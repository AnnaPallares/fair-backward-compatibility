import tensorflow as tf

class FairModel(tf.keras.Model):
    def __init__(self, base_model, target_group, lambda_, NFtype, use_weights=False, mitig2b=False):
        """
        Implementation of Fair Backward-Compatible Empirical Risk Minimization.
        
        Args:
            base_model: Keras model outputting sigmoid probabilities.
            target_group:  sensitive group (0 or 1) prone to unfair regression.
            lambda_: weight for the FBC penalty.
            NFtype: 'all' for DP, 'positive', or 'negative' for EO (check labels to set this correctly).
            use_weights: boolean for accounting for differentiable relaxation, FBC-D .
            mitig2b: boolean for accounting for convex relaxation, FBC-C.
        """
        super().__init__()
        self.base           = base_model
        self.target_group   = int(target_group)
        self.lambda_        = float(lambda_)
        self.NFtype         = NFtype
        
        # Paper Terminology Mapping
        self.fbc_d_mode     = bool(use_weights)
        self.fbc_c_mode     = bool(mitig2b)

        if self.fbc_d_mode and self.fbc_c_mode:
            raise ValueError("FBC-D and FBC-C strategies cannot be active simultaneously.")

        # Trackers
        self.loss_tracker     = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker      = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.auc_tracker      = tf.keras.metrics.AUC(name="auc")
        self.ur_tracker       = tf.keras.metrics.Mean(name="unfair_regression")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker, self.auc_tracker, self.ur_tracker]

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    def compute_fair_loss(self, y_true, y_pred, y_old, s):
        """
        Calculates the FBC loss: L = L_base + lambda * (FBC_Constraint)
        """
        y_pred = tf.squeeze(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.float32)
        y_old = tf.cast(y_old, tf.float32)
        y_old_bin = tf.cast(y_old > 0.5, tf.float32)        
        s = tf.cast(s, tf.int32)
        eps = 1e-7

        # Standard Binary Cross-Entropy (BCE) per sample
        y_pred_clipped = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(y_true * tf.math.log(y_pred_clipped) + (1 - y_true) * tf.math.log(1 - y_pred_clipped))

        # Identify Negative Flips (NF)
        if self.NFtype == "positive":
            flip_mask = tf.logical_and(tf.equal(y_old_bin, y_true), tf.equal(y_true, 1.0))
        elif self.NFtype == "negative":
            flip_mask = tf.logical_and(tf.equal(y_old_bin, y_true), tf.equal(y_true, 0.0))
        else:
            flip_mask = tf.equal(y_old_bin, y_true)
        
        flip_mask = tf.cast(flip_mask, tf.float32)

        # FBC mitigation Selection
        if self.fbc_d_mode:
            # FBC-D: Differentiable Relaxation. Weight binary cross-entropy specifically for the sensitive target group
            target_mask = tf.cast(tf.equal(s, self.target_group), tf.float32)
            penalty = flip_mask * target_mask * self.lambda_
            total_bce = (1.0 + penalty) * bce
            return tf.reduce_mean(total_bce)

        elif self.fbc_c_mode:
            # FBC-C: Convex Relaxation. Squared difference between group-wise losses
            mask_s0 = tf.cast(tf.equal(s, 0), tf.float32)
            mask_s1 = tf.cast(tf.equal(s, 1), tf.float32)
            
            bce_s0 = tf.reduce_sum(bce * mask_s0) / (tf.reduce_sum(mask_s0) + eps)
            bce_s1 = tf.reduce_sum(bce * mask_s1) / (tf.reduce_sum(mask_s1) + eps)
            
            diff_penalty = tf.square(bce_s0 - bce_s1) * self.lambda_
            return tf.reduce_mean(bce) + diff_penalty
             
        else:
            # Standard Binary Cross Entropy
            return tf.reduce_mean(bce)

    def train_step(self, data):
        (img, y_old, s), y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base(img, training=True)
            loss = self.compute_fair_loss(y_true, y_pred, y_old, s)

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

        # Update core metrics
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_true, y_pred)
        self.auc_tracker.update_state(y_true, y_pred)
        
        # Calculate and track current batch Unfair Regression (i.e. FBC)
        self._update_ur_metric(y_true, y_pred, y_old, s)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (img, y_old, s), y_true = data
        y_pred = self.base(img, training=False)
        loss = self.compute_fair_loss(y_true, y_pred, y_old, s)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_true, y_pred)
        self.auc_tracker.update_state(y_true, y_pred)
        self._update_ur_metric(y_true, y_pred, y_old, s)

        return {m.name: m.result() for m in self.metrics}

    def _update_ur_metric(self, y_true, y_pred, y_old, s):
        """Internal helper to track FBC during training."""
        y_old = tf.cast(y_old, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        y_old_bin = tf.cast(y_old > 0.5, tf.float32)
        y_new_bin = tf.cast(y_pred > 0.5, tf.float32)
        
        # Negative Flip: Old was correct, New is wrong
        nf_mask = tf.logical_and(tf.equal(y_old_bin, y_true), tf.not_equal(y_new_bin, y_true))
        nf_mask = tf.cast(nf_mask, tf.float32)
        
        ur_s0 = tf.reduce_sum(nf_mask * tf.cast(tf.equal(s, 0), tf.float32)) / (tf.reduce_sum(tf.cast(tf.equal(s, 0), tf.float32)) + 1e-7)
        ur_s1 = tf.reduce_sum(nf_mask * tf.cast(tf.equal(s, 1), tf.float32)) / (tf.reduce_sum(tf.cast(tf.equal(s, 1), tf.float32)) + 1e-7)
        
        self.ur_tracker.update_state(tf.abs(ur_s0 - ur_s1))