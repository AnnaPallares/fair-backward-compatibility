import os
import gc
import tensorflow as tf
import keras_hub

try:
    from keras_hub.src.models.vit.vit_layers import ViTPatchingAndEmbedding, ViTEncoder, ViTEncoderBlock
    from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
    from keras_hub.src.models.vit.vit_backbone import ViTBackbone
    
    original_vit_from_config = ViTPatchingAndEmbedding.from_config
    @classmethod
    def patched_vit_from_config(cls, config):
        config.pop("num_patches", None)
        config.pop("num_positions", None)
        if isinstance(config, dict):
             return cls(**config)
        return original_vit_from_config(config)
    
    ViTPatchingAndEmbedding.from_config = patched_vit_from_config
    
    tf.keras.utils.get_custom_objects()["ViTPatchingAndEmbedding"] = ViTPatchingAndEmbedding
    tf.keras.utils.get_custom_objects()["TransformerEncoder"] = TransformerEncoder
    tf.keras.utils.get_custom_objects()["ViTEncoder"] = ViTEncoder
    tf.keras.utils.get_custom_objects()["ViTEncoderBlock"] = ViTEncoderBlock
    tf.keras.utils.get_custom_objects()["ViTBackbone"] = ViTBackbone

    from keras_hub.src.models.deit.deit_layers import DeiTEncoder, DeiTEmbeddings
    
    # Fix bug in DeiTEncoder.get_config (missing key_dim attribute)
    def robust_deit_get_config(self):
        config = super(DeiTEncoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "key_dim": getattr(self, "key_dim", self.hidden_dim // self.num_heads),
            "use_mha_bias": getattr(self, "use_mha_bias", True),
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config
    DeiTEncoder.get_config = robust_deit_get_config

    original_deit_encoder_from_config = DeiTEncoder.from_config
    @classmethod
    def patched_deit_encoder_from_config(cls, config):
        config.pop("key_dim", None)
        config.pop("use_mha_bias", None)
        if isinstance(config, dict):
             return cls(**config)
        return original_deit_encoder_from_config(config)
    DeiTEncoder.from_config = patched_deit_encoder_from_config
    
    original_deit_emb_from_config = DeiTEmbeddings.from_config
    @classmethod
    def patched_deit_emb_from_config(cls, config):
        config.pop("num_patches", None)
        config.pop("num_positions", None)
        if isinstance(config, dict):
             return cls(**config)
        return original_deit_emb_from_config(config)
    DeiTEmbeddings.from_config = patched_deit_emb_from_config

    tf.keras.utils.get_custom_objects()["DeiTEncoder"] = DeiTEncoder
    tf.keras.utils.get_custom_objects()["DeiTEmbeddings"] = DeiTEmbeddings
    
except ImportError:
    pass

import pandas as pd
import wandb
import time
import json
from data_loader import (
    load_fairfaces_data, load_utkface_data,
    load_fitzpatrick_data, load_ddi_data, load_marvel_data
) 
from utils.modelUtils import get_builder_and_preproc
from training.runners import run_baselines, run_cv
from training.fair_logging import log_final_fair_model
from utils.datasetUtils import make_image_ds
from utils.fairnessUtils import get_sensitive_target


DATA_LOADERS = {
    'FairFace':       load_fairfaces_data,
    'UTKFace':        load_utkface_data,
    'fitzpatrick17k': load_fitzpatrick_data,
    'DDI':            load_ddi_data,
    'marvel':         load_marvel_data,
}

def run_experiment(args):
    """
    Orchestrates the FBC experiment: 
    1. Data Loading/Augmentation
    2. Baseline Training (Naive update)
    3. Cross-validation
    4. Final Fair Model training
    """
    
    if args.dataset not in DATA_LOADERS:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
        
    loader = DATA_LOADERS[args.dataset]
    train_df, val_df = loader(
        args.base_dir,
        seed=args.seed,
        sensitive_attribute=getattr(args, 'sensitive_attribute', None)
    )

    # --- Dataset-Specific Preprocessing (DDI) ---
    if args.dataset == 'DDI':
        # DDI requires augmentation and balancing due to small/imbalanced nature
        args.augment = True  
        
        df_pos = train_df[train_df['target'] == 1]
        df_neg = train_df[train_df['target'] == 0]
        
        if len(df_pos) < len(df_neg):
            n_to_add = len(df_neg) - len(df_pos)
            df_oversampled = df_pos.sample(n=n_to_add, replace=True, random_state=args.seed)
            train_df = pd.concat([train_df, df_oversampled], axis=0).reset_index(drop=True)

    # --- Dataset-Specific Preprocessing if needed ---
    if args.dataset == 'fitzpatrick17k':
        args.augment = True

        df_maj = train_df[train_df['target'] == 1] 
        df_min = train_df[train_df['target'] == 0] 

        n_to_add = len(df_maj) - len(df_min)
        df_oversampled = df_min.sample(n=n_to_add, replace=True, random_state=args.seed)
        train_df = pd.concat([train_df, df_oversampled], axis=0).reset_index(drop=True)
        print(f"[Fitzpatrick] Oversampled minority class-0: {len(df_min)} → {len(df_maj)} samples. "
              f"New train size: {len(train_df)}")

    # Prepare architectures
    builder_old, pre_fn_old = get_builder_and_preproc(args.arch_old)
    builder_new, pre_fn_new = get_builder_and_preproc(args.arch_new)

    # Define paths for baseline models
    dataset_dir = os.path.dirname(os.path.dirname(args.output_dir))
    baseline_dir = os.path.join(dataset_dir, 'baselines')
    os.makedirs(baseline_dir, exist_ok=True)
    
    old_path = os.path.join(baseline_dir, f'baseline_{args.arch_old}_seed{args.seed}.h5')
    new_path = os.path.join(baseline_dir, f'baseline_{args.arch_new}_seed{args.seed}.h5')

    # --- STAGE 1: Baselines (Reference & New Naive) ---
    t0_baseline = time.time()
    if args.stage in ('baseline', 'all'):
        # Check if already exists to skip redundant training across scenarios of the same seed
        if os.path.exists(old_path) and os.path.exists(new_path):
            print(f"Loading existing baseline models for seed {args.seed}...")
            with tf.keras.utils.custom_object_scope(keras_hub.layers.__dict__):
                m_old = tf.keras.models.load_model(old_path)
                m_new = tf.keras.models.load_model(new_path)
            
            # Generate predictions for the reference model
            train_ds_old = make_image_ds(train_df['file'].values, args.batch_size, pre_fn_old, args.image_size)
            val_ds_old   = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_old, args.image_size)
            val_ds_new   = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_new, args.image_size)

            train_df['y_old_pred'] = (m_old.predict(train_ds_old) > 0.5).astype(int).flatten()
            val_df['y_old_pred']   = (m_old.predict(val_ds_old) > 0.5).astype(int).flatten()
            y_new_val              = (m_new.predict(val_ds_new) > 0.5).astype(int).flatten()

            sens_target, _ = get_sensitive_target(
                y_true     = val_df['target'].values,
                y_pred_old = val_df['y_old_pred'].values,
                y_pred_new = y_new_val,
                sens_attr  = val_df['sensitive'].values,
                NFtype     = args.nf_type
            )
        else:
            sens_target = run_baselines(
                args, train_df, val_df,
                builder_old, pre_fn_old,
                builder_new, pre_fn_new,
                old_path=old_path,
                new_path=new_path
            )
    else:
        # Load pre-trained baselines if skipping baseline stage
        if not os.path.exists(old_path) or not os.path.exists(new_path):
            raise FileNotFoundError(f"Baseline models not found in {baseline_dir}. Run with --stage all first.")
            
        m_old = tf.keras.models.load_model(old_path)
        m_new = tf.keras.models.load_model(new_path)

        # Generate predictions for the reference model
        train_ds_old = make_image_ds(train_df['file'].values, args.batch_size, pre_fn_old, args.image_size)
        val_ds_old   = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_old, args.image_size)
        val_ds_new   = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_new, args.image_size)

        train_df['y_old_pred'] = (m_old.predict(train_ds_old) > 0.5).astype(int).flatten()
        val_df['y_old_pred']   = (m_old.predict(val_ds_old) > 0.5).astype(int).flatten()
        y_new_val              = (m_new.predict(val_ds_new) > 0.5).astype(int).flatten()

        # Identify targets for FBC (Negative Flips)
        sens_target, _ = get_sensitive_target(
            y_true     = val_df['target'].values,
            y_pred_old = val_df['y_old_pred'].values,
            y_pred_new = y_new_val,
            sens_attr  = val_df['sensitive'].values,
            NFtype     = args.nf_type
        )
        
        # Save sensitive target for future stages (final training)
        meta_path = os.path.join(args.output_dir, f"meta_{args.scenario}.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        meta['sens_target'] = int(sens_target)
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        # Free up baseline model memory before CV and final training
        del m_old, m_new
        tf.keras.backend.clear_session()
        gc.collect()
    t1_baseline = time.time()
    baseline_time = t1_baseline - t0_baseline

    # --- STAGE 2: Mitigation / Hyperparameter Tuning ---
    t0_cv = time.time()
    meta_path = os.path.join(args.output_dir, f"meta_{args.scenario}.json")
    
    if (args.stage in ('mitigation', 'cv', 'final', 'all')) and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if 'sens_target' in meta:
            sens_target = meta['sens_target']
            print(f"Loaded sens_target={sens_target} from {meta_path}")

    if args.stage in ('mitigation', 'cv', 'all'):
        if sens_target is None:
            print("[Error] sens_target is missing. Did the baseline stage run successfully?")
            return
            
        best_lam, best_lr, best_bs = run_cv(
            args, train_df, sens_target,
            builder_new, pre_fn_new
        )

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta.update({'lambda': best_lam, 'lr': best_lr, 'bs': best_bs})
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    else:
        best_lam, best_lr, best_bs = None, None, None
        
    t1_cv = time.time()
    cv_time = t1_cv - t0_cv

    # --- STAGE 3: Final FBC Model Training ---
    t0_final = time.time()
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    if args.stage in ('mitigation', 'final', 'all'):
        if (args.stage == 'final' or best_lam is None) and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            sens_target = meta.get('sens_target', sens_target)
            best_lam = meta.get('lambda')
            best_lr = meta.get('lr')
            best_bs = meta.get('bs')
            print(f"Loaded params from {meta_path}: Lam={best_lam}, LR={best_lr}, BS={best_bs}")

        if best_bs is not None:
            best_bs = int(best_bs)

        if best_lam is not None:
            fm = log_final_fair_model(
                    args        = args,
                    project     = args.project,
                    run_name    = f"FBC_Model_{args.scenario}",
                    train_df    = train_df,
                    val_df      = val_df,
                    builder     = builder_new,
                    pre_fn      = pre_fn_new,
                    image_size  = args.image_size,
                    seed        = args.seed,
                    sens_target = sens_target,
                    best_lam    = best_lam,
                    best_lr     = best_lr,
                    best_bs     = best_bs,
                    nf_type     = args.nf_type,
                    use_weights = args.use_weights, # FBC-D relaxation
                    mitig2b     = args.mitig2b,     # FBC-C relaxation
                    epochs      = args.epochs
                )
    
        else:
            print("[Warning] Skipping final training because no best parameters were found/provided.")
    t1_final = time.time()
    final_time = t1_final - t0_final

    # --- Save Timing Data ---
    timing_data = {
        "baseline_time": baseline_time,
        "cv_time": cv_time,
        "final_time": final_time,
        "total_time": baseline_time + cv_time + final_time
    }
    
    with open(os.path.join(args.output_dir, f"train_time_{args.scenario}.json"), "w") as f:
        json.dump(timing_data, f, indent=4)
        
    try:
        wandb.log(timing_data)
    except Exception as e:
        pass