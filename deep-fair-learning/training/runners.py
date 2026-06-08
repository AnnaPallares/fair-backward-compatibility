import os
import gc
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from tqdm.keras import TqdmCallback
from wandb.integration.keras import WandbMetricsLogger

from utils.datasetUtils import make_baseline_dataset, make_image_ds
from training.cv import cvFair_double_step
from utils.fairnessUtils import get_sensitive_target
from utils.loggingUtils import save_metrics_json

def print_balanced_details(y_true, y_pred, model_label="Model"):
    """Prints specificity, sensitivity, and balanced accuracy for fairness contexts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    print(f"\n--- {model_label} Performance ---")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:         {specificity:.4f}")
    print(f"Balanced Accuracy:   {balanced_acc:.4f}")
    print(f"Confusion Matrix:    [TN: {tn}, FP: {fp} / FN: {fn}, TP: {tp}]")

def run_baselines(args, train_df, val_df, builder_old, pre_fn_old, builder_new, pre_fn_new,
                  old_path=None, new_path=None, class_weight=None):
    """
    Trains the Reference (Old) and Naive (New) models.
    Computes initial Negative Flips to identify the target group for mitigation.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Standardized logging configuration
    common_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lr": args.lr,
        "scenario": args.scenario, # e.g., 'naive', 'fbc-s'
        "nf_type": args.nf_type,
        "seed": args.seed
    }

    # 1. Reference Model (Old Architecture/Data subset)
    old_sample_df = train_df.sample(frac=args.subset_frac, random_state=args.seed)
    ds_old = make_baseline_dataset(old_sample_df, args.batch_size, args.seed, pre_fn_old, args.image_size, augment=args.augment)
    ds_val_old = make_baseline_dataset(val_df, args.batch_size, args.seed, pre_fn_old, args.image_size, augment=False)

    wandb.init(project=args.project, name=f"ref-old_s{args.seed}", group=args.exp_name, reinit=True, config=common_config)

    model_old = builder_old(input_shape=args.image_size + (3,))
    model_old.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    model_old.fit(ds_old, validation_data=ds_val_old, epochs=args.epochs,
                  callbacks=[WandbMetricsLogger(), TqdmCallback()],
                  class_weight=class_weight, verbose=0)
    wandb.finish()
    
    if old_path:
        model_old.save(old_path)

    # 2. Naive Update Model (New Architecture/Full data)
    ds_new = make_baseline_dataset(train_df, args.batch_size, args.seed, pre_fn_new, args.image_size, augment=args.augment)
    ds_val_new = make_baseline_dataset(val_df, args.batch_size, args.seed, pre_fn_new, args.image_size, augment=False)

    wandb.init(project=args.project, name=f"naive-new_s{args.seed}", group=args.exp_name, reinit=True, config=common_config)

    model_new = builder_new(input_shape=args.image_size + (3,))
    model_new.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    model_new.fit(ds_new, validation_data=ds_val_new, epochs=args.epochs,
                  callbacks=[WandbMetricsLogger(), TqdmCallback()],
                  class_weight=class_weight, verbose=0)

    if new_path:
        model_new.save(new_path)

    # 3. Compute Predictions & FBC Metrics
    val_imgs_old = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_old, args.image_size)
    val_imgs_new = make_image_ds(val_df['file'].values, args.batch_size, pre_fn_new, args.image_size)
    train_imgs_old = make_image_ds(train_df['file'].values, args.batch_size, pre_fn_old, args.image_size)

    train_df['y_old_pred'] = (model_old.predict(train_imgs_old) > 0.5).astype(int).flatten()
    val_df['y_old_pred']   = (model_old.predict(val_imgs_old) > 0.5).astype(int).flatten()
    y_new_val              = (model_new.predict(val_imgs_new) > 0.5).astype(int).flatten()

    print_balanced_details(val_df['target'].values, val_df['y_old_pred'].values, "REFERENCE (Old)")
    print_balanced_details(val_df['target'].values, y_new_val, "NAIVE (New)")

    # Identify the sensitive target group (samples that suffered backward incompatibility)
    sens_target, baseline_metrics = get_sensitive_target(
        val_df['target'].values, val_df['y_old_pred'].values, y_new_val,
        val_df['sensitive'].values, args.nf_type
    )
    
    # Unfair Regression (FBC)
    ur_disparity = abs(baseline_metrics[0] - baseline_metrics[1])
    wandb.log({"initial_UR_disparity": ur_disparity})
    wandb.finish()

    # Compute overall performance metrics
    y_true = val_df['target'].values
    acc_old = accuracy_score(y_true, val_df['y_old_pred'].values)
    acc_new = accuracy_score(y_true, y_new_val)
    bacc_old = balanced_accuracy_score(y_true, val_df['y_old_pred'].values)
    bacc_new = balanced_accuracy_score(y_true, y_new_val)

    # Save metrics and cleanup
    metrics_to_save = {
        "UR_disparity": ur_disparity,
        "acc_old": float(acc_old),
        "acc_new": float(acc_new),
        "bacc_old": float(bacc_old),
        "bacc_new": float(bacc_new)
    }
    save_metrics_json(os.path.join(args.output_dir, f"baseline_metrics_{timestamp}.json"), metrics_to_save)
    
    del model_old, model_new
    tf.keras.backend.clear_session()
    gc.collect()

    return sens_target

def run_cv(args, train_df, sens_target, builder_new, pre_fn_new):
    """
    Executes Cross-Validation to find optimal hyperparameters (Lambda, LR, Batch Size).
    Supports single-step and double-step (FBC) validation.
    """
    threshold = args.acc_threshold if args.double_step else 0.0

    (best_params, _, _) = cvFair_double_step(
        df=train_df,
        lambda_vals=args.lambda_vals,
        lr_vals=args.lr_vals,
        batch_size_vals=args.batch_size_vals,
        threshold=threshold,
        target_group=sens_target,
        nf_type=args.nf_type,
        args=args,
        builder=builder_new,
        pre_fn=pre_fn_new,
        img_size=args.image_size
    )

    best_lam, best_lr, best_bs = best_params

    # Final cleanup before returning to main trainer
    tf.keras.backend.clear_session()
    gc.collect()

    return best_lam, best_lr, best_bs