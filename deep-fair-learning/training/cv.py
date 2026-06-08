import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tensorflow.python.framework.errors_impl import ResourceExhaustedError

from utils.datasetUtils import make_fair_dataset, make_image_ds
from models.FairModelClass_ResNet import FairModel
from utils.fairnessUtils import negativeFlip_rate

def cvFair_double_step(df, lambda_vals, lr_vals, batch_size_vals, threshold, target_group,
                        nf_type, args, builder, pre_fn, img_size):
    """
    Implements the Double-Step Cross-Validation strategy (mitigation FBC-S) described in the paper:
    Step 1: Grid search over Lambda (constraint weight), LR, and Batch Size.
    Step 2: Filter models within a tolerable accuracy drop (threshold) of the max accuracy.
    Step 3: Select the candidate with the lowest Unfair Regression (UR), namely FBC.
    """

    # Define hyperparameter grid based on the chosen FBC scenario
    if args.use_weights or args.mitig2b:
        param_grid = {
            'lambda':        lambda_vals,
            'learning_rate': lr_vals,
            'batch_size':    batch_size_vals
        }
    else:
        # For Naive or standard updates, lambda unused
        param_grid = {
            'lambda':        [0.0], 
            'learning_rate': lr_vals,
            'batch_size':    batch_size_vals
        }

    results_list = []
    param_list = list(ParameterGrid(param_grid))
    
    print(f"Starting Double-Step CV: {len(param_list)} combinations, {args.folds} folds each.")

    for params in tqdm(param_list, desc="Hyperparameter Grid"):
        lam = params['lambda']
        lr  = params['learning_rate']
        bs  = int(params['batch_size'])

        acc_folds, ur_folds = [], []
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(df['file'], df['target'])):
            # Memory safety: clear GPU memory at the start of every fold
            tf.keras.backend.clear_session()
            gc.collect()

            df_tr, df_vl = df.iloc[tr_idx], df.iloc[vl_idx]

            ds_tr = make_fair_dataset(df_tr, bs, args.seed, pre_fn, img_size, augment=args.augment)
            ds_vl = make_fair_dataset(df_vl, bs, args.seed, pre_fn, img_size, augment=False)

            try:
                # Initialize architecture
                base_model = builder(input_shape=img_size + (3,))
                
                # Wrap in FBC specialized Model
                fm = FairModel(
                    base_model, 
                    target_group=target_group, 
                    lambda_=lam, 
                    NFtype=nf_type, 
                    use_weights=args.use_weights, 
                    mitig2b=args.mitig2b
                )
                
                fm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
                ]
                
                fm.fit(ds_tr, validation_data=ds_vl, epochs=args.epochs, callbacks=callbacks, verbose=0)
            except ResourceExhaustedError:
                print(f"\n[Warning] OOM with Batch Size {bs}. Skipping combination.")
                break # Exit fold loop for this BS

            # Evaluate Fold Performance
            y_old_vl = df_vl['y_old_pred'].values
            y_true_vl = df_vl['target'].values
            
            ds_imgs = make_image_ds(df_vl['file'].values, bs, pre_fn, img_size)
            y_new_vl = (fm.predict(ds_imgs) > 0.5).astype(int).ravel()

            acc_folds.append((y_new_vl == y_true_vl).mean())
            
            # Calculate Unfair Regression rate for this specific fold
            metrics = negativeFlip_rate(y_true_vl, y_old_vl, y_new_vl, df_vl['sensitive'].values, nf_type)
            ur_folds.append(metrics['Unfair Regression'])

        # Store mean results for this hyperparameter combo
        if acc_folds:
            results_list.append({
                'lambda':        lam,
                'learning_rate': lr,
                'batch_size':    bs,
                'mean_acc':      np.mean(acc_folds),
                'mean_UR':       np.mean(ur_folds)
            })

    # --- Phase 2: Selection Logic ---
    df_results = pd.DataFrame(results_list)
    
    # 1. Find max accuracy across all combos
    max_acc = df_results['mean_acc'].max()
    
    # 2. Define the minimum accuracy  (e.g., within 10% of max_acc if threshold is 0.1)
    acc_floor = max_acc * (1.0 - threshold)
    
    # 3. Filter candidates that meet the accuracy requirement
    candidates = df_results[df_results['mean_acc'] >= acc_floor].copy()
    
    if candidates.empty:
        # Fallback: if no model meets the threshold, take the most accurate one
        candidates = df_results[df_results['mean_acc'] == max_acc].copy()

    # 4. Of the accurate candidates, pick the one with the lowest Unfair Regression
    candidates = candidates.sort_values('mean_UR').reset_index(drop=True)
    best_row = candidates.iloc[0]

    print(f"\nCV Complete. Best Params Selected:")
    print(f" > Lambda: {best_row['lambda']} | LR: {best_row['learning_rate']} | BS: {best_row['batch_size']}")
    print(f" > Expected Acc: {best_row['mean_acc']:.4f} | Expected UR: {best_row['mean_UR']:.4f}")

    # Final cleanup after CV grid search (helps avoiding OOM error)
    tf.keras.backend.clear_session()
    gc.collect()

    return (best_row['lambda'], best_row['learning_rate'], best_row['batch_size']), df_results, candidates