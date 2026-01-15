import os
import tensorflow as tf
import pandas as pd
import wandb

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
    3. Cross-validation for FBC parameters
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

    # Prepare architectures
    builder_old, pre_fn_old = get_builder_and_preproc(args.arch_old)
    builder_new, pre_fn_new = get_builder_and_preproc(args.arch_new)

    # Define paths for baseline models
    # Note: These are stored in a 'baselines' subfolder shared across seeds for efficiency
    baseline_dir = os.path.join(os.path.dirname(args.output_dir), 'baselines')
    os.makedirs(baseline_dir, exist_ok=True)
    
    old_path = os.path.join(baseline_dir, f'baseline_{args.arch_old}.h5')
    new_path = os.path.join(baseline_dir, f'baseline_{args.arch_new}.h5')

    # --- STAGE 1: Baselines (Reference & New Naive) ---
    if args.stage in ('baseline', 'all'):
        sens_target = run_baselines(
            args, train_df, val_df,
            builder_old, pre_fn_old,
            builder_new, pre_fn_new
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

    # --- STAGE 2: Mitigation / Hyperparameter Tuning ---
    if args.stage in ('mitigation', 'all'):
        best_lam, best_lr, best_bs = run_cv(
            args, train_df, sens_target,
            builder_new, pre_fn_new
        )
    else:
        best_lam, best_lr, best_bs = None, None, None

    # --- STAGE 3: Final FBC Model Training ---
    if args.stage in ('mitigation', 'all'):
        # Pass the renamed mitig2b -> mitig2b (keeping internal name for now or refactoring logging)
        # Note: log_final_fair_model should ideally be updated to use the new nomenclature in its logs
        log_final_fair_model(
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