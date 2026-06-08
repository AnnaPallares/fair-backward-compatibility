import tensorflow as tf
import wandb
import datetime
import os
import gc
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from wandb.integration.keras import WandbMetricsLogger
from tqdm.keras import TqdmCallback

from models.FairModelClass_ResNet import FairModel
from utils.datasetUtils import make_fair_dataset, make_image_ds
from utils.fairnessUtils import negativeFlip_rate
from utils.loggingUtils import save_metrics_json


def log_final_fair_model(
    args,
    project: str,
    run_name: str,
    train_df,
    val_df,
    builder,
    pre_fn,
    image_size,
    seed,
    sens_target: int,
    best_lam: float,
    best_lr: float,
    best_bs: int,
    nf_type: str,
    use_weights: bool,
    mitig2b: bool,
    epochs: int
):
    """
    Retrains the FairModel with the chosen hyperparameters on the full training set,
    logs per-epoch train/val curves (loss, accuracy, auc, ur) to W&B, then logs final UR.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fair_name = f"{run_name}_{args.scenario}_seed{seed}_{args.run_id}"
    
    tf.keras.backend.clear_session()
    gc.collect()

    # 1) init W&B run
    wandb.init(
        project=project,
        name=fair_name,
        group=args.exp_name,
        reinit=True,
        config=vars(args)
    )


    # 2) build & compile model
    base_model = builder(input_shape=image_size + (3,))

    # Fine-tuning logic: 
    if args.fine_tune:
        base_model.trainable = True 
        
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
            
        # Unfreeze last 10 layers (except BatchNorm for stability)
        for layer in base_model.layers[-10:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # Log trainable layers info 
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    wandb.log({"trainable_layers_count": trainable_count})

    fm = FairModel(
        base_model,
        target_group=sens_target,
        lambda_=best_lam,
        NFtype=nf_type,
        use_weights=use_weights,
        mitig2b=mitig2b
    )
    fm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr))

    # 3) Setup Callbacks for 100 epochs
    cbs = [
        WandbMetricsLogger(),
        TqdmCallback(verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=12,          # Stops if no improvement after 12 epochs
            mode='min', 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7
        )
    ]

    # 4) fit
    ds_tr = make_fair_dataset(train_df, best_bs, seed, pre_fn, image_size, augment=args.augment)
    ds_vl = make_fair_dataset(val_df,   best_bs, seed, pre_fn, image_size, augment=False)

    fm.fit(
        ds_tr,
        validation_data=ds_vl,
        epochs=epochs,
        callbacks=cbs,
        verbose=0
    )

    # 5) final UR and accuracu metric on validation set
    y_old = val_df['y_old_pred'].values
    y_new = (fm.predict(make_image_ds(val_df['file'].values, best_bs, pre_fn, image_size)) > 0.5).astype(int).flatten()
    y_true = val_df['target'].values

    ur_metrics = negativeFlip_rate(
        y_true,
        y_old,
        y_new,
        val_df['sensitive'].values,
        nf_type
    )
    ur_disparity = ur_metrics['Unfair Regression']

    val_acc_old = accuracy_score(y_true, y_old)
    val_acc_new = accuracy_score(y_true, y_new)
    val_bacc_old = balanced_accuracy_score(y_true, y_old)
    val_bacc_new = balanced_accuracy_score(y_true, y_new)

    wandb.log({
        "val_accuracy": val_acc_new,
        "val_bacc": val_bacc_new,
        "UR_disparity": ur_disparity      
    })

    metrics_out = {
        "UR_disparity": float(ur_disparity),
        "acc_old": float(val_acc_old),
        "acc_new": float(val_acc_new),
        "bacc_old": float(val_bacc_old),
        "bacc_new": float(val_bacc_new)
    }

    fm.save(os.path.join(args.output_dir, f"fair_model_{args.scenario}.h5"))
    save_metrics_json(os.path.join(args.output_dir, f"fair_metrics_{args.scenario}.json"), metrics_out)


    wandb.finish()

    return fm
