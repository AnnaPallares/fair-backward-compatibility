import tensorflow as tf
import wandb
import datetime
import os
import gc
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
    ur = negativeFlip_rate(
        val_df['target'].values,
        y_old,
        y_new,
        val_df['sensitive'].values,
        nf_type
    )['Unfair Regression']

    y_true = val_df['target'].values
    val_acc = (y_new == y_true).mean()

    wandb.log({
        "val_accuracy": val_acc,
        "UR": ur      
    })

    metrics_out = {
        "timestamp": timestamp,
        "model_type": "fair",
        "nf_type": nf_type,
        "use_weights": use_weights,
        "mitig2b": mitig2b,
        "biased on": sens_target,
        "lambda": best_lam,
        "learning_rate": best_lr,
        "batch_size": best_bs,
        "epochs": epochs,
        "val_accuracy": val_acc,
        "UR": ur,
        "train_loss": [float(x) for x in fm.history.history["loss"]],
        "train_acc":  [float(x) for x in fm.history.history["accuracy"]],
        "train_auc":  [float(x) for x in fm.history.history["auc"]],
        "val_loss":   [float(x) for x in fm.history.history["val_loss"]],
        "val_acc":    [float(x) for x in fm.history.history["val_accuracy"]],
        "val_auc":    [float(x) for x in fm.history.history["val_auc"]],

    }

    # outdir = os.path.join(args.output_dir, args.scenario)
    # os.makedirs(outdir, exist_ok=True)

    fm.save(os.path.join(args.output_dir, f"fair_model_{args.scenario}.h5"))
    save_metrics_json(os.path.join(args.output_dir, f"fair_metrics_{args.scenario}.json"), metrics_out)


    wandb.finish()

    return fm
