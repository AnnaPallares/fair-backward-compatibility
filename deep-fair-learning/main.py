import os
import wandb
import sys
import datetime
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli import parse_args
from training.trainer import run_experiment

wandb.init(
    project="fair-backward-compatibility",
    mode="offline" if os.environ.get("WANDB_API_KEY") is None else "online"
)

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU Setup Error: {e}")

def main():
    setup_gpu()
    args = parse_args()

    # Mapping Paper Terminology to Internal Logic
    scenario_map = {
        "naive": {"double_step": False, "use_weights": False, "mitig2b": False},
        "fbc-s": {"double_step": True,  "use_weights": False, "mitig2b": False},
        "fbc-d": {"double_step": True,  "use_weights": True,  "mitig2b": False},
        "fbc-c": {"double_step": True,  "use_weights": False, "mitig2b": True}
    }

    if args.scenario in scenario_map:
        config = scenario_map[args.scenario]
        args.double_step = config["double_step"]
        args.use_weights = config["use_weights"]
        args.mitig2b     = config["mitig2b"]

    # Generate experiment name if not provided
    if args.exp_name is None:
        ts = datetime.datetime.now().strftime("%y%m%d_%H%M")
        args.exp_name = (
            f"{args.dataset}_{args.scenario}"
            f"_ft{int(args.fine_tune)}"
            f"_seed{args.seed}_{ts}"
        )

    # Finalize output directory structure
    run_id = f"seed-{args.seed}"
    args.run_id = run_id
    args.output_dir = os.path.join(args.output_dir, args.dataset, run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # Kick off the experiment
    run_experiment(args)

if __name__ == "__main__":
    main()