import argparse
import numpy as np
import os

# Relative paths to data from the project root
DATA_PATHS = {
    'FairFace':       "data/FairFace",
    'UTKFace':        "data/UTKFace",
    'fitzpatrick17k': "data/fitzpatrick17k",
    'DDI':            "data/DDI",
    'marvel':         "data/marvel",
}

def str2bool(v: str) -> bool:
    """Helper for parsing boolean flags in command line."""
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', '1')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fair Backward Compatibility (FBC) Training Framework"
    )

    # Dataset Selection
    parser.add_argument(
        '--dataset',
        required=True,
        choices=list(DATA_PATHS.keys()),
        help="Target dataset for the experiment."
    )

    parser.add_argument(
        '--scenario', 
        type=str, 
        default="naive",
        choices=["naive", "fbc-s", "fbc-d", "fbc-c"],
        help="FBC Mitigation strategy: naive (none), fbc-s (model selection phase, 2-step cv), fbc-d (differentiable relaxation), fbc-c (convex relaxation)"
    )

    # Core Hyperparameters
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--subset_frac', type=float, default=0.2)
    parser.add_argument('--folds',       type=int,   default=5)
    parser.add_argument('--epochs',      type=int,   default=25)


    # Fixed hyperparameters for Baseline mdoels
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)


    # FBC Specific Configurations
    parser.add_argument('--sensitive_attribute', type=str, default=None, choices=["gender", "race"]) # only needed for FairFace and UTKFace datasets
    parser.add_argument('--acc_threshold',       type=float, default=0.10)
    parser.add_argument('--nf_type',             choices=['all','positive','negative'], default='all', help="Fairness definition: all=DP, positive/negative=EO") 

    # Internal logic flags (set automatically by scenario, but available for override)
    parser.add_argument('--double_step', type=str2bool, nargs='?', const=True, default=False) # FBC-S
    parser.add_argument('--use_weights', type=str2bool, nargs='?', const=True, default=False) # FBC-D
    parser.add_argument('--mitig2b',     type=str2bool, nargs='?', const=True, default=False) # FBC-C
    parser.add_argument('--fine_tune',   type=str2bool, nargs='?', const=True, default=False) # True in all new and fair models
    parser.add_argument('--augment',    type=str2bool, nargs='?', const=True, default=False) # data augmentation for specific cases


    # Cross-Validation Grids
    parser.add_argument('--lambda_vals',     type=float, nargs='+', default=list(np.logspace(-3, 2, 5)))
    parser.add_argument('--batch_size_vals', type=int,   nargs='+', default=[16, 32, 64])
    parser.add_argument('--lr_vals',         type=float, nargs='+', default=[1e-3, 1e-4, 1e-5])    

    # Execution Flow
    parser.add_argument('--stage', choices=['baseline','mitigation','cv','final','all'], default='all')
    parser.add_argument('--arch_old', choices=['resnet50', 'mobilenetv2', 'resnet18', 'vit-small'], default='resnet18')
    parser.add_argument('--arch_new', choices=['resnet18', 'resnet50', 'mobilenetv2', 'vit-small'], default='resnet50')

    # Logging & Outputs
    parser.add_argument('--project',    type=str, default='FBC_Project', help='Wandb project name')
    parser.add_argument('--exp_name',   type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="results/rebuttal")

    args = parser.parse_args()
    
    # Path handling
    if args.nf_type in ('positive', 'negative'):
        args.output_dir = os.path.join(args.output_dir, args.nf_type)

    # Set absolute data path relative to script location
    args.base_dir = os.path.join(DATA_PATHS[args.dataset])
    
    os.makedirs(args.output_dir, exist_ok=True)
    args.image_size = (224, 224)
    
    return args