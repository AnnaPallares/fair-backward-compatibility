import time
from sklearn.model_selection import ParameterGrid

def estimate_runtime(config):
    # Calculate combinations for f_new
    if config['model_new'] == 'svm':
        # Sum of combinations across all kernels
        h_combs = sum(len(list(ParameterGrid(g))) for g in config['param_grid_new'].values())
    else:
        h_combs = len(list(ParameterGrid(config['param_grid_new'])))

    # FBC-S: Just h_combs * folds
    # FBC-D: h_combs * lambdas * 2 groups * folds
    multiplier = (len(config['l_values']) * 2) if config['method'] == 'fbc-d' else 1
    total_fits = h_combs * multiplier * 5 # 5 for k-fold
    
    return total_fits