from sklearn.model_selection import ParameterGrid
import numpy as np

def get_grid(model_type):
    """Returns the hyperparameter search space for each model."""
    if model_type == 'svm':
        return {
            'linear': { 'C': np.logspace(-4, 3, 21) },
            'rbf':    { 'C': np.logspace(-4, 3, 21), 'gamma': np.logspace(-4, 1, 10) }
        }
    if model_type == 'xgb':
        return {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'reg_lambda': [0, 0.1, 1],
            'alpha': [0, 0.1, 1],
            'gamma': [0, 0.1, 1],
            'tree_method': ['hist'], # For faster training
            'device': ['cpu']
        }

def get_default_config(m_old, m_new, method, k_old='linear', k_new='rbf'):
    """
    Generates the full experimental setup.
    """
    return {
        'seed': 0, 
        'size0': 0.2, 
        'size1': 1,
        'model_old': m_old, 
        'kernel_old': k_old,  
        'model_new': m_new, 
        'kernel_new': k_new,  
        'method': method,
        'l_values': np.logspace(-3, 2, 5),
        'NFtype': 'all', 
        'CVmetric': 'balanced_accuracy',
        'p_thresh': 0.15,
        'n_jobs': -1,
        'param_grid_old': get_grid(m_old),
        'param_grid_new': get_grid(m_new)
    }