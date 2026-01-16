import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils.metrics import neg_flip_cond
from utils.data_preprocessing import prepare_data
from engine import engine_fbc_s, engine_fbc_d, engine_fbc_c, METRIC_MAP

def run_fbc_experiment(dataset_name, X, y, config):
    """
    Unified execution pipeline.
    config keys: seed, size0, size1, model_old ('svm'/'xgb'), 
    model_new ('svm'/'xgb'), method ('fbc-s'/'fbc-d'/'fbc-c'),
    param_grid_old, param_grid_new, l_values, NFtype, CVmetric, p_thresh, n_jobs
    """
    # 1. Data Preparation
    data = prepare_data(X, y, config['seed'], config['size0'], config['size1'], 
                        dataset_name, config['model_new'])
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['seed'])

    # 2. Train f_old
    print(f"--- Training f_old ({config['model_old']} - {config['kernel_old']}) ---")
    
    if config['model_old'] == 'svm':
        # Select only the grid corresponding to the chosen kernel
        k_old = config['kernel_old']
        svc_grid = [{'kernel': [k_old], **config['param_grid_old'][k_old]}]
        base_old = SVC()
    else:
        svc_grid = config['param_grid_old']
        base_old = XGBClassifier(n_jobs=config['n_jobs'], tree_method="hist")
        
    f_old = GridSearchCV(base_old, svc_grid, scoring=config['CVmetric'], cv=kfold, n_jobs=config['n_jobs'])
    f_old.fit(data['X0'], data['y0'])
    
    # 3. Train f_new 
    print(f"--- Training f_new ({config['model_new']} - {config['kernel_new']}) ---")
    
    if config['model_new'] == 'svm':
        # Select only the grid corresponding to the chosen kernel
        k_new = config['kernel_new']
        svc_grid_new = [{'kernel': [k_new], **config['param_grid_new'][k_new]}]
        base_new = SVC()
    else:
        svc_grid_new = config['param_grid_new']
        base_new = XGBClassifier(n_jobs=config['n_jobs'], tree_method="hist")

    f_new = GridSearchCV(base_new, svc_grid_new, scoring=config['CVmetric'], cv=kfold, n_jobs=config['n_jobs'])
    f_new.fit(data['X1'], data['y1'])

    # 4. Mitigation Step
    print(f"--- Running Mitigation: {config['method']} ---")
    f_mitig = None
    
    # Prepare the specific grid for the NEW model based on user kernel choice
    if config['model_new'] == 'svm':
        current_k = config['kernel_new']
        mitig_grid = {current_k: config['param_grid_new'][current_k]}
    else:
        mitig_grid = config['param_grid_new']['xgb']

    if config['method'] == 'fbc-s':
        best_p = engine_fbc_s(data['X1'], data['y1'], data['s1'], f_old, 
                              config['model_new'], mitig_grid, 
                              kfold, config['NFtype'], config['CVmetric'], 
                              config['p_thresh'], config['n_jobs'])
        f_mitig = SVC(**best_p) if config['model_new'] == 'svm' else XGBClassifier(**best_p)
        f_mitig.fit(data['X1'], data['y1'])

    elif config['method'] == 'fbc-d':
        best = engine_fbc_d(data['X1'], data['y1'], data['s1'], f_old, 
                            config['model_new'], mitig_grid, 
                            config['l_values'], kfold, config['NFtype'], 
                            config['CVmetric'], config['p_thresh'], config['n_jobs'])
        
        y_old_train = f_old.predict(data['X1'])

        mask = (y_old_train == np.array(data['y1']))
        if config['NFtype'] == 'positive': mask &= (np.array(data['y1']) == 1)
        elif config['NFtype'] == 'negative': mask &= (np.array(data['y1']) == 0)
        
        weights = np.ones(len(data['y1']))
        weights[mask & (np.array(data['s1']) == best['ind_flip'])] += best['l1']
        
        f_mitig = SVC(**best['params']) if config['model_new'] == 'svm' else XGBClassifier(**best['params'])
        f_mitig.fit(data['X1'], data['y1'], sample_weight=weights)

    elif config['method'] == 'fbc-c':
        f_mitig_dict = engine_fbc_c(data['X1'], data['y1'], data['s1'], f_old, 
                                   config['kernel_new'], f_new.best_params_, 
                                   config['NFtype'])
        
        from engine import linear_kernel, rbf_kernel
        def predict_c(X_test):
            if f_mitig_dict['kernel'] == 'rbf':
                K = rbf_kernel(X_test, f_mitig_dict['X_train'], gamma=f_mitig_dict['gamma'])
            else:
                K = linear_kernel(X_test, f_mitig_dict['X_train'])
            raw = np.dot(K, f_mitig_dict['alpha'] * f_mitig_dict['y_train_bin']) + f_mitig_dict['b']
            return np.where(raw >= 0, 1, 0)
        
        y_pred_mit = predict_c(data['X_te'])

    # 5. Final Evaluation
    y_pred_old = f_old.predict(data['X_te'])
    y_pred_new = f_new.predict(data['X_te'])
    if config['method'] != 'fbc-c':
        y_pred_mit = f_mitig.predict(data['X_te'])

    y_true = np.array(data['y_te'])
    s_true = np.array(data['s_te'])

    res_new = neg_flip_cond(y_true, y_pred_old, y_pred_new, s_true, config['NFtype'])
    res_mit = neg_flip_cond(y_true, y_pred_old, y_pred_mit, s_true, config['NFtype'])
    
    results = {
        'acc_old': accuracy_score(y_true, y_pred_old),
        'acc_new': accuracy_score(y_true, y_pred_new),
        'acc_mit': accuracy_score(y_true, y_pred_mit),
        'nfd_new': abs(res_new[4] - res_new[3]),
        'nfd_mit': abs(res_mit[4] - res_mit[3])
    }
    
    return results