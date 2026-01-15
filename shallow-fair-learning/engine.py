import numpy as np
import logging
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils.metrics import neg_flip_cond

# Global metric map for consistency
METRIC_MAP = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score
}

def engine_fbc_s(X, y, s, f_old, model_type, param_grid, kfold, NFtype, CVmetric, p_thresh, n_jobs):
    """
    FBC-S: Selects best hyperparameters to maximize B-ACC while minimize NFD.
    """
    X_arr, y_arr, s_arr = np.asarray(X), np.asarray(y), np.asarray(s).astype(int)
    
    # Step 1: Maximize Accuracy
    if model_type == 'svm':
        svc_param_grid = []
        for kernel, grid in param_grid.items():
            pg = grid.copy(); pg['kernel'] = [kernel]
            svc_param_grid.append(pg)
        base_model = SVC()
        gs = GridSearchCV(base_model, svc_param_grid, scoring=CVmetric, cv=kfold, n_jobs=n_jobs)
    else:
        base_model = XGBClassifier(n_jobs=n_jobs, tree_method="hist")
        gs = GridSearchCV(base_model, param_grid, scoring=CVmetric, cv=kfold, n_jobs=n_jobs)

    gs.fit(X, y)
    cutoff = gs.best_score_ * (1 - p_thresh)
    accepted = [p for p, sc in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']) if sc >= cutoff]
    
    # Step 2: Minimize NFD
    best_nfd, best_params = float('inf'), (accepted[0] if accepted else None)
    for p in accepted:
        fold_nfds = []
        for tr_idx, vl_idx in kfold.split(X_arr, y_arr):
            clf = SVC(**p) if model_type == 'svm' else XGBClassifier(**p, n_jobs=n_jobs)
            clf.fit(X_arr[tr_idx], y_arr[tr_idx])
            y_pred = clf.predict(X_arr[vl_idx])
            y_old = f_old.predict(X_arr[vl_idx])
            _, _, _, nfr0, nfr1 = neg_flip_cond(y_arr[vl_idx], y_old, y_pred, s_arr[vl_idx], NFtype)
            fold_nfds.append(abs(nfr1 - nfr0))
        
        avg_nfd = np.mean(fold_nfds)
        if avg_nfd < best_nfd:
            best_nfd, best_params = avg_nfd, p
            
    return best_params

def engine_fbc_d(X, y, s, f_old, model_type, param_grid, l_values, kfold, NFtype, CVmetric, p_thresh, n_jobs):
    """
    FBC-D: Differentiable relaxation via weighted Double-Step CV. Optimizes (Params + Lambda + FlipGroup).
    """
    X_arr, y_arr, s_arr = np.asarray(X), np.asarray(y), np.asarray(s).astype(int)
    metric_fn = METRIC_MAP[CVmetric]
    
    # Precompute mask for weight candidates
    y_old_full = f_old.predict(X)
    if NFtype == 'positive': bool_fold = (y_old_full == y_arr) & (y_arr == 1)
    elif NFtype == 'negative': bool_fold = (y_old_full == y_arr) & (y_arr == 0)
    else: bool_fold = (y_old_full == y_arr)

    # Flatten SVM grid if needed for ParameterGrid compatibility
    if model_type == 'svm':
        flat_grid = []
        for k, v in param_grid.items():
            for p in ParameterGrid(v):
                flat_grid.append({'kernel': k, **p})
    else:
        flat_grid = list(ParameterGrid(param_grid))

    results = []
    for p in flat_grid:
        for l1 in l_values:
            for ind_flip in [0, 1]:
                w_full = np.ones_like(y_arr, dtype=float)
                w_full[bool_fold & (s_arr == ind_flip)] += l1
                
                fold_accs, fold_nfds = [], []
                for tr_idx, vl_idx in kfold.split(X_arr, y_arr):
                    model = SVC(**p) if model_type == 'svm' else XGBClassifier(**p, n_jobs=n_jobs)
                    model.fit(X_arr[tr_idx], y_arr[tr_idx], sample_weight=w_full[tr_idx])
                    y_pred = model.predict(X_arr[vl_idx])
                    y_old = f_old.predict(X_arr[vl_idx])
                    fold_accs.append(metric_fn(y_arr[vl_idx], y_pred))
                    _, _, _, nfr0, nfr1 = neg_flip_cond(y_arr[vl_idx], y_old, y_pred, s_arr[vl_idx], NFtype)
                    fold_nfds.append(abs(nfr1 - nfr0))
                
                results.append({'params': p, 'l1': l1, 'ind_flip': ind_flip, 'acc': np.mean(fold_accs), 'nfd': np.mean(fold_nfds)})

    max_acc = max(r['acc'] for r in results)
    candidates = [r for r in results if r['acc'] >= max_acc * (1 - p_thresh)]
    return min(candidates, key=lambda x: x['nfd'])

# --- FBC-C Relaxation (SVM ONLY) ---
try:
    from gurobipy import Model, GRB, quicksum
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def engine_fbc_c(X, y, s, f_old, kernel, best_params, NFtype):
    """
    FBC-C: Convex relaxation (SVM ONLY).
    """
    if not GUROBI_AVAILABLE:
        logging.warning("Gurobi not found. Skipping FBC-C.")
        return None
    
    y_bin = np.where(np.asarray(y) == 0, -1, 1)
    X_arr, s_arr = np.asarray(X), np.asarray(s).astype(int)
    N = len(X_arr)
    C = best_params.get('C', 1.0)
    gamma = best_params.get('gamma', None)

    # Kerenel computation
    K = rbf_kernel(X_arr, X_arr, gamma=gamma) if kernel == "rbf" else linear_kernel(X_arr, X_arr)

    # Fairness constraint vector
    y_old = f_old.predict(X)
    if NFtype == 'positive': bool_mask = (y_old == y) & (y == 1)
    elif NFtype == 'negative': bool_mask = (y_old == y) & (y == 0)
    else: bool_mask = (y_old == y)

    s0_mask, s1_mask = (s_arr == 0) & bool_mask, (s_arr == 1) & bool_mask
    ns0, ns1 = max(np.sum(s0_mask), 1e-6), max(np.sum(s1_mask), 1e-6)

    f_vec = np.array([np.sum(K[i, s0_mask])/ns0 - np.sum(K[i, s1_mask])/ns1 for i in range(N)])
    f_vec = (f_vec - np.mean(f_vec)) / (np.linalg.norm(f_vec) + 1e-9)

    # Problem optimization with gurobi
    m = Model("FBC-C")
    m.Params.LogToConsole = 0
    alpha = m.addVars(N, lb=0, ub=C, name="alpha")
    obj = 0.5 * quicksum(alpha[i]*alpha[j]*y_bin[i]*y_bin[j]*K[i,j] 
                         for i in range(N) for j in range(N)) - quicksum(alpha[i] for i in range(N))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstr(quicksum(alpha[i] * y_bin[i] for i in range(N)) == 0)
    m.addConstr(quicksum(alpha[i] * y_bin[i] * f_vec[i] for i in range(N)) == 0)
    m.optimize()

    a_vals = np.array([alpha[i].X for i in range(N)])

    # Bias calculation
    sv = (a_vals > 1e-5) & (a_vals < C)
    b = np.mean(y_bin[sv] - np.dot(K[sv], a_vals * y_bin)) if np.any(sv) else 0

    return {'alpha': a_vals, 'b': b, 'kernel': kernel, 'gamma': gamma, 'X_train': X_arr, 'y_train_bin': y_bin}