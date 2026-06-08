import numpy as np
import logging
from sklearn.model_selection import ParameterGrid, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils.metrics import neg_flip_cond

# Global metric map for consistency
METRIC_MAP = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score
}

from joblib import Parallel, delayed

def evaluate_candidate_s(p, X_arr, y_arr, s_arr, f_old, model_type, n_splits, cv_seed, NFtype, metric_fn):
    """Evaluate a single hyperparameter config: compute BOTH accuracy and NFD in one CV pass."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    fold_accs, fold_nfds = [], []
    for tr_idx, vl_idx in kfold.split(X_arr, y_arr):
        clf = SVC(**p) if model_type == 'svm' else XGBClassifier(**p)
        clf.fit(X_arr[tr_idx], y_arr[tr_idx])
        y_pred = clf.predict(X_arr[vl_idx])
        y_old = f_old.predict(X_arr[vl_idx])
        fold_accs.append(metric_fn(y_arr[vl_idx], y_pred))
        _, _, _, nfr0, nfr1 = neg_flip_cond(y_arr[vl_idx], y_old, y_pred, s_arr[vl_idx], NFtype)
        fold_nfds.append(abs(nfr1 - nfr0))
    return np.mean(fold_accs), np.mean(fold_nfds), p

def engine_fbc_s(X, y, s, f_old, model_type, param_grid, kfold, NFtype, CVmetric, p_thresh, n_jobs):
    """
    FBC-S: Single CV pass computes accuracy and NFD simultaneously.
    Step 1: Filter candidates by accuracy threshold.
    Step 2: PICK (not re-CV) the candidate with the lowest NFD from the same CV results.
    """
    X_arr, y_arr, s_arr = np.asarray(X), np.asarray(y), np.asarray(s).astype(int)
    metric_fn = METRIC_MAP[CVmetric]

    # Build flat parameter grid
    if model_type == 'svm':
        flat_grid = [{'kernel': k, **p} for k, v in param_grid.items() for p in ParameterGrid(v)]
    else:
        flat_grid = list(ParameterGrid(param_grid))

    # Single parallelized CV pass: compute both accuracy AND NFD for every candidate
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_candidate_s)(p, X_arr, y_arr, s_arr, f_old, model_type,
                                      kfold.n_splits, kfold.random_state, NFtype, metric_fn)
        for p in flat_grid
    )

    # Step 1: apply accuracy threshold
    max_acc = max(r[0] for r in results)
    cutoff = max_acc * (1 - p_thresh)
    accepted = [(acc, nfd, p) for acc, nfd, p in results if acc >= cutoff]

    if not accepted:
        return flat_grid[0]  # fallback: return first candidate

    # Step 2: Pick the accepted candidate with the lowest NFD
    _, _, best_params = min(accepted, key=lambda x: x[1])
    return best_params

def evaluate_params_d(p, l1, ind_flip, X_arr, y_arr, s_arr, f_old, model_type, n_splits, cv_seed, NFtype, bool_fold, metric_fn, n_jobs):
    # Create a fresh kfold inside each worker to avoid shared/stale state across parallel jobs
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
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
    
    return {'params': p, 'l1': l1, 'ind_flip': ind_flip, 'acc': np.mean(fold_accs), 'nfd': np.mean(fold_nfds)}

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

    # Parallelized sweep — pass n_splits and seed, not the kfold object, to avoid stale state
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params_d)(p, l1, ind_flip, X_arr, y_arr, s_arr, f_old, model_type,
                                   kfold.n_splits, kfold.random_state, NFtype, bool_fold, metric_fn, 1)
        for p in flat_grid
        for l1 in l_values
        for ind_flip in [0, 1]
    )

    max_acc = max(r['acc'] for r in results)
    candidates = [r for r in results if r['acc'] >= max_acc * (1 - p_thresh)]
    return min(candidates, key=lambda x: x['nfd'])

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# --- FBC-C Relaxation (SVM ONLY) ---
try:
    from gurobipy import Model, GRB, quicksum
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