import numpy as np

def negativeFlip_rate(y_true, y_pred_old, y_pred_new, sens_attr, NFtype='all'):
    """
    Calculates Negative Flip Rates (NFR) per group using vectorized NumPy operations.
    """
    y_true = np.array(y_true).flatten()
    y_old = np.array(y_pred_old).flatten()
    y_new = np.array(y_pred_new).flatten()
    s = np.array(sens_attr).flatten()

    # Define type of fairness (i.e., all samples for DP, positive/negative for EO)
    if NFtype == "positive":
        subset_mask = (y_true == 1)
    elif NFtype == "negative":
        subset_mask = (y_true == 0)
    else:
        subset_mask = np.ones_like(y_true, dtype=bool) 

    # A 'Negative Flip' candidate is a sample the old model got correct
    correct_old_mask = (y_old == y_true) & subset_mask
    
    # A 'Negative Flip' occurs if the new model gets that same sample wrong
    nf_mask = correct_old_mask & (y_new != y_true)

    # Group masks
    mask_s0 = (s == 0)
    mask_s1 = (s == 1)

    # NFR = (Count of Flips in Group) / (Count of samples group correctly predicted by old model)
    denom_s0 = np.sum(correct_old_mask & mask_s0)
    denom_s1 = np.sum(correct_old_mask & mask_s1)

    nfr_s0 = np.sum(nf_mask & mask_s0) / denom_s0 if denom_s0 > 0 else 0.0
    nfr_s1 = np.sum(nf_mask & mask_s1) / denom_s1 if denom_s1 > 0 else 0.0

    return {
        "sensitive NF rate": nfr_s0,
        "non-sensitive NF rate": nfr_s1,
        "Unfair Regression": abs(nfr_s1 - nfr_s0),
        "total_flips": np.sum(nf_mask)
    }

def get_sensitive_target(y_true, y_pred_old, y_pred_new, sens_attr, NFtype):
    """Identifies which group (0 or 1) is suffering more from backward incompatibility, used to weighten in the FBC-D relax.
    """
    stats = negativeFlip_rate(y_true, y_pred_old, y_pred_new, sens_attr, NFtype)
    nfr_map = {0: stats["sensitive NF rate"], 1: stats["non-sensitive NF rate"]}
    # Target the group with the highest NFR for mitigation
    sens_target = max(nfr_map, key=nfr_map.get)
    return sens_target, nfr_map