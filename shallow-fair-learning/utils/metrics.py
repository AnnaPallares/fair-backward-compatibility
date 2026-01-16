import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def neg_flip_cond(y_true,y_mod1,y_mod2,s0,NFtype):
    
    y_true = np.array(y_true)

    s0 = np.array(s0).astype(int) # Converts [True, False] to [1, 0]
    
    # Setting fairness definition (DP/EO) 
    if NFtype == 'positive':
        NFcond = [y_true[i]==1 for i in range(len(y_true))]
    elif NFtype == 'negative':
        NFcond = [y_true[i]==0 for i in range(len(y_true))]
    else:
        NFcond = [True]*len(y_true)
    
    y_true,y_mod1,y_mod2, s0 = list(y_true),list(y_mod1),list(y_mod2),list(s0)
    s1 = [-i+1 for i in s0] 
    count = 0
    countS0 = 0
    countS1 = 0
    denS0 = 0
    denS1 = 0
    for i in range(len(y_true)):
        if y_true[i] == y_mod1[i] and NFcond[i]:
            denS0 = denS0+s0[i]
            denS1 = denS1+s1[i]
            if y_true[i] != y_mod2[i]:
                count = count+1
                countS0 = countS0 + s0[i]
                countS1 = countS1 + s1[i]
    
    nfrS0 = 0
    nfrS1 = 0    
    if denS0 > 0:
        nfrS0 = countS0/denS0
    if denS1 > 0:
        nfrS1 = countS1/denS1       
    
    return count,countS0,countS1,nfrS0,nfrS1

def acc_nf(
    y_test, 
    y_pred_old, 
    y_pred_new, 
    y_pred_mitig, 
    boolS0_test, 
    NFtype: str = "all"
) -> pd.DataFrame:
    """
    Computes accuracy and negative flip statistics.
    Returns a one-row DataFrame.
    """

    # Metrics
    ACCold = accuracy_score(y_test, y_pred_old)
    ACCnew = accuracy_score(y_test, y_pred_new)
    ACCmitig = accuracy_score(y_test, y_pred_mitig)

    AUCold = balanced_accuracy_score(y_test, y_pred_old)
    AUCnew = balanced_accuracy_score(y_test, y_pred_new)
    AUCmitig = balanced_accuracy_score(y_test, y_pred_mitig)

    # Flip condition: Only consider y=1 if NFtype is "positive"
    NFcond = [(y == 1) if NFtype == 'positive' else True for y in y_test]

    # Negative flip stats
    NFref = neg_flip_cond(y_test, y_pred_old, y_pred_new, boolS0_test, NFtype)
    NFmitig = neg_flip_cond(y_test, y_pred_old, y_pred_mitig, boolS0_test, NFtype)

    NFRDref = abs(NFref[3] - NFref[4])
    NFRDmitig = abs(NFmitig[3] - NFmitig[4])

    # Create result row
    row = {
        "ACC_old": round(ACCold, 4),
        "ACC_new": round(ACCnew, 4),
        "ACC_mitig": round(ACCmitig, 4),
        "AUC_old": round(AUCold, 4),
        "AUC_new": round(AUCnew, 4),
        "AUC_mitig": round(AUCmitig, 4),
        "NF_ref": NFref[0],
        "NF_mitig": NFmitig[0],
        "NFRD_ref": round(NFRDref, 4),
        "NFRD_mitig": round(NFRDmitig, 4)
    }

    return pd.DataFrame([row])
