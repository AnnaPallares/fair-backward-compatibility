import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import normalize

from utils.sens_mapping import sens_loc

def get_Xy(df, dataset_name):
    """
    Extracts X and y from the dataframe provided and performs one-hot
    encoding to get rid of categorical variables.
    """
    dataset_name = dataset_name.lower()  
    dim = df.shape
    # Standard cleanup for  datasets (removing index column and extracting label)
    X = df.iloc[:, 1:(dim[1]-1)]
    y = df.iloc[:, (dim[1]-1)]

    # Specific handling for missing data in Arrhythmia
    if dataset_name == "arrhythmia":
        X = X.replace('?', np.nan)
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.median(), inplace=True)

    X = pd.get_dummies(X, drop_first=True) 
    
    # Ensure y is a single binary column (0 and 1)
    y = pd.get_dummies(y)
    y = y.iloc[:, 1].astype(int) 
    return X, y

def prepare_data(X, y, seed, size0, size1, dataset_name, model_type):
    # Data spliting for old models (subsets of 20%)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    
    n = len(X_tr)
    random.seed(seed)
    ind1 = random.sample(range(n), int(n * size1))
    ind0 = random.sample(ind1, int(n * size0)) # ind0 is a subset of ind1

    # Apply your specific Normalization for SVM
    if model_type == 'svm':
        X0_tr = normalize(X_tr.iloc[ind0, :], axis=0)
        X1_tr = normalize(X_tr.iloc[ind1, :], axis=0)
        X_te_final = normalize(X_te, axis=0)
    else:
        X0_tr, X1_tr, X_te_final = X_tr.iloc[ind0, :], X_tr.iloc[ind1, :], X_te

    sens_idx = sens_loc(dataset_name)
    return {
        'X0': X0_tr, 'y0': y_tr.iloc[ind0],
        'X1': X1_tr, 'y1': y_tr.iloc[ind1], 's1': X_tr.iloc[ind1, sens_idx],
        'X_te': X_te_final, 'y_te': y_te, 's_te': X_te.iloc[:, sens_idx]
    }