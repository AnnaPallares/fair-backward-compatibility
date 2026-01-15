import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_fairfaces_data(base_dir, seed, sensitive_attribute):
    young = ['0-2','3-9','10-19','20-29']
    old   = ['30-39','40-49','50-59','60-69','more than 70']

    df = pd.read_csv(os.path.join(base_dir, "fairface_label_train.csv"))
    df['file'] = df['file'].apply(lambda x: os.path.join(base_dir, "images", x))
    df = df[df['file'].apply(os.path.exists)]
    df = df[df['age'].isin(young + old)].copy()
    print("raw", df.shape)
    df['target'] = df['age'].apply(lambda x: 0 if x in young else 1)
    df['sensitive'] = df['gender'].map({'Female': 0, 'Male': 1})

    if sensitive_attribute == "gender":
        df['sensitive'] = df['gender'].map({'Female': 0, 'Male': 1})
    elif sensitive_attribute == "race":
        df['sensitive'] = df['race'].apply(lambda r: 0 if r == 'White' else 1)
    else:
        raise ValueError(f"Unsupported sensitive_attribute: {sensitive_attribute}")
    
    df = df.sample(frac=0.25, random_state=seed).reset_index(drop=True)
    print(df.shape)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def load_utkface_data(base_dir, seed, sensitive_attribute):
    young=range(0, 30)
    old=range(50, 100)

    df = pd.read_csv(os.path.join(base_dir, "labels.csv"))
    df['file'] = df['file'].apply(lambda x: os.path.join(base_dir, "images", x))
    df = df[df['file'].apply(os.path.exists)]
    df = df[df['age'].isin(list(young) + list(old))].copy()
    df['target'] = df['age'].apply(lambda x: 0 if x in young else 1)
    df['sensitive'] = df['gender'] 
    print("raw", df.shape)

    if sensitive_attribute == "gender":
        df['sensitive'] = df['gender']
    elif sensitive_attribute == "race":
        df['sensitive'] = df['race'].apply(lambda x: 0 if x == 0 else 1)
    else:
        raise ValueError(f"Unsupported sensitive_attribute: {sensitive_attribute}")
    
    df = df.sample(frac=0.4, random_state=seed).reset_index(drop=True)
    print(df.shape)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def load_fitzpatrick_data(base_dir, seed, sensitive_attribute):
    csv_path = os.path.join(base_dir, "fitzpatrick_train.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df['file'] = df['file'].apply(lambda x: os.path.join(base_dir, os.path.normpath(x)))
    df = df.rename(columns={"diagnosis_label": "target", "tone_label": "sensitive"})
    df = df[df['file'].apply(os.path.exists)] 
    print(df.shape)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["target"], random_state=seed)
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def load_ddi_data(base_dir, seed, sensitive_attribute):
    """
    Load the DDI dataset for binary classification (malignant vs benign),
    with skin tone as the sensitive attribute.
    """
    csv_path = os.path.join(base_dir, "ddi_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found. Make sure 'ddi_metadata.csv' is in the base_dir.")

    df = pd.read_csv(csv_path)

    df['file'] = df['DDI_file'].apply(lambda x: os.path.join(base_dir, "images", x))
    df = df[df['file'].apply(os.path.exists)]
    print(df.shape)

    df['target'] = df['malignant'].astype(int)  # 1 = malignant, 0 = benign
    print(df["target"].shape)

    # Use skin tone as sensitive attribute (continuous: 0–100)
    df['sensitive'] = df['skin_tone'].apply(lambda x: 0 if x < 56 else 1)
    print(df['sensitive'].shape)

    # Optional: filter out missing values if needed
    df = df.dropna(subset=['target', 'sensitive'])

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['target'], random_state=seed)
    print(train_df.shape, val_df.shape)
    print(np.sum(df["sensitive"]==0))
    print(np.sum(df["target"]==0))

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def load_marvel_data(base_dir, seed, sensitive_attribute):
    """
    Load Marvel character image data from CSV and return train/val DataFrames with:
    - target: hero/villain
    - sensitive: gender (or custom logic if needed)
    """
    train_csv = os.path.join(base_dir, "marvel_train_labels.csv")
    val_csv = os.path.join(base_dir, "marvel_valid_labels.csv")
    img_dir = os.path.join(base_dir, "train")  

    # Load both CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(train_df.shape)
    print(val_df.shape)

    # Add full image path
    train_df['file'] = train_df['FileName'].apply(lambda x: os.path.join(base_dir, "train", x))
    val_df['file']   = val_df['FileName'].apply(lambda x: os.path.join(base_dir, "valid", x))

    # Drop if image not found
    train_df = train_df[train_df['file'].apply(os.path.exists)]
    val_df = val_df[val_df['file'].apply(os.path.exists)]

    # Assign target and sensitive
    train_df['target'] = train_df['y']
    train_df['sensitive'] = train_df['s']

    val_df['target'] = val_df['y']
    val_df['sensitive'] = val_df['s']

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)