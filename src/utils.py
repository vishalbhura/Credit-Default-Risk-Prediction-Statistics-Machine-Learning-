import os
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_csv(path):
    return pd.read_csv(path)

def save_df(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def merge_homecredit_lending(home_path, lending_path, out_path='data/merged_data.csv', n_samples=None):
    """Simple merge logic: outer-append with source label. Customize per your schema."""
    hc = pd.read_csv(home_path)
    lc = pd.read_csv(lending_path)
    hc['source'] = 'home_credit'
    lc['source'] = 'lending_club'
    df = pd.concat([hc, lc], axis=0, ignore_index=True)
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)
    return df
