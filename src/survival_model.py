import pandas as pd
from lifelines import CoxPHFitter
from src.preprocessing import basic_cleaning
from src.feature_engineering import create_features

def train_cox(data_path, duration_col='time_to_default', event_col='default', covariates=None):
    df = pd.read_csv(data_path)
    df = basic_cleaning(df)
    df = create_features(df)
    if covariates is None:
        covariates = [c for c in df.columns if c not in [duration_col, event_col, 'id', 'index', 'source']]
    cph = CoxPHFitter()
    df_sub = df[[duration_col, event_col] + covariates].dropna()
    cph.fit(df_sub, duration_col=duration_col, event_col=event_col)
    return cph
