import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    # Example features - adjust to dataset columns
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1e-9)
    if 'emp_length' in df.columns:
        # convert strings like '10+ years' to numeric where possible
        df['emp_length_num'] = df['emp_length'].astype(str).str.extract('(\d+)').astype(float)
    # Flag prior delinquencies
    if 'delinq_2yrs' in df.columns:
        df['prior_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    return df
