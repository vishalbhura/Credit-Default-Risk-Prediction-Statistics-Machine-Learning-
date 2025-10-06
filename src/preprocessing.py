import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def basic_cleaning(df):
    df = df.copy()
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def build_preprocessing_pipeline(numeric_features, categorical_features, scaler=True):
    numeric_transformers = [('imputer', SimpleImputer(strategy='median'))]
    if scaler:
        numeric_transformers.append(('scaler', StandardScaler()))
    numeric_pipeline = Pipeline(numeric_transformers)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop')

    return preprocessor
