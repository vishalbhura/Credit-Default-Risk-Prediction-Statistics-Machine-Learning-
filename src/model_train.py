import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessing_pipeline, basic_cleaning
from src.feature_engineering import create_features

def train_models(data_path, target='default', save_dir='models', test_size=0.2, random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df = basic_cleaning(df)
    df = create_features(df)
    # Select features - simple heuristic: numeric + a few engineered
    exclude = [target, 'id', 'index', 'source']
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [float, int]]
    X = df[features].fillna(0)
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Preprocessor
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = []  # keep simple for now
    preproc = build_preprocessing_pipeline(numeric_features, categorical_features, scaler=True)

    # Handle imbalance with SMOTE
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = {
        'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'lightgbm': LGBMClassifier(random_state=random_state)
    }

    results = {}
    for name, mdl in models.items():
        pipe = Pipeline([('preproc', preproc), ('model', mdl)])
        print(f"Training {name}...")
        pipe.fit(X_res, y_res)
        preds = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, preds)
        print(f"{name} AUC: {auc:.4f}")
        joblib.dump(pipe, os.path.join(save_dir, f"{name}.joblib"))
        results[name] = auc

    # Save results
    pd.Series(results).to_csv(os.path.join(save_dir, 'results.csv'))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--target', default='default')
    parser.add_argument('--save-dir', default='models')
    args = parser.parse_args()
    train_models(args.data_path, target=args.target, save_dir=args.save_dir)
