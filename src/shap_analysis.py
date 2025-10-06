import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.preprocessing import basic_cleaning
from src.feature_engineering import create_features

def explain_model(model_path, data_path, model_name='lightgbm', n_samples=1000, save_dir='models'):
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df = basic_cleaning(df)
    df = create_features(df)
    # Extract features used by pipeline (assumes ColumnTransformer with numeric + ohe)
    # For simplicity, pass raw X to explainer after preprocessing numeric part
    X = df.drop(columns=['id','default','source'], errors='ignore').fillna(0)
    # Use KernelExplainer if tree explainer is incompatible with pipeline; try TreeExplainer first
    model = pipe.named_steps['model']
    try:
        explainer = shap.TreeExplainer(model)
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_shap_summary.png")
    except Exception as e:
        print('TreeExplainer failed, fallback to KernelExplainer.', e)
        explainer = shap.KernelExplainer(lambda x: pipe.predict_proba(x)[:,1], X.sample(100, random_state=42))
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
    return True
