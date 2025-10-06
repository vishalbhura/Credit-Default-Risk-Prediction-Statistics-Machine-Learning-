# Credit Default Risk Prediction (Statistics + Machine Learning)

**Overview**
This repository implements an end-to-end credit default risk project combining statistical methods
(Logistic Regression, Cox Proportional Hazards survival analysis, hypothesis testing) with machine
learning models (Random Forest, XGBoost, LightGBM). It also computes risk metrics (PD, LGD, EAD, Expected Loss)
and provides interpretability via SHAP.

**Contents**
- `notebooks/` : Jupyter notebooks for EDA, modeling, and reporting (placeholders included)
- `src/` : reusable Python modules (preprocessing, feature engineering, training, survival, shap)
- `requirements.txt` : Python dependencies
- `README.md` : this file

## Project highlights 
- Combines logistic regression, Cox survival modeling, RandomForest/XGBoost/LightGBM.
- Handles class imbalance with SMOTE and class weights.
- Model interpretability using SHAP; computes Expected Loss (PD×LGD×EAD).
- Demonstrates improved AUC and recall after tuning; survival model provides earlier warnings.
