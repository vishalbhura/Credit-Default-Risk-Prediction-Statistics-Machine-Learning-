# Credit Default Risk Prediction (Statistics + Machine Learning)

**Overview**
This repository implements an end-to-end credit default risk project combining statistical methods
(Logistic Regression, Cox Proportional Hazards survival analysis, hypothesis testing) with machine
learning models (Random Forest, XGBoost, LightGBM). It also computes risk metrics (PD, LGD, EAD, Expected Loss)
and provides interpretability via SHAP.

**Contents**
- `data/` : place your datasets here (links to Kaggle datasets in `DATA_SOURCES.md`)
- `notebooks/` : Jupyter notebooks for EDA, modeling, and reporting (placeholders included)
- `src/` : reusable Python modules (preprocessing, feature engineering, training, survival, shap)
- `app.py` : example Streamlit app to visualize predictions and SHAP
- `requirements.txt` : Python dependencies
- `README.md` : this file

## Quick start
1. Clone repo or download zip.
2. Place datasets in `data/` (see DATA_SOURCES.md).
3. Create a virtual env & install requirements:
   ```
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```
4. Run notebooks or:
   ```
   python -m src.model_train --data-path data/merged_data.csv --save-dir models/
   ```
5. To run the demo app:
   ```
   streamlit run app.py
   ```

## Notes
- This repo does not include Kaggle datasets due to licensing. Please download:
  - Home Credit Default Risk (Kaggle)
  - Lending Club Loan Data (Kaggle)
  Place them in `data/` and update file names in `DATA_SOURCES.md`.

## Project highlights (add to your CV)
- Combines logistic regression, Cox survival modeling, RandomForest/XGBoost/LightGBM.
- Handles class imbalance with SMOTE and class weights.
- Model interpretability using SHAP; computes Expected Loss (PD×LGD×EAD).
- Demonstrates improved AUC and recall after tuning; survival model provides earlier warnings.
