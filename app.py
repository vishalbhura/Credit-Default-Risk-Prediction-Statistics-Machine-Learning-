import streamlit as st
import pandas as pd
import joblib
from src.utils import load_csv

st.title('Credit Default Risk Demo')
st.markdown('Upload a CSV with features (or use sample) and a trained model (.joblib) to predict PD and show SHAP.')

uploaded = st.file_uploader('Upload CSV', type=['csv'])
model_file = st.file_uploader('Upload model (.joblib)', type=['joblib'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df.head())
    if model_file:
        pipe = joblib.load(model_file)
        preds = pipe.predict_proba(df)[:,1]
        df['pd_pred'] = preds
        st.write('Predicted PD (probability of default):')
        st.write(df[['pd_pred']].describe())
        st.download_button('Download predictions', df.to_csv(index=False), 'preds.csv')
