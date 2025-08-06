import streamlit as st
import pandas as pd
from eda_utils import run_eda
from ml_utils import run_ml

st.set_page_config(page_title="Auto Data Sorcerer", layout="wide")
st.title("🔮 Auto Data Sorcerer")

st.sidebar.title("Upload or Use Sample Data")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
use_sample = st.sidebar.checkbox("Use Titanic Sample Dataset")

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
elif use_sample:
    df = pd.read_csv("sample_data/titanic.csv")
else:
    st.warning("Please upload a file or use sample dataset.")
    st.stop()

st.success(" Data loaded successfully!")

# Tabs for EDA and ML
tab1, tab2 = st.tabs([" Exploratory Data Analysis", " AutoML Model Training"])

with tab1:
    run_eda(df)

with tab2:
    run_ml(df)
