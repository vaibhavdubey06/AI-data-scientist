import streamlit as st
import pandas as pd
from eda_utils import run_eda
# Temporarily comment out ML to test EDA
# from ml_utils import run_ml

st.set_page_config(page_title="Auto Data Sorcerer", layout="wide")
st.title("🔮 Auto Data Sorcerer")

st.sidebar.title("Upload or Use Sample Data")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
use_sample = st.sidebar.checkbox("Use Titanic Sample Dataset")

@st.cache_data
def load_and_clean_data(file_path=None, uploaded_file=None):
    """Load and clean data to avoid PyArrow compatibility issues"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(file_path)
    
    # Convert problematic data types to standard types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert object columns to string to avoid mixed types
            df[col] = df[col].astype(str)
            # Replace 'nan' string with actual NaN
            df[col] = df[col].replace('nan', pd.NA)
        elif 'Int64' in str(df[col].dtype):
            # Convert nullable integer to standard int64
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif 'Float64' in str(df[col].dtype):
            # Convert nullable float to standard float64
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    return df

if file:
    df = load_and_clean_data(uploaded_file=file)
elif use_sample:
    df = load_and_clean_data(file_path="sample_data/titanic.csv")
else:
    st.warning("Please upload a file or use sample dataset.")
    st.stop()

st.success(" Data loaded successfully!")

# Tabs for EDA and ML
tab1, tab2 = st.tabs([" Exploratory Data Analysis", " AutoML Model Training"])

with tab1:
    # Run EDA and get potentially processed dataframe
    df = run_eda(df)
    
    # Check if processed data is available in session state
    if 'df_processed' in st.session_state:
        st.info("📊 Using processed dataset with outlier handling applied.")
        df = st.session_state['df_processed']

with tab2:
    st.info("🚧 AutoML functionality is temporarily disabled due to library conflicts. Working on fixing it!")
    st.write("In the meantime, you can use the comprehensive EDA features in the first tab.")
    # run_ml(df)
