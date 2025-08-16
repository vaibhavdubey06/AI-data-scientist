import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from data_quality import run_outlier_analysis, outlier_summary_report


def safe_dataframe_display(df, title="DataFrame"):
    """Safely display dataframe avoiding PyArrow issues"""
    try:
        st.dataframe(df)
    except Exception as e:
        st.warning(f"⚠️ Display issue with {title}. Showing as HTML table instead.")
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

def run_eda(df):
    st.subheader("🔍 Data Preview")
    safe_dataframe_display(df.head(), "Data Preview")

    st.subheader("📐 Dataset Info")
    st.text(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column Types:")
    st.write(df.dtypes)

    st.subheader("🔍 Advanced Column Type Analysis")
    column_types = detect_column_types(df)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Numerical Columns:**")
        if column_types['continuous']:
            st.write("🔸 **Continuous:**", column_types['continuous'])
        if column_types['discrete']:
            st.write("🔹 **Discrete:**", column_types['discrete'])
            
        st.write("**📝 Categorical Columns:**")
        if column_types['categorical_nominal']:
            st.write("🔸 **Nominal:**", column_types['categorical_nominal'])
        if column_types['categorical_ordinal']:
            st.write("🔹 **Ordinal:**", column_types['categorical_ordinal'])
    
    with col2:
        st.write("**🕒 DateTime Columns:**")
        if column_types['datetime']:
            st.write("📅", column_types['datetime'])
        else:
            st.write("None detected")
            
        st.write("**🔤 Text Columns:**")
        if column_types['text']:
            st.write("📄", column_types['text'])
        else:
            st.write("None detected")
            
        st.write("**🆔 ID Columns:**")
        if column_types['id_columns']:
            st.write("🔑", column_types['id_columns'])
        else:
            st.write("None detected")
            
        st.write("**✅ Boolean Columns:**")
        if column_types['boolean']:
            st.write("☑️", column_types['boolean'])
        else:
            st.write("None detected")

    st.subheader("📉 Missing Values")
    st.write(df.isnull().sum())
    st.bar_chart(df.isnull().sum())

    st.subheader("🧹 Handle Missing Values")
    handle_option = st.radio("Choose method:", ["Do Nothing", "Drop Rows", "Fill with Mean", "Fill with Mode"])

    if handle_option == "Drop Rows":
        df.dropna(inplace=True)
        st.success("✅ Missing values dropped!")

    elif handle_option == "Fill with Mean":
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        st.success("✅ Numeric missing values filled with mean!")

    elif handle_option == "Fill with Mode":
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        st.success("✅ All missing values filled with mode!")

    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include='all'))

    st.subheader("🧩 Correlation Heatmap")
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns to compute correlation.")

    st.subheader("📈 Distribution Plot / Pie Chart")
    col = st.selectbox("Select a column to visualize", df.columns)

    if df[col].dtype != "object":
        fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
        st.plotly_chart(fig)
    else:
        fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
        st.plotly_chart(fig)

    # Add outlier analysis section
    st.subheader("🎯 Outlier Analysis")
    
    outlier_tab1, outlier_tab2 = st.tabs(["🔍 Detailed Analysis", "📋 Summary Report"])
    
    with outlier_tab1:
        df = run_outlier_analysis(df)
    
    with outlier_tab2:
        outlier_summary_report(df)
    
    return df

def detect_column_types(df):
    """Detect and categorize column types comprehensively"""
    import re
    
    results = {
        'continuous': [],
        'discrete': [],
        'categorical_nominal': [],
        'categorical_ordinal': [],
        'datetime': [],
        'text': [],
        'boolean': [],
        'id_columns': []
    }
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df[col].dropna())
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Skip if all values are null
        if total_count == 0:
            continue
            
        # Check for DateTime columns
        if df[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df[col]):
            results['datetime'].append(col)
            continue
            
        # Try to parse as datetime
        try:
            # Only try datetime parsing on string columns that might contain dates
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10).astype(str)
                # Check if any values look like dates (contains numbers and separators)
                date_like = any(any(char in str(val) for char in ['-', '/', ':', ' ']) 
                               and any(char.isdigit() for char in str(val)) 
                               for val in sample_values)
                if date_like:
                    pd.to_datetime(df[col].dropna().head(100), errors='raise', infer_datetime_format=True)
                    results['datetime'].append(col)
                    continue
        except (ValueError, TypeError, pd.errors.ParserError):
            pass
            
        # Check for Boolean columns
        if df[col].dtype == 'bool' or set(df[col].dropna().unique()).issubset({0, 1, True, False, 'True', 'False', 'yes', 'no', 'Y', 'N'}):
            results['boolean'].append(col)
            continue
            
        # Check for ID columns (high cardinality, unique values)
        if unique_ratio > 0.95 and unique_count > 100:
            # Additional checks for ID patterns
            sample_values = df[col].dropna().astype(str).head(50)
            if any(re.match(r'^[A-Z0-9]+$', str(val)) for val in sample_values) or col.lower() in ['id', 'userid', 'customer_id', 'order_id']:
                results['id_columns'].append(col)
                continue
                
        # Check for Numerical columns
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int8', 'int16', 'float16']:
            # Check if it's whole numbers (even if float type)
            try:
                is_whole_numbers = df[col].dropna().apply(lambda x: float(x).is_integer()).all()
            except (ValueError, TypeError):
                is_whole_numbers = False
            
            # Discrete numerical criteria
            if (unique_ratio < 0.05 or 
                unique_count < 20 or 
                (df[col].dtype in ['int64', 'int32'] and is_whole_numbers and unique_count < 50)):
                results['discrete'].append(col)
            else:
                results['continuous'].append(col)
            continue
            
        # Check for Text columns (long strings)
        if df[col].dtype == 'object':
            avg_length = df[col].dropna().astype(str).str.len().mean()
            max_length = df[col].dropna().astype(str).str.len().max()
            
            if avg_length > 50 or max_length > 100:
                results['text'].append(col)
                continue
                
            # Check for Categorical columns
            if unique_ratio < 0.5 and unique_count < 50:
                # Check if it might be ordinal (has ordering)
                sample_values = set(df[col].dropna().astype(str).str.lower())
                ordinal_patterns = [
                    {'low', 'medium', 'high'},
                    {'small', 'medium', 'large'},
                    {'poor', 'fair', 'good', 'excellent'},
                    {'bad', 'average', 'good'},
                    {'first', 'second', 'third'},
                    {'primary', 'secondary', 'tertiary'}
                ]
                
                is_ordinal = any(pattern.issubset(sample_values) for pattern in ordinal_patterns)
                
                if is_ordinal:
                    results['categorical_ordinal'].append(col)
                else:
                    results['categorical_nominal'].append(col)
            else:
                results['text'].append(col)
                
    return results
