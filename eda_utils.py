import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def run_eda(df):
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📐 Dataset Info")
    st.text(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column Types:")
    st.write(df.dtypes)

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
