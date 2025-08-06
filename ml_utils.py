import streamlit as st
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run_ml(df):
    st.subheader("🤖 Train Machine Learning Model")

    target = st.selectbox("🎯 Select Target Variable", df.columns)

    if not target:
        st.warning("Please select a valid target column.")
        return

    df_ml = df.copy()

    # Label encode object (categorical) columns
    obj_cols = df_ml.select_dtypes(include='object').columns
    for col in obj_cols:
        df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))

    X = df_ml.drop(columns=[target])
    y = df_ml[target]

    if X.empty:
        st.warning("No usable features found for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    problem_type = st.radio("📌 Select Problem Type", ["Classification", "Regression"])

    if problem_type == "Classification":
        model = LazyClassifier()
    else:
        model = LazyRegressor()

    st.info("⏳ Training models. Please wait...")
    models, predictions = model.fit(X_train, X_test, y_train, y_test)

    st.success("✅ Training Complete!")
    st.dataframe(models)
