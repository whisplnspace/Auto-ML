import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="AutoML with Gemini", layout="wide")

st.title("🤖 AutoML Trainer with Smart Algorithm Selection")
st.markdown("Upload your dataset (CSV) and let the app detect and train the best ML model automatically.")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("Train Model"):
        with st.spinner("🔍 Analyzing and training..."):

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Handle missing and categorical values
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            # Check task type
            if y.nunique() <= 20 and y.dtype in ['object', 'bool', 'category']:
                task = "classification"
            elif y.nunique() <= 20 and y.dtype in ['int64']:
                task = "classification"
            else:
                task = "regression"

            # Encode target if classification
            if task == "classification" and y.dtype == 'object':
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            best_model = None
            best_score = -np.inf
            model_results = {}

            if task == "classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC()
                }
                scoring = 'accuracy'
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "SVR": SVR()
                }
                scoring = 'r2'

            for name, model in models.items():
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                try:
                    pipe.fit(X_train, y_train)
                    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring=scoring).mean()
                    model_results[name] = (pipe, score)
                    if score > best_score:
                        best_score = score
                        best_model = (name, pipe)
                except Exception as e:
                    st.warning(f"{name} failed to train: {e}")

            if best_model:
                model_name, model_pipeline = best_model
                y_pred = model_pipeline.predict(X_test)
                st.success(f"✅ Best Model: {model_name} | Score: {round(best_score, 4)}")

                if task == "classification":
                    st.subheader("📊 Classification Report")
                    st.text(classification_report(y_test, y_pred))
                    st.subheader("🧩 Confusion Matrix")
                    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
                else:
                    st.subheader("📈 Regression Metrics")
                    st.write(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
                    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

                # Save model
                os.makedirs("saved_models", exist_ok=True)
                model_path = f"saved_models/{model_name.replace(' ', '_')}.pkl"
                joblib.dump(model_pipeline, model_path)
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download Best Model", f, file_name=f"{model_name}.pkl")

            else:
                st.error("❌ No model trained successfully.")

