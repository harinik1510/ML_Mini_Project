import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.markdown("""
<style>
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Approval Prediction")
st.subheader("Comparison of Decision Tree and Support Vector Machine")

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    # Update the path if necessary
    return pd.read_csv("dataset/train.csv")

df = load_data()

# -------------------- Preprocessing Function --------------------
def preprocess_data(df):
    df = df.copy()

    # Drop Loan_ID if present (for Kaggle dataset compatibility)
    df.drop("Loan_ID", axis=1, inplace=True, errors="ignore")

    # Handle missing values
    for col in ["Gender", "Married", "Dependents",
                "Education", "Self_Employed", "Property_Area"]:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    if "LoanAmount" in df.columns:
        df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    if "Loan_Amount_Term" in df.columns:
        df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

    if "Credit_History" in df.columns:
        df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    # Convert Dependents column
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Encode target variable
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # One-hot encoding for categorical variables
    df = pd.get_dummies(
        df,
        columns=["Gender", "Married", "Education",
                 "Self_Employed", "Property_Area"],
        drop_first=True
    )

    return df

# -------------------- Apply Preprocessing --------------------
df_processed = preprocess_data(df)

# -------------------- Feature and Target --------------------
X = df_processed.drop("Loan_Status", axis=1)
y = df_processed["Loan_Status"]

# -------------------- Feature Scaling --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Train Models --------------------
@st.cache_resource
def train_models(X_train, y_train):
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)

    dt_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    return dt_model, svm_model

dt_model, svm_model = train_models(X_train, y_train)

# -------------------- Evaluation Metrics --------------------
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

dt_pred = dt_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

dt_metrics = compute_metrics(y_test, dt_pred)
svm_metrics = compute_metrics(y_test, svm_pred)

# -------------------- User Input Form --------------------
st.header("📝 Enter Applicant Details")

with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        self_emp = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000)
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=200000, max_value=1000000, value=500000)

    with col3:
        loan_term = st.selectbox("Loan Amount Term (Months)", [120, 180, 240, 300, 360])
        credit_history = st.selectbox("Credit History", [1.0, 0.0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit_button = st.form_submit_button("🔍 Predict Loan Status")

# -------------------- Prediction and Dynamic Visualizations --------------------
if submit_button:

    # Prepare input data
    input_dict = {
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_emp == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # Predictions
    dt_prediction = dt_model.predict(input_scaled)[0]
    svm_prediction = svm_model.predict(input_scaled)[0]

    dt_prob = dt_model.predict_proba(input_scaled)[0]
    svm_prob = svm_model.predict_proba(input_scaled)[0]

    # -------------------- Display Predictions --------------------
    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Decision Tree",
            "Approved" if dt_prediction == 1 else "Rejected",
            f"Confidence: {dt_prob[1]:.2%}"
        )
    with col2:
        st.metric(
            "Support Vector Machine",
            "Approved" if svm_prediction == 1 else "Rejected",
            f"Confidence: {svm_prob[1]:.2%}"
        )

    # -------------------- Dynamic Graphs --------------------
    st.subheader("📊 Model Confidence Comparison")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(["Decision Tree", "SVM"], [dt_prob[1], svm_prob[1]])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Approval Probability")
    st.pyplot(fig1)

    st.subheader("📈 Approval vs Rejection Probability")
    prob_df = pd.DataFrame({
        "Outcome": ["Rejected", "Approved"],
        "Decision Tree": dt_prob,
        "SVM": svm_prob
    })
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    prob_df.set_index("Outcome").plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Probability")
    st.pyplot(fig2)

    # -------------------- Static Evaluation --------------------
    st.subheader("📈 Model Performance Comparison")
    metrics_df = pd.DataFrame({
        "Decision Tree": dt_metrics,
        "SVM": svm_metrics
    })
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    metrics_df.plot(kind="bar", ax=ax3)
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)

    # TXT Metrics
    st.subheader("📄 Metrics Comparison")
    metrics_text = f"""
Decision Tree:
Accuracy : {dt_metrics['Accuracy']:.4f}
Precision: {dt_metrics['Precision']:.4f}
Recall   : {dt_metrics['Recall']:.4f}
F1 Score : {dt_metrics['F1 Score']:.4f}

Support Vector Machine:
Accuracy : {svm_metrics['Accuracy']:.4f}
Precision: {svm_metrics['Precision']:.4f}
Recall   : {svm_metrics['Recall']:.4f}
F1 Score : {svm_metrics['F1 Score']:.4f}
"""
    st.code(metrics_text, language="text")

    # Confusion Matrices
    st.subheader("📌 Confusion Matrices")
    col1, col2 = st.columns(2)

    with col1:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, dt_pred),
                    annot=True, fmt="d", cmap="Blues", ax=ax4)
        ax4.set_title("Decision Tree")
        st.pyplot(fig4)

    with col2:
        fig5, ax5 = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, svm_pred),
                    annot=True, fmt="d", cmap="Greens", ax=ax5)
        ax5.set_title("SVM")
        st.pyplot(fig5)

    # Decision Tree Visualization
    st.subheader("🌳 Decision Tree Visualization")
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    plot_tree(
        dt_model,
        feature_names=X.columns,
        class_names=["Rejected", "Approved"],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax7
    )
    st.pyplot(fig7)

    # Feature Importance
    st.subheader("⭐ Feature Importance")
    importance = pd.Series(
        dt_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    fig8, ax8 = plt.subplots(figsize=(5, 3))
    importance.plot(kind="barh", ax=ax8)
    ax8.set_xlabel("Importance Score")
    st.pyplot(fig8)

