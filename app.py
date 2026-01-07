import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ------------------ STYLE ------------------
st.markdown("""
<style>
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("Customer Churn Prediction App")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Churn Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload the churn dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------ SAMPLE DATA ------------------
st.subheader("Sample Dataset")
st.dataframe(df.head())

# ------------------ FEATURE SELECTION ------------------
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# ------------------ DATA CLEANING ------------------
X = X.copy()
X['TotalCharges'] = X['TotalCharges'].replace(' ', np.nan)
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'])
X = X.dropna()
y = y.loc[X.index]

# Encode target
y = y.map({'Yes': 1, 'No': 0})

# ------------------ CHURN DISTRIBUTION ------------------
st.subheader("Churn Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(x=y, ax=ax1)
ax1.set_xticklabels(['No Churn', 'Churn'])
ax1.set_title("Churn Count")
st.pyplot(fig1)

# ------------------ TRAIN MODEL ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ------------------ MODEL ACCURACY ------------------
y_pred_test = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_test)

st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# ------------------ CONFUSION MATRIX ------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_test)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No', 'Yes'],
    yticklabels=['No', 'Yes'],
    ax=ax_cm
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ------------------ ROC CURVE ------------------
st.subheader("ROC Curve")

y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
auc_score = roc_auc_score(y_test, y_prob_test)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax_roc.plot([0, 1], [0, 1], linestyle='--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

# ------------------ PREDICTION SECTION ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Predict Customer Churn")

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

if st.button("Predict"):
    user_data = np.array([[tenure, monthly, total]])
    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    st.progress(int(probability * 100))

    if prediction == 1:
        st.error(f"Customer is likely to CHURN\n\nProbability: {probability:.2%}")
    else:
        st.success(f"Customer is likely to STAY\n\nProbability: {probability:.2%}")

st.markdown("</div>", unsafe_allow_html=True)
