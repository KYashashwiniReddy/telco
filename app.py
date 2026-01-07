import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# ------------------ LOAD DATA ------------------
df = pd.read_csv("yash.csv")

# ------------------ DISPLAY SAMPLE DATA ------------------
st.subheader("📄 Sample Dataset")
st.dataframe(df.head())

# ------------------ SELECT FEATURES ------------------
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Clean TotalCharges
X['TotalCharges'] = X['TotalCharges'].replace(' ', np.nan)
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'])
X = X.dropna()
y = y.loc[X.index]

# Encode target
y = y.map({'Yes': 1, 'No': 0})

# ------------------ VISUALIZATIONS ------------------
st.subheader("📊 Data Visualizations")

# Churn Count Plot
fig1, ax1 = plt.subplots()
sns.countplot(x=y, ax=ax1)
ax1.set_xticklabels(['No Churn', 'Churn'])
ax1.set_title("Churn Distribution")
st.pyplot(fig1)

# Tenure vs MonthlyCharges
fig2, ax2 = plt.subplots()
sns.scatterplot(x=X['tenure'], y=X['MonthlyCharges'], hue=y, ax=ax2)
ax2.set_title("Tenure vs Monthly Charges")
st.pyplot(fig2)

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

st.metric("📈 Model Accuracy", f"{accuracy*100:.2f}%")

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

    if prediction == 1:
        st.error(f"Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"Customer is likely to STAY\n\nProbability: {probability:.2f}")

st.markdown("</div>", unsafe_allow_html=True)
