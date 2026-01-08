import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/Users/ethanjohn/Desktop/Data Science/Projects/MentalHealthDashboard/Students_Social_Media_Addiction_FE.csv")

bias_params = joblib.load("bias_params.pkl")
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
MH_bias = bias_params["MH_Bias_Factor"]
A_bias = bias_params["AS_Bias_Factor"]

# Note to self: verify these scores
min_addicted_score = 0
max_addicted_score = 10
min_usage_hrs = 0
max_usage_hrs = 24
min_sleep_hrs = 0
max_sleep_hrs = 12
min_mh_score = 0
max_mh_score = 10

st.title("Student Mental Health Dashboard")
st.write("This interactive tool estimates mental health, addiction, academic risk, and likelihood of social media affecting academic performance.")

col1, col2 = st.columns(2)

with col1:

    sleep_hrs = st.slider(
        label="Hours of sleep per night",
        min_value=0,
        max_value=12,
        value=7
    )
    
    usage_hrs = st.slider(
        label="Hours spent on social media per day",
        min_value=0,
        max_value=12,
        value=3
    )

    conflicts_count = st.slider(
        label="Number of social media conflicts",
        min_value=0,
        max_value=50,
        value=0)

    academic_level = st.selectbox(
        "Enter academic level",
        ("High School", "Undergraduate", "Graduate"),
        placeholder="Enter academic level")

# Calc MH and A scores
def calc_mh_score(usage_hrs,sleep_hrs,conflicts):
    raw = (
        0.6 * sleep_hrs - 
        0.25 * usage_hrs - 
        0.15 * conflicts
    )

    return np.clip(raw + MH_bias, 0, 10)

def calc_addicted_score(usage_hrs,sleep_hrs,conflicts):
    raw = (
        0.7 * usage_hrs +
        0.15 * conflicts + 
        0.15 * max(0, 8-sleep_hrs)
    )

    return np.clip(raw + A_bias, 0, 10)

mh_score_pred = calc_mh_score(usage_hrs,sleep_hrs,conflicts_count)

addicted_score_pred = calc_addicted_score(usage_hrs,sleep_hrs,conflicts_count)

def rescale(value,min_val,max_val):
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

addicted_scaled = rescale(addicted_score_pred, min_addicted_score, max_addicted_score)
usage_scaled = rescale(usage_hrs,min_usage_hrs,max_usage_hrs)
sleep_scaled = rescale(sleep_hrs,min_sleep_hrs,max_sleep_hrs)
mh_scaled = rescale(mh_score_pred,min_mh_score,max_mh_score)

academic_risk_index = np.mean([
    addicted_scaled,
    usage_scaled,
    1 - sleep_scaled,
    1 - mh_scaled
])
st.subheader("Estimated Scores")
st.metric(
    "Estimated Addiction Score (0-10)", round(addicted_score_pred,2)
)
st.metric(
    "Estimated Mental Health Score Score (0-10)", round(mh_score_pred,2)
)
st.metric(
    "Predicted Academic Risk (0-1)", f"{academic_risk_index:.2f}"
)


# Integrate model
academic_mapping = {
    "High School": 1,
    "Undergraduate": 2,
    "Graduate": 3
}

academic_level_encoded = academic_mapping[academic_level]

X_input = np.array([[
    addicted_score_pred,
    usage_hrs,
    sleep_hrs,
    mh_score_pred,
    academic_level_encoded,
    conflicts_count,
    academic_risk_index
]])

X_input_scaled = scaler.transform(X_input)

pred_class = model.predict(X_input_scaled)[0]
pred_proba = model.predict_proba(X_input_scaled)[0,1]

st.subheader("Predicted Academic Impact")
st.metric(
    label="Likely to Affect Academic Performance?", 
    value="Yes" if pred_class == 1 else "No",
    delta=f"Confidence: {pred_proba*100:.1f}%"
)

with col2:
    st.subheader("Score Distribution")

    all_addicted_scores = df["Addicted_Score"].values
    all_mh_scores = df["Mental_Health_Score"].values

    fig, axes = plt.subplots(2,1,figsize=(6,8))

    axes[0].hist(all_addicted_scores, bins=6, color="lightcoral", alpha=0.7)
    axes[0].axvline(addicted_score_pred, color="red", linestyle="dashed", linewidth=2)
    axes[0].set_title("Addiction Score Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")

    axes[1].hist(all_mh_scores, bins=6, color="lightgreen", alpha=0.7)
    axes[1].axvline(mh_score_pred, color="green", linestyle="dashed", linewidth=2)
    axes[1].set_title("Mental Health Score Distribution")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)






st.caption("This tool is for educational purposes, not medical advice. ")
