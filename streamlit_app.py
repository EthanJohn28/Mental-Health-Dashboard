import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load_css("style.css")

df = pd.read_csv("./data/Students_Social_Media_Addiction_FE.csv")

bias_params = joblib.load("bias_params_production.pkl")
model = joblib.load("logistic_model_production.pkl")
scaler = joblib.load("scaler_production.pkl")


MH_bias = bias_params["MH_Bias_Factor"]
A_bias = bias_params["AS_Bias_Factor"]

# Note to self: verify these scores
min_addicted_score = 0
max_addicted_score = 10
min_usage_hrs = 0
max_usage_hrs = 12
min_sleep_hrs = 0
max_sleep_hrs = 12
min_mh_score = 0
max_mh_score = 10

st.title("Student Mental Health Dashboard")
st.write("This interactive tool estimates mental health, addiction, academic risk, and likelihood of social media affecting academic performance.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data Here: ")

    sleep_hrs = st.slider(
        label="Hours of sleep per night",
        min_value=min_sleep_hrs,
        max_value=max_sleep_hrs,
        value=7
    )
    
    usage_hrs = st.slider(
        label="Hours spent on social media per day",
        min_value=min_usage_hrs,
        max_value=max_usage_hrs,
        value=3
    )

    academic_level = st.selectbox(
        "Enter academic level",
        ("High School", "Undergraduate", "Graduate"),
        )

# Calc MH and A scores
def calc_mh_score(usage_hrs,sleep_hrs):
    raw = (
        0.75 * sleep_hrs - 
        0.25 * usage_hrs
    )

    return np.clip(raw + MH_bias, min_mh_score, max_mh_score)

def calc_addicted_score(usage_hrs,sleep_hrs):
    raw = (
        0.75 * usage_hrs +
        0.25 * max(0, 8-sleep_hrs)
    )

    return np.clip(raw + A_bias, min_addicted_score, max_addicted_score)

mh_score_pred = calc_mh_score(usage_hrs,sleep_hrs)

addicted_score_pred = calc_addicted_score(usage_hrs,sleep_hrs)

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




# Integrate model
academic_mapping = {
    "High School": 1,
    "Undergraduate": 2,
    "Graduate": 3
}

academic_level_encoded = academic_mapping[academic_level]

X_input = np.array([[
    sleep_hrs,
    usage_hrs,
    academic_level_encoded

]])

X_input_scaled = scaler.transform(X_input)

pred_class = model.predict(X_input_scaled)[0]
pred_proba = model.predict_proba(X_input_scaled)[0,1]
with col2: 
    st.subheader("Estimated Scores")
    st.metric(
        label="Estimated Addiction Score (0-10):", 
        value = f"{addicted_score_pred:.2f}"
    )
    st.metric(
        label="Estimated Mental Health Score Score (0-10):",
        value= f"{mh_score_pred:.2f}"
    )
    st.metric(
        label="Predicted Academic Risk (0-1): ",
        value=f"{academic_risk_index:.2f}"
    )
    st.subheader("Predicted Academic Impact")
    st.metric(
        label="Likely to Affect Academic Performance?", 
        value="Yes" if pred_class == 1 else "No",
        #delta=f"Confidence: {pred_proba*100:.1f}%"
    )


st.subheader("Score Distribution")

all_addicted_scores = df["Addicted_Score"].values
all_mh_scores = df["Mental_Health_Score"].values
all_ari_scores = df["Academic_Risk_Index"].values

fig, axes = plt.subplots(3,1,figsize=(5,6))

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

axes[2].hist(all_ari_scores, bins=6, color="lightblue", alpha=0.7)
axes[2].axvline(academic_risk_index, color="blue",linestyle="dashed", linewidth=2)
axes[2].set_title("Academic Risk Index Distribution")
axes[2].set_xlabel("Score")
axes[2].set_ylabel("Count")

plt.tight_layout()
st.pyplot(fig)


st.caption("This tool is for educational purposes, not medical advice. ")
