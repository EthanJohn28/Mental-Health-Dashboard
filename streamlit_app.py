import streamlit as st
import joblib
import numpy as np

bias_params = joblib.load("bias_params.pkl")
MH_bias = bias_params["MH_Bias_Factor"]
A_bias = bias_params["AS_Bias_Factor"]

st.title("Student Mental Health Dashboard")
st.write("Description goes here: ")

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
        value=0
    )

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

st.subheader("Estimated Scores")
st.metric(
    "Estimated Addiction Score (0-10)", round(addicted_score_pred,2)
)
st.metric(
    "Estimated Mental Health Score Score (0-10)", round(mh_score_pred,2)
)
st.caption("This tool is for educational purposes, not medical advice. ")
