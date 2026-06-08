import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import altair as alt
from utils import rescale, create_histogram, load_css
from model_utils import calc_mh_score, calc_addicted_score
import requests

def generate_explanation(
    sleep_hrs,
    usage_hrs,
    addicted_score,
    mental_health_score,
    academic_risk_index,
    prediction
):

    prompt = f"""
You are an educational analytics assistant.

Student data:
- Sleep: {sleep_hrs} hours
- Social media: {usage_hrs} hours
- Addiction score: {addicted_score:.1f}/10
- Mental health score: {mental_health_score:.1f}/10
- Academic risk index: {academic_risk_index:.2f}
- Academic impact prediction: {"Yes" if prediction == 1 else "No"}

Explain:
1. What these results mean
2. The most important factors affecting the prediction
3. Three practical recommendations

Keep the response under 150 words.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        return response.json()["response"]

    except Exception as e:
        return f"AI explanation unavailable: {e}"


st.set_page_config(
    page_title="Student Mental Health Dashboard",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)
with st.spinner("Loading Dashboard..."):
    time.sleep(0.3)

#load_css("style.css")

df = pd.read_csv("./data/Students_Social_Media_Addiction_FE.csv")

bias_params = joblib.load("bias_params_production.pkl")
model = joblib.load("logistic_model_production.pkl")
scaler = joblib.load("scaler_production.pkl")

MH_bias = bias_params["MH_Bias_Factor"]
A_bias = bias_params["AS_Bias_Factor"]

min_addicted_score = 0
max_addicted_score = 10
min_usage_hrs = 0
max_usage_hrs = 12
min_sleep_hrs = 0
max_sleep_hrs = 12
min_mh_score = 0
max_mh_score = 10

st.title("Student Mental Health Dashboard")
st.subheader("Academic Risk Predictor")
st.write("This interactive tool estimates mental health, addiction, academic risk, and likelihood of social media affecting academic performance.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Data Here: ")
    
    expander = st.expander("Instructions")
    expander.write("""
    Please enter your daily sleep hours and social media usage, and select your academic level using the inputs below.  
    This information will help the model estimate your mental health and social media addiction risk.
    """)
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

mh_score_pred = calc_mh_score(usage_hrs, sleep_hrs)

addicted_score_pred = calc_addicted_score(usage_hrs, sleep_hrs)

addicted_scaled = rescale(addicted_score_pred, min_addicted_score, max_addicted_score)
usage_scaled = rescale(usage_hrs, min_usage_hrs, max_usage_hrs)
sleep_scaled = rescale(sleep_hrs, min_sleep_hrs, max_sleep_hrs)
mh_scaled = rescale(mh_score_pred, min_mh_score, max_mh_score)

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

X_input = pd.DataFrame([{
    "Sleep_Hours_Per_Night": sleep_hrs,
    "Avg_Daily_Usage_Hours": usage_hrs,
    "Academic_Level_Encoded": academic_level_encoded,
}])

X_input_scaled = scaler.transform(X_input)

pred_class = model.predict(X_input_scaled)[0]
pred_proba = model.predict_proba(X_input_scaled)[0,1]

with col2:

    st.subheader("Estimated Scores")

    expander = st.expander("Instructions")
    expander.write("""
This section provides a summary of your individualized scores based on your input

**Estimated Addicted Score:** An indication of your potential risk level of phone addiction, where higher scores reflect increased risk.

**Estimated Mental Health Score:** Represents your projected mental health status on a scale from 0 (poor) to 10 (excellent).

**Predicted Academic Risk:** A combined risk index, ranging from 0 (no risk) to 1 (high risk), incorporating addiction, mental health, usage, and sleep patterns.

**Predicted Academic Impact:** Uses a predictive model to estimate whether your social media habits are likely to negatively affect your academic performance.
""")

    st.metric(
        label="Estimated Addicted Score (0-10):",
        value=f"{addicted_score_pred:.2f}"
    )

    st.metric(
        label="Estimated Mental Health Score (0-10):",
        value=f"{mh_score_pred:.2f}"
    )

    st.metric(
        label="Predicted Academic Risk (0-1):",
        value=f"{academic_risk_index:.2f}"
    )

    st.metric(
        label="Likely to Affect Academic Performance?",
        value="Yes" if pred_class == 1 else "No",
        #delta=f"Confidence: {pred_proba*100:.1f}%"
    )

st.subheader("Score Distribution")

expander = st.expander("Score Distribution Overview")

expander.write("""
The histograms below show how your scores compare to the rest of the sample data. Each distribution represents all users, while the **dashed line** highlights your individual score within each category.

- The wider area of each histogram displays how common different scores are in our dataset.
- The **dashed vertical line** is your personal value, helping you see where you stand (e.g., average, above, or below the typical range).

These visual summaries can help you interpret your scores and understand how your results relate to others.
""")

all_addicted_scores = df["Addicted_Score"].values
all_mh_scores = df["Mental_Health_Score"].values
all_ari_scores = df["Academic_Risk_Index"].values


addicted_hist = create_histogram(
    all_addicted_scores,
    addicted_score_pred,
    "lightcoral",
    "Addiction Score Distribution",
    "red",
    x_domain=[0, 10]
)
mh_hist = create_histogram(
    all_mh_scores,
    mh_score_pred,
    "lightgreen",
    "Mental Health Score Distribution",
    "green",
    x_domain=[0, 10]
)
ari_hist = create_histogram(
    all_ari_scores,
    academic_risk_index,
    "lightblue",
    "Academic Risk Index Distribution",
    "blue",
    x_domain=[0, 1]
)

all_charts = alt.vconcat(addicted_hist, mh_hist, ari_hist).resolve_scale(
    x="independent",
    y="shared")
st.altair_chart(all_charts,width="stretch")

st.caption("This tool is for educational purposes, not medical advice. ")

st.subheader("Generate AI Explanation")

if "explanation" not in st.session_state:
    st.session_state.explanation = ""

if st.button("Generate Explanation"):
    with st.spinner("Generating explanation..."):
        st.session_state.explanation = generate_explanation(
            sleep_hrs,
            usage_hrs,
            addicted_score_pred,
            mh_score_pred,
            academic_risk_index,
            pred_class
        )

if st.session_state.explanation:
    st.write(st.session_state.explanation)
