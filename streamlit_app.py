import streamlit as st
import joblib


model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")


st.title("Student Mental Health Dashboard")
st.write("This interactive web app hosts a model that can be used in identifying whether or not one's social media habits has an impact on their academic performance. ")



col1,col2 = st.columns(2)

with col1:
    sleep_hrs = st.slider(
        label="Input # hours of sleep you get in general",
        min_value=0,
        max_value=24,
        value=7)

    usage_hrs = st.slider(
        label="Input # hours you spend on social media in general",
        min_value=0,
        max_value=24,
        value=3)

    conflicts_count = st.slider(
        label="Input # conflicts you've had on social media",
        min_value=0,
        max_value=100,
        value=0) # Check what to set max value

    academic_level = st.selectbox(
        label="What is your academic level?", 
        options=["High School", "Undergraduate", "Postgraduate"], 
        index=None,
        placeholder="Select Academic Level")

academic_mapping = {
    "High School": 0,
    "Undergraduate": 1,
    "Postgraduate": 2
}

def calc_mh_score(usage_hrs, sleep_hrs, conflicts):
    score = (1.2 * sleep_hrs - 
    0.7 * usage_hrs - 
    0.3 * conflicts
    )

    return round(max(min(score,10), 0), 2)

def calc_addicted_score(usage_hrs, sleep_hrs, conflicts):
    score = (
        0.6 * usage_hrs + 
        0.25 * conflicts + 
        0.5 * max(0, 8 - sleep_hrs)

    )
    return round(min(score,10),2)

mental_health_score_pred = calc_mh_score(usage_hrs,sleep_hrs,conflicts_count)
addicted_score_pred = calc_addicted_score(usage_hrs,sleep_hrs,conflicts_count)

st.subheader("Estimated Scores")
st.metric("Estimated Addiction Score (0-10)", addicted_score_pred)
st.metric("Estimated Mental Health Score (0-10)", mental_health_score_pred)


st.caption("NOTE - This is not meant to serve medical advice/diagnosis")