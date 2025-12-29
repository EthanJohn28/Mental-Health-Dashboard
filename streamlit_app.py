import streamlit as st

st.title("Student Mental Health Dashboard")
st.write("This interactive web app hosts a model that can be used in identifying whether or not one's social media habits has an impact on their academic performance. ")

sleep_hrs = st.slider(label="Input # hours of sleep you get in general",min_value=0,max_value=24)
usage_hrs = st.slider(label="Input # hours you spend on social media in general",min_value=0,max_value=24)
conflicts_count = st.slider(label="Input # conflicts you've had on social media",min_value=0,max_value=100) # Check what to set max value
academic_level = st.selectbox(label="What is your academic level?", options=["High School", "Undergraduate", "Postgraduate"], index=None, placeholder="Select Academic Level")



