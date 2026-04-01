import altair as alt
import pandas as pd
import streamlit as st

def rescale(value,min_val,max_val):
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def create_histogram(data, pred_value, color, title, marker_color, x_domain):
    hist = alt.Chart(pd.DataFrame({"value": data})).mark_bar(
        opacity=0.7,
        color=color
    ).encode(
        x=alt.X(
            'value',
            bin=alt.Bin(maxbins=10, extent=x_domain),
            scale=alt.Scale(domain=x_domain),
            title="Score"
        ),
        y=alt.Y('count()', title="Count"),
        tooltip=[alt.Tooltip('count()', title="Count")]
    )

    marker = alt.Chart(pd.DataFrame({"x": [pred_value]})).mark_rule(
        color=marker_color,
        size=2,
        strokeWidth=6,
        strokeDash=[8,4]
    ).encode(
        x="x",
        tooltip=[alt.Tooltip('x', title="Your Score")])
    return (hist + marker).properties(title=title, height=200, width=600)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)