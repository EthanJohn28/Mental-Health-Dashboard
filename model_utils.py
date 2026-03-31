import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("./data/Students_Social_Media_Addiction_FE.csv")

bias_params = joblib.load("bias_params_production.pkl")
model = joblib.load("logistic_model_production.pkl")
scaler = joblib.load("scaler_production.pkl")

MH_bias = bias_params["MH_Bias_Factor"]
A_bias = bias_params["AS_Bias_Factor"]

min_addicted_score = 0
max_addicted_score = 10
min_mh_score = 0
max_mh_score = 10

def calc_mh_score(usage_hrs, sleep_hrs):
    raw = (
        0.75 * sleep_hrs - 
        0.25 * usage_hrs
    )

    return np.clip(raw + MH_bias, min_mh_score, max_mh_score)

def calc_addicted_score(usage_hrs, sleep_hrs):
    raw = (
        0.75 * usage_hrs +
        0.25 * max(0, 8-sleep_hrs)
    )

    return np.clip(raw + A_bias, min_addicted_score, max_addicted_score)