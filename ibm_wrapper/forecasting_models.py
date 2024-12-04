import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def logistic_f(logit):
    return np.exp(logit) / (1 + np.exp(logit))


def calculate_tarspot_risk_function(meanAT, maxRH, rh90_night_tot):
    logit_LR4 = 32.06987 - (0.89471 * meanAT) - (0.14373 * maxRH)
    logit_LR6 = 20.35950 - (0.91093 * meanAT) - (0.29240 * rh90_night_tot)
    probabilities = [logistic_f(logit_LR4), logistic_f(logit_LR6)]
    ensemble_prob = np.mean(probabilities)

    if ensemble_prob < 0.2:
        risk_class = "low"
    elif ensemble_prob > 0.35:
        risk_class = "high"
    else:
        risk_class = "moderate"

    return pd.Series({"tarspot_risk": ensemble_prob, "tarspot_risk_class": risk_class})


def calculate_gray_leaf_spot_risk_function(minAT21, minDP30):
    prob = logistic_f(-2.9467 - (0.03729 * minAT21) + (0.6534 * minDP30))

    if prob < 0.2:
        risk_class = "low"
    elif prob > 0.6:
        risk_class = "high"
    else:
        risk_class = "moderate"

    return pd.Series({"gls_risk": prob, "gls_risk_class": risk_class})


def calculate_non_irrigated_risk(maxAT30MA, maxWS30MA):
    logit_nirr = (-0.47 * maxAT30MA) - (1.01 * maxWS30MA) + 16.65
    ensemble_prob = logistic_f(logit_nirr)

    return pd.Series({"sporec_nirr_risk": ensemble_prob, "sporec_nirr_risk_class": "NoClass"})


def calculate_irrigated_risk(maxAT30MA, maxRH30MA):
    logit_irr_30 = (-2.38 * 1) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_30 = logistic_f(logit_irr_30)

    logit_irr_15 = (-2.38 * 0) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_15 = logistic_f(logit_irr_15)

    return pd.Series({
        "sporec_irr_30in_risk": prob_logit_irr_30,
        "sporec_irr_15in_risk": prob_logit_irr_15
    })


def calculate_frogeye_leaf_spot_function(maxAT30, rh80tot30):
    logit_fe = -5.92485 - (0.1220 * maxAT30) + (0.1732 * rh80tot30)
    prob_logit_fe = logistic_f(logit_fe)

    if prob_logit_fe < 0.5:
        risk_class = "low"
    elif prob_logit_fe > 0.6:
        risk_class = "high"
    else:
        risk_class = "moderate"

    return pd.Series({"fe_risk": prob_logit_fe, "fe_risk_class": risk_class})
