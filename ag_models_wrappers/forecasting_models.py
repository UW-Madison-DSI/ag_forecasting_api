import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

@staticmethod
def fahrenheit_to_celsius(x):
    '''

    Args:
        x:

    Returns:

    '''
    return (x - 32) * 5/9

@staticmethod
def rolling_mean(series, window):
    '''

    Args:
        series:
        window:

    Returns:

    '''
    return series.rolling(window=window, min_periods=window).mean()

@staticmethod
def logistic_f(logit):
    '''

    Args:
        logit:

    Returns:

    '''
    return np.exp(logit) / (1 + np.exp(logit))


###################### Damon et. al.
def calculate_tarspot_risk_function(meanAT, maxRH, rh90_night_tot):
    '''

    Args:
        meanAT:
        maxRH:
        rh90_night_tot:

    Returns:

    '''
    logit_LR4 = 32.06987 - (0.89471 * meanAT) - (0.14373 * maxRH)
    logit_LR6 = 20.35950 - (0.91093 * meanAT) - (0.29240 * rh90_night_tot)
    probabilities = [logistic_f(logit_LR4), logistic_f(logit_LR6)]
    ensemble_prob = np.mean(probabilities)

    if ensemble_prob < 0.2:
        risk_class = "Low"
    elif ensemble_prob > 0.35:
        risk_class = "High"
    elif ensemble_prob >= .2 or ensemble_prob <= .35:
        risk_class = "Moderate"
    else:
        risk_class= "No class"

    return pd.Series({"tarspot_risk": ensemble_prob, "tarspot_risk_class": risk_class})


def calculate_gray_leaf_spot_risk_function(minAT21, minDP30):
    '''

    Args:
        minAT21:
        minDP30:

    Returns:

    '''
    prob = logistic_f(-2.9467 - (0.03729 * minAT21) + (0.6534 * minDP30))

    if prob < 0.2:
        risk_class = "Low"
    elif prob > 0.6:
        risk_class = "High"
    elif prob >= .2 or prob <= .6:
        risk_class = "Moderate"
    else:
        risk_class= "No class"

    return pd.Series({"gls_risk": prob, "gls_risk_class": risk_class})


def calculate_non_irrigated_risk(maxAT30MA, maxWS30MA):
    '''

    Args:
        maxAT30MA:
        maxWS30MA:

    Returns:

    '''
    logit_nirr = (-0.47 * maxAT30MA) - (1.01 * maxWS30MA) + 16.65
    prob = logistic_f(logit_nirr)

    return pd.Series({"whitemold_nirr_risk": prob, "whitemold_nirr_risk_class": "NoClass"})


def calculate_irrigated_risk(maxAT30MA, maxRH30MA):
    '''

    Args:
        maxAT30MA:
        maxRH30MA:

    Returns:

    '''
    logit_irr_30 = (-2.38 * 1) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_30 = logistic_f(logit_irr_30)

    logit_irr_15 = (-2.38 * 0) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_15 = logistic_f(logit_irr_15)

    return pd.Series({
        "whitemold_irr_30in_risk": prob_logit_irr_30,
        "whitemold_irr_15in_risk": prob_logit_irr_15
    })


def calculate_frogeye_leaf_spot_function(maxAT30, rh80tot30):
    '''

    Args:
        maxAT30:
        rh80tot30:

    Returns:

    '''
    logit_fe = -5.92485 - (0.1220 * maxAT30) + (0.1732 * rh80tot30)
    prob_logit_fe = logistic_f(logit_fe)

    if prob_logit_fe < 0.5:
        risk_class = "Low"
    elif prob_logit_fe > 0.6:
        risk_class = "High"
    elif prob_logit_fe>=.5 or prob_logit_fe<=.6:
        risk_class = "Moderate"
    else:
        risk_class = "No class"

    return pd.Series({"fe_risk": prob_logit_fe, "fe_risk_class": risk_class})
