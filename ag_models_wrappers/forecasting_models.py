import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

'''
Units:
- Air temp are Celsius
- Dew Point Celsius
- RH %
'''

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
def calculate_tarspot_risk_function(meanAT30d, maxRH30d, rh90_night_tot14d):
    '''

    Args:
        meanAT: Where MeanAT is the mean of all hourly observed temperatures for that day, 30 days moving average
        Max RH is the hourly maximum observed relative humidity for that day, 30 days moving average
        RH90.night.TOT is the daily total hours between 20:00 and 6:00 where the humidity was 90% or above, 14 days moving average.

    Returns:

    '''
    logit_LR4 = 32.06987 - (0.89471 * meanAT30d) - (0.14373 * maxRH30d)
    logit_LR6 = 20.35950 - (0.91093 * meanAT30d) - (0.29240 * rh90_night_tot14d)
    probabilities = [logistic_f(logit_LR4), logistic_f(logit_LR6)]
    ensemble_prob = np.mean(probabilities)

    risk_class = 'Inactive'
    if meanAT30d<10:
        risk_class='Inactive'
    else:
        if ensemble_prob < 0.2:
            risk_class = "1.Low"
        elif ensemble_prob > 0.35:
            risk_class = "3.High"
        elif ensemble_prob >= .2 or ensemble_prob <= .35:
            risk_class = "2.Moderate"

    return pd.Series({"tarspot_risk": ensemble_prob, "tarspot_risk_class": risk_class})


def calculate_gray_leaf_spot_risk_function(minAT21, minDP30):
    '''

    Args:
        minAT21: MinAT is the hourly minimum observed temperature for that day, 21 moving averaged
        minDP30: is the hourly minimum observed dew point for that day, 30 days moving averaged

    Returns:

    Implicit rules: Growth stage within V10 and R3 and Not irrigation total needed
    '''
    prob = logistic_f(-2.9467 - (0.03729 * minAT21) + (0.6534 * minDP30))

    risk_class = "No class"
    if prob < 0.2:
        risk_class = "1.Low"
    elif prob > 0.6:
        risk_class = "3.High"
    elif prob >= .2 or prob <= .6:
        risk_class = "2.Moderate"


    return pd.Series({"gls_risk": prob, "gls_risk_class": risk_class})


def calculate_non_irrigated_risk(maxAT30MA, maxWS30MA):
    '''

    Args:
        maxAT30MA: maximum air temperature Celsius 30 days moving average
        maxWS30MA: maximum wind speed 30 days moving average

    Returns:

    '''
    logit_nirr = (-0.47 * maxAT30MA) - (1.01 * maxWS30MA) + 16.65
    prob = logistic_f(logit_nirr)

    risk_class = 'Inactive'

    if maxAT30MA<10:
        risk_class='Inactive'
    else:
        if prob<.2:
            risk_class = '1.Low'
        elif prob>.35:
            risk_class = '3.High'
        else:
            risk_class = '2.Moderate'

    return pd.Series({"whitemold_nirr_risk": prob, "whitemold_nirr_risk_class": risk_class})


def calculate_irrigated_risk(maxAT30MA, maxRH30MA):
    '''

    Args:
        maxAT30MA: maximum air temp Celsius, 30 days moving average
        maxRH30MA: maximum relative humidity pct, 30 days moving average

    Returns:

    '''
    logit_irr_30 = (-2.38 * 1) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_30 = logistic_f(logit_irr_30)

    logit_irr_15 = (-2.38 * 0) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_15 = logistic_f(logit_irr_15)

    risk_class = 'Inactive'

    if maxAT30MA<10:
        risk_class='Inactive'
    else:

        if prob_logit_irr_15<.05:
            risk_class = '1.Low'
        elif prob_logit_irr_15>.05:
            risk_class = '3.High'
        else:
            risk_class = '2.Moderate'

    return pd.Series({
        "whitemold_irr_30in_risk": prob_logit_irr_30,
        "whitemold_irr_15in_risk": prob_logit_irr_15,
        "whitemold_irr_class": risk_class
    })


def calculate_frogeye_leaf_spot_function(maxAT30, rh80tot30):
    '''

    Args:
        maxAT30: MaxAT is the hourly maximum observed temperature for that day., 30 days moving average
        rh80tot30: is the daily total hours, where humidity was 80% or above, 30 days moving averaged

    Returns:

    Implicit rules: Growth stage within R1 and R5, No irrigation total needed
    '''
    logit_fe = -5.92485 + (0.1220 * maxAT30) + (0.1732 * rh80tot30)
    prob_logit_fe = logistic_f(logit_fe)

    risk_class = "No class"
    if prob_logit_fe < 0.5:
        risk_class = "1.Low"
    elif prob_logit_fe > 0.6:
        risk_class = "3.High"
    elif prob_logit_fe>=.5 or prob_logit_fe<=.6:
        risk_class = "2.Moderate"

    return pd.Series({"fe_risk": prob_logit_fe, "fe_risk_class": risk_class})
