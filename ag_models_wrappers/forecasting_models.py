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

def compute_logit(const, terms):
    """
    Compute a logit value given a constant and a list of (coefficient, variable) tuples.
    """
    return const - sum(coef * var for coef, var in terms)

###################### Damon et. al.
def calculate_tarspot_risk_function(meanAT30d, maxRH30d, rh90_night_tot14d):
    '''

    Args:
        meanAT: Where MeanAT is the mean of all hourly observed temperatures for that day, 30 days moving average
        Max RH is the hourly maximum observed relative humidity for that day, 30 days moving average
        RH90.night.TOT is the daily total hours between 20:00 and 6:00 where the humidity was 90% or above, 14 days moving average.

    Returns:

    '''
    if meanAT30d is None or meanAT30d==np.nan:
        #in the moving average computation I wont have maxAT30MA if not enough number of days, so then I will report NoData
        risk_class = 'NoData'
        ensemble_prob = -1
    else:
        risk_class = 'Inactive'
        if meanAT30d<10:
            risk_class='Inactive'
            ensemble_prob = -1
        else:
            logit_models = [
                (32.06987, [(0.89471, meanAT30d), (0.14373, maxRH30d)]),
                (20.35950, [(0.91093, meanAT30d), (0.29240, rh90_night_tot14d)])
            ]

            # Compute the logits for each model
            logits = [compute_logit(const, terms) for const, terms in logit_models]

            # Compute the probabilities using your logistic function
            probabilities = [logistic_f(logit) for logit in logits]

            # Calculate the ensemble probability by averaging the individual probabilities
            ensemble_prob = sum(probabilities) / 2

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
    if minAT21 is None or minAT21==np.nan:
        #in the moving average computation I wont have minAT21MA if not enough number of days, so then I will report NoData
        risk_class = 'NoData'
        prob = -1
    else:
        if minAT21 < 5:
            risk_class = 'Inactive'
            prob = -1
        else:
            prob = logistic_f(-2.9467 - (0.03729 * minAT21) + (0.6534 * minDP30))
            risk_class = 'Inactive'

            if prob < 0.2:
                risk_class = "1.Low"
            elif prob > 0.6:
                risk_class = "3.High"
            elif prob >= .2 or prob <= .6:
                risk_class = "2.Moderate"
            else:
                risk_class = 'NoData'

    return pd.Series({"gls_risk": prob, "gls_risk_class": risk_class})


def calculate_non_irrigated_risk(maxAT30MA, maxRH30MA, maxWS30MA):
    '''

    Args:
        maxAT30MA: maximum air temperature Celsius 30 days moving average
        maxWS30MA: maximum wind speed 30 days moving average

    Returns:

    '''
    if maxAT30MA is None or maxAT30MA==np.nan:
        #in the moving average computation I wont have maxAT30MA if not enough number of days, so then I will report NoData
        risk_class = 'NoData'
        prob = -1
    else:
        if maxAT30MA<15:
            risk_class='Inactive'
            prob = -1
        else:
            logits = [
                -0.47 * maxAT30MA - 1.01 * maxWS30MA + 16.65,
                -0.68 * maxAT30MA + 17.19,
                -0.56 * maxAT30MA + 0.10 * maxRH30MA - 0.75 * maxWS30MA + 8.20
            ]

            # Compute probabilities using the logistic function
            probs = [logistic_f(logit) for logit in logits]

            # Compute the average probability
            prob = sum(probs) / 3

            if prob<.2:
                risk_class = '1.Low'
            elif prob>.35:
                risk_class = '3.High'
            elif prob<=.35 and prob>=.2:
                risk_class = '2.Moderate'
            else:
                risk_class = 'NoData'

    return pd.Series({"whitemold_nirr_risk": prob, "whitemold_nirr_risk_class": risk_class})



def calculate_irrigated_risk(maxAT30MA, maxRH30MA):
    """
    Calculate risk probabilities and classes for whitemold exposure.

    Parameters:
    maxAT30MA (float): Maximum temperature measure over 30 minutes.
    maxRH30MA (float): Maximum relative humidity measure over 30 minutes.

    Returns:
    pd.Series: A pandas Series containing:
    - whitemold_irr_30in_risk: Probability risk based on the 30 multiplier.
    - whitemold_irr_15in_risk: Probability risk based on the 15 multiplier.
    - whitemold_irr_15in_class: Risk classification for the 15 model.
    - whitemold_irr_30in_class: Risk classification for the 30 model.
    """
    # Calculate the logit for the 30 multiplier case
    logit_irr_30 = (-2.38 * 1) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_30 = logistic_f(logit_irr_30)

    # Calculate the logit for the 15 multiplier case
    logit_irr_15 = (-2.38 * 0) + (0.65 * maxAT30MA) + (0.38 * maxRH30MA) - 52.65
    prob_logit_irr_15 = logistic_f(logit_irr_15)

    # Initialize risk classes
    risk_class_15 = 'Inactive'
    risk_class_30 = 'Inactive'

    if maxAT30MA < 15:
        risk_class = 'Inactive'
        prob_logit_irr_30 = -1
        prob_logit_irr_15 = -1
    else:
        # Set risk classifications based on thresholds
        # Classification for 15in model
        if prob_logit_irr_15 < 0.05:
            risk_class_15 = '1.Low'
        elif prob_logit_irr_15 > 0.1:
            risk_class_15 = '3.High'
        elif prob_logit_irr_15 <= 0.1 and prob_logit_irr_15 >= 0.05:
            risk_class_15 = '2.Moderate'
        else:
            risk_class_15 = 'NoData'

        # Classification for 30in model
        if prob_logit_irr_30 < 0.05:
            risk_class_30 = '1.Low'
        elif prob_logit_irr_30 > 0.1:
            risk_class_30 = '3.High'
        elif prob_logit_irr_30 <= 0.1 and prob_logit_irr_30 >= 0.05:
            risk_class_30 = '2.Moderate'
        else:
            risk_class_30 = 'NoData'

    # Return the computed values as a pandas Series
    return pd.Series({
        "whitemold_irr_30in_risk": prob_logit_irr_30,
        "whitemold_irr_15in_risk": prob_logit_irr_15,
        "whitemold_irr_15in_class": risk_class_15,
        "whitemold_irr_30in_class": risk_class_30,
    })

def calculate_frogeye_leaf_spot_function(maxAT30, rh80tot30):
    '''

    Args:
        maxAT30: MaxAT is the hourly maximum observed temperature for that day., 30 days moving average
        rh80tot30: is the daily total hours, where humidity was 80% or above, 30 days moving averaged

    Returns:

    Implicit rules: Growth stage within R1 and R5, No irrigation total needed
    '''


    if maxAT30 is None or maxAT30==np.nan:
        risk_class = 'NoData'
        prob_logit_fe = -1
    else:
        logit_fe = -5.92485 + (0.1220 * maxAT30) + (0.1732 * rh80tot30)
        prob_logit_fe = logistic_f(logit_fe)
        risk_class = '2.Moderate'
        if maxAT30 < 15:
            risk_class = 'Inactive'
            prob_logit_fe = -1
        else:
            if prob_logit_fe < 0.5:
                risk_class = "1.Low"
            elif prob_logit_fe > 0.6:
                risk_class = "3.High"
            elif prob_logit_fe>=.5 or prob_logit_fe<=.6:
                risk_class = "2.Moderate"
            else:
                risk_class = 'NoData'

    return pd.Series({"fe_risk": prob_logit_fe, "fe_risk_class": risk_class})


def cereal_rye_report(cgdd_ap, rain_ap, cgdd_bt):
    # 1. Calculate base biomass (assuming original coefficients are kg/ha)
    # If your coefficients already result in lb/acre, remove the 0.892 conversion.
    biomass_kgha = -5725.79 + (6.19 * cgdd_ap) - (403.59 * rain_ap) + (7.02 * cgdd_bt)

    # Convert kg/ha to lb/acre (1 kg/ha ≈ 0.892179 lb/acre)
    biomass_lb_acre = max(0, biomass_kgha * 0.892)

    # 2. Determine Color and Text based on thresholds
    if biomass_lb_acre <= 1999:
        color = "Gray"
        status_text = ("Reduction in nitrate leaching, with limited reduction "
                       "in soil erosion and phosphorus loss in runoff.")

    elif 2000 <= biomass_lb_acre <= 4500:
        color = "Yellow"
        status_text = ("Significant reduction in soil erosion, phosphorus loss in runoff, "
                       "and nitrate leaching, with limited weed suppression.")

    else:  # Biomass is > 4500
        color = "Green"
        status_text = ("Significant weed suppression, reduction in soil erosion, "
                       "reduction in phosphorus loss in runoff, and mitigation of nitrate leaching.")

    # 3. Return a structured report
    return {
        "biomass": round(biomass_lb_acre, 2),
        "color": color,
        "message": status_text
    }