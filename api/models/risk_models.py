"""
Agricultural disease-risk and biomass models.

Each function is a pure computation: given scalar weather inputs it returns
a ``pd.Series`` whose keys are the risk score and risk class for that disease.
This makes it trivial to unit-test models in isolation and to apply them via
``DataFrame.apply``.

Units
-----
- Air temperature  : °C
- Dew-point        : °C
- Relative humidity: %
- Wind speed       : m/s
"""

import pandas as pd

from api.utils.math_helpers import logistic, compute_logit

# ── Sentinel values ────────────────────────────────────────────────────────────
_INACTIVE_PROB: float = -1.0
_INACTIVE_CLASS: str = "Inactive"
_NO_DATA_CLASS: str = "NoData"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_missing(value) -> bool:
    """True when a value is None or a float NaN."""
    import math
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


# ── Tarspot (Damon et al.) ─────────────────────────────────────────────────────

def calculate_tarspot_risk(
    mean_at_30d: float,
    max_rh_30d: float,
    rh90_night_14d: float,
) -> pd.Series:
    """
    Ensemble tarspot risk from two logistic sub-models.

    Args:
        mean_at_30d:    30-day moving average of mean hourly air temperature (°C).
        max_rh_30d:     30-day moving average of maximum hourly RH (%).
        rh90_night_14d: 14-day moving average of nightly hours with RH ≥ 90 %.

    Returns:
        pd.Series with keys ``tarspot_risk`` (float) and ``tarspot_risk_class`` (str).
    """
    if _is_missing(mean_at_30d):
        return pd.Series({"tarspot_risk": _INACTIVE_PROB, "tarspot_risk_class": _NO_DATA_CLASS})

    if mean_at_30d < 10:
        return pd.Series({"tarspot_risk": _INACTIVE_PROB, "tarspot_risk_class": _INACTIVE_CLASS})

    logit_models = [
        (32.06987, [(0.89471, mean_at_30d), (0.14373, max_rh_30d)]),
        (20.35950, [(0.91093, mean_at_30d), (0.29240, rh90_night_14d)]),
    ]
    ensemble_prob = sum(logistic(compute_logit(c, t)) for c, t in logit_models) / 2

    if ensemble_prob < 0.20:
        risk_class = "1.Low"
    elif ensemble_prob > 0.35:
        risk_class = "3.High"
    else:
        risk_class = "2.Moderate"

    return pd.Series({"tarspot_risk": ensemble_prob, "tarspot_risk_class": risk_class})


# ── Gray Leaf Spot ─────────────────────────────────────────────────────────────

def calculate_gray_leaf_spot_risk(
    min_at_21d: float,
    min_dp_30d: float,
) -> pd.Series:
    """
    Gray leaf spot risk from a single logistic model.

    Implicit assumptions: growth stage V10–R3, no irrigation total needed.

    Args:
        min_at_21d: 21-day moving average of minimum hourly air temperature (°C).
        min_dp_30d: 30-day moving average of minimum hourly dew-point (°C).

    Returns:
        pd.Series with keys ``gls_risk`` and ``gls_risk_class``.
    """
    if _is_missing(min_at_21d):
        return pd.Series({"gls_risk": _INACTIVE_PROB, "gls_risk_class": _NO_DATA_CLASS})

    if min_at_21d < 5:
        return pd.Series({"gls_risk": _INACTIVE_PROB, "gls_risk_class": _INACTIVE_CLASS})

    prob = logistic(-2.9467 - (0.03729 * min_at_21d) + (0.6534 * min_dp_30d))

    if prob < 0.20:
        risk_class = "1.Low"
    elif prob > 0.60:
        risk_class = "3.High"
    elif 0.20 <= prob <= 0.60:
        risk_class = "2.Moderate"
    else:
        risk_class = _NO_DATA_CLASS

    return pd.Series({"gls_risk": prob, "gls_risk_class": risk_class})


# ── Frogeye Leaf Spot ──────────────────────────────────────────────────────────

def calculate_frogeye_leaf_spot_risk(
    max_at_30d: float,
    rh80_hours_30d: float,
) -> pd.Series:
    """
    Frogeye leaf spot risk from a single logistic model.

    Implicit assumptions: growth stage R1–R5, no irrigation total needed.

    Args:
        max_at_30d:     30-day moving average of maximum hourly air temperature (°C).
        rh80_hours_30d: 30-day moving average of daily hours with RH ≥ 80 %.

    Returns:
        pd.Series with keys ``fe_risk`` and ``fe_risk_class``.
    """
    if _is_missing(max_at_30d):
        return pd.Series({"fe_risk": _INACTIVE_PROB, "fe_risk_class": _NO_DATA_CLASS})

    if max_at_30d < 15:
        return pd.Series({"fe_risk": _INACTIVE_PROB, "fe_risk_class": _INACTIVE_CLASS})

    prob = logistic(-5.92485 + (0.1220 * max_at_30d) + (0.1732 * rh80_hours_30d))

    if prob < 0.50:
        risk_class = "1.Low"
    elif prob > 0.60:
        risk_class = "3.High"
    else:
        risk_class = "2.Moderate"

    return pd.Series({"fe_risk": prob, "fe_risk_class": risk_class})


# ── White Mold – Irrigated ─────────────────────────────────────────────────────

def calculate_whitemold_irrigated_risk(
    max_at_30d: float,
    max_rh_30d: float,
) -> pd.Series:
    """
    White mold risk for irrigated fields (two irrigation-level sub-models).

    Args:
        max_at_30d:  30-day moving average of maximum air temperature (°C).
        max_rh_30d:  30-day moving average of maximum RH (%).

    Returns:
        pd.Series with keys:
        ``whitemold_irr_30in_risk``, ``whitemold_irr_15in_risk``,
        ``whitemold_irr_15in_class``, ``whitemold_irr_30in_class``.
    """
    _inactive = pd.Series({
        "whitemold_irr_30in_risk": _INACTIVE_PROB,
        "whitemold_irr_15in_risk": _INACTIVE_PROB,
        "whitemold_irr_15in_class": _INACTIVE_CLASS,
        "whitemold_irr_30in_class": _INACTIVE_CLASS,
    })

    if max_at_30d < 15:
        return _inactive

    base_logit = (0.65 * max_at_30d) + (0.38 * max_rh_30d) - 52.65
    prob_30in = logistic(-2.38 + base_logit)   # irrigated 30 in/season
    prob_15in = logistic(base_logit)            # irrigated 15 in/season

    def _classify(prob: float) -> str:
        if prob < 0.05:
            return "1.Low"
        if prob > 0.10:
            return "3.High"
        return "2.Moderate"

    return pd.Series({
        "whitemold_irr_30in_risk":  prob_30in,
        "whitemold_irr_15in_risk":  prob_15in,
        "whitemold_irr_15in_class": _classify(prob_15in),
        "whitemold_irr_30in_class": _classify(prob_30in),
    })


# ── White Mold – Non-Irrigated ─────────────────────────────────────────────────

def calculate_whitemold_non_irrigated_risk(
    max_at_30d: float,
    max_rh_30d: float,
    max_ws_30d: float,
) -> pd.Series:
    """
    White mold risk for non-irrigated fields (three-model ensemble).

    Args:
        max_at_30d:  30-day moving average of maximum air temperature (°C).
        max_rh_30d:  30-day moving average of maximum RH (%).
        max_ws_30d:  30-day moving average of maximum wind speed (m/s).

    Returns:
        pd.Series with keys ``whitemold_nirr_risk`` and ``whitemold_nirr_risk_class``.
    """
    if _is_missing(max_at_30d):
        return pd.Series({"whitemold_nirr_risk": _INACTIVE_PROB, "whitemold_nirr_risk_class": _NO_DATA_CLASS})

    if max_at_30d < 15:
        return pd.Series({"whitemold_nirr_risk": _INACTIVE_PROB, "whitemold_nirr_risk_class": _INACTIVE_CLASS})

    logits = [
        -0.47 * max_at_30d - 1.01 * max_ws_30d + 16.65,
        -0.68 * max_at_30d + 17.19,
        -0.56 * max_at_30d + 0.10 * max_rh_30d - 0.75 * max_ws_30d + 8.20,
    ]
    prob = sum(logistic(l) for l in logits) / 3

    if prob < 0.20:
        risk_class = "1.Low"
    elif prob > 0.35:
        risk_class = "3.High"
    elif 0.20 <= prob <= 0.35:
        risk_class = "2.Moderate"
    else:
        risk_class = _NO_DATA_CLASS

    return pd.Series({"whitemold_nirr_risk": prob, "whitemold_nirr_risk_class": risk_class})


# ── Biomass (Cereal Rye) ───────────────────────────────────────────────────────

def calculate_cereal_rye_biomass(
    cgdd_ap: float,
    rain_ap: float,
    cgdd_bt: float,
) -> dict:
    """
    Estimate cereal-rye cover-crop biomass and classify it.

    Args:
        cgdd_ap:  Cumulative GDD (°F) for the first 60 days after planting.
        rain_ap:  Total rainfall (in) for the first 60 days after planting.
        cgdd_bt:  Cumulative GDD (°F) for the 30 days before termination.

    Returns:
        dict with keys:
        ``biomass`` (lb/acre, float),
        ``color``   ("Gray" | "Yellow" | "Green"),
        ``message`` (str).
    """
    biomass_kgha = -5725.79 + (6.19 * cgdd_ap) - (403.59 * rain_ap) + (7.02 * cgdd_bt)
    biomass_lb_acre = max(0.0, biomass_kgha * 0.892)

    if biomass_lb_acre <= 1999:
        color = "Gray"
        message = (
            "Reduction in nitrate leaching, with limited reduction "
            "in soil erosion and phosphorus loss in runoff."
        )
    elif biomass_lb_acre <= 4500:
        color = "Yellow"
        message = (
            "Significant reduction in soil erosion, phosphorus loss in runoff, "
            "and nitrate leaching, with limited weed suppression."
        )
    else:
        color = "Green"
        message = (
            "Significant weed suppression, reduction in soil erosion, "
            "reduction in phosphorus loss in runoff, and mitigation of nitrate leaching."
        )

    return {"biomass": round(biomass_lb_acre, 2), "color": color, "message": message}