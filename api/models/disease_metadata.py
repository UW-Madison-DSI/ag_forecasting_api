from typing import Dict

DISEASE_MODELS: Dict[str, dict] = {
    "tarspot": {
        "name": "Tarspot",
        "description": (
            "Tarspot risk model based on 30-day moving averages of temperature, "
            "relative humidity, and nighttime humidity exposure."
        ),
        "variables": [
            "air_temp_avg_c_30d_ma",
            "rh_max_30d_ma",
            "rh_above_90_night_14d_ma"
        ],
        "model_type": "Logistic regression",
        "risk_output": "Probability scaled to 0–100",
        "inactive_rule": "Inactive when 30-day average temperature < threshold",
        "version": "1.0"
    },
    "frogeye_leaf_spot": {
        "name": "Frogeye Leaf Spot",
        "description": (
            "Risk model using 30-day max temperature and daytime humidity exposure."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_above_80_day_30d_ma"
        ],
        "model_type": "Logistic regression",
        "risk_output": "Probability scaled to 0–100",
        "version": "1.0"
    },
    "whitemold_irrigated": {
        "name": "White Mold (Irrigated)",
        "description": (
            "White mold risk under irrigated conditions using 30-day max temperature "
            "and relative humidity."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_max_30d_ma"
        ],
        "model_type": "Logistic regression",
        "version": "1.0"
    },
    "whitemold_non_irrigated": {
        "name": "White Mold (Non-Irrigated)",
        "description": (
            "White mold risk under rainfed conditions using temperature, humidity, "
            "and wind speed."
        ),
        "variables": [
            "air_temp_max_c_30d_ma",
            "rh_max_30d_ma",
            "max_ws_30d_ma"
        ],
        "model_type": "Logistic regression",
        "version": "1.0"
    }
}