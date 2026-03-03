"""
Model registry — single source of truth for all pest model metadata.

The /models router, the prediction formatters, and the dependency
validators all read from here. Adding a new model means one entry here
plus one apply-block in api/services/risk_processor.py.
"""

MODELS: dict[str, dict] = {
    "tarspot": {
        "id":           "tarspot",
        "name":         "Tar Spot",
        "description":  "Ensemble logistic model for Phyllachora maydis (Damon et al.).",
        "crop":         "corn",
        "risk_classes": ["inactive", "low", "moderate", "high"],
        "units":        {"risk": "probability 0–1"},
        "inputs":       [
            "mean_air_temp_30d_avg_c",
            "max_rh_30d_avg_pct",
            "rh90_night_hours_14d_avg",
        ],
    },
    "gls": {
        "id":           "gls",
        "name":         "Gray Leaf Spot",
        "description":  "Logistic model for Cercospora zeae-maydis.",
        "crop":         "corn",
        "risk_classes": ["inactive", "low", "moderate", "high"],
        "units":        {"risk": "probability 0–1"},
        "inputs":       [
            "min_air_temp_21d_avg_c",
            "min_dew_point_30d_avg_c",
        ],
    },
    "frogeye": {
        "id":           "frogeye",
        "name":         "Frogeye Leaf Spot",
        "description":  "Logistic model for Cercospora sojina.",
        "crop":         "soybean",
        "risk_classes": ["inactive", "low", "moderate", "high"],
        "units":        {"risk": "probability 0–1"},
        "inputs":       [
            "max_air_temp_30d_avg_c",
            "rh80_hours_30d_avg",
        ],
    },
    "whitemold": {
        "id":           "whitemold",
        "name":         "White Mold",
        "description":  "Ensemble logistic model for Sclerotinia sclerotiorum — irrigated and non-irrigated variants.",
        "crop":         "soybean",
        "risk_classes": ["inactive", "low", "moderate", "high"],
        "units":        {"risk": "probability 0–1"},
        "inputs":       [
            "max_air_temp_30d_avg_c",
            "max_rh_30d_avg_pct",
            "max_wind_speed_30d_avg_mps",
        ],
        "variants":     ["non_irrigated", "irrigated_15in", "irrigated_30in"],
    },
}

ALL_MODEL_IDS: list[str] = list(MODELS.keys())


def get_model(model_id: str) -> dict | None:
    return MODELS.get(model_id)


def validate_models(requested: list[str]) -> list[str]:
    """Raise ValueError for any unknown model IDs; return the list unchanged on success."""
    unknown = [m for m in requested if m not in MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Valid options: {', '.join(ALL_MODEL_IDS)}."
        )
    return requested
