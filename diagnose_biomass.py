"""
Comprehensive biomass diagnostic script.
Run this from the repository root to identify where the winter rye biomass pipeline breaks.
"""

import json
from datetime import datetime
from pprint import pprint

import pandas as pd

print("""
============================================================
WINTER RYE BIOMASS DIAGNOSTIC
============================================================
""")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST MODEL MATH
# ─────────────────────────────────────────────────────────────────────────────
print("[1/4] Testing model math...")
print("------------------------------------------------------------")
try:
    from api.models.risk_models import calculate_winter_rye_biomass

    result = calculate_winter_rye_biomass(
        plant_doy=260,
        precip_fall=100.0,
        gdd_total=1500.0,
    )

    print("✅ Model callable")
    print(f"   Output: {result}")
    if result.get("biomass_lb_acre") is not None:
        print("✅ Biomass value present")
    else:
        print("❌ Biomass value missing")
except Exception as exc:
    print("❌ Model math failed:")
    print(exc)

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST GDD HELPER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Testing GDD helper...")
print("------------------------------------------------------------")
try:
    from api.utils.math_helpers import gdd_sine

    gdd = gdd_sine(5.0, 15.0)
    print("✅ GDD helper callable")
    print(f"   gdd_sine(5, 15) = {gdd:.3f}")
    if gdd > 0:
        print("✅ GDD output looks valid")
    else:
        print("⚠️ GDD output is zero or negative")
except Exception as exc:
    print("❌ GDD helper failed:")
    print(exc)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEST INTERNAL ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Testing biomass annotation function...\n")
print("------------------------------------------------------------")
try:
    from api.services.ibm_service import _annotate_winter_rye_biomass

    dates = pd.date_range("2024-09-15", "2025-05-15", freq="D")
    sample_daily = pd.DataFrame(
        {
            "date": dates,
            "temperature_min": 5.0 + (dates.dayofyear % 50) * 0.2,
            "temperature_max": 15.0 + (dates.dayofyear % 50) * 0.2,
            "precip1Hour_sum": [2.0 if d.dayofyear % 5 == 0 else 0.0 for d in dates],
        }
    )
    sample_daily["date"] = pd.to_datetime(sample_daily["date"]).dt.date

    print(f"✅ Created sample daily DataFrame with {len(sample_daily)} rows")
    result_df = _annotate_winter_rye_biomass(sample_daily.copy(), "2024-09-15")

    print("✅ Annotation function executed")
    biomass_cols = [c for c in result_df.columns if "biomass" in c]
    print(f"   biomass columns: {biomass_cols}")
    for col in biomass_cols:
        null_count = int(result_df[col].isna().sum())
        print(f"   {col}: {len(result_df) - null_count} non-null, {null_count} null")
    if biomass_cols:
        print("   Sample last values:")
        print(result_df.loc[result_df.index[-3:], biomass_cols].to_dict(orient="records"))
except Exception as exc:
    print("❌ Biomass annotation failed:")
    print(exc)

# ─────────────────────────────────────────────────────────────────────────────
# 4. TEST API ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Testing local API endpoint...\n")
print("------------------------------------------------------------")
try:
    import requests

    url = "http://127.0.0.1:8000/v2/test/rye_biomass"
    print(f"Calling {url}")
    response = requests.get(url, timeout=10)
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        body = response.json()
        print(json.dumps(body, indent=2))
    else:
        print("Response text:")
        print(response.text)
except Exception as exc:
    print("❌ API endpoint call failed:")
    print(exc)

print("\n============================================================")
print("DIAGNOSTIC COMPLETE")
print("============================================================")
