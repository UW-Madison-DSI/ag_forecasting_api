# Winter Rye Biomass Model - Debugging Guide

## Quick Diagnostic Checklist

Use this checklist to systematically verify each layer of the biomass pipeline:

```
[ ] 1. Model Math - Verify calculation logic works in isolation
[ ] 2. Data Input - Confirm weather data columns exist and have correct values
[ ] 3. Parameter Flow - Trace planting_date through API → service → model
[ ] 4. Integration - Test full IBM/WiscoNet pipeline end-to-end
[ ] 5. Output - Verify results appear in API response with expected values
```

---

## Level 1: Test Model Math in Isolation

### Option A: Use the Built-in Test Endpoint

```bash
# Start the API (if not already running)
uvicorn app:app --reload

# In another terminal, call the test endpoint
curl "http://localhost:8000/v2/test/rye_biomass"
```

**Expected Output:**
```json
{
  "test_inputs": {
    "plant_doy": 260,
    "plant_doy_interpretation": "Mid-September (day 260 of year)",
    "precip_fall": 100.0,
    "gdd_total": 1500.0
  },
  "result": {
    "biomass_lb_acre": 3289.14,
    "biomass_color": "Yellow",
    "biomass_message": "Moderate rye coverage (2000-4500 lb/acre)"
  }
}
```

**If this works:** Model math is correct ✅ → Move to Level 2

**If this fails:** Check `api/models/risk_models.py` for `calculate_winter_rye_biomass()` function

---

### Option B: Test Model Directly in Python

```python
# Create a test script: test_biomass_model.py

from api.models.risk_models import calculate_winter_rye_biomass

# Test case: Mid-September planting, moderate GDD accumulation
result = calculate_winter_rye_biomass(
    plant_doy=260,      # Day 260 = Sept 17
    precip_fall=100.0,  # 100mm rain during fall establishment
    gdd_total=1500.0    # 1500 GDD accumulated
)

print("Test Result:")
print(f"  Biomass: {result['biomass_lb_acre']:.2f} lb/acre")
print(f"  Color: {result['biomass_color']}")
print(f"  Message: {result['biomass_message']}")

# Expected: biomass ~3289 lb/acre, Yellow color
```

Run it:
```bash
cd /Users/mariaoros/Documents/DataScienceInstitute/2026/ag_forecasting_api
python test_biomass_model.py
```

---

## Level 2: Check Weather Data Input

### Verify IBM API Returns Required Columns

```bash
# Query IBM endpoint and inspect raw weather data
curl -s "http://localhost:8000/v2/ag_models_wrappers/ibm?forecasting_date=2025-05-15&latitude=43.0&longitude=-89.0&planting_date=2024-09-15&API_KEY=xxx&TENANT_ID=yyy&ORG_ID=zzz" | python -m json.tool | head -100
```

**Look for these columns in properties:**
- ✓ `temperature_min` (or `temperature_min_c`)
- ✓ `temperature_max` (or `temperature_max_c`)
- ✓ `precip1Hour_sum` (cumulative hourly precipitation)
- ✓ `date` (in YYYY-MM-DD format)

**If missing:** IBM API may not be returning expected columns. Check:
1. IBM API credentials are valid
2. API response schema hasn't changed
3. `_build_daily()` in `ibm_service.py` is creating these columns correctly

---

### Debug Data Manually (Python Script)

Create `debug_biomass_data.py`:

```python
import pandas as pd
from datetime import datetime, date
from api.models.risk_models import calculate_winter_rye_biomass
from api.utils.math_helpers import gdd_sine

# Simulate weather data (like IBM API would return)
dates = pd.date_range('2024-09-15', '2025-05-15', freq='D')

data = {
    'date': [d.strftime('%Y-%m-%d') for d in dates],
    'temperature_min': 10 + (d.dayofyear % 100) * 0.05,  # Simulated temp curve
    'temperature_max': 20 + (d.dayofyear % 100) * 0.05,
    'precip1Hour_sum': 2.5 if d.dayofyear % 5 == 0 else 0,
}

df = pd.DataFrame(data)

# Convert types
df['date'] = pd.to_datetime(df['date'])
df['temperature_min'] = pd.to_numeric(df['temperature_min'])
df['temperature_max'] = pd.to_numeric(df['temperature_max'])
df['precip1Hour_sum'] = pd.to_numeric(df['precip1Hour_sum'])

print("Sample weather data:")
print(df.head(10))
print(f"\nTotal rows: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Now manually compute biomass step by step
planting_date = '2024-09-15'
plant_date_obj = datetime.fromisoformat(planting_date).date()
plant_doy = plant_date_obj.timetuple().tm_yday

print(f"\nPlanting date: {planting_date}")
print(f"Plant day-of-year: {plant_doy}")

# Filter to dates >= planting_date
rye_df = df[df['date'] >= planting_date].copy()
print(f"Rows after planting: {len(rye_df)}")

# Calculate GDD
rye_df['gdd_daily'] = rye_df.apply(
    lambda row: gdd_sine(row['temperature_min'], row['temperature_max']),
    axis=1
)

rye_df['gdd_cumsum'] = rye_df['gdd_daily'].cumsum()

print("\nGDD calculation:")
print(rye_df[['date', 'temperature_min', 'temperature_max', 'gdd_daily', 'gdd_cumsum']].head(10))
print("...")
print(rye_df[['date', 'temperature_min', 'temperature_max', 'gdd_daily', 'gdd_cumsum']].tail(10))

# Calculate fall precipitation (Sept 1 - planting date)
fall_start = plant_date_obj.replace(month=9, day=1)
fall_precip = rye_df[
    (rye_df['date'] >= str(fall_start)) & 
    (rye_df['date'] < planting_date)
]['precip1Hour_sum'].sum()

print(f"\nFall precipitation (Sept 1 - {planting_date}): {fall_precip:.2f} mm")

# Compute biomass for last row
last_row = rye_df.iloc[-1]
biomass = calculate_winter_rye_biomass(
    plant_doy=plant_doy,
    precip_fall=fall_precip,
    gdd_total=last_row['gdd_cumsum']
)

print(f"\nBiomass result (as of {last_row['date'].strftime('%Y-%m-%d')}):")
print(f"  Input: plant_doy={plant_doy}, precip_fall={fall_precip:.2f}, gdd_total={last_row['gdd_cumsum']:.2f}")
print(f"  Output: {biomass['biomass_lb_acre']:.2f} lb/acre ({biomass['biomass_color']})")
```

Run it:
```bash
python debug_biomass_data.py
```

---

## Level 3: Trace Parameter Flow

### Check if `planting_date` Reaches the Model

#### Step 3a: Verify Route Accepts Parameter

Edit `app.py` and add a debug print:

```python
@app.get("/v2/ag_models_wrappers/ibm")
def ibm_weather_query(
    forecasting_date: str,
    latitude: float = Query(...),
    longitude: float = Query(...),
    planting_date: str | None = Query(None),  # <-- Should capture this
    termination_date: str | None = Query(None),
    API_KEY: str = Query(...),
    TENANT_ID: str = Query(...),
    ORG_ID: str = Query(...)
):
    print(f"DEBUG: Route received planting_date={planting_date}")  # ADD THIS LINE
    
    result = get_weather_with_risk(
        forecasting_date=forecasting_date,
        latitude=latitude,
        longitude=longitude,
        planting_date=planting_date,  # Make sure it's passed
        # ...rest of params
    )
    return result
```

**Test it:**
```bash
curl "http://localhost:8000/v2/ag_models_wrappers/ibm?forecasting_date=2025-05-15&latitude=43.0&longitude=-89.0&planting_date=2024-09-15&API_KEY=xxx&TENANT_ID=yyy&ORG_ID=zzz"
```

Check console logs for: `DEBUG: Route received planting_date=2024-09-15`

---

#### Step 3b: Verify Service Function Receives It

In `api/services/ibm_service.py`, check the `get_weather_with_risk()` signature:

```python
def get_weather_with_risk(
    lat: float,
    lng: float,
    forecasting_date: str,
    planting_date: str | None = None,  # <-- Should have this parameter
    API_KEY: str,
    TENANT_ID: str,
    ORG_ID: str
) -> dict:
```

And make sure it's being passed to `_annotate_winter_rye_biomass()`:

```python
def _annotate_risks(daily_df: pd.DataFrame, planting_date: str | None = None) -> pd.DataFrame:
    """Apply all risk models + biomass calculation."""
    
    # ... disease risk calculations ...
    
    # Biomass annotation (if planting_date provided)
    if planting_date:
        print(f"DEBUG: _annotate_risks calling biomass with planting_date={planting_date}")  # ADD
        daily_df = _annotate_winter_rye_biomass(daily_df, planting_date)
    
    return daily_df
```

---

#### Step 3c: Check if Biomass Function is Called

In `api/services/ibm_service.py`, inside `_annotate_winter_rye_biomass()`, look for logging:

```python
logger.info(f"Starting biomass annotation with planting_date={planting_date}")
logger.info(f"Plant DOY: {plant_doy}")
logger.info(f"Filtered rows: {len(rye_daily)}")
logger.info(f"Fall precipitation: {fall_precip:.2f} mm")
logger.info(f"Sample GDD values: {rye_daily['gdd_total'].tail(3).values}")
logger.info(f"Sample biomass predictions: {rye_daily['biomass_lb_acre'].tail(3).values}")
```

**Run with logging:**
```bash
# Enable debug logging
PYTHONPATH=/Users/mariaoros/Documents/DataScienceInstitute/2026/ag_forecasting_api \
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from api.services.ibm_service import get_weather_with_risk
result = get_weather_with_risk(
    lat=43.0, lng=-89.0,
    forecasting_date='2025-05-15',
    planting_date='2024-09-15',
    API_KEY='xxx', TENANT_ID='yyy', ORG_ID='zzz'
)
" 2>&1 | grep -i biomass
```

---

## Level 4: Test Full Pipeline End-to-End

### Test with WiscoNet (No IBM Credentials Needed)

```bash
# This is easier because WiscoNet is public API
curl "http://localhost:8000/v2/ag_models_wrappers/wisconet_g?forecasting_date=2025-05-15&planting_date=2024-09-15&station_id=ALTN"
```

**Inspect the response for biomass columns:**

```python
import json
import subprocess

result = subprocess.run([
    'curl', '-s',
    'http://localhost:8000/v2/ag_models_wrappers/wisconet_g?forecasting_date=2025-05-15&planting_date=2024-09-15&station_id=ALTN'
], capture_output=True, text=True)

data = json.loads(result.stdout)

# Check first feature's properties
if data['features']:
    props = data['features'][0]['properties']
    
    # Look for biomass fields
    biomass_fields = {k: v for k, v in props.items() if 'biomass' in k.lower()}
    
    print("Biomass fields in response:")
    for key, value in biomass_fields.items():
        print(f"  {key}: {value}")
    
    if not biomass_fields:
        print("  ❌ NO BIOMASS FIELDS FOUND")
    else:
        print("  ✅ Biomass fields present")
```

---

## Level 5: Deep Inspection Using Python Script

Create `diagnose_biomass.py`:

```python
"""
Comprehensive biomass diagnostic script.
Run this to identify where the pipeline breaks.
"""

import json
import pandas as pd
from datetime import datetime
from pprint import pprint

print("=" * 70)
print("WINTER RYE BIOMASS DIAGNOSTIC")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# 1. TEST MODEL MATH
# ─────────────────────────────────────────────────────────────────────────

print("\n[1/4] Testing Model Math...")
print("-" * 70)

try:
    from api.models.risk_models import calculate_winter_rye_biomass
    
    result = calculate_winter_rye_biomass(
        plant_doy=260,
        precip_fall=100.0,
        gdd_total=1500.0
    )
    
    print(f"✅ Model callable")
    print(f"   Input: plant_doy=260, precip_fall=100.0, gdd_total=1500.0")
    print(f"   Output: {result}")
    
    if result.get('biomass_lb_acre') and result['biomass_lb_acre'] > 0:
        print(f"✅ Model produces numeric output")
    else:
        print(f"❌ Model output invalid: {result.get('biomass_lb_acre')}")
        
except Exception as e:
    print(f"❌ Model math failed: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────
# 2. TEST GDD CALCULATION
# ─────────────────────────────────────────────────────────────────────────

print("\n[2/4] Testing GDD Helper...")
print("-" * 70)

try:
    from api.utils.math_helpers import gdd_sine
    
    # Typical spring day: min 5°C, max 15°C
    gdd = gdd_sine(5.0, 15.0)
    print(f"✅ GDD helper callable")
    print(f"   Input: tmin=5°C, tmax=15°C")
    print(f"   Output: {gdd:.2f} GDD")
    
    if gdd > 0:
        print(f"✅ GDD calculation working")
    else:
        print(f"❌ GDD returned {gdd}, expected > 0")
        
except Exception as e:
    print(f"❌ GDD helper failed: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────
# 3. TEST ANNOTATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────

print("\n[3/4] Testing Biomass Annotation Function...")
print("-" * 70)

try:
    from api.services.ibm_service import _annotate_winter_rye_biomass
    
    # Create sample daily data (like IBM API returns)
    dates = pd.date_range('2024-09-15', '2025-05-15', freq='D')
    sample_daily = pd.DataFrame({
        'date': dates,
        'temperature_min': 10 + (dates.dayofyear % 100) * 0.02,
        'temperature_max': 20 + (dates.dayofyear % 100) * 0.02,
        'precip1Hour_sum': 2.0 if (dates.dayofyear % 5) == 0 else 0,
    })
    
    print(f"✅ Created sample daily data ({len(sample_daily)} rows)")
    print(f"   Date range: {sample_daily['date'].min()} to {sample_daily['date'].max()}")
    
    # Call annotation
    result_df = _annotate_winter_rye_biomass(sample_daily, '2024-09-15')
    
    print(f"✅ Annotation function executed")
    
    # Check for biomass columns
    biomass_cols = [c for c in result_df.columns if 'biomass' in c.lower()]
    
    if biomass_cols:
        print(f"✅ Biomass columns created: {biomass_cols}")
        
        # Check for null values
        for col in biomass_cols:
            null_count = result_df[col].isna().sum()
            non_null_count = (result_df[col].notna()).sum()
            
            print(f"   {col}: {non_null_count} values, {null_count} nulls")
            
            if null_count == len(result_df):
                print(f"   ❌ ALL VALUES ARE NULL")
            elif null_count > 0:
                print(f"   ⚠️  PARTIAL NULLS ({null_count}/{len(result_df)})")
            else:
                print(f"   ✅ No nulls")
                
                # Show sample values
                print(f"      Sample values: {result_df[col].tail(3).values}")
    else:
        print(f"❌ No biomass columns created")
        print(f"   Available columns: {list(result_df.columns)}")
        
except Exception as e:
    print(f"❌ Annotation function failed: {e}")
    import traceback
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────
# 4. TEST API ENDPOINT
# ─────────────────────────────────────────────────────────────────────────

print("\n[4/4] Testing API Endpoint...")
print("-" * 70)

try:
    import requests
    
    # Try test endpoint (simplest)
    response = requests.get('http://localhost:8000/v2/test/rye_biomass')
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Test endpoint responds")
        
        if data.get('result', {}).get('biomass_lb_acre'):
            print(f"   Biomass value: {data['result']['biomass_lb_acre']:.2f} lb/acre")
            print(f"   Color: {data['result']['biomass_color']}")
        else:
            print(f"❌ Test endpoint returned null biomass")
            pprint(data)
    else:
        print(f"❌ Test endpoint returned {response.status_code}")
        print(f"   {response.text}")
        
except Exception as e:
    print(f"❌ Could not reach API: {e}")

print("\n" + "=" * 70)
print("END OF DIAGNOSTIC")
print("=" * 70)
```

Run it:
```bash
cd /Users/mariaoros/Documents/DataScienceInstitute/2026/ag_forecasting_api
python diagnose_biomass.py
```

---

## Common Issues & Solutions

### Issue 1: "Biomass fields are null in response"

**Diagnosis:**
```python
# Check if biomass columns exist but have null values
features = response_geojson['features']
for feature in features:
    props = feature['properties']
    if 'biomass_lb_acre' in props:
        if props['biomass_lb_acre'] is None:
            print("❌ Column exists but is null")
        else:
            print(f"✅ Biomass: {props['biomass_lb_acre']}")
    else:
        print("❌ Column doesn't exist in response")
```

**Likely Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| `planting_date` not passed to API | Add query parameter: `&planting_date=2024-09-15` |
| `planting_date` not forwarded to service | Check `app.py` route passes param to `get_weather_with_risk()` |
| Biomass function not called | Add `if planting_date:` check in `_annotate_risks()` |
| No weather data after planting | Check forecast_date is AFTER planting_date |
| Weather columns have wrong names | Check IBM response has `temperature_min`, `temperature_max`, `precip1Hour_sum` |

---

### Issue 2: "Model produces wrong/unexpected values"

**Check these inputs:**

```python
# These should be realistic for Wisconsin rye
plant_doy = 260          # Should be 250-290 (mid-Sep to mid-Oct)
precip_fall = 100.0      # Should be 50-200mm
gdd_total = 2000.0       # Should be 1500-3000 by May

# Check your actual values
print(f"plant_doy: {plant_doy} (expected 250-290)")
print(f"precip_fall: {precip_fall} (expected 50-200)")
print(f"gdd_total: {gdd_total} (expected 1500-3000)")

# Typical output ranges
# - Biomass at May 15: 3000-5000 lb/acre (Green color)
# - Biomass at March 15: 1500-2500 lb/acre (Yellow color)
# - Biomass at Jan 15: 500-1500 lb/acre (Gray color)
```

---

### Issue 3: "GDD keeps returning 0"

Check temperature data:

```python
# If both temps are below base (0°C), GDD = 0
# This is normal in winter
tmin, tmax = -5, -2  # Both below 0°C base
gdd = gdd_sine(tmin, tmax)  # Returns 0

# In spring, should get positive GDD
tmin, tmax = 5, 15  # Both above 0°C
gdd = gdd_sine(tmin, tmax)  # Returns ~7-8 GDD

# Check your temperature data is in Celsius
if tmin > 50 or tmax > 50:
    print("❌ Temperatures likely in Fahrenheit, not Celsius!")
```

---

## Viewing Logs in Production

If running with Docker:

```bash
# View real-time logs
docker-compose logs -f ag_forecasting_api

# View with search for biomass-related logs
docker-compose logs ag_forecasting_api | grep -i biomass
```

If running locally:

```bash
# In the terminal where uvicorn is running, you'll see logs
# Look for lines containing "biomass" or your debug statements

# Or redirect to file and tail it
uvicorn app:app --reload > api.log 2>&1 &
tail -f api.log | grep -i biomass
```

---

## Summary Decision Tree

```
Does /v2/test/rye_biomass return a non-null biomass?
├─ YES → Model math works ✅
│         Try full endpoint: /v2/ag_models_wrappers/ibm?planting_date=...
│         └─ Does it return biomass?
│            ├─ YES → All working! ✅
│            └─ NO → Check planting_date is being passed through route
│
└─ NO → Model math broken ❌
        Check calculate_winter_rye_biomass() in api/models/risk_models.py
```

---

**Next Steps:**
1. Run `diagnose_biomass.py` above - it will tell you exactly which layer is failing
2. Share the output and I'll help you fix the specific issue
3. Once fixed, verify with `/v2/test/rye_biomass` then full endpoint tests
