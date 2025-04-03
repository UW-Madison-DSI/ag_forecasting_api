# Open Source Ag Forecasting API

This API provides access to weather-related data using a FastAPI-based backend. The application integrates weather station data and forecasting from multiple sources, including IBM Weather and Wisconet. It also supports querying bulk measurements for stations over specified date ranges.

Table of Contents

- Features
- A use case of our API Integration: visualizing the Ag forecasting for Wisconsin
- Agriculture models: ag_models_wrappers folder
- Installation
- Running the API
- Ag Forecasting Endpoints
  - IBM Weather Data
  - Wisconet Weather Data
- pywisconet Endpoints
  - Get Station Fields
  - Bulk Measurements
  - Active Stations
- Root Endpoint
- WSGI Compatibility
- Environment Variables
- Notes

## Features

- Station Fields Retrieval: Get details of a weather station by its ID.
- Bulk Measurements: Retrieve weather measurements for a station within a specified date range and frequency.
- Active Stations Query: List stations based on active days and a provided start date.
- IBM Weather Integration: Fetch and clean daily weather data from IBM Weather API.
- Wisconet Weather Data: Access weather data aggregated daily from Wisconet.

## A use case of our API Integration: visualizing the Ag forecasting for Wisconsin
Visit our API application through our interactive Dashboard:
- [Link](https://connect.doit.wisc.edu/ag_forecasting/)
- [GitHub Repo](https://github.com/UW-Madison-DSI/corn_disease_forecast_api.git)

## Agriculture models: ag_models_wrappers folder
The ag_models_wrappers serve as the critical layer for providing crop model-based risk assessments tailored to weather data on specific locations eg Wisconet Stations or punctual locations in Wisconsin by IBM data. This component integrates various forecasting models to deliver localized risk predictions for plant diseases for a given forecasting date, enabling informed decision-making in agricultural management.
- Ag Forecasting based on Wisconet API Wrapper: This REST API simplifies communication between the Wisconet API service and our ag forecasting logic. We developed an API that dynamically retrieves data from available weather stations for a given forecasting date, fetching daily and hourly variables necessary for the input of various crop disease models.
- Ag Forecasting based on IBM API Wrapper: This REST API facilitates communication between the IBM paid service and our ag forecasting logic. Access to the IBM service is secured with an API key.
Both APIs are open-source and can be integrated into your processes. Note that the IBM API requires an API key for access.

See a python example on how to programatically use our API: https://github.com/UW-Madison-DSI/ag_forecasting_api/blob/main/materials/example_callapi.ipynb

## Installation

```commandline
git clone https://github.com/UW-Madison-DSI/ag_forecasting_api.git
cd pywisconet

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

```

## Running the API
```commandline
uvicorn main:app --reload

```
The API will be available at http://127.0.0.1:8000.

## Ag Forecasting Endpoints

### Ag Forecasting for a given (or all) Available Wisconet Stations Weather Data for a given forecasting date
Endpoint: /ag_models_wrappers/wisconet
Method: GET
Query Parameters:

- forecasting_date: Date for the forecast in YYYY-MM-DD format.
- risk_days: (Optional) Number of risk days (default is 1).
- station_id: (Optional) Station ID for filtering the results.
Usage Example:
```commandline
GET /ag_models_wrappers/wisconet?forecasting_date=2024-07-01&risk_days=1&station_id=ALTN

```

**See more examples in the** notebooks/examples.ipynb



### Ag Forecasting for a given forecasting date and location (lat, lon) sourced from IBM Weather Data
Endpoint: /ag_models_wrappers/ibm
Method: GET
Path Parameter:

- forecasting_date: Date for the forecast in YYYY-MM-DD format.
- Query Parameters:
- latitude: Latitude of the location.
- longitude: Longitude of the location.
- API_KEY: IBM API key.
- TENANT_ID: Tenant ID.
- ORG_ID: Organization ID.
Usage Example:
```commandline
GET /ag_models_wrappers/ibm?forecasting_date=2024-07-01&latitude=41.8781&longitude=-87.6298&API_KEY=your_key&TENANT_ID=your_tenant&ORG_ID=your_org

```

## Root Endpoint
Endpoint: /
Method: GET
Description: A simple welcome message to indicate that the API is up and running.
Usage Example:
```commandline
GET /
```

## WSGI Compatibility

This API includes a WSGI application wrapped with Starlette’s WSGIMiddleware for compatibility with WSGI servers. The create_wsgi_app function creates a WSGI application that delegates HTTP requests to the FastAPI app. This can be useful when integrating with legacy systems or deploying in environments that require a WSGI interface.

### Environment Variables

For the IBM Weather data endpoint to work properly, ensure that you set the following environment variables in your deployment environment:

- IBM_API_KEY
- TENANT_ID
- ORG_ID
These credentials are validated against the values provided in the query parameters.

## pywisconet
We provide also an API wrapper that represents an interface for interacting with Wisconet v1 data. The intended objective is to provide a streamlined method for accessing weather and environmental data from Wisconet's network of stations, enabling users to integrate this data into their own applications and forecasting models for agricultural and environmental analysis.


API Documentation: [API](https://connect.doit.wisc.edu/pywisconet_wrapper/docs)

For more information on how to use our API, please visit the material section.

### Active Wisconet Stations
Endpoint: /wisconet/active_stations/
Method: GET
Query Parameters:

min_days_active: Minimum number of days a station should have been active.
start_date: Start date in YYYY-MM-DD format.
Usage Example:
```commandline
GET /wisconet/active_stations/?min_days_active=10&start_date=2024-07-01

```

### Get Wisconet Station Fields
Endpoint: /station_fields/{station_id}
Method: GET
Description: Retrieves the fields associated with the specified station.
Usage Example:
```commandline
GET /station_fields/ALTN

```

### Bulk Wisconet Weather Measurements
Endpoint: /bulk_measures/{station_id}
Method: GET
Query Parameters:

- start_date: Start date in YYYY-MM-DD format (e.g., 2024-07-01) assumed to be in Central Time (CT).
- end_date: End date in YYYY-MM-DD format (e.g., 2024-07-02) assumed to be in CT.
- measurements: Measurement types (e.g., AIRTEMP, DEW_POINT, WIND_SPEED, RELATIVE_HUMIDITY, or ALL for the last four).
- frequency: Frequency of measurements (e.g., MIN60, MIN5, DAILY).
Usage Example:
```commandline
GET /bulk_measures/ALTN?start_date=2024-07-01&end_date=2024-07-02&measurements=AIRTEMP&frequency=MIN60

```

## Acknowledgements:
This is an initiative of the Open source Program Office at the University of Madison-Wisconsin.

[API](https://connect.doit.wisc.edu/pywisconet_wrapper/docs#/default/all_data_from_wisconet_query_ag_models_wrappers_wisconet_get)


**Matainer**
- Maria Oros, Data Scientist, Data Science Institute at UW-Madison