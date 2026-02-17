# Open Source Agricultural Forecasting API

This API provides access to crop disease models developed by the University of wisconesin Madison experts in plant pathology. The API uses a FastAPI-based backend and integrates weather station data from public and private sources, including mesonet weather stations and IBM Environmental Intelligence. It also supports querying Wisconet by a wrapper built in top of it in a custom way. 


[API](https://connect.doit.wisc.edu/ag_forecasting_api/docs)

Table of Contents

- [Features](#features)
- [A use case of our API Integration: visualizing the Ag forecasting for Wisconsin](#a-use-case-of-our-api-integration-visualizing-the-ag-forecasting-for-wisconsin)
- [Agriculture models: ag_models_wrappers folder](#agriculture-models-ag_models_wrappers-folder)
- [Plant Disease models](#plant-disease-models)
- [Running the API](#running-the-api)
- [Ag Forecasting Endpoints](#ag-forecasting-endpoints)
  - [IBM Weather Data](#ibm-weather-data)
  - [Wisconet Weather Data](#wisconet-weather-data)
- [pywisconet Endpoints](#pywisconet-endpoints)
  - [Get Station Fields](#get-station-fields)
  - [Bulk Measurements](#bulk-measurements)
  - [Active Stations](#active-stations)
- [Root Endpoint](#root-endpoint)
- [WSGI Compatibility](#wsgi-compatibility)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Features

- Crop disease forecasting models for corn and soybean.
  - IBM Weather Integration: Fetch and clean daily weather data from IBM Weather API.
  - Wisconet Weather Data: Access weather data aggregated daily from Wisconet.
- Station Fields Retrieval: Get details of a weather station by its ID.
- Bulk Measurements: Retrieve weather measurements for a station within a specified date range and frequency.
- Active Stations Query: List stations based on active days and a provided start date.

## A use case of our API Integration: visualizing the Ag forecasting for Wisconsin

Visit our API application through our interactive Dashboard:
- [Link](https://connect.doit.wisc.edu/ag_forecasting/)
- [GitHub Repo](https://github.com/UW-Madison-DSI/corn_disease_forecast_api.git)

## Agricultural crop disease forecasting models: ag_models_wrappers folder

The ag_models_wrappers serve as the critical layer for providing crop model-based risk assessments tailored to weather data on specific locations eg Wisconet Stations or punctual locations in Wisconsin by IBM data. This component integrates various forecasting models to deliver localized risk predictions for plant diseases for a given forecasting date, enabling informed decision-making in agricultural management.
- Ag Forecasting based on Wisconet API Wrapper: This REST API simplifies communication between the Wisconet API service and our ag forecasting logic. We developed an API that dynamically retrieves data from available weather stations for a given forecasting date, fetching daily and hourly variables necessary for the input of various crop disease models.
- Ag Forecasting based on IBM API Wrapper: This REST API facilitates communication between the IBM paid service and our ag forecasting logic. Access to the IBM service is secured with an API key.
Both APIs are open-source and can be integrated into your processes. Note that the IBM API requires an API key for access.

See a python example on how to programatically use our API: https://github.com/UW-Madison-DSI/ag_forecasting_api/blob/main/materials/example_callapi.ipynb

### Plant disease models

Selected field crops and vegetable disease model outputs are provided. These models are subject to change. The calculations used to generate each model prediction can be viewed in the source code.

**Soybean Crop Disease** White mold (aka Sporecaster), probability of apothecial presence. More information: https://cropprotectionnetwork.org/news/smartphone-application-to-forecast-white-mold-in-soybean-now-available-to-growers

  - dry 
  - irrigated 15-inch row spacing
  - irrigated 30-inch row spacing
  
**Corn Crop Disease**

- Frogeye Leaf Spot - More information: https://cropprotectionnetwork.org/encyclopedia/frogeye-leaf-spot-of-soybean
- Gray Leaf Spot of corn - More information: https://cropprotectionnetwork.org/encyclopedia/gray-leaf-spot-of-corn
- Tar Spot of corn (aka Tarspotter) - More information: https://cropprotectionnetwork.org/encyclopedia/tar-spot-of-corn

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


## Running the API Locally

### Installation

```commandline
git clone https://github.com/UW-Madison-DSI/ag_forecasting_api.git
cd pywisconet

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

```

### Run locally


```commandline
uvicorn app:app --reload

```
The API will be available at http://127.0.0.1:8000.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements
- This work is an Open-Source initiative from the [Open Source Program Office at the University of Madison Wisconsin](https://ospo.wisc.edu), aimed at fostering collaboration and innovation in open source forecasting tools.
- The models presented are based on plant pathology research in the University of Madison Wisconsin, paper: [Nature Scientific Reports, 2023](https://www.nature.com/articles/s41598-023-44338-6)
- This software was created by the [Data Science Institute](https://datascience.wisc.edu) at the [University of Wisconsin-Madison](https://www.wisc.edu)

Mantainer: Maria Oros, maria.oros@wisc.edu
