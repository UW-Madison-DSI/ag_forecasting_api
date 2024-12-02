from pathlib import Path
import httpx
from datetime import datetime

from pywisconet.schema import (
       BASE_URL, Station, Field, BulkMeasures, 
)


def all_stations(n_days_active: int) -> list[Station]:
    """
    Get all current Wisconet stations.
    :return: list of Station objects
    """
    route = "/stations/"
    stations = []
    if n_days_active is None:
        n_days_active=0

    with httpx.Client(base_url=BASE_URL) as client:
        response = client.get(route)
        response.raise_for_status()

    for station in response.json():
        station_tz = station.pop("station_timezone")
        earliest_api_date = datetime.strptime(station.pop("earliest_api_date"), "%m/%d/%Y")
        current_date = datetime.now()

        # Calculate the number of days the station has been active
        days_active = (current_date - earliest_api_date).days
        #print("Days active:", days_active)
        elevation = float(station.pop("elevation"))
        latitude = float(station.pop("latitude"))
        longitude = float(station.pop("longitude"))
        if days_active > n_days_active:
            stations.append(
                Station(
                    station_timezone=station_tz,
                    earliest_api_date=earliest_api_date,
                    days_active=days_active,
                    elevation=elevation,
                    latitude=latitude,
                    longitude=longitude,
                    **station,  # Include any additional fields dynamically
                )
            )
    return stations


def station_fields(station_id: str) -> list[Field]:
    """
    Get the Field objects available for a station.
    :param station_id: station_id e.g "ALTN".
    :return: list of Field objects
    """
    route = f"/fields/{station_id}/available_fields"
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.get(route)
        response.raise_for_status()
    return [Field(**field) for field in response.json()]


def bulk_measures(station_id: str, start_time: datetime, end_time: datetime, fields: list[str] | None = None, timeout: float = 30.0) -> BulkMeasures:
    """
    Get measures for a station between two times.
    :param station_id: Station.station_id e.g "ALTN".
    :param start_time: datetime, fetch start time in UTC.
    :param end_time: datetime fetch end time in UTC
    :param fields: optional list of Field.standard_name strings of fields to return. If not specified, returns all fields.
    :param timeout: float, httpx timeout.
    :return: BulkMeasures object
    """
    start_time_epoch = int(start_time.timestamp())
    end_time_epoch = int(end_time.timestamp())
    route = f"/stations/{station_id}/measures"
    params = {
        "start_time": start_time_epoch,
        "end_time": end_time_epoch,
    }
    if fields:
        params["fields"] = ",".join(fields)
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        response = client.get(route, params=params)
        response.raise_for_status()
    return BulkMeasures(**response.json())


def all_data_for_station(s: Station, fields: list[str] | None = None, timeout=60.0) -> BulkMeasures:
    """
    Get all available data for a Station.
    :param s: Station object.
    :param fields: optional list of Field.standard_name strings of fields to return. If not specified, returns all fields.
    :param timeout: float, httpx timeout.
    :return: BulkMeasures object.
    """
    return bulk_measures(
        station_id=s.station_id,
        start_time=s.earliest_api_date,
        end_time=datetime.now(),
        fields=fields,
        timeout=timeout,
    )


def all_data_for_station_stream_to_disk(s: Station, output_dir: Path, overwrite=False, timeout: float = 60.0) -> Path:
    """
    Get all available dagta for a Station.
    :param s: Station object.
    :param output_dir: Dir to write the data to, filename: "<STATION_ID>_measures.json".
    :param overwrite: bool, overwrite the file if it already exists.
    :param timeout: float, httpx timeout.
    :return: BulkMeasures object.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    file_name = f"{s.station_id}_measures.json"
    file_path = output_dir / file_name
    if file_path.exists():
        if not overwrite:
            raise FileExistsError(f"{file_path} already exists.")
        else:
            file_path.unlink()
    params = {
        "start_time": int(s.earliest_api_date.timestamp()),
        "end_time": int(datetime.now().timestamp()),
    }
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    url = BASE_URL + f"/stations/{s.station_id}/measures?{param_str}"
    with open(file_path, "a") as f:
        with httpx.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                f.write(line)
    return file_path
