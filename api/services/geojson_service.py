def dataframe_to_geojson(df):
    features = []

    for station, group in df.groupby("station_id"):
        first_row = group.iloc[0]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    first_row["longitude"],
                    first_row["latitude"],
                ],
            },
            "properties": {
                "station_id": station,
                "station_name": first_row.get("station_name"),
                "city": first_row.get("city"),
                "county": first_row.get("county"),
                "region": first_row.get("region"),
                "state": first_row.get("state"),
                "time_series": group.drop(
                    columns=["latitude", "longitude"]
                ).to_dict("records"),
            },
        }

        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }