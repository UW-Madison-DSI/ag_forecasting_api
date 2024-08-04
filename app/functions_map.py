import folium
import streamlit as st
from streamlit_folium import st_folium
import requests
from geopy.distance import geodesic

def map_creation1(stationslist, highlight_station_id=None):
    # Create a map centered at the average latitude and longitude of all stations
    MAP_TILER_KEY = st.secrets["MAP_TILER_KEY"]
    average_latitude = sum(station["latitude"] for station in stationslist.values()) / len(stationslist)
    average_longitude = sum(station["longitude"] for station in stationslist.values()) / len(stationslist)

    map = folium.Map(location=[average_latitude, average_longitude], zoom_start=7)

    # Map layers
    base_maps = {
        "Esri": folium.TileLayer(
            name='Carto CDN',
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
                      'World_Topo_Map/MapServer/tile/{z}/{y}/{x}.png',
            attr='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, '
                       'FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, '
                       'Esri China (Hong Kong), and the GIS User Community'
        ),
        "Map Tiler": folium.TileLayer(
            name='Map Tiler',
            tiles='https://api.maptiler.com/maps/topo/{z}/{x}/{y}.png?key=' + str(MAP_TILER_KEY),
            attr='Map data © OpenStreetMap contributors, Imagery © MapTiler'
        )
    }

    # base map
    for name, tile_layer in base_maps.items():
        tile_layer.add_to(map)

    # Layers
    folium.LayerControl().add_to(map)

    # map marks
    for station_id, station in stationslist.items():
        marker_color = "blue"
        if station_id == highlight_station_id:
            marker_color = "yellow"

        folium.Marker(
            location=[station["latitude"], station["longitude"]],
            popup=(
                f"<strong>{station['name']}</strong><br>"
                f"Location: {station['location']}<br>"
                f"Region: {station['region']}<br>"
                f"State: {station['state']}"
            ),
            icon=folium.Icon(color=marker_color)
        ).add_to(map)

    # Render
    st.title("Wisconet Station Map")
    st.write("Wisconet's Weather Map Stations. Click on any station to display its information, or choose from the list alongside the map.")
    st_folium(map, width=700, height=500)

def get_ip_location():
    """

    :return:
    """
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    loc = data.get('loc', '').split(',')
    return float(loc[0]), float(loc[1])

# Function to find the nearest station
def find_nearest_station(user_location):
    """

    :param user_location:
    :return: station name, station code
    """
    min_distance = float('inf')
    nearest_station = None
    cd = None
    for code, info in stationslist.items():
        station_location = (info['latitude'], info['longitude'])
        distance = geodesic(user_location, station_location).km
        if distance < min_distance:
            min_distance = distance
            nearest_station = info
            cd = code
    return nearest_station, cd
