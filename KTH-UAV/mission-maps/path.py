import folium
import pandas as pd
from folium import plugins

def add_mission_path(m: folium.Map, df: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lot', zoom_start: int = 11):
    loc = df[[lat_col,lon_col]].values.tolist()
    plugins.AntPath(loc).add_to(m)
    return m