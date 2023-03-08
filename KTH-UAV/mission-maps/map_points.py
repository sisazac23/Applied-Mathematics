import folium
from folium import plugins
import pandas as pd

def heatmap_points(df: pd.DataFrame, lat_col: str = 'lat', lon_col: str = 'lot', zoom_start: int = 11, \
                plot_points: bool = False, pt_radius: int = 15, \
                draw_heatmap: bool = False, heat_map_weights_col: str = None, \
                heat_map_weights_normalize: bool = True, heat_map_radius: int = 15):
    """Creates a map given a dataframe of points. Can also produce a heatmap overlay

    Arg:
        df: dataframe containing points to maps
        lat_col: Column containing latitude (string)
        lon_col: Column containing longitude (string)
        zoom_start: Integer representing the initial zoom of the map
        plot_points: Add points to map (boolean)
        pt_radius: Size of each point
        draw_heatmap: Add heatmap to map (boolean)
        heat_map_weights_col: Column containing heatmap weights
        heat_map_weights_normalize: Normalize heatmap weights (boolean)
        heat_map_radius: Size of heatmap point

    Returns:
        folium map object
    """

    ## center map in the middle of points center in
    middle_lat = df[lat_col].median()
    middle_lon = df[lon_col].median()

    curr_map = folium.Map(location=[middle_lat, middle_lon],
                          zoom_start=zoom_start)

    # add points to map with circles without circunference


    if plot_points:
        for _, row in df.iterrows():
            #Define radius of circle based on the values of the heat_maps_weights_col, making int bigger if the value is bigger
            folium.CircleMarker([row[lat_col], row[lon_col]],
                                radius=pt_radius*(row[heat_map_weights_col]/df[heat_map_weights_col].max()),
                                #popup=row['name'],
                                fill_color="#3db7e4", # divvy color,
                                border_width=0
                               ).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            cols_to_pull = [lat_col, lon_col]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                df[heat_map_weights_col] = \
                    df[heat_map_weights_col] / df[heat_map_weights_col].sum()

            cols_to_pull = [lat_col, lon_col, heat_map_weights_col]

        stations = df[cols_to_pull].values
        curr_map.add_child(plugins.HeatMap(stations, radius=heat_map_radius))

    return curr_map





