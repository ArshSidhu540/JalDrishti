import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_and_spatialise_data(csv_path: str) -> gpd.GeoDataFrame:
    """Loads the real Kaggle groundwater dataset and projects it to Indian UTM coordinates."""
    df = pd.read_csv(csv_path)
    
    # Map directly to your true Kaggle column headers
    lat_col = 'latitude'  
    lon_col = 'longitude'
    
    # Clean rows missing geographical positioning data
    df = df.dropna(subset=[lat_col, lon_col])
    
    # Build spatial geometries
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # Convert WGS84 degrees to Projected flat meters (India UTM Zone 43N)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf_projected = gdf.to_crs(epsg=32643)
    
    return gdf_projected