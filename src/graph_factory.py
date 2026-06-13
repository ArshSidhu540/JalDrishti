import networkx as nx
import torch
from src.data_pipeline import load_and_spatialise_data

def build_hydrological_graph(csv_path: str, radius_km: float = 100.0) -> nx.Graph:
    gdf = load_and_spatialise_data(csv_path)
    num_stations = len(gdf)
    G = nx.Graph()
    
    for idx, row in gdf.iterrows():
        G.add_node(idx, station_id=row['station_id'], name=row['station_name'], pos=(row['geometry'].x, row['geometry'].y))
        
    radius_meters = radius_km * 1000.0
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            point_i = gdf.loc[i, 'geometry']
            point_j = gdf.loc[j, 'geometry']
            distance_meters = point_i.distance(point_j)
            if distance_meters <= radius_meters:
                G.add_edge(i, j, distance_km=(distance_meters / 1000.0))
    return G

def extract_graph_matrices(G: nx.Graph):
    edges = list(G.edges())
    if len(edges) > 0:
        edge_list_directed = []
        for u, v in edges:
            edge_list_directed.append([u, v])
            edge_list_directed.append([v, u])
        edge_index = torch.tensor(edge_list_directed, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return None, edge_index