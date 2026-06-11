import numpy as np
import networkx as nx
import torch
from spatial_pipeline import load_and_spatialise_data

def build_hydrological_graph(csv_path: str, radius_km: float = 100.0) -> nx.Graph:
    """
    Takes the path to groundwater data, spatializes it, and constructs
    a spatial network graph based on a maximum distance radius threshold.
    """
    # 1. Fetch our projected GeoDataFrame from Phase 2
    gdf = load_and_spatialise_data(csv_path)
    num_stations = len(gdf)
    
    # 2. Initialize an empty undirected NetworkX Graph
    G = nx.Graph()
    
    # 3. Add Nodes with features
    print("\n--- Initializing Graph Nodes ---")
    for idx, row in gdf.iterrows():
        G.add_node(
            idx,  # Integer ID representing the node index
            station_id=row['station_id'],
            name=row['station_name'],
            water_depth=row['water_table_depth_bgl'],
            # Store the projected Easting and Northing coordinates
            pos=(row['geometry'].x, row['geometry'].y)
        )
    print(f"Added {G.number_of_nodes()} groundwater monitoring nodes to the graph topology.")
    
    # 4. Compute Pairwise Distances and Construct Edges
    print(f"\n--- Constructing Edges (Radius Threshold: {radius_km} km) ---")
    radius_meters = radius_km * 1000.0  # Convert km to meters to match our projected CRS unit
    
    # Nested loops to compare every station pair
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            # Extract geometry points
            point_i = gdf.loc[i, 'geometry']
            point_j = gdf.loc[j, 'geometry']
            
            # Calculate standard Euclidean distance in flat meters
            distance_meters = point_i.distance(point_j)
            distance_km = distance_meters / 1000.0
            
            if distance_meters <= radius_meters:
                G.add_edge(i, j, distance_km=distance_km)
                print(f"🔗 Edge Created: {gdf.loc[i, 'station_name']} <---> {gdf.loc[j, 'station_name']} | Distance: {distance_km:.2f} km")
                
    return G

def extract_graph_matrices(G: nx.Graph):
    """
    Converts a NetworkX graph into standard matrices and tensors 
    required for deep learning applications.
    """
    print("\n--- Exporting Graph Representations ---")
    
    # 1. Extract Dense Adjacency Matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    print("Formulated Adjacency Matrix (A):")
    print(adj_matrix)
    
    # 2. Extract PyTorch Geometric Edge Index format
    # nx.to_edgelist returns tuple list: (source, target, attributes_dict)
    edges = list(G.edges())
    
    if len(edges) > 0:
        # For undirected graphs, we must add edges in BOTH directions (source->target AND target->source)
        edge_list_directed = []
        for u, v in edges:
            edge_list_directed.append([u, v])
            edge_list_directed.append([v, u])
            
        # Convert to a PyTorch long Tensor and transpose to get [2, num_edges] layout
        edge_index = torch.tensor(edge_list_directed, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    print("\nGenerated PyTorch Geometric Edge Index shape:")
    print(edge_index.shape)
    print("Edge Index Tensor layout:")
    print(edge_index)
    
    return adj_matrix, edge_index

if __name__ == "__main__":
    csv_file_path = "data/raw/india_groundwater_2025.csv"
    
    # Build graph with a strict 100km connection window
    hydrology_graph = build_hydrological_graph(csv_file_path, radius_km=100.0)
    extract_graph_matrices(hydrology_graph)