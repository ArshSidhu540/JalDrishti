import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Append current directory path so python locates our src/ package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import load_and_spatialise_data
from src.graph_factory import build_hydrological_graph

def run_real_jaldrishti_map():
    print("=========================================================")
    print("🌐 JALDRISHTI: SCALED GEOSPATIAL VISUALIZATION ENGINE  🌐")
    print("=========================================================\n")
    
    real_csv_path = "data/raw/india_groundwater_2025.csv"
    
    if not os.path.exists(real_csv_path):
        print(f"❌ Error: Could not find your dataset at {real_csv_path}")
        print("Please ensure your downloaded Kaggle file is renamed and moved to that path.")
        return

    print("[Step 1/3] Ingesting multi-year dataset and extracting unique spatial stations...")
    
    # Pre-processing: If the file has multiple years of readings per station,
    # we filter for unique stations to construct a clean spatial network map.
    raw_df = pd.read_csv(real_csv_path)
    unique_stations_df = raw_df.drop_duplicates(subset=['station_id'])
    
    # Save a temporary static file representing unique physical nodes
    filtered_path = "data/raw/unique_stations_topology.csv"
    unique_stations_df.to_csv(filtered_path, index=False)
    
    print(f" -> Found {len(unique_stations_df)} unique monitoring stations across regions.")
    print(" -> Constructing spatial network topology using a 100km radius cutoff...")
    
    # 2. Build the graph using our real unique geographical nodes
    G = build_hydrological_graph(filtered_path, radius_km=100.0)
    
    # 3. Render the Map
    print("\n[Step 2/3] Computing graph layout coordinates...")
    positions = nx.get_node_attributes(G, 'pos')
    
    if len(positions) == 0:
        print("❌ Error: No geographical positions were generated. Check your coordinate formatting.")
        return
        
    print("\n[Step 3/3] Launching Interactive Map Window...")
    plt.figure(figsize=(12, 10))
    
    # Draw physical connection links (Edges) between sharing aquifers
    nx.draw_networkx_edges(G, positions, width=0.6, edge_color='skyblue', alpha=0.5)
    
    # Draw physical station markers (Nodes)
    nx.draw_networkx_nodes(G, positions, node_size=15, node_color='teal', alpha=0.5)
    
    plt.title(f"JalDrishti: Hydrological Network Map\nTopology consisting of {G.number_of_nodes()} Interconnected Stations (100km Sharing Radius)", fontsize=12, fontweight='bold')
    plt.xlabel("Easting Coordinate (UTM Zone 43N - Meters)", fontsize=10)
    plt.ylabel("Northing Coordinate (UTM Zone 43N - Meters)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Clean up temporary filter file
    if os.path.exists(filtered_path):
        os.remove(filtered_path)
        
    print("\n📊 Success! Displaying map window. Close the map viewer window to exit.")
    plt.show()

if __name__ == "__main__":
    run_real_jaldrishti_map()