========================================================================
🌐 JALDRISHTI: SPATIO-TEMPORAL HYDROLOGICAL INTELLIGENCE PLATFORM
========================================================================
Author: Arshdeep Kaur
Degree: B.Tech Computer Science Engineering
University: Maharaja Ranjit Singh Punjab Technical University
Campus: Giani Zail Singh Campus College of Engineering & Technology, Bathinda
Domain: Spatio-Temporal Machine Learning, Graph Neural Networks (GNN), MLOps
Dataset Focus: India Groundwater Climate Time-Series (1994–2025)

---
📌 PROJECT OVERVIEW
---
JalDrishti ("Water Vision") is an advanced end-to-end Machine Learning and MLOps 
pipeline designed to track, model, and predict critical groundwater depletion 
and aquifer health parameters across India.

Traditional hydrological models analyze monitoring stations as isolated time-series, 
failing to account for the spatial reality of contiguous underground aquifers. 
When multiple agricultural zones pump water simultaneously, they create "cones of 
depression" that draw groundwater away from neighboring rural community sectors 
laterally. 

JalDrishti addresses this limitation by fusing spatial GIS data with climate 
time-series records, constructing a Hydrological Network Graph where physical 
monitoring stations serve as Nodes, and shared aquifer boundaries act as Edges. 
Using a Spatio-Temporal Graph Convolutional Network (STGNN) architecture, the 
engine models physical message-passing between adjacent well coordinates over a 
30-year horizon to predict regional water table shifts and issue preemptive community 
advisories.

---
🚀 CORE ARCHITECTURAL MODULES
---
1. Ingestion & GIS Pipeline (`src/data_pipeline.py`)
   - Parses multi-year raw national hydro-meteorological data.
   - Cleans spatial coordinate pairs and projects standard WGS84 degrees (EPSG:4326) 
     into flat Cartesian meters matching the localized Indian UTM reference frame 
     (India UTM Zone 43N - EPSG:32643) via GeoPandas.

2. Topological Graph Factory (`src/graph_factory.py`)
   - Computes physical geospatial Haversine distance matrices across monitoring pairs.
   - Dynamically instantiates network links (Edges) using an adjustable spatial radius 
     cutoff threshold (default: 100km sharing boundary) via NetworkX.

3. Graph Deep Learning Core (`main.py`)
   - Formulates sparse tensor data structures (Edge Index, Node Feature Matrix) 
     compatible with PyTorch Geometric (PyG).
   - Simulates forward message-passing loops via spectral Graph Convolutional 
     Layers (GCN) implementing the network aggregation equation:
     H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )

4. Unified Web Intelligence Dashboard (`app.py`)
   - A live, production-grade web interactive interface engineered in Streamlit.
   - Integrates an advanced Web File Uploader Widget, allowing users or evaluators 
     to drag and drop new regional groundwater CSV files dynamically.
   - Automatically re-calculates graph layouts, updates KPI cards, renders zoomable 
     geographical map marker grids via Folium, charts climate-drawdown metrics, 
     and outputs GNN projection data frames along with localized strategic solutions.

---
📂 DETAILED REPOSITORY ARCHITECTURE
---
The JalDrishti project is built following strict MLOps modular design principles, 
separating data ingestion, graph processing models, exploratory work, and user 
interfaces. Below is the production tree structure:

```text
JalDrishti/
├── .gitignore                   # Excludes massive datasets (.csv) and caches from tracking
├── app.py                       # Core Streamlit web application & user interface logic
├── main.py                      # Orchestrator running baseline structural GNN framework checks
├── README.md                    # Main repository documentation & deployment guide
│
├── data/                        # Integrated Data Storage Layer (Excluded from GitHub)
│   ├── raw/                     # Houses Kushvinth Madhavan's 30-Year source CSV files
│   └── variants/                # Clean regional testing sub-slices for dynamic web uploads
│
├── notebooks/                   # R&D and Exploratory Sandbox Environment
│   ├── inspect_kaggle.py        # Validates incoming raw CSV structural layouts
│   ├── clean_incoming_data.py   # Maps raw external columns to internal system features
│   └── generate_dashboard_variants.py # Automatic script generating targeted regional datasets
│
└── src/                         # Production-Grade Modular Source Package
    ├── __init__.py              # Bundles directory as a seamless, importable Python package
    ├── data_pipeline.py         # Handles GeoPandas GIS spatial datum UTM re-projections
    └── graph_factory.py         # Spatially computes radial distance-based NetworkX topologies
```

---
📦 SYSTEM PRE-REQUISITES & INSTALLATION
---
To install dependencies and run the project locally, open your terminal within the 
root `JalDrishti` folder and execute the following deployment sequence:

# 1. Create and activate a clean isolated Python environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Update core pip utilities
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Install required GIS, Map, Deep Learning, and Web Dashboard packages
.\.venv\Scripts\python.exe -m pip install pandas geopandas shapely networkx matplotlib streamlit streamlit-folium folium torch

---
🖥️ OPERATIONAL EXECUTION
---
Run the pipeline elements step-by-step to verify files and launch the dashboard web panel:

### Step 1: Verify the Kaggle Data Columns
Ensure your raw source dataset file is placed at `data/raw/india_groundwater_2025.csv`, then run the column inspector:
```powershell
.\.venv\Scripts\python.exe notebooks/inspect_kaggle.py
```

### Step 2: Test the Graph Deep Learning Engine
Run the main pipeline to verify spatial tensor configurations and geometric data structures:
```powershell
.\.venv\Scripts\python.exe main.py
```

### Step 3: Launch the Web Intelligence Dashboard
Spin up the local Streamlit application server to open the interactive page automatically in your browser:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Once initialized, your browser will immediately open `http://localhost:8501` to display the map and prediction metrics!

---
💡 DESIGN PATTERNS & DATA-DRIVEN INTERVENTIONS
---
The dashboard translates complex Spatio-Temporal calculations into 3 distinct layers 
of real-world strategic policy actions:
1. Crop Re-alignment: Automatically flags agricultural grids where the GNN forecasts a 
   drop exceeding 15%, issuing community alerts to transition high-stress sectors from 
   water-intensive crops (Paddy/Sugarcane) into robust indigenous millets or pulses.
2. Managed Aquifer Recharge: Isolates high-weight structural node hubs within the spatial 
   network graph, guiding government agencies to build check-dams precisely where they 
   will recharge multiple adjacent water streams simultaneously.
3. Proactive Health Security: Tracks heavy mineral data (Fluoride/Arsenic) to chart 
   plume migration coordinates, prompting health boards to deploy treatment facilities 
   before toxic columns reach rural drinking wells.
========================================================================