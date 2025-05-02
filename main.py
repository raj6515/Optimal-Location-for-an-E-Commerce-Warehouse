import os
import osmnx as ox
import cudf
import cupy as cp
import cugraph
from cuml.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Ensure RAPIDS and OSMnx are configured
ox.settings.use_cache = True
ox.settings.log_console = True

# 1. Load or cache network
def load_network(place_name="Kolkata, India", cache_file="kolkata_network.graphml"):
    if os.path.exists(cache_file):
        print(f"Loading network from {cache_file}...")
        G = ox.load_graphml(cache_file)
    else:
        print(f"Downloading network for {place_name}...")
        G = ox.graph_from_place(place_name)
        ox.save_graphml(G, cache_file)
    G_proj = ox.project_graph(G)
    return G, G_proj

# 2. Load points CSV into GPU DataFrame
def load_points(csv_file):
    print(f"Loading points from {csv_file}...")
    df = pd.read_csv(csv_file)
    for col in ["latitude","longitude","frequency"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in CSV")
    return cudf.DataFrame.from_pandas(df)

# 3. Snap points to nearest OSMnx node using GPU KNN
def snap_to_network(G, pts_gdf):
    print("Snapping points to network nodes on GPU...")
    nodes, data = zip(*G.nodes(data=True))
    coords = cp.asarray([[d['x'],d['y']] for d in data], dtype=cp.float32)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute')
    nbrs.fit(coords)
    pt_coords = pts_gdf[['longitude','latitude']].to_cupy().astype(cp.float32)
    _, idx = nbrs.kneighbors(pt_coords)
    nearest = [nodes[i] for i in cp.asnumpy(idx).flatten()]
    pts_gdf['nearest_node'] = nearest
    return pts_gdf

# 4. Convert OSMnx graph to cuGraph
def to_cugraph(G):
    print("Converting OSMnx graph to cuGraph...")
    src, dst, w = [],[],[]
    for u,v,d in G.edges(data=True):
        src.append(u); dst.append(v)
        w.append(d.get('length',1.0))
    df = cudf.DataFrame({'src':src,'dst':dst,'weight':w})
    G_gpu = cugraph.Graph()
    G_gpu.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
    return G_gpu

# 5. Haversine distance on CPU (vectorized)
def haversine(pt1, pt2):
    # pt in (lat,lon) degrees
    lat1, lon1 = np.deg2rad(pt1)
    lat2, lon2 = np.deg2rad(pt2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return 6371000*c  # meters

# 6. Find global haversine-optimal node
def find_haversine_center(G, pts_gdf):
    print("Finding global center via haversine...")
    # Map freq per node
    freq = pts_gdf.groupby('nearest_node')['frequency'].sum().to_pandas()
    # Node coord lookup
    node_coord = {n:(d['y'],d['x']) for n,d in G.nodes(data=True)}
    best, best_node = float('inf'), None
    for node,(lat,lon) in tqdm(node_coord.items(), desc="Global Haversine pass"):
        total = 0
        for tgt,f in freq.items():
            total += f * haversine((lat,lon), node_coord[tgt])
        if total < best:
            best, best_node = total, node
    return best_node, best

# 7. Local A* (SSSP) over all city nodes but summing only within radius

def find_local_optimal(G, pts_gdf, center_node, radius=5000):
    print(f"Finding local A* optimal relative to points within {radius}m of {center_node}...")
    # Determine target nodes within radius
    center_coord = (G.nodes[center_node]['y'], G.nodes[center_node]['x'])
    target_freq = {}
    # build freq map and filter
    for tgt, group in pts_gdf.groupby('nearest_node'):
        coord = (G.nodes[tgt]['y'], G.nodes[tgt]['x'])
        if haversine(center_coord, coord) <= radius:
            target_freq[tgt] = int(group['frequency'].sum())
    print(f"{len(target_freq)} target nodes within radius")
    # convert graph once
    G_gpu = to_cugraph(G)
    # search over all nodes
    best, best_node = float('inf'), None
    for src in tqdm(G.nodes(), desc="Local A* pass over city", unit="node"):
        paths = cugraph.sssp(G_gpu, src)
        df = paths.to_pandas()
        df = df[df['vertex'].isin(target_freq)]
        df['w'] = df['distance'] * df['vertex'].map(target_freq)
        total = df['w'].sum()
        if total < best:
            best, best_node = total, src
    return best_node, best

# 8. Visualization
def visualize(G, node, pts_gdf):
    fig,ax = plt.subplots(figsize=(10,8))
    ox.plot_graph(G,ax=ax,node_size=0,edge_color='gray',edge_linewidth=0.5)
    df = pts_gdf.to_pandas()
    ax.scatter(df['longitude'],df['latitude'],s=df['frequency']*5,alpha=0.6)
    x,y = G.nodes[node]['x'],G.nodes[node]['y']
    ax.scatter(x,y,marker='*',c='red',s=200)
    plt.show()

# 9. Main pipeline
def main():
    G,_ = load_network()
    pts = load_points("ECOM.csv")
    pts = snap_to_network(G,pts)
    center, _ = find_haversine_center(G,pts)
    optimal, _ = find_local_optimal(G,pts,center, radius=5000)
    print(f"Optimal node: {optimal}")
    visualize(G,optimal,pts)

if __name__ == '__main__':
    main()
