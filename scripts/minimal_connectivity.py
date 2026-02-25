import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import src.my_cm as my_cm

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Parameters
datasets = config['parameters']['datasets']
avoidance = config['parameters']['avoidance']

for dataset_name in datasets:
    print("\n" + "="*70)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*70)
    
    # Data 
    input_dir = Path(f'fitted_data_{dataset_name}/')
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directory
    output_dir = Path(f'iterative_projected_data/{dataset_name}/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------
    # 0. Load data
    # --------------------------------------------------
    csv_path = f"data/binary/{dataset_name}_observed.csv"
    df = pd.read_csv(csv_path, index_col=0)

    # Drop all-zero rows/columns 
    df = df.loc[df.sum(axis=1) > 0, :]
    df = df.loc[:, df.sum(axis=0) > 0]

    # Save original row and column names BEFORE any sorting
    original_rows = df.index.tolist()  # Species names
    original_cols = df.columns.tolist()  # Sample names
    
    print(f'Loading data from {csv_path}')
    # --------------------------------------------------
    # 1. Load p-values
    # --------------------------------------------------
    p_path = input_dir / 'pvalues' / 'row_pvalues.csv' 
    df = pd.read_csv(p_path, header=None)
    if avoidance:
        df = 1 - df
    df = df.clip(1e-10, 1 - 1e-10)

        # Assuming the p-values are in a single column or need to be flattened
    if df.shape[1] == 1:
        pvalues = df.iloc[:, 0].values
    else:
        pvalues = df.values.flatten()
    
    print(f"\nLoaded {len(pvalues)} p-values")
    print(f"P-value range: [{pvalues.min():.6f}, {pvalues.max():.6f}]")
    print(f"First 10 sorted p-values: {np.sort(pvalues)[:10]}")

    # --------------------------------------------------
    # 2. Build connected projected network
    # --------------------------------------------------

    N = my_cm.flat2triumat_dim(len(pvalues))

   # Sort p-values by ascending order, get sorted indices
    sorted_indices = np.argsort(pvalues)

   # Initialize graph with all species as nodes
    G = nx.Graph()
    G.add_nodes_from(range(N))

# Add edges one at a time until graph is connected
    for k in sorted_indices:
        if nx.is_connected(G):
            break
        i, j = my_cm.flat2triumat_idx(k, N)
        G.add_edge(i, j)

    print(f"\nGraph connected: {nx.is_connected(G)}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Relabel nodes with species names
    mapping = {idx: name for idx, name in enumerate(original_rows)}
    G = nx.relabel_nodes(G, mapping)

# --------------------------------------------------
# 3. Save network
# --------------------------------------------------
    net_dir = output_dir / 'network'
    net_dir.mkdir(parents=True, exist_ok=True)

# GraphML preserves node labels and edge weights
    nx.write_graphml(G, net_dir / f'{dataset_name}_connected.graphml')

# Also save adjacency matrix with species labels
    adj = nx.to_pandas_adjacency(G, nodelist=original_rows)
    adj.to_csv(net_dir / f'{dataset_name}_adjacency.csv')

    print(f"\nNetwork saved to {net_dir}")
