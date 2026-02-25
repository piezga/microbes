import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
import yaml

# --------------------------------------------------
# Config
# --------------------------------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

datasets = config['parameters']['datasets']
avoidance = config['parameters']['avoidance']

INTERACTIVE = True  # Set to False to suppress plt.show()

# --------------------------------------------------
# Helper
# --------------------------------------------------
def maybe_show():
    if INTERACTIVE:
        plt.show()
    else:
        plt.close()

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for dataset_name in datasets:
    print("\n" + "="*70)
    print(f"ANALYSING NETWORK: {dataset_name.upper()}")
    print("="*70)

    # Paths
    net_dir = Path(f'iterative_projected_data/{dataset_name}/network/')
    viz_dir = Path(f'iterative_projected_data/{dataset_name}/visualization/')
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load
    G = nx.read_graphml(net_dir / f'{dataset_name}_connected.graphml')
    N = G.number_of_nodes()
    E = G.number_of_edges()

    # --------------------------------------------------
    # 1. Global properties
    # --------------------------------------------------
    density         = nx.density(G)
    avg_clustering  = nx.average_clustering(G)
    transitivity    = nx.transitivity(G)
    avg_path_length = nx.average_shortest_path_length(G)
    diameter        = nx.diameter(G)
    assortativity   = nx.degree_assortativity_coefficient(G)

    print(f"\n--- Global properties ---")
    print(f"  Nodes            : {N}")
    print(f"  Edges            : {E}")
    print(f"  Density          : {density:.4f}")
    print(f"  Avg clustering   : {avg_clustering:.4f}")
    print(f"  Transitivity     : {transitivity:.4f}")
    print(f"  Avg path length  : {avg_path_length:.4f}")
    print(f"  Diameter         : {diameter}")
    print(f"  Assortativity    : {assortativity:.4f}")

    # --------------------------------------------------
    # 2. Node-level metrics
    # --------------------------------------------------
    degrees      = dict(G.degree())
    betweenness  = nx.betweenness_centrality(G)
    closeness    = nx.closeness_centrality(G)
    eigenvector  = nx.eigenvector_centrality(G, max_iter=1000)
    clustering   = nx.clustering(G)

    node_df = pd.DataFrame({
        'degree'      : degrees,
        'betweenness' : betweenness,
        'closeness'   : closeness,
        'eigenvector' : eigenvector,
        'clustering'  : clustering,
    }).sort_values('degree', ascending=False)

    print(f"\n--- Top 10 nodes by degree ---")
    print(node_df.head(10).to_string())

    node_df.to_csv(viz_dir / f'{dataset_name}_node_metrics.csv')

    # --------------------------------------------------
    # 3. Degree distribution
    # --------------------------------------------------
    degree_sequence = sorted(degrees.values(), reverse=True)
    degree_counts   = Counter(degree_sequence)
    deg_vals        = np.array(sorted(degree_counts.keys()))
    deg_freq        = np.array([degree_counts[d] for d in deg_vals])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{dataset_name} — Degree Distribution', fontsize=13)

    axes[0].bar(deg_vals, deg_freq, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Linear scale')

    axes[1].scatter(deg_vals, deg_freq, color='steelblue', s=30, zorder=3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Log-log scale')

    plt.tight_layout()
    plt.savefig(viz_dir / f'{dataset_name}_degree_distribution.png', dpi=150)
    maybe_show()

    # --------------------------------------------------
    # 4. Centrality distributions
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'{dataset_name} — Centrality Distributions', fontsize=13)

    for ax, metric, label in zip(axes,
                                  [betweenness, closeness, eigenvector],
                                  ['Betweenness', 'Closeness', 'Eigenvector']):
        vals = list(metric.values())
        ax.hist(vals, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(viz_dir / f'{dataset_name}_centrality_distributions.png', dpi=150)
    maybe_show()

    # --------------------------------------------------
    # 5. Centrality correlation heatmap
    # --------------------------------------------------
    corr = node_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    ax.set_title(f'{dataset_name} — Metric Correlations')
    plt.tight_layout()
    plt.savefig(viz_dir / f'{dataset_name}_metric_correlations.png', dpi=150)
    maybe_show()

    # --------------------------------------------------
    # 6. Connected components & core-periphery
    # --------------------------------------------------
    components = list(nx.connected_components(G))
    print(f"\n--- Connected components: {len(components)} ---")

    # k-core decomposition
    core_numbers = nx.core_number(G)
    max_core     = max(core_numbers.values())
    core_sizes   = Counter(core_numbers.values())
    print(f"  Max k-core       : {max_core}")
    print(f"  Core size dist   : {dict(sorted(core_sizes.items()))}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(core_sizes.keys(), core_sizes.values(), color='steelblue', edgecolor='white')
    ax.set_xlabel('Core number (k)')
    ax.set_ylabel('Number of nodes')
    ax.set_title(f'{dataset_name} — k-core decomposition')
    plt.tight_layout()
    plt.savefig(viz_dir / f'{dataset_name}_kcore.png', dpi=150)
    maybe_show()

    # --------------------------------------------------
    # 7. Network visualisation
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(N))

    node_sizes  = [300 * degrees[n] / max(degrees.values()) + 20 for n in G.nodes()]
    node_colors = [core_numbers[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5, edge_color='gray')
    nc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap=plt.cm.plasma,
                                 alpha=0.9)
    plt.colorbar(nc, ax=ax, label='k-core number')

    # Label only top-degree nodes to avoid clutter
    top_nodes = node_df.head(10).index.tolist()
    labels    = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(f'{dataset_name} — Network (size=degree, colour=k-core)', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(viz_dir / f'{dataset_name}_network.png', dpi=150)
    maybe_show()

    print(f"\nAll outputs saved to {viz_dir}")
