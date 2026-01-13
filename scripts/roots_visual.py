import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from networkx.algorithms import bipartite
import seaborn as sns
from pathlib import Path
import json

# Flags
drawing = False  # Draw graph with nodes and edges

# Set style
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

# Datasets to process
datasets = ['root', 'soil']


def print_degree_stats(G, nodes, label):
    degrees = np.array([G.degree(n) for n in nodes])

    if len(degrees) == 0:
        print(f"\n{label}: no nodes")
        return

    print(f"\n{label} stats")
    print("-" * (len(label) + 6))
    print(f"Nodes           : {len(degrees)}")
    print(f"Edges (incident): {degrees.sum()}")
    print(f"Min degree      : {degrees.min()}")
    print(f"Max degree      : {degrees.max()}")
    print(f"Mean degree     : {degrees.mean():.2f}")
    print(f"Median degree   : {np.median(degrees):.2f}")
    print(f"Degree std dev  : {degrees.std():.2f}")


def compute_network_stats(G, species_nodes, sample_nodes):
    """Compute various network statistics"""
    stats = {}
    
    # Basic stats
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['num_species'] = len(species_nodes)
    stats['num_samples'] = len(sample_nodes)
    
    # Density
    stats['density'] = nx.density(G)
    
    # Degree stats
    species_degrees = [G.degree(n) for n in species_nodes]
    sample_degrees = [G.degree(n) for n in sample_nodes]
    
    stats['species_degree_mean'] = np.mean(species_degrees)
    stats['species_degree_std'] = np.std(species_degrees)
    stats['sample_degree_mean'] = np.mean(sample_degrees)
    stats['sample_degree_std'] = np.std(sample_degrees)
    
    # Connectivity
    stats['is_connected'] = nx.is_connected(G)
    stats['num_connected_components'] = nx.number_connected_components(G)
    
    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
        stats['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    else:
        # Compute for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        stats['diameter_largest_component'] = nx.diameter(G_largest)
        stats['avg_shortest_path_largest_component'] = nx.average_shortest_path_length(G_largest)
        stats['largest_component_size'] = len(largest_cc)
    
    # Clustering (for bipartite, use projection)
    try:
        species_projection = bipartite.weighted_projected_graph(G, species_nodes)
        stats['species_clustering_coefficient'] = nx.average_clustering(species_projection)
    except:
        stats['species_clustering_coefficient'] = None
    
    try:
        sample_projection = bipartite.weighted_projected_graph(G, sample_nodes)
        stats['sample_clustering_coefficient'] = nx.average_clustering(sample_projection)
    except:
        stats['sample_clustering_coefficient'] = None
    
    # Centrality (sample top nodes)
    degree_centrality = nx.degree_centrality(G)
    stats['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
    
    betweenness = nx.betweenness_centrality(G)
    stats['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
    
    # Top 5 nodes by degree
    top_species = sorted(species_nodes, key=lambda x: G.degree(x), reverse=True)[:5]
    top_samples = sorted(sample_nodes, key=lambda x: G.degree(x), reverse=True)[:5]
    
    stats['top_5_species'] = [(s, G.degree(s)) for s in top_species]
    stats['top_5_samples'] = [(s, G.degree(s)) for s in top_samples]
    
    return stats


# --------------------------------------------------
# Main processing loop
# --------------------------------------------------
for dataset_name in datasets:
    print("\n" + "="*70)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*70)
    
    # Create output directory for this dataset
    output_dir = Path(f"exploration/{dataset_name}_binary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------
    # 1. Load incidence matrix
    # --------------------------------------------------
    csv_path = f"data/binary/{dataset_name}_observed.csv"
    df = pd.read_csv(csv_path, index_col=0)

    # Drop all-zero rows/columns 
    df = df.loc[df.sum(axis=1) > 0, :]
    df = df.loc[:, df.sum(axis=0) > 0]

    # --------------------------------------------------
    # Order rows/columns by degree
    # --------------------------------------------------
    sample_degree = df.sum(axis=0)
    df = df.loc[:, sample_degree.sort_values(ascending=False).index]

    species_degree = df.sum(axis=1)
    df = df.loc[species_degree.sort_values(ascending=False).index, :]

    # --------------------------------------------------
    # 2. Build bipartite graph
    # --------------------------------------------------
    B = nx.Graph()

    species_nodes = list(df.index)
    sample_nodes = list(df.columns)

    B.add_nodes_from(species_nodes, bipartite=0, node_type="species")
    B.add_nodes_from(sample_nodes, bipartite=1, node_type="sample")

    edges = df.stack()
    edges = edges[edges != 0]

    B.add_edges_from(
        (species, sample)
        for (species, sample), value in edges.items()
    )

    print(f"Nodes: {B.number_of_nodes()}")
    print(f"Edges: {B.number_of_edges()}")

    # --------------------------------------------------
    # 3. Compute and save network statistics
    # --------------------------------------------------
    print("\n" + "="*60)
    print("COMPUTING NETWORK STATISTICS")
    print("="*60)

    network_stats = compute_network_stats(B, species_nodes, sample_nodes)

    # Print stats
    print("\n--- Global Network Statistics ---")
    print(f"Total nodes: {network_stats['num_nodes']}")
    print(f"Total edges: {network_stats['num_edges']}")
    print(f"Species: {network_stats['num_species']}")
    print(f"Samples: {network_stats['num_samples']}")
    print(f"Density: {network_stats['density']:.4f}")
    print(f"Connected: {network_stats['is_connected']}")
    print(f"Connected components: {network_stats['num_connected_components']}")

    if 'diameter' in network_stats:
        print(f"Diameter: {network_stats['diameter']}")
        print(f"Avg shortest path: {network_stats['average_shortest_path_length']:.2f}")
    else:
        print(f"Diameter (largest component): {network_stats.get('diameter_largest_component', 'N/A')}")
        print(f"Avg shortest path (largest): {network_stats.get('avg_shortest_path_largest_component', 'N/A'):.2f}")

    print(f"\nAvg degree centrality: {network_stats['avg_degree_centrality']:.4f}")
    print(f"Avg betweenness centrality: {network_stats['avg_betweenness_centrality']:.4f}")

    if network_stats['species_clustering_coefficient'] is not None:
        print(f"Species clustering coefficient: {network_stats['species_clustering_coefficient']:.4f}")
    if network_stats['sample_clustering_coefficient'] is not None:
        print(f"Sample clustering coefficient: {network_stats['sample_clustering_coefficient']:.4f}")

    print("\n--- Top 5 Species by Degree ---")
    for species, degree in network_stats['top_5_species']:
        print(f"{species}: {degree}")

    print("\n--- Top 5 Samples by Degree ---")
    for sample, degree in network_stats['top_5_samples']:
        print(f"{sample}: {degree}")

    # Save stats to JSON
    stats_file = output_dir / "network_statistics.json"
    # Convert tuples to lists for JSON serialization
    json_stats = network_stats.copy()
    json_stats['top_5_species'] = [list(x) for x in json_stats['top_5_species']]
    json_stats['top_5_samples'] = [list(x) for x in json_stats['top_5_samples']]

    with open(stats_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    # --------------------------------------------------
    # 4. Degree distributions (Enhanced)
    # --------------------------------------------------
    species_degrees = np.array([B.degree(n) for n in species_nodes])
    sample_degrees = np.array([B.degree(n) for n in sample_nodes])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.capitalize()} - Species × Sample Network: Degree Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)

    # Species degree histogram
    axes[0, 0].hist(species_degrees, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0, 0].axvline(species_degrees.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {species_degrees.mean():.1f}')
    axes[0, 0].axvline(np.median(species_degrees), color='#3498db', linestyle='--', linewidth=2, label=f'Median: {np.median(species_degrees):.1f}')
    axes[0, 0].set_title('Species Degree Distribution', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Degree (number of samples)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend(fontsize=10, framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Sample degree histogram
    axes[0, 1].hist(sample_degrees, bins=30, color='#e67e22', alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0, 1].axvline(sample_degrees.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {sample_degrees.mean():.1f}')
    axes[0, 1].axvline(np.median(sample_degrees), color='#3498db', linestyle='--', linewidth=2, label=f'Median: {np.median(sample_degrees):.1f}')
    axes[0, 1].set_title('Sample Degree Distribution', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Degree (number of species)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend(fontsize=10, framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # Log-log degree distribution (Species)
    species_degree_counts = np.bincount(species_degrees)
    species_degrees_unique = np.arange(len(species_degree_counts))
    species_mask = species_degree_counts > 0
    axes[1, 0].loglog(species_degrees_unique[species_mask], species_degree_counts[species_mask], 
                       'o', markersize=8, color='#2ecc71', alpha=0.7, markeredgecolor='black', markeredgewidth=1)
    axes[1, 0].set_title('Species Degree Distribution (Log-Log)', fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Degree (log scale)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', which='both')

    # Log-log degree distribution (Samples)
    sample_degree_counts = np.bincount(sample_degrees)
    sample_degrees_unique = np.arange(len(sample_degree_counts))
    sample_mask = sample_degree_counts > 0
    axes[1, 1].loglog(sample_degrees_unique[sample_mask], sample_degree_counts[sample_mask], 
                       'o', markersize=8, color='#e67e22', alpha=0.7, markeredgecolor='black', markeredgewidth=1)
    axes[1, 1].set_title('Sample Degree Distribution (Log-Log)', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Degree (log scale)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', which='both')

    plt.tight_layout()
    plt.savefig(output_dir / "degree_distributions.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'degree_distributions.png'}")
    plt.show()
    plt.close()

    # --------------------------------------------------
    # 5. Incidence Matrix Heatmap (only for root)
    # --------------------------------------------------
    if dataset_name == 'root':
        fig, ax = plt.subplots(figsize=(14, max(8, 0.2 * df.shape[0])))

        sns.heatmap(
            df,
            cmap=sns.color_palette("YlOrRd", as_cmap=True),
            cbar_kws={'label': 'Presence', 'shrink': 0.8},
            linewidths=0.1,
            linecolor='white',
            square=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax
        )

        ax.set_title(f'{dataset_name.capitalize()} - Species × Sample Incidence Matrix\n(Ordered by degree)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f'Samples (n={len(sample_nodes)})', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Species (n={len(species_nodes)})', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "incidence_matrix.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'incidence_matrix.png'}")
        plt.show()
        plt.close()

    # --------------------------------------------------
    # 6. Degree rank plots
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{dataset_name.capitalize()} - Degree Rank Plots', fontsize=18, fontweight='bold')

    # Species
    species_sorted = np.sort(species_degrees)[::-1]
    axes[0].plot(range(1, len(species_sorted) + 1), species_sorted, 
                 linewidth=2.5, color='#2ecc71', marker='o', markersize=4, alpha=0.7)
    axes[0].set_title('Species Degree Rank', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Rank', fontsize=12)
    axes[0].set_ylabel('Degree', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_yscale('log')

    # Samples
    sample_sorted = np.sort(sample_degrees)[::-1]
    axes[1].plot(range(1, len(sample_sorted) + 1), sample_sorted, 
                 linewidth=2.5, color='#e67e22', marker='o', markersize=4, alpha=0.7)
    axes[1].set_title('Sample Degree Rank', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Rank', fontsize=12)
    axes[1].set_ylabel('Degree', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "degree_rank_plots.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'degree_rank_plots.png'}")
    plt.show()
    plt.close()

    # --------------------------------------------------
    # 7. Print degree stats
    # --------------------------------------------------
    print_degree_stats(B, species_nodes, "Species")
    print_degree_stats(B, sample_nodes, "Samples")

    print(f"\n{'='*60}")
    print(f"All outputs for {dataset_name} saved to: {output_dir.absolute()}")
    print(f"{'='*60}")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print("\nAll datasets processed successfully!")
print(f"Output directories:")
for dataset_name in datasets:
    print(f"  - exploration/{dataset_name}_binary/")
