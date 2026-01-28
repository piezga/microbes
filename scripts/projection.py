import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Helper functions for array-to-matrix conversion
def flat2triumat_dim(L):
    """
    Calculate the dimension N of a square matrix from the length L 
    of its upper triangular part (excluding diagonal).
    
    L = N * (N - 1) / 2
    Solving for N: N = (1 + sqrt(1 + 8*L)) / 2
    """
    N = int((1 + np.sqrt(1 + 8 * L)) / 2)
    return N

def flat2triumat_idx(k, N):
    """
    Convert flat array index k to upper triangular matrix indices (i, j).
    
    Parameters:
    -----------
    k : int
        Index in the flattened array
    N : int
        Dimension of the square matrix
    
    Returns:
    --------
    (i, j) : tuple
        Row and column indices in the upper triangular matrix
    """
    # Calculate row index i
    i = int(N - 2 - np.floor(np.sqrt(4 * N * (N - 1) - 7 - 8 * k) / 2.0 - 0.5))
    # Calculate column index j
    j = int(k + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2)
    return (i, j)

def fdr_procedure(pvalues, t=0.05):
    """
    Apply the False Discovery Rate (FDR) procedure.
    
    Parameters:
    -----------
    pvalues : array-like
        Array of p-values to test
    t : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    dict with:
        - 'i_hat': largest index satisfying FDR condition
        - 'p_threshold': p-value threshold
        - 'rejected_indices': indices of rejected hypotheses
        - 'num_rejected': number of rejected hypotheses
        - 'num_accepted': number of accepted hypotheses
    """
    M = len(pvalues)
    
    # Sort p-values in increasing order and keep track of original indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    
    # Find largest i satisfying: p_value[i] <= (i * t) / M
    # IMPORTANT: We must check ALL indices and keep the LARGEST one that satisfies
    i_hat = -1
    for i in range(M):
        threshold = ((i + 1) * t) / M  # i+1 because Python is 0-indexed
        if sorted_pvalues[i] <= threshold:
            i_hat = i  # Update i_hat, don't break!
        # Note: We continue checking all values to find the LARGEST i
    
    # If i_hat was found, reject all hypotheses up to and including i_hat
    if i_hat >= 0:
        p_threshold = sorted_pvalues[i_hat]
        rejected_indices = sorted_indices[:i_hat + 1]
        num_rejected = i_hat + 1
    else:
        p_threshold = 0
        rejected_indices = np.array([])
        num_rejected = 0
    
    return {
        'i_hat': i_hat,
        'p_threshold': p_threshold,
        'rejected_indices': rejected_indices,
        'num_rejected': num_rejected,
        'num_accepted': M - num_rejected,
        'total_tests': M
    }

# Parameters
desktop = True  # Set to False if not on desktop
datasets = ['soil']
t = 0.05

if desktop:
    virt_env = 'conda'
else:
    virt_env = 'mamba'

for dataset_name in datasets:
    print("\n" + "="*70)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*70)
    
    # Data 
    input_dir = Path(f'fitted_data_{dataset_name}/')
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directory
    output_dir = Path(f'projected_data_{dataset_name}/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------
    # 1. Load p-values
    # --------------------------------------------------
    p_path = input_dir / 'pvalues' / 'row_pvalues.csv' 
    df = pd.read_csv(p_path, header=None)
    
    # Assuming the p-values are in a single column or need to be flattened
    if df.shape[1] == 1:
        pvalues = df.iloc[:, 0].values
    else:
        pvalues = df.values.flatten()
    
    print(f"\nLoaded {len(pvalues)} p-values")
    print(f"P-value range: [{pvalues.min():.6f}, {pvalues.max():.6f}]")
    print(f"Number of p-values < {t}: {np.sum(pvalues < t)}")
    print(f"First 10 sorted p-values: {np.sort(pvalues)[:10]}")

    # Histogram of p-values
    plt.figure(figsize=(8, 5))
    plt.hist(pvalues, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=t, color='red', linestyle='--', label=f'Significance level (t={t})')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.title('Histogram of P-values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # --------------------------------------------------
    # 2. Validate links using FDR procedure
    # --------------------------------------------------
    print(f"\nApplying FDR procedure with significance level t={t}")
    
    fdr_results = fdr_procedure(pvalues, t=t)
    
    # Debug: check what's happening
    M = len(pvalues)
    sorted_pvalues = np.sort(pvalues)
    print(f"\nDEBUG INFO:")
    print(f"First threshold (i=0): {(1 * t) / M:.10f}")
    print(f"Smallest p-value: {sorted_pvalues[0]:.10f}")
    print(f"Last threshold (i={M-1}): {(M * t) / M:.10f}")
    print(f"Largest p-value: {sorted_pvalues[-1]:.10f}")
    
    # Check where p-values might cross threshold
    # Find first i where threshold might exceed p-value
    target_p = sorted_pvalues[0]  # smallest p-value
    target_i = int((target_p * M) / t)
    print(f"\nTo validate smallest p-value {target_p:.6f}, need i >= {target_i}")
    
    # Check a few strategic points
    check_indices = [0, 10, 100, 1000, 10000, min(100000, M-1), M-1]
    print("\nChecking strategic indices:")
    for i in check_indices:
        if i < M:
            threshold = ((i + 1) * t) / M
            print(f"  i={i}: p={sorted_pvalues[i]:.8f}, threshold={threshold:.8f}, passes={sorted_pvalues[i] <= threshold}")
    
    # Try to find where it should pass
    print("\nSearching for crossover point...")
    crossover_found = False
    for i in range(min(50000, M)):
        threshold = ((i + 1) * t) / M
        if sorted_pvalues[i] <= threshold:
            print(f"  Found! i={i}: p={sorted_pvalues[i]:.8f}, threshold={threshold:.8f}")
            crossover_found = True
            break
    
    if not crossover_found:
        print(f"  No crossover found in first {min(50000, M)} indices")
        # Check around the predicted crossover point
        target_i = int((sorted_pvalues[0] * M) / t)
        print(f"\n  Checking around predicted crossover i={target_i}:")
        for offset in [-10, -5, -1, 0, 1, 5, 10]:
            i = target_i + offset
            if 0 <= i < M:
                threshold = ((i + 1) * t) / M
                print(f"    i={i}: p={sorted_pvalues[i]:.8f}, threshold={threshold:.8f}, passes={sorted_pvalues[i] <= threshold}")
    
    # --------------------------------------------------
    # Visualize FDR procedure
    # --------------------------------------------------
    print("\nGenerating FDR visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.capitalize()} - FDR Procedure Visualization (t={t})', 
                 fontsize=18, fontweight='bold')
    
    # Calculate threshold line
    indices = np.arange(M)
    threshold_line = ((indices + 1) * t) / M
    
    # Plot 1: Full view
    axes[0, 0].plot(indices, sorted_pvalues, 'b.', markersize=1, alpha=0.3, label='P-values')
    axes[0, 0].plot(indices, threshold_line, 'r-', linewidth=2, label=f'Threshold (t={t})')
    axes[0, 0].set_xlabel('Rank (i)', fontsize=12)
    axes[0, 0].set_ylabel('P-value', fontsize=12)
    axes[0, 0].set_title('Full View: P-values vs FDR Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed to first 10000
    zoom_n = min(10000, M)
    axes[0, 1].plot(indices[:zoom_n], sorted_pvalues[:zoom_n], 'b.', markersize=2, alpha=0.5, label='P-values')
    axes[0, 1].plot(indices[:zoom_n], threshold_line[:zoom_n], 'r-', linewidth=2, label=f'Threshold (t={t})')
    axes[0, 1].set_xlabel('Rank (i)', fontsize=12)
    axes[0, 1].set_ylabel('P-value', fontsize=12)
    axes[0, 1].set_title(f'Zoomed View: First {zoom_n} ranks', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Log-scale view
    axes[1, 0].plot(indices, sorted_pvalues, 'b.', markersize=1, alpha=0.3, label='P-values')
    axes[1, 0].plot(indices, threshold_line, 'r-', linewidth=2, label=f'Threshold (t={t})')
    axes[1, 0].set_xlabel('Rank (i)', fontsize=12)
    axes[1, 0].set_ylabel('P-value (log scale)', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Log Scale: P-values vs FDR Threshold', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Plot 4: Very early ranks (first 2000)
    early_n = min(2000, M)
    axes[1, 1].plot(indices[:early_n], sorted_pvalues[:early_n], 'b.', markersize=3, alpha=0.6, label='P-values')
    axes[1, 1].plot(indices[:early_n], threshold_line[:early_n], 'r-', linewidth=2, label=f'Threshold (t={t})')
    axes[1, 1].set_xlabel('Rank (i)', fontsize=12)
    axes[1, 1].set_ylabel('P-value', fontsize=12)
    axes[1, 1].set_title(f'Early Ranks: First {early_n}', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fdr_plot_path = output_dir / 'fdr_visualization.png'
    plt.savefig(fdr_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved FDR visualization to: {fdr_plot_path}")
    plt.show()
    plt.close()
    
    print("\n" + "-"*70)
    print("FDR RESULTS")
    print("-"*70)
    print(f"Total hypotheses tested: {fdr_results['total_tests']}")
    print(f"Largest index i_hat: {fdr_results['i_hat']}")
    print(f"P-value threshold: {fdr_results['p_threshold']:.6f}")
    print(f"Number of rejected hypotheses (validated links): {fdr_results['num_rejected']}")
    print(f"Number of accepted hypotheses (non-significant): {fdr_results['num_accepted']}")
    print(f"Proportion of validated links: {fdr_results['num_rejected']/fdr_results['total_tests']:.2%}")
    
    # --------------------------------------------------
    # 3. Convert to matrix format
    # --------------------------------------------------
    L = len(pvalues)
    N = flat2triumat_dim(L)
    
    print(f"\nMatrix conversion:")
    print(f"Array length L: {L}")
    print(f"Matrix dimension N: {N}")
    print(f"Expected array length: {N * (N - 1) // 2}")
    
    # Create adjacency matrix for validated links
    validated_matrix = np.zeros((N, N), dtype=int)
    pvalue_matrix = np.zeros((N, N))
    pvalue_matrix[:] = np.nan
    
    # Fill the matrices
    for k in range(L):
        i, j = flat2triumat_idx(k, N)
        pvalue_matrix[i, j] = pvalues[k]
        
        # Mark as validated if this hypothesis was rejected (p-value significant)
        if k in fdr_results['rejected_indices']:
            validated_matrix[i, j] = 1
            validated_matrix[j, i] = 1  # Make symmetric
    
    print(f"\nValidated links matrix shape: {validated_matrix.shape}")
    print(f"Total validated links (edges): {validated_matrix.sum() // 2}")
    
    # --------------------------------------------------
    # 4. Save results
    # --------------------------------------------------
    # Save validated links matrix
    validated_df = pd.DataFrame(validated_matrix)
    validated_path = output_dir / 'validated_links.csv'
    validated_df.to_csv(validated_path)
    print(f"\nSaved validated links matrix to: {validated_path}")
    
    # Save p-value matrix
    pvalue_matrix_df = pd.DataFrame(pvalue_matrix)
    pvalue_matrix_path = output_dir / 'pvalue_matrix.csv'
    pvalue_matrix_df.to_csv(pvalue_matrix_path)
    print(f"Saved p-value matrix to: {pvalue_matrix_path}")
    
    # Save FDR results summary
    fdr_summary = pd.DataFrame({
        'metric': ['total_tests', 'significance_level', 'i_hat', 'p_threshold', 
                   'num_rejected', 'num_accepted', 'proportion_rejected'],
        'value': [fdr_results['total_tests'], t, fdr_results['i_hat'], 
                  fdr_results['p_threshold'], fdr_results['num_rejected'],
                  fdr_results['num_accepted'], 
                  fdr_results['num_rejected']/fdr_results['total_tests']]
    })
    fdr_summary_path = output_dir / 'fdr_summary.csv'
    fdr_summary.to_csv(fdr_summary_path, index=False)
    print(f"Saved FDR summary to: {fdr_summary_path}")
    
    # Save list of validated link indices
    validated_links_list = []
    for k in fdr_results['rejected_indices']:
        i, j = flat2triumat_idx(k, N)
        validated_links_list.append({
            'flat_index': k,
            'node_i': i,
            'node_j': j,
            'p_value': pvalues[k]
        })
    
    validated_links_df = pd.DataFrame(validated_links_list)
    validated_links_path = output_dir / 'validated_links_list.csv'
    validated_links_df.to_csv(validated_links_path, index=False)
    print(f"Saved validated links list to: {validated_links_path}")
    
    print("\n" + "="*70)
    print(f"PROCESSING COMPLETE FOR {dataset_name.upper()}")
    print("="*70)
