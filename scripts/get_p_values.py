import pandas as pd 
from pathlib import Path 
import subprocess
import numpy as np 
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Flags
debug = config['general']['debug']
desktop = config['general']['desktop']

print(f'Desktop is set to {desktop}')

# Parameters
datasets = config['parameters']['datasets']
N_top = config['parameters']['N_top'] # top species to drop
N_bottom = config['parameters']['N_bottom'] # bottom species to drop

print(f'Dropping {N_top} most abundant species')
print(f'Dropping {N_bottom} most rare species')

if desktop:
    virt_env = 'conda'
else:
    virt_env = 'mamba'



for dataset_name in datasets:
    print("\n" + "="*70)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("="*70)
    
    # Output directory
    output_dir = Path(f'fitted_data_{dataset_name}/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------
    # 1. Load incidence matrix
    # --------------------------------------------------
    csv_path = f"data/binary/{dataset_name}_observed.csv"
    df = pd.read_csv(csv_path, index_col=0)

    # Drop all-zero rows/columns 
    df = df.loc[df.sum(axis=1) > 0, :]
    df = df.loc[:, df.sum(axis=0) > 0]

    # Save original row and column names BEFORE any sorting
    original_rows = df.index.tolist()  # Species names
    original_cols = df.columns.tolist()  # Sample names
      
    # Drop N most abundant species 
    row_sums = df.sum(axis=1)

    rows_top_drop = row_sums.nlargest(N_top).index
    rows_bottom_drop = row_sums.nsmallest(N_bottom).index
    all_rows_to_drop = rows_top_drop.union(rows_bottom_drop)
    # Create list of rows to KEEP
    rows_to_keep = [row for row in original_rows if row not in all_rows_to_drop]

    df = df.drop(index=all_rows_to_drop)


    if debug:
        print(df)

    # --------------------------------------------------
    # Turn into 2d numpy array
    # --------------------------------------------------
    mat = df.to_numpy().astype(int)
    mat_path = temp_dir / 'mat.npy'
    np.save(mat_path, mat)

    # --------------------------------------------------
    # 2. Fit that mat 
    # --------------------------------------------------
    subprocess.run([
        f"{virt_env}", "run", "-n", "bicm", "python", "src/fit_bicm.py", str(mat_path), str(output_dir)
    ])

    # --------------------------------------------------
    # 3. Load probability matrix and convert to DataFrame
    # --------------------------------------------------
    print("\n" + "="*70)
    print("CONVERTING TO DATAFRAME")
    print("="*70)
    
    # Load the probability matrix
    prob_matrix_path = output_dir / 'adj_matrix.tsv'
    prob_matrix = np.loadtxt(prob_matrix_path)

    # Get indices of rows to keep in the original matrix
    keep_indices = [original_rows.index(row) for row in rows_to_keep]
    # Filter probability matrix to only include kept rows
    filtered_prob_matrix = prob_matrix

    # Create DataFrame with correct (filtered) rows
    prob_df = pd.DataFrame(
        filtered_prob_matrix,
        index=rows_to_keep,  # Use filtered species names
        columns=original_cols  # All samples (no column dropping)
    )
   
    # Save as CSV
    prob_csv_path = output_dir / 'probability_dataframe.csv'
    prob_df.to_csv(prob_csv_path)
    
    print(f"Saved probability DataFrame to: {prob_csv_path}")
    print(f"DataFrame shape: {prob_df.shape}")

    # Show a preview
    print("\nPreview of probability matrix:")
    print(prob_df.iloc[:5, :5])
