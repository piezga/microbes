import pandas as pd 
from pathlib import Path 
import subprocess
import numpy as np 

# Flags
debug = False
desktop = True

# Parameters
datasets = ['root']



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
    
    # Create DataFrame with original row/column names
    prob_df = pd.DataFrame(
        prob_matrix,
        index=original_rows,  # Use original species names
        columns=original_cols  # Use original sample names
    )
    
    # Save as CSV
    prob_csv_path = output_dir / 'probability_dataframe.csv'
    prob_df.to_csv(prob_csv_path)
    
    print(f"Saved probability DataFrame to: {prob_csv_path}")
    print(f"DataFrame shape: {prob_df.shape}")
    print(f"Species (rows): {len(original_rows)}")
    print(f"Samples (columns): {len(original_cols)}")
    
    # Show a preview
    print("\nPreview of probability matrix:")
    print(prob_df.iloc[:5, :5])
