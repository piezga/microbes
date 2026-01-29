import sys
import os
import numpy as np
import time
from bicm import BiCM

#################################
### WARNING: THIS IS PYTHON 2 ###
#################################

if len(sys.argv) > 2:
    mat_path = sys.argv[1]
    output_path = sys.argv[2]
    mat = np.load(mat_path)
    print "Loaded matrix from {0}".format(mat_path)
    print "Matrix shape: {0}".format(mat.shape)
    print "Output path: {0}".format(output_path)
else:
    print "Error: No matrix file provided"
    sys.exit(1)

print "\n" + "="*70
print "FITTING BICM MODEL"
print "="*70
start_fit = time.time()

cm = BiCM(bin_mat=mat)
cm.make_bicm()

fit_time = time.time() - start_fit
print "Model fitting completed in {0:.2f} seconds".format(fit_time)

# Save adjacency matrix
output_dir = output_path
adj_npy = os.path.join(output_dir, 'adj_matrix.npy')
adj_csv = os.path.join(output_dir, 'adj_matrix.tsv')

np.save(adj_npy, cm.adj_matrix)
np.savetxt(adj_csv, cm.adj_matrix, delimiter='\t', fmt='%.6f')
adj_matrix_path = os.path.join(output_dir, 'adj_matrix.npy')

# Check dimensions
print "\nMatrix dimensions:"
print "  - adj_matrix shape:", cm.adj_matrix.shape if hasattr(cm, 'adj_matrix') else "N/A"
print "  - Original mat shape:", mat.shape

# --------------------------------------------------
# Calculate p-values for ROWS ONLY (species)
# --------------------------------------------------
print "\n" + "="*70
print "CALCULATING P-VALUES FOR ROW NODES (SPECIES)"
print "="*70

# Create p-values directory
pvalues_dir = os.path.join(output_dir, 'pvalues')
if not os.path.exists(pvalues_dir):
    os.makedirs(pvalues_dir)

# Save human-readable CSV for row p-values
row_csv_filename = os.path.join(pvalues_dir, 'row_pvalues.csv')

print "\nStarting p-value calculation for rows..."
print "This may take a while for large matrices..."

start_pvalues = time.time()

try:
    # Calculate and save row p-values as CSV
    cm.lambda_motifs(True, filename=row_csv_filename, delim='\t', binary=False)
    
    pvalues_time = time.time() - start_pvalues
    print "\n Row p-values calculation completed in {0:.2f} seconds".format(pvalues_time)
    print " P-values saved to: {0}".format(row_csv_filename)
    
    # Show file size
    if os.path.exists(row_csv_filename):
        file_size = os.path.getsize(row_csv_filename)
        print "File size: {0:.2f} MB".format(file_size / (1024.0 * 1024.0))
    
except Exception as e:
    pvalues_time = time.time() - start_pvalues
    print "\n Error calculating row p-values: {0}".format(e)
    print " Calculation ran for {0:.2f} seconds before failing".format(pvalues_time)

# --------------------------------------------------
# Summary
# --------------------------------------------------
total_time = time.time() - start_fit
print "\n" + "="*70
print "SUMMARY"
print "="*70
print "Total processing time: {0:.2f} seconds".format(total_time)
print "  - Model fitting: {0:.2f} seconds".format(fit_time)
print "  - P-value calculation: {0:.2f} seconds".format(pvalues_time)
print "\nOutput files saved in: {0}".format(output_dir)
print "  - Adjacency matrix: {0}".format(adj_matrix_path)
print "  - Row p-values (CSV): {0}".format(row_csv_filename)
print "PROCESSING COMPLETE"
