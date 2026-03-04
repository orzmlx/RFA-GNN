
import pandas as pd
import json

h5_path = "data/cmap/level3_beta_ctl_n188708x12328_landmark_aligned.h5"
landmark_path = "data/landmark_genes.json"

print(f"Reading {h5_path}...")
df = pd.read_hdf(h5_path, key='data')
print(f"Shape: {df.shape}")
print(f"Index head: {df.index[:5]}")
print(f"Columns head: {df.columns[:5]}")

print("\nReading landmark genes...")
with open(landmark_path, 'r') as f:
    genes = json.load(f)
    target_ids = [str(g['entrez_id']) for g in genes]
    print(f"Target IDs head: {target_ids[:5]}")

# Check intersection
if df.shape[0] == 978:
    print("\nAssuming (Genes, Samples), checking Index vs Target IDs:")
    inter = set(df.index.astype(str)).intersection(target_ids)
    print(f"Intersection size: {len(inter)}")
    if len(inter) == 0:
        print("Index samples:")
        print(df.index[:5])
else:
    print("\nAssuming (Samples, Genes), checking Columns vs Target IDs:")
    inter = set(df.columns.astype(str)).intersection(target_ids)
    print(f"Intersection size: {len(inter)}")
