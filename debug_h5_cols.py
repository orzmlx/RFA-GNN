
import pandas as pd
import json

h5_path = "data/cmap/level3_beta_ctl_n188708x12328_landmark_aligned.h5"
landmark_path = "data/landmark_genes.json"

print(f"Reading {h5_path}...")
try:
    df = pd.read_hdf(h5_path, key='data')
    print(f"Shape: {df.shape}")
    print(f"Index head: {df.index[:5]}")
    print(f"Columns head: {df.columns[:5]}")
    print(f"Columns tail: {df.columns[-5:]}")
except Exception as e:
    print(e)

print("\nReading landmark genes...")
with open(landmark_path, 'r') as f:
    genes = json.load(f)
    target_ids = [str(g['entrez_id']) for g in genes]
    print(f"Target IDs count: {len(target_ids)}")
    print(f"Target IDs head: {target_ids[:5]}")
