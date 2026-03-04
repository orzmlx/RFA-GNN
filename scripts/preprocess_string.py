
import pandas as pd
import mygene
import os

def map_string_ids():
    input_path = "data/string_directed.csv"
    output_path = "data/string_interactions_mapped.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Get unique ENSP IDs
    unique_ids = pd.unique(df[['source', 'target']].values.ravel('K'))
    print(f"Found {len(unique_ids)} unique Ensembl Protein IDs.")
    
    # Query MyGene
    mg = mygene.MyGeneInfo()
    print("Querying MyGene.info (this may take a moment)...")
    res = mg.querymany(unique_ids, scopes='ensembl.protein', fields='entrezgene,symbol', species='human')
    
    # Create mapping dict
    ensp_to_entrez = {}
    ensp_to_symbol = {}
    
    for item in res:
        if 'query' in item and 'entrezgene' in item:
            ensp_to_entrez[item['query']] = str(item['entrezgene'])
        if 'query' in item and 'symbol' in item:
            ensp_to_symbol[item['query']] = item['symbol']
            
    print(f"Mapped {len(ensp_to_entrez)} IDs to Entrez Gene ID.")
    
    # Map DataFrame
    df['source_entrez'] = df['source'].map(ensp_to_entrez)
    df['target_entrez'] = df['target'].map(ensp_to_entrez)
    df['source_symbol'] = df['source'].map(ensp_to_symbol)
    df['target_symbol'] = df['target'].map(ensp_to_symbol)
    
    # Filter out unmapped
    original_len = len(df)
    df_mapped = df.dropna(subset=['source_entrez', 'target_entrez'])
    print(f"Kept {len(df_mapped)} / {original_len} interactions after mapping.")
    
    # Save
    print(f"Saving to {output_path}...")
    df_mapped.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    map_string_ids()
