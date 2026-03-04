import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from drug_encoder import DrugEncoder

def generate_fingerprints(compound_path, output_path):
    print(f"正在读取 {compound_path} ...")
    if not os.path.exists(compound_path):
        print("错误: 找不到 compoundinfo 文件")
        return

    # 读取 pert_id 和 canonical_smiles
    try:
        df = pd.read_csv(compound_path, sep="\t", usecols=["pert_id", "canonical_smiles"], engine='python')
    except ValueError as e:
        print(f"尝试读取列名失败: {e}")
        return

    # 去重
    df_unique = df.drop_duplicates(subset=["pert_id"])
    df_unique = df_unique.dropna(subset=["canonical_smiles"])
    
    print(f"共找到 {len(df_unique)} 个唯一的药物 ID (含 SMILES)。开始生成 Morgan 指纹...")
    
    encoder = DrugEncoder()
    fingerprints = []
    pert_ids = []
    
    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique)):
        smiles = row["canonical_smiles"]
        pid = row["pert_id"]
        fp = encoder.encode(smiles)
        fingerprints.append(fp)
        pert_ids.append(pid)
        
    cols = [f"fp{i}" for i in range(2048)]
    df_fp = pd.DataFrame(fingerprints, index=pert_ids, columns=cols)
    
    print(f"Morgan 指纹生成完毕。Shape: {df_fp.shape}")
    print(f"保存至 {output_path} ...")
    df_fp.to_csv(output_path)
    print("完成！")


if __name__ == "__main__":
    # 1. 生成 Morgan Fingerprints
    compound_path = "data/compoundinfo_beta.txt"
    morgan_output = "data/new_morgan_fingerprints.csv"
    generate_fingerprints(compound_path, morgan_output)
    
    # 2. 生成 GO Fingerprints
    # landmark_path = "data/landmark.txt"
    # go_original = "DeepCOP/Data/go_fingerprints.csv"
    # go_output = "data/landmark_go_fingerprints.csv"
    # meta_json = "DeepCOP/Data/landmark_genes.json"
    
    # generate_landmark_go_fingerprints(landmark_path, go_original, go_output, meta_json)
