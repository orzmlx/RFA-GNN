import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

class DrugEncoder:
    def __init__(self):
        self.smiles_map = {}
        
    def load_data(self, path=None):
        # 简单模拟或加载真实 SMILES 数据
        # 这里为了演示，我们假设可以直接从 pert_info 中获取
        pass
        
    def encode(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(2048)
            
            # 使用新版 API: MorganGenerator
            # 旧版 AllChem.GetMorganFingerprintAsBitVect 已弃用
            try:
                mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp = mfgen.GetFingerprint(mol)
            except ImportError:
                # 回退到旧版 (如果 RDKit 版本很老)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                
            return np.array(fp)
        except:
            return np.zeros(2048)
            
    def encode_batch(self, smiles_list):
        return np.array([self.encode(s) for s in smiles_list])
