import pandas as pd
import requests
import gzip
import io
import os

def download_and_process_string_actions(output_file='data/string_actions_high_confidence.csv', min_score=700):
    """
    下载 STRING 数据库的 "Protein Actions" 数据（包含方向和作用类型）。
    
    Args:
        output_file: 保存路径
        min_score: 最小置信度分数 (0-1000). 
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # URL for Protein Actions (Human v12.0)
    url = "https://stringdb-static.org/download/protein.actions.v12.0/9606.protein.actions.v12.0.txt.gz"
    
    print(f"正在下载 STRING Actions 数据: {url} ...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("下载完成，正在处理...")
        
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            df = pd.read_csv(f, sep='\t')
            
        print(f"原始数据包含 {len(df)} 条记录。")
        print("前5行数据:")
        print(df.head())
        
        # 1. 筛选高分边
        df = df[df['score'] >= min_score].copy()
        print(f"筛选分数 >= {min_score} 后剩余: {len(df)}")
        
        # 2. 筛选有方向的边 (is_directional = 't')
        df = df[df['is_directional'] == 't'].copy()
        print(f"筛选有方向的边后剩余: {len(df)}")
        
        # 3. 筛选明确的激活/抑制关系 (mode = 'activation' or 'inhibition')
        # STRING 还有 'binding', 'catalysis' 等，这些可能没有明确的正负号意义
        target_modes = ['activation', 'inhibition', 'expression']
        df = df[df['mode'].isin(target_modes)].copy()
        print(f"筛选 activation/inhibition/expression 后剩余: {len(df)}")

        # 4. 处理 ID (去前缀)
        df['item_id_a'] = df['item_id_a'].str.replace('9606.', '', regex=False)
        df['item_id_b'] = df['item_id_b'].str.replace('9606.', '', regex=False)
        
        # 5. 转换成 RFA 需要的格式
        # 如果 a_is_acting = 't', 则是 A -> B
        # STRING 的 action 文件通常已经把有方向的行排列好了，但为了保险起见：
        # 我们只保留那些确实有明确方向的行
        
        # 映射符号
        # activation/expression -> +1
        # inhibition -> -1
        
        def get_sign(mode):
            if mode == 'inhibition':
                return -1
            return 1
            
        df['weight'] = df['mode'].apply(get_sign)
        
        # 重命名列以匹配 OmniPath 风格
        df_final = df[['item_id_a', 'item_id_b', 'mode', 'score', 'weight']].rename(columns={
            'item_id_a': 'source',
            'item_id_b': 'target'
        })
        
        print(f"正在保存到 {output_file} ...")
        df_final.to_csv(output_file, index=False)
        print("完成！")
        print(df_final.head())
        
        return df_final
        
    except Exception as e:
        print(f"发生错误: {e}")
        return None

if __name__ == "__main__":
    download_and_process_string_actions(min_score=700)
