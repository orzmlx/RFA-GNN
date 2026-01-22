import pandas as pd
import requests
import gzip
import io
import os

def download_and_process_string(output_file='data/string_high_confidence.csv', min_score=700):
    """
    下载 STRING 数据库的人类蛋白相互作用数据，并筛选高置信度边。
    
    Args:
        output_file: 保存路径
        min_score: 最小置信度分数 (0-1000). 
                   700 = High Confidence
                   900 = Very High Confidence
    """
    # 1. 准备目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 2. STRING 数据库 URL (人类, v12.0)
    # 9606 是人类的 Taxonomy ID
    url = "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    
    print(f"正在下载 STRING 数据: {url} ...")
    print("这可能需要几分钟，取决于网速...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("下载完成，正在解压和读取...")
        
        # 3. 读取数据 (空格分隔)
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            df = pd.read_csv(f, sep=' ')
            
        print(f"原始数据包含 {len(df)} 条相互作用。")
        print("前5行数据:")
        print(df.head())
        
        # 4. 筛选高置信度边
        print(f"\n正在筛选 combined_score >= {min_score} 的边...")
        df_high = df[df['combined_score'] >= min_score].copy()
        
        print(f"筛选后剩余 {len(df_high)} 条相互作用。")
        
        # 5. 处理 ID (可选: 去掉 '9606.' 前缀)
        # STRING ID 格式为 '9606.ENSP00000000233'
        df_high['protein1'] = df_high['protein1'].str.replace('9606.', '', regex=False)
        df_high['protein2'] = df_high['protein2'].str.replace('9606.', '', regex=False)
        
        # 6. 保存
        print(f"正在保存到 {output_file} ...")
        df_high.to_csv(output_file, index=False)
        print("完成！")
        
        return df_high
        
    except Exception as e:
        print(f"发生错误: {e}")
        return None

if __name__ == "__main__":
    # 默认下载 High Confidence (>=700)
    # 你也可以改成 900 (Very High)
    download_and_process_string(min_score=900)
