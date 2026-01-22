import pandas as pd
import requests
import gzip
import io
import os

def process_string_links_only(output_file='data/string_all_merged.csv', min_score=0):
    """
    只下载 STRING Links 数据（最全的无向网络）。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 使用 v11.5 Links (这个链接刚刚验证是有效的)
    url = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
    
    print(f"正在下载 Links 数据: {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("下载完成，正在处理...")
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            df = pd.read_csv(f, sep=' ')
            
        # 去掉 ID 前缀
        df['protein1'] = df['protein1'].str.replace('9606.', '', regex=False)
        df['protein2'] = df['protein2'].str.replace('9606.', '', regex=False)
        
        # 重命名列
        df = df.rename(columns={'protein1': 'source', 'protein2': 'target', 'combined_score': 'score'})
        
        # 标记为无向边 (direction = 0)
        df['direction'] = 0
        df['type'] = 'ppi'
        
        if min_score > 0:
            df = df[df['score'] >= min_score]
            
        print(f"最终保存 {len(df)} 条相互作用。")
        print(df.head())
        
        df.to_csv(output_file, index=False)
        print(f"已保存到 {output_file}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    process_string_links_only()
