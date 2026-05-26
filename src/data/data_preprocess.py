
import os
import pandas as pd
import fsspec
import numpy as np
import json
import h5py
import sys
import re
import copy
import csv


def get_landmark_ids(landmark_file):
    if landmark_file is None or len(landmark_file) == 0:
        return
    gene_info_path = landmark_file
    if not os.path.exists(gene_info_path):
        raise FileNotFoundError(f"找不到基因信息文件: {gene_info_path}")
    print(f"正在读取基因信息: {gene_info_path}")
    with open(gene_info_path, "r") as f:
        genes = json.load(f)
    # 用entrez_id（数字字符串）作为landmark_ids
    landmark_ids = [str(g["entrez_id"]) for g in genes if g.get("l1000_type") == "landmark" and g.get("entrez_id")]
    print(f"成功提取 {len(landmark_ids)} 个 Landmark 基因 entrez_id。示例: {landmark_ids[:5]}")
    return landmark_ids


def extract_landmark_from_local(gctx_path, landmark_ids=None, siginfo_path=None, cell_lines=None, durations=[24], doses=["10uM"]):
    """
    从本地 GCTX 文件提取基因数据。
    支持过滤：
    1. landmark_ids: 基因过滤 (行)
    2. siginfo_path + conditions: 样本过滤 (列)
    """

    print(f"\n开始处理本地文件: {gctx_path}")
    if not os.path.exists(gctx_path):
        print(f"错误: 找不到文件 {gctx_path}")
        return

    # 1. 样本过滤逻辑
    target_sample_ids = None
    if siginfo_path and os.path.exists(siginfo_path) and (cell_lines or durations or doses):
        print(f"正在根据 {siginfo_path} 过滤样本...")
        print(f"条件: Cells={cell_lines}, Time={durations}, Dose={doses}")
        try:
            info_df = pd.read_csv(siginfo_path, sep='\t', engine='python')
            filtered_df = filter_by_condition(info_df, cell_lines, durations, doses)
            # distil_ids 可能包含多个 ID (以|分隔)
            target_sample_ids = set[str](_expand_distil_ids(filtered_df['distil_ids'].tolist()))
            print(f"符合条件的样本 ID 数: {len(target_sample_ids)}")
            
            if len(target_sample_ids) == 0:
                print("警告: 没有符合条件的样本。")
                return None
        except Exception as e:
            print(f"样本过滤失败: {e}")
            return None

    try:
        with h5py.File(gctx_path, 'r') as h5:
            matrix_shape = h5['0/DATA/0/matrix'].shape
            print(f"Matrix shape: {matrix_shape}")
            row_ids = h5['0/META/ROW/id'][:].astype(str)
            col_ids = h5['0/META/COL/id'][:].astype(str)
            print(f"Row IDs (Genes): {len(row_ids)}")
            print(f"Col IDs (Samples): {len(col_ids)}")
            
            # 2. 计算样本索引 (Columns)
            is_sample_row = False
            if matrix_shape[0] == len(col_ids): # (Samples, Genes)
                print("检测到格式: (Samples, Genes)")
                is_sample_row = True
                sample_axis_len = len(col_ids)
            elif matrix_shape[0] == len(row_ids): # (Genes, Samples)
                print("检测到格式: (Genes, Samples)")
                is_sample_row = False
                sample_axis_len = len(col_ids)
            else:
                print("警告: 矩阵维度与 Metadata 不匹配！")
                return

            sample_indices = slice(None)
            final_col_ids = col_ids
            
            if target_sample_ids is not None:
                # 找出 col_ids 中在 target_sample_ids 里的索引
                # 为了加速，使用字典映射
                col_to_idx = {cid: i for i, cid in enumerate(col_ids)}
                # 只需要交集
                valid_ids = [cid for cid in target_sample_ids if cid in col_to_idx]
                if not valid_ids:
                    print("错误: H5 文件中没有包含目标样本 ID。")
                    return None
                    
                indices = [col_to_idx[cid] for cid in valid_ids]
                indices.sort() # H5 读取通常要求索引有序
                sample_indices = indices
                final_col_ids = col_ids[indices]
                print(f"最终读取样本数: {len(final_col_ids)}")

            # 3. 计算基因索引 (Rows)
            if landmark_ids is None or len(landmark_ids) == 0:
                print("未指定 Landmark IDs，将提取所有基因。")
                sorted_indices = slice(None)
                sorted_landmark_ids = row_ids
            else:
                gene_id_to_idx = {gid: i for i, gid in enumerate(row_ids)}
                target_indices = []
                valid_landmark_ids = []
                for lid in landmark_ids:
                    if lid in gene_id_to_idx:
                        target_indices.append(gene_id_to_idx[lid])
                        valid_landmark_ids.append(lid)
                
                if not target_indices:
                    print("错误: 未找到匹配的 Landmark 基因。")
                    return None
                    
                sorted_idx_order = np.argsort(target_indices)
                sorted_indices = [target_indices[i] for i in sorted_idx_order]
                sorted_landmark_ids = [valid_landmark_ids[i] for i in sorted_idx_order]
                print(f"匹配到 {len(sorted_indices)} 个 Landmark 基因。")

            # 4. 读取数据
            # h5[...] 支持花式索引 (fancy indexing)，但效率不如 slice
            # 最好是一次性读取? 还是分块?
            # 如果 indices 是列表，h5py 会自动处理
            
            print("正在读取矩阵数据...")
            if is_sample_row:
                # data shape: (Samples, Genes)
                # 先按样本切，再按基因切
                # 注意: h5py 不支持同时在两个轴上使用花式索引 (list indices)
                # 必须分两步: data[samples, :][:, genes]
                
                # Step 1: Filter Samples (Rows)
                if isinstance(sample_indices, list):
                    # 为了避免内存爆炸，如果样本太多，可能需要分块。但这里假设过滤后内存够用。
                    # h5py 读取花式索引比较慢，且需要将数据集读入内存? 
                    # 不，h5py 支持 selection。
                    # 但 h5py 要求索引是递增的。我们已经 sort 了。
                    temp = h5['0/DATA/0/matrix'][sample_indices, :]
                else:
                    temp = h5['0/DATA/0/matrix'][:]
                
                # Step 2: Filter Genes (Cols)
                if isinstance(sorted_indices, list):
                    data = temp[:, sorted_indices]
                else:
                    data = temp
                    
                data = data.T # 转置为 (Genes, Samples)
                
            else:
                # data shape: (Genes, Samples)
                # Step 1: Filter Genes (Rows)
                if isinstance(sorted_indices, list):
                    temp = h5['0/DATA/0/matrix'][sorted_indices, :]
                else:
                    temp = h5['0/DATA/0/matrix'][:]
                    
                # Step 2: Filter Samples (Cols)
                if isinstance(sample_indices, list):
                    data = temp[:, sample_indices]
                else:
                    data = temp

            print(f"提取完成，数据形状: {data.shape}")
            print("正在构建 DataFrame...")
            df = pd.DataFrame(data, index=sorted_landmark_ids, columns=final_col_ids)
            return df

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

def save_df_to_h5_csv(df, output_name,save_path="data/cmap"):
    output_path = os.path.join(save_path, output_name)
    output_csv_path = output_path.replace('.h5', '.csv')
    print(f"正在保存到 {output_path} (h5) ...")
    df.to_hdf(output_path, key='data', mode='w')
    print("h5保存成功。")
    print(f"正在保存到 {output_csv_path} (csv) ...")
    df.to_csv(output_csv_path)
    print("csv保存成功。")



def extrac_data_from_gctx(landmark_file, info_path, cell_lines=[], durations=[], doses=[], to_csv=False):
    if landmark_file:
        landmarks = get_landmark_ids(landmark_file)
    else:
        landmarks = None
        
    # 处理 CTL 文件
    print("处理 CTL 文件...")
    # CTL 不需要过滤剂量，因为 Vehicle 通常没有剂量或剂量为0/无关
    # 但如果是 align 逻辑，CTL 应该和 TRT 匹配。
    # 这里的 extract_landmark_from_local 逻辑是：如果有过滤条件，就过滤。
    # 对于 CTL，我们通常只关心 Cell Line 和 Duration。
    # 如果传了 doses，可能会把 CTL 滤光（因为 CTL 剂量可能标记为 -666）。
    # 所以传给 CTL 的 doses 应该是 None。
    
    ctl_df = extract_landmark_from_local(
        "data/cmap/level3_beta_ctl_n188708x12328.gctx", 
        landmarks,
        siginfo_path=info_path,
        cell_lines=cell_lines,
        durations=durations,
        doses=None # CTL 不过滤剂量
    )
    
    # 处理 TRT 文件
    print("处理 TRT 文件...")
    trt_df = extract_landmark_from_local(
        "data/cmap/level3_beta_trt_cp_n1805898x12328.gctx", 
        landmarks,
        siginfo_path=info_path,
        cell_lines=cell_lines,
        durations=durations,
        doses=doses
    )
    
    if to_csv:
        if ctl_df is not None:
            save_df_to_h5_csv(ctl_df, "level3_beta_ctl_n188708x12328.h5")
        if trt_df is not None:
            save_df_to_h5_csv(trt_df, "level3_beta_trt_cp_n1805898x12328.h5")

    return ctl_df, trt_df

# def extract_cell_lines(col_list, cell_lines= []):
#     if cell_lines is not None and len(cell_lines) > 0:
#         print(f"正在过滤细胞系: {cell_lines}")
#         filtered_cols = [col for col in col_list if any(cl in col for cl in cell_lines)]
#         if len(filtered_cols) == 0:
#             print("警告: 未在样本ID中找到指定细胞系，未做细胞系过滤。")
#         else:
#             print(f"细胞系过滤后剩余样本数: {len(filtered_cols)}")
#     return filtered_cols

# def extract_durations(col_list, durations= []):
#     if durations is not None and len(durations) > 0:
#         print(f"正在过滤处理时间: {durations}")
#         filtered_cols = [col for col in col_list if any(dur in col for dur in durations)]
#         if len(filtered_cols) == 0:
#             print("警告: 未在样本ID中找到指定处理时间，未做处理时间过滤。")
#         else:
#             print(f"处理时间过滤后剩余样本数: {len(filtered_cols)}")
           
#     return filtered_cols



# def extract_doses(col_list, infopath, doses= []):
#     if doses is not None and len(doses) > 0:
#         print(f"正在根据注释文件过滤剂量: {doses}")
#         dose_info = pd.read_csv(infopath, sep=None, engine='python')
#         # 支持多剂量和单位
#         dose_mask = False
#         for dose in doses:
#             num = re.search(r'\d+', dose).group()
#             # 提取第一个数字
#             unit = re.search(r'[A-Za-z]+',dose).group() 
#             # 支持剂量单位为 UM（不区分大小写）
#             mask = (dose_info['pert_dose'] >= float(num) - 0.1) & (dose_info['pert_dose'] <= float(num) + 0.1) & (dose_info['pert_dose_unit'].str.upper() == unit.upper())
#             dose_mask = dose_mask | mask if isinstance(dose_mask, pd.Series) else mask
#         filtered_sample_ids_raw = dose_info.loc[dose_mask, 'distil_ids'].tolist()
#         # 拆分以|分隔的id
#         filtered_sample_ids = []
#         for item in filtered_sample_ids_raw:
#             if isinstance(item, str) and '|' in item:
#                 filtered_sample_ids.extend(item.split('|'))
#             else:
#                 filtered_sample_ids.append(item)
#         filtered_sample_ids = [x for x in filtered_sample_ids if isinstance(x, str) and x.strip()]
#         filtered_set = set(filtered_sample_ids)
#         filtered_cols = [col for col in col_list if col in filtered_set]
#         if len(filtered_cols) == 0:
#             print("警告: 注释文件未找到指定剂量样本，未做剂量过滤。")
#         else:
#             print(f"剂量过滤后剩余样本数: {len(filtered_cols)}")
#     return filtered_cols


def filter_by_condition(info_df, cell_lines=[], durations=[], doses=[]):

    # import csv
    # if not os.path.exists(csv_path):
    #     print(f"文件不存在: {csv_path}")
    #     return
    # only read columns names
    # with open(csv_path, newline='') as f:
    #     col_list = next(csv.reader(f))
    # filtered_cols = extract_doses(col_list, infopath, doses)
    # if len(filtered_cols) == 0:
    #     print("剂量过滤后无样本，退出。")
    #     sys.exit(1)
    # 细胞系过滤
    # 如需用注释文件过滤处理时间，可用：
    # info_df = pd.read_csv(infopath, sep=None, engine='python')
    if durations is not None and len(durations) > 0:
        info_df = info_df[info_df['pert_time'].isin(durations)]
    if cell_lines is not None and len(cell_lines) > 0:
        info_df = info_df[info_df['cell_mfc_name'].str.upper().isin([c.upper() for c in cell_lines])]
    if doses is not None and len(doses) > 0:
        for dose in doses:
            num = re.search(r'\d+', dose).group()
            # 提取第一个数字
            unit = re.search(r'[A-Za-z]+',dose).group() 
            # 支持剂量单位为 UM（不区分大小写）
            mask = (info_df['pert_dose'] >= float(num) - 0.1) & (info_df['pert_dose'] <= float(num) + 0.1) & (info_df['pert_dose_unit'].str.upper() == unit.upper())
            info_df = info_df[mask]
    return info_df



def preliminary_filter(df):


    # 删除pert_dose列是0或者空的行
   # df = df[df['pert_dose'].notna() & (df['pert_dose'] != 0)]
    # 剔除 det_plates（实验板）、cell_id（细胞系 ID）缺失的样本（用于后续配对）；
    df = df[df['det_plates'].notna() & df['cell_iname'].notna()]
    # XPert 药物样本筛选（正确逻辑）

    trt_df = copy.deepcopy(df)
    ctl_df = copy.deepcopy(df)

    trt_df = trt_df[trt_df['pert_type'] == 'trt_cp']  # 对应论文中的“pert_type=="drug"”
    # 同时筛选对照组（用于配对）
    ctl_df = ctl_df[ctl_df['pert_type'] =='ctl_vehicle']
    ctl_df = ctl_df[ctl_df['pert_id'].str.upper() == 'DMSO']
    return trt_df, ctl_df


def _expand_distil_ids(distil_ids):
    expanded = []
    for item in distil_ids:
        if isinstance(item, str) and '|' in item:
            expanded.extend([x for x in item.split('|') if x.strip()])
        else:
            expanded.append(item)
    return [x for x in expanded if isinstance(x, str) and x.strip()]


def _create_match_tag(row):
    # 对齐字段：cell_mfc_name（细胞系名称）、pert_time（处理时间）、vehicle（对照类型）、bead_batch（实验批次）
    return f"{row['cell_mfc_name']}_{row['pert_time']}_{row['bead_batch']}"


def align_trt_ctl_by_siginfo(info_path, cell_lines=None, durations=None, doses=None):
    """基于siginfo匹配条件对齐trt/ctl样本，返回对应distil_ids列表。"""
    info_df = pd.read_csv(info_path, sep='\t', engine='python')
    trt_df, ctl_df = preliminary_filter(info_df)
    # 筛选暴露时间和剂量条件
    trt_df_filter = filter_by_condition(trt_df, cell_lines, durations, doses)
    # ctl group no need to filter dose
    ctl_df_filter = filter_by_condition(ctl_df, cell_lines, durations, None)

    # 构造匹配标签
    trt_df_filter['match_tag'] = trt_df_filter.apply(_create_match_tag, axis=1)
    ctl_df_filter['match_tag'] = ctl_df_filter.apply(_create_match_tag, axis=1)

    # trt_meta = info_df[info_df['type'] == 'trt']
    # ctl_meta = info_df[info_df['type'] == 'ctl']
    common_tags = list(set(trt_df_filter['match_tag']) & set(ctl_df_filter['match_tag']))
    trt_df_filter = trt_df_filter[trt_df_filter['match_tag'].isin(common_tags)]
    ctl_df_filter = ctl_df_filter[ctl_df_filter['match_tag'].isin(common_tags)]

    trt_ids = _expand_distil_ids(trt_df_filter['distil_ids'].tolist())
    ctl_ids = _expand_distil_ids(ctl_df_filter['distil_ids'].tolist())
    return trt_ids, ctl_ids


def align_trt_ctl_csv(trt_csv_path, ctl_csv_path, info_path, cell_lines=None, durations=None, doses=None, save_path="data/cmap",suffix = "_aligned"):
    """按对齐后的distil_ids过滤TRT/CTL CSV并保存。"""
    trt_ids, ctl_ids = align_trt_ctl_by_siginfo(info_path, cell_lines, durations, doses)

    
    for csv_path, keep_ids in [(trt_csv_path, trt_ids), (ctl_csv_path, ctl_ids)]:
        if not os.path.exists(csv_path):
            print(f"文件不存在: {csv_path}")
            continue
        with open(csv_path, newline='') as f:
            col_list = next(csv.reader(f))
        index_col_name = col_list[0] if col_list else None
        keep_set = set(keep_ids)
        filtered_cols = [col for col in col_list if col in keep_set]
        if index_col_name and index_col_name not in filtered_cols:
            filtered_cols = [index_col_name] + filtered_cols
        if len(filtered_cols) <= 1:
            print(f"未匹配到列，跳过: {csv_path}")
            continue
        df = pd.read_csv(csv_path, usecols=filtered_cols, index_col=0)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_name = f"{base_name}{suffix}.h5"
        save_df_to_h5_csv(df, output_name, save_path=save_path)


# 冷启动划分函数
def cold_split(df, split_type='drug', test_ratio=0.2, random_state=42):
    """
    冷启动划分：将未见过的新药或新细胞划分为测试集。
    split_type: 'drug' 或 'cell'
    test_ratio: 测试集比例
    返回：训练集索引、测试集索引
    """
    np.random.seed(random_state)
    if split_type == 'drug':
        unique_drugs = df['pert_id'].unique()
        test_drugs = np.random.choice(unique_drugs, int(len(unique_drugs) * test_ratio), replace=False)
        test_idx = df['pert_id'].isin(test_drugs)
    elif split_type == 'cell':
        unique_cells = df['cell_mfc_name'].unique()
        test_cells = np.random.choice(unique_cells, int(len(unique_cells) * test_ratio), replace=False)
        test_idx = df['cell_mfc_name'].isin(test_cells)
    else:
        raise ValueError('split_type must be "drug" or "cell"')
    train_idx = ~test_idx
    return df[train_idx], df[test_idx]



if __name__ == "__main__":
    # 加载 Landmark 列表
    # landmark_file = "data/landmark_genes.json" # 或者从 gene_info 获取
    # # 这里为了演示，我们先加载 gene_info 获取 ID
    #cell_lines  = ['LNCAP']
    cell_lines = []
    durations = [24]
    doses = ["10uM"]
    infopath = "data/siginfo_beta.txt"
    ctl_df, trt_df = extrac_data_from_gctx(None, infopath, cell_lines, durations, doses, to_csv=True)
   # align_trt_ctl_csv(ctl_csv_path, trt_csv_path, infopath, cell_lines, durations, doses)
    # # 读取siginfo元数据
    # info_df = pd.read_csv(infopath, sep='\t', engine='python')
    # # 冷启动划分：以新药为测试集
    # train_df, test_df = cold_split(info_df, split_type='drug', test_ratio=0.2)
    # print(f"冷启动划分结果：训练集{len(train_df)}, 测试集{len(test_df)}")
