
import h5py
import numpy as np

ctl_path = "data/cmap/level3_beta_ctl_n188708x12328.h5"
trt_path = "data/cmap/level3_beta_trt_cp_n1805898x12328.h5"

def inspect_h5(path):
    print(f"\nInspecting {path}...")
    try:
        with h5py.File(path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            def print_attrs(name, obj):
                print(name)
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
                    if obj.shape[0] > 0:
                        try:
                            print(f"  Head: {obj[:5]}")
                        except:
                            pass
            
            f.visititems(print_attrs)
            
    except Exception as e:
        print(f"Error: {e}")

inspect_h5(ctl_path)
inspect_h5(trt_path)
