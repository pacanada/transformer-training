import torch
import numpy as np
# god bless gpt4 for speeding up this process
def create_sequences(t:np.array, block_size:int, dtype=torch.float32):
    n, f = t.shape[0], t.shape[-1]
    
    # Preallocate memory for efficiency
    X_all = np.empty((n-block_size+1, block_size, f))
    
    for i in range(n-block_size+1):
        X_all[i] = t[i:i+block_size]
        
    return torch.tensor(X_all, dtype=dtype)



def create_sequences_target(t:np.array, block_size:int):
    n = t.shape[0]
    X_all = np.empty((n-block_size+1, block_size))
        
    for i in range(n-block_size+1):
        X_all[i] = t[i:i+block_size]
    return torch.tensor(X_all, dtype=torch.uint8)

def standardize(data):
    means = data.mean(dim=1, keepdim=True)
    stds = data.std(dim=1, keepdim=True)
    # if stds has 0 values, replace them with 1
    stds[stds==0] = 1
    normalized_data = (data - means) / stds
    return normalized_data

def add_target(df, column_to_apply, target_list=None):
    """Will result in -float when price goes up in n-'target' minutes"""
    if target_list is None:
        targets = [1,2,5,10,20,50,80,100]
    else:
        targets = target_list
    for target in targets:
        df[f"target_{target}"] = df[column_to_apply].pct_change(-target)
    return df
