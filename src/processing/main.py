import pandas as pd
from src.config import config
from src.processing.utils import create_sequences, create_sequences_target, standardize
import numpy as np
import torch
import os



def main():


    df = pd.read_parquet(config.root_dir / "data" / "historical" / "dataset.parquet")
    # Add classification target
    # old [0.025, 0.15, 0.5 , 0.85, 0.975]
    bins_no_inf = df[config.numeric_target].quantile([0.5]).to_list()
    print(bins_no_inf)
    bins = [-np.inf]+bins_no_inf+[np.inf]
    labels = [int(i) for i in range(len(bins)-1)]

    df[config.class_target] = pd.cut(df[config.numeric_target], bins=bins, labels=labels).astype(int)

    # Generate sequences normalized by the mean and std of the set defined by the pair_name and the sequence length (block_size)
    X_train_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float32)
    X_test_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float32)

    y_train_all = torch.empty((0, config.block_size), dtype=torch.uint8)
    y_test_all = torch.empty((0, config.block_size), dtype=torch.uint8)

    for name, group in df.groupby("pair_name"):
        print("Generating trainining data for", name)
        X_raw = create_sequences(group[config.features].values, config.block_size)
        y = create_sequences_target(group[config.class_target].values, config.block_size)

        X = standardize(X_raw)


        length = len(X)
        idx_train = int(length*config.training_ratio)   
        X_train, y_train = X[:idx_train], y[:idx_train]
        X_test, y_test = X[idx_train:], y[idx_train:]
        X_train_all = torch.cat((X_train_all, X_train), dim=0)
        y_train_all = torch.cat((y_train_all, y_train), dim=0)
        X_test_all = torch.cat((X_test_all, X_test), dim=0)
        y_test_all = torch.cat((y_test_all, y_test), dim=0)


    # Serialize all variables to pytorch
    torch.save(X_train_all, config.root_dir / "data" / "pytorch" / "X_train_all.pt")
    torch.save(y_train_all, config.root_dir / "data" / "pytorch" / "y_train_all.pt")
    torch.save(X_test_all, config.root_dir / "data" / "pytorch" / "X_test_all.pt")
    torch.save(y_test_all, config.root_dir / "data" / "pytorch" / "y_test_all.pt")




if __name__ == "__main__":
    main()