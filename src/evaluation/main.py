from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.processing.utils import add_target
import pandas as pd
from src.config import config
from src.models.transformers import Transformer
from src.processing.utils import create_sequences, create_sequences_target, standardize


pairs =["xlmeur", "bcheur","compeur","xdgeur", "etheur", "algoeur", "bateur", "adaeur","xrpeur"]
input_directory = config.data_dir / "validation"
for pair_name in pairs:
    df = pd.read_csv(input_directory / f"{pair_name}.csv").head(700)

    # Assign target
    df = add_target(df, column_to_apply="open", target_list=[5])

    #print(df)
    # dropna
    df = df.dropna()
    bins = [-np.inf]+ config.bins_label+[np.inf]
    labels = [int(i) for i in range(len(bins)-1)]
    
    df[config.class_target] = pd.cut(df[config.numeric_target], bins=bins, labels=labels).astype(int)

    # Generate sequence

    X_raw = create_sequences(df[config.features].values, config.block_size)
    X = standardize(X_raw)
    y = create_sequences_target(df[config.class_target].values, config.block_size)

    # Load model

    m = Transformer(config)
    # load weights from gpu

    idx = -1

    m.load_state_dict(torch.load(config.weights_dir / "weights.pt", map_location=torch.device("cpu")))

    # Evaluate model
    m.eval()
    out = m(X.squeeze())
    loss_fn_aux = nn.CrossEntropyLoss(reduction="none")
    loss_fn = nn.CrossEntropyLoss()
    loss_all = loss_fn(out.view(-1,config.vocab_size),y.view(-1))
    loss_last = loss_fn(out[:,idx,:], y[:,idx])

    # verctor of loss across all blocks
    loss_aux = loss_fn_aux(out.view(-1,config.vocab_size),y.view(-1)) #.detach().numpy()
    #plt.imshow(loss_aux.reshape(-1,100).detach().numpy())
    #plt.show()
    plt.plot(loss_aux.reshape(-1,100).mean(dim=0).detach().numpy())
    plt.show()
    print(loss_aux)
    print(f"{pair_name}:Loss all ", loss_all.item(), "Loss last", loss_last.item())

    predicted_class = out[:,idx,:].softmax(-1).argmax(1).detach().numpy()
    target = y[:,idx].detach().numpy()

    predicted_all = out.view(-1, config.vocab_size).softmax(-1).argmax(1).detach().numpy()
    target_all = y.view(-1).detach().numpy()
    print("Accuracy", (predicted_class==target).mean())
    print("Predicted class")
    print(pd.DataFrame(predicted_class).value_counts().to_dict())
    print("Target")
    print(pd.DataFrame(target).value_counts().to_dict())
    print("Predicted class all")
    print(pd.DataFrame(predicted_all).value_counts().to_dict())
    print("Target all")
    print(pd.DataFrame(target_all).value_counts().to_dict())

    # is around 1.3 a good loss for unbalance 6 classes?

    # plt.plot(predicted_class, label="predicted")
    # plt.plot(target, label="target")
    # plt.legend()
    # plt.show()

    # TODO: nan in bateur is because of open price not changing