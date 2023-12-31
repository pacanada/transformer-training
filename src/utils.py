import pickle
import json
import os
from pathlib import Path
import torch
import wandb
from src.config import config
class WandbLogger:
    def __init__(self, project_name:str, entity_name:str, model_params:dict):
        self.run = wandb.init(config=model_params,reinit=True, project=project_name, entity=entity_name)
    def log(self, kwargs):
        self.run.log(kwargs)
    def log_weights(self, weights: torch.Tensor, name:str):
        torch.save(weights, config.weights_dir / f"{name}.pt")
        artifact = wandb.Artifact(name=name, type='model')
        artifact.add_file(config.weights_dir / f"{name}.pt")
        self.run.log_artifact(artifact)

    def log_artifact(self, name:str, model, metrics_dict):
        temp_dir = Path("temp/")
        os.makedirs(temp_dir, exist_ok=True)
        
        pickle.dump(model, open(temp_dir / f"{name}.pickle", 'wb'))
        json.dump(metrics_dict, open(temp_dir /f"{name}.json", 'w'))
        self.artifact = wandb.Artifact(name=name, type='model')
        self.artifact.add_file(temp_dir /f"{name}.pickle")
        self.artifact.add_file(temp_dir /f"{name}.json")
        self.run.log_artifact(self.artifact)