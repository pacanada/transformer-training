from pathlib import Path


class Config:
    # TODO: split experiment and model config
    root_dir = Path(".").parent
    data_dir = root_dir / "data"
    historical_dir = data_dir / "historical"
    pytorch_dir = data_dir / "pytorch"
    weights_dir = data_dir / "weights"
    features = ["open"]
    numeric_target = "target_5"
    class_target = "target_5_class"
    training_ratio = 0.9
    percentiles = [0.5]
    bins_label = [0]
    block_size = 100
    embedding_dim = 8
    n_head = 4
    dropout = 0.1
    batch_size = 100
    epochs = 1000
    evaluation_steps = 100
    learning_rate = 1e-3
    load_model = True
    n_blocks=6
    vocab_size = 2


    def __post_init__(self):
        if self.embedding_dim%self.n_head!=0:
            raise ValueError(f"Embedding dimension {self.embedding_dim} should be a multiple of n_head={self.n_head}")
        # create path if it does not exist
        import os
        if not os.path.exists(Path(Config.path_model)):
            os.makedirs(self.path_model)
        
        
    def dict(self):
        list_of_attributes = [a for a in dir(Config) if not a.startswith('__') and not callable(getattr(Config,a))]
        dict_of_attributes = {k:v for k,v in Config.__dict__.items() if k in list_of_attributes}
        return dict_of_attributes
    
    def json(self):
        import json
        return json.dumps(self.dict())

config = Config()