from pathlib import Path


class Config:
    root_dir = Path(".").parent
    numeric_target = "target_5"
    class_target = "target_5_class"

config = Config()