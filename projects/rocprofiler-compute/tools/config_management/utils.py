from pathlib import Path
from typing import Union

import yaml


def str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_representer)


def load_yaml(filepath: Union[str, Path]) -> dict:
    with open(filepath) as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, filepath: Union[str, Path]) -> None:
    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False)
