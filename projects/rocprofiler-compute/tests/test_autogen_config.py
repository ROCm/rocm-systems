import hashlib
from pathlib import Path

import yaml


def test_modification_time():
    # Ensure hash map consistency
    hash_path = Path("utils/autogen_hash.yaml")
    with open(hash_path, "r") as f:
        hash_map = yaml.safe_load(f)
    for file, hash in hash_map.items():
        file_hash = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        assert (
            file_hash == hash
        ), f"Hash mismatch for {file}: expected {hash}, got {file_hash}"
