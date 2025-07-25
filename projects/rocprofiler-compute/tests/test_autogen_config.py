import hashlib
from pathlib import Path

import pytest
import yaml


@pytest.mark.autogen_config
def test_modification_time():
    # Ensure modification time of utils/unified_config.yaml is older than
    # utils/autogen_hash.yaml
    # docs/data/metrics_description.yaml and
    # src/rocprof_compute_soc/analysis_configs/gfx*/*.yaml

    unified_config_path = Path("utils/unified_config.yaml")
    hash_path = Path("utils/autogen_hash.yaml")
    docs_config_path = Path("docs/data/metrics_description.yaml")
    analysis_config_paths = list(
        Path("src/rocprof_compute_soc/analysis_configs").glob("gfx*/*.yaml")
    )

    assert (
        unified_config_path.stat().st_mtime < hash_path.stat().st_mtime
    ), f"{unified_config_path} is not older than {hash_path}"

    assert (
        unified_config_path.stat().st_mtime < docs_config_path.stat().st_mtime
    ), f"{unified_config_path} is not older than {docs_config_path}"

    for analysis_config_path in analysis_config_paths:
        assert (
            unified_config_path.stat().st_mtime < analysis_config_path.stat().st_mtime
        ), f"{unified_config_path} is not older than {analysis_config_path}"

    # Ensure hash map consistency

    with open(hash_path, "r") as f:
        hash_map = yaml.safe_load(f)
    for file, hash in hash_map.items():
        file_hash = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        assert (
            file_hash == hash
        ), f"Hash mismatch for {file}: expected {hash}, got {file_hash}"
