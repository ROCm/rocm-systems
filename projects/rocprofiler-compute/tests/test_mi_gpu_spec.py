# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.mi_gpu_spec import MIGPUSpecs


# ---------------------------------------------------------------------------
# YAML parsing / initialization
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_yaml_loads_successfully():
    yaml_path = Path(__file__).parent.parent / "src" / "utils" / "mi_gpu_spec.yaml"
    data = MIGPUSpecs._load_yaml(str(yaml_path))
    assert isinstance(data, dict)
    assert "mi_gpu_spec" in data
    assert len(data["mi_gpu_spec"]) > 0


@pytest.mark.mi_gpu_spec
def test_gpu_model_dict_consistent_with_all_models():
    all_models = set(MIGPUSpecs.get_all_gpu_models())
    models_from_dict = set()
    for gpu_models in MIGPUSpecs._gpu_model_dict.values():
        models_from_dict.update(gpu_models)
    assert models_from_dict == all_models


# ---------------------------------------------------------------------------
# get_gpu_series
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_get_gpu_series_all_archs():
    for arch in MIGPUSpecs._gpu_series_dict:
        result = MIGPUSpecs.get_gpu_series(arch)
        assert result is not None, f"get_gpu_series({arch!r}) returned None"
        assert result == result.upper()


# ---------------------------------------------------------------------------
# get_gpu_model
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_get_gpu_model_legacy_archs():
    result = MIGPUSpecs.get_gpu_model("gfx908", None)
    assert result is not None
    assert "MI100" == result

    result = MIGPUSpecs.get_gpu_model("gfx90a", None)
    assert result is not None


@pytest.mark.mi_gpu_spec
def test_get_gpu_model_chip_id_lookup():
    for chip_id, expected_model in MIGPUSpecs._chip_id_dict.items():
        if chip_id is None:
            continue
        result = MIGPUSpecs.get_gpu_model("gfx942", str(chip_id))
        assert result is not None, (
            f"get_gpu_model('gfx942', {chip_id!r}) returned None"
        )
        assert result.lower() == expected_model.lower()


# ---------------------------------------------------------------------------
# get_perfmon_config
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_get_perfmon_config_all_archs():
    for arch in MIGPUSpecs._perfmon_config:
        result = MIGPUSpecs.get_perfmon_config(arch)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# get_num_xcds
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_get_num_xcds_legacy_returns_1():
    legacy_cases = [
        ("gfx908", "mi100"),
        ("gfx90a", "mi210"),
        ("gfx90a", "mi250"),
        ("gfx90a", "mi250x"),
    ]
    for arch, model in legacy_cases:
        result = MIGPUSpecs.get_num_xcds(gpu_arch=arch, gpu_model=model)
        assert result == 1, (
            f"get_num_xcds({arch!r}, {model!r}) returned {result}, expected 1"
        )


@pytest.mark.mi_gpu_spec
def test_get_num_xcds_with_partition():
    for arch, partitions in MIGPUSpecs._gpu_arch_to_compute_partition_dict.items():
        if not isinstance(partitions, dict):
            continue
        for partition, num_xcds in partitions.items():
            if num_xcds is None:
                continue
            result = MIGPUSpecs.get_num_xcds(
                gpu_arch=arch, compute_partition=partition
            )
            assert result == num_xcds, (
                f"get_num_xcds({arch!r}, partition={partition!r}) "
                f"returned {result}, expected {num_xcds}"
            )


# ---------------------------------------------------------------------------
# get_num_dies
# ---------------------------------------------------------------------------


@pytest.mark.mi_gpu_spec
def test_get_num_dies_all_models():
    for arch, models in MIGPUSpecs._gpu_model_dict.items():
        for model in models:
            result = MIGPUSpecs.get_num_dies(arch, model)
            assert isinstance(result, int) and result >= 1, (
                f"get_num_dies({arch!r}, {model!r}) returned {result!r}"
            )


@pytest.mark.mi_gpu_spec
def test_get_num_dies_cdna_no_design():
    with patch.object(
        MIGPUSpecs, "_gpu_design", {"mi100": {}}
    ), patch.object(
        MIGPUSpecs, "_gpu_series_dict", {"gfx908": "mi100"}
    ):
        result = MIGPUSpecs.get_num_dies("gfx908", "mi100")
        assert result == 1


@pytest.mark.mi_gpu_spec
def test_get_num_dies_cdna_with_design():
    design = {"testmodel": {"physical_aid": 4, "logical_partitions_per_die": 2}}
    with patch.object(
        MIGPUSpecs, "_gpu_design", design
    ), patch.object(
        MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}
    ):
        assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 8


@pytest.mark.mi_gpu_spec
def test_get_num_dies_cdna_partial_design():
    design = {"testmodel": {"physical_aid": 4}}
    with patch.object(
        MIGPUSpecs, "_gpu_design", design
    ), patch.object(
        MIGPUSpecs, "_gpu_series_dict", {"gfx942": "mi300"}
    ):
        assert MIGPUSpecs.get_num_dies("gfx942", "testmodel") == 4


@pytest.mark.mi_gpu_spec
def test_get_num_dies_rdna_with_memory_die():
    design = {"rdna_model": {"memory_die": 3}}
    with patch.object(
        MIGPUSpecs, "_gpu_design", design
    ), patch.object(
        MIGPUSpecs, "_gpu_series_dict", {"gfx1151": "navi3"}
    ):
        assert MIGPUSpecs.get_num_dies("gfx1151", "rdna_model") == 3


@pytest.mark.mi_gpu_spec
def test_get_num_dies_rdna_no_memory_die():
    design = {"rdna_model": {}}
    with patch.object(
        MIGPUSpecs, "_gpu_design", design
    ), patch.object(
        MIGPUSpecs, "_gpu_series_dict", {"gfx1151": "navi3"}
    ):
        assert MIGPUSpecs.get_num_dies("gfx1151", "rdna_model") == 1
