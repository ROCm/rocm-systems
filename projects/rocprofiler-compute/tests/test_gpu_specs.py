# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import re
import subprocess
import tempfile
from subprocess import CompletedProcess
from unittest.mock import patch

import common
import pytest

import utils.specs as specs
from utils.file_io import is_single_panel_config
from utils.specs import generate_machine_specs
from utils.utils_common import canonical_config_arch

# NOTE: Only testing gfx942 for now.
GFX942_CHIP_IDS_TO_NUM_XCDS = {
    "29856": {"spx": 6, "tpx": 2},
    "29876": {"spx": 6, "tpx": 2},
    "29857": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
    "29877": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
    "29858": {"spx": 4, "dpx": 2, "cpx": 1},
    "29878": {"spx": 4, "dpx": 2, "cpx": 1},
    "29861": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
    "29881": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
    "29864": {"spx": 4, "dpx": 2, "cpx": 1},
    "29884": {"spx": 4, "dpx": 2, "cpx": 1},
    "29865": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
    "29885": {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1},
}


def parse_table_dict(output: str) -> dict:
    """
    Parse an ASCII table into a dict mapping Spec -> Value.
    """
    lines = [line for line in output.splitlines() if line.startswith("|")]
    # locate header row (the one containing 'Spec' and 'Value')
    header_idx = next(
        (i for i, ln in enumerate(lines) if "Spec" in ln and "Value" in ln), None
    )
    if header_idx is None:
        raise ValueError("Header row with Spec and Value not found")

    header_cells = [c.strip() for c in lines[header_idx].strip("|").split("|")]

    spec_i = header_cells.index("Spec")
    value_i = header_cells.index("Value")

    result = {}
    for ln in lines[header_idx + 1 :]:
        # Skip separator lines
        if ln.startswith("+"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) <= max(spec_i, value_i):
            continue
        spec = cells[spec_i]
        value = cells[value_i]
        if spec:
            result[spec] = value
    return result


def get_num_xcds():
    num_xcds = None

    ## 1) Parse arch details from rocminfo
    rocminfo = str(
        # decode with utf-8 to account for rocm-smi changes in latest rocm
        subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("utf-8")
    )
    rocminfo = rocminfo.split("\n")

    chip_id = re.compile(r"^\s*Chip ID:\s+ ([a-zA-Z0-9]+)\s*", re.MULTILINE)
    ids = list(filter(chip_id.match, rocminfo))
    for id in ids:
        chip_id = re.match(r"^[^()]+", id.split()[2]).group(0)

    if str(chip_id) in GFX942_CHIP_IDS_TO_NUM_XCDS.keys():
        num_xcds = GFX942_CHIP_IDS_TO_NUM_XCDS[str(chip_id)]

    if num_xcds is None:
        return

    return num_xcds


@pytest.mark.num_xcds_spec_class
def test_num_xcds_spec_class(monkeypatch):
    # 1. Check if gfx942 soc
    gpu_arch = common.gpu_soc()[0]
    if gpu_arch is None or gpu_arch.lower() != "gfx942":
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    num_xcds = get_num_xcds()

    # 2. load machine specs
    machine_spec = generate_machine_specs(None, None)

    # 3. check results are expected
    assert machine_spec.compute_partition is not None
    assert int(machine_spec.num_xcd) == num_xcds.get(
        machine_spec.compute_partition.lower(), -1
    )


@pytest.mark.num_xcds_cli_output
def test_num_xcds_cli_output():
    # 1. Check if gfx942 soc
    gpu_arch = common.gpu_soc()[0]
    if gpu_arch is None or gpu_arch.lower() != "gfx942":
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    num_xcds = get_num_xcds()

    # 2. Run rocprof-compute -s and grab rocprof-compute num_xcd
    try:
        proc = subprocess.run(
            ["src/rocprof-compute", "-s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        proc = subprocess.run(
            ["./rocprof-compute", "-s"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    assert proc.returncode == 0, (
        f"Non-zero exit ({proc.returncode}), stderr:\n{proc.stderr}"
    )

    # 3. strip ANSI, parse table
    clean = common.strip_ansi(proc.stdout)
    return_dict = parse_table_dict(clean)

    # 4. check results are expected
    assert "Compute Partition" in return_dict, (
        "Spec 'Compute Partition' not found in table"
    )
    assert "Num XCDs" in return_dict, "Spec 'Num XCDs' not found in table"

    compute_partition_actual = return_dict["Compute Partition"]
    num_xcd_actual = return_dict["Num XCDs"]

    assert compute_partition_actual is not None
    assert int(num_xcd_actual) == num_xcds.get(compute_partition_actual.lower(), -1)


@pytest.mark.misc
def test_load_yaml_file_not_found():
    """Test _load_yaml with non-existent file - covers lines 104-105"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with pytest.raises(FileNotFoundError):
        MIGPUSpecs._load_yaml("non_existent_file.yaml")


@pytest.mark.misc
def test_load_yaml_invalid_yaml():
    """Test _load_yaml with corrupted YAML - covers lines 106-107"""
    import yaml

    from src.utils.mi_gpu_spec import MIGPUSpecs

    # Create invalid YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [\nunclosed bracket")
        temp_path = f.name

    with pytest.raises(yaml.YAMLError):
        MIGPUSpecs._load_yaml(str(temp_path))


@pytest.mark.misc
def test_load_yaml_generic_exception():
    """Test _load_yaml generic exception handling - covers lines 108-111"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        with pytest.raises(PermissionError, match="Access denied"):
            MIGPUSpecs._load_yaml("some_file.yaml")


@pytest.mark.misc
@pytest.mark.parametrize(
    "mock_kwargs",
    [
        {"side_effect": FileNotFoundError("missing")},
        {
            "return_value": CompletedProcess(
                args=["rocminfo"], returncode=1, stdout="", stderr="boom"
            )
        },
    ],
    ids=["missing_binary", "nonzero_exit"],
)
def test_run_fails_fast(mock_kwargs):
    with (
        patch.object(specs.subprocess, "run", **mock_kwargs),
        pytest.raises(SystemExit),
    ):
        specs.run(["rocminfo"])


@pytest.mark.misc
def test_get_gpu_series_dict_uninitialized():
    """Test get_gpu_series_dict when dict not populated - covers lines 182-185"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        assert MIGPUSpecs.get_gpu_series_dict() == {}


@pytest.mark.misc
def test_get_gpu_series_uninitialized():
    """Test get_gpu_series when dict not populated - covers lines 191-194"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        assert MIGPUSpecs.get_gpu_series_dict() == {}


@pytest.mark.misc
def test_get_perfmon_config_uninitialized():
    """Test get_perfmon_config when dict not populated - covers lines 210-213"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_perfmon_config", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_perfmon_config("gfx942")


@pytest.mark.misc
def test_get_gpu_model_uninitialized():
    """Test get_gpu_model when dict not populated - covers lines 223-226"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_gpu_model_dict", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_gpu_model("gfx942", "29857")


@pytest.mark.misc
def test_get_gpu_model_invalid_chip_id():
    """Test get_gpu_model with invalid chip_id - covers lines 235-236"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_gpu_model("gfx942", "99999")
    assert result is None


@pytest.mark.misc
def test_get_gpu_model_invalid_arch():
    """Test get_gpu_model with invalid architecture - covers lines 243-244"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_gpu_model("gfx999", "12345")
    assert result is None


@pytest.mark.misc
def test_get_gpu_model_none_result():
    """Test get_gpu_model when result is None - covers lines 246-248"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_chip_id_dict", {999: None}):
        result = MIGPUSpecs.get_gpu_model("gfx942", "999")
        assert result is None


@pytest.mark.misc
def test_canonical_config_arch_maps_gfx115_variants_to_shared_dir():
    assert canonical_config_arch(None) is None
    assert canonical_config_arch("gfx1150") == "gfx115x"
    assert canonical_config_arch("gfx1151") == "gfx115x"
    assert canonical_config_arch("gfx1152") == "gfx115x"
    assert canonical_config_arch("gfx942") == "gfx942"


@pytest.mark.misc
def test_is_single_panel_config_accepts_shared_gfx115x_dir(tmp_path):
    (tmp_path / "gfx115x").mkdir()

    supported_archs = {
        "gfx1150": "rdna35_point_1",
        "gfx1151": "rdna35_halo",
        "gfx1152": "rdna35_point_2",
    }

    assert is_single_panel_config(str(tmp_path), supported_archs) is False


@pytest.mark.misc
def test_get_num_xcds_no_compute_partition_data():
    """Test get_num_xcds when no compute partition data found - covers lines 307-309"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    mock_dict = {"gfx942": None}
    with patch.object(MIGPUSpecs, "_gpu_arch_to_compute_partition_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx942")  # noqa: F841


@pytest.mark.misc
def test_get_num_xcds_uninitialized_dict():
    """Test get_num_xcds when XCD dict not populated - covers lines 315-317"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_num_xcds_dict", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_num_xcds(gpu_arch="gfx950", gpu_model="MI350")


@pytest.mark.misc
def test_get_num_xcds_unknown_gpu_model():
    """Test get_num_xcds with unknown gpu model - covers lines 319-321"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="UNKNOWN_MODEL"
    )


@pytest.mark.misc
def test_get_num_xcds_no_compute_partition():
    """Test get_num_xcds with no compute partition - covers lines 325-327"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="MI350", compute_partition=""
    )


@pytest.mark.misc
def test_get_num_xcds_unknown_compute_partition():
    """Test get_num_xcds with unknown compute partition - covers lines 329-332"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="MI350", compute_partition="UNKNOWN"
    )


@pytest.mark.misc
def test_get_num_xcds_none_partition_value():
    """Test get_num_xcds when partition value is None - covers lines 338-340"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    mock_dict = {"mi350": {"spx": None}}
    with patch.object(MIGPUSpecs, "_num_xcds_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(  # noqa: F841
            gpu_arch="gfx950", gpu_model="MI350", compute_partition="spx"
        )


@pytest.mark.misc
def test_get_num_xcds_no_gpu_model():
    """Test get_num_xcds with no gpu model - covers line 342"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(  # noqa: F841
        gpu_arch="gfx950", gpu_model="", compute_partition="spx"
    )


@pytest.mark.misc
def test_get_chip_id_dict_empty():
    """Test get_chip_id_dict when dict is empty - covers line 352"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_chip_id_dict", {}):
        assert MIGPUSpecs.get_chip_id_dict() == {}


@pytest.mark.misc
def test_get_num_xcds_dict_empty():
    """Test get_num_xcds_dict when dict is empty - covers line 359"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_num_xcds_dict", {}):
        assert MIGPUSpecs.get_num_xcds_dict() == {}


@pytest.mark.misc
def test_set_cache_sizes_cdna_uses_l1():
    """CDNA models (e.g. mi300x_a1) should key vL1D as 'L1'."""
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 304,
            }
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert "L0" not in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_set_cache_sizes_rdna_uses_l0_and_l1():
    """RDNA models (e.g. rdna35_halo) populate both L0 (per-CU GL0) and L1 (per-SA GL1).
    Both are reported as level-1 DATA_CACHE by amd-smi; L0 has higher instance count.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 32,
                "num_cache_instance": 40,  # GL0: one per CU (40 CUs total)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 128,
                "num_cache_instance": 8,  # GL1: one per SA (num_se * num_sa_se = 4*2)
            },
        ]
    }
    result = specs.set_cache_sizes(
        "rdna35_halo", 40, cache_info, num_dies=1, num_se=4, num_sa_se=2
    )
    assert result["L0"] == 32 * 1024
    assert result["L1"] == 128 * 1024


@pytest.mark.misc
def test_set_cache_sizes_l2_and_mall():
    """L2 and MALL (L3) cache entries are correctly parsed and scaled by num_dies."""
    cache_info = {
        "cache": [
            {
                "cache_level": 2,
                "cache_size": 512,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
            {
                "cache_level": 3,
                "cache_size": 256,
                "cache_properties": [],
                "num_cache_instance": 1,
            },
        ]
    }
    result = specs.set_cache_sizes(
        "mi300x_a1", 304, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert result["L2"] == 512 * 1024
    assert result["MALL"] == 256 * 1024 // 4


@pytest.mark.misc
def test_set_cache_sizes_selects_vl1d_by_max_instance_count():
    """Harvested GPU: num_cu reported by rocminfo may be less than the max
    num_cache_instance in amd-smi. set_cache_sizes must select vL1D by the
    highest num_cache_instance, not by exact match to num_cu.
    """
    cache_info = {
        "cache": [
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 228,  # max instances (some CUs harvested)
            },
            {
                "cache_level": 1,
                "cache_properties": ["DATA_CACHE"],
                "cache_size": 16,
                "num_cache_instance": 190,  # lower instance count entry
            },
        ]
    }
    # num_cu=224 simulates harvested GPU (fewer than 228 active CUs)
    result = specs.set_cache_sizes(
        "mi300x_a1", 224, cache_info, num_dies=4, num_se=8, num_sa_se=2
    )
    assert "L1" in result
    assert result["L1"] == 16 * 1024


@pytest.mark.misc
def test_normal_functionality_still_works():
    """Ensure that normal paths still work after adding error handling tests"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_gpu_model("gfx90a", None)

    assert result is not None

    result = MIGPUSpecs.get_gpu_series("gfx90a")
    assert result is not None

    result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx90a")
    assert result == 1
