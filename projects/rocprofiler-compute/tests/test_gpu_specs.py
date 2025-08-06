##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import os
import re
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import yaml

from src.utils.specs import generate_machine_specs

rocprof_compute = SourceFileLoader("rocprof-compute", "src/rocprof-compute").load_module()


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

# helper to strip ANSI color codes if your app uses them
ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE.sub("", s)


def parse_table_dict(output: str) -> dict:
    """
    Parse an ASCII table into a dict mapping Spec -> Value.
    """
    lines = [l for l in output.splitlines() if l.startswith("│")]
    # locate header row (the one containing 'Spec' and 'Value')
    header_idx = next(
        (i for i, ln in enumerate(lines) if "Spec" in ln and "Value" in ln), None
    )
    if header_idx is None:
        raise ValueError("Header row with Spec and Value not found")

    header_cells = [c.strip() for c in lines[header_idx].strip("│").split("│")]

    spec_i = header_cells.index("Spec")
    value_i = header_cells.index("Value")

    result = {}
    for ln in lines[header_idx + 2 :]:
        if ln.startswith("├") or ln.startswith("╘"):
            continue
        cells = [c.strip() for c in ln.strip("│").split("│")]
        if len(cells) <= max(spec_i, value_i):
            continue
        spec = cells[spec_i]
        value = cells[value_i]
        if spec:
            result[spec] = value
    return result


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "amd-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load amd-smi")
        assert 0
    return p.stdout.decode("utf-8")


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


def get_gpu_arch():

    rocminfo = str(
        # decode with utf-8 to account for rocm-smi changes in latest rocm
        subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("utf-8")
    )
    rocminfo = rocminfo.split("\n")

    soc_regex = re.compile(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", re.MULTILINE)
    devices = list(filter(soc_regex.match, rocminfo))
    gpu_arch = devices[0].split()[1]
    return gpu_arch


@pytest.mark.num_xcds_spec_class
def test_num_xcds_spec_class(monkeypatch):
    # 1. Check if gfx942 soc
    gpu_arch = get_gpu_arch()
    if gpu_arch is None or gpu_arch.lower() != "gfx942":
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    num_xcds = get_num_xcds()

    # 2. load machine specs
    machine_spec = generate_machine_specs(None)

    # 3. check results are expected
    assert machine_spec.compute_partition is not None
    assert int(machine_spec.num_xcd) == num_xcds.get(
        machine_spec.compute_partition.lower(), -1
    )


@pytest.mark.num_xcds_cli_output
def test_num_xcds_cli_output():
    # 1. Check if gfx942 soc
    gpu_arch = get_gpu_arch()
    if gpu_arch is None or gpu_arch.lower() != "gfx942":
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    num_xcds = get_num_xcds()

    # 2. Run rocprof-compute -s and grab rocprof-compute num_xcd
    proc = subprocess.run(
        ["src/rocprof-compute", "-s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert (
        proc.returncode == 0
    ), f"Non-zero exit ({proc.returncode}), stderr:\n{proc.stderr}"

    # 3. strip ANSI, parse table
    clean = strip_ansi(proc.stdout)
    return_dict = parse_table_dict(clean)

    # 4. check results are expected
    assert (
        "Compute Partition" in return_dict
    ), "Spec 'Compute Partition' not found in table"
    assert "Num XCDs" in return_dict, "Spec 'Num XCDs' not found in table"

    compute_partition_actual = return_dict["Compute Partition"]
    num_xcd_actual = return_dict["Num XCDs"]

    assert compute_partition_actual is not None
    assert int(num_xcd_actual) == num_xcds.get(compute_partition_actual.lower(), -1)


@pytest.mark.misc
def test_load_yaml_file_not_found():
    """Test _load_yaml with non-existent file - covers lines 104-105"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    non_existent_path = "/path/that/does/not/exist/file.yaml"

    with pytest.raises(SystemExit):
        MIGPUSpecs._load_yaml(non_existent_path)


@pytest.mark.misc
def test_load_yaml_invalid_yaml():
    """Test _load_yaml with corrupted YAML - covers lines 106-107"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [\nunclosed bracket")
        temp_path = f.name

    try:
        with pytest.raises(SystemExit):
            MIGPUSpecs._load_yaml(temp_path)
    finally:
        os.unlink(temp_path)


@pytest.mark.misc
def test_load_yaml_generic_exception():
    """Test _load_yaml generic exception handling - covers lines 108-111"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch("builtins.open", side_effect=PermissionError("Access denied")):
        with pytest.raises(SystemExit):
            MIGPUSpecs._load_yaml("some_file.yaml")


@pytest.mark.misc
def test_get_gpu_series_dict_uninitialized():
    """Test get_gpu_series_dict when dict not populated - covers lines 182-185"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        with pytest.raises(SystemExit):
            MIGPUSpecs.get_gpu_series_dict()


@pytest.mark.misc
def test_get_gpu_series_uninitialized():
    """Test get_gpu_series when dict not populated - covers lines 191-194"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_gpu_series_dict", {}):
        with pytest.raises(SystemExit):
            result = MIGPUSpecs.get_gpu_series("gfx942")


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
def test_get_num_xcds_no_compute_partition_data():
    """Test get_num_xcds when no compute partition data found - covers lines 307-309"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    mock_dict = {"gfx942": None}
    with patch.object(MIGPUSpecs, "_gpu_arch_to_compute_partition_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx942")


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

    result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx950", gpu_model="UNKNOWN_MODEL")


@pytest.mark.misc
def test_get_num_xcds_no_compute_partition():
    """Test get_num_xcds with no compute partition - covers lines 325-327"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(
        gpu_arch="gfx950", gpu_model="MI350", compute_partition=""
    )


@pytest.mark.misc
def test_get_num_xcds_unknown_compute_partition():
    """Test get_num_xcds with unknown compute partition - covers lines 329-332"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(
        gpu_arch="gfx950", gpu_model="MI350", compute_partition="UNKNOWN"
    )


@pytest.mark.misc
def test_get_num_xcds_none_partition_value():
    """Test get_num_xcds when partition value is None - covers lines 338-340"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    mock_dict = {"mi350": {"spx": None}}
    with patch.object(MIGPUSpecs, "_num_xcds_dict", mock_dict):
        result = MIGPUSpecs.get_num_xcds(
            gpu_arch="gfx950", gpu_model="MI350", compute_partition="spx"
        )


@pytest.mark.misc
def test_get_num_xcds_no_gpu_model():
    """Test get_num_xcds with no gpu model - covers line 342"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_num_xcds(
        gpu_arch="gfx950", gpu_model="", compute_partition="spx"
    )


@pytest.mark.misc
def test_get_chip_id_dict_empty():
    """Test get_chip_id_dict when dict is empty - covers line 352"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_chip_id_dict", {}):
        with patch("src.utils.mi_gpu_spec.console_error") as mock_error:
            result = MIGPUSpecs.get_chip_id_dict()
            mock_error.assert_called_once()


@pytest.mark.misc
def test_get_num_xcds_dict_empty():
    """Test get_num_xcds_dict when dict is empty - covers line 359"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    with patch.object(MIGPUSpecs, "_num_xcds_dict", {}):
        with patch("src.utils.mi_gpu_spec.console_error") as mock_error:
            result = MIGPUSpecs.get_num_xcds_dict()
            mock_error.assert_called_once()


@pytest.mark.misc
def test_normal_functionality_still_works():
    """Ensure that normal paths still work after adding error handling tests"""
    from src.utils.mi_gpu_spec import MIGPUSpecs

    result = MIGPUSpecs.get_gpu_model("gfx906", None)
    assert result is not None

    result = MIGPUSpecs.get_gpu_series("gfx906")
    assert result is not None

    result = MIGPUSpecs.get_num_xcds(gpu_arch="gfx906")
    assert result == 1
