import re
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.specs import generate_machine_specs

rocprof_compute = SourceFileLoader("rocprof-compute", "src/rocprof-compute").load_module()


# NOTE: Only testing gfx942 for now.
GFX942_CHIP_IDS = ["29856", "29876", "29857", "29877", "29858", "29878", "29861", "29881"]

# Expected result
GFX942_NUM_XCDS = {"spx": 8, "dpx": 4, "qpx": 2, "cpx": 1}


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


def gpu_arch():
    gpu_arch = None

    ## 1) Parse arch details from rocminfo
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

    if not gpu_arch:
        ## 2) Try parse arch details from chip id.
        chip_id = re.compile(r"^\s*Chip ID:\s+ ([a-zA-Z0-9]+)\s*", re.MULTILINE)
        ids = list(filter(chip_id.match, rocminfo))
        for id in ids:
            chip_id = re.match(r"^[^()]+", id.split()[2]).group(0)

        if chip_id in GFX942_CHIP_IDS:
            gpu_arch = "gfx942"

    return gpu_arch


@pytest.mark.num_xcds_spec_class
def test_num_xcds_spec_class(monkeypatch):
    arch = gpu_arch()

    # 1. Check if gfx942 soc
    if not arch or "gfx942" not in arch.lower():
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    # 2. load machine specs
    machine_spec = generate_machine_specs(None)

    # 3. check results are expected
    assert machine_spec.compute_partition is not None
    assert int(machine_spec.num_xcd) == GFX942_NUM_XCDS.get(
        machine_spec.compute_partition.lower(), -1
    )


@pytest.mark.num_xcds_cli_output
def test_num_xcds_cli_output():
    arch = gpu_arch()

    # 1. Check if gfx942 soc
    if not arch or "gfx942" not in arch.lower():
        pytest.skip("Skipping num xcds test for non-gfx942 socs.")

    # 2. Run rocprof-compute -s and grab rocprof-compute num_xcd
    proc = subprocess.run(
        ["rocprof-compute", "-s"],
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
    assert int(num_xcd_actual) == GFX942_NUM_XCDS.get(
        compute_partition_actual.lower(), -1
    )
