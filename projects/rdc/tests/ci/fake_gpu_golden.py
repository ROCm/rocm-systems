"""Expected RDC readings for the fake amd-smi in tests/fake_amdsmi.

The fake answers every amdsmi_* call with per-GPU readings from fake_values.inc. This
module reads the same table and converts each reading to the unit RDC documents for the
field (see `rdci dmon -l`), so tests compare RDC's output with an expectation that does
not depend on RDC's own conversion code.
"""

import os
import re

_ROW = re.compile(r"^FAKE_VALUE\((\w+),\s*(-?\d+),\s*(-?\d+),\s*(\d+)\)$")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _table_path():
    for path in (
        os.path.join(_HERE, "fake_values.inc"),  # installed next to this file
        os.path.join(_HERE, "..", "fake_amdsmi", "fake_values.inc"),  # source tree
    ):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("fake_values.inc not found next to this file or in ../fake_amdsmi")


def _load():
    table, rows = {}, 0
    with open(_table_path()) as f:
        for line in f:
            rows += line.startswith("FAKE_VALUE(")
            m = _ROW.match(line.strip())
            if m:
                table[m.group(1)] = tuple(int(m.group(i)) for i in (2, 3, 4))
    if not table or len(table) != rows:
        raise ValueError(f"parsed {len(table)} of {rows} FAKE_VALUE rows in fake_values.inc")
    return table


TABLE = _load()

GIB = 1 << 30


def reading(name, gpu):
    """Raw reading `name` of GPU `gpu`, in amd-smi's unit."""
    base, step, modulo = TABLE[name]
    value = base + step * gpu
    return value % modulo if modulo else value


# RDC field id -> (fake reading, factor to the unit RDC documents for the field).
FIELDS = {
    100: ("GFX_CLK_MHZ", 1_000_000),  # RDC_FI_GPU_CLOCK, Hz
    101: ("MEM_CLK_MHZ", 1_000_000),  # RDC_FI_MEM_CLOCK, Hz
    200: ("TEMP_MEM_C", 1000),  # RDC_FI_MEMORY_TEMP, millidegrees Celsius
    201: ("TEMP_EDGE_C", 1000),  # RDC_FI_GPU_TEMP, millidegrees Celsius
    300: ("SOCKET_POWER_W", 1_000_000),  # RDC_FI_POWER_USAGE, microwatts
    500: ("GFX_ACTIVITY_PCT", 1),  # RDC_FI_GPU_UTIL, percent
    501: ("VRAM_USED_GIB", GIB),  # RDC_FI_GPU_MEMORY_USAGE, bytes
    502: ("VRAM_TOTAL_GIB", GIB),  # RDC_FI_GPU_MEMORY_TOTAL, bytes
    504: ("VCN_ACTIVITY_PCT", 1),  # RDC_FI_GPU_MM_DEC_UTIL, percent
}


def expected(field_id, gpu):
    """Value RDC must report for `field_id` on fake GPU `gpu`."""
    name, factor = FIELDS[field_id]
    return reading(name, gpu) * factor


def device_name(gpu):
    return f"Fake AMD Instinct GPU {gpu}"
