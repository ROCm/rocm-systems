#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate the metric library.

For every arch in src/rocprof_compute_soc/metric_library/, check that:
  1. each file loads, and a child file inherits from its parent arch's file;
  2. metrics and variables have the required fields and types; "peak" is a
     theoretical peak and "peak_empirical" a roofline benchmark ceiling;
  3. every ceiling the roofline benchmark measures on the arch is the
     "peak_empirical" of exactly one metric, or is in CEILINGS_WITHOUT_METRIC;
  4. metric ids look like "{category}.{subcategory}_{name}", name hardware rather than
     a view, and live in the file named after their category, so each id is unique on
     the arch;
  5. no two metrics on the arch share a formula or a name;
  6. units use one spelling;
  7. the library matches analysis_configs: every equation a row shows is a library
     formula (DISPLAY_VARIANTS list the rows that show one in another form), every
     formula is shown, every row's unit is its metric's unit (except UNIT_VARIANTS),
     each row's peak is its metric's "peak" or "peak_empirical" and each library peak
     is shown by a row, the no-filter counters equal detect_counters(), and the
     variables equal get_build_in_vars(). Check 7 is temporary; see "Equivalence with
     analysis_configs" below.
"""

import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_yaml import normalize_equation

import config
from rocprof_compute_soc.soc_base import OmniSoC_Base
from roofline.benchmark.ceilings import measured_ceilings
from roofline.run_benchmark import BENCHMARKING_SUPPORTED
from utils.metric_library import (
    METRIC_LIBRARY_ID_RE,
    METRIC_LIBRARY_PARENT_ARCH,
    VARIABLES_FILENAME,
    MetricLibrary,
    arch_files,
    default_metric_library_dir,
    load_metric_library,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import (
    MEMBW_ANALYSIS_PANEL_ID,
    canonical_config_arch,
    load_panel_configs,
)
from utils.utils_counter_defs import get_build_in_vars, metric_value_keys
from utils.yaml_inherit_loader import (
    INHERIT_KEY,
    InheritanceError,
    InheritLoader,
    load_inherited_yaml,
)
from vendored import yaml

# Categories named after panels rather than hardware.
VIEW_CATEGORIES = ("system", "memchart", "roofline", "wavefront")

# Alias -> canonical unit. A display scale such as Bytes/s vs GB/s is not an alias.
UNIT_ALIASES = {
    "Gbps": "GB/s",
    "GFLOPs": "GFLOP/s",
    "GIOPs": "GIOP/s",
    "(Req + $normUnit)": "(Requests + $normUnit)",
    "Instr/cycle": "Instructions per Cycle",
    "Waves/cycle": "Wavefronts per Cycle",
    "Cycles/wave": "Cycles per Wavefront",
    "KB per Wave": "KB per Wavefront",
    "Wave": "Wavefronts",
    "Waves": "Wavefronts",
}

METRIC_REQUIRED_FIELDS = ("name", "formula", "unit", "description")
METRIC_OPTIONAL_FIELDS = ("peak", "peak_empirical")
# A roofline benchmark ceiling, the measured peak in "peak_empirical".
EMPIRICAL_PEAK_RE = re.compile(r"\$(\w+)_empirical_peak")
# Ceilings the roofline benchmark measures that no row shows as a peak: on every arch,
# the VALU integer rates and MALL bandwidth (analyze skips MALL, roofline_calc.py); per
# arch, the rates the roofline combines or does not show.
CEILINGS_WITHOUT_METRIC = ("I8Ops", "I32Ops", "I64Ops", "MALLBw")
ARCH_CEILINGS_WITHOUT_METRIC = {
    # The roofline shows the combined F6F4 MFMA rate.
    "gfx950": ("MFMAF4Flops", "MFMAF6Flops"),
    # The roofline shows the F32 and F64 VALU rates only.
    "gfx115x": ("FP16Flops", "BF16Flops"),
}
VARIABLE_FIELDS = ("formula",)
# Ids allowed to share another id's formula (id -> that id): the same counters bounded
# by a different measured ceiling. gfx950 counts MXF8 MFMA work in the F8 counter.
SHARED_FORMULA_IDS = {"compute.mfma_flops_mxf8": "compute.mfma_flops_f8"}

# A row's equation comes from the first of these columns it has; a row with none
# shows one equation in each of its table's value columns.
MAIN_FORMS = ("avg", "value", "expr")
# Value columns that _Row reads separately rather than as equations.
ROW_FIELDS = ("unit", "peak")
ROUND_RE = re.compile(r"^\s*ROUND\((.*),\s*\d+\s*\)\s*$", re.DOTALL)
# Time in seconds (Bytes/s rows) and in nanoseconds (GB/s).
SECONDS_SCALE = "/(SUM(End_Timestamp-Start_Timestamp)/1e9)"
NANOSECONDS = "/SUM(End_Timestamp-Start_Timestamp)"
# Metric unit -> the unit of a row that shows the metric in seconds.
SECONDS_SCALE_UNITS = {"GB/s": "Bytes/s"}
# TODO(AIPROFCOMP-820): remove with the row equation check when the old panel files are
# deleted (see "Equivalence with analysis_configs" below).
# Rows that write a metric's formula differently (positional id -> label, metric id,
# row equation). A row whose time is in seconds needs no entry: it shows a GB/s formula
# scaled by 1e9, and its unit must say Bytes/s.
# The roofline's L2 bandwidth at its bytes per request (64 up to gfx941, then 128).
_ROOFLINE_L2_BW_EQUATION = (
    "{} * SUM(TCP_TCC_WRITE_REQ_sum + TCP_TCC_ATOMIC_WITH_RET_REQ_sum "
    "+ TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum + TCP_TCC_READ_REQ_sum) "
    "/ SUM(End_Timestamp - Start_Timestamp)"
)
_ROOFLINE_L2_BW = (
    "L2 Cache Bandwidth",
    "vl1d.l2_bw",
    _ROOFLINE_L2_BW_EQUATION.format(64),
)
_ROOFLINE_L2_BW_128B = (
    "L2 Cache Bandwidth",
    "vl1d.l2_bw",
    _ROOFLINE_L2_BW_EQUATION.format(128),
)
_ROOFLINE_LDS_BW = (
    "LDS Bandwidth",
    "lds.theoretical_bw",
    "SUM((SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT) * 4 * $lds_banks_per_cu) "
    "/ SUM(End_Timestamp - Start_Timestamp)",
)
_PIPELINE_MFMA_UTIL = (
    "MFMA Utilization",
    "compute.mfma_util",
    "100 * SUM(SQ_VALU_MFMA_BUSY_CYCLES) "
    "/ (4 * SUM($cu_per_gpu * $GRBM_GUI_ACTIVE_PER_XCD))",
)
DISPLAY_VARIANTS = {
    "gfx908": {
        "4.1.9": _ROOFLINE_L2_BW,
        "4.1.11": _ROOFLINE_LDS_BW,
    },
    "gfx90a": {
        "4.1.9": _ROOFLINE_L2_BW,
        "4.1.11": _ROOFLINE_LDS_BW,
        "11.2.7": _PIPELINE_MFMA_UTIL,
    },
    "gfx940": {
        "4.1.10": _ROOFLINE_L2_BW,
        "4.1.12": _ROOFLINE_LDS_BW,
        "11.2.7": _PIPELINE_MFMA_UTIL,
    },
}
DISPLAY_VARIANTS["gfx941"] = DISPLAY_VARIANTS["gfx940"]
DISPLAY_VARIANTS["gfx950"] = {
    "2.1.23": (
        "L2-Fabric Read BW",
        "fabric.rd_bw",
        "SUM(128 * TCC_EA0_RDREQ_128B_sum + 64 * TCC_EA0_RDREQ_64B_sum "
        "+ 32 * TCC_EA0_RDREQ_32B_sum) / SUM(End_Timestamp - Start_Timestamp)",
    ),
    "4.1.12": _ROOFLINE_L2_BW_128B,
    "4.1.14": _ROOFLINE_LDS_BW,
    "11.2.9": _PIPELINE_MFMA_UTIL,
    # Panel 3000 shows each latency as SUM/SUM and as AVG/AVG over the same rows: the
    # counters are summed across CUs (channels), so both give the same value.
    "30.2.6": (
        "L1 latency",
        "vl1d.average_lat",
        "AVG(TCP_TCP_LATENCY_sum) / AVG(TCP_TOTAL_ACCESSES_sum)",
    ),
    "30.2.10": (
        "L2 read latency",
        "vl1d.l2_rd_average_lat",
        "AVG(TCP_TCC_READ_REQ_LATENCY_sum) / AVG(TCP_TCC_READ_REQ_sum)",
    ),
    "30.2.12": (
        "L2 write latency",
        "vl1d.l2_wr_average_lat",
        "AVG(TCP_TCC_WRITE_REQ_LATENCY_sum) / AVG(TCP_TCC_WRITE_REQ_sum)",
    ),
    "30.2.14": (
        "UTCL1 latency",
        "vl1d.utcl1_average_lat",
        "AVG(TCP_CLIENT_UTCL1_INFLIGHT_sum) / AVG(TCP_UTCL1_REQUEST_sum)",
    ),
    "30.2.16": (
        "UTCL1 to UTCL2 latency",
        "vl1d.utcl1_to_utcl2_average_lat",
        "AVG(TCP_UTCL1_UTCL2_INFLIGHT_sum) / AVG(TCP_UTCL1_TRANSLATION_MISS_sum)",
    ),
    "30.8.4": (
        "L2 to EA read latency",
        "fabric.rd_lat",
        "AVG(TCC_EA0_RDREQ_LEVEL_sum) / AVG(TCC_EA0_RDREQ_sum)",
    ),
    "30.8.6": (
        "L2 to EA write latency",
        "fabric.wr_lat",
        "AVG(TCC_EA0_WRREQ_LEVEL_sum) / AVG(TCC_EA0_WRREQ_sum)",
    ),
    "30.8.8": (
        "L2 to EA atomic latency",
        "fabric.atomic_lat",
        "AVG(TCC_EA0_ATOMIC_LEVEL_sum) / AVG(TCC_EA0_ATOMIC_sum)",
    ),
    "30.14.11": (
        "EA read latency",
        "fabric.rd_lat",
        "AVG(TCC_EA0_RDREQ_LEVEL_sum) / AVG(TCC_EA0_RDREQ_sum)",
    ),
    "30.14.13": (
        "EA write latency",
        "fabric.wr_lat",
        "AVG(TCC_EA0_WRREQ_LEVEL_sum) / AVG(TCC_EA0_WRREQ_sum)",
    ),
    "30.14.15": (
        "EA atomic latency",
        "fabric.atomic_lat",
        "AVG(TCC_EA0_ATOMIC_LEVEL_sum) / AVG(TCC_EA0_ATOMIC_sum)",
    ),
    "30.7.1": (
        "L2 utilization",
        "l2.util",
        "100 * SUM(TCC_BUSY_sum) "
        "/ (SUM($GRBM_GUI_ACTIVE_PER_XCD) * TO_INT($total_l2_chan))",
    ),
}
DISPLAY_VARIANTS["gfx942"] = {
    **DISPLAY_VARIANTS["gfx940"],
    "4.1.10": _ROOFLINE_L2_BW_128B,
    # TODO(AIPROFCOMP-820): the clamp is deferred (FOLLOWUPS); the other vL1D hit rate
    # rows have no NOISE_CLAMP.
    "16.3.5": (
        "Cache Hit Rate",
        "vl1d.hit_rate",
        "100 * SUM(NOISE_CLAMP(TCP_TOTAL_CACHE_ACCESSES_sum - (TCP_TCC_READ_REQ_sum "
        "+ TCP_TCC_WRITE_REQ_sum + TCP_TCC_ATOMIC_WITH_RET_REQ_sum "
        "+ TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum), TCP_TOTAL_CACHE_ACCESSES_sum)) "
        "/ SUM(TCP_TOTAL_CACHE_ACCESSES_sum)",
    ),
}
DISPLAY_VARIANTS["gfx115x"] = {
    "4.1.0": (
        "VALU FLOPs",
        "compute.valu_flops",
        "SUM($wave_size * SQ_INSTS_VALU_sum) / SUM(End_Timestamp - Start_Timestamp)",
    ),
    "4.1.1": (
        "VALU FLOPs (F64)",
        "compute.valu_flops_f64",
        "SUM($wave_size * SQ_INSTS_VALU_DP_sum) / SUM(End_Timestamp - Start_Timestamp)",
    ),
}
# Rows whose unit differs from their metric's (positional id -> label, unit).
_CDNA_UNIT_VARIANTS = {"7.1.8": ("LDS Allocation", "Bytes")}
UNIT_VARIANTS = {
    arch: _CDNA_UNIT_VARIANTS
    for arch in ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942")
}
UNIT_VARIANTS["gfx115x"] = {
    "3.3.0": ("TCP Read Requests", "(Requests + $normUnit)"),
    "3.3.1": ("TCP Write Requests", "(Requests + $normUnit)"),
    "3.4.0": ("LDS Req", "(Requests + $normUnit)"),
    "7.2.4": ("LDS Allocation", "Bytes"),
}
UNIT_VARIANTS["gfx950"] = {
    **_CDNA_UNIT_VARIANTS,
    "30.3.6": ("TD - atomic wavefronts", "(Wavefronts + $normUnit)"),
    "30.3.7": ("TD - store wavefronts", "(Wavefronts + $normUnit)"),
    "30.6.30": ("TCP_CLIENT_UTCL1_INFLIGHT", "(Cycles + $normUnit)"),
    "30.6.43": ("TD_LOAD_WAVEFRONT", "(Instructions + $normUnit)"),
    "30.8.3": ("L2 to EA read average latency", "Cycles per Request"),
    "30.8.5": ("L2 to EA write average latency", "Cycles per Request"),
    "30.8.7": ("L2 to EA atomic average latency", "Cycles per Request"),
    "30.14.10": ("EA read average latency", "Cycles per Request"),
    "30.14.12": ("EA write average latency", "Cycles per Request"),
    "30.14.14": ("EA atomic average latency", "Cycles per Request"),
    "30.8.4": ("L2 to EA read latency", "Cycles per Request"),
    "30.8.6": ("L2 to EA write latency", "Cycles per Request"),
    "30.8.8": ("L2 to EA atomic latency", "Cycles per Request"),
    "30.14.11": ("EA read latency", "Cycles per Request"),
    "30.14.13": ("EA write latency", "Cycles per Request"),
    "30.14.15": ("EA atomic latency", "Cycles per Request"),
}


def validate(metric_dir: Path, analysis_dir: Optional[Path] = None) -> list[str]:
    """Validate every arch in a metric library tree. Return error messages.

    With analysis_dir, also check each valid arch against its analysis configs.
    """
    errors: list[str] = []
    for arch in METRIC_LIBRARY_PARENT_ARCH:
        if not (metric_dir / arch).is_dir():
            continue
        arch_errors = validate_arch(metric_dir, arch)
        if not arch_errors and analysis_dir is not None:
            arch_errors = equivalence_errors(metric_dir, analysis_dir, arch)
        errors.extend(arch_errors)
    errors.extend(
        f"[{path.name}] is not a known metric library arch"
        for path in sorted(metric_dir.iterdir())
        if path.is_dir() and path.name not in METRIC_LIBRARY_PARENT_ARCH
    )
    return errors


def validate_arch(metric_dir: Path, arch: str) -> list[str]:
    """Validate one arch's library files."""
    errors: list[str] = []
    metrics: dict[str, Any] = {}
    for name, path in arch_files(metric_dir, arch).items():
        source = f"{path.parent.name}/{name}"
        if path.parent.name == arch:
            errors.extend(_inherit_errors(metric_dir, arch, path, source))
        try:
            document = load_inherited_yaml(path)
        except (InheritanceError, yaml.YAMLError) as error:
            errors.append(f"[{source}] {error}")
            continue
        if name == VARIABLES_FILENAME:
            errors.extend(_variables_errors(document, source))
        else:
            errors.extend(_metrics_errors(document, source, path.stem))
            metrics.update(
                (metric_id, entry)
                for metric_id, entry in document.items()
                if isinstance(entry, dict)
            )
    errors.extend(_ceiling_errors(arch, metrics))
    errors.extend(_shared_formula_errors(arch, metrics))
    errors.extend(_shared_name_errors(arch, metrics))
    return errors


def equivalence_errors(metric_dir: Path, analysis_dir: Path, arch: str) -> list[str]:
    """Check one arch's library against its analysis configs. Temporary; see below."""
    library = load_metric_library(metric_dir, arch)
    rows = list(_row_equations(analysis_dir / arch))
    variants = DISPLAY_VARIANTS.get(arch, {})
    formulas = {
        _canonical(metric["formula"]): metric_id
        for metric_id, metric in library.metrics.items()
        if metric_id not in SHARED_FORMULA_IDS
    }
    names = {
        metric["name"].casefold(): metric_id
        for metric_id, metric in library.metrics.items()
    }
    metric_ids = [
        _row_metric_id(row, variants, formulas, names, set(library.metrics))
        for row in rows
    ]
    return [
        *_row_errors(arch, rows, metric_ids, library),
        *_unshown_errors(arch, rows, metric_ids, library),
        *_unit_errors(arch, rows, metric_ids, library),
        *_peak_errors(arch, rows, metric_ids, library),
        *_counter_errors(arch, library, analysis_dir),
        *_builtin_var_errors(arch, library),
        *_unused_variant_errors(arch, rows),
    ]


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(default_metric_library_dir(), config.analysis_configs_dir)
    if errors:
        print("Metric library validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Metric library validation passed.")
    return 0


# ---------------------------------------------------------------------------
# Library files
# ---------------------------------------------------------------------------


def _inherit_errors(metric_dir: Path, arch: str, path: Path, source: str) -> list[str]:
    parent = METRIC_LIBRARY_PARENT_ARCH[arch]
    parent_path = arch_files(metric_dir, parent).get(path.name) if parent else None
    return _parent_errors(path, _declared_parent(path), parent_path, source)


def _declared_parent(path: Path) -> object:
    """Return the file's "inherit" value, or None if it does not load."""
    try:
        with open(path, encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=InheritLoader)
    except (InheritanceError, yaml.YAMLError):
        return None
    return raw.get(INHERIT_KEY) if isinstance(raw, dict) else None


def _parent_errors(
    path: Path, named: object, parent_path: Optional[Path], source: str
) -> list[str]:
    if parent_path is None:
        return [] if named is None else [f"[{source}] has no parent file to inherit"]
    if (
        not isinstance(named, str)
        or (path.parent / named).resolve() != parent_path.resolve()
    ):
        expected = f"{parent_path.parent.name}/{parent_path.name}"
        return [f"[{source}] must inherit from {expected}"]
    return []


def _variables_errors(document: dict[str, Any], source: str) -> list[str]:
    errors = []
    for name, entry in document.items():
        errors.extend(_field_errors(entry, VARIABLE_FIELDS, (), source, str(name)))
        if isinstance(entry, dict):
            errors.extend(_text_errors(entry, VARIABLE_FIELDS, source, str(name)))
    return errors


def _metrics_errors(document: dict[str, Any], source: str, category: str) -> list[str]:
    errors = []
    for metric_id, entry in document.items():
        item = str(metric_id)
        if not METRIC_LIBRARY_ID_RE.match(item):
            errors.append(
                f"[{source}] {item}: the id must look like "
                "'{category}.{subcategory}_{name}'"
            )
        if item.split(".")[0] in VIEW_CATEGORIES:
            errors.append(
                f"[{source}] {item}: the id names a view; use a hardware category"
            )
        if item.split(".")[0] != category:
            errors.append(
                f"[{source}] {item}: the id must have the file's category '{category}'"
            )
        errors.extend(_metric_errors(entry, source, item))
    return errors


def _metric_errors(entry: object, source: str, item: str) -> list[str]:
    errors = _field_errors(
        entry, METRIC_REQUIRED_FIELDS, METRIC_OPTIONAL_FIELDS, source, item
    )
    if errors:
        return errors
    errors.extend(_text_errors(entry, ("name", "formula"), source, item))
    for field in ("unit", "description"):
        if not isinstance(entry[field], str):
            errors.append(f"[{source}] {item}: '{field}' must be a string")
    errors.extend(
        f"[{source}] {item}: '{field}' must be plain ASCII, without \\u escapes"
        for field in ("name", "unit", "description")
        if isinstance(entry[field], str)
        and (not entry[field].isascii() or "\\u" in entry[field])
    )
    if isinstance(entry["unit"], str) and entry["unit"] in UNIT_ALIASES:
        errors.append(
            f"[{source}] {item}: 'unit' {entry['unit']!r} is an alias; "
            f"use {UNIT_ALIASES[entry['unit']]!r}"
        )
    peak = entry.get("peak")
    if isinstance(peak, bool) or not isinstance(peak, (str, int, float, type(None))):
        errors.append(f"[{source}] {item}: 'peak' must be a string or a number")
    elif EMPIRICAL_PEAK_RE.search(str(peak)):
        errors.append(
            f"[{source}] {item}: 'peak' is a measured peak; use 'peak_empirical'"
        )
    if "peak_empirical" in entry and not EMPIRICAL_PEAK_RE.fullmatch(
        str(entry["peak_empirical"])
    ):
        errors.append(
            f"[{source}] {item}: 'peak_empirical' must be '$<ceiling>_empirical_peak'"
        )
    return errors


def _ceiling_errors(arch: str, metrics: dict[str, Any]) -> list[str]:
    measured = _arch_ceilings(arch)
    without_metric = {
        *CEILINGS_WITHOUT_METRIC,
        *ARCH_CEILINGS_WITHOUT_METRIC.get(arch, ()),
    }
    users = defaultdict(list)
    for metric_id, metric in metrics.items():
        match = EMPIRICAL_PEAK_RE.fullmatch(str(metric.get("peak_empirical")))
        if match:
            users[match.group(1)].append(metric_id)
    errors = []
    for ceiling, metric_ids in sorted(users.items()):
        if ceiling not in measured:
            errors.append(
                f"[{arch}] {metric_ids}: the roofline benchmark does not measure "
                f"{ceiling} on {arch}"
            )
        elif len(metric_ids) > 1:
            errors.append(f"[{arch}] {metric_ids}: share the ceiling {ceiling}")
        elif ceiling in without_metric:
            errors.append(
                f"[{arch}] {metric_ids}: {ceiling} is in CEILINGS_WITHOUT_METRIC"
            )
    errors.extend(
        f"[{arch}] the roofline benchmark measures {ceiling}, but no metric has it as "
        "'peak_empirical'; add it to the metric it bounds, or to "
        "CEILINGS_WITHOUT_METRIC"
        for ceiling in sorted(measured - set(users) - without_metric)
    )
    return errors


def _shared_formula_errors(arch: str, metrics: dict[str, Any]) -> list[str]:
    formulas = {
        metric_id: _canonical(metric.get("formula"))
        for metric_id, metric in metrics.items()
    }
    errors = _shared_value_errors(
        arch,
        formulas,
        "share a formula; merge them into one id",
        [{shared, partner} for shared, partner in SHARED_FORMULA_IDS.items()],
    )
    errors.extend(
        f"[{arch}] {shared}: SHARED_FORMULA_IDS lists it with {partner}, but their "
        "formulas differ"
        for shared, partner in SHARED_FORMULA_IDS.items()
        if shared in formulas
        and partner in formulas
        and formulas[shared] != formulas[partner]
    )
    return errors


def _shared_name_errors(arch: str, metrics: dict[str, Any]) -> list[str]:
    return _shared_value_errors(
        arch,
        {
            metric_id: m["name"].casefold()
            for metric_id, m in metrics.items()
            if isinstance(m.get("name"), str)
        },
        "share a name; name each after what it measures",
    )


def _shared_value_errors(
    arch: str,
    values: dict[str, Optional[str]],
    problem: str,
    allowed_groups: Sequence[set[str]] = (),
) -> list[str]:
    ids_by_value = defaultdict(list)
    for metric_id, value in values.items():
        ids_by_value[value].append(metric_id)
    return [
        f"[{arch}] {metric_ids}: {problem}"
        for value, metric_ids in ids_by_value.items()
        if value is not None
        and len(metric_ids) > 1
        and set(metric_ids) not in allowed_groups
    ]


def _arch_ceilings(arch: str) -> set[str]:
    """Return the ceilings the roofline benchmark measures on any device of arch."""
    return set().union(
        *(
            measured_ceilings(device)
            for device in BENCHMARKING_SUPPORTED
            if canonical_config_arch(device) == arch
        )
    )


# ---------------------------------------------------------------------------
# Equivalence with analysis_configs
#
# TODO(AIPROFCOMP-820): these checks are temporary. Remove them in two steps, as the
# LLD's Stage 3 "Removal" lists:
# - at the cutover (Stage 8): _counter_errors() and _builtin_var_errors(), since
#   detect_counters() and get_build_in_vars() then read the library;
# - when the old panel files are deleted, after the one-release fallback: the rest of
#   this section, DISPLAY_VARIANTS, equivalence_errors() and the analysis_dir argument
#   of validate().
# ---------------------------------------------------------------------------


def _row_metric_id(
    row: "_Row",
    variants: dict[str, tuple],
    formulas: dict[str, str],
    names: dict[str, str],
    library_ids: set[str],
) -> Optional[str]:
    """Return the id of the metric a row shows, or None if it shows none.

    A row without an equation (value: None) is hidden; its label names the metric. A
    row whose label names an id of SHARED_FORMULA_IDS shows that id.
    """
    if row.legacy_id in variants:
        metric_id = variants[row.legacy_id][1]
        return metric_id if metric_id in library_ids else None
    named_id = names.get(row.label.casefold())
    if row.equation is None:
        return named_id
    metric_id = formulas.get(_gb_per_second(row))
    if metric_id is not None and SHARED_FORMULA_IDS.get(named_id) == metric_id:
        return named_id
    return metric_id


def _row_errors(
    arch: str,
    rows: list["_Row"],
    metric_ids: list[Optional[str]],
    library: MetricLibrary,
) -> list[str]:
    variants = DISPLAY_VARIANTS.get(arch, {})
    errors = []
    for row, metric_id in zip(rows, metric_ids):
        if row.legacy_id in variants:
            errors.extend(_variant_errors(arch, row, variants[row.legacy_id], library))
        elif row.equation is not None and metric_id is None:
            errors.append(
                f"[{arch}] {row.legacy_id} {row.label}: not a library formula"
            )
    return errors


def _variant_errors(
    arch: str, row: "_Row", variant: tuple, library: MetricLibrary
) -> list[str]:
    label, metric_id, row_equation = variant
    if row.label != label:
        return [f"[{arch}] {row.legacy_id} is now {row.label!r}; update variants"]
    if metric_id not in library.metrics:
        return [f"[{arch}] {row.legacy_id} {label}: no metric {metric_id}"]
    if row.equation == _canonical(row_equation):
        return []
    return [f"[{arch}] {row.legacy_id} {label}: not the listed form of {metric_id}"]


def _unused_variant_errors(arch: str, rows: list["_Row"]) -> list[str]:
    """Report DISPLAY_VARIANTS and UNIT_VARIANTS entries at a position no row has."""
    positions = {row.legacy_id for row in rows}
    return [
        f"[{arch}] {name} lists {legacy_id}, but no row has that position"
        for name, listed in (
            ("DISPLAY_VARIANTS", DISPLAY_VARIANTS.get(arch, {})),
            ("UNIT_VARIANTS", UNIT_VARIANTS.get(arch, {})),
        )
        for legacy_id in listed
        if legacy_id not in positions
    ]


def _unshown_errors(
    arch: str,
    rows: list["_Row"],
    metric_ids: list[Optional[str]],
    library: MetricLibrary,
) -> list[str]:
    shown = {
        metric_id
        for row, metric_id in zip(rows, metric_ids)
        if row.equation is not None
    }
    return [
        f"[{arch}] {metric_id}: no analysis config row shows its formula"
        for metric_id in library.metrics
        if metric_id not in shown
    ]


def _unit_errors(
    arch: str,
    rows: list["_Row"],
    metric_ids: list[Optional[str]],
    library: MetricLibrary,
) -> list[str]:
    unit_variants = UNIT_VARIANTS.get(arch, {})
    errors = []
    for row, metric_id in zip(rows, metric_ids):
        if row.unit is None or metric_id is None:
            continue
        label, unit = unit_variants.get(
            row.legacy_id, (row.label, library.metrics[metric_id]["unit"])
        )
        if _in_seconds(row):
            unit = SECONDS_SCALE_UNITS.get(unit, unit)
        if row.label != label:
            errors.append(
                f"[{arch}] {row.legacy_id} is now {row.label!r}; update unit variants"
            )
        elif UNIT_ALIASES.get(row.unit, row.unit) != unit:
            errors.append(
                f"[{arch}] {row.legacy_id} {row.label}: unit {row.unit!r} is not "
                f"{unit!r} ({metric_id})"
            )
    return errors


def _peak_errors(
    arch: str,
    rows: list["_Row"],
    metric_ids: list[Optional[str]],
    library: MetricLibrary,
) -> list[str]:
    """Check that row peaks and library peaks match, both ways.

    Each row's peak is its metric's "peak" or "peak_empirical", and each library peak
    is the peak of a row that shows the metric. A Bytes/s row shows its peak in Bytes/s.
    A measured peak on an arch the roofline benchmark does not run on is skipped.
    """
    benchmarked = bool(_arch_ceilings(arch))
    errors = []
    shown_fields = defaultdict(set)
    for row, metric_id in zip(rows, metric_ids):
        if row.peak is None or metric_id is None:
            continue
        if not benchmarked and EMPIRICAL_PEAK_RE.search(row.peak):
            continue
        metric = library.metrics[metric_id]
        fields = [
            field
            for field in ("peak", "peak_empirical")
            if field in metric and row.peak == _row_peak(row, _canonical(metric[field]))
        ]
        shown_fields[metric_id].update(fields)
        if not fields:
            errors.append(
                f"[{arch}] {row.legacy_id} {row.label}: peak {row.peak} is neither the "
                f"'peak' nor the 'peak_empirical' of {metric_id}"
            )
    errors.extend(
        f"[{arch}] {metric_id}: no analysis config row shows its '{field}'"
        for metric_id, metric in library.metrics.items()
        for field in ("peak", "peak_empirical")
        if field in metric and field not in shown_fields[metric_id]
    )
    return errors


def _gb_per_second(row: "_Row") -> Optional[str]:
    """Return a row's equation, with time in seconds turned into nanoseconds (GB/s)."""
    if _in_seconds(row):
        return row.equation[: -len(SECONDS_SCALE)] + NANOSECONDS
    return row.equation


def _in_seconds(row: "_Row") -> bool:
    """Return whether a row divides by the kernel time in seconds (a Bytes/s row)."""
    return row.equation is not None and row.equation.endswith(SECONDS_SCALE)


def _row_peak(row: "_Row", peak: str) -> str:
    """Return a library peak (GB/s) as row shows it.

    A Bytes/s row shows a theoretical "<MHz term> / 1000" as "<MHz term> * 1e6", and a
    measured "$<ceiling>_empirical_peak" as "... * 1e9".
    """
    if not _in_seconds(row):
        return peak
    if peak.endswith("/1000"):
        return peak[: -len("/1000")] + "*1e6"
    return f"{peak}*1e9"


def _counter_errors(arch: str, library: MetricLibrary, analysis_dir: Path) -> list[str]:
    has_membw_panel = any(
        (analysis_dir / arch).glob(f"{MEMBW_ANALYSIS_PANEL_ID}_*.yaml")
    )
    gpu_series = _gpu_series(arch)
    soc = _make_soc(arch, gpu_series, analysis_dir, has_membw_panel)
    expected, _ = soc.detect_counters()
    actual = soc._expand_tcc_template_counters(
        library.get_counters_for_metrics(list(library.metrics), gpu_series)
    )
    errors = []
    if expected - actual:
        errors.append(f"[{arch}] counters missing: {sorted(expected - actual)}")
    if actual - expected:
        errors.append(f"[{arch}] extra counters: {sorted(actual - expected)}")
    return errors


def _builtin_var_errors(arch: str, library: MetricLibrary) -> list[str]:
    actual = {
        name: _canonical(variable["formula"])
        for name, variable in library.variables.items()
    }
    gpu_series = _gpu_series(arch)
    expected = {
        name: _canonical(formula)
        for name, formula in get_build_in_vars(gpu_series).items()
    }
    if actual == expected:
        return []
    return [f"[{arch}] variables differ from get_build_in_vars()"]


def _row_equations(arch_dir: Path) -> Iterator["_Row"]:
    """Yield each equation a row of arch_dir shows."""
    for panel in load_panel_configs([str(arch_dir)]).values():
        for source in panel["data source"]:
            table = source.get("metric_table")
            if table is None:
                continue
            labels = [
                label for label in table["metric"] if label != "placeholder_range"
            ]
            for index, label in enumerate(labels):
                row = table["metric"][label]
                legacy_id = f"{table['id'] // 100}.{table['id'] % 100}.{index}"
                for form in _row_forms(table, row):
                    yield _Row(
                        legacy_id,
                        label,
                        _canonical(row[form]),
                        _canonical(row.get("peak")),
                        row.get("unit"),
                    )


def _row_forms(table: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """Return the columns of row that show an equation."""
    main_forms = [form for form in MAIN_FORMS if form in row]
    if main_forms:
        return main_forms[:1]
    return [
        key for key in row if key in metric_value_keys(table) and key not in ROW_FIELDS
    ]


def _canonical(equation: object) -> Optional[str]:
    """Return equation in format_yaml.py form without display ROUND(); None if empty.

    Many analysis_configs rows are unformatted (format_yaml.py skips AUTOGENERATED).
    """
    if equation is None or str(equation) == "None":
        return None
    match = ROUND_RE.match(str(equation))
    text = normalize_equation(match.group(1) if match else str(equation))
    return re.sub(r"\s+", "", text)


def _make_soc(
    arch: str, gpu_series: Optional[str], analysis_dir: Path, membw_analysis: bool
) -> OmniSoC_Base:
    args = SimpleNamespace(
        config_dir=analysis_dir,
        filter_blocks=[],
        membw_analysis=membw_analysis,
        roof_only=False,
        set_selected=None,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=_member_arch(arch),
        gpu_series=gpu_series,
        l2_banks=16,
        num_xcd=1,
        rocminfo_lines=None,
    )
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(_member_arch(arch))
    return soc


def _gpu_series(arch: str) -> Optional[str]:
    return mi_gpu_specs.get_gpu_series(_member_arch(arch))


def _member_arch(arch: str) -> str:
    """Return a GPU arch that uses arch's analysis configs (gfx1151 for gfx115x)."""
    return next(
        gpu_arch
        for gpu_arch in sorted(mi_gpu_specs.get_gpu_series_dict())
        if canonical_config_arch(gpu_arch) == arch
    )


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _field_errors(
    entry: object,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    source: str,
    item: str,
) -> list[str]:
    if not isinstance(entry, dict):
        return [f"[{source}] {item}: must be a mapping"]
    errors = [
        f"[{source}] {item}: '{field}' is required"
        for field in required
        if field not in entry
    ]
    errors.extend(
        f"[{source}] {item}: '{field}' is not a known field"
        for field in entry
        if field not in required and field not in optional
    )
    return errors


def _text_errors(
    entry: dict[str, Any], fields: tuple[str, ...], source: str, item: str
) -> list[str]:
    return [
        f"[{source}] {item}: '{field}' must be a non-empty string"
        for field in fields
        if not isinstance(entry.get(field), str) or not entry[field].strip()
    ]


class _Row(NamedTuple):
    """One equation an analysis config row shows."""

    legacy_id: str
    label: str
    equation: Optional[str]
    peak: Optional[str]
    unit: Optional[str]


if __name__ == "__main__":
    sys.exit(main())
