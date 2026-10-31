#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: validate the metric library.

For every arch in src/rocprof_compute_soc/metric_library/, check that:
  1. each file loads, and a child file inherits from its parent arch's file;
  2. metrics and variables have the required fields and types; "peak" is a
     theoretical peak and "peak_empirical" a roofline benchmark ceiling;
  3. every ceiling the roofline benchmark measures on the arch is the
     "peak_empirical" of exactly one metric, or is in CEILINGS_WITHOUT_METRIC or the
     arch's ARCH_CEILINGS_WITHOUT_METRIC, and each entry there is measured;
  4. metric ids look like "{category}.{subcategory}_{name}", name hardware rather than
     a view, and live in the file named after their category, so each id is unique on
     the arch;
  5. no two metrics on the arch share a formula (except SHARED_FORMULA_IDS) or a name,
     and a formula has one id on every arch (except SPLIT_FORMULA_IDS); each set has a
     title and a list of the arch's metric ids, and fits in one profiling pass on
     every GPU arch it serves;
  6. units use one spelling, and each metric id has one name and one unit on every
     arch;
  7. the library matches analysis_configs: every equation a row shows is a library
     formula (DISPLAY_VARIANTS list the rows that show one in another form), every
     formula is shown, every row's unit is its metric's unit (except UNIT_VARIANTS),
     each row's peak is its metric's "peak" or "peak_empirical" and each library peak
     is shown by a row, the no-filter counters equal detect_counters(), and the
     variables equal get_build_in_vars(); each set of profile_configs/sets is a library
     set (except SETS_WITHOUT_METRICS) that collects the same counters as `--set`.
     Check 7 is temporary; see "Equivalence with analysis_configs" below.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from rocprof_compute_soc.soc_base import OmniSoC_Base, iter_analysis_config_metrics
from utils import metric_library
from utils.analysis_views import (
    DISPLAY_FORMS,
    MAIN_COLUMNS,
    NANOSECONDS_DENOMINATOR,
    ROUND_RE,
    SECONDS_DENOMINATOR,
    SECONDS_SCALED_UNITS,
    UNIT_ALIASES,
)
from utils.equation_format import normalize_equation
from utils.metric_library import (
    SETS_FILENAME,
    VARIABLES_FILENAME,
    MetricLibrary,
    arch_files,
    default_metric_library_dir,
    iter_library_documents,
    load_metric_library,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import (
    MEMBW_ANALYSIS_PANEL_ID,
    canonical_config_arch,
    parse_sets_yaml,
)
from utils.utils_counter_defs import (
    counters_over_block_limits,
    extract_counters_and_variables,
    get_build_in_vars,
    metric_value_keys,
)
from utils.yaml_inherit_loader import InheritanceError, declared_parent
from vendored import yaml

# Categories named after panels rather than hardware.
VIEW_CATEGORIES = ("system", "memchart", "roofline", "wavefront")

# A metric id, "{category}.{subcategory}_{name}": lowercase words joined by
# underscores. utils_common.METRIC_ID_RE matches the numeric panel ids instead.
METRIC_LIBRARY_ID_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z0-9]+(_[a-z0-9]+)*$")
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
    # The VALU rows show theoretical peaks.
    "gfx1250": ("FP16Flops", "FP32Flops", "FP64Flops"),
}
VARIABLE_FIELDS = ("formula",)
SET_FIELDS = ("title", "metrics")
SET_OPTION_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
# Sets of profile_configs/sets with no library set, by arch: no row of the set shows an
# equation on the arch, so `--set` collects only the normalization counters. gfx908 has
# no MFMA counters.
SETS_WITHOUT_METRICS = {"gfx908": ("compute_thruput_flops",)}
# Ids allowed to share another id's formula (id -> that id): the same counters bounded
# by a different measured ceiling. gfx950 and gfx1250 count MXF8 matrix work in the F8
# counter.
SHARED_FORMULA_IDS = {
    "compute.mfma_flops_mxf8": "compute.mfma_flops_f8",
    "compute.wmma_flops_mxfp8": "compute.wmma_flops_fp8",
}
# Ids that hold another id's formula on a different arch (id -> that id): one label
# shows two equations on gfx940 and gfx941 (D8 Kind 3), and the parent arch's equation
# keeps the id until the metric owners choose one.
SPLIT_FORMULA_IDS = {
    "fabric.rd_bw_incl_128b": "fabric.rd_bw",
    "vl1d.cache_bw_128b": "vl1d.cache_bw",
}

# Value columns that _Row reads separately rather than as equations.
ROW_FIELDS = ("unit", "peak")
# Time in seconds (Bytes/s rows) and in nanoseconds (GB/s), in _canonical() form.
SECONDS_SCALE = re.sub(r"\s+", "", SECONDS_DENOMINATOR)
NANOSECONDS = re.sub(r"\s+", "", NANOSECONDS_DENOMINATOR)
# TODO(AIPROFCOMP-820): remove with the row equation check when the old panel files are
# deleted (see "Equivalence with analysis_configs" below).
# Rows that show a display form of their metric (positional id -> label, metric id,
# DISPLAY_FORMS name). A row whose time is in seconds needs no entry: it shows a GB/s
# formula scaled by 1e9, and its unit must say Bytes/s.
_ROOFLINE_L2_BW = ("L2 Cache Bandwidth", "vl1d.l2_bw", "roofline_l2_bw_64b")
_ROOFLINE_L2_BW_128B = ("L2 Cache Bandwidth", "vl1d.l2_bw", "roofline_l2_bw_128b")
_ROOFLINE_LDS_BW = ("LDS Bandwidth", "lds.theoretical_bw", "roofline_lds_bw")
_PIPELINE_MFMA_UTIL = ("MFMA Utilization", "compute.mfma_util", "pipeline_mfma_util")
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
    "2.1.23": ("L2-Fabric Read BW", "fabric.rd_bw", "fabric_rd_bw_by_request_size"),
    "4.1.12": _ROOFLINE_L2_BW_128B,
    "4.1.14": _ROOFLINE_LDS_BW,
    "11.2.9": _PIPELINE_MFMA_UTIL,
    "30.2.6": ("L1 latency", "vl1d.average_lat", "avg_vl1d_average_lat"),
    "30.2.10": (
        "L2 read latency",
        "vl1d.l2_rd_average_lat",
        "avg_vl1d_l2_rd_average_lat",
    ),
    "30.2.12": (
        "L2 write latency",
        "vl1d.l2_wr_average_lat",
        "avg_vl1d_l2_wr_average_lat",
    ),
    "30.2.14": (
        "UTCL1 latency",
        "vl1d.utcl1_average_lat",
        "avg_vl1d_utcl1_average_lat",
    ),
    "30.2.16": (
        "UTCL1 to UTCL2 latency",
        "vl1d.utcl1_to_utcl2_average_lat",
        "avg_vl1d_utcl1_to_utcl2_average_lat",
    ),
    "30.8.4": ("L2 to EA read latency", "fabric.rd_lat", "avg_fabric_rd_lat"),
    "30.8.6": ("L2 to EA write latency", "fabric.wr_lat", "avg_fabric_wr_lat"),
    "30.8.8": ("L2 to EA atomic latency", "fabric.atomic_lat", "avg_fabric_atomic_lat"),
    "30.14.11": ("EA read latency", "fabric.rd_lat", "avg_fabric_rd_lat"),
    "30.14.13": ("EA write latency", "fabric.wr_lat", "avg_fabric_wr_lat"),
    "30.14.15": ("EA atomic latency", "fabric.atomic_lat", "avg_fabric_atomic_lat"),
    "30.7.1": ("L2 utilization", "l2.util", "l2_util_whole_channels"),
}
DISPLAY_VARIANTS["gfx942"] = {
    **DISPLAY_VARIANTS["gfx940"],
    "4.1.10": _ROOFLINE_L2_BW_128B,
    # TODO(AIPROFCOMP-820): the clamp is deferred (FOLLOWUPS); the other vL1D hit rate
    # rows have no NOISE_CLAMP.
    "16.3.5": ("Cache Hit Rate", "vl1d.hit_rate", "noise_clamped_vl1d_hit_rate"),
}
DISPLAY_VARIANTS["gfx115x"] = {
    "4.1.0": ("VALU FLOPs", "compute.valu_flops", "valu_flops_per_valu_inst"),
    "4.1.1": (
        "VALU FLOPs (F64)",
        "compute.valu_flops_f64",
        "valu_flops_f64_per_dp_inst",
    ),
}
DISPLAY_VARIANTS["gfx1250"] = {
    "3.3.4": ("GL0 Hit Rate", "gl0.hit_rate", "gl0_hit_rate_times_100"),
}
# Rows whose unit differs from their metric's (positional id -> label, unit).
_CDNA_UNIT_VARIANTS = {"7.1.8": ("LDS Allocation", "Bytes")}
UNIT_VARIANTS = {arch: _CDNA_UNIT_VARIANTS for arch in ("gfx908", "gfx90a", "gfx942")}
# gfx940 and gfx941 label integer ops in FLOPs.
_GFX940_UNIT_VARIANTS = {
    **_CDNA_UNIT_VARIANTS,
    "11.3.1": ("IOPs (Total)", "(FLOPs + $normUnit)"),
    "11.3.7": ("INT8 OPs", "(FLOPs + $normUnit)"),
}
UNIT_VARIANTS["gfx940"] = _GFX940_UNIT_VARIANTS
UNIT_VARIANTS["gfx941"] = _GFX940_UNIT_VARIANTS
UNIT_VARIANTS["gfx115x"] = {
    "3.4.0": ("LDS Req", "(Requests + $normUnit)"),
    "7.2.4": ("LDS Allocation", "Bytes"),
    "9.2.0": ("Total Requests", "(Count + $normUnit)"),
    "9.2.1": ("Read Requests", "(Count + $normUnit)"),
    "9.2.2": ("Write Requests", "(Count + $normUnit)"),
    "9.4.0": ("GL1 Read Requests", "(Count + $normUnit)"),
    "9.4.2": ("GL1 Write Requests", "(Count + $normUnit)"),
    "11.2.0": ("Total Requests", "(Count + $normUnit)"),
    "11.2.1": ("Read Requests", "(Count + $normUnit)"),
    "11.2.2": ("Write Requests", "(Count + $normUnit)"),
    "11.4.0": ("GL2 Read Requests", "(Count + $normUnit)"),
    "11.4.1": ("GL2 Read 128B Requests", "(Count + $normUnit)"),
    "11.4.2": ("GL2 Write Requests", "(Count + $normUnit)"),
}
UNIT_VARIANTS["gfx1250"] = {
    "2.1.17": ("Active WGPs", "WGPs per Cycle"),
    "3.4.0": ("LDS Load Requests", "(Requests + $normUnit)"),
    "3.4.1": ("LDS Store Requests", "(Requests + $normUnit)"),
    "3.4.2": ("LDS Atomic Requests", "(Requests + $normUnit)"),
    "3.4.3": ("Async Load Requests", "(Requests + $normUnit)"),
    "3.4.4": ("Async Store Requests", "(Requests + $normUnit)"),
    "3.4.14": ("TDM Load Requests", "(Requests + $normUnit)"),
    "3.4.15": ("TDM Store Requests", "(Requests + $normUnit)"),
    "6.2.0": ("Dispatched Workgroups", "Count"),
    "6.2.1": ("Dispatched Wavefronts", "Count"),
    "7.2.5": ("LDS Allocation", "Bytes"),
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


def validate(metric_dir: Path, analysis_dir: Optional[Path] = None) -> List[str]:
    """Validate every arch in a metric library tree. Return error messages.

    With analysis_dir, also check each valid arch against its analysis configs.
    """
    errors: List[str] = []
    valid_archs = []
    for arch in metric_library.METRIC_LIBRARY_PARENT_ARCH:
        if not (metric_dir / arch).is_dir():
            continue
        arch_errors = validate_arch(metric_dir, arch)
        if not arch_errors:
            valid_archs.append(arch)
            if analysis_dir is not None:
                arch_errors = equivalence_errors(metric_dir, analysis_dir, arch)
        errors.extend(arch_errors)
    errors.extend(_formula_id_errors(metric_dir, valid_archs))
    errors.extend(_name_unit_errors(metric_dir, valid_archs))
    errors.extend(
        f"[{path.name}] is not a known metric library arch"
        for path in sorted(metric_dir.iterdir())
        if path.is_dir() and path.name not in metric_library.METRIC_LIBRARY_PARENT_ARCH
    )
    return errors


def validate_arch(metric_dir: Path, arch: str) -> List[str]:
    """Validate one arch's library files."""
    errors = [
        error
        for path in arch_files(metric_dir, arch).values()
        if path.parent.name == arch
        for error in _inherit_errors(metric_dir, arch, path)
    ]
    load_errors: Dict[Path, Exception] = {}
    metrics: Dict[str, Any] = {}
    sets: Dict[str, Any] = {}
    for path, document in iter_library_documents(metric_dir, arch, load_errors):
        if path.name == VARIABLES_FILENAME:
            errors.extend(_variables_errors(document, _source(path)))
            continue
        if path.name == SETS_FILENAME:
            errors.extend(_sets_errors(document, _source(path)))
            sets = document
            continue
        errors.extend(_metrics_errors(document, _source(path), path.stem))
        metrics.update(
            (metric_id, entry)
            for metric_id, entry in document.items()
            if isinstance(entry, dict)
        )
    errors.extend(f"[{_source(path)}] {error}" for path, error in load_errors.items())
    errors.extend(_ceiling_errors(arch, metrics))
    errors.extend(_shared_formula_errors(arch, metrics))
    errors.extend(_shared_name_errors(arch, metrics))
    if not errors:
        errors.extend(_set_metric_errors(arch, sets, metrics))
    return errors


def equivalence_errors(metric_dir: Path, analysis_dir: Path, arch: str) -> List[str]:
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
        *_set_counter_errors(arch, library, analysis_dir),
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


def _source(path: Path) -> str:
    """Return the "<arch>/<file>" name of a library file used in error messages."""
    return f"{path.parent.name}/{path.name}"


def _inherit_errors(metric_dir: Path, arch: str, path: Path) -> List[str]:
    """Check that a file of arch inherits from its parent arch's file of that name."""
    parent = metric_library.METRIC_LIBRARY_PARENT_ARCH[arch]
    expected = arch_files(metric_dir, parent).get(path.name) if parent else None
    try:
        declared = declared_parent(path)
    except (InheritanceError, yaml.YAMLError):
        declared = None  # validate_arch reports the load error
    if expected is None:
        if declared is None:
            return []
        return [f"[{_source(path)}] has no parent file to inherit"]
    if declared != expected.resolve():
        return [f"[{_source(path)}] must inherit from {_source(expected)}"]
    return []


def _variables_errors(document: Dict[str, Any], source: str) -> List[str]:
    errors = []
    for name, entry in document.items():
        errors.extend(_field_errors(entry, VARIABLE_FIELDS, (), source, str(name)))
        if isinstance(entry, dict):
            errors.extend(_text_errors(entry, VARIABLE_FIELDS, source, str(name)))
    return errors


def _sets_errors(document: Dict[str, Any], source: str) -> List[str]:
    errors = []
    for set_option, entry in document.items():
        item = str(set_option)
        if not SET_OPTION_RE.match(item):
            errors.append(f"[{source}] {item}: a set option must be snake_case")
        field_errors = _field_errors(entry, SET_FIELDS, (), source, item)
        errors.extend(field_errors)
        if field_errors:
            continue
        errors.extend(_text_errors(entry, ("title",), source, item))
        metric_ids = entry["metrics"]
        if (
            not isinstance(metric_ids, list)
            or not metric_ids
            or not all(isinstance(metric_id, str) for metric_id in metric_ids)
        ):
            errors.append(f"[{source}] {item}: 'metrics' must list metric ids")
        elif len(set(metric_ids)) != len(metric_ids):
            errors.append(f"[{source}] {item}: 'metrics' lists an id twice")
    return errors


def _set_metric_errors(
    arch: str, sets: Dict[str, Any], metrics: Dict[str, Any]
) -> List[str]:
    """Check that each set lists the arch's metrics and fits in one pass."""
    errors = []
    library = MetricLibrary(metrics, {}, sets)
    for set_option, entry in sets.items():
        unknown = [
            metric_id for metric_id in entry["metrics"] if metric_id not in metrics
        ]
        if unknown:
            errors.append(f"[{arch}] set {set_option}: unknown metric ids {unknown}")
            continue
        for gpu_arch in _member_archs(arch):
            limits = mi_gpu_specs.get_perfmon_config(gpu_arch)
            counters = library.get_counters_for_metrics(
                entry["metrics"], mi_gpu_specs.get_gpu_series(gpu_arch)
            )
            errors.extend(
                f"[{gpu_arch}] set {set_option}: block {block} needs "
                f"{len(block_counters)} counters but one pass holds {limits[block]}"
                for block, block_counters in counters_over_block_limits(
                    counters, limits
                ).items()
            )
    return errors


def _metrics_errors(document: Dict[str, Any], source: str, category: str) -> List[str]:
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


def _metric_errors(entry: object, source: str, item: str) -> List[str]:
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


def _ceiling_errors(arch: str, metrics: Dict[str, Any]) -> List[str]:
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
                f"[{arch}] {metric_ids}: {ceiling} is listed as a ceiling without a "
                "metric"
            )
    errors.extend(
        f"[{arch}] the roofline benchmark measures {ceiling}, but no metric has it as "
        "'peak_empirical'; add it to the metric it bounds, or to "
        "CEILINGS_WITHOUT_METRIC or ARCH_CEILINGS_WITHOUT_METRIC"
        for ceiling in sorted(measured - set(users) - without_metric)
    )
    errors.extend(
        f"[{arch}] ARCH_CEILINGS_WITHOUT_METRIC lists {ceiling}, which the roofline "
        f"benchmark does not measure on {arch}; remove it"
        for ceiling in ARCH_CEILINGS_WITHOUT_METRIC.get(arch, ())
        if ceiling not in measured
    )
    return errors


def _shared_formula_errors(arch: str, metrics: Dict[str, Any]) -> List[str]:
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


def _shared_name_errors(arch: str, metrics: Dict[str, Any]) -> List[str]:
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
    values: Dict[str, Optional[str]],
    problem: str,
    allowed_groups: Sequence[Set[str]] = (),
) -> List[str]:
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


def _formula_id_errors(metric_dir: Path, archs: List[str]) -> List[str]:
    """Check that each formula has one id on every arch (D8).

    Apart from SHARED_FORMULA_IDS, the archs' own checks leave only ids that share a
    formula on different archs: a quantity a later arch renamed. SPLIT_FORMULA_IDS lists
    the pairs kept on purpose.
    """
    allowed_groups = [
        {metric_id, partner}
        for metric_id, partner in {**SHARED_FORMULA_IDS, **SPLIT_FORMULA_IDS}.items()
    ]
    archs_by_formula = _archs_by_value(
        metric_dir, archs, lambda metric_id, m: (_canonical(m["formula"]), metric_id)
    )
    return [
        f"{sorted(found)}: share a formula on different archs ("
        + "; ".join(
            f"{metric_id} on {', '.join(on)}" for metric_id, on in found.items()
        )
        + "); use one id on every arch"
        for found in archs_by_formula.values()
        if len(found) > 1 and set(found) not in allowed_groups
    ]


def _name_unit_errors(metric_dir: Path, archs: List[str]) -> List[str]:
    """Check that each metric id has one name and one unit on every arch (D8)."""
    values = _archs_by_value(
        metric_dir, archs, lambda metric_id, m: (metric_id, (m["name"], m["unit"]))
    )
    return [
        f"{metric_id}: one name and unit on every arch, not "
        + "; ".join(f"{name!r} [{unit}] on {on}" for (name, unit), on in found.items())
        for metric_id, found in sorted(values.items())
        if len(found) > 1
    ]


def _archs_by_value(
    metric_dir: Path,
    archs: List[str],
    key: Callable[[str, Dict[str, Any]], Tuple[Hashable, Hashable]],
) -> Dict[Hashable, Dict[Hashable, List[str]]]:
    """Return group -> value -> the archs whose metrics key() maps to (group, value)."""
    groups = defaultdict(lambda: defaultdict(list))
    for arch in archs:
        for metric_id, metric in load_metric_library(metric_dir, arch).metrics.items():
            group, value = key(metric_id, metric)
            groups[group][value].append(arch)
    return groups


def _arch_ceilings(arch: str) -> Set[str]:
    """Return the ceilings the roofline benchmark measures on any device of arch."""
    return set().union(
        *(
            mi_gpu_specs.get_roofline_columns(device)
            for device in mi_gpu_specs.get_roofline_archs()
            if canonical_config_arch(device) == arch
        )
    )


# ---------------------------------------------------------------------------
# Equivalence with analysis_configs
#
# TODO(AIPROFCOMP-820): these checks are temporary. Remove them at the cutover
# (Stage 8), with the old panel files, as the LLD's Stage 3 "Removal" says: this
# section, DISPLAY_VARIANTS, UNIT_VARIANTS, equivalence_errors() and the analysis_dir
# argument of validate().
# ---------------------------------------------------------------------------


def _row_metric_id(
    row: "_Row",
    variants: Dict[str, tuple],
    formulas: Dict[str, str],
    names: Dict[str, str],
    library_ids: Set[str],
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
    rows: List["_Row"],
    metric_ids: List[Optional[str]],
    library: MetricLibrary,
) -> List[str]:
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
) -> List[str]:
    label, metric_id, form = variant
    if row.label != label:
        return [f"[{arch}] {row.legacy_id} is now {row.label!r}; update variants"]
    if metric_id not in library.metrics:
        return [f"[{arch}] {row.legacy_id} {label}: no metric {metric_id}"]
    if row.equation == _canonical(DISPLAY_FORMS[form]["formula"]):
        return []
    return [f"[{arch}] {row.legacy_id} {label}: not the listed form of {metric_id}"]


def _unused_variant_errors(arch: str, rows: List["_Row"]) -> List[str]:
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
    rows: List["_Row"],
    metric_ids: List[Optional[str]],
    library: MetricLibrary,
) -> List[str]:
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
    rows: List["_Row"],
    metric_ids: List[Optional[str]],
    library: MetricLibrary,
) -> List[str]:
    unit_variants = UNIT_VARIANTS.get(arch, {})
    errors = []
    for row, metric_id in zip(rows, metric_ids):
        if row.unit is None or metric_id is None:
            continue
        metric_unit = library.metrics[metric_id]["unit"]
        label, unit = unit_variants.get(row.legacy_id, (row.label, metric_unit))
        if row.legacy_id in unit_variants and unit == metric_unit:
            errors.append(
                f"[{arch}] {row.legacy_id} {row.label}: the unit of {metric_id}; "
                "remove it from UNIT_VARIANTS"
            )
            continue
        if _in_seconds(row):
            unit = SECONDS_SCALED_UNITS.get(unit, unit)
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
    rows: List["_Row"],
    metric_ids: List[Optional[str]],
    library: MetricLibrary,
) -> List[str]:
    """Check that row peaks and library peaks match, both ways.

    Each row's peak is its metric's "peak" or "peak_empirical", and each library peak
    is the peak of a row that shows the metric. A Bytes/s row shows its peak in Bytes/s.
    """
    errors = []
    shown_fields = defaultdict(set)
    for row, metric_id in zip(rows, metric_ids):
        if row.peak is None or metric_id is None:
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


def _make_soc(
    analysis_dir: Path,
    arch: str,
    membw_analysis: bool,
    set_selected: Optional[str] = None,
) -> OmniSoC_Base:
    """Return an OmniSoC_Base that reads analysis_dir without a GPU (16 L2 channels),
    with set_selected as the --set option, if any."""
    args = SimpleNamespace(
        config_dir=analysis_dir,
        filter_blocks=[],
        membw_analysis=membw_analysis,
        roof_only=False,
        set_selected=set_selected,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=arch,
        gpu_series=mi_gpu_specs.get_gpu_series(arch),
        l2_banks=16,
        num_xcd=1,
        rocminfo_lines=None,
    )
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(arch)
    return soc


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


def _counter_errors(arch: str, library: MetricLibrary, analysis_dir: Path) -> List[str]:
    has_membw_panel = any(
        (analysis_dir / arch).glob(f"{MEMBW_ANALYSIS_PANEL_ID}_*.yaml")
    )
    gpu_series = _gpu_series(arch)
    soc = _make_soc(analysis_dir, _member_arch(arch), has_membw_panel)
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


def _set_counter_errors(
    arch: str, library: MetricLibrary, analysis_dir: Path
) -> List[str]:
    """Compare each library set with `--set` on profile_configs/sets."""
    gpu_series = _gpu_series(arch)
    without_metrics = SETS_WITHOUT_METRICS.get(arch, ())
    shipped = set(parse_sets_yaml(_member_arch(arch)))
    # Every collection includes the normalization counters.
    no_metric_counters, _ = extract_counters_and_variables("", gpu_series)
    errors = []
    if set(library.sets) != shipped - set(without_metrics):
        errors.append(
            f"[{arch}] sets differ from profile_configs/sets: "
            f"missing {sorted(shipped - set(without_metrics) - set(library.sets))}, "
            f"extra {sorted(set(library.sets) - shipped)}"
        )
    for set_option in sorted(shipped & {*library.sets, *without_metrics}):
        soc = _make_soc(analysis_dir, _member_arch(arch), False, set_option)
        expected, _ = soc.detect_counters()
        if set_option in without_metrics:
            if expected - no_metric_counters or set_option in library.sets:
                errors.append(
                    f"[{arch}] set {set_option} is in SETS_WITHOUT_METRICS but has "
                    "metrics; remove the entry"
                )
            continue
        actual = soc._expand_tcc_template_counters(
            library.get_counters_for_metrics(
                library.sets[set_option]["metrics"], gpu_series
            )
        )
        if actual != expected:
            errors.append(
                f"[{arch}] set {set_option}: counters missing "
                f"{sorted(expected - actual)}, extra {sorted(actual - expected)}"
            )
    return errors


def _builtin_var_errors(arch: str, library: MetricLibrary) -> List[str]:
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
    """Yield each equation a row of arch_dir shows.

    A per-channel table's last key, placeholder_range, shows no column and adds no row.
    """
    for item in iter_analysis_config_metrics(arch_dir, set()):
        legacy_id = f"{item.table_id // 100}.{item.table_id % 100}.{item.metric_idx}"
        for form in _row_forms(item.table, item.metric):
            yield _Row(
                legacy_id,
                item.metric_name,
                _canonical(item.metric[form]),
                _canonical(item.metric.get("peak")),
                item.metric.get("unit"),
            )


def _row_forms(table: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    """Return the columns of row that show an equation."""
    main_forms = [form for form in MAIN_COLUMNS if form in row]
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


def _gpu_series(arch: str) -> Optional[str]:
    return mi_gpu_specs.get_gpu_series(_member_arch(arch))


def _member_arch(arch: str) -> str:
    """Return a GPU arch that uses arch's analysis configs (gfx1150 for gfx115x)."""
    return _member_archs(arch)[0]


def _member_archs(arch: str) -> List[str]:
    """Return the GPU archs that use arch's configs; none for an unknown arch."""
    return [
        gpu_arch
        for gpu_arch in sorted(mi_gpu_specs.get_gpu_series_dict())
        if canonical_config_arch(gpu_arch) == arch
    ]


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _field_errors(
    entry: object,
    required: Tuple[str, ...],
    optional: Tuple[str, ...],
    source: str,
    item: str,
) -> List[str]:
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
    entry: Dict[str, Any], fields: Tuple[str, ...], source: str, item: str
) -> List[str]:
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
