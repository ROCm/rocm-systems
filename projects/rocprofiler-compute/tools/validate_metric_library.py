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
     and a formula has one id on every arch (except SPLIT_FORMULA_IDS);
  6. units use one spelling, and each metric id has one name and one unit on every
     arch.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_yaml import normalize_equation

from utils import metric_library
from utils.metric_library import (
    VARIABLES_FILENAME,
    arch_files,
    default_metric_library_dir,
    iter_library_documents,
    load_metric_library,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import canonical_config_arch
from utils.yaml_inherit_loader import InheritanceError, declared_parent
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

# A metric id, "{category}.{subcategory}_{name}": lowercase words joined by
# underscores. utils_common.METRIC_ID_RE matches the numeric panel ids instead.
METRIC_LIBRARY_ID_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z0-9]+(_[a-z0-9]+)*$")
METRIC_REQUIRED_FIELDS = ("name", "formula", "unit", "description")
METRIC_OPTIONAL_FIELDS = ("peak", "peak_empirical")
# A roofline benchmark ceiling, the measured peak in "peak_empirical".
EMPIRICAL_PEAK_RE = re.compile(r"\$(\w+)_empirical_peak")
# Ceilings the roofline benchmark measures that no row shows as a peak: on every arch,
# the VALU integer rates; per arch, the rates the roofline combines or does not show.
CEILINGS_WITHOUT_METRIC = ("I8Ops", "I32Ops", "I64Ops")
ARCH_CEILINGS_WITHOUT_METRIC = {}
VARIABLE_FIELDS = ("formula",)
# Ids allowed to share another id's formula (id -> that id): the same counters bounded
# by a different measured ceiling.
SHARED_FORMULA_IDS = {}
# Ids that hold another id's formula on a different arch (id -> that id): one label
# shows two equations on an arch (D8 Kind 3), and the parent arch's equation keeps
# the id until the metric owners choose one.
SPLIT_FORMULA_IDS = {}

ROUND_RE = re.compile(r"^\s*ROUND\((.*),\s*\d+\s*\)\s*$", re.DOTALL)


def validate(metric_dir: Path) -> List[str]:
    """Validate every arch in a metric library tree. Return error messages."""
    errors: List[str] = []
    valid_archs = []
    for arch in metric_library.METRIC_LIBRARY_PARENT_ARCH:
        if not (metric_dir / arch).is_dir():
            continue
        arch_errors = validate_arch(metric_dir, arch)
        if not arch_errors:
            valid_archs.append(arch)
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
    for path, document in iter_library_documents(metric_dir, arch, load_errors):
        if path.name == VARIABLES_FILENAME:
            errors.extend(_variables_errors(document, _source(path)))
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
    return errors


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(default_metric_library_dir())
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


def _canonical(equation: object) -> Optional[str]:
    """Return equation in format_yaml.py form without display ROUND(); None if empty.

    Many analysis_configs rows are unformatted (format_yaml.py skips AUTOGENERATED).
    """
    if equation is None or str(equation) == "None":
        return None
    match = ROUND_RE.match(str(equation))
    text = normalize_equation(match.group(1) if match else str(equation))
    return re.sub(r"\s+", "", text)


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


if __name__ == "__main__":
    sys.exit(main())
