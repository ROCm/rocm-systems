#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook: check the metric library files.

For every arch in METRIC_LIBRARY_PARENT_ARCH, check that:

- the arch has a directory, and each file loads and merges with its parent's file;
- each metric has the required fields and plain ASCII text;
- each id is named after hardware and lives in the file of its category;
- units use one spelling, and an id has one name and one unit on every arch;
- no two metrics on an arch share a formula or a name;
- a formula has one id on every arch;
- every formula and peak parses;
- each roofline ceiling the arch measures is one metric's peak_empirical;
- top-level keys are sorted.

src/rocprof_compute_soc/metric_library/README.md explains the rules.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple

from format_yaml import normalize_equation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Make src/ importable to reuse the library loader and the expression parser.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rocprof_compute_soc.metric_library import loader  # noqa: E402
from utils.metrics.expression import build_eval_string  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils_common import canonical_config_arch  # noqa: E402

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
# underscores.
METRIC_LIBRARY_ID_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z0-9]+(_[a-z0-9]+)*$")
METRIC_REQUIRED_FIELDS = ("name", "formula", "unit", "description")
METRIC_OPTIONAL_FIELDS = ("peak", "peak_empirical")
# Metric fields the runtime evaluates.
EXPRESSION_FIELDS = ("formula", *METRIC_OPTIONAL_FIELDS)
VARIABLE_FIELDS = ("formula",)
# A roofline benchmark ceiling, the measured peak in "peak_empirical".
EMPIRICAL_PEAK_RE = re.compile(r"\$(\w+)_empirical_peak")
# Ceilings the roofline benchmark measures that no row shows as a peak: on every arch,
# the VALU integer rates and MALL bandwidth (analyze skips MALL, roofline_calc.py); per
# arch, the rates the roofline combines or does not show.
CEILINGS_WITHOUT_METRIC = ("I8Ops", "I32Ops", "I64Ops", "MALLBw")
ARCH_CEILINGS_WITHOUT_METRIC = {
    # The roofline shows the combined F6F4 MFMA rate.
    "gfx950": ("MFMAF4Flops", "MFMAF6Flops"),
}
# Ids allowed to share another id's formula (id -> that id): the same counters bounded
# by a different measured ceiling. gfx950 counts MXF8 matrix work in the F8 counter.
SHARED_FORMULA_IDS = {
    "compute.mfma_flops_mxf8": "compute.mfma_flops_f8",
}
# Ids that hold another id's formula on a different arch (id -> that id): one label
# shows two equations on an arch, and the parent arch's equation keeps the id until
# the metric owners choose one.
SPLIT_FORMULA_IDS = {}
# A display-only ROUND(x, digits) around a whole equation.
ROUND_RE = re.compile(r"^\s*ROUND\((.*),\s*\d+\s*\)\s*$", re.DOTALL)


def validate(metric_dir: Path) -> List[str]:
    """Return the errors of every arch in a metric library tree."""
    errors = _directory_errors(metric_dir)
    if errors:
        return errors
    libraries = {}
    for arch in loader.METRIC_LIBRARY_PARENT_ARCH:
        arch_errors = validate_arch(metric_dir, arch)
        errors.extend(arch_errors)
        if not arch_errors:
            libraries[arch] = loader.MetricLibrary.load(metric_dir, arch)
    errors.extend(_formula_id_errors(libraries))
    errors.extend(_name_unit_errors(libraries))
    return errors


def validate_arch(metric_dir: Path, arch: str) -> List[str]:
    """Return the errors of one arch's library files."""
    files = loader.arch_files(metric_dir, arch)
    load_errors = _load_errors(files)
    if load_errors:
        return load_errors
    documents = {
        file_name: loader.load_file(paths) for file_name, paths in files.items()
    }
    variables = documents.pop(loader.VARIABLES_FILENAME, {})
    return [
        *_key_order_errors(arch, files),
        *_variables_errors(variables, f"{arch}/{loader.VARIABLES_FILENAME}"),
        *(
            error
            for file_name, document in documents.items()
            for error in _metrics_errors(document, f"{arch}/{file_name}")
        ),
        *_arch_metrics_errors(arch, documents),
    ]


def canonical_formula(equation: object) -> Optional[str]:
    """Return equation in format_yaml.py form without display ROUND() and spaces.

    Returns None for an empty equation.
    """
    if equation is None or str(equation) == "None":
        return None
    match = ROUND_RE.match(str(equation))
    text = normalize_equation(match.group(1) if match else str(equation))
    return re.sub(r"\s+", "", text)


def arch_ceilings(arch: str) -> Set[str]:
    """Return the ceilings the roofline benchmark measures on any device of arch."""
    return set().union(
        *(
            mi_gpu_specs.get_roofline_columns(device)
            for device in mi_gpu_specs.get_roofline_archs()
            if canonical_config_arch(device) == arch
        )
    )


def main() -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    errors = validate(loader.default_metric_library_dir())
    if errors:
        print("Metric library validation failed:\n")
        for error in errors:
            print(f"  ERROR: {error}")
        print(f"\n{len(errors)} error(s) found.")
        return 1
    print("Metric library validation passed.")
    return 0


def _directory_errors(metric_dir: Path) -> List[str]:
    """Report mapped archs without a directory and arch directories that are not
    mapped."""
    mapped_archs = loader.METRIC_LIBRARY_PARENT_ARCH
    return [
        *(
            f"[{arch}] is in METRIC_LIBRARY_PARENT_ARCH but has no directory"
            for arch in mapped_archs
            if not (metric_dir / arch).is_dir()
        ),
        *(
            f"[{path.name}] is not a known metric library arch"
            for path in sorted(metric_dir.glob("gfx*"))
            if path.is_dir() and path.name not in mapped_archs
        ),
    ]


def _load_errors(files: Dict[str, List[Path]]) -> List[str]:
    """Report each file that does not load or merge."""
    errors = []
    for paths in files.values():
        try:
            loader.load_file(paths)
        except loader.MetricLibraryError as error:
            errors.append(str(error))
    return errors


def _key_order_errors(arch: str, files: Dict[str, List[Path]]) -> List[str]:
    """Report each file of arch's own directory whose ids are not sorted."""
    own_paths = [paths[-1] for paths in files.values() if paths[-1].parent.name == arch]
    return [
        f"[{arch}/{path.name}] top-level keys are not sorted"
        for path in own_paths
        if not _keys_sorted(loader.read_file(path))
    ]


def _keys_sorted(document: Dict[str, Any]) -> bool:
    """Return whether the keys of document, apart from the remove list, are sorted."""
    keys = [key for key in document if key != loader.REMOVE_KEY]
    return keys == sorted(keys)


def _variables_errors(document: Dict[str, Any], source: str) -> List[str]:
    """Report variables without a non-empty formula."""
    errors = []
    for name, entry in document.items():
        errors.extend(_field_errors(entry, VARIABLE_FIELDS, (), source, str(name)))
        if isinstance(entry, dict):
            errors.extend(_text_errors(entry, VARIABLE_FIELDS, source, str(name)))
    return errors


def _metrics_errors(document: Dict[str, Any], source: str) -> List[str]:
    """Report bad ids and bad metrics of one merged file."""
    category = Path(source).stem
    errors = []
    for metric_id, entry in document.items():
        errors.extend(_id_errors(str(metric_id), source, category))
        errors.extend(_metric_errors(entry, source, str(metric_id)))
    return errors


def _id_errors(metric_id: str, source: str, category: str) -> List[str]:
    """Report an id that is malformed, names a view, or is not of the file's
    category."""
    errors = []
    if not METRIC_LIBRARY_ID_RE.match(metric_id):
        errors.append(
            f"[{source}] {metric_id}: the id must look like "
            "'{category}.{subcategory}_{name}'"
        )
    id_category = metric_id.split(".")[0]
    if id_category in VIEW_CATEGORIES:
        errors.append(
            f"[{source}] {metric_id}: the id names a view; use a hardware category"
        )
    if id_category != category:
        errors.append(
            f"[{source}] {metric_id}: the id must have the file's category '{category}'"
        )
    return errors


def _metric_errors(entry: object, source: str, metric_id: str) -> List[str]:
    """Report bad fields of one metric."""
    errors = _field_errors(
        entry, METRIC_REQUIRED_FIELDS, METRIC_OPTIONAL_FIELDS, source, metric_id
    )
    if errors:
        return errors
    return [
        *_text_errors(entry, ("name", "formula"), source, metric_id),
        *_string_errors(entry, source, metric_id),
        *_peak_errors(entry, source, metric_id),
    ]


def _string_errors(entry: Dict[str, Any], source: str, metric_id: str) -> List[str]:
    """Report a unit or description that is no string, non-ASCII text, and a unit
    alias."""
    errors = [
        f"[{source}] {metric_id}: '{field}' must be a string"
        for field in ("unit", "description")
        if not isinstance(entry[field], str)
    ]
    errors.extend(
        f"[{source}] {metric_id}: '{field}' must be plain ASCII, without \\u escapes"
        for field in ("name", "unit", "description")
        if isinstance(entry[field], str)
        and (not entry[field].isascii() or "\\u" in entry[field])
    )
    if isinstance(entry["unit"], str) and entry["unit"] in UNIT_ALIASES:
        errors.append(
            f"[{source}] {metric_id}: 'unit' {entry['unit']!r} is an alias; "
            f"use {UNIT_ALIASES[entry['unit']]!r}"
        )
    return errors


def _peak_errors(entry: Dict[str, Any], source: str, metric_id: str) -> List[str]:
    """Report a peak that is not a theoretical expression, and a malformed
    peak_empirical."""
    errors = []
    peak = entry.get("peak")
    if isinstance(peak, bool) or not isinstance(peak, (str, int, float, type(None))):
        errors.append(f"[{source}] {metric_id}: 'peak' must be a string or a number")
    elif EMPIRICAL_PEAK_RE.search(str(peak)):
        errors.append(
            f"[{source}] {metric_id}: 'peak' is a measured peak; use 'peak_empirical'"
        )
    if "peak_empirical" in entry and not EMPIRICAL_PEAK_RE.fullmatch(
        str(entry["peak_empirical"])
    ):
        errors.append(
            f"[{source}] {metric_id}: 'peak_empirical' must be "
            "'$<ceiling>_empirical_peak'"
        )
    return errors


def _arch_metrics_errors(arch: str, documents: Dict[str, Dict[str, Any]]) -> List[str]:
    """Report problems across all metrics of arch."""
    try:
        combined = loader.combine_metric_documents(documents)
    except loader.MetricLibraryError as error:
        return [f"[{arch}] {error}"]
    metrics = {
        metric_id: entry
        for metric_id, entry in combined.items()
        if isinstance(entry, dict)
    }
    return [
        *_parse_errors(arch, metrics),
        *_ceiling_errors(arch, metrics),
        *_shared_formula_errors(arch, metrics),
        *_shared_name_errors(arch, metrics),
    ]


def _parse_errors(arch: str, metrics: Dict[str, Any]) -> List[str]:
    """Report each expression that build_eval_string() rejects."""
    errors = []
    for metric_id, metric in metrics.items():
        for field in EXPRESSION_FIELDS:
            error = _expression_error(metric.get(field))
            if error is not None:
                errors.append(f"[{arch}] {metric_id}: {field} does not parse: {error}")
    return errors


def _expression_error(expression: object) -> Optional[str]:
    """Return why build_eval_string() rejects an expression, or None if it accepts it.

    Values that are not strings or numbers are left to the field type checks.
    """
    if not isinstance(expression, (str, int, float)):
        return None
    try:
        eval_string = build_eval_string(str(expression))
    except SyntaxError as error:
        return error.msg
    return None if eval_string else "the expression parser does not allow an element"


def _ceiling_errors(arch: str, metrics: Dict[str, Any]) -> List[str]:
    """Check that each measured ceiling bounds exactly one metric, or is listed as a
    ceiling without a metric, and that no metric names an unmeasured ceiling."""
    measured = arch_ceilings(arch)
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
    """Report ids of arch with one formula, apart from SHARED_FORMULA_IDS."""
    formulas = {
        metric_id: canonical_formula(metric.get("formula"))
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
    """Report ids of arch with one name, ignoring case."""
    return _shared_value_errors(
        arch,
        {
            metric_id: metric["name"].casefold()
            for metric_id, metric in metrics.items()
            if isinstance(metric.get("name"), str)
        },
        "share a name; name each after what it measures",
    )


def _shared_value_errors(
    arch: str,
    value_by_id: Dict[str, Optional[str]],
    problem: str,
    allowed_groups: Sequence[Set[str]] = (),
) -> List[str]:
    """Report each group of ids with one value, apart from allowed_groups."""
    ids_by_value = defaultdict(list)
    for metric_id, value in value_by_id.items():
        ids_by_value[value].append(metric_id)
    return [
        f"[{arch}] {metric_ids}: {problem}"
        for value, metric_ids in ids_by_value.items()
        if value is not None
        and len(metric_ids) > 1
        and set(metric_ids) not in allowed_groups
    ]


def _formula_id_errors(libraries: Dict[str, loader.MetricLibrary]) -> List[str]:
    """Report a formula with a different id on another arch.

    The per-arch checks leave only ids that share a formula on different archs: a
    quantity a later arch renamed. SPLIT_FORMULA_IDS lists the pairs kept on purpose.
    """
    allowed_groups = [
        {metric_id, partner}
        for metric_id, partner in {**SHARED_FORMULA_IDS, **SPLIT_FORMULA_IDS}.items()
    ]
    archs_by_formula = _archs_by_value(
        libraries,
        lambda metric_id, metric: (canonical_formula(metric["formula"]), metric_id),
    )
    return [
        f"{sorted(archs_by_id)}: share a formula on different archs ("
        + "; ".join(
            f"{metric_id} on {', '.join(archs)}"
            for metric_id, archs in archs_by_id.items()
        )
        + "); use one id on every arch"
        for archs_by_id in archs_by_formula.values()
        if len(archs_by_id) > 1 and set(archs_by_id) not in allowed_groups
    ]


def _name_unit_errors(libraries: Dict[str, loader.MetricLibrary]) -> List[str]:
    """Report an id with another name or unit on another arch."""
    archs_by_id = _archs_by_value(
        libraries,
        lambda metric_id, metric: (metric_id, (metric["name"], metric["unit"])),
    )
    return [
        f"{metric_id}: one name and unit on every arch, not "
        + "; ".join(
            f"{name!r} [{unit}] on {archs}"
            for (name, unit), archs in archs_by_name_unit.items()
        )
        for metric_id, archs_by_name_unit in sorted(archs_by_id.items())
        if len(archs_by_name_unit) > 1
    ]


def _archs_by_value(
    libraries: Dict[str, loader.MetricLibrary],
    key: Callable[[str, Dict[str, Any]], Tuple[Hashable, Hashable]],
) -> Dict[Hashable, Dict[Hashable, List[str]]]:
    """Return group -> value -> the archs whose metrics key() maps to (group, value)."""
    groups = defaultdict(lambda: defaultdict(list))
    for arch, library in libraries.items():
        for metric_id, metric in library.metrics.items():
            group, value = key(metric_id, metric)
            groups[group][value].append(arch)
    return groups


def _field_errors(
    entry: object,
    required: Tuple[str, ...],
    optional: Tuple[str, ...],
    source: str,
    item: str,
) -> List[str]:
    """Report a missing required field and an unknown field of a mapping entry."""
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
    """Report each field that is not a non-empty string."""
    return [
        f"[{source}] {item}: '{field}' must be a non-empty string"
        for field in fields
        if not isinstance(entry.get(field), str) or not entry[field].strip()
    ]


if __name__ == "__main__":
    sys.exit(main())
