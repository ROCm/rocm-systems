# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load each arch's metric library, merged through its parent archs.

README.md in this directory describes the files and the inheritance rules.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from utils.utils_common import canonical_config_arch
from utils.utils_counter_defs import extract_counters_and_variables
from vendored import yaml

# Each arch inherits from its immediate predecessor; None marks a family base.
METRIC_LIBRARY_PARENT_ARCH: Dict[str, Optional[str]] = {
    "gfx908": None,
    "gfx90a": "gfx908",
    "gfx942": "gfx90a",
    "gfx950": "gfx942",
    "gfx115x": None,
    "gfx1250": None,
}
VARIABLES_FILENAME = "variables.yaml"
SETS_FILENAME = "sets.yaml"
# Top-level key for the list of parent ids to drop in a child file.
REMOVE_KEY = "remove"
# Metric fields whose expressions can name hardware counters.
COUNTER_EXPRESSION_FIELDS = ("formula", "peak")


class MetricLibraryError(Exception):
    """A metric library file cannot be loaded or merged."""


def default_metric_library_dir() -> Path:
    """Return the metric library tree shipped with rocprof-compute."""
    return Path(__file__).resolve().parent


def arch_chain(arch: str) -> List[str]:
    """Return the chain of archs for arch, base first."""
    config_arch = canonical_config_arch(arch)
    if config_arch not in METRIC_LIBRARY_PARENT_ARCH:
        raise ValueError(f"No metric library is defined for arch {arch}")
    chain = [config_arch]
    while METRIC_LIBRARY_PARENT_ARCH[chain[-1]] is not None:
        parent = METRIC_LIBRARY_PARENT_ARCH[chain[-1]]
        if parent in chain:
            raise ValueError(f"METRIC_LIBRARY_PARENT_ARCH has a cycle at {parent}")
        chain.append(parent)
    return chain[::-1]


def arch_files(metric_dir: Path, arch: str) -> Dict[str, List[Path]]:
    """Return each file name of the library of arch, with its files base first."""
    chain = arch_chain(arch)
    missing_dirs = [
        str(metric_dir / chain_arch)
        for chain_arch in chain
        if not (metric_dir / chain_arch).is_dir()
    ]
    if missing_dirs:
        raise MetricLibraryError(f"No metric library directory {missing_dirs}")
    files_by_name: Dict[str, List[Path]] = defaultdict(list)
    for chain_arch in chain:
        for path in sorted((metric_dir / chain_arch).glob("*.yaml")):
            files_by_name[path.name].append(path)
    return dict(sorted(files_by_name.items()))


def read_file(path: Path) -> Dict[str, Any]:
    """Return the top-level mapping of one library file."""
    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise MetricLibraryError(f"{path}: {error}") from error
    if not isinstance(document, dict):
        raise MetricLibraryError(f"{path}: the top level must be a mapping")
    return document


def load_file(paths: List[Path]) -> Dict[str, Any]:
    """Return one library file merged through its parent archs' files of that name."""
    merged: Dict[str, Any] = {}
    for path in paths:
        merged = merge_documents(merged, read_file(path), str(path))
    return merged


def merge_documents(
    parent: Dict[str, Any], child: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Return parent without the removed ids, with child merged on top.

    Mappings merge key by key; scalars and lists replace the parent's value.
    """
    removed_ids = _removed_ids(parent, child, source)
    merged = {key: value for key, value in parent.items() if key not in removed_ids}
    for key, value in child.items():
        if key != REMOVE_KEY:
            merged[key] = _merged_value(parent.get(key), value, key, source)
    return merged


def combine_metric_documents(
    documents: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return the metrics of every file; raise if two files define one id."""
    metrics: Dict[str, Dict[str, Any]] = {}
    for file_name, document in documents.items():
        duplicate_ids = sorted(metrics.keys() & document.keys())
        if duplicate_ids:
            raise MetricLibraryError(
                f"{file_name}: metric ids {duplicate_ids} are defined in another file"
            )
        metrics.update(document)
    return metrics


def _removed_ids(
    parent: Dict[str, Any], child: Dict[str, Any], source: str
) -> Set[str]:
    """Return the checked ids of the REMOVE_KEY list in child."""
    removed_ids = child.get(REMOVE_KEY, [])
    if not isinstance(removed_ids, list) or not all(
        isinstance(metric_id, str) for metric_id in removed_ids
    ):
        raise MetricLibraryError(f"{source}: '{REMOVE_KEY}' must be a list of ids")
    missing_ids = [metric_id for metric_id in removed_ids if metric_id not in parent]
    if missing_ids:
        raise MetricLibraryError(
            f"{source}: cannot remove {missing_ids}: the parent has no such ids"
        )
    redefined_ids = [metric_id for metric_id in removed_ids if metric_id in child]
    if redefined_ids:
        raise MetricLibraryError(f"{source}: {redefined_ids} are removed and defined")
    return set(removed_ids)


def _merged_value(
    parent_value: object, child_value: object, key: str, source: str
) -> object:
    """Return child_value merged onto parent_value; mappings merge key by key."""
    parent_kind = _kind(parent_value)
    child_kind = _kind(child_value)
    if None not in (parent_value, child_value) and parent_kind != child_kind:
        raise MetricLibraryError(
            f"{source}: '{key}' replaces a {parent_kind} with a {child_kind}"
        )
    if not isinstance(child_value, dict) or not isinstance(parent_value, dict):
        return child_value
    merged = dict(parent_value)
    for child_key, value in child_value.items():
        merged[child_key] = _merged_value(
            parent_value.get(child_key), value, child_key, source
        )
    return merged


def _kind(value: object) -> str:
    """Return whether value is a mapping, a list or a scalar."""
    if isinstance(value, dict):
        return "mapping"
    return "list" if isinstance(value, list) else "scalar"


class MetricLibrary:
    """One arch's metrics, built-in variables and sets."""

    def __init__(
        self,
        arch: str,
        metrics: Dict[str, Dict[str, Any]],
        variables: Dict[str, Dict[str, Any]],
        sets: Dict[str, Dict[str, Any]],
    ) -> None:
        self.arch = arch
        self.metrics = metrics
        self.variables = variables
        self.sets = sets

    @classmethod
    def load(cls, metric_dir: Path, arch: str) -> "MetricLibrary":
        """Read the library of arch from metric_dir, merged through its parent archs."""
        documents = {
            file_name: load_file(paths)
            for file_name, paths in arch_files(metric_dir, arch).items()
        }
        variables = documents.pop(VARIABLES_FILENAME, {})
        sets = documents.pop(SETS_FILENAME, {})
        metrics = combine_metric_documents(documents)
        return cls(canonical_config_arch(arch), metrics, variables, sets)

    def get_counters_for_metrics(
        self, metric_ids: List[str], gpu_series: str
    ) -> Set[str]:
        """Return the hardware counters in the formulas and peaks of metric_ids.

        Raises ValueError if an id is not a metric of the arch.
        """
        unknown_metric_ids = [
            metric_id for metric_id in metric_ids if metric_id not in self.metrics
        ]
        if unknown_metric_ids:
            raise ValueError(
                f"Unknown metric ids for {self.arch}: {unknown_metric_ids}"
            )
        expressions = [
            str(self.metrics[metric_id][field])
            for metric_id in metric_ids
            for field in COUNTER_EXPRESSION_FIELDS
            if self.metrics[metric_id].get(field) is not None
        ]
        counters, _ = extract_counters_and_variables("\n".join(expressions), gpu_series)
        return counters
