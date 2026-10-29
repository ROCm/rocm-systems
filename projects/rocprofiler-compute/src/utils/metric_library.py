# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The metric library: each arch's metric definitions, and how they are loaded.

metric_library/<arch>/ defines each metric once per arch, keyed by id, with no display
data. A metric lives in the file named after its id category (l2.yaml holds l2.*). A
child arch keeps only the files that differ from its parent; each names its parent with
an "inherit" key. variables.yaml holds the built-in variables.

Only tests and tools use this module so far. At the cutover it becomes the source of
metrics for analyze and profile, so its scope grows past loading. Loading only rejects
an id that two files define.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import config
from utils.utils_common import canonical_config_arch
from utils.utils_counter_defs import extract_counters_and_variables
from utils.yaml_inherit_loader import InheritanceError, load_inherited_yaml
from vendored import yaml

# Each arch inherits from its immediate predecessor; None marks a family base.
METRIC_LIBRARY_PARENT_ARCH: Dict[str, Optional[str]] = {
    "gfx908": None,
    "gfx90a": "gfx908",
    "gfx940": "gfx90a",
    "gfx941": "gfx940",
    "gfx942": "gfx940",
    "gfx950": "gfx942",
    "gfx115x": None,
    "gfx1250": None,
}
VARIABLES_FILENAME = "variables.yaml"

# Metric fields that may name counters.
COUNTER_FIELDS = ("formula", "peak")


def default_metric_library_dir() -> Path:
    """Return the metric library tree shipped with rocprof-compute."""
    return Path(config.rocprof_compute_home) / "rocprof_compute_soc" / "metric_library"


def arch_chain(arch: str) -> List[str]:
    """Return the metric library archs that arch inherits from, base first.

    Args:
        arch: A GPU arch, e.g. "gfx942" or "gfx1151".

    Returns:
        The chain of canonical archs, ending with arch itself.
    """
    config_arch = canonical_config_arch(arch)
    if config_arch not in METRIC_LIBRARY_PARENT_ARCH:
        raise FileNotFoundError(f"No metric library is defined for arch {arch}")
    chain = [config_arch]
    parent = METRIC_LIBRARY_PARENT_ARCH[config_arch]
    while parent is not None:
        chain.insert(0, parent)
        parent = METRIC_LIBRARY_PARENT_ARCH[parent]
    return chain


def arch_files(metric_dir: Path, arch: str) -> Dict[str, Path]:
    """Return each library file of arch, keyed by file name.

    A file the arch lacks comes from its nearest ancestor.

    Args:
        metric_dir: The metric library tree.
        arch: A GPU arch.

    Returns:
        The path of each file, keyed by file name, in file name order.
    """
    chain = arch_chain(arch)
    arch_dir = Path(metric_dir) / chain[-1]
    if not arch_dir.is_dir():
        raise FileNotFoundError(f"No metric library for {arch}: {arch_dir} is missing")
    files: Dict[str, Path] = {}
    for chain_arch in chain:
        for path in (Path(metric_dir) / chain_arch).glob("*.yaml"):
            files[path.name] = path
    return dict(sorted(files.items()))


def load_metric_library(metric_dir: Path, arch: str) -> "MetricLibrary":
    """Load the metric library of arch, merged through its chain. Cached per arch.

    Args:
        metric_dir: The metric library tree.
        arch: A GPU arch.

    Returns:
        The arch's metrics and built-in variables.
    """
    key = (str(Path(metric_dir).resolve()), canonical_config_arch(arch))
    if key not in MetricLibrary._cache:
        MetricLibrary._cache[key] = _read_metric_library(metric_dir, arch)
    return MetricLibrary._cache[key]


def iter_library_documents(
    metric_dir: Path,
    arch: str,
    load_errors: Optional[Dict[Path, Exception]] = None,
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Yield each library file of arch with its document, merged through its chain.

    A file that does not load raises, or with load_errors is recorded there and skipped.
    """
    for path in arch_files(metric_dir, arch).values():
        try:
            document = load_inherited_yaml(path)
        except (InheritanceError, yaml.YAMLError) as error:
            if load_errors is None:
                raise
            load_errors[path] = error
            continue
        yield path, document


def _read_metric_library(metric_dir: Path, arch: str) -> "MetricLibrary":
    metrics: Dict[str, Dict[str, Any]] = {}
    variables: Dict[str, Dict[str, Any]] = {}
    for path, document in iter_library_documents(metric_dir, arch):
        if path.name == VARIABLES_FILENAME:
            variables = document
            continue
        duplicates = sorted(set(document) & set(metrics))
        if duplicates:
            raise ValueError(f"{path}: metric ids {duplicates} are defined twice")
        metrics.update(document)
    return MetricLibrary(metrics, variables)


class MetricLibrary:
    """One arch's metrics and built-in variables."""

    _cache: Dict[Tuple[str, str], "MetricLibrary"] = {}

    def __init__(
        self, metrics: Dict[str, Dict[str, Any]], variables: Dict[str, Dict[str, Any]]
    ) -> None:
        self.metrics = metrics
        self.variables = variables

    def get_counters_for_metrics(
        self, metric_ids: List[str], gpu_series: str
    ) -> Set[str]:
        """Return the counters the metrics need, from formula and peak."""
        texts = [
            str(self.metrics[metric_id][field])
            for metric_id in metric_ids
            for field in COUNTER_FIELDS
            if self.metrics[metric_id].get(field) is not None
        ]
        counters, _ = extract_counters_and_variables("\n".join(texts), gpu_series)
        return counters
