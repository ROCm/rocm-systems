# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Each arch keeps one formula per collectable, in its collectables file."""

from pathlib import Path
from typing import Dict, List, NamedTuple

import pytest

from utils.utils_common import (
    COLLECTABLE_REF_RE,
    COLLECTABLES_FILENAME,
    load_collectable_formulas,
    load_panel_configs,
    panel_config_paths,
)
from vendored import yaml

_CONFIGS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
)
ARCH_DIRS = sorted(path for path in _CONFIGS.glob("gfx*") if path.is_dir())
# (_collect rows on loaded panels, unique collectable ids) of each arch. The
# rows are the ones the panel files held inline before the collectables file.
ROWS_AND_IDS_BY_ARCH = {
    "gfx115x": (5, 5),
    "gfx1201": (0, 0),
    "gfx1250": (7, 7),
    "gfx908": (35, 29),
    "gfx90a": (42, 33),
    "gfx942": (46, 38),
    "gfx950": (48, 39),
}


class CollectableCopy(NamedTuple):
    collectable_id: str
    formula: str
    location: str


def formula_drifts(copies: List[CollectableCopy]) -> List[str]:
    """Ids whose copies on this arch do not share one formula."""
    grouped: Dict[str, List[CollectableCopy]] = {}
    for copy in copies:
        grouped.setdefault(copy.collectable_id, []).append(copy)
    drifts = []
    for collectable_id, group in grouped.items():
        formulas = {copy.formula for copy in group}
        if len(formulas) == 1 and collectable_id and "" not in formulas:
            continue
        shown = "; ".join(f"{copy.location}={copy.formula!r}" for copy in group)
        drifts.append(f"{collectable_id or '(no id)'}: {shown}")
    return drifts


def definition_copies(arch_dir: Path) -> List[CollectableCopy]:
    """Every definition in the collectables file, duplicate keys included."""
    collectables_path = arch_dir / COLLECTABLES_FILENAME
    root = yaml.compose(collectables_path.read_text(encoding="utf-8"))
    collectables_node = next(
        value for key, value in root.value if key.value == "collectables"
    )
    copies = []
    for key_node, definition_node in collectables_node.value:
        fields = {key.value: value.value for key, value in definition_node.value}
        copies.append(
            CollectableCopy(
                fields.get("id", ""),
                str(fields.get("formula", "")).strip(),
                f"{COLLECTABLES_FILENAME}:{key_node.value}",
            )
        )
    return copies


def inline_collectable_rows(arch_dir: Path) -> List[str]:
    """Panel files that still define a _collect row themselves."""
    found = []
    for panel_path in panel_config_paths(arch_dir):
        document = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
        for source in document["Panel Config"]["data source"]:
            table = source.get("metric_table") or {}
            found.extend(
                f"{panel_path.name}:{name}"
                for name in table.get("metric") or {}
                if name.startswith("_collect.")
            )
    return found


def loaded_tables(arch_dir: Path) -> List[dict]:
    """Every metric_table of the arch, as the analysis loader returns it."""
    return [
        source["metric_table"]
        for panel in load_panel_configs([str(arch_dir)]).values()
        for source in panel["data source"]
        if source.get("metric_table")
    ]


def first_references(metrics: dict) -> List[str]:
    """_collect row names that the table's own metrics name, in order."""
    names: List[str] = []
    for metric in metrics.values():
        for text in metric.values():
            if isinstance(text, str):
                names.extend(
                    f"_collect.{name}" for name in COLLECTABLE_REF_RE.findall(text)
                )
    return list(dict.fromkeys(names))


def table_shape_errors(table: dict, formulas: Dict[str, str]) -> List[str]:
    """How the table's _collect rows differ from the pre-move layout."""
    metrics = table["metric"]
    names = list(metrics)
    rows = [name for name in names if name.startswith("_collect.")]
    own_metrics = {name: metrics[name] for name in names if name not in rows}
    header = table.get("header") or {}
    expression_key = "value" if "value" in header else "avg"
    errors = []
    if rows and names[-len(rows) :] != rows:
        errors.append(f"{table['id']}: _collect rows are not at the end")
    if rows != first_references(own_metrics):
        errors.append(f"{table['id']}: rows {rows} are not in first-reference order")
    for name in rows:
        expected = {expression_key: formulas.get(name[1:]), "_collectable_id": name[1:]}
        if metrics[name] != expected:
            errors.append(f"{table['id']}:{name} is {metrics[name]}")
    return errors


@pytest.mark.misc
def test_same_id_with_two_formulas_is_drift() -> None:
    copies = [
        CollectableCopy("collect.x", "SUM(A)", "a.yaml:_collect.x"),
        CollectableCopy("collect.x", "SUM(B)", "b.yaml:_collect.x"),
    ]
    assert formula_drifts(copies) == [
        "collect.x: a.yaml:_collect.x='SUM(A)'; b.yaml:_collect.x='SUM(B)'"
    ]


@pytest.mark.misc
@pytest.mark.parametrize("arch_dir", ARCH_DIRS, ids=lambda path: path.name)
def test_each_collectable_has_one_definition(arch_dir: Path) -> None:
    copies = definition_copies(arch_dir)
    assert formula_drifts(copies) == []
    assert len(copies) == len({copy.collectable_id for copy in copies})
    for copy in copies:
        key = copy.location.split(":", 1)[1]
        assert copy.collectable_id == f"collect.{key}", copy.location


@pytest.mark.misc
@pytest.mark.parametrize("arch_dir", ARCH_DIRS, ids=lambda path: path.name)
def test_panel_files_hold_no_collectable_copies(arch_dir: Path) -> None:
    assert inline_collectable_rows(arch_dir) == []


@pytest.mark.misc
@pytest.mark.parametrize("arch_dir", ARCH_DIRS, ids=lambda path: path.name)
def test_loaded_panels_keep_the_pre_move_collectable_rows(arch_dir: Path) -> None:
    """Loaded tables end with the _collect rows the panel files used to hold."""
    formulas = load_collectable_formulas(arch_dir)
    tables = loaded_tables(arch_dir)
    errors = [
        error for table in tables for error in table_shape_errors(table, formulas)
    ]
    row_ids = [
        row["_collectable_id"]
        for table in tables
        for name, row in table["metric"].items()
        if name.startswith("_collect.")
    ]
    assert errors == []
    assert (len(row_ids), len(set(row_ids))) == ROWS_AND_IDS_BY_ARCH[arch_dir.name]
    assert set(row_ids) == set(formulas), "a definition no panel references"


@pytest.mark.misc
def test_gfx942_atom_piece_follows_each_table_header() -> None:
    """One formula, stored under value in 1601 and under avg in 1603."""
    tables = {table["id"]: table for table in loaded_tables(_CONFIGS / "gfx942")}
    in_1601 = tables[1601]["metric"]
    in_1603 = tables[1603]["metric"]
    formula = load_collectable_formulas(_CONFIGS / "gfx942")["collect.vl1d_hit_atom"]
    assert in_1601["_collect.vl1d_hit_atom"]["value"] == formula
    assert in_1603["_collect.vl1d_hit_atom"]["avg"] == formula
    assert "_collect.vl1d_hit_non_rw" in in_1601
    assert "_collect.vl1d_chr_non_rw" in in_1603
    assert "_collect.vl1d_hit_non_rw" not in in_1603
