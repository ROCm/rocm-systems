# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Same-arch collectable copies must keep one formula."""

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pytest

from vendored import yaml

_CONFIGS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
)
_EXPRESSION_FIELDS = ("value", "avg")


class CollectableCopy(NamedTuple):
    collectable_id: str
    formula: str
    location: str


def expression_text(row: dict) -> Optional[str]:
    """The formula on a collectable row. ``value`` and ``avg`` are one field."""
    texts = []
    for field in _EXPRESSION_FIELDS:
        text = row.get(field)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    unique = list(dict.fromkeys(texts))
    if len(unique) != 1:
        return None
    return unique[0]


def collectable_copies(arch_dir: Path) -> List[CollectableCopy]:
    """Every ``_collect.*`` row in one architecture's panel files."""
    copies: List[CollectableCopy] = []
    for yaml_path in sorted(arch_dir.glob("*.yaml")):
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        for metric_name, row in collectable_rows_in(document):
            formula = None
            collectable_id = None
            if isinstance(row, dict):
                formula = expression_text(row)
                collectable_id = row.get("_collectable_id")
            has_id = isinstance(collectable_id, str) and bool(collectable_id)
            missing = not has_id or formula is None
            if missing:
                copies.append(
                    CollectableCopy(
                        collectable_id or "",
                        formula or "",
                        f"{yaml_path.name}:{metric_name}: missing id or formula",
                    )
                )
                continue
            copies.append(
                CollectableCopy(
                    collectable_id,
                    formula,
                    f"{yaml_path.name}:{metric_name}",
                )
            )
    return copies


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


def collectable_rows_in(node: object) -> List[tuple]:
    found: List[tuple] = []
    if isinstance(node, dict):
        for key, value in node.items():
            is_collectable = (
                isinstance(key, str)
                and key.startswith("_collect.")
                and isinstance(value, dict)
            )
            if is_collectable:
                found.append((key, value))
            else:
                found.extend(collectable_rows_in(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(collectable_rows_in(item))
    return found


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
def test_value_and_avg_with_the_same_text_are_one_formula() -> None:
    assert expression_text({"value": " SUM(A) ", "avg": "SUM(A)"}) == "SUM(A)"
    assert expression_text({"value": "SUM(A)", "avg": "SUM(B)"}) is None


@pytest.mark.misc
def test_collectable_copies_on_one_arch_share_a_formula() -> None:
    """A copied piece may use value in one table and avg in another."""
    drifts = []
    atom_copies = []
    for arch_dir in sorted(path for path in _CONFIGS.glob("gfx*") if path.is_dir()):
        copies = collectable_copies(arch_dir)
        drifts.extend(f"{arch_dir.name} {drift}" for drift in formula_drifts(copies))
        if arch_dir.name == "gfx942":
            atom_copies = [
                copy
                for copy in copies
                if copy.collectable_id == "collect.vl1d_hit_atom"
            ]
    assert drifts == []
    assert len(atom_copies) >= 2
    assert len({copy.formula for copy in atom_copies}) == 1
