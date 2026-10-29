# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/yaml_inherit_loader.py."""

from pathlib import Path

import pytest
from common import write_metric_library_family, write_text_file

from utils.yaml_inherit_loader import (
    InheritanceError,
    load_inherited_yaml,
)

METRIC = """\
{id}:
  name: {name}
  formula: SUM(SQ_WAVES)
  unit: Waves
  description: ""
"""


def write_one_metric_base(root):
    """Write base/x.yaml with the one metric a.one."""
    return write_text_file(root / "base/x.yaml", METRIC.format(id="a.one", name="One"))


def write_child(root, text):
    """Write child/x.yaml, which inherits base/x.yaml."""
    return write_text_file(root / "child/x.yaml", "inherit: ../base/x.yaml\n" + text)


# =============================================================================
# Base files and chains
# =============================================================================


def test_base_file_loads_as_is(tmp_path):
    document = load_inherited_yaml(write_one_metric_base(tmp_path))
    assert document == {
        "a.one": {
            "name": "One",
            "formula": "SUM(SQ_WAVES)",
            "unit": "Waves",
            "description": "",
        }
    }


def test_single_level_modifies_adds_and_keeps_order(tmp_path):
    write_text_file(
        tmp_path / "base/x.yaml",
        METRIC.format(id="a.one", name="One") + METRIC.format(id="a.two", name="Two"),
    )
    child = write_child(
        tmp_path,
        "a.one:\n  unit: Instructions\n" + METRIC.format(id="a.three", name="Three"),
    )
    document = load_inherited_yaml(child)
    assert list(document) == ["a.one", "a.two", "a.three"]
    assert document["a.one"]["unit"] == "Instructions"
    assert document["a.one"]["name"] == "One"
    assert list(document["a.one"]) == ["name", "formula", "unit", "description"]
    assert "inherit" not in document


def test_four_deep_chain(tmp_path):
    files = write_metric_library_family(tmp_path)
    document = load_inherited_yaml(files["gfx_c4"])
    assert list(document) == [
        "compute.salu_util",
        "compute.valu_util",
        "compute.mfma_flops_f8",
    ]
    assert "peak" not in document["compute.valu_util"]
    assert document["compute.valu_util"]["formula"].endswith("* 2)")
    assert (
        document["compute.salu_util"]["description"] == "How often the SALU was busy."
    )


# =============================================================================
# Removal
# =============================================================================


@pytest.mark.parametrize(
    "text, key",
    [("a.missing: !remove\n", "a.missing"), ("a.one:\n  peak: !remove\n", "peak")],
)
def test_remove_missing_key_fails(tmp_path, text, key):
    write_one_metric_base(tmp_path)
    child = write_child(tmp_path, text)
    with pytest.raises(InheritanceError, match=f"cannot remove '{key}'"):
        load_inherited_yaml(child)


def test_remove_takes_no_value(tmp_path):
    base = write_text_file(tmp_path / "base.yaml", "a.one: !remove a.one\n")
    with pytest.raises(InheritanceError, match="takes no value"):
        load_inherited_yaml(base)


@pytest.mark.parametrize("tags", ["  - !remove\n", "  - name: !remove\n"])
def test_remove_not_allowed_in_a_list(tmp_path, tags):
    base = write_text_file(tmp_path / "base.yaml", "tags:\n" + tags)
    with pytest.raises(InheritanceError, match="not allowed in 'tags'"):
        load_inherited_yaml(base)


# =============================================================================
# Keys and plain values
# =============================================================================


@pytest.mark.parametrize(
    "text, key",
    [
        (
            METRIC.format(id="a.one", name="One")
            + METRIC.format(id="a.one", name="Again"),
            "a.one",
        ),
        ("a.one:\n  unit: X\n  unit: Y\n", "unit"),
    ],
)
def test_duplicate_key_fails(tmp_path, text, key):
    base = write_text_file(tmp_path / "base.yaml", text)
    with pytest.raises(InheritanceError, match=rf"duplicate keys \['{key}'\]"):
        load_inherited_yaml(base)


def test_lists_and_scalars_are_replaced(tmp_path):
    write_text_file(tmp_path / "base/x.yaml", "tags: [a, b]\ntitle: Old\n")
    child = write_child(tmp_path, "tags: [c]\ntitle: New\n")
    assert load_inherited_yaml(child) == {"tags": ["c"], "title": "New"}


def test_merge_does_not_change_yaml_aliases(tmp_path):
    """A child that changes one use of an anchored mapping leaves the others alone."""
    write_text_file(
        tmp_path / "base/x.yaml", "a.one: &metric\n  unit: X\na.two: *metric\n"
    )
    child = write_child(tmp_path, "a.one:\n  unit: Y\n")
    document = load_inherited_yaml(child)
    assert document["a.one"] == {"unit": "Y"}
    assert document["a.two"] == {"unit": "X"}


# =============================================================================
# Errors in the chain
# =============================================================================


def test_cycle_names_the_chain(tmp_path):
    write_text_file(tmp_path / "a.yaml", "inherit: b.yaml\n")
    write_text_file(tmp_path / "b.yaml", "inherit: a.yaml\n")
    with pytest.raises(InheritanceError, match="inheritance cycle: ") as error:
        load_inherited_yaml(tmp_path / "a.yaml")
    chain = str(error.value).split(": ", 1)[1].split(" -> ")
    assert [Path(path).name for path in chain] == ["a.yaml", "b.yaml", "a.yaml"]


@pytest.mark.parametrize("parent", ["5", "''", "/base/x.yaml"])
def test_inherit_must_be_a_relative_path(tmp_path, parent):
    path = write_text_file(tmp_path / "a.yaml", f"inherit: {parent}\n")
    with pytest.raises(InheritanceError, match="must be a relative file path"):
        load_inherited_yaml(path)


def test_missing_parent_fails(tmp_path):
    path = write_text_file(tmp_path / "a.yaml", "inherit: ../nowhere.yaml\n")
    with pytest.raises(InheritanceError, match="parent file not found"):
        load_inherited_yaml(path)


def test_top_level_must_be_a_mapping(tmp_path):
    path = write_text_file(tmp_path / "a.yaml", "- one\n- two\n")
    with pytest.raises(InheritanceError, match="top level must be a mapping"):
        load_inherited_yaml(path)
