# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pytest

from membw.tree_spec import (
    _parse_tree_spec,
    _validate_tree_spec,
    collect_metric_keys,
    load_tree_spec,
    tree_spec_path,
)
from utils.utils_common import load_yaml


class TestLoadTreeSpec:
    def test_gfx950_loads_and_validates(self):
        """Load the real gfx950 spec -- verifies YAML parses and validates."""
        spec = load_tree_spec("gfx950")
        assert len(spec.thresholds) == 2
        assert len(spec.guidance_templates) == 21
        assert len(spec.schema_hash) == 16

    def test_missing_arch_raises(self):
        with pytest.raises(SystemExit):
            load_tree_spec("gfx_nonexistent")

    def test_spec_path_exists_for_gfx950(self):
        assert tree_spec_path("gfx950").exists()

    def test_spec_path_missing_for_unknown_arch(self):
        assert not tree_spec_path("gfx_nonexistent").exists()


class TestCollectMetricKeys:
    def test_gfx950_has_21_metric_keys(self):
        spec = load_tree_spec("gfx950")
        keys = collect_metric_keys(spec)
        assert len(keys) == 21

    def test_keys_span_all_three_tables(self):
        """Metric keys should reference L1 (3001), L2 (3012), EA (3018)."""
        spec = load_tree_spec("gfx950")
        keys = collect_metric_keys(spec)
        assert any(k.startswith("L1") for k in keys)
        assert any(k.startswith("L2") for k in keys)
        assert any(k.startswith("EA") for k in keys)


class TestValidation:
    """Verify that malformed specs are rejected at load time."""

    def test_invalid_threshold_reference(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    op: gte\n    threshold: bad_ref\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_invalid_guidance_id(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    op: gte\n    threshold: t1\n    label: n\n"
            "    guidance_id: missing\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_invalid_op(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    op: bad_op\n    threshold: t1\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_invalid_level(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: INVALID\n    metric: m\n"
            "    op: gte\n    threshold: t1\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_missing_level_on_root_raises(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    metric: m\n"
            "    op: gte\n    threshold: t1\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_metric_without_op(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    threshold: t1\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_metric_without_threshold(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    op: gte\n    label: n\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)

    def test_children_as_list_raises(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  n:\n    level: GL1\n    metric: m\n"
            "    op: gte\n    threshold: t1\n    label: n\n"
            "    children:\n      - bad_item\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            _parse_tree_spec(load_yaml(spec_file))

    def test_invalid_sibling_reference(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(
            "thresholds:\n  t1: 10.0\n"
            "nodes:\n  p:\n    level: GL1\n    metric: m\n"
            "    op: gte\n    threshold: t1\n    label: p\n"
            "    children:\n"
            "      a:\n        metric: m\n        op: gte\n"
            "        threshold: t1\n        label: a\n"
            "      catch:\n        requires_parent: true\n"
            "        requires_siblings_false: [missing]\n"
            "        label: catch\n"
            "guidance_templates: {}\n",
            encoding="utf-8",
        )
        spec = _parse_tree_spec(load_yaml(spec_file))
        with pytest.raises(SystemExit):
            _validate_tree_spec(spec)
