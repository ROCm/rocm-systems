# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""RDNA4 descriptions retain their memory hierarchy and documentation units."""

from pathlib import Path

import pytest
import yaml
from config_management import metric_description_manager

import config

MEMORY_PANEL = """\
Panel Config:
  id: 300
  title: Memory Chart
  data source:
  - metric_table:
      id: 308
      metric:
        GL2 Requests:
          value: GL2C_REQ_sum
          unit: Count
  - metric_table:
      id: 309
      metric:
        DRAM-Bound Reads:
          value: GL2C_EA_RDREQ_DRAM_32B_sum
          unit: Bytes/s
  - metric_table:
      id: 311
      metric:
        Physical GDDR6 Reads:
          value: None
          unit: Bytes/s
  metrics_description:
    GL2 Requests: Requests that reach GL2.
    DRAM-Bound Reads: Traffic may be served by Infinity Cache.
    Physical GDDR6 Reads: This measurement is unavailable.
"""


@pytest.fixture
def memory_arch_dir(tmp_path: Path) -> Path:
    arch_dir = tmp_path / "gfx1201"
    arch_dir.mkdir()
    (arch_dir / "0300_memory_chart.yaml").write_text(MEMORY_PANEL, encoding="utf-8")
    return arch_dir


def test_navi48_descriptions_use_distinct_memory_sections(memory_arch_dir):
    descriptions = metric_description_manager.extract_descriptions_from_arch(
        memory_arch_dir
    )

    assert descriptions == {
        "Memory chart - GL2 Cache (L2)": {
            "GL2 Requests": {
                "plain": "Requests that reach GL2.",
                "rst": "Requests that reach GL2.",
                "unit": "Count",
            }
        },
        "Memory chart - EA to DRAM": {
            "DRAM-Bound Reads": {
                "plain": "Traffic may be served by Infinity Cache.",
                "rst": "Traffic may be served by Infinity Cache.",
                "unit": "Bytes/s",
            }
        },
        "Memory chart - Infinity Cache and GDDR6": {
            "Physical GDDR6 Reads": {
                "plain": "This measurement is unavailable.",
                "rst": "This measurement is unavailable.",
                "unit": "Bytes/s",
            }
        },
    }


@pytest.mark.parametrize(
    ("table_id", "expected_section"),
    [
        (306, "Memory chart - GLARB (GL1-GL2 Arbiter)"),
        (308, "Memory chart - EA to HBM"),
    ],
)
def test_gfx1250_memory_sections_remain_unchanged(table_id, expected_section):
    assert (
        metric_description_manager.panel_id_to_section("gfx1250", table_id)
        == expected_section
    )


def test_navi48_doc_generation_preserves_sections_and_units(memory_arch_dir, tmp_path):
    per_arch_dir = tmp_path / "per_arch"
    docs_dir = tmp_path / "docs"
    descriptions = metric_description_manager.extract_descriptions_from_arch(
        memory_arch_dir
    )
    metric_description_manager.update_per_arch_metrics_file(
        "gfx1201", descriptions, per_arch_dir
    )

    assert metric_description_manager.generate_docs_from_per_arch(
        per_arch_dir, docs_dir, ["gfx1201"]
    )

    docs = yaml.safe_load(
        (docs_dir / "gfx1201_metrics.yaml").read_text(encoding="utf-8")
    )
    assert set(docs) == set(descriptions)
    for section, metrics in descriptions.items():
        for metric_name, metric in metrics.items():
            assert docs[section][metric_name] == {
                "rst": metric["rst"],
                "unit": metric["unit"],
            }


def test_navi48_doc_mapping_covers_shipped_tables():
    arch_dir = config.analysis_configs_dir / "gfx1201"
    table_ids = set()
    for path in arch_dir.glob("*.yaml"):
        panel = yaml.safe_load(path.read_text(encoding="utf-8"))["Panel Config"]
        for source in panel["data source"]:
            if "metric_table" in source:
                table_ids.add(source["metric_table"]["id"])

    assert table_ids
    for table_id in table_ids:
        assert metric_description_manager.panel_id_to_section("gfx1201", table_id)


def test_navi48_shipped_metrics_have_valid_descriptions():
    is_valid, warnings, errors = metric_description_manager.validate_descriptions(
        config.analysis_configs_dir / "gfx1201"
    )

    assert is_valid, errors
    assert not warnings
