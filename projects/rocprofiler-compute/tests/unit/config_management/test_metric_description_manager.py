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
      id: 306
      metric:
        GL1 Returns:
          value: GL1C_GL1_RET_32B_sum
          unit: Bytes
  - metric_table:
      id: 308
      metric:
        DRAM-Bound Reads:
          value: GL2C_EA_RDREQ_DRAM_32B_sum
          unit: Bytes/s
  - metric_table:
      id: 310
      metric:
        Physical GDDR6 Reads:
          value: None
          unit: Bytes/s
  metrics_description:
    GL1 Returns: Bytes returned to GL1 clients.
    DRAM-Bound Reads: Traffic may be served by Infinity Cache.
    Physical GDDR6 Reads: This measurement is unavailable.
"""

BANDWIDTH_PANEL = """\
Panel Config:
  id: 1300
  title: GL2 Cache
  data source:
  - metric_table:
      id: 1304
      title: GL2C Request Bandwidth
      metric:
        GL2 Cache Read BW:
          avg: GL2C_READ_32_REQ_sum
          unit: Bytes/s
        GL2 Cache Write BW:
          avg: GL2C_WRITE_32_REQ_sum
          unit: Bytes/s
  metrics_description:
    GL2 Cache Read BW: Updated L2 read bandwidth description.
    GL2 Cache Write BW: Updated L2 write bandwidth description.
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
        "Memory chart - GL1 (Write-Combining)": {
            "GL1 Returns": {
                "plain": "Bytes returned to GL1 clients.",
                "rst": "Bytes returned to GL1 clients.",
                "unit": "Bytes",
            }
        },
        "Memory chart - L2 Memory Interface": {
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


@pytest.mark.parametrize(
    (
        "arch_name",
        "hbm_rst",
        "cache_rst",
        "current_rst",
        "expected_section",
        "expected_read_rst",
    ),
    [
        pytest.param(
            "gfx1201",
            "Legacy HBM ``L2`` guidance.",
            None,
            None,
            "GL2C Request Bandwidth",
            "Legacy HBM ``L2`` guidance.",
            id="old-hbm-only",
        ),
        pytest.param(
            "gfx1201",
            None,
            "Previous cache ``L2`` guidance.",
            None,
            "GL2C Request Bandwidth",
            "Previous cache ``L2`` guidance.",
            id="old-cache-only",
        ),
        pytest.param(
            "gfx1201",
            "Legacy HBM ``L2`` guidance.",
            None,
            "Current autogenerated description.",
            "GL2C Request Bandwidth",
            "Legacy HBM ``L2`` guidance.",
            id="current-autogenerated-preserves-hbm-authored-rst",
        ),
        pytest.param(
            "gfx1201",
            None,
            "Previous cache ``L2`` guidance.",
            "Current autogenerated description.",
            "GL2C Request Bandwidth",
            "Previous cache ``L2`` guidance.",
            id="current-autogenerated-preserves-cache-authored-rst",
        ),
        pytest.param(
            "gfx1201",
            "Legacy HBM ``L2`` guidance.",
            "Previous cache ``L2`` guidance.",
            "Current authored ``L2`` guidance.",
            "GL2C Request Bandwidth",
            "Current authored ``L2`` guidance.",
            id="current-authored-rst-takes-precedence",
        ),
        pytest.param(
            "gfx1201",
            "Legacy HBM ``L2`` guidance.",
            "Previous cache ``L2`` guidance.",
            "Current autogenerated description.",
            "GL2C Request Bandwidth",
            "Previous cache ``L2`` guidance.",
            id="cache-authored-rst-takes-precedence-over-hbm",
        ),
        pytest.param(
            "gfx1201",
            "Legacy HBM ``L2`` guidance.",
            "Cache autogenerated description.",
            "Current autogenerated description.",
            "GL2C Request Bandwidth",
            "Legacy HBM ``L2`` guidance.",
            id="autogenerated-cache-cannot-replace-hbm-authored-rst",
        ),
        pytest.param(
            "gfx1250",
            "Legacy HBM ``L2`` guidance.",
            None,
            None,
            "GL2C-EA Interface - HBM Bandwidth",
            "Legacy HBM ``L2`` guidance.",
            id="gfx1250-remains-unchanged",
        ),
    ],
)
def test_sync_bandwidth_section_preserves_authored_rst_without_duplicates(
    tmp_path,
    arch_name,
    hbm_rst,
    cache_rst,
    current_rst,
    expected_section,
    expected_read_rst,
) -> None:
    configs_dir = tmp_path / "configs"
    arch_dir = configs_dir / arch_name
    arch_dir.mkdir(parents=True)
    (arch_dir / "1300_gl2_cache.yaml").write_text(BANDWIDTH_PANEL, encoding="utf-8")
    per_arch_dir = tmp_path / "per_arch"
    per_arch_dir.mkdir()
    per_arch_file = per_arch_dir / f"{arch_name}_metrics_description.yaml"
    existing = {}
    for section, read_plain, read_rst in [
        (
            "GL2C-EA Interface - HBM Bandwidth",
            "HBM autogenerated description.",
            hbm_rst,
        ),
        ("GL2C Cache Bandwidth", "Cache autogenerated description.", cache_rst),
        (
            "GL2C Request Bandwidth",
            "Current autogenerated description.",
            current_rst,
        ),
    ]:
        if read_rst is None:
            continue
        existing[section] = {
            "GL2 Cache Read BW": {
                "plain": read_plain,
                "rst": read_rst,
                "unit": "Bytes/s",
            },
            "GL2 Cache Write BW": {
                "plain": "Legacy write description.",
                "rst": "Legacy write description.",
                "unit": "Bytes/s",
            },
        }
    per_arch_file.write_text(yaml.safe_dump(existing), encoding="utf-8")

    assert metric_description_manager.sync_arch(
        arch_name, str(configs_dir), str(per_arch_dir)
    )

    synced = yaml.safe_load(per_arch_file.read_text(encoding="utf-8"))
    assert synced == {
        expected_section: {
            "GL2 Cache Read BW": {
                "plain": "Updated L2 read bandwidth description.",
                "rst": expected_read_rst,
                "unit": "Bytes/s",
            },
            "GL2 Cache Write BW": {
                "plain": "Updated L2 write bandwidth description.",
                "rst": "Updated L2 write bandwidth description.",
                "unit": "Bytes/s",
            },
        }
    }
    assert metric_description_manager.sync_arch(
        arch_name, str(configs_dir), str(per_arch_dir)
    )
    assert yaml.safe_load(per_arch_file.read_text(encoding="utf-8")) == synced
