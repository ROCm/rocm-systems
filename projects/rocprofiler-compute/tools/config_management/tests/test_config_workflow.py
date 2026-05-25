# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Panel-YAML fixture builders for master_config_workflow_script tests."""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_panel_dict(
    panel_id: int,
    title: str,
    tables: tuple,
    descriptions: dict | None = None,
) -> dict:
    """
    tables: tuple of (table_id, table_title, metrics_dict)
    metrics_dict example:
      {
        "Metric A": {
          "avg": "AVG(A)",
          "min": "MIN(A)",
          "max": "MAX(A)",
          "unit": "Percent",
        },
      }
    """
    data_sources = []
    for tid, ttitle, metrics in tables:
        data_sources.append({
            "metric_table": {
                "id": tid,
                "title": ttitle,
                "header": {
                    "metric": "Metric",
                    "avg": "Avg",
                    "min": "Min",
                    "max": "Max",
                    "unit": "Unit",
                },
                "metric": metrics or {},
            }
        })

    panel = {
        "Panel Config": {
            "id": panel_id,
            "title": title,
            "data source": data_sources,
        }
    }
    if descriptions:
        panel["Panel Config"]["metrics_description"] = descriptions
    return panel


def write_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
