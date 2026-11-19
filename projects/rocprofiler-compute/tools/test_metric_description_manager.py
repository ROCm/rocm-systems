# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/config_management/metric_description_manager.py.

Run from the project directory:
    python3 -m pytest tools/test_metric_description_manager.py
"""

import shutil

import yaml
from config_management.metric_description_manager import (
    DOCS_DIR,
    DOCS_SECTION_NAMES,
    docs_file,
    sync_arch,
)

ARCH = "gfx115x"
# A section whose docs name differs from the name the views give it
VIEW_SECTION, DOCS_SECTION = next(iter(DOCS_SECTION_NAMES[ARCH].items()))


def test_sync_keeps_hand_edited_rst_under_the_docs_section_name(tmp_path):
    shutil.copy(docs_file(ARCH, DOCS_DIR), tmp_path)
    path = docs_file(ARCH, tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    metric = next(iter(data[DOCS_SECTION]))
    data[DOCS_SECTION][metric]["rst"] = "Edited by hand, see :ref:`desc-vmem`."
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    sync_arch(ARCH, tmp_path)

    synced = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert VIEW_SECTION not in synced
    assert synced[DOCS_SECTION][metric]["rst"] == (
        "Edited by hand, see :ref:`desc-vmem`."
    )
    assert synced[DOCS_SECTION][metric]["plain"] == data[DOCS_SECTION][metric]["plain"]
