# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Shared test fixtures for the ais-check suite.

The tool ships as an executable named ``ais-check`` (no ``.py`` extension and a
hyphen), so it cannot be imported with a normal ``import`` statement. This shim
loads it from source as a module named ``ais_check`` and exposes it as a
session-scoped fixture.
"""

import importlib.util
import os
from importlib.machinery import SourceFileLoader

import pytest

_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ais-check")


def _load_ais_check():
    # The script has no .py extension, so spec_from_file_location can't infer a
    # loader; supply a source loader explicitly.
    loader = SourceFileLoader("ais_check", _SCRIPT_PATH)
    spec = importlib.util.spec_from_loader("ais_check", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ais_check():
    """The ais-check script loaded as an importable module."""
    return _load_ais_check()


@pytest.fixture
def make_sysfs_disk():
    """
    Build a fake sysfs block-device directory under a base path.

    Keyword flags mirror the sysfs attributes the device walk reads:
      dm_uuid   -> dm/uuid (a device-mapper layer; "LVM-" prefix means LVM)
      slaves    -> existing device dirs to symlink under slaves/
      partition -> a partition attribute (resolved up to the parent disk)
    """

    def build(root, name, *, dm_uuid=None, slaves=None, partition=False):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        if dm_uuid is not None:
            (d / "dm").mkdir(exist_ok=True)
            (d / "dm" / "uuid").write_text(dm_uuid + "\n", encoding="utf-8")
        (d / "slaves").mkdir(exist_ok=True)
        for child in slaves or []:
            (d / "slaves" / child.name).symlink_to(child, target_is_directory=True)
        if partition:
            (d / "partition").write_text("1\n", encoding="utf-8")
        return d

    return build


@pytest.fixture
def make_fs_result():
    """Build a filesystem result dict like the one _inspect_filesystem returns."""

    def build(mountpoint, source, supported, reasons=None):
        return {
            "mountpoint": mountpoint,
            "source": source,
            "fstype": "ext4",
            "supported": supported,
            "reasons": reasons or [],
        }

    return build
