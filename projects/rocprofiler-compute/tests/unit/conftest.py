# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here probes a GPU, starts a CLI process, or accesses the network.
"""

import sys
from pathlib import Path

import common
import pytest

from rocprof_compute_base import RocProfCompute
from utils.analysis_orm import Database


@pytest.fixture
def parse_cli_args(monkeypatch):
    """Parse CLI arguments through startup without initializing hardware."""
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__supported_archs = common.SUPPORTED_ARCHS
    instance._RocProfCompute__version = {"ver_pretty": "unit test"}
    monkeypatch.setattr("config.rocprof_compute_home", Path.cwd() / "src")

    def parse(argv):
        monkeypatch.setattr(sys, "argv", ["rocprof-compute", *argv])
        instance.parse_args()
        return instance.get_args()

    return parse


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init(":memory:")
    yield Database.get_session()
    Database._session.close()
    Database._engine.dispose()
    Database._session = None
    Database._engine = None
