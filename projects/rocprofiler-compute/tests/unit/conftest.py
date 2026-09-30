# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pytest

from utils.analysis_orm import Database


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init()
    yield Database.get_session()
    Database.close()
