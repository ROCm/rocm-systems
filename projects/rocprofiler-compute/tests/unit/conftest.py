# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pandas as pd
import pytest

from utils.analysis_orm import Database


@pytest.fixture
def repeated_dispatch_frame():
    """Frame where the longest kernel by total time is not the longest dispatch.

    ``kernel_frequent`` runs three times for 500ns each, ``kernel_long`` once
    for 1000ns.
    """
    return pd.DataFrame({
        "Kernel_Name": [
            "kernel_long",
            "kernel_frequent",
            "kernel_frequent",
            "kernel_frequent",
        ],
        "GPU_ID": [0, 0, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [0, 2000, 3000, 4000],
        "End_Timestamp": [1000, 2500, 3500, 4500],
    })


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init(":memory:")
    yield Database.get_session()
    Database._session.close()
    Database._engine.dispose()
    Database._session = None
    Database._engine = None
