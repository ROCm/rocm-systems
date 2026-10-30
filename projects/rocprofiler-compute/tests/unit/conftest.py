# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pytest

from utils.analysis_orm import Database


@pytest.fixture
def gfx942_sys_info():
    """Every required sys.info field, as sysinfo.csv writes it for a gfx942 part."""
    return {
        "gpu_arch": "gfx942",
        "ip_blocks": "all",
        "se_per_gpu": 8,
        "pipes_per_gpu": 4,
        "cu_per_gpu": 304,
        "simd_per_cu": 4,
        "sqc_per_gpu": 38,
        "lds_banks_per_cu": 32,
        "cur_sclk": 2100.0,
        "cur_mclk": 1300.0,
        "max_mclk": 1300.0,
        "max_sclk": 2100.0,
        "max_waves_per_cu": 32,
        "wave_size": 64,
        "total_l2_chan": 16,
        "num_xcd": 8,
    }


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init(":memory:")
    yield Database.get_session()
    Database._session.close()
    Database._engine.dispose()
    Database._session = None
    Database._engine = None
