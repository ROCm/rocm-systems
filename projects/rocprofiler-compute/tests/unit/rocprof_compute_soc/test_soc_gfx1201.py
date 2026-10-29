# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from argparse import Namespace
from types import SimpleNamespace

from rocprof_compute_soc.soc_base import OmniSoC_Base
from rocprof_compute_soc.soc_gfx1201 import gfx1201_soc


def test_gfx1201_soc_initializes_with_current_base_api(monkeypatch):
    monkeypatch.setattr(OmniSoC_Base, "populate_mspec", lambda self: None)
    mspec = SimpleNamespace()
    soc = gfx1201_soc(Namespace(), mspec)

    assert soc.get_arch() == "gfx1201"
    assert mspec.lds_banks_per_cu == 32
