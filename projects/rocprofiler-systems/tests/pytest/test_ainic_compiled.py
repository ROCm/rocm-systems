# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests that AI NIC support was compiled into the rocprofiler-systems binaries.

This test does NOT require NIC hardware to be present. The AI NIC settings
(e.g. ROCPROFSYS_USE_AINIC) are only registered when the binaries are built
with ROCPROFSYS_BUILD_AINIC=ON, so their presence in ``rocprof-sys-avail
--settings`` is a direct indicator of compile-time AI NIC support.
"""

from __future__ import annotations

import subprocess
import pytest
from rocprofsys import RocprofsysConfig

# AI NIC support is only compiled in when AMD SMI >= 26.3. Skip this build-artifact
# check on builds against older AMD SMI, where AI NIC is legitimately disabled.
pytestmark = [
    pytest.mark.rocprof_binary,
    pytest.mark.amdsmi_min_version("26.3"),
]

# Settings registered only when ROCPROFSYS_BUILD_AINIC=ON (see cmake/Packages.cmake
# and the guarded ROCPROFSYS_CONFIG_SETTING blocks in config.cpp).
_AINIC_SETTINGS = [
    "ROCPROFSYS_USE_AINIC",
    "ROCPROFSYS_SAMPLING_AINICS",
]


class TestAiNic:
    """Verify AI NIC support was compiled into the rocprofiler-systems binaries."""

    def test_settings_present(self, rocprof_config: RocprofsysConfig):
        """AI NIC settings must be listed by ``rocprof-sys-avail --settings``.

        These settings are only registered when the binaries are compiled with
        ROCPROFSYS_BUILD_AINIC=ON, so their presence proves AI NIC support was
        compiled in. No NIC hardware is required.
        """
        result = subprocess.run(
            [str(rocprof_config.rocprofsys_avail), "--settings"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"rocprof-sys-avail failed: {result.stderr}"

        settings = result.stdout
        missing = [s for s in _AINIC_SETTINGS if s not in settings]
        assert not missing, (
            "AI NIC settings not reported by rocprof-sys-avail --settings — "
            "were the binaries built with ROCPROFSYS_BUILD_AINIC=OFF?\n"
            f"Missing: {missing}"
        )
