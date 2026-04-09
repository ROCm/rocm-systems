# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.unit_tests]


# =============================================================================
# Units tests
# =============================================================================


class TestUnitTests(RocprofsysTest):
    def test(self, rocprof_config):
        if rocprof_config.is_installed:
            pytest.skip("Test only runs in build mode")
        self.run_test("baseline", "rocprof-sys-unit-tests")
