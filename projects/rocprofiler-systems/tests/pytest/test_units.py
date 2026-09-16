# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.unit_tests]


# =============================================================================
# Unit tests
# =============================================================================


@pytest.mark.build_only
@pytest.mark.class_name("unit-tests")
class TestUnitTests(RocprofsysTest):
    def test(self):
        self.run_test("baseline", "rocprof-sys-unit-tests", no_base_env=True)
