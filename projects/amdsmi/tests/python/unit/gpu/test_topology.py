#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Topology API handle-validation unit tests (hardware-free)."""

from __future__ import annotations

import unittest

from common.common import Common, amdsmi

_common = Common.for_unit_tests()


class TestAmdSmiTopologyHandleValidation(unittest.TestCase):
    """The interface layer rejects non-handle arguments before any C call."""

    def test_link_type_rejects_non_handle(self):
        with _common.expect_status(
            "\t### amdsmi_topo_get_link_type(gpu=not_a_handle, gpu=not_a_handle):",
            raises=amdsmi.AmdSmiParameterException,
        ):
            amdsmi.amdsmi_topo_get_link_type("not_a_handle", "not_a_handle")

    def test_p2p_status_rejects_non_handle(self):
        with _common.expect_status(
            "\t### amdsmi_topo_get_p2p_status(gpu=not_a_handle, gpu=not_a_handle):",
            raises=amdsmi.AmdSmiParameterException,
        ):
            amdsmi.amdsmi_topo_get_p2p_status("not_a_handle", "not_a_handle")
