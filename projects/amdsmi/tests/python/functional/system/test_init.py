#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""System initialization and shutdown functional tests."""

import unittest

import common.common as common
from common.common import amdsmi


class TestAmdSmiInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)
        return

    def test_init_shutdown(self):
        self.common.print("## test_init_shutdown()")

        msg = "\t### amdsmi_init():"
        try:
            ret = self.common.amdsmi_smart_init()[0]
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e

        msg = "\t### amdsmi_shut_down():"
        try:
            ret = amdsmi.amdsmi_shut_down()
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e
        return
