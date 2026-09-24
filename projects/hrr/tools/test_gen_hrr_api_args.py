#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


GENERATOR = Path(__file__).with_name("gen_hrr_api_args.py")
SPEC = importlib.util.spec_from_file_location("hrr_codegen", GENERATOR)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestManualCaptureCoverage(unittest.TestCase):
    def test_missing_manual_capture_shim_fails(self):
        entry = MODULE.ApiEntry(
            name="hipMemcpy", ret_type="hipError_t", params=[], table="runtime")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp") as source:
            source.write("hipError_t capture_hipOther() { return hipSuccess; }\n")
            source.flush()
            with self.assertRaisesRegex(SystemExit, "manual capture shim definition is missing"):
                MODULE.validate_capture_coverage([entry], Path(source.name), silent=True)

    def test_manual_capture_shim_definition_passes(self):
        entry = MODULE.ApiEntry(
            name="hipMemcpy", ret_type="hipError_t", params=[], table="runtime")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp") as source:
            source.write("hipError_t capture_hipMemcpy() { return hipSuccess; }\n")
            source.flush()
            MODULE.validate_capture_coverage([entry], Path(source.name), silent=True)


if __name__ == "__main__":
    unittest.main()
