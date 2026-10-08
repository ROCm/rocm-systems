#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise the control CLI against the loaded shared library, without a GPU."""

import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parent
LIBRARY = Path(sys.argv.pop(1)).resolve()
PROBE = Path(sys.argv.pop(1)).resolve()
SUCCESS, INVAL, NOT_SUPPORTED, NOT_FOUND = 0, 1, 2, 31


class ErrorCount(ctypes.Structure):
    _fields_ = [
        ("correctable", ctypes.c_uint64),
        ("uncorrectable", ctypes.c_uint64),
        ("deferred", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 5),
    ]


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="amdsmi-mock-control-")
        self.directory = Path(self.temporary.name)
        self.previous_directory = os.environ.get("AMDSMI_MOCK_STATE_DIR")
        os.environ["AMDSMI_MOCK_STATE_DIR"] = str(self.directory)
        self.lib = ctypes.CDLL(str(LIBRARY))
        self.lib.amdsmi_init.argtypes = [ctypes.c_uint64]
        self.lib.amdsmi_shut_down.argtypes = []
        self.lib.amdsmi_get_socket_handles.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.amdsmi_get_processor_handles.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.amdsmi_get_gpu_total_ecc_count.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ErrorCount),
        ]
        self.lib.amdsmi_get_temp_metric.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.initialized = False

    def tearDown(self):
        if self.initialized:
            self.assertEqual(self.lib.amdsmi_shut_down(), SUCCESS)
        if self.previous_directory is None:
            os.environ.pop("AMDSMI_MOCK_STATE_DIR", None)
        else:
            os.environ["AMDSMI_MOCK_STATE_DIR"] = self.previous_directory
        self.temporary.cleanup()

    def control(self, *arguments, success=True):
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "control.py"),
                "--state-dir",
                str(self.directory),
                *arguments,
            ],
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def discover(self):
        self.assertEqual(self.lib.amdsmi_init(2), SUCCESS)
        self.initialized = True
        count = ctypes.c_uint32(1)
        socket = ctypes.c_void_p()
        self.assertEqual(
            self.lib.amdsmi_get_socket_handles(ctypes.byref(count), ctypes.byref(socket)), SUCCESS
        )
        count.value = 0
        self.assertEqual(
            self.lib.amdsmi_get_processor_handles(socket, ctypes.byref(count), None), SUCCESS
        )
        devices = (ctypes.c_void_p * count.value)()
        self.assertEqual(
            self.lib.amdsmi_get_processor_handles(socket, ctypes.byref(count), devices), SUCCESS
        )
        return devices

    def ecc(self, device):
        result = ErrorCount()
        self.assertEqual(
            self.lib.amdsmi_get_gpu_total_ecc_count(device, ctypes.byref(result)), SUCCESS
        )
        return result.correctable, result.uncorrectable

    def test_live_fault_and_reset_through_c_abi(self):
        self.control("init", "--gpus", "2", "--total-mib", "98304")
        baseline = json.loads(self.control("show").stdout)
        devices = self.discover()
        self.assertEqual(len(devices), 2)
        self.assertEqual(self.ecc(devices[0]), (0, 0))
        self.control(
            "set",
            "--gpu",
            "0",
            "--temperature-c",
            "95",
            "--utilization",
            "100",
            "--correctable",
            "3",
            "--uncorrectable",
            "7",
        )
        temperature = ctypes.c_int64()
        self.assertEqual(
            self.lib.amdsmi_get_temp_metric(devices[0], 0, 0, ctypes.byref(temperature)), SUCCESS
        )
        self.assertEqual(temperature.value, 95)
        self.assertEqual(self.ecc(devices[0]), (3, 7))
        self.assertEqual(self.ecc(devices[1]), (0, 0))
        self.control("reset", "--gpu", "0")
        self.assertEqual(self.ecc(devices[0]), (0, 0))
        self.assertEqual(json.loads(self.control("show").stdout), baseline)

    def test_example_consumer(self):
        self.control("init", "--gpus", "2")
        self.control("set", "--gpu", "0", "--temperature-c", "95", "--uncorrectable", "7")
        output = subprocess.check_output([str(PROBE)], text=True).splitlines()
        self.assertEqual(
            output,
            [
                "gpu=0 temperature_c=95 utilization_percent=0 power_w=100 vram_used_mib=0 ecc_correctable=0 ecc_uncorrectable=7",
                "gpu=1 temperature_c=45 utilization_percent=0 power_w=100 vram_used_mib=0 ecc_correctable=0 ecc_uncorrectable=0",
            ],
        )

    def test_invalid_updates_preserve_last_state(self):
        self.control("init")
        before = (self.directory / "gpu0").read_bytes()
        for arguments in (
            ("--utilization", "101"),
            ("--temperature-c", "-1"),
            ("--used-mib", "196609"),
            ("--power-w", str(2**32 - 1)),
            ("--uncorrectable", str(2**64)),
            (),
        ):
            self.control("set", "--gpu", "0", *arguments, success=False)
            self.assertEqual((self.directory / "gpu0").read_bytes(), before)
        self.control("set", "--gpu", "1", "--utilization", "50", success=False)
        self.control("init", success=False)
        self.assertEqual((self.directory / "gpu0").read_bytes(), before)

    def test_all_public_gpu_symbols_are_exported(self):
        # The build generator sees the preprocessed header, so CPU-only APIs
        # gated by ENABLE_ESMI_LIB are intentionally excluded.
        header = subprocess.check_output(
            [
                os.environ.get("CC", "cc"),
                "-E",
                "-P",
                "-x",
                "c",
                str(HERE / "../../include/amd_smi/amdsmi.h"),
            ],
            text=True,
        )
        symbols = set(re.findall(r"amdsmi_status_t\s+(amdsmi_\w+)\s*\(", header))
        self.assertGreater(len(symbols), 100)
        for symbol in symbols:
            with self.subTest(symbol=symbol):
                self.assertIsNotNone(getattr(self.lib, symbol))
        self.lib.amdsmi_reset_gpu.argtypes = [ctypes.c_void_p]
        self.assertEqual(self.lib.amdsmi_reset_gpu(None), NOT_SUPPORTED)

    def test_atomic_updates_during_reads(self):
        self.control("init")
        device = self.discover()[0]
        stop = threading.Event()
        failures = []

        def read():
            while not stop.is_set():
                value = ErrorCount()
                status = self.lib.amdsmi_get_gpu_total_ecc_count(device, ctypes.byref(value))
                if status != SUCCESS or value.correctable != value.uncorrectable:
                    failures.append((status, value.correctable, value.uncorrectable))
                    break

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for value in range(1, 6):
                self.control(
                    "set", "--gpu", "0", "--correctable", str(value), "--uncorrectable", str(value)
                )
        finally:
            stop.set()
            reader.join()
        self.assertEqual(failures, [])
        self.assertEqual(self.ecc(device), (5, 5))

    def test_missing_state_is_reported(self):
        self.control("init")
        device = self.discover()[0]
        (self.directory / "gpu0").unlink()
        value = ErrorCount()
        self.assertEqual(
            self.lib.amdsmi_get_gpu_total_ecc_count(device, ctypes.byref(value)), NOT_FOUND
        )

    def test_empty_fixture_and_count_limit(self):
        self.control("init", "--gpus", "33", success=False)
        self.control("init", "--gpus", "0")
        self.assertEqual(self.lib.amdsmi_init(2), SUCCESS)
        self.initialized = True
        count = ctypes.c_uint32(99)
        self.assertEqual(self.lib.amdsmi_get_socket_handles(ctypes.byref(count), None), SUCCESS)
        self.assertEqual(count.value, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
