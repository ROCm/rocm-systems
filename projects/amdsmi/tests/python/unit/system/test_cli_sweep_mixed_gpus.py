#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free checks of the CLI sweep on hosts whose GPUs differ, such as a dGPU
next to an iGPU that lacks some clock tables and gpu_metrics.
"""

import unittest
from unittest import mock

from common.common import amdsmi

try:
    from cli.base import AmdSmiExitCode, TestCliBase
except (ImportError, AttributeError) as exc:  # needs the amd-smi CLI under test
    _CLI_SKIP = f"amd-smi CLI unavailable: {exc}"
else:
    _CLI_SKIP = None

NOT_SUPPORTED = int(amdsmi.AmdSmiStatus.NOT_SUPPORTED)
INVAL = int(amdsmi.AmdSmiStatus.INVAL)
UNEXPECTED_DATA = int(amdsmi.AmdSmiStatus.UNEXPECTED_DATA)


@unittest.skipIf(_CLI_SKIP, _CLI_SKIP)
class TestCliSweepMixedGpus(unittest.TestCase):
    def _run_cmd(self, cmd, cond, codes, gpu_count=2):
        """Grade *cmd* through RunCmds; *codes* maps each command line to its exit code.

        Returns:
            tuple: (whether RunCmds passed, the command lines it ran).
        """
        sweep = TestCliBase()
        sweep.common = mock.Mock()
        sweep.list_data = [{"gpu": index} for index in range(gpu_count)]
        ran = []

        def run(command, **_):
            ran.append(command)
            return codes[command], "", ""

        sweep.util = mock.Mock(RunCmdSync=run)
        try:
            sweep.RunCmds([(cmd, cond)])
        except AssertionError:
            return False, ran
        return True, ran

    def test_mixed_codes_pass_when_each_gpu_answers_an_allowed_code(self):
        cmd = "amd-smi set --memory-partition NPS2 --loglevel DEBUG"
        codes = {
            cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS),
            f"{cmd} --gpu 0": INVAL,
            f"{cmd} --gpu 1": NOT_SUPPORTED,
        }
        passed, ran = self._run_cmd(cmd, [0, NOT_SUPPORTED, INVAL], codes)
        self.assertTrue(passed)
        self.assertEqual(ran, [cmd, f"{cmd} --gpu 0", f"{cmd} --gpu 1"])

    def test_mixed_codes_fail_when_a_gpu_answers_a_code_not_allowed(self):
        cmd = "amd-smi set --clk-level FCLK 0 1 2 --loglevel DEBUG"
        codes = {
            cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS),
            f"{cmd} --gpu 0": NOT_SUPPORTED,
            f"{cmd} --gpu 1": UNEXPECTED_DATA,
        }
        passed, _ = self._run_cmd(cmd, [0, NOT_SUPPORTED], codes)
        self.assertFalse(passed)

    def test_gpu_all_is_rerun_once_per_gpu(self):
        cmd = "amd-smi set --clk-level SOCCLK 0 1 --gpu all --loglevel DEBUG"
        per_gpu = "amd-smi set --clk-level SOCCLK 0 1 --loglevel DEBUG --gpu"
        codes = {
            cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS),
            f"{per_gpu} 0": NOT_SUPPORTED,
            f"{per_gpu} 1": 0,
        }
        passed, ran = self._run_cmd(cmd, [0, NOT_SUPPORTED], codes)
        self.assertTrue(passed)
        self.assertEqual(ran, [cmd, f"{per_gpu} 0", f"{per_gpu} 1"])

    def test_mixed_codes_on_one_named_gpu_are_not_rerun(self):
        cmd = "amd-smi reset --clocks --gpu 0"
        passed, ran = self._run_cmd(
            cmd, [0, NOT_SUPPORTED], {cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)}
        )
        self.assertFalse(passed)
        self.assertEqual(ran, [cmd])

    def test_mixed_codes_on_a_single_gpu_host_are_not_rerun(self):
        cmd = "amd-smi reset --clocks"
        passed, ran = self._run_cmd(
            cmd, [0, NOT_SUPPORTED], {cmd: int(AmdSmiExitCode.MIXED_DEVICE_ERRORS)}, gpu_count=1
        )
        self.assertFalse(passed)
        self.assertEqual(ran, [cmd])

    def test_all_gpu_clk_level_is_dropped_when_a_gpu_has_no_table(self):
        sweep = TestCliBase()
        sweep.clk_freq = [{"FCLK": {"num_supported": 3, "has_deep_sleep": False}}, {"FCLK": None}]
        self.assertEqual(sweep._clk_level_count("FCLK", 0, explicit_gpu=False), 0)
        self.assertEqual(sweep._clk_level_count("FCLK", 0, explicit_gpu=True), 3)

    def test_all_gpu_clk_level_fits_the_smallest_table(self):
        sweep = TestCliBase()
        sweep.clk_freq = [
            {"SOCCLK": {"num_supported": 4, "has_deep_sleep": True}},
            {"SOCCLK": {"num_supported": 2, "has_deep_sleep": False}},
        ]
        self.assertEqual(sweep._clk_level_count("SOCCLK", 0, explicit_gpu=False), 2)

    def test_clock_table_is_kept_when_clock_info_is_not_supported(self):
        dgpu, igpu = object(), object()

        def clk_freq(handle, clk_type):
            if handle is igpu and clk_type == amdsmi.AmdSmiClkType.DF:
                raise amdsmi.AmdSmiLibraryException(UNEXPECTED_DATA)
            return {"num_supported": 3}

        def clock_info(handle, _clk_type):
            if handle is igpu:
                raise amdsmi.AmdSmiLibraryException(NOT_SUPPORTED)
            return {"clk_deep_sleep": 28}

        library = mock.patch.multiple(
            amdsmi,
            amdsmi_get_processor_handles=lambda: [dgpu, igpu],
            amdsmi_get_clk_freq=clk_freq,
            amdsmi_get_clock_info=clock_info,
            amdsmi_shut_down=lambda: None,
        )
        with library, mock.patch.object(TestCliBase, "common", mock.Mock(), create=True):
            tables = TestCliBase._read_clk_freq()

        self.assertTrue(tables[0]["SOCCLK"]["has_deep_sleep"])
        self.assertEqual(tables[1]["SOCCLK"], {"num_supported": 3, "has_deep_sleep": False})
        self.assertIsNone(tables[1]["FCLK"])
