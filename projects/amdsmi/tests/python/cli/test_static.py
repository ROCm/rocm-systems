#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: static command (incl. mem-carveout / node GTT display)."""

import json

from cli.base import TestCliBase


class TestStatic(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "static", "Static Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_mem_carveout_gtt(self):
        """Test static --mem-carveout and node --gtt flags (display mode only)"""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static --mem-carveout and node --gtt"
        self.common.print(msg)

        # Test mem-carveout display (static subcommand)
        cmd = "amd-smi static --mem-carveout"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT display (node subcommand — GTT is system-wide, not per-GPU)
        cmd = "amd-smi node --gtt"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test mem-carveout with JSON output
        cmd = "amd-smi static --mem-carveout --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test GTT with JSON output (node subcommand)
        cmd = "amd-smi node --gtt --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test mem-carveout with CSV output
        cmd = "amd-smi static --mem-carveout --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test GTT with CSV output (node subcommand)
        cmd = "amd-smi node --gtt --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Note: We do NOT test set/reset operations (--mem-carveout in set, --gtt in set/reset) because:
        # 1. They require root/sudo permissions
        # 2. They require system reboot to take effect
        # 3. They could interfere with the test system configuration
        # These operations should be tested manually or in dedicated integration test environments

        msg = f"{self.tab}Static mem-carveout and node GTT tests passed (display mode only)"
        self.common.print(msg)
        return

    def test_tray(self):
        """Test node --tray flag (display/--json/--csv)"""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node --tray"
        self.common.print(msg)

        # Test tray display
        cmd = "amd-smi node --tray"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        # Test tray with JSON output
        cmd = "amd-smi node --tray --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        if data:
            try:
                json_data = json.loads(data)
                self.assertIsInstance(json_data, (list, dict))
                node_info = json_data.get("node", {}) if isinstance(json_data, dict) else {}
                tray_info = node_info.get("tray")
                if tray_info:
                    self.assertIn("max_acc_per_tray", tray_info)
                    self.assertIn("tray_type", tray_info)
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON output for command '{cmd}'")

        # Test tray with CSV output
        cmd = "amd-smi node --tray --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

        msg = f"{self.tab}Node tray tests passed"
        self.common.print(msg)
        return
