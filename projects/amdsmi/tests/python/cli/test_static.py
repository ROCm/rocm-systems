#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: static command (incl. mem-carveout / node GTT display)."""

import csv
import io
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

    def test_asic_revision_id_keys(self):
        """The revision ids must reach JSON and CSV, not only the human output."""
        self.common.print_func_name("")
        keys = ("chip_rev_id", "external_rev_id")

        cmd = "amd-smi static --asic --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        payload = json.loads(data)
        gpus = payload["gpu_data"] if isinstance(payload, dict) else payload
        self.assertTrue(gpus, f"no GPU entries in '{cmd}'")
        for gpu in gpus:
            for key in keys:
                self.assertIn(key, gpu["asic"])
        json_values = {key: [gpu["asic"][key] for gpu in gpus] for key in keys}

        cmd = "amd-smi static --asic --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        # vendor_name carries a comma on some backends, so honour the quoting.
        reader = csv.DictReader(io.StringIO(data))
        rows = [row for row in reader]
        # Compare the values, not just the header, so a dropped field is caught.
        for key in keys:
            self.assertIn(key, reader.fieldnames)
            self.assertEqual([row[key] for row in rows], json_values[key])
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

    def test_power_stabilization(self):
        """Test static --power-stabilization (display mode only, human/json/csv)."""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static --power-stabilization"
        self.common.print(msg)

        ps_fields = (
            "host_control_enabled",
            "enabled",
            "power_floor",
            "delay",
            "insertion_priority",
            "ramp_enabled",
            "ramp_max_ramp_up_rate",
            "ramp_max_ramp_down_rate",
            "ramp_min_time_at_power_floor",
            "ramp_stop_delay",
        )

        ps_baseline = self.static_data["gpu_data"][0]["power_stabilization"]
        if ps_baseline == "N/A":
            self.skipTest("Power Stabilization not supported on gpu 0")

        cmd = "amd-smi static --gpu 0 --power-stabilization"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        self.assertIn("POWER_STABILIZATION", data)

        cmd = "amd-smi static --gpu 0 --power-stabilization --json"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        payload = json.loads(data)
        gpus = payload["gpu_data"] if isinstance(payload, dict) else payload
        self.assertTrue(gpus, f"no GPU entries in '{cmd}'")
        ps = gpus[0]["power_stabilization"]
        self.assertIsInstance(ps, dict, f"unexpected power_stabilization value: {ps!r}")
        for field in ps_fields:
            self.assertIn(field, ps)

        cmd = "amd-smi static --gpu 0 --power-stabilization --csv"
        (rc, data, std_err) = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
        reader = csv.DictReader(io.StringIO(data))
        rows = [row for row in reader]
        self.assertTrue(rows, f"no CSV rows for '{cmd}'")
        for field in ps_fields:
            self.assertIn(f"power_stabilization_{field}", reader.fieldnames)

        # A GPU without power_stabilization support (if any exists on this
        # node) exercises the AMDSMI_STATUS_NOT_SUPPORTED path: distinct text
        # in human output, bare "N/A" in json/csv. Discovered dynamically
        # rather than assumed at a fixed index: which GPU (if any) lacks
        # support is topology-dependent (e.g. an APU, or none on a
        # homogeneous multi-MI4XX node).
        unsupported_gpu = next(
            (
                i
                for i, gpu in enumerate(self.static_data["gpu_data"])
                if gpu.get("power_stabilization") == "N/A"
            ),
            None,
        )
        if unsupported_gpu is None:
            msg = (
                f"{self.tab}Every GPU on this node supports Power Stabilization; "
                "skipping N/A-path checks"
            )
            self.common.print(msg)
        else:
            cmd = f"amd-smi static --gpu {unsupported_gpu} --power-stabilization"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
            self.assertIn(
                "POWER_STABILIZATION: N/A (Power Stabilization is not supported on this ASIC/platform)",
                data,
            )

            cmd = f"amd-smi static --gpu {unsupported_gpu} --power-stabilization --json"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
            payload = json.loads(data)
            gpus = payload["gpu_data"] if isinstance(payload, dict) else payload
            self.assertEqual(gpus[0]["power_stabilization"], "N/A")

            cmd = f"amd-smi static --gpu {unsupported_gpu} --power-stabilization --csv"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
            reader = csv.DictReader(io.StringIO(data))
            rows = [row for row in reader]
            # CSV emits the full column set (not a single bare "power_stabilization"
            # column) even when unsupported, so a mixed node doesn't produce a
            # ragged header.
            for field in ps_fields:
                self.assertEqual(rows[0][f"power_stabilization_{field}"], "N/A")

        msg = f"{self.tab}Static power-stabilization tests passed"
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
