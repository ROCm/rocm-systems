#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: static command (incl. mem-carveout / node GTT display)."""

import csv
import io
import json
import os
import tempfile

from cli.base import TestCliBase

# CSV expands NUMA affinity into one row per CPU list under these names.
NUMA_CSV_COLUMNS = {
    "cpu_list": None,
    "bitmask": ("cpu_affinity", "bitmask"),
    "cpu_cores_affinity": ("cpu_affinity", "cpu_cores_affinity"),
    "socket_bitmask": ("socket_affinity", "bitmask"),
    "socket_cpu_cores_affinity": ("socket_affinity", "cpu_cores_affinity"),
}


class TestStatic(TestCliBase):
    def test_numa_json(self) -> None:
        cmd = "amd-smi static --numa --json"
        rc, output, stderr = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
        entries = json.loads(output)["gpu_data"]
        self.assertEqual(
            {entry["gpu"] for entry in entries}, {entry["gpu"] for entry in self.list_data}
        )
        for entry in entries:
            self.assertIn("numa", entry, cmd)

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi static"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "static", "Static Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def _assert_numa_rows(self, json_entries, csv_rows):
        """Each GPU's CSV NUMA rows carry the affinity its JSON output reports."""
        for entry in json_entries:
            numa = entry.get("numa")
            if not isinstance(numa, dict):
                continue
            gpu = entry["gpu"]
            rows = {row["cpu_list"]: row for row in csv_rows if int(row["gpu"]) == gpu}
            cpu_lists = set()
            for scope in ("cpu_affinity", "socket_affinity"):
                if isinstance(numa[scope], dict):
                    cpu_lists |= set(numa[scope])
            self.assertEqual(set(rows), cpu_lists or {"N/A"}, f"GPU {gpu} NUMA CSV rows")
            for cpu_list, row in rows.items():
                for column, source in NUMA_CSV_COLUMNS.items():
                    if source is None:
                        continue
                    scope, field = source
                    affinity = numa[scope]
                    expected = "N/A"
                    if isinstance(affinity, dict) and cpu_list in affinity:
                        expected = str(affinity[cpu_list][field])
                    self.assertEqual(row[column], expected, f"GPU {gpu} {cpu_list} {column}")

    def test_csv(self):
        """`amd-smi static --csv` is well-formed CSV agreeing with JSON for every GPU."""
        self.common.print_func_name("")
        json_entries, csv_rows = self.assert_json_csv_consistent(
            "amd-smi static", skip=NUMA_CSV_COLUMNS
        )
        self.assert_gpu_ids([row["gpu"] for row in csv_rows], "amd-smi static --csv")
        self._assert_numa_rows(json_entries, csv_rows)

    def test_csv_file(self):
        """`amd-smi static --csv --file` writes the same parseable table as stdout."""
        self.common.print_func_name("")
        stdout_rows = self.run_csv("amd-smi static --csv")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "static.csv")
            cmd = f"amd-smi static --csv --file {path}"
            self.assertEqual(self.run_ok(cmd).strip(), "", f"'{cmd}' also wrote to stdout")
            self.assertTrue(os.path.isfile(path), f"'{cmd}' did not create {path}")
            with open(path, encoding="utf-8") as fin:
                file_rows = self.parse_csv(fin.read(), cmd)
        self.assertEqual(file_rows, stdout_rows, "--file CSV differs from stdout CSV")

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
