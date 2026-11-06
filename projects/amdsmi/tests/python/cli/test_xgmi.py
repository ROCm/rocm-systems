#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: xgmi command."""

import csv
import io

from cli.base import TestCliBase


class TestXgmi(TestCliBase):
    def test_multiple_gpus_csv(self) -> None:
        if len(self.list_data) < 2:
            self.skipTest("requires at least two GPUs")
        selected = [entry["gpu"] for entry in self.list_data[:2]]
        cmd = f"amd-smi xgmi -g {' '.join(map(str, selected))} --csv"
        rc, output, stderr = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
        tables = output.strip().split("\n\n")
        self.assertTrue(output.strip(), cmd)
        for table in tables:
            with self.subTest(header=table.splitlines()[0]):
                reader = csv.DictReader(io.StringIO(table))
                self.assertIn("gpu", reader.fieldnames)
                rows = list(reader)
                self.assertTrue(rows, cmd)
                self.assertTrue(all(None not in row and None not in row.values() for row in rows))
                self.assertEqual(
                    {
                        int(row["gpu"][3:] if row["gpu"].startswith("GPU") else row["gpu"])
                        for row in rows
                    },
                    set(selected),
                )

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi xgmi"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "xgmi", "XGMI arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
