#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""CLI topology commands: topology, xgmi, partition, ras, node."""

import ctypes
import json
import os
import shutil
import stat
import tempfile
import unittest

import common.common as common
import common.runcmd as runcmd
from cli.base import TestCliBase

# common.common owns path resolution, sys.path setup, and amdsmi loading — borrow the
# reference so AMDSMI_PATH/ROCM_HOME/ROCM_PATH resolution and the stale-package check
# (see ROCM-1552 / PR #6359) are not duplicated or bypassed here.
from common.common import amdsmi


class TestCliTopology(TestCliBase):
    TMP_FILENAME = "_tmp.log"
    TMP_FOLDER = "_tmp"

    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)
        cls.util = runcmd.Util("WARNING")

        # Record starting values; running here (once per class) rather than in
        # __init__ (once per test method) reduces setup overhead from O(N) to
        # O(1) — N being the number of test methods in this class.
        cmds = [
            ("metric", "amd-smi metric --json"),
            ("static", "amd-smi static --json"),
            ("list", "amd-smi list --json"),
            ("partition", "amd-smi partition --current --json"),
        ]
        for name, cmd in cmds:
            (rc, data, std_err) = cls.util.RunCmdSync(cmd)
            if rc:
                raise RuntimeError(f'Error executing "{cmd}": {std_err}')
            if not data:
                raise RuntimeError(f'Empty JSON output from "{cmd}". stderr: {std_err}')
            try:
                setattr(cls, f"{name}_data", json.loads(data))
            except (json.JSONDecodeError, TypeError) as e:
                # TODO(amdsmi_team): Known issue — several AI NIC and CPU commands can produce
                # malformed JSON/CSV/error output, causing parsing & other failures.
                # We need to log tickets on these issues.

                # Log warning but continue — malformed JSON output is a CLI bug,
                # not a test infrastructure failure; tests that depend on this
                # data will fail individually with a KeyError pointing to the
                # missing key, making the root cause clear.
                cls.common.print(f'\n\tERROR: Could not parse JSON from "{cmd}": {e}')
                setattr(cls, f"{name}_data", {})

        cls.gpus = ["all"]
        for entry in cls.list_data:
            cls.gpus.append(entry["gpu"])
            if entry["gpu"] == 0:
                # Only test bdf and uuid when gpu=0
                cls.gpus.append(entry["bdf"])
                cls.gpus.append(entry["uuid"])

        # When parsing, expand each arg with array element
        cls.sub_args = {
            "CLOCK": ["SYS", "DF", "DCEF", "SOC", "MEM", "VCLK0", "VCLK1", "DCLK0", "DCLK1", "ALL"],
            "PID": [123],
            "NAME": ["AMD"],
            "GPU": cls.gpus,
            "FILE": [
                cls.TMP_FILENAME,
                f"{cls.TMP_FILENAME} --overwrite",
                f"{cls.TMP_FILENAME} --append",
            ],
            "SEVERITY": ["nonfatal-uncorrected", "fatal", "nonfatal-corrected", "all"],
            "FOLDER": [cls.TMP_FOLDER],
            "FILE_LIMIT": [10],
            #'LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        }

    def test_node(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "node", "Node arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_partition(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi partition"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "partition", "Partition arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_ras(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi ras"
        self.common.print(msg)

        # TODO Yazen
        # TODO allow event commands to be executed
        if not self.PrintCmdsOnly:
            if self.common.TODO_SKIP_FAIL:
                msg = f"{self.tab}Not Yet Implemented"
                # self.common.print(msg)
                self.skipTest(msg)

        cmds = self.CreateCmds(
            "ras", "RAS arguments:", "CPER Arguments", "Device Arguments:", "Command Modifiers:"
        )
        self.RunCmds(cmds)
        return

    def test_ras_afid_folder(self):
        """Exercise the pure-Python validation/decode branches of
        ``amd-smi ras --afid --folder`` against on-disk fixtures (no GPU needed).
        """
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi ras --afid --folder"
        self.common.print(msg)

        if self.PrintCmdsOnly:
            return

        tmp_dir = tempfile.mkdtemp(prefix="amdsmi_ras_afid_")
        try:
            # A real but undecodable .cper (non-empty garbage bytes): decodes to
            # "decode failed" in the table, but the command still exits 0.
            garbage = os.path.join(tmp_dir, "garbage.cper")
            with open(garbage, "wb") as fout:
                fout.write(b"\x00" * 64)

            # An existing folder with no .cper files.
            empty_dir = os.path.join(tmp_dir, "empty")
            os.mkdir(empty_dir)

            # A symlink to a folder that does contain a .cper, so only the
            # symlink rejection (not a missing-file error) can make this FAIL.
            target_dir = os.path.join(tmp_dir, "target")
            os.mkdir(target_dir)
            with open(os.path.join(target_dir, "g.cper"), "wb") as fout:
                fout.write(b"\x00" * 64)
            symlink_dir = os.path.join(tmp_dir, "link")
            os.symlink(target_dir, symlink_dir)

            cmds = [
                # Folder with a (garbage) .cper: undecodable rows are reported but
                # the command exits 0.
                (f"amd-smi ras --afid --folder {tmp_dir}", self.PASS),
                # --cper-file and --folder are mutually exclusive under --afid.
                (f"amd-smi ras --afid --cper-file {garbage} --folder {tmp_dir}", self.FAIL),
                # Nonexistent folder.
                (f"amd-smi ras --afid --folder {os.path.join(tmp_dir, 'nope')}", self.FAIL),
                # Existing folder with no .cper files.
                (f"amd-smi ras --afid --folder {empty_dir}", self.FAIL),
                # Symlinked folder is refused even though it contains a .cper.
                (f"amd-smi ras --afid --folder {symlink_dir}", self.FAIL),
            ]
            self.RunCmds(cmds)

            # The --json output must be a flat list of per-file objects, not a
            # doubly-wrapped [[...]]. Guards against the logger wrapping a list
            # assigned to self.output a second time.
            cmd = f"amd-smi ras --afid --folder {tmp_dir} --json"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
            json_data = json.loads(data)
            self.assertIsInstance(json_data, list, f"'{cmd}' did not emit a JSON list")
            for entry in json_data:
                self.assertIsInstance(
                    entry,
                    dict,
                    f"'{cmd}' emitted a non-object element (double-wrapped?): {entry!r}",
                )
                self.assertIn("cper_file", entry)
                self.assertIn("afids", entry)
                self.assertIn("decode_failed", entry)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    def test_topology(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi topology"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "topology", "Topology arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_xgmi(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi xgmi"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "xgmi", "XGMI arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_fabric(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi fabric"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "fabric", "Fabric arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
