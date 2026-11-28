#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: node command."""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from cli.base import TestCliBase
from common.common import amdsmi_path


class TestNode(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi node"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "node", "Node arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return


class TestNodeCuidNamespace(unittest.TestCase):
    @unittest.skipUnless(shutil.which("unshare") and shutil.which("mount"), "needs util-linux")
    def test_missing_machine_id_exits_with_no_data(self):
        cli = str(Path(amdsmi_path).resolve().parents[1] / "bin" / "amd-smi")
        cmd = [cli, "node", "--cuid", "--json"]
        normal = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True, timeout=30)
        if (
            normal.returncode == 0
            and json.loads(normal.stdout)[0]["node"]["cuid"]["components"] == "N/A"
        ):
            self.skipTest("built without CUID support")
        ns = ["unshare", "--mount", "--propagation", "private"]
        if os.geteuid() != 0:
            ns += ["--user", "--map-root-user"]
        if subprocess.run(ns + ["true"]).returncode != 0:
            self.skipTest("cannot create a private mount namespace")
        hide_ids = (
            "for p in /etc/machine-id /var/lib/dbus/machine-id; do "
            'if [ -e "$p" ]; then mount --bind /dev/null "$p" || exit 77; fi; done; exec "$@"'
        )
        hidden = subprocess.run(
            ns + ["sh", "-c", hide_ids, "sh"] + cmd,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
        if hidden.returncode == 77:
            self.skipTest("bind mounts are not permitted")
        self.assertEqual(hidden.returncode, 40, hidden.stdout)
        self.assertIn(
            "AMDSMI_STATUS_NO_DATA", json.loads(hidden.stdout)[0]["node"]["cuid"]["components"]
        )
