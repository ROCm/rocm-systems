#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: ras command (incl. --afid --folder fixtures)."""

import json
import os
import shutil
import tempfile

from cli.base import TestCliBase


class TestRas(TestCliBase):
    def test_command(self):
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

    def test_afid_folder(self):
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
            self.assertIsNotNone(data, f"Command '{cmd}' produced no output")
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
