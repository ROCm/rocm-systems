#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: firmware (and ucode) command."""

import json

from common.common import amdsmi

from cli.base import TestCliBase


class TestFirmware(TestCliBase):
    def test_pldm_bundle(self) -> None:
        cmd = "amd-smi firmware --json"
        rc, output, stderr = self.util.RunCmdSync(cmd)
        self.assertEqual(rc, self.PASS, f"{cmd}: {stderr}")
        entries = {entry["gpu"]: entry for entry in json.loads(output)}
        supported = 0
        self.common.amdsmi_smart_init()
        try:
            for gpu, processor in enumerate(amdsmi.amdsmi_get_processor_handles()):
                try:
                    firmware = amdsmi.amdsmi_get_fw_info(processor)["fw_list"]
                except amdsmi.AmdSmiLibraryException as exc:
                    if exc.err_code == amdsmi.AmdSmiStatus.NOT_SUPPORTED:
                        continue
                    raise
                for entry in firmware:
                    if entry["fw_name"].name == "AMDSMI_FW_ID_PLDM_BUNDLE":
                        supported += 1
                        reported = {
                            item["fw_id"]: item["fw_version"] for item in entries[gpu]["fw_list"]
                        }
                        self.assertIn("PLDM_BUNDLE", reported)
                        self.assertEqual(reported["PLDM_BUNDLE"], entry["fw_version"])
        finally:
            amdsmi.amdsmi_shut_down()
        if not supported:
            self.skipTest("no GPU reports PLDM_BUNDLE firmware")

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi firmware"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "firmware", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        cmds = self.CreateCmds(
            "ucode", "Firmware Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return
