#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: fabric command."""

from cli.base import TestCliBase

# Distinctive fragment of the --topology deprecation warning: matching the whole
# sentence breaks on rewording, matching only "warning" trips on driver noise.
TOPOLOGY_WARNING = "--topology now reports"


class TestFabric(TestCliBase):
    def setUp(self):
        # The fabric subcommand exists only when the amdgpu driver is initialized;
        # without it these assertions report its absence, not the flag contract.
        super().setUp()
        (rc, _, _) = self.util.RunCmdSync("amd-smi fabric --help")
        if rc != 0:
            self.skipTest("fabric subcommand not registered")
        return

    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi fabric"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "fabric", "Fabric arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        self.RunCmds(cmds)
        return

    def test_flag_contract(self):
        self.common.print_func_name("")

        (rc, help_out, _) = self.util.RunCmdSync("amd-smi fabric --help")
        help_out = help_out or ""
        self.assertEqual(0, rc)
        self.assertIn("--topology", help_out)
        self.assertIn("--telemetry", help_out)
        self.assertNotIn("--info", help_out)

        # --info was removed rather than aliased, so it must not parse.
        (rc, _, _) = self.util.RunCmdSync("amd-smi fabric --info --json")
        self.assertNotEqual(0, rc)

        # Routing:
        #   --topology emits the config key
        #   --telemetry the counters key
        # Guards the flag swap, not just that the flags parse.
        # Key presence holds without fabric hardware: the API-error path still stores
        # an "N/A" value under the key.
        # Validating the actual config/counter values needs IFoE/UALink hardware
        # and is deferred.
        (rc, topo_out, topo_err) = self.util.RunCmdSync("amd-smi fabric --topology --json")
        topo_out = topo_out or ""
        self.assertEqual(0, rc)
        self.assertIn("fabric_info", topo_out)
        self.assertNotIn("fabric_telemetry", topo_out)

        # --topology changed meaning, so explicit use warns. Keeping the warning off
        # stdout is what leaves --json and --csv parseable.
        self.assertIn(TOPOLOGY_WARNING, topo_err or "")
        self.assertNotIn(TOPOLOGY_WARNING, topo_out)

        (rc, telem_out, telem_err) = self.util.RunCmdSync("amd-smi fabric --telemetry --json")
        telem_out = telem_out or ""
        self.assertEqual(0, rc)
        self.assertIn("fabric_telemetry", telem_out)
        self.assertNotIn("fabric_info", telem_out)
        self.assertNotIn(TOPOLOGY_WARNING, telem_err or "")

        # The no-flag default turns both flags on internally; that is not a user
        # passing --topology, so it must stay quiet.
        (rc, both_out, both_err) = self.util.RunCmdSync("amd-smi fabric --json")
        both_out = both_out or ""
        self.assertEqual(0, rc)
        self.assertIn("fabric_info", both_out)
        self.assertIn("fabric_telemetry", both_out)
        self.assertNotIn(TOPOLOGY_WARNING, both_err or "")
        return
