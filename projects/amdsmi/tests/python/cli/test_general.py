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
"""CLI leaf test: general behavior (help, invalid)."""

from cli.base import TestCliBase


class TestGeneral(TestCliBase):
    def test_help(self):
        self.common.print_func_name("")
        msg = "### amd-smi help"
        self.common.print(msg)

        cmd = "amd-smi --help"
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        lines = std_out.split("\n") if std_out else []
        # Find all available command line args
        cmd_args = []
        found = False
        cmd_indent = None
        for line in lines:
            if found:
                if not line:
                    break
                indent = len(line) - len(line.lstrip())
                # The first command establishes the command column. Lines that
                # are indented further are wrapped description continuations
                # (e.g. the "devices" tail of the long fabric help text) and
                # must be skipped so they aren't parsed as subcommands.
                if cmd_indent is None:
                    cmd_indent = indent
                elif indent > cmd_indent:
                    continue
                items = line.split()
                cmd_args.append(items[0])
                continue
            if "Descriptions" in line:
                found = True

        cmds = [("amd-smi --help", self.PASS)]
        for cmd_arg in cmd_args:
            cmds.append((f"amd-smi {cmd_arg} --help", self.PASS))

        self.RunCmds(cmds)
        return

    def test_invalid(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi"
        self.common.print(msg)

        # Create bad bdf and uuid gpus
        bdf = self.list_data[0]["bdf"]
        if bdf[-1] == "0":
            bad_bdf = self.list_data[0]["bdf"][:-1] + "1"
        else:
            bad_bdf = self.list_data[0]["bdf"][:-1] + "0"
        uuid = self.list_data[0]["uuid"]
        if uuid[-1] == "0":
            bad_uuid = self.list_data[0]["uuid"][:-1] + "1"
        else:
            bad_uuid = self.list_data[0]["uuid"][:-1] + "0"

        cmds = [
            # Test invalid command
            ("amd-smi invalid_cmd", self.FAIL),
            # Test invalid sub command
            ("amd-smi version --invalid", self.FAIL),
            ("amd-smi list --invalid", self.FAIL),
            ("amd-smi static --invalid", self.FAIL),
            ("amd-smi firmware --invalid", self.FAIL),
            ("amd-smi bad_pages --invalid", self.FAIL),
            ("amd-smi metric --invalid", self.FAIL),
            ("amd-smi process --invalid", self.FAIL),
            ("amd-smi event --invalid", self.FAIL),
            ("amd-smi topology --invalid", self.FAIL),
            ("amd-smi set --invalid", self.FAIL),
            ("amd-smi reset", self.FAIL),
            ("amd-smi reset --invalid", self.FAIL),
            ("amd-smi monitor --invalid", self.FAIL),
            ("amd-smi xgmi --invalid", self.FAIL),
            ("amd-smi partition --invalid", self.FAIL),
            ("amd-smi ras --invalid", self.FAIL),
            ("amd-smi node --invalid", self.FAIL),
            # Test invalid gpu value
            ("amd-smi version --gpu 0", self.FAIL),
            ("amd-smi version --gpu -1", self.FAIL),
            ("amd-smi version --gpu ALL", self.FAIL),
            (f"amd-smi version --gpu {len(self.common.processors)}", self.FAIL),
            ("amd-smi static --gpu -1", self.FAIL),
            ("amd-smi static --gpu _ALL", self.FAIL),
            (f"amd-smi static --gpu {len(self.common.processors)}", self.FAIL),
            (f"amd-smi static --gpu {bad_bdf}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['bdf'][:-1]}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['bdf'] + '0'}", self.FAIL),
            (f"amd-smi static --gpu {bad_uuid}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['uuid'][:-1]}", self.FAIL),
            (f"amd-smi static --gpu {self.list_data[0]['uuid'] + '0'}", self.FAIL),
            # Test invalid loglevel
            ("amd-smi metric --loglevel DDEBUG", self.FAIL),
            ("amd-smi metric --loglevel DEBUGG", self.FAIL),
            ("amd-smi metric --loglevel BADLEVEL", self.FAIL),
            # Test invalid set options
            ("amd-smi set", self.FAIL),
            ("amd-smi set --fan", self.FAIL),
            ("amd-smi set --fan 500", self.FAIL),
            ("amd-smi set --fan 150%", self.FAIL),
            ("amd-smi set --perf-level", self.FAIL),
            ("amd-smi set --perf-level INVALID", self.FAIL),
            ("amd-smi set --profile", self.FAIL),
            ("amd-smi set --profile INVALID", self.FAIL),
            ("amd-smi set --perf-determinism", self.FAIL),
            ("amd-smi set --compute-partition", self.FAIL),
            ("amd-smi set --compute-partition INVALID", self.FAIL),
            ("amd-smi set --memory-partition", self.FAIL),
            ("amd-smi set --memory-partition NPS3", self.FAIL),
            ("amd-smi set --memory-partition INVALID", self.FAIL),
            ("amd-smi set --compute-partition-mem-alloc-mode", self.FAIL),
            ("amd-smi set --compute-partition-mem-alloc-mode HALF", self.FAIL),
            ("amd-smi set --compute-partition-mem-alloc-mode INVALID", self.FAIL),
            ("amd-smi set --process-isolation", self.FAIL),
            ("amd-smi set --process-isolation 2", self.FAIL),
            ("amd-smi set --clk-limit", self.FAIL),
            ("amd-smi set --clk-limit INVALID", self.FAIL),
            ("amd-smi set --clk-limit SCLK INVALID", self.FAIL),
            ("amd-smi set --clk-limit MCLK INVALID", self.FAIL),
            ("amd-smi set --clk-limit SCLK MIN", self.FAIL),
            ("amd-smi set --clk-limit MCLK MAX", self.FAIL),
            ("amd-smi set --clk-level SCLK", self.FAIL),
            ("amd-smi set --clk-level SCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level MCLK", self.FAIL),
            ("amd-smi set --clk-level MCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level FCLK", self.FAIL),
            ("amd-smi set --clk-level FCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level SOCCLK", self.FAIL),
            ("amd-smi set --clk-level SOCCLK INVALID", self.FAIL),
            ("amd-smi set --clk-level PCIE", self.FAIL),
            ("amd-smi set --clk-level PCIE INVALID", self.FAIL),
            # Test invalid process PID, NAME
            ("amd-smi process --name", self.FAIL),
            ("amd-smi process --pid", self.FAIL),
            ("amd-smi process --pid NOT_A_NUMBER", self.FAIL),
            # Test invalid ras options
            ("amd-smi ras", self.FAIL),
            ("amd-smi ras --cper INVALID", self.FAIL),
            ("amd-smi ras --cper --severity INVALID", self.FAIL),
            ("amd-smi ras --afid", self.FAIL),
            ("amd-smi ras --afid INVALID", self.FAIL),
            # --afid --folder requires an existing directory of CPER records
            ("amd-smi ras --afid --folder /nonexistent_amdsmi_pr4812_dir", self.FAIL),
            # --decode is not a valid flag
            ("amd-smi ras --decode", self.FAIL),
            # Test invalid watch order
            ("amd-smi monitor --interval 2 --watch 1", self.FAIL),
            ("amd-smi monitor --watch_time 2 --watch 1", self.FAIL),
        ]

        for index, gpu in enumerate(self.common.processors):
            # Test invalid power-cap values
            cmds.append((f"amd-smi set --power-cap --gpu {index}", self.FAIL))
            for power_type in self.power_types:
                cmds.append((f"amd-smi set --power-cap {power_type} --gpu {index}", self.FAIL))
                _power_type = self.static_data["gpu_data"][index]["limit"][power_type]
                socket_power_limit = _power_type["socket_power_limit"]
                if socket_power_limit != "N/A":
                    min_power = _power_type["min_power_limit"]["value"]
                    max_power = _power_type["max_power_limit"]["value"]
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {min_power - 1} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {max_power + 1} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )
                    cmds.append(
                        (
                            f"amd-smi set --power-cap {int(max_power * 1.10)} {power_type} --gpu {index}",
                            self.FAIL,
                        )
                    )

            # Test invalid soc-pstate values
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                cmds.append((f"amd-smi set --soc-pstate --gpu {index}", self.FAIL))
                num_supported = int(soc_pstate["num_supported"])
                cmds.append((f"amd-smi set --soc-pstate {num_supported} --gpu {index}", self.FAIL))

            # Test invalid xgmi-plpd values
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                cmds.append((f"amd-smi set --xgmi-plpd --gpu {index}", self.FAIL))
                num_supported = int(xgmi_plpd["num_supported"])
                cmds.append((f"amd-smi set --xgmi-plpd {num_supported} --gpu {index}", self.FAIL))

        self.RunCmds(cmds)
        return
