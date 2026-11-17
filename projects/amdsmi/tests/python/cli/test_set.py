#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: set command."""

from common.common import amdsmi

from cli.base import TestCliBase


def _strip_prefix(value, prefix):
    # Backport of str.removeprefix() (added in Python 3.9) because the test suite
    # still supports Python 3.8. Strips a leading enum-name prefix (e.g.
    # "AMDSMI_DEV_PERF_LEVEL_") when present, otherwise returns value unchanged.
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


class TestSet(TestCliBase):
    def test_command(self):
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set"
        self.common.print(msg)

        cmds = self.CreateCmds(
            "set", "Set Arguments:", "Device Arguments:", "Command Modifiers:", ""
        )
        # Registered before the sweep: RunCmds raises on the first failure, and a
        # sweep that aborts partway is exactly when the GPU is left mid-change.
        self.addCleanup(self._restore_starting_values)
        self.RunCmds(cmds)

        return

    def _restore_starting_values(self):
        """Put the values the sweep changed back to what setUpClass recorded."""
        cmds, problems = self._restore_commands()
        self.common.print("Restore Starting Values")
        # Run every restore command before reporting anything, so a bad baseline
        # field never leaves the GPU in the state the sweep put it in.
        errors = []
        try:
            self.RunCmds(cmds)
        except AssertionError as e:
            errors.append(str(e))
        errors.extend(problems)
        if errors:
            self.fail("\n".join(errors))

    def _restore_commands(self):
        """Commands that restore the recorded values, and baseline problems found.

        Returns:
            tuple: (list of (cmd, cond), list of problem messages).
        """
        cmds = []
        problems = []
        for index, gpu in enumerate(self.common.processors):
            # The bare-metal default metric dump omits the fan section on an APU and
            # reports it on a dGPU, with "N/A" fields when there is no fan.
            # A dGPU fan is present when max is not "N/A". max is the constant pwm1_max,
            # which the driver hides only along with every other fan attribute, pwm1_enable
            # included (an APU, or a dGPU with no fan). Speed is not used: reading pwm1
            # needs the device active, so a runtime-suspended dGPU reports it as "N/A".
            fan = self.metric_data["gpu_data"][index].get("fan")
            if self.is_apu[index]:
                if fan is not None:
                    problems.append(f"GPU {index}: APU metric dump has a fan section: {fan}")
                fan_reset = [amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            elif not isinstance(fan, dict) or "max" not in fan or "speed" not in fan:
                problems.append(f"GPU {index}: dGPU metric dump has no fan max/speed: {fan}")
                fan_reset = [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            elif fan["max"] == "N/A":
                fan_reset = [amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            else:
                # max fan speed is the hwmon pwm1_max ceiling on every interface
                fan_max = fan["max"]
                numeric = isinstance(fan_max, (int, float)) and not isinstance(fan_max, bool)
                if not numeric or not 0 < fan_max <= 255:
                    problems.append(f"GPU {index}: max fan speed {fan_max!r} is not in 1-255")
                fan_reset = [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
            # reset --fans (works for both legacy hwmon and gpu_od interfaces)
            # A present fan is not necessarily controllable, so NOT_SUPPORTED is allowed.
            cmds.append((f"amd-smi reset --fans --gpu {index}", fan_reset))

            # set --profile defaults
            # Read from the static baseline: the library is shut down once setup
            # ends, so querying it here fails and the restore was skipped.
            profile = self.static_data["gpu_data"][index].get("profile")
            if isinstance(profile, dict) and profile["current"] in profile["available_profiles"]:
                cmds.append(
                    (f"amd-smi set --profile {profile['current']} --gpu {index}", self.PASS)
                )

            # set --perf-determinism defaults
            clock_sys = self.static_data["gpu_data"][index]["clock"]["sys"]
            if clock_sys != "N/A":
                num = len(clock_sys["frequency_levels"])
                level = f"Level {num - 1}"
                clock_freq = int(clock_sys["frequency_levels"][level]["value"])
                # Readable clock levels do not imply the determinism feature exists.
                cmds.append(
                    (
                        f"amd-smi set --perf-determinism {clock_freq} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

            # set --memory-partition defaults
            # Safe to write back: the mode only takes effect after a driver reload.
            memory_partition = self.partition_data["current_partition"][index]["memory"]
            if memory_partition != "N/A":
                cmds.append(
                    (f"amd-smi set --memory-partition {memory_partition} --gpu {index}", self.PASS)
                )

            # set --compute-partition defaults
            # Must come after memory partition sets, since the memory partition set can override
            # this setting (their defaults may differ).
            accelerator_type = self.partition_data["current_partition"][index]["accelerator_type"]
            if accelerator_type != "N/A":
                cmds.append(
                    (f"amd-smi set --compute-partition {accelerator_type} --gpu {index}", self.PASS)
                )

            # set --compute-partition-mem-alloc-mode defaults
            try:
                mem_alloc_mode = self.static_data["gpu_data"][index]["partition"][
                    "compute_partition_mem_alloc_mode"
                ]
            except (KeyError, TypeError):
                mem_alloc_mode = "N/A"
            if mem_alloc_mode not in ("N/A", "INVALID"):
                cmds.append(
                    (
                        f"amd-smi set --compute-partition-mem-alloc-mode {mem_alloc_mode} --gpu {index}",
                        self.PASS,
                    )
                )

            # reset --power-cap
            # Writes each supported sensor's default_power_cap. Replaying the recorded
            # value did not restore it: min and max were written after it, so the cap
            # was left at max. A GPU with no supported cap sensor (e.g. an APU) has no
            # power#_cap to write.
            limits = self.static_data["gpu_data"][index]["limit"]
            reports_cap = any(
                isinstance(limits[sensor], dict) and limits[sensor]["max_power_limit"] != "N/A"
                for sensor in self.power_types
            )
            cmds.append(
                (
                    f"amd-smi reset --power-cap --gpu {index}",
                    self.PASS if reports_cap else [amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                )
            )

            # set --soc-pstate defaults
            soc_pstate = self.static_data["gpu_data"][index]["soc_pstate"]
            if soc_pstate != "N/A":
                current = int(soc_pstate["current_id"])
                cmds.append((f"amd-smi set --soc-pstate {current} --gpu {index}", self.PASS))

            # set --xgmi-plpd defaults
            xgmi_plpd = self.static_data["gpu_data"][index]["xgmi_plpd"]
            if xgmi_plpd != "N/A":
                current = int(xgmi_plpd["current_id"])
                cmds.append((f"amd-smi set --xgmi-plpd {current} --gpu {index}", self.PASS))

            # set --ptl-status defaults
            ptl_state = self.static_data["gpu_data"][index]["limit"]["ptl_state"]
            if ptl_state != "N/A":
                if ptl_state == "Disabled":
                    ptl_state_value = 0
                else:
                    ptl_state_value = 1
                cmds.append(
                    (f"amd-smi set --ptl-status {ptl_state_value} --gpu {index}", self.PASS)
                )

            # set --ptl-format defaults
            ptl_format = self.static_data["gpu_data"][index]["limit"]["ptl_format"]
            if ptl_format != "N/A":
                # TODO: get the right ptl-format
                cmds.append((f"amd-smi set --ptl-format {ptl_format} --gpu {index}", self.PASS))

            # set --clk-limit defaults
            clock = self.metric_data["gpu_data"][index]["clock"]
            for clk_type in self.clk_limits:
                for limit_type in self.limit_types:
                    value = self._clk_limit_value(clock, clk_type, limit_type)
                    if value is not None:
                        # A readable min/max clock does not imply the device
                        # supports setting a limit on it.
                        cmds.append(
                            (
                                f"amd-smi set --clk-limit {clk_type} {limit_type} {value} --gpu {index}",
                                [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                            )
                        )

            # reset --clocks
            # A clk-level write is a bitmask, so replaying the recorded level would pin DPM
            # to that one level; only AUTO hands every level back. Overdrive is reset too,
            # and reports NOT_SUPPORTED on parts that lack it even though the rest worked.
            cmds.append(
                (
                    f"amd-smi reset --clocks --gpu {index}",
                    [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                )
            )

            # set --perf-level defaults
            # Last of the clock writes: the reset above forces AUTO, and --clk-limit and
            # --perf-determinism each leave the device on a level of their own.
            perf_level = self.metric_data["gpu_data"][index]["perf_level"]
            if perf_level not in ("N/A", "AMDSMI_DEV_PERF_LEVEL_UNKNOWN"):
                perf_level = _strip_prefix(perf_level, "AMDSMI_DEV_PERF_LEVEL_")
                cmds.append((f"amd-smi set --perf-level {perf_level} --gpu {index}", self.PASS))
            # set --process-isolation defaults
            process_isolation = self.static_data["gpu_data"][index]["process_isolation"]
            # Put back whatever value the sweep left; "N/A" means unreadable, so
            # there is nothing to restore.
            if process_isolation != "N/A":
                original = 0 if process_isolation == "Disabled" else 1
                cmds.append(
                    (
                        f"amd-smi set --process-isolation {original} --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

        return cmds, problems
