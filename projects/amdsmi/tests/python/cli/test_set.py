#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI leaf test: set command."""

import os
import shutil
import tempfile

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
        cmds = []
        for index, gpu in enumerate(self.common.processors):
            # max fan speed is the hwmon pwm1_max ceiling on every interface
            fan_max = self.metric_data["gpu_data"][index]["fan"]["max"]
            if fan_max != "N/A":
                self.assertGreater(fan_max, 0, f"GPU {index}: max fan speed must be > 0")
                self.assertLessEqual(fan_max, 255, f"GPU {index}: max fan speed must be <= 255")

            # reset --fans (works for both legacy hwmon and gpu_od interfaces)
            # A readable fan speed does not imply the fan can be controlled.
            fan_speed = self.metric_data["gpu_data"][index]["fan"]["speed"]
            if fan_speed != "N/A":
                cmds.append(
                    (
                        f"amd-smi reset --fans --gpu {index}",
                        [self.PASS, amdsmi.AmdSmiStatus.NOT_SUPPORTED],
                    )
                )

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
            # was left at max.
            cmds.append((f"amd-smi reset --power-cap --gpu {index}", self.PASS))

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

        self.common.print("Restore Starting Values")
        self.RunCmds(cmds)

        return

    def test_power_stabilization_configure(self):
        """Test set --power-stabilization-configure (write path, gpu 0)."""
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set --power-stabilization-configure"
        self.common.print(msg)

        ps_baseline = self.static_data["gpu_data"][0]["power_stabilization"]
        if ps_baseline == "N/A":
            self.skipTest("Power Stabilization not supported on gpu 0")

        # Negative paths: malformed input is rejected before any sysfs write.
        cmds = [
            ("amd-smi set --gpu 0 --power-stabilization-configure invalidkey=1", self.FAIL),
            ("amd-smi set --gpu 0 --power-stabilization-configure host_control=1", self.FAIL),
        ]
        self.RunCmds(cmds)

        # Missing "=" must produce a clean argparse-style error, not an
        # uncaught traceback.
        cmd = "amd-smi set --gpu 0 --power-stabilization-configure enable"
        (rc, std_out, std_err) = self.util.RunCmdSync(cmd)
        self.assertNotEqual(rc, self.PASS, f"Command '{cmd}' unexpectedly succeeded")
        combined_output = f"{std_out}\n{std_err}".lower()
        self.assertIn(
            "invalid key=value pair",
            combined_output,
            f"Command '{cmd}' did not produce the expected clean error message; "
            f"got: {combined_output}",
        )
        self.assertNotIn(
            "traceback",
            combined_output,
            f"Command '{cmd}' crashed with an uncaught traceback instead of a clean error",
        )

        # Missing / empty / invalid JSON file, via @<path>.
        tmp_dir = tempfile.mkdtemp(prefix="amdsmi_ps_configure_")
        try:
            missing_path = os.path.join(tmp_dir, "missing.json")
            empty_path = os.path.join(tmp_dir, "empty.json")
            with open(empty_path, "w"):
                pass
            invalid_path = os.path.join(tmp_dir, "invalid.json")
            with open(invalid_path, "w") as fout:
                fout.write("not json")

            cmds = [
                (f"amd-smi set --gpu 0 --power-stabilization-configure @{missing_path}", self.FAIL),
                (f"amd-smi set --gpu 0 --power-stabilization-configure @{empty_path}", self.FAIL),
                (f"amd-smi set --gpu 0 --power-stabilization-configure @{invalid_path}", self.FAIL),
            ]
            self.RunCmds(cmds)

            # Positive path: KEY=VALUE tokens (comma- and space-separated),
            # insertion_priority by name, then an @<path> round-trip of the
            # exact `static --json` dump (host_control is read-only and must
            # be silently skipped).
            cmds = [
                (
                    "amd-smi set --gpu 0 --power-stabilization-configure enable=1,power_floor=55",
                    self.PASS,
                ),
                (
                    "amd-smi set --gpu 0 --power-stabilization-configure "
                    "insertion_priority=GUARANTEED ramp_enable=1",
                    self.PASS,
                ),
            ]
            self.RunCmds(cmds)

            dump_path = os.path.join(tmp_dir, "dump.json")
            cmd = "amd-smi static --gpu 0 --power-stabilization --json"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")
            # Write the full 'gpu_data' dump as-is: set resolves the GPU 0
            # entry by matching its integer "gpu" id, same as --ampp-configure.
            with open(dump_path, "w") as fout:
                fout.write(data)

            cmds = [
                (f"amd-smi set --gpu 0 --power-stabilization-configure @{dump_path}", self.PASS)
            ]
            self.RunCmds(cmds)
        finally:
            # Restore the shared mock sysfs fixture to its starting values.
            # The read-path dump uses stative names (enabled/ramp_enabled);
            # the setter's KEY=VALUE keys are the param_id operation names
            # (enable/ramp_enable). host_control_enabled has no param_id
            # (read-only), so it is never set.
            read_to_write_key = {"enabled": "enable", "ramp_enabled": "ramp_enable"}
            restore_tokens = ",".join(
                f"{read_to_write_key.get(key, key)}="
                f"{int(value) if isinstance(value, bool) else value}"
                for key, value in ps_baseline.items()
                if key != "host_control_enabled"
            )
            self.RunCmds(
                [
                    (
                        f"amd-smi set --gpu 0 --power-stabilization-configure {restore_tokens}",
                        self.PASS,
                    )
                ]
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return

    def test_power_stabilization_configure_case_sensitive_path(self):
        """End-to-end regression test for the '@<path>' case-preservation bug (F-4/G-1).

        The CLI's argv-lowercasing pass must leave an `@<path>` value from
        --power-stabilization-configure untouched in *both* legal spellings:
        the space-separated form (`--power-stabilization-configure @path`)
        and the `=` form (`--power-stabilization-configure=@path`) -- only
        the flag name itself may be lowercased. Uses a deterministic
        mixed-case directory and filename rather than tempfile.mkdtemp's
        random suffix (which draws from ascii_letters, so any uppercase
        coverage there is accidental, not a guard): a regression that
        lowercases the path turns into a file-not-found FAIL here, since the
        lowercased path does not exist on this case-sensitive filesystem.

        This exercises the real CLI binary end-to-end, but self-skips below
        when the GPU does not report Power Stabilization support -- i.e. on
        every hardware-free CI runner (review finding F-3). The guard that is
        actually CI-reachable regardless of hardware is
        tests/python/unit/system/test_cli_argv_case_normalization.py, which
        unit-tests amdsmi_cli.normalize_argv() directly with no GPU
        dependency. Keep both: this one additionally proves the fix holds
        through the real argparse/dispatch path on real hardware.
        """
        self.common.print_func_name("")
        msg = f"{self.tab}### amd-smi set --power-stabilization-configure (case-sensitive @<path>)"
        self.common.print(msg)

        ps_baseline = self.static_data["gpu_data"][0]["power_stabilization"]
        if ps_baseline == "N/A":
            self.skipTest("Power Stabilization not supported on gpu 0")

        tmp_dir = tempfile.mkdtemp(prefix="amdsmi_ps_configure_case_")
        try:
            cmd = "amd-smi static --gpu 0 --power-stabilization --json"
            (rc, data, std_err) = self.util.RunCmdSync(cmd)
            self.assertEqual(rc, self.PASS, f"Command '{cmd}' failed with rc={rc}")

            mixed_case_dir = os.path.join(tmp_dir, "PS_Config")
            os.makedirs(mixed_case_dir)
            mixed_case_path = os.path.join(mixed_case_dir, "Dump.json")
            with open(mixed_case_path, "w") as fout:
                fout.write(data)

            # Space-separated form: "--power-stabilization-configure @<path>".
            self.RunCmds(
                [
                    (
                        f"amd-smi set --gpu 0 --power-stabilization-configure @{mixed_case_path}",
                        self.PASS,
                    )
                ]
            )

            # "=" form: "--power-stabilization-configure=@<path>". argparse
            # only accepts a single token after "=" for a nargs="+" flag, so
            # this is the only shape that spelling supports.
            self.RunCmds(
                [
                    (
                        f"amd-smi set --gpu 0 --power-stabilization-configure=@{mixed_case_path}",
                        self.PASS,
                    )
                ]
            )
        finally:
            # Restore the shared mock sysfs fixture to its starting values,
            # same idiom as test_power_stabilization_configure above.
            read_to_write_key = {"enabled": "enable", "ramp_enabled": "ramp_enable"}
            restore_tokens = ",".join(
                f"{read_to_write_key.get(key, key)}="
                f"{int(value) if isinstance(value, bool) else value}"
                for key, value in ps_baseline.items()
                if key != "host_control_enabled"
            )
            self.RunCmds(
                [
                    (
                        f"amd-smi set --gpu 0 --power-stabilization-configure {restore_tokens}",
                        self.PASS,
                    )
                ]
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return
