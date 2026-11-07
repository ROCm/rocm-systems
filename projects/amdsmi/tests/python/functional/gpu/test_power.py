#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU power: power cap set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_gpu_power_cap(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)

            # Get Power Cap Info
            try:
                msg = f"\t### amdsmi_get_supported_power_cap(gpu={i}):"
                power_cap_types = amdsmi.amdsmi_get_supported_power_cap(gpu)
                # TODO(amdsmi_team): we should be iterating through all supported power cap sensors,
                #                    but for now we will just test the first one.
                #                    See amdsmi_get_supported_power_cap for more details
                #                    on the structure of power_cap_types
                sensor_type = power_cap_types["sensor_types"][0]
                self.common.print(msg, power_cap_types)
                self.common.check_ret("", "", self.common.PASS)

                msg = f"\t### amdsmi_get_power_cap_info(gpu={i}):"
                power_cap_info = amdsmi.amdsmi_get_power_cap_info(gpu, sensor_type)
                self.common.print(msg, power_cap_info)
                self.common.check_ret("", "", self.common.PASS)
                cap = int((power_cap_info["max_power_cap"] + power_cap_info["min_power_cap"]) / 2)
                current_cap = power_cap_info["power_cap"]
                # A power_cap of 0 means no user cap is set (driver uses default).
                # Restoring 0 would be rejected as out-of-range, so use default_power_cap instead.
                if current_cap == 0:
                    current_cap = power_cap_info["default_power_cap"]
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                # Have to be able to get info before setting
                continue

            # Set to Average Power Cap
            msg = f"\t### amdsmi_set_power_cap(gpu={i}, sensor={sensor_type}, power_cap={cap}):"
            try:
                ret = amdsmi.amdsmi_set_power_cap(gpu, sensor_type, cap)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Restore Power Cap
            msg = f"\t### amdsmi_set_power_cap(gpu={i}, sensor={sensor_type}, power_cap={current_cap}):"
            try:
                ret = amdsmi.amdsmi_set_power_cap(gpu, sensor_type, current_cap)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_get_gpu_power_profile_presets(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_power_profile_presets=amdsmi.amdsmi_get_gpu_power_profile_presets,
            index=0,
        )
        return

    def test_get_power_cap_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_cap_info=amdsmi.amdsmi_get_power_cap_info)
        return

    def test_get_power_info(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_get_power_info=amdsmi.amdsmi_get_power_info)
        return

    def test_is_gpu_power_management_enabled(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_is_gpu_power_management_enabled=amdsmi.amdsmi_is_gpu_power_management_enabled
        )
        return

    def test_set_gpu_power_profile(self):
        self.common.print_func_name("")
        # Valid rejections (although driver may include others)
        refused = [
            amdsmi.AmdSmiStatus.NOT_SUPPORTED,  # eg. guest devices, or sysfs node absent
            amdsmi.AmdSmiStatus.INPUT_OUT_OF_BOUNDS,  # eg. invalid input
            amdsmi.AmdSmiStatus.NO_PERM,  # eg. R/O sysfs in a container (EROFS)
        ]
        # All valid responses
        applied_or_refused = [amdsmi.AmdSmiStatus.SUCCESS, *refused]
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)
                for mask_name, mask, _ in common.POWER_PROFILE_PRESET_MASKS:
                    accept = refused if mask_name == "INVALID" else applied_or_refused
                    msg = (
                        f"\t### amdsmi_set_gpu_power_profile(gpu={i}, "
                        f"power_profile_preset_mask={mask_name}):"
                    )
                    with self.common.expect_status(msg, accept):
                        amdsmi.amdsmi_set_gpu_power_profile(gpu, 0, mask)
                    self.common.print("")
        return

    def test_get_power_stabilization_info(self):
        """Read Power Stabilization info; asserts field shape on supported hardware.

        A GPU with no power_stabilization sysfs node (eg. an APU) exercises the
        AMDSMI_STATUS_NOT_SUPPORTED path.
        """
        self.common.print_func_name("")
        fields = (
            "host_control_enabled",
            "enabled",
            "power_floor",
            "delay",
            "insertion_priority",
            "ramp_enabled",
            "ramp_max_ramp_up_rate",
            "ramp_max_ramp_down_rate",
            "ramp_min_time_at_power_floor",
            "ramp_stop_delay",
        )
        accept = [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)
                msg = f"\t### amdsmi_get_power_stabilization_info(gpu={i}):"
                info = None
                with self.common.expect_status(msg, accept):
                    info = amdsmi.amdsmi_get_power_stabilization_info(gpu)
                if info is None:
                    continue
                self.common.print("", info)
                for field in fields:
                    self.assertIn(field, info)
                self.assertIsInstance(
                    info["insertion_priority"], amdsmi.AmdSmiHawkInsertionPriority
                )
        return

    def test_set_power_stabilization_param(self):
        """Set each Power Stabilization param and confirm the read-back changed, then restore.

        A GPU with no power_stabilization sysfs node exercises the NOT_SUPPORTED
        path; host_control_enabled has no param_id (read-only), so it is never set.
        """
        self.common.print_func_name("")
        param_id = amdsmi.AmdSmiPowerStabilizationParamId
        priority = amdsmi.AmdSmiHawkInsertionPriority
        # (param_id, info field, test value). ENABLE is last so the other
        # params -- some of which the driver may gate on Power Stabilization
        # being enabled (e.g. ramp-down) -- are exercised while still at
        # their baseline enabled state, not freshly disabled.
        param_specs = (
            (param_id.POWER_FLOOR, "power_floor", 60),
            (param_id.DELAY, "delay", 20),
            (param_id.INSERTION_PRIORITY, "insertion_priority", None),
            (param_id.RAMP_ENABLE, "ramp_enabled", 0),
            (param_id.RAMP_MAX_RAMP_UP_RATE, "ramp_max_ramp_up_rate", 30.0),
            (param_id.RAMP_MAX_RAMP_DOWN_RATE, "ramp_max_ramp_down_rate", 30.0),
            (param_id.RAMP_MIN_TIME_AT_POWER_FLOOR, "ramp_min_time_at_power_floor", 8),
            (param_id.RAMP_STOP_DELAY, "ramp_stop_delay", 4),
            (param_id.ENABLE, "enabled", 0),
        )

        with self.common.status_sweep():
            for i, gpu in enumerate(self.common.processors):
                self.common.print_device_header(i)

                msg = f"\t### amdsmi_get_power_stabilization_info(gpu={i}) baseline:"
                original = None
                with self.common.expect_status(
                    msg, [amdsmi.AmdSmiStatus.SUCCESS, amdsmi.AmdSmiStatus.NOT_SUPPORTED]
                ):
                    original = amdsmi.amdsmi_get_power_stabilization_info(gpu)
                supported = original is not None
                accept = (
                    [amdsmi.AmdSmiStatus.SUCCESS]
                    if supported
                    else [amdsmi.AmdSmiStatus.NOT_SUPPORTED]
                )

                # Derive the insertion_priority target from the baseline so the
                # set/read-back assertion isn't a tautology when the baseline
                # already happens to be GUARANTEED.
                insertion_priority_target = (
                    (int(original["insertion_priority"]) + 1) % 3
                    if supported
                    else int(priority.GUARANTEED)
                )
                params = tuple(
                    (pid, field, insertion_priority_target)
                    if pid == param_id.INSERTION_PRIORITY
                    else (pid, field, value)
                    for pid, field, value in param_specs
                )

                try:
                    for pid, field, value in params:
                        set_msg = (
                            f"\t### amdsmi_set_power_stabilization_param(gpu={i}, "
                            f"param_id={pid.name}, value={value}):"
                        )
                        with self.common.expect_status(set_msg, accept):
                            amdsmi.amdsmi_set_power_stabilization_param(gpu, pid, value)

                        if not supported:
                            continue

                        read_msg = f"\t### amdsmi_get_power_stabilization_info(gpu={i}) after set:"
                        info = None
                        with self.common.expect_status(read_msg, amdsmi.AmdSmiStatus.SUCCESS):
                            info = amdsmi.amdsmi_get_power_stabilization_info(gpu)
                        if info is not None:
                            self.assertEqual(
                                int(info[field]),
                                value,
                                f"gpu({i}): set {field}={value} but read back {info[field]}",
                            )
                finally:
                    if supported:
                        for pid, field, _ in params:
                            restore_msg = (
                                f"\t### restore amdsmi_set_power_stabilization_param(gpu={i}, "
                                f"param_id={pid.name}):"
                            )
                            # Restore with the baseline's own Python type: the
                            # two ramp-rate params require a float
                            # (AMDSMI_VALUE_TYPE_DOUBLE), so coercing every
                            # field to int() makes the API reject them with
                            # INVAL and leaves them un-restored.
                            restore_value = original[field]
                            if not isinstance(restore_value, float):
                                restore_value = int(restore_value)
                            with self.common.expect_status(
                                restore_msg, amdsmi.AmdSmiStatus.SUCCESS
                            ):
                                amdsmi.amdsmi_set_power_stabilization_param(gpu, pid, restore_value)
        return
