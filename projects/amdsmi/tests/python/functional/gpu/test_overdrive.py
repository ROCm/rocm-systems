#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU performance: overdrive level set, performance level set."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuOverdrive(unittest.TestCase):
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

    def test_set_gpu_overdrive_level(self):
        self.common.print_func_name("")

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            # Find current overdrive value
            msg = f"\t### amdsmi_get_gpu_overdrive_level(gpu={i}):"
            try:
                overdrive_value_current = amdsmi.amdsmi_get_gpu_overdrive_level(gpu)
                self.common.print(msg, overdrive_value_current)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            if overdrive_value_current != 1:
                overdrive_value = 1
            else:
                overdrive_value = 2

            # Set overdrive value
            msg = (
                f"\t### amdsmi_set_gpu_overdrive_level(gpu={i}, overdrive_value={overdrive_value}):"
            )
            try:
                ret = amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set back to original overdrive value
            msg = f"\t### amdsmi_set_gpu_overdrive_level(gpu={i}, overdrive_value={overdrive_value_current}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_overdrive_level(gpu, overdrive_value_current)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_set_gpu_perf_level(self):
        self.common.print_func_name("")

        dev_perf_level_current = common.DEV_PERF_LEVELS[0][1]
        dev_perf_level_current_cond = common.DEV_PERF_LEVELS[0][2]
        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"\t### amdsmi_get_gpu_perf_level(gpu={i}):"
            try:
                dev_perf_level_name_current = amdsmi.amdsmi_get_gpu_perf_level(gpu)
                items = dev_perf_level_name_current.split("_")
                level_name_current = items[-1]
                for name, level, cond in common.DEV_PERF_LEVELS:
                    if name == level_name_current:
                        dev_perf_level_current = level
                        dev_perf_level_current_cond = cond
                        break
                self.common.print(msg, level_name_current)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                self.common.print(msg, e)
                continue

            for dev_perf_level_name, dev_perf_level, dev_perf_level_cond in common.DEV_PERF_LEVELS:
                msg = f"\t### amdsmi_set_gpu_perf_level(gpu={i}, dev_perf_level={dev_perf_level_name}):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, dev_perf_level_cond):
                        self.raise_exception = e

            msg = f"\t### amdsmi_set_gpu_perf_level(gpu={i}, dev_perf_level={dev_perf_level_name_current}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_perf_level(gpu, dev_perf_level_current)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, dev_perf_level_current_cond):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_get_gpu_reg_table_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_reg_table_info as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_get_gpu_reg_table_info=amdsmi.amdsmi_get_gpu_reg_table_info,
            reg_type=common.REG_TYPES,
        )
        return

    def test_set_gpu_clk_limit(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_clk_limit as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO Find better way to set value
        value = 0

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for clk_type_name, _, clk_cond in common.CLK_TYPES:
                for clk_limit_type_name, clk_limit_type, clk_limit_cond in common.CLK_LIMIT_TYPES:
                    msg = f"\t### amdsmi_set_gpu_clk_limit(gpu={i}, clk_type={clk_type_name}, clk_limit_type={clk_limit_type_name}, value={value}):"
                    try:
                        amdsmi.amdsmi_set_gpu_clk_limit(
                            gpu, clk_type_name, clk_limit_type_name, value
                        )
                        self.common.print(msg, "")
                        self.common.check_ret("", "", self.common.PASS)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not clk_cond == self.common.PASS:
                            self.common.check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        elif not clk_limit_type == self.common.PASS:
                            self.common.check_ret(msg, e, clk_limit_cond)
                            self.raise_exception = e
                        else:
                            self.common.check_ret(msg, e, self.common.PASS)
                            self.raise_exception = e
                    self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_od_clk_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_od_clk_info as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO value = 0
        value = 200

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            for freq_ind_name, freq_ind, freq_ind_cond in common.FREQ_INDS:
                for clk_type_name, clk_type, clk_cond in common.CLK_TYPES:
                    msg = f"\t### amdsmi_set_gpu_od_clk_info(gpu={i}, freq_ind={freq_ind_name}, value={value}, clk_type={clk_type_name}):"
                    try:
                        amdsmi.amdsmi_set_gpu_od_clk_info(gpu, freq_ind, value, clk_type)
                        self.common.print(msg, "")
                        self.common.check_ret("", "", self.common.PASS)
                    except amdsmi.AmdSmiLibraryException as e:
                        if not freq_ind_cond == self.common.PASS:
                            self.common.check_ret(msg, e, freq_ind_cond)
                            self.raise_exception = e
                        elif not clk_cond == self.common.PASS:
                            self.common.check_ret(msg, e, clk_cond)
                            self.raise_exception = e
                        else:
                            self.common.check_ret(msg, e, self.common.PASS)
                            self.raise_exception = e
                    self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_set_gpu_od_volt_info(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_od_volt_info as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO vpoint = 0 clk_value = 0 volt_value = 0
        vpoint = 0
        clk_value = 0
        volt_value = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_od_volt_info=amdsmi.amdsmi_set_gpu_od_volt_info,
            vpoint=vpoint,
            clk_value=clk_value,
            volt_value=volt_value,
        )
        return

    def test_set_gpu_perf_determinism_mode(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_perf_determinism_mode as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO clk_value = 0
        clk_value = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_perf_determinism_mode=amdsmi.amdsmi_set_gpu_perf_determinism_mode,
            clk_value=clk_value,
        )
        return

    # out of order: 0 then power_profile_preset_mask
