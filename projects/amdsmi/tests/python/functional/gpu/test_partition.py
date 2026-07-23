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
"""GPU partition: compute partition set, XGMI PLPD."""

import unittest

import common.common as common
from common.common import amdsmi


class TestGpuPartition(unittest.TestCase):
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

    def test_set_gpu_compute_partition(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_compute_partition as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            default_compute_partition_type = common.COMPUTE_PARTITION_TYPES[0][1]
            msg = f"\t### amdsmi_get_gpu_compute_partition(gpu={i}):"
            try:
                default_compute_partition_name = amdsmi.amdsmi_get_gpu_compute_partition(gpu)
                self.common.print(msg, default_compute_partition_name)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            for (
                compute_partition_type_name,
                compute_partition_type,
                compute_partition_type_cond,
            ) in common.COMPUTE_PARTITION_TYPES:
                if default_compute_partition_name == compute_partition_type_name:
                    default_compute_partition_type = compute_partition_type
                msg = f"\t### amdsmi_set_gpu_compute_partition(gpu={i}, compute_partition_type={compute_partition_type_name}):"
                try:
                    ret = amdsmi.amdsmi_set_gpu_compute_partition(gpu, compute_partition_type)
                    self.common.print(msg, ret)
                    self.common.check_ret("", "", self.common.PASS)
                except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                    if self.common.check_ret(msg, e, compute_partition_type_cond):
                        self.raise_exception = e

            msg = f"\t### amdsmi_set_gpu_compute_partition(gpu={i}, default_compute_partition={default_compute_partition_name}):"
            try:
                ret = amdsmi.amdsmi_set_gpu_compute_partition(gpu, default_compute_partition_type)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

        if self.raise_exception:
            raise self.raise_exception
        return

    # integration

    def test_xgmi_plpd(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_xgmi_plpd as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        for i, gpu in enumerate(self.common.processors):
            self.common.print_device_header(i)
            msg = f"gpu({i}):"

            # Get current policy info
            msg = f"\t### amdsmi_get_xgmi_plpd(gpu={i}):"
            try:
                policy_info = amdsmi.amdsmi_get_xgmi_plpd(gpu)
                self.common.print(msg, "")
                self.common.check_ret("", "", self.common.PASS)

                num_supported = policy_info["num_supported"]
                if not isinstance(num_supported, int):
                    self.common.print(f"Cannot determine num_supported={num_supported}", "")
                    continue
                policy_id_current = policy_info["current_id"]
                if not isinstance(policy_id_current, int):
                    self.common.print(f"Cannot determine policy_id_current={policy_id_current}", "")
                    continue
                policy_id_orig = policy_info["policies"][policy_id_current]["policy_id"]
                if not isinstance(policy_id_orig, int):
                    self.common.print(f"Cannot determine orig policy_id={policy_id_orig}", "")
                    continue
                index = 0
                if num_supported >= 2:
                    if policy_id_current != 0:
                        index = 1
                policy_id = policy_info["policies"][index]["policy_id"]
                if not isinstance(policy_id, int):
                    self.common.print(f"Cannot determine policy_id={policy_id}", "")
                    continue
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
                continue

            # Set policy
            msg = f"\t### amdsmi_set_xgmi_plpd(gpu={i}, policy_id={policy_id}):"
            try:
                ret = amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id)
                self.common.print(msg, ret)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

            # Set back to original policy
            msg = f"\t### amdsmi_set_xgmi_plpd(gpu={i}, policy_id={policy_id_orig}):"
            try:
                ret = amdsmi.amdsmi_set_xgmi_plpd(gpu, policy_id_orig)
                self.common.check_ret("", "", self.common.PASS)
            except (amdsmi.AmdSmiLibraryException, amdsmi.AmdSmiParameterException) as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e

        if self.raise_exception:
            raise self.raise_exception
        return

    # import multiprocessing
    # Unstable on workstation cards
    # def test_walkthrough_multiprocess(self):
    #     print("\n\n========> test_walkthrough_multiprocess start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     p0 = multiprocessing.Process(target=walk_through, args=[self])
    #     p1 = multiprocessing.Process(target=walk_through, args=[self])
    #     p2 = multiprocessing.Process(target=walk_through, args=[self])
    #     p3 = multiprocessing.Process(target=walk_through, args=[self])
    #     p0.start()
    #     p1.start()
    #     p2.start()
    #     p3.start()
    #     p0.join()
    #     p1.join()
    #     p2.join()
    #     p3.join()
    #     print("\n========> test_walkthrough_multiprocess end <========\n")

    # import threading
    # Unstable on workstation cards
    # def test_walkthrough_multithread(self):
    #     print("\n\n========> test_walkthrough_multithread start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     t0 = threading.Thread(target=walk_through, args=[self])
    #     t1 = threading.Thread(target=walk_through, args=[self])
    #     t2 = threading.Thread(target=walk_through, args=[self])
    #     t3 = threading.Thread(target=walk_through, args=[self])
    #     t0.start()
    #     t1.start()
    #     t2.start()
    #     t3.start()
    #     t0.join()
    #     t1.join()
    #     t2.join()
    #     t3.join()
    #     print("\n========> test_walkthrough_multithread end <========\n")

    # # Unstable - do not run
    # def test_z_gpureset_asicinfo_multithread(self):
    #     def get_asic_info(processor):
    #         try:
    #             print("\n###Test amdsmi_get_gpu_asic_info \n")
    #             asic_info = amdsmi.amdsmi_get_gpu_asic_info(processor)
    #         except amdsmi.AmdSmiLibraryException as e:
    #             self.common.check_exception(e)
    #             continue
    #         print("  asic_info['market_name'] is: {}".format(
    #             asic_info['market_name']))
    #         print("  asic_info['vendor_id'] is: {}".format(
    #             asic_info['vendor_id']))
    #         print("  asic_info['vendor_name'] is: {}".format(
    #             asic_info['vendor_name']))
    #         print("  asic_info['device_id'] is: {}".format(
    #             asic_info['device_id']))
    #         print("  asic_info['rev_id'] is: {}".format(
    #             asic_info['rev_id']))
    #         print("  asic_info['asic_serial'] is: {}".format(
    #             asic_info['asic_serial']))
    #         print("  asic_info['oam_id'] is: {}\n".format(
    #             asic_info['oam_id']))
    #     def gpu_reset(processor):
    #         print("\n###Test amdsmi_reset_gpu \n")
    #         amdsmi.amdsmi_reset_gpu(processor)
    #         print("  GPU reset completed.\n")
    #     print("\n\n========> test_z_gpureset_asicinfo_multithread start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), self.common.max_num_physical_devices)
    #     for i in range(0, len(processors)):
    #         bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
    #         print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
    #         t0 = threading.Thread(target=get_asic_info, args=[processors[i]])
    #         t1 = threading.Thread(target=gpu_reset, args=[processors[i]])
    #         # t2 = threading.Thread(target=walk_through, args=[self])
    #         # t3 = threading.Thread(target=walk_through, args=[self])
    #         t0.start()
    #         t1.start()
    #         # t2.start()
    #         # t3.start()
    #         t0.join()
    #         t1.join()
    #         # t2.join()
    #         # t3.join()
    #     print("\n========> test_z_gpureset_asicinfo_multithread end <========\n")

    def test_get_gpu_accelerator_partition_profile(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_accelerator_partition_profile=amdsmi.amdsmi_get_gpu_accelerator_partition_profile
        )
        return

    def test_get_gpu_accelerator_partition_profile_config(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_accelerator_partition_profile_config=amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config
        )
        return

    def test_get_gpu_compute_partition(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_compute_partition=amdsmi.amdsmi_get_gpu_compute_partition
        )
        return

    def test_get_gpu_compute_partition_mem_alloc_mode(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_compute_partition_mem_alloc_mode=amdsmi.amdsmi_get_gpu_compute_partition_mem_alloc_mode
        )
        return

    def test_get_gpu_memory_partition(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_partition=amdsmi.amdsmi_get_gpu_memory_partition
        )
        return

    def test_get_gpu_memory_partition_config(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_get_gpu_memory_partition_config as it fails on MI300."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(
            amdsmi_get_gpu_memory_partition_config=amdsmi.amdsmi_get_gpu_memory_partition_config
        )
        return

    def test_set_gpu_accelerator_partition_profile(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_NOT_COMPLETE:
            msg = "\tSkipping test_set_gpu_accelerator_partition_profile as it is not complete."
            self.common.print(msg)
            self.skipTest(msg)

        # TODO profile_index = 0
        profile_index = 0

        self.common.Test_API_Per_GPU(
            amdsmi_set_gpu_accelerator_partition_profile=amdsmi.amdsmi_set_gpu_accelerator_partition_profile,
            profile_index=profile_index,
        )
        return

    # Uses clk_type_name instead of clk_type
    # Uses clk_limit_type_name instead of clk_limit_type

    def test_set_gpu_memory_partition(self):
        self.common.print_func_name("")

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_set_gpu_memory_partition=amdsmi.amdsmi_set_gpu_memory_partition,
            memory_partition_type=common.MEMORY_PARTITION_TYPES,
        )
        return

    def test_set_gpu_memory_partition_mode(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_set_gpu_memory_partition_mode as it fails (MI350X, AMDSMI_STATUS_UNEXPECTED_DATA)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_set_gpu_memory_partition_mode=amdsmi.amdsmi_set_gpu_memory_partition_mode,
            memory_partition_mode=common.MEMORY_PARTITION_TYPES,
        )
        return

    def test_set_gpu_compute_partition_mem_alloc_mode(self):
        self.common.print_func_name("")

        self.common.Test_Per_GPU_With_One_Enum(
            amdsmi_set_gpu_compute_partition_mem_alloc_mode=amdsmi.amdsmi_set_gpu_compute_partition_mem_alloc_mode,
            compute_partition_mem_alloc_mode=common.COMPUTE_PARTITION_MEM_ALLOC_MODE_TYPES,
        )
        return

    # out of order freq_ind then value then clk_type
