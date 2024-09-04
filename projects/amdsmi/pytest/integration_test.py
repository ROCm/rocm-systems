#!/usr/bin/env python3
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import sys
sys.path.append("/opt/rocm/libexec/amdsmi_cli/")

try:
    import amdsmi
except ImportError:
    raise ImportError("Could not import /opt/rocm/libexec/amdsmi_cli/amdsmi_cli.py")

import unittest
import threading
import multiprocessing
from datetime import datetime

def handle_exceptions(func):
    """Exposes, silences, and logs AMD SMI exceptions to users what exception was raised.

        params:
            func: test function(s) that use decorator to expose AMD SMI exceptions
        return:
            On success - original function is returned
            On failure - silences error and prints to user what exception was caught
        """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except amdsmi.AmdSmiRetryException as e:
            print("**** [ERROR] | Test: " + str(func.__name__) + " | Caught AmdSmiRetryException: {}".format(e))
            pass
        except amdsmi.AmdSmiTimeoutException as e:
            print("**** [ERROR] | Test: " + str(func.__name__) + " | Caught AmdSmiTimeoutException: {}".format(e))
            pass
        except amdsmi.AmdSmiLibraryException as e:
            print("**** [ERROR] | Test: " + str(func.__name__) + " | Caught AmdSmiLibraryException: {}".format(e))
            pass
        except Exception as e:
            print("**** [ERROR] | Test: " + str(func.__name__) + " | Caught unknown exception: {}".format(e))
            pass
    return wrapper

class TestAmdSmiInit(unittest.TestCase):
    @handle_exceptions
    def test_init(self):
        amdsmi.amdsmi_init()
        amdsmi.amdsmi_shut_down()

class TestAmdSmiPythonInterface(unittest.TestCase):
    @handle_exceptions
    def setUp(self):
        amdsmi.amdsmi_init()
    @handle_exceptions
    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    # Bad page is not supported in Navi21 and Navi31
    @handle_exceptions
    def test_bad_page_info(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_gpu_bad_page_info \n")
            bad_page_info = amdsmi.amdsmi_get_gpu_bad_page_info(processors[i])
            print("bad_page_info: " + str(bad_page_info))
            print("Number of bad pages: {}".format(len(bad_page_info)))
            j = 0
            for table_record in bad_page_info:
                print("\ntable_record[\"value\"]" + str(table_record["value"]))
                print("Page: {}".format(j))
                print("Page Address: " + str(table_record["page_address"]))
                print("Page Size: " + str(table_record["page_size"]))
                print("Status: " + str(table_record["status"]))
                print()
                j += 1
        print()

    def test_bdf_device_id(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            processor = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            print("\n###Test amdsmi_get_gpu_vbios_info \n")
            vbios_info = amdsmi.amdsmi_get_gpu_vbios_info(processor)
            print("  vbios_info['part_number'] is: {}".format(
                vbios_info['part_number']))
            print("  vbios_info['build_date'] is: {}".format(
                vbios_info['build_date']))
            print("  vbios_info['version'] is: {}".format(
                vbios_info['version']))
            print("  vbios_info['name'] is: {}".format(
                vbios_info['name']))
            print("\n###Test amdsmi_get_gpu_device_uuid \n")
            uuid = amdsmi.amdsmi_get_gpu_device_uuid(processor)
            print("  uuid is: {}".format(uuid))
        print()

    def test_ecc(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_gpu_total_ecc_count \n")
            ecc_info = amdsmi.amdsmi_get_gpu_total_ecc_count(processors[i])
            print("Number of uncorrectable errors: {}".format(
                ecc_info['uncorrectable_count']))
            print("Number of correctable errors: {}".format(
                ecc_info['correctable_count']))
            print("Number of deferred errors: {}".format(
                ecc_info['deferred_count']))
            self.assertGreaterEqual(ecc_info['uncorrectable_count'], 0)
            self.assertGreaterEqual(ecc_info['correctable_count'], 0)
            self.assertGreaterEqual(ecc_info['deferred_count'], 0)
        print()

    # RAS is not supported in Navi21 and Navi31
    @handle_exceptions
    def test_ras(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_gpu_ras_feature_info \n")
            ras_feature = amdsmi.amdsmi_get_gpu_ras_feature_info(processors[i])
            print("ras_feature: " + str(ras_feature))
            if ras_feature != None:
                print("ras_feature: " + str(ras_feature))
                print("RAS eeprom version: {}".format(ras_feature['eeprom_version']))
                print("RAS parity schema: {}".format(ras_feature['parity_schema']))
                print("RAS single bit schema: {}".format(ras_feature['single_bit_schema']))
                print("RAS double bit schema: {}".format(ras_feature['double_bit_schema']))
                print("Poisioning supported: {}".format(ras_feature['poison_schema']))
        print()

    def test_clock_info(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_clock_info \n")
            clock_measure = amdsmi.amdsmi_get_clock_info(
                    processors[i], amdsmi.AmdSmiClkType.GFX)
            print("  Current clock for domain GFX is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain GFX is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain GFX is: {}".format(
                clock_measure['min_clk']))
            print("  Is GFX clock locked: {}".format(
                clock_measure['clk_locked']))
            print("  Is GFX clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
            clock_measure = amdsmi.amdsmi_get_clock_info(
                processors[i], amdsmi.AmdSmiClkType.MEM)
            print("  Current clock for domain MEM is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain MEM is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain MEM is: {}".format(
                clock_measure['min_clk']))
            print("  Is MEM clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
        print()

    # VCLK0 and DCLK0 are not supported in MI210
    @handle_exceptions
    def test_gpu_clock_vclk0_dclk0(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_clock_info \n")
            clock_measure = amdsmi.amdsmi_get_clock_info(
                processors[i], amdsmi.AmdSmiClkType.VCLK0)
            print("  Current clock for domain VCLK0 is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain VCLK0 is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain VCLK0 is: {}".format(
                clock_measure['min_clk']))
            print("  Is VCLK0 clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
            clock_measure = amdsmi.amdsmi_get_clock_info(
                processors[i], amdsmi.AmdSmiClkType.DCLK0)
            print("  Current clock for domain DCLK0 is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain DCLK0 is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain DCLK0 is: {}".format(
                clock_measure['min_clk']))
            print("  Is DCLK0 clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
        print()

    # VCLK1 and DCLK1 are not supported in Navi 31, MI210, and MI300
    @handle_exceptions
    def test_gpu_clock_vclk1_dclk1(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_clock_info \n")
            clock_measure = amdsmi.amdsmi_get_clock_info(
                processors[i], amdsmi.AmdSmiClkType.VCLK1)
            print("  Current clock for domain VCLK1 is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain VCLK1 is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain VCLK1 is: {}".format(
                clock_measure['min_clk']))
            print("  Is VCLK1 clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
            clock_measure = amdsmi.amdsmi_get_clock_info(
                processors[i], amdsmi.AmdSmiClkType.DCLK1)
            print("  Current clock for domain DCLK1 is: {}".format(
                clock_measure['clk']))
            print("  Max clock for domain DCLK1 is: {}".format(
                clock_measure['max_clk']))
            print("  Min clock for domain DCLK1 is: {}".format(
                clock_measure['min_clk']))
            print("  Is DCLK1 clock in deep sleep: {}".format(
                clock_measure['clk_deep_sleep']))
        print()

    def test_gpu_activity(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_gpu_activity \n")
            engine_usage = amdsmi.amdsmi_get_gpu_activity(processors[i])
            print("  engine_usage['gfx_activity'] is: {} %".format(
                engine_usage['gfx_activity']))
            print("  engine_usage['umc_activity'] is: {} %".format(
                engine_usage['umc_activity']))
            print("  engine_usage['mm_activity'] is: {} %".format(
                engine_usage['mm_activity']))
        print()

    def test_pcie(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_pcie_info \n")
            pcie_info = amdsmi.amdsmi_get_pcie_info(processors[i])
            print("  pcie_info['pcie_metric']['pcie_width'] is: {}".format(
                pcie_info['pcie_metric']['pcie_width']))
            print("  pcie_info['pcie_static']['max_pcie_width'] is: {} ".format(
                pcie_info['pcie_static']['max_pcie_width']))
            print("  pcie_info['pcie_metric']['pcie_speed'] is: {} MT/s".format(
                pcie_info['pcie_metric']['pcie_speed']))
            print("  pcie_info['pcie_static']['max_pcie_speed'] is: {} ".format(
                pcie_info['pcie_static']['max_pcie_speed']))
            print("  pcie_info['pcie_static']['pcie_interface_version'] is: {}".format(
                pcie_info['pcie_static']['pcie_interface_version']))
            print("  pcie_info['pcie_static']['slot_type'] is: {}".format(
                pcie_info['pcie_static']['slot_type']))
            print("  pcie_info['pcie_metric']['pcie_replay_count'] is: {}".format(
                pcie_info['pcie_metric']['pcie_replay_count']))
            print("  pcie_info['pcie_metric']['pcie_bandwidth'] is: {}".format(
                pcie_info['pcie_metric']['pcie_bandwidth']))
            print("  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: {}".format(
                pcie_info['pcie_metric']['pcie_l0_to_recovery_count']))
            print("  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: {}".format(
                pcie_info['pcie_metric']['pcie_replay_roll_over_count']))
            print("  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: {}".format(
                pcie_info['pcie_metric']['pcie_nak_sent_count']))
            print("  pcie_info['pcie_metric']['pcie_nak_received_count'] is: {}".format(
                pcie_info['pcie_metric']['pcie_nak_received_count']))
        print()

    def test_power(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_power_info \n")
            power_info = amdsmi.amdsmi_get_power_info(processors[i])
            print("  power_info['current_socket_power'] is: {}".format(
                power_info['current_socket_power']))
            print("  power_info['average_socket_power'] is: {}".format(
                power_info['average_socket_power']))
            print("  power_info['gfx_voltage'] is: {}".format(
                power_info['gfx_voltage']))
            print("  power_info['soc_voltage'] is: {}".format(
                power_info['soc_voltage']))
            print("  power_info['mem_voltage'] is: {}".format(
                power_info['mem_voltage']))
            print("  power_info['power_limit'] is: {}".format(
                power_info['power_limit']))
            print("\n###Test amdsmi_is_gpu_power_management_enabled \n")
            is_power_management_enabled = amdsmi.amdsmi_is_gpu_power_management_enabled(processors[i])
            print("  Is power management enabled is: {}".format(
                is_power_management_enabled))
        print()

    def test_temperature(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_temp_metric \n")
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.HOTSPOT, amdsmi.AmdSmiTemperatureMetric.CURRENT)
            print("  Current temperature for HOTSPOT is: {}".format(
                temperature_measure))
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.VRAM, amdsmi.AmdSmiTemperatureMetric.CURRENT)
            print("  Current temperature for VRAM is: {}".format(
                temperature_measure))
            print("\n###Test amdsmi_get_temp_metric \n")
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.HOTSPOT, amdsmi.AmdSmiTemperatureMetric.CRITICAL)
            print("  Limit (critical) temperature for HOTSPOT is: {}".format(
                temperature_measure))
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.VRAM, amdsmi.AmdSmiTemperatureMetric.CRITICAL)
            print("  Limit (critical) temperature for VRAM is: {}".format(
                temperature_measure))
            print("\n###Test amdsmi_get_temp_metric \n")
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.HOTSPOT, amdsmi.AmdSmiTemperatureMetric.EMERGENCY)
            print("  Shutdown (emergency) temperature for HOTSPOT is: {}".format(
                temperature_measure))
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.VRAM, amdsmi.AmdSmiTemperatureMetric.EMERGENCY)
            print("  Shutdown (emergency) temperature for VRAM is: {}".format(
                temperature_measure))
        print()

    # Edge temperature is not supported in MI300
    @handle_exceptions
    def test_temperature_edge(self):
        processors = amdsmi.amdsmi_get_processor_handles()
        self.assertGreaterEqual(len(processors), 1)
        self.assertLessEqual(len(processors), 32)
        for i in range(0, len(processors)):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
            print("\n\n###Test Processor {}, bdf: {}".format(i, bdf))
            print("\n###Test amdsmi_get_temp_metric \n")
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.EDGE, amdsmi.AmdSmiTemperatureMetric.CURRENT) # current
            print("  Current temperature for EDGE is: {}".format(
                temperature_measure))
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.EDGE, amdsmi.AmdSmiTemperatureMetric.CRITICAL) # slowdown/limit
            print("  Limit (critical) temperature for EDGE is: {}".format(
                temperature_measure))
            temperature_measure = amdsmi.amdsmi_get_temp_metric(
                processors[i], amdsmi.AmdSmiTemperatureType.EDGE, amdsmi.AmdSmiTemperatureMetric.EMERGENCY) # shutdown
            print("  Shutdown (emergency) temperature for EDGE is: {}".format(
                temperature_measure))
        print()

    def test_walkthrough(self):
        walk_through(self)

    # Unstable on workstation cards
    # @handle_exceptions
    # def test_walkthrough_multiprocess(self):
    #     print("\n\n========> test_walkthrough_multiprocess start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), 32)
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

    # Unstable on workstation cards
    # @handle_exceptions
    # def test_walkthrough_multithread(self):
    #     print("\n\n========> test_walkthrough_multithread start <========\n")
    #     processors = amdsmi.amdsmi_get_processor_handles()
    #     self.assertGreaterEqual(len(processors), 1)
    #     self.assertLessEqual(len(processors), 32)
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
    # @handle_exceptions
    # def test_z_gpureset_asicinfo_multithread(self):
    #     def get_asic_info(processor):
    #         print("\n###Test amdsmi_get_gpu_asic_info \n")
    #         asic_info = amdsmi.amdsmi_get_gpu_asic_info(processor)
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
    #     self.assertLessEqual(len(processors), 32)
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

def walk_through(self):
    print("\n###Test amdsmi_get_processor_handles() \n")
    processors = amdsmi.amdsmi_get_processor_handles()
    for i in range(0, len(processors)):
        print("\n###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = " + str(i) + "\n")
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(processors[i])
        print("###Test Processor {}, bdf: {} ".format(i, bdf))
        print("\n###Test amdsmi_get_gpu_asic_info \n")
        asic_info = amdsmi.amdsmi_get_gpu_asic_info(processors[i])
        print("  asic_info['market_name'] is: {}".format(
            asic_info['market_name']))
        print("  asic_info['vendor_id'] is: {}".format(
            asic_info['vendor_id']))
        print("  asic_info['vendor_name'] is: {}".format(
            asic_info['vendor_name']))
        print("  asic_info['device_id'] is: {}".format(
            asic_info['device_id']))
        print("  asic_info['rev_id'] is: {}\n".format(
            asic_info['rev_id']))
        print("  asic_info['asic_serial'] is: {}\n".format(
            asic_info['asic_serial']))
        print("  asic_info['oam_id'] is: {}\n".format(
            asic_info['oam_id']))
        print("###Test amdsmi_get_power_cap_info \n")
        power_info = amdsmi.amdsmi_get_power_cap_info(processors[i])
        print("  power_info['dpm_cap'] is: {}".format(
            power_info['dpm_cap']))
        print("  power_info['power_cap'] is: {}\n".format(
            power_info['power_cap']))
        print("###Test amdsmi_get_gpu_vbios_info \n")
        vbios_info = amdsmi.amdsmi_get_gpu_vbios_info(processors[i])
        print("  vbios_info['part_number'] is: {}".format(
            vbios_info['part_number']))
        print("  vbios_info['build_date'] is: {}".format(
            vbios_info['build_date']))
        print("  vbios_info['name'] is: {}\n".format(
            vbios_info['name']))
        print("  vbios_info['version'] is: {}\n".format(
            vbios_info['version']))
        print("###Test amdsmi_get_gpu_board_info \n")
        board_info = amdsmi.amdsmi_get_gpu_board_info(processors[i])
        print("  board_info['model_number'] is: {}\n".format(
            board_info['model_number']))
        print("  board_info['product_serial'] is: {}\n".format(
            board_info['product_serial']))
        print("  board_info['fru_id'] is: {}\n".format(
            board_info['fru_id']))
        print("  board_info['manufacturer_name'] is: {}\n".format(
            board_info['manufacturer_name']))
        print("  board_info['product_name'] is: {}\n".format(
            board_info['product_name']))
        print("###Test amdsmi_get_fw_info \n")
        fw_info = amdsmi.amdsmi_get_fw_info(processors[i])
        fw_num = len(fw_info['fw_list'])
        self.assertLessEqual(fw_num, len(amdsmi.AmdSmiFwBlock))
        for j in range(0, fw_num):
            fw = fw_info['fw_list'][j]
            if fw['fw_version'] != 0:
                print("FW name:           {}".format(
                    fw['fw_name'].name))
                print("FW version:        {}".format(
                    fw['fw_version']))
        print("\n###Test amdsmi_get_gpu_driver_info \n")
        driver_info = amdsmi.amdsmi_get_gpu_driver_info(processors[i])
        print("Driver info:  {}".format(driver_info))
        print("\n###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = " + str(i) + "\n")

if __name__ == '__main__':
    unittest.main()