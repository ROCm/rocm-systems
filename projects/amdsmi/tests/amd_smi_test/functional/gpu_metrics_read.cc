/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <string>
#include <map>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "gpu_metrics_read.h"
#include "../test_common.h"


TestGpuMetricsRead::TestGpuMetricsRead() : TestBase() {
  set_title("AMDSMI GPU Metrics Read Test");
  set_description("The GPU Metrics tests verifies that "
                  "the gpu metrics info can be read properly.");
}

TestGpuMetricsRead::~TestGpuMetricsRead(void) {
}

void TestGpuMetricsRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestGpuMetricsRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestGpuMetricsRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestGpuMetricsRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestGpuMetricsRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    std::cout << "Device #" << std::to_string(i) << "\n";

    IF_VERB(STANDARD) {
        std::cout << "\t**GPU METRICS: Using static struct (Backwards Compatibility):\n";
    }
    amdsmi_gpu_metrics_t smu;
    err =  amdsmi_get_gpu_metrics_info(processor_handles_[i], &smu);
    const char *status_string;
    amdsmi_status_code_to_string(err, &status_string);
    std::cout << "\t\t** amdsmi_get_gpu_metrics_info(): " << status_string
    << "\n";
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << "METRIC TABLE HEADER:\n";
          std::cout << "structure_size=" << std::dec
          << static_cast<int>(smu.common_header.structure_size) << '\n';
          std::cout << "format_revision=" << std::dec
          << static_cast<int>(smu.common_header.format_revision) << '\n';
          std::cout << "content_revision=" << std::dec
          << static_cast<int>(smu.common_header.content_revision) << '\n';
          std::cout << "\n";
          std::cout << "TIME STAMPS (ns):\n";
          std::cout << std::dec << "system_clock_counter="
          << smu.system_clock_counter << '\n';
          std::cout << "firmware_timestamp (10ns resolution)=" << std::dec
          << smu.firmware_timestamp << '\n';
          std::cout << "\n";
          std::cout << "TEMPERATURES (C):\n";
          std::cout << std::dec << "temperature_edge= "
          << static_cast<uint16_t>(smu.temperature_edge) << '\n';
          std::cout << std::dec << "temperature_hotspot= "
          << static_cast<uint16_t>(smu.temperature_hotspot) << '\n';
          std::cout << std::dec << "temperature_mem= "
          << static_cast<uint16_t>(smu.temperature_mem) << '\n';
          std::cout << std::dec << "temperature_vrgfx= "
          << static_cast<uint16_t>(smu.temperature_vrgfx) << '\n';
          std::cout << std::dec << "temperature_vrsoc= "
          << static_cast<uint16_t>(smu.temperature_vrsoc) << '\n';
          std::cout << std::dec << "temperature_vrmem= "
          << static_cast<uint16_t>(smu.temperature_vrmem) << '\n';
          for (int i = 0; i < AMDSMI_NUM_HBM_INSTANCES; ++i) {
            std::cout << "temperature_hbm[" << i << "]= " << std::dec
            << static_cast<uint16_t>(smu.temperature_hbm[i]) << '\n';
          }
          std::cout << "\n";
          std::cout << "UTILIZATION (%):\n";
          std::cout << std::dec << "average_gfx_activity="
          << static_cast<uint16_t>(smu.average_gfx_activity) << '\n';
          std::cout << std::dec << "average_umc_activity="
          << static_cast<uint16_t>(smu.average_umc_activity) << '\n';
          std::cout << std::dec << "average_mm_activity="
          << static_cast<uint16_t>(smu.average_mm_activity) << '\n';
          std::cout << std::dec << "vcn_activity= [";
          uint16_t size = static_cast<uint16_t>(
            sizeof(smu.vcn_activity)/sizeof(smu.vcn_activity[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.vcn_activity[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.vcn_activity[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << "\n";
          std::cout << std::dec << "jpeg_activity= [";
          size = static_cast<uint16_t>(
            sizeof(smu.jpeg_activity)/sizeof(smu.jpeg_activity[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.jpeg_activity[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.jpeg_activity[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << "\n";
          std::cout << "POWER (W)/ENERGY (15.259uJ per 1ns):\n";
          std::cout << std::dec << "average_socket_power="
          << static_cast<uint16_t>(smu.average_socket_power) << '\n';
          std::cout << std::dec << "current_socket_power="
          << static_cast<uint16_t>(smu.current_socket_power) << '\n';
          std::cout << std::dec << "energy_accumulator="
          << static_cast<uint16_t>(smu.energy_accumulator) << '\n';
          std::cout << "\n";
          std::cout << "AVG CLOCKS (MHz):\n";
          std::cout << std::dec << "average_gfxclk_frequency="
          << static_cast<uint16_t>(smu.average_gfxclk_frequency) << '\n';
          std::cout << std::dec << "average_gfxclk_frequency="
          << static_cast<uint16_t>(smu.average_gfxclk_frequency) << '\n';
          std::cout << std::dec << "average_uclk_frequency="
          << static_cast<uint16_t>(smu.average_uclk_frequency) << '\n';
          std::cout << std::dec << "average_vclk0_frequency="
          << static_cast<uint16_t>(smu.average_vclk0_frequency) << '\n';
          std::cout << std::dec << "average_dclk0_frequency="
          << static_cast<uint16_t>(smu.average_dclk0_frequency) << '\n';
          std::cout << std::dec << "average_vclk1_frequency="
          << static_cast<uint16_t>(smu.average_vclk1_frequency) << '\n';
          std::cout << std::dec << "average_dclk1_frequency="
          << static_cast<uint16_t>(smu.average_dclk1_frequency) << '\n';
          std::cout << "\n";
          std::cout << "CURRENT CLOCKS (MHz):\n";
          std::cout << std::dec << "current_gfxclk="
          << smu.current_gfxclk << '\n';
          std::cout << std::dec << "current_gfxclks= [";
          size = static_cast<uint16_t>(
            sizeof(smu.current_gfxclks)/sizeof(smu.current_gfxclks[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_gfxclks[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_gfxclks[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << std::dec << "current_socclk="
          << smu.current_socclk << '\n';
          std::cout << std::dec << "current_socclks= [";
          size = static_cast<uint16_t>(
            sizeof(smu.current_socclks)/sizeof(smu.current_socclks[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_socclks[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_socclks[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << std::dec << "current_uclk="
          << static_cast<uint16_t>(smu.current_uclk) << '\n';
          std::cout << std::dec << "current_vclk0="
          << static_cast<uint16_t>(smu.current_vclk0) << '\n';
          std::cout << std::dec << "current_vclk0s= [";
          size = static_cast<uint16_t>(
            sizeof(smu.current_vclk0s)/sizeof(smu.current_vclk0s[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_vclk0s[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_vclk0s[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << std::dec << "current_dclk0="
          << smu.current_dclk0 << '\n';
          std::cout << std::dec << "current_dclk0s= [";
          size = static_cast<uint16_t>(
            sizeof(smu.current_dclk0s)/sizeof(smu.current_dclk0s[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_dclk0s[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint16_t>(smu.current_dclk0s[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << std::dec << "current_vclk1="
          << static_cast<uint16_t>(smu.current_vclk1) << '\n';
          std::cout << std::dec << "current_dclk1="
          << static_cast<uint16_t>(smu.current_dclk1) << '\n';
          std::cout << "\n";
          std::cout << "TROTTLE STATUS:\n";
          std::cout << std::dec << "throttle_status="
          << static_cast<uint32_t>(smu.throttle_status) << '\n';
          std::cout << "\n";
          std::cout << "FAN SPEED:\n";
          std::cout << std::dec << "current_fan_speed="
          << static_cast<uint16_t>(smu.current_fan_speed) << '\n';
          std::cout << "\n";
          std::cout << "LINK WIDTH (number of lanes) /SPEED (0.1 GT/s):\n";
          std::cout << "pcie_link_width="
          << std::to_string(smu.pcie_link_width) << '\n';
          std::cout << "pcie_link_speed="
          << std::to_string(smu.pcie_link_speed) << '\n';
          std::cout << "xgmi_link_width="
          << std::to_string(smu.xgmi_link_width) << '\n';
          std::cout << "xgmi_link_speed="
          << std::to_string(smu.xgmi_link_speed) << '\n';

          std::cout << "\n";
          std::cout << "Utilization Accumulated(%):\n";
          std::cout << "gfx_activity_acc="
          << std::dec << static_cast<uint32_t>(smu.gfx_activity_acc) << '\n';
          std::cout << "mem_activity_acc="
          << std::dec << static_cast<uint32_t>(smu.mem_activity_acc)  << '\n';

          std::cout << "\n";
          std::cout << "XGMI ACCUMULATED DATA TRANSFER SIZE (KB):\n";
          std::cout << std::dec << "xgmi_read_data_acc= [";
          size = static_cast<uint16_t>(
            sizeof(smu.xgmi_read_data_acc)/sizeof(smu.xgmi_read_data_acc[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint64_t>(smu.xgmi_read_data_acc[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint64_t>(smu.xgmi_read_data_acc[i]);
            }
          }
          std::cout << std::dec << "]\n";
          std::cout << std::dec << "xgmi_write_data_acc= [";
          size = static_cast<uint16_t>(
            sizeof(smu.xgmi_write_data_acc)/sizeof(smu.xgmi_write_data_acc[0]));
          for (uint16_t i= 0; i < size; i++) {
            if (i+1 < size) {
              std::cout << std::dec << static_cast<uint64_t>(smu.xgmi_write_data_acc[i]) << ", ";
            } else {
              std::cout << std::dec << static_cast<uint64_t>(smu.xgmi_write_data_acc[i]);
            }
          }
          std::cout << std::dec << "]\n";

          // Voltage (mV)
          std::cout << "voltage_soc = "
          << std::dec << static_cast<uint16_t>(smu.voltage_soc) << "\n";
          std::cout << "voltage_soc = "
          << std::dec << static_cast<uint16_t>(smu.voltage_gfx) << "\n";
          std::cout << "voltage_mem = "
          << std::dec << static_cast<uint16_t>(smu.voltage_mem) << "\n";

          std::cout << "indep_throttle_status = "
          << std::dec << static_cast<uint64_t>(smu.indep_throttle_status) << "\n";

          // Clock Lock Status. Each bit corresponds to clock instance
          std::cout << "gfxclk_lock_status (in hex) = "
          << std::hex << static_cast<uint32_t>(smu.gfxclk_lock_status) << std::dec <<"\n";

          // Bandwidth (GB/sec)
          std::cout << "pcie_bandwidth_acc=" << std::dec
          << static_cast<uint64_t>(smu.pcie_bandwidth_acc) << "\n";
          std::cout << "pcie_bandwidth_inst=" << std::dec
          << static_cast<uint64_t>(smu.pcie_bandwidth_inst) << "\n";

          // Counts
          std::cout << "pcie_l0_to_recov_count_acc= " << std::dec
          << static_cast<uint64_t>(smu.pcie_l0_to_recov_count_acc) << "\n";
          std::cout << "pcie_replay_count_acc= " << std::dec
          << static_cast<uint64_t>(smu.pcie_replay_count_acc) << "\n";
          std::cout << "pcie_replay_rover_count_acc= " << std::dec
          << static_cast<uint64_t>(smu.pcie_replay_rover_count_acc) << "\n";
          std::cout << "pcie_nak_rcvd_count_acc= " << std::dec
          << static_cast<uint32_t>(smu.pcie_nak_rcvd_count_acc) << "\n";
          std::cout << "pcie_replay_rover_count_acc= " << std::dec
          << static_cast<uint64_t>(smu.pcie_replay_rover_count_acc) << "\n";

          // Check for constant changes/refresh metrics
          std::cout << "\n";
          std::cout << "\t ** -> Checking metrics with constant changes ** " << "\n";
          constexpr uint16_t kMAX_ITER_TEST = 10;
          amdsmi_gpu_metrics_t gpu_metrics_check;
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
            amdsmi_get_gpu_metrics_info(processor_handles_[i], &gpu_metrics_check);
            std::cout << "\t\t -> firmware_timestamp [" << idx << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.firmware_timestamp << "\n";
          }

          std::cout << "\n";
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
            amdsmi_get_gpu_metrics_info(processor_handles_[i], &gpu_metrics_check);
            std::cout << "\t\t -> system_clock_counter [" << idx << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.system_clock_counter << "\n";
          }
          std::cout << "\n";
      }
    }

    // Verify api support checking functionality is working
    err =  amdsmi_get_gpu_metrics_info(processor_handles_[i], nullptr);
    if (err !=AMDSMI_STATUS_INVAL) {
      DISPLAY_AMDSMI_ERR(err);
    }
    amdsmi_status_code_to_string(err, &status_string);
    std::cout << "\t\t** amdsmi_get_gpu_metrics_info(nullptr check): " << status_string << "\n";
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
