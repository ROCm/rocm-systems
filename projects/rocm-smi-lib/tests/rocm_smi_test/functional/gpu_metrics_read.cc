/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020-2025, Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi_test/functional/gpu_metrics_read.h"
#include "rocm_smi_test/test_common.h"


TestGpuMetricsRead::TestGpuMetricsRead() : TestBase() {
  set_title("RSMI GPU Metrics Read Test");
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
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}



void TestGpuMetricsRead::Run(void) {
  rsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    IF_VERB(STANDARD) {
        std::cout << "\n\n";
        std::cout << "\t**GPU METRICS: Using static struct (Backwards Compatibility):\n";

        metrics_table_header_t header_values;
        auto ret = rsmi_dev_metrics_header_info_get(i, &header_values);
        if (ret == rsmi_status_t::RSMI_STATUS_SUCCESS) {
            std::cout << "\t[Metrics Header]" << "\n";
            std::cout << "\t  -> format_revision  : "
                      << static_cast<uint16_t>(header_values.format_revision) << "\n";
            std::cout << "\t  -> content_revision : "
                      << static_cast<uint16_t>(header_values.content_revision) << "\n";
            std::cout << "\t--------------------" << "\n";
        }
    }

    rsmi_gpu_metrics_t smu = {};
    err = rsmi_dev_gpu_metrics_info_get(i, &smu);
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          continue;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << "METRIC TABLE HEADER:\n";
          std::cout << "structure_size=" << std::dec
          << static_cast<uint16_t>(smu.common_header.structure_size) << "\n";
          std::cout << "format_revision=" << std::dec
          << static_cast<uint16_t>(smu.common_header.format_revision) << "\n";
          std::cout << "content_revision=" << std::dec
          << static_cast<uint16_t>(smu.common_header.content_revision) << "\n";

          std::cout << "\n";
          std::cout << "TIME STAMPS (ns):\n";
          std::cout << std::dec << "system_clock_counter=" << smu.system_clock_counter << "\n";
          std::cout << "firmware_timestamp (10ns resolution)=" << std::dec << smu.firmware_timestamp
                    << "\n";

          std::cout << "\n";
          std::cout << "TEMPERATURES (C):\n";
          std::cout << std::dec << "temperature_edge= " << smu.temperature_edge << "\n";
          std::cout << std::dec << "temperature_hotspot= " << smu.temperature_hotspot << "\n";
          std::cout << std::dec << "temperature_mem= " << smu.temperature_mem << "\n";
          std::cout << std::dec << "temperature_vrgfx= " << smu.temperature_vrgfx << "\n";
          std::cout << std::dec << "temperature_vrsoc= " << smu.temperature_vrsoc << "\n";
          std::cout << std::dec << "temperature_vrmem= " << smu.temperature_vrmem << "\n";
          std::cout << "temperature_hbm = [";
          std::copy(std::begin(smu.temperature_hbm),
                    std::end(smu.temperature_hbm),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << "\n";
          std::cout << "UTILIZATION (%):\n";
          std::cout << std::dec << "average_gfx_activity=" << smu.average_gfx_activity << "\n";
          std::cout << std::dec << "average_umc_activity=" << smu.average_umc_activity << "\n";
          std::cout << std::dec << "average_mm_activity=" << smu.average_mm_activity << "\n";
          std::cout << std::dec << "vcn_activity= [";
          std::copy(std::begin(smu.vcn_activity),
                    std::end(smu.vcn_activity),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << "\n";
          std::cout << std::dec << "jpeg_activity= [";
          std::copy(std::begin(smu.jpeg_activity),
                    std::end(smu.jpeg_activity),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << "\n";
          std::cout << "POWER (W)/ENERGY (15.259uJ per 1ns):\n";
          std::cout << std::dec << "average_socket_power=" << smu.average_socket_power << "\n";
          std::cout << std::dec << "current_socket_power=" << smu.current_socket_power << "\n";
          std::cout << std::dec << "energy_accumulator=" << smu.energy_accumulator << "\n";

          std::cout << "\n";
          std::cout << "AVG CLOCKS (MHz):\n";
          std::cout << std::dec << "average_gfxclk_frequency=" << smu.average_gfxclk_frequency
                    << "\n";
          std::cout << std::dec << "average_gfxclk_frequency=" << smu.average_gfxclk_frequency
                    << "\n";
          std::cout << std::dec << "average_uclk_frequency=" << smu.average_uclk_frequency << "\n";
          std::cout << std::dec << "average_vclk0_frequency=" << smu.average_vclk0_frequency
                    << "\n";
          std::cout << std::dec << "average_dclk0_frequency=" << smu.average_dclk0_frequency
                    << "\n";
          std::cout << std::dec << "average_vclk1_frequency=" << smu.average_vclk1_frequency
                    << "\n";
          std::cout << std::dec << "average_dclk1_frequency=" << smu.average_dclk1_frequency
                    << "\n";

          std::cout << "\n";
          std::cout << "CURRENT CLOCKS (MHz):\n";
          std::cout << std::dec << "current_gfxclk=" << smu.current_gfxclk << "\n";
          std::cout << std::dec << "current_gfxclks= [";
          std::copy(std::begin(smu.current_gfxclks),
                    std::end(smu.current_gfxclks),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "current_socclk=" << smu.current_socclk << "\n";
          std::cout << std::dec << "current_socclks= [";
          std::copy(std::begin(smu.current_socclks),
                    std::end(smu.current_socclks),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "current_uclk=" << smu.current_uclk << "\n";
          std::cout << std::dec << "current_vclk0=" << smu.current_vclk0 << "\n";
          std::cout << std::dec << "current_vclk0s= [";
          std::copy(std::begin(smu.current_vclk0s),
                    std::end(smu.current_vclk0s),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "current_dclk0=" << smu.current_dclk0 << "\n";
          std::cout << std::dec << "current_dclk0s= [";
          std::copy(std::begin(smu.current_dclk0s),
                    std::end(smu.current_dclk0s),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "current_vclk1=" << smu.current_vclk1 << "\n";
          std::cout << std::dec << "current_dclk1=" << smu.current_dclk1 << "\n";

          std::cout << "\n";
          std::cout << "TROTTLE STATUS:\n";
          std::cout << std::dec << "throttle_status=" << smu.throttle_status << "\n";

          std::cout << "\n";
          std::cout << "FAN SPEED:\n";
          std::cout << std::dec << "current_fan_speed=" << smu.current_fan_speed << "\n";

          std::cout << "\n";
          std::cout << "LINK WIDTH (number of lanes) /SPEED (0.1 GT/s):\n";
          std::cout << "pcie_link_width=" << smu.pcie_link_width << "\n";
          std::cout << "pcie_link_speed=" << smu.pcie_link_speed << "\n";
          std::cout << "xgmi_link_width=" << smu.xgmi_link_width << "\n";
          std::cout << "xgmi_link_speed=" << smu.xgmi_link_speed << "\n";

          std::cout << "\n";
          std::cout << "Utilization Accumulated(%):\n";
          std::cout << "gfx_activity_acc=" << std::dec << smu.gfx_activity_acc << "\n";
          std::cout << "mem_activity_acc=" << std::dec << smu.mem_activity_acc  << "\n";

          std::cout << "\n";
          std::cout << "XGMI ACCUMULATED DATA TRANSFER SIZE (KB):\n";
          std::cout << std::dec << "xgmi_read_data_acc= [";
          std::copy(std::begin(smu.xgmi_read_data_acc),
                    std::end(smu.xgmi_read_data_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "xgmi_write_data_acc= [";
          std::copy(std::begin(smu.xgmi_write_data_acc),
                    std::end(smu.xgmi_write_data_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          std::cout << std::dec << "xgmi_link_status= [";
          std::copy(std::begin(smu.xgmi_link_status),
                    std::end(smu.xgmi_link_status),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << std::dec << "]\n";

          // Voltage (mV)
          std::cout << "voltage_soc = " << std::dec << smu.voltage_soc << "\n";
          std::cout << "voltage_gfx = " << std::dec << smu.voltage_gfx << "\n";
          std::cout << "voltage_mem = " << std::dec << smu.voltage_mem << "\n";

          std::cout << "indep_throttle_status = " << std::dec << smu.indep_throttle_status << "\n";

          // Clock Lock Status. Each bit corresponds to clock instance
          std::cout << "gfxclk_lock_status (in hex) = " << std::hex
                    << smu.gfxclk_lock_status << std::dec <<"\n";

          // Bandwidth (GB/sec)
          std::cout << "pcie_bandwidth_acc=" << std::dec << smu.pcie_bandwidth_acc << "\n";
          std::cout << "pcie_bandwidth_inst=" << std::dec << smu.pcie_bandwidth_inst << "\n";

          // VRAM max bandwidth at max memory clock
          std::cout << "vram_max_bandwidth=" << std::dec << smu.vram_max_bandwidth << "\n";

          // Counts
          std::cout << "pcie_l0_to_recov_count_acc= " << std::dec << smu.pcie_l0_to_recov_count_acc
                    << "\n";
          std::cout << "pcie_replay_count_acc= " << std::dec << smu.pcie_replay_count_acc << "\n";
          std::cout << "pcie_replay_rover_count_acc= " << std::dec
                    << smu.pcie_replay_rover_count_acc << "\n";
          std::cout << "pcie_nak_sent_count_acc= " << std::dec << smu.pcie_nak_sent_count_acc
                    << "\n";
          std::cout << "pcie_nak_rcvd_count_acc= " << std::dec << smu.pcie_nak_rcvd_count_acc
                    << "\n";

          // PCIE other end recovery counter
          std::cout << "pcie_lc_perf_other_end_recovery = "
                    << std::dec << smu.pcie_lc_perf_other_end_recovery << "\n";

          // Accumulation cycle counter
          // Accumulated throttler residencies
          std::cout << "\n";
          std::cout << "RESIDENCY ACCUMULATION / COUNTER:\n";
          std::cout << "accumulation_counter = " << std::dec << smu.accumulation_counter << "\n";
          std::cout << "prochot_residency_acc = " << std::dec << smu.prochot_residency_acc << "\n";
          std::cout << "ppt_residency_acc = " << std::dec << smu.ppt_residency_acc << "\n";
          std::cout << "socket_thm_residency_acc = " << std::dec << smu.socket_thm_residency_acc
                    << "\n";
          std::cout << "vr_thm_residency_acc = " << std::dec << smu.vr_thm_residency_acc
                    << "\n";
          std::cout << "hbm_thm_residency_acc = " << std::dec << smu.hbm_thm_residency_acc << "\n";

          // Number of current partitions
          std::cout << "num_partition = " << std::dec << smu.num_partition << "\n";


          std::cout << std::dec << "xcp_stats.gfx_busy_inst = \n";
          auto xcp = 0;
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_busy_inst),
                    std::end(row.gfx_busy_inst),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.jpeg_busy = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.jpeg_busy),
                    std::end(row.jpeg_busy),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.vcn_busy = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.vcn_busy),
                    std::end(row.vcn_busy),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_busy_acc = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_busy_acc),
                    std::end(row.gfx_busy_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_below_host_limit_acc = \n";  // new for 1.7
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_below_host_limit_acc),
                    std::end(row.gfx_below_host_limit_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }
          // new for gpu metrics v1.8
          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_below_host_limit_ppt_acc = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_below_host_limit_ppt_acc),
                    std::end(row.gfx_below_host_limit_ppt_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_below_host_limit_thm_acc = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_below_host_limit_thm_acc),
                    std::end(row.gfx_below_host_limit_thm_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_low_utilization_acc = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_low_utilization_acc),
                    std::end(row.gfx_low_utilization_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          xcp = 0;
          std::cout << std::dec << "xcp_stats.gfx_below_host_limit_total_acc = \n";
          for (auto& row : smu.xcp_stats) {
            std::cout << "XCP[" << xcp << "] = " << "[ ";
            std::copy(std::begin(row.gfx_below_host_limit_total_acc),
                    std::end(row.gfx_below_host_limit_total_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
            std::cout << " ]\n";
            xcp++;
          }

          std::cout << "\n\n";
          std::cout << "\t ** -> Checking metrics with constant changes ** " << "\n";
          constexpr uint16_t kMAX_ITER_TEST = 10;
          rsmi_gpu_metrics_t gpu_metrics_check;
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
              rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
              std::cout << "\t\t -> firmware_timestamp [" << idx << "/" << kMAX_ITER_TEST << "]: "
                        << gpu_metrics_check.firmware_timestamp << "\n";
          }

          std::cout << "\n";
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
              rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
              std::cout << "\t\t -> system_clock_counter [" << idx << "/" << kMAX_ITER_TEST << "]: "
                        << gpu_metrics_check.system_clock_counter << "\n";
          }

          std::cout << "\n";
          std::cout << " ** Note: Values MAX'ed out "
                    << "(UINTX MAX are unsupported for the version in question) ** " << "\n\n";
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_gpu_metrics_info_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    auto temp_xcd_counter_value = uint16_t(0);
    err = rsmi_dev_metrics_xcd_counter_get(i, &temp_xcd_counter_value);
    if (err != RSMI_STATUS_NOT_SUPPORTED) {
      CHK_ERR_ASRT(err);
    }
  }
}
