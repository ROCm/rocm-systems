/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
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


using GPUMetricResults_t = std::map<std::string, rsmi_status_t>;
GPUMetricResults_t MetricResults{};

template <typename T>
auto print_error_or_value(std::string title, std::string func_name, const T& metric) {
  auto str_values = title;
  const auto status_code = MetricResults.at(func_name);
  if (status_code == rsmi_status_t::RSMI_STATUS_SUCCESS) {
    if constexpr (std::is_array_v<T>) {
      auto idx = uint16_t(0);

      const auto num_elems = static_cast<uint16_t>(std::end(metric) - std::begin(metric));
      str_values += ("\n\t\t num of values: " + std::to_string(num_elems) + "\n");
      for (const auto& el : metric) {
        str_values += "\t\t  [" + std::to_string(idx) + "]: " + std::to_string(el) + "\n";
        ++idx;
      }
      return str_values;
    }
    else if constexpr ((std::is_same_v<T, std::uint16_t>) ||
                      (std::is_same_v<T, std::uint32_t>) ||
                      (std::is_same_v<T, std::uint64_t>)) {

      return str_values += std::to_string(metric);
    }
  }
  else {
    return str_values += ("\n\t\tStatus: [" + std::to_string(status_code) + "] " + "-> " + amd::smi::getRSMIStatusString(status_code));
  }
};

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
            std::cout << "\t  -> format_revision  : " << amd::smi::print_unsigned_int(header_values.format_revision) << "\n";
            std::cout << "\t  -> content_revision : " << amd::smi::print_unsigned_int(header_values.content_revision) << "\n";
            std::cout << "\t--------------------" << "\n";
        }
    }

    rsmi_gpu_metrics_t smu;
    err = rsmi_dev_gpu_metrics_info_get(i, &smu);
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << std::dec << "\tsystem_clock_counter=" << smu.system_clock_counter << '\n';
          std::cout << std::dec << "\ttemperature_edge=" << smu.temperature_edge << '\n';
          std::cout << std::dec << "\ttemperature_hotspot=" << smu.temperature_hotspot << '\n';
          std::cout << std::dec << "\ttemperature_mem=" << smu.temperature_mem << '\n';
          std::cout << std::dec << "\ttemperature_vrgfx=" << smu.temperature_vrgfx << '\n';
          std::cout << std::dec << "\ttemperature_vrsoc=" << smu.temperature_vrsoc << '\n';
          std::cout << std::dec << "\ttemperature_vrmem=" << smu.temperature_vrmem << '\n';
          std::cout << std::dec << "\taverage_gfx_activity=" << smu.average_gfx_activity << '\n';
          std::cout << std::dec << "\taverage_umc_activity=" << smu.average_umc_activity << '\n';
          std::cout << std::dec << "\taverage_mm_activity=" << smu.average_mm_activity << '\n';
          std::cout << std::dec << "\taverage_socket_power=" << smu.average_socket_power << '\n';
          std::cout << std::dec << "\tenergy_accumulator=" << smu.energy_accumulator << '\n';
          std::cout << std::dec << "\taverage_gfxclk_frequency=" << smu.average_gfxclk_frequency << '\n';
          std::cout << std::dec << "\taverage_uclk_frequency=" << smu.average_uclk_frequency << '\n';
          std::cout << std::dec << "\taverage_vclk0_frequency=" << smu.average_vclk0_frequency << '\n';
          std::cout << std::dec << "\taverage_dclk0_frequency=" << smu.average_dclk0_frequency << '\n';
          std::cout << std::dec << "\taverage_vclk1_frequency=" << smu.average_vclk1_frequency << '\n';
          std::cout << std::dec << "\taverage_dclk1_frequency=" << smu.average_dclk1_frequency << '\n';
          std::cout << std::dec << "\tcurrent_gfxclk=" << smu.current_gfxclk << '\n';
          std::cout << std::dec << "\tcurrent_socclk=" << smu.current_socclk << '\n';
          std::cout << std::dec << "\tcurrent_uclk=" << smu.current_uclk << '\n';
          std::cout << std::dec << "\tcurrent_vclk0=" << smu.current_vclk0 << '\n';
          std::cout << std::dec << "\tcurrent_dclk0=" << smu.current_dclk0 << '\n';
          std::cout << std::dec << "\tcurrent_vclk1=" << smu.current_vclk1 << '\n';
          std::cout << std::dec << "\tcurrent_dclk1=" << smu.current_dclk1 << '\n';
          std::cout << std::dec << "\tthrottle_status=" << smu.throttle_status << '\n';
          std::cout << std::dec << "\tcurrent_fan_speed=" << smu.current_fan_speed << '\n';
          std::cout << std::dec << "\tpcie_link_width=" << smu.pcie_link_width << '\n';
          std::cout << std::dec << "\tpcie_link_speed=" << smu.pcie_link_speed << '\n';
          std::cout << std::dec << "\tgfx_activity_acc=" << std::dec << smu.gfx_activity_acc << '\n';
          std::cout << std::dec << "\tmem_activity_acc=" << std::dec << smu.mem_activity_acc  << '\n';

          for (int i = 0; i < RSMI_NUM_HBM_INSTANCES; ++i) {
            std::cout << "\ttemperature_hbm[" << i << "]=" << std::dec << smu.temperature_hbm[i] << '\n';
          }
          std::cout << "\n";
          std::cout << "\tfirmware_timestamp=" << std::dec << smu.firmware_timestamp  << '\n';
          std::cout << "\tvoltage_soc=" << std::dec << smu.voltage_soc  << '\n';
          std::cout << "\tvoltage_gfx=" << std::dec << smu.voltage_gfx  << '\n';
          std::cout << "\tvoltage_mem=" << std::dec << smu.voltage_mem  << '\n';
          std::cout << "\tindep_throttle_status=" << std::dec << smu.indep_throttle_status  << '\n';
          std::cout << "\tcurrent_socket_power=" << std::dec << smu.current_socket_power  << '\n';

          for (int i = 0; i < RSMI_MAX_NUM_VCNS; ++i) {
            std::cout << "\tvcn_activity[" << i << "]=" << std::dec << smu.vcn_activity[i] << '\n';
          }
          std::cout << "\n";

          for (int i = 0; i < RSMI_MAX_NUM_JPEG_ENGS; ++i) {
            std::cout << "\tjpeg_activity[" << i << "]=" << std::dec << smu.jpeg_activity[i] << '\n';
          }
          std::cout << "\n";

          std::cout << "\tgfxclk_lock_status=" << std::dec << smu.gfxclk_lock_status  << '\n';
          std::cout << "\txgmi_link_width=" << std::dec << smu.xgmi_link_width  << '\n';
          std::cout << "\txgmi_link_speed=" << std::dec << smu.xgmi_link_speed  << '\n';
          std::cout << "\tpcie_bandwidth_acc=" << std::dec << smu.pcie_bandwidth_acc  << '\n';
          std::cout << "\tpcie_bandwidth_inst=" << std::dec << smu.pcie_bandwidth_inst  << '\n';
          std::cout << "\tpcie_l0_to_recov_count_acc=" << std::dec << smu.pcie_l0_to_recov_count_acc << '\n';
          std::cout << "\tpcie_replay_count_acc=" << std::dec << smu.pcie_replay_count_acc << '\n';
          std::cout << "\tpcie_replay_rover_count_acc=" << std::dec << smu.pcie_replay_rover_count_acc << '\n';
          for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; ++i) {
            std::cout << "\txgmi_read_data_acc[" << i << "]=" << std::dec << smu.xgmi_read_data_acc[i] << '\n';
          }

          std::cout << "\n";
          for (int i = 0; i < RSMI_MAX_NUM_XGMI_LINKS; ++i) {
            std::cout << "\txgmi_write_data_acc[" << i << "]=" << std::dec << smu.xgmi_write_data_acc[i] << '\n';
          }

          std::cout << "\n";
          for (int i = 0; i < RSMI_MAX_NUM_GFX_CLKS; ++i) {
            std::cout << "\tcurrent_gfxclks[" << i << "]=" << std::dec << smu.current_gfxclks[i] << '\n';
          }

          std::cout << "\n";
          for (int i = 0; i < RSMI_MAX_NUM_CLKS; ++i) {
            std::cout << "\tcurrent_socclks[" << i << "]=" << std::dec << smu.current_socclks[i] << '\n';
          }

          std::cout << "\n";
          for (int i = 0; i < RSMI_MAX_NUM_CLKS; ++i) {
            std::cout << "\tcurrent_vclk0s[" << i << "]=" << std::dec << smu.current_vclk0s[i] << '\n';
          }

          std::cout << "\n";
          for (int i = 0; i < RSMI_MAX_NUM_CLKS; ++i) {
            std::cout << "\tcurrent_dclk0s[" << i << "]=" << std::dec << smu.current_dclk0s[i] << '\n';
          }


          std::cout << "\n\n";
          std::cout << "\t ** -> Checking metrics with constant changes ** " << "\n";
          constexpr uint16_t kMAX_ITER_TEST = 10;
          rsmi_gpu_metrics_t gpu_metrics_check;
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
              rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
              std::cout << "\t\t -> firmware_timestamp [" << idx << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.firmware_timestamp << "\n";
          }

          std::cout << "\n";
          for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
              rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
              std::cout << "\t\t -> system_clock_counter [" << idx << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.system_clock_counter << "\n";
          }

          std::cout << "\n";
          std::cout << " ** Note: Values MAX'ed out (UINTX MAX are unsupported for the version in question) ** " << "\n\n";
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
