/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "gpu_metrics_read.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <string>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "libdrm/amdgpu_drm.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace {

// APU metrics version constants
constexpr uint8_t kApuMetricsV24FormatRevision = 2;
constexpr uint8_t kApuMetricsV24ContentRevision = 4;
constexpr uint8_t kApuMetricsV30FormatRevision = 3;
constexpr uint8_t kApuMetricsV30ContentRevision = 0;
constexpr size_t kApuMetricsV24CoreCount = AMDSMI_APU_V24_CORES;

template <typename T>
void PrintArrayLine(const std::string& label, const T* values, size_t count) {
  std::cout << label << " = [";
  std::copy(values, values + count, amd::smi::make_ostream_joiner(&std::cout, ", "));
  std::cout << "]\n";
}

void PrintApuMetrics(const amdsmi_gpu_metrics_t& smu) {
  if (smu.apu_metrics == nullptr) {
    return;
  }

  const auto& apu = *smu.apu_metrics;
  const bool is_v24 = smu.common_header.format_revision == kApuMetricsV24FormatRevision &&
                      smu.common_header.content_revision == kApuMetricsV24ContentRevision;
  const bool is_v30 = smu.common_header.format_revision == kApuMetricsV30FormatRevision &&
                      smu.common_header.content_revision == kApuMetricsV30ContentRevision;

  const size_t core_count = is_v24 ? kApuMetricsV24CoreCount : AMDSMI_APU_MAX_CORES;
  const size_t l3_count = is_v24 ? AMDSMI_APU_MAX_L3 : 0;

  std::cout << "\n";
  std::cout << "APU METRICS AUXILIARY DATA:\n";
  std::cout << "temperature_gfx = " << std::dec << apu.temperature_gfx << "\n";
  std::cout << "temperature_soc = " << std::dec << apu.temperature_soc << "\n";
  PrintArrayLine("temperature_core", apu.temperature_core, core_count);
  if (l3_count != 0) {
    PrintArrayLine("temperature_l3", apu.temperature_l3, l3_count);
  }
  if (is_v30) {
    std::cout << "temperature_skin = " << std::dec << apu.temperature_skin << "\n";
  }

  std::cout << "\n";
  std::cout << "APU UTILIZATION:\n";
  std::cout << "average_gfx_activity = " << std::dec << apu.average_gfx_activity << "\n";
  if (is_v24) {
    std::cout << "average_mm_activity = " << std::dec << apu.average_mm_activity << "\n";
  }
  if (is_v30) {
    std::cout << "average_vcn_activity = " << std::dec << apu.average_vcn_activity << "\n";
    PrintArrayLine("average_ipu_activity", apu.average_ipu_activity, AMDSMI_APU_MAX_IPU);
    PrintArrayLine("average_core_c0_activity", apu.average_core_c0_activity, core_count);
    std::cout << "average_dram_reads = " << std::dec << apu.average_dram_reads << "\n";
    std::cout << "average_dram_writes = " << std::dec << apu.average_dram_writes << "\n";
    std::cout << "average_ipu_reads = " << std::dec << apu.average_ipu_reads << "\n";
    std::cout << "average_ipu_writes = " << std::dec << apu.average_ipu_writes << "\n";
  }

  std::cout << "\n";
  std::cout << "APU POWER (mW):\n";
  std::cout << "average_socket_power = " << std::dec << apu.average_socket_power << "\n";
  if (is_v24) {
    std::cout << "average_cpu_power = " << std::dec << apu.average_cpu_power << "\n";
    std::cout << "average_soc_power = " << std::dec << apu.average_soc_power << "\n";
  }
  std::cout << "average_gfx_power = " << std::dec << apu.average_gfx_power << "\n";
  PrintArrayLine("average_core_power", apu.average_core_power, core_count);
  if (is_v30) {
    std::cout << "average_ipu_power = " << std::dec << apu.average_ipu_power << "\n";
    std::cout << "average_apu_power = " << std::dec << apu.average_apu_power << "\n";
    std::cout << "average_dgpu_power = " << std::dec << apu.average_dgpu_power << "\n";
    std::cout << "average_all_core_power = " << std::dec << apu.average_all_core_power << "\n";
    std::cout << "average_sys_power = " << std::dec << apu.average_sys_power << "\n";
    std::cout << "stapm_power_limit = " << std::dec << apu.stapm_power_limit << "\n";
    std::cout << "current_stapm_power_limit = " << std::dec << apu.current_stapm_power_limit
              << "\n";
  }

  std::cout << "\n";
  std::cout << "APU AVERAGE CLOCKS (MHz):\n";
  std::cout << "average_gfxclk_frequency = " << std::dec << apu.average_gfxclk_frequency << "\n";
  std::cout << "average_socclk_frequency = " << std::dec << apu.average_socclk_frequency << "\n";
  std::cout << "average_uclk_frequency = " << std::dec << apu.average_uclk_frequency << "\n";
  std::cout << "average_fclk_frequency = " << std::dec << apu.average_fclk_frequency << "\n";
  std::cout << "average_vclk_frequency = " << std::dec << apu.average_vclk_frequency << "\n";
  if (is_v24) {
    std::cout << "average_dclk_frequency = " << std::dec << apu.average_dclk_frequency << "\n";
  }
  if (is_v30) {
    std::cout << "average_vpeclk_frequency = " << std::dec << apu.average_vpeclk_frequency << "\n";
    std::cout << "average_ipuclk_frequency = " << std::dec << apu.average_ipuclk_frequency << "\n";
    std::cout << "average_mpipu_frequency = " << std::dec << apu.average_mpipu_frequency << "\n";
  }

  std::cout << "\n";
  std::cout << "APU CURRENT CLOCKS (MHz):\n";
  if (is_v24) {
    std::cout << "current_gfxclk = " << std::dec << apu.current_gfxclk << "\n";
    std::cout << "current_socclk = " << std::dec << apu.current_socclk << "\n";
    std::cout << "current_uclk = " << std::dec << apu.current_uclk << "\n";
    std::cout << "current_fclk = " << std::dec << apu.current_fclk << "\n";
    std::cout << "current_vclk = " << std::dec << apu.current_vclk << "\n";
    std::cout << "current_dclk = " << std::dec << apu.current_dclk << "\n";
  }
  PrintArrayLine("current_coreclk", apu.current_coreclk, core_count);
  if (l3_count != 0) {
    PrintArrayLine("current_l3clk", apu.current_l3clk, l3_count);
  }
  if (is_v30) {
    std::cout << "current_core_maxfreq = " << std::dec << apu.current_core_maxfreq << "\n";
    std::cout << "current_gfx_maxfreq = " << std::dec << apu.current_gfx_maxfreq << "\n";
  }

  std::cout << "\n";
  std::cout << "APU THROTTLE:\n";
  if (is_v24) {
    std::cout << "throttle_status = " << std::dec << apu.throttle_status << "\n";
    std::cout << "indep_throttle_status = " << std::dec << apu.indep_throttle_status << "\n";
  }
  if (is_v30) {
    std::cout << "throttle_residency_prochot = " << std::dec << apu.throttle_residency_prochot
              << "\n";
    std::cout << "throttle_residency_spl = " << std::dec << apu.throttle_residency_spl << "\n";
    std::cout << "throttle_residency_fppt = " << std::dec << apu.throttle_residency_fppt << "\n";
    std::cout << "throttle_residency_sppt = " << std::dec << apu.throttle_residency_sppt << "\n";
    std::cout << "throttle_residency_thm_core = " << std::dec << apu.throttle_residency_thm_core
              << "\n";
    std::cout << "throttle_residency_thm_gfx = " << std::dec << apu.throttle_residency_thm_gfx
              << "\n";
    std::cout << "throttle_residency_thm_soc = " << std::dec << apu.throttle_residency_thm_soc
              << "\n";
  }

  if (is_v24) {
    std::cout << "\n";
    std::cout << "APU FAN / VOLTAGE / CURRENT:\n";
    std::cout << "fan_pwm = " << std::dec << apu.fan_pwm << "\n";
    std::cout << "average_cpu_voltage = " << std::dec << apu.average_cpu_voltage << "\n";
    std::cout << "average_soc_voltage = " << std::dec << apu.average_soc_voltage << "\n";
    std::cout << "average_gfx_voltage = " << std::dec << apu.average_gfx_voltage << "\n";
    std::cout << "average_cpu_current = " << std::dec << apu.average_cpu_current << "\n";
    std::cout << "average_soc_current = " << std::dec << apu.average_soc_current << "\n";
    std::cout << "average_gfx_current = " << std::dec << apu.average_gfx_current << "\n";

    std::cout << "\n";
    std::cout << "APU AVERAGE TEMPERATURE:\n";
    std::cout << "average_temperature_gfx = " << std::dec << apu.average_temperature_gfx << "\n";
    std::cout << "average_temperature_soc = " << std::dec << apu.average_temperature_soc << "\n";
    PrintArrayLine("average_temperature_core", apu.average_temperature_core, core_count);
    PrintArrayLine("average_temperature_l3", apu.average_temperature_l3, l3_count);
  }

  if (is_v30) {
    std::cout << "\n";
    std::cout << "APU OTHER:\n";
    std::cout << "time_filter_alphavalue = " << std::dec << apu.time_filter_alphavalue << "\n";
  }
}

}  // namespace

TestGpuMetricsRead::TestGpuMetricsRead() : TestBase() {
  set_title("AMDSMI GPU Metrics Read Test");
  set_description(
      "The GPU Metrics tests verifies that "
      "the gpu metrics info can be read properly.");
}

TestGpuMetricsRead::~TestGpuMetricsRead(void) {}

void TestGpuMetricsRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestGpuMetricsRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

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
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    std::cout << "Device #" << std::to_string(i) << "\n";

    IF_VERB(STANDARD) {
      std::cout << "\n\n";
      std::cout << "\t**GPU METRICS: Using static struct (Backwards Compatibility):\n";
    }
    amdsmi_gpu_metrics_t smu = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_metrics_info(processor_handles_[i], &smu);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        continue;
      }
    } else {
      auto temp_xcd_counter_value = uint16_t(0);
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_xcd_counter", "gpu=" + std::to_string(i), VERB(STANDARD));
      auto ret_xcd = amdsmi_get_gpu_xcd_counter(processor_handles_[i], &temp_xcd_counter_value);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret_xcd, AMDSMI_STATUS_SUCCESS);
      IF_VERB(STANDARD) {
        std::cout << "\t\t** amdsmi_get_gpu_xcd_counter(): "
                  << smi_amdgpu_get_status_string(ret_xcd, false)
                  << "\n\t\t** XCD Counter Value: " << temp_xcd_counter_value << "\n";
      }
      CHK_ERR_ASRT(err);
      amdsmi_asic_info_t asic_info = {};
      err = amdsmi_get_gpu_asic_info(processor_handles_[i], &asic_info);
      ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
      if ((asic_info.flags & AMDGPU_IDS_FLAGS_FUSION) != 0 && smu.apu_metrics == nullptr) {
        std::cout << "Fusion/APU device detected but APU metrics not available, skipping device "
                  << i << std::endl;
        continue;
      }
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
        std::copy(std::begin(smu.temperature_hbm), std::end(smu.temperature_hbm),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << "\n";
        std::cout << "UTILIZATION (%):\n";
        std::cout << std::dec << "average_gfx_activity=" << smu.average_gfx_activity << "\n";
        std::cout << std::dec << "average_umc_activity=" << smu.average_umc_activity << "\n";
        std::cout << std::dec << "average_mm_activity=" << smu.average_mm_activity << "\n";
        std::cout << std::dec << "vcn_activity= [";
        std::copy(std::begin(smu.vcn_activity), std::end(smu.vcn_activity),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << "\n";
        std::cout << std::dec << "jpeg_activity= [";
        std::copy(std::begin(smu.jpeg_activity), std::end(smu.jpeg_activity),
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
        std::cout << std::dec << "average_vclk0_frequency=" << smu.average_vclk0_frequency << "\n";
        std::cout << std::dec << "average_dclk0_frequency=" << smu.average_dclk0_frequency << "\n";
        std::cout << std::dec << "average_vclk1_frequency=" << smu.average_vclk1_frequency << "\n";
        std::cout << std::dec << "average_dclk1_frequency=" << smu.average_dclk1_frequency << "\n";

        std::cout << "\n";
        std::cout << "CURRENT CLOCKS (MHz):\n";
        std::cout << std::dec << "current_gfxclk=" << smu.current_gfxclk << "\n";
        std::cout << std::dec << "current_gfxclks= [";
        std::copy(std::begin(smu.current_gfxclks), std::end(smu.current_gfxclks),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "current_socclk=" << smu.current_socclk << "\n";
        std::cout << std::dec << "current_socclks= [";
        std::copy(std::begin(smu.current_socclks), std::end(smu.current_socclks),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "current_uclk=" << smu.current_uclk << "\n";
        std::cout << std::dec << "current_vclk0=" << smu.current_vclk0 << "\n";
        std::cout << std::dec << "current_vclk0s= [";
        std::copy(std::begin(smu.current_vclk0s), std::end(smu.current_vclk0s),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "current_dclk0=" << smu.current_dclk0 << "\n";
        std::cout << std::dec << "current_dclk0s= [";
        std::copy(std::begin(smu.current_dclk0s), std::end(smu.current_dclk0s),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "current_vclk1=" << smu.current_vclk1 << "\n";
        std::cout << std::dec << "current_dclk1=" << smu.current_dclk1 << "\n";

        std::cout << "\n";
        std::cout << "THROTTLE STATUS:\n";
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
        std::cout << "mem_activity_acc=" << std::dec << smu.mem_activity_acc << "\n";

        std::cout << "\n";
        std::cout << "XGMI ACCUMULATED DATA TRANSFER SIZE (KB):\n";
        std::cout << std::dec << "xgmi_read_data_acc= [";
        std::copy(std::begin(smu.xgmi_read_data_acc), std::end(smu.xgmi_read_data_acc),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "xgmi_write_data_acc= [";
        std::copy(std::begin(smu.xgmi_write_data_acc), std::end(smu.xgmi_write_data_acc),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        std::cout << std::dec << "xgmi_link_status= [";
        std::copy(std::begin(smu.xgmi_link_status), std::end(smu.xgmi_link_status),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << std::dec << "]\n";

        // Voltage (mV)
        std::cout << "voltage_soc = " << std::dec << smu.voltage_soc << "\n";
        std::cout << "voltage_gfx = " << std::dec << smu.voltage_gfx << "\n";
        std::cout << "voltage_mem = " << std::dec << smu.voltage_mem << "\n";

        std::cout << "indep_throttle_status = " << std::dec << smu.indep_throttle_status << "\n";

        // Clock Lock Status. Each bit corresponds to clock instance
        std::cout << "gfxclk_lock_status (in hex) = " << std::hex << smu.gfxclk_lock_status
                  << std::dec << "\n";

        // Bandwidth (GB/sec)
        std::cout << "pcie_bandwidth_acc=" << std::dec << smu.pcie_bandwidth_acc << "\n";
        std::cout << "pcie_bandwidth_inst=" << std::dec << smu.pcie_bandwidth_inst << "\n";

        // VRAM max bandwidth at max memory clock (GB/sec)
        std::cout << "vram_max_bandwidth=" << std::dec << smu.vram_max_bandwidth << "\n";

        // Counts
        std::cout << "pcie_l0_to_recov_count_acc= " << std::dec << smu.pcie_l0_to_recov_count_acc
                  << "\n";
        std::cout << "pcie_replay_count_acc= " << std::dec << smu.pcie_replay_count_acc << "\n";
        std::cout << "pcie_replay_rover_count_acc= " << std::dec << smu.pcie_replay_rover_count_acc
                  << "\n";
        std::cout << "pcie_nak_sent_count_acc= " << std::dec << smu.pcie_nak_sent_count_acc << "\n";
        std::cout << "pcie_nak_rcvd_count_acc= " << std::dec << smu.pcie_nak_rcvd_count_acc << "\n";

        // Accumulation cycle counter
        // Accumulated throttler residencies
        std::cout << "\n";
        std::cout << "RESIDENCY ACCUMULATION / COUNTER:\n";
        std::cout << "accumulation_counter = " << std::dec << smu.accumulation_counter << "\n";
        std::cout << "prochot_residency_acc = " << std::dec << smu.prochot_residency_acc << "\n";
        std::cout << "ppt_residency_acc = " << std::dec << smu.ppt_residency_acc << "\n";
        std::cout << "socket_thm_residency_acc = " << std::dec << smu.socket_thm_residency_acc
                  << "\n";
        std::cout << "vr_thm_residency_acc = " << std::dec << smu.vr_thm_residency_acc << "\n";
        std::cout << "hbm_thm_residency_acc = " << std::dec << smu.hbm_thm_residency_acc << "\n";

        // Number of current partitions
        std::cout << "num_partition = " << std::dec << smu.num_partition << "\n";

        // PCIE other end recovery counter
        std::cout << "pcie_lc_perf_other_end_recovery = " << std::dec
                  << smu.pcie_lc_perf_other_end_recovery << "\n";

        std::cout << std::dec << "xcp_stats.gfx_busy_inst = \n";
        auto xcp = 0;
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_busy_inst), std::end(row.gfx_busy_inst),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.jpeg_busy = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.jpeg_busy), std::end(row.jpeg_busy),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.vcn_busy = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.vcn_busy), std::end(row.vcn_busy),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.gfx_busy_acc = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_busy_acc), std::end(row.gfx_busy_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.gfx_below_host_limit_acc = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
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
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_below_host_limit_ppt_acc),
                    std::end(row.gfx_below_host_limit_ppt_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.gfx_below_host_limit_thm_acc = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_below_host_limit_thm_acc),
                    std::end(row.gfx_below_host_limit_thm_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.gfx_low_utilization_acc = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_low_utilization_acc), std::end(row.gfx_low_utilization_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        xcp = 0;
        std::cout << std::dec << "xcp_stats.gfx_below_host_limit_total_acc = \n";
        for (auto& row : smu.xcp_stats) {
          std::cout << "XCP[" << xcp << "] = "
                    << "[ ";
          std::copy(std::begin(row.gfx_below_host_limit_total_acc),
                    std::end(row.gfx_below_host_limit_total_acc),
                    amd::smi::make_ostream_joiner(&std::cout, ", "));
          std::cout << " ]\n";
          xcp++;
        }

        if (smu.apu_metrics != nullptr) {
          PrintApuMetrics(smu);
        }

        std::cout << "\n\n";
        std::cout << "\t ** -> Checking metrics with constant changes ** "
                  << "\n";
        constexpr uint16_t kMAX_ITER_TEST = 10;
        amdsmi_gpu_metrics_t gpu_metrics_check = {};
        for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
          DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i),
                             VERB(STANDARD));
          auto ret = amdsmi_get_gpu_metrics_info(processor_handles_[i], &gpu_metrics_check);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
          std::cout << "\t\t -> firmware_timestamp [" << idx << "/" << kMAX_ITER_TEST
                    << "]: " << gpu_metrics_check.firmware_timestamp << "\n";
        }

        std::cout << "\n";
        for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
          DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i),
                             VERB(STANDARD));
          auto ret = amdsmi_get_gpu_metrics_info(processor_handles_[i], &gpu_metrics_check);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
          std::cout << "\t\t -> system_clock_counter [" << idx << "/" << kMAX_ITER_TEST
                    << "]: " << gpu_metrics_check.system_clock_counter << "\n";
        }

        std::cout << "\n";
        std::cout << " ** Note: Values MAX'ed out "
                  << "(UINTX MAX are unsupported for the version in question) ** "
                  << "\n\n";
      }
    }

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_metrics_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_metrics_info(processor_handles_[i], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
