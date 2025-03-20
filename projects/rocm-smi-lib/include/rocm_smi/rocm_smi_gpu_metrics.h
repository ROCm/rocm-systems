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

#ifndef ROCM_SMI_ROCM_SMI_GPU_METRICS_H_
#define ROCM_SMI_ROCM_SMI_GPU_METRICS_H_

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <tuple>
#include <variant>
#include <vector>


/**
 *  All 1.4 and newer GPU metrics are now defined in this header.
 *
 */
namespace amd::smi {

constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MAJOR_VER_1 = 1;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_1 = 1;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_2 = 2;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_3 = 3;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_4 = 4;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_8 = 8;
constexpr uint32_t kRSMI_LATEST_GPU_METRICS_API_CONTENT_MAJOR_VER
                    = kRSMI_GPU_METRICS_API_CONTENT_MAJOR_VER_1;
constexpr uint32_t kRSMI_LATEST_GPU_METRICS_API_CONTENT_MINOR_VER
                    = kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_8;


//  Note: This *must* match NUM_HBM_INSTANCES
constexpr uint32_t kRSMI_MAX_NUM_HBM_INSTANCES = 4;

//  Note: This *must* match NUM_XGMI_LINKS
constexpr uint32_t kRSMI_MAX_NUM_XGMI_LINKS = 8;

//  Note: This *must* match MAX_GFX_CLKS
constexpr uint32_t kRSMI_MAX_NUM_GFX_CLKS = 8;

//  Note: This *must* match MAX_CLKS
constexpr uint32_t kRSMI_MAX_NUM_CLKS = 4;

//  Note: This *must* match NUM_VCN
constexpr uint32_t kRSMI_MAX_NUM_VCNS = 4;

//  Note: This *must* match NUM_JPEG_ENG
constexpr uint32_t kRSMI_MAX_JPEG_ENGINES = 32;

//  Note: Updated for amdgpu_xcp_metrics_v1_2.
//  Document provides NUM_JPEG_ENG_V1 but will rename to kRSMI_MAX_NUM_JPEG_ENG_V1
constexpr uint32_t kRSMI_MAX_NUM_JPEG_ENG_V1 = 40;

//  Note: This *must* match MAX_XCC
constexpr uint32_t kRSMI_MAX_NUM_XCC = 8;

//  Note: This *must* match MAX_XCP
constexpr uint32_t kRSMI_MAX_NUM_XCP = 8;


struct AMDGpuMetricsHeader_v1_t {
  uint16_t m_structure_size;
  uint8_t  m_format_revision;
  uint8_t  m_content_revision;
};
struct amdgpu_xcp_metrics {
  /* Utilization Instantaneous (%) */
  uint32_t gfx_busy_inst[kRSMI_MAX_NUM_XCC];
  uint16_t jpeg_busy[kRSMI_MAX_JPEG_ENGINES];
  uint16_t vcn_busy[kRSMI_MAX_NUM_VCNS];

  /* Utilization Accumulated (%) */
  uint64_t gfx_busy_acc[kRSMI_MAX_NUM_XCC];
};

struct amdgpu_xcp_metrics_v1_1 {
  /* Utilization Instantaneous (%) */
  uint32_t gfx_busy_inst[kRSMI_MAX_NUM_XCC];
  uint16_t jpeg_busy[kRSMI_MAX_JPEG_ENGINES];
  uint16_t vcn_busy[kRSMI_MAX_NUM_VCNS];

  /* Utilization Accumulated (%) */
  uint64_t gfx_busy_acc[kRSMI_MAX_NUM_XCC];

  /* Total App Clock Counter Accumulated */
  uint64_t gfx_below_host_limit_acc[kRSMI_MAX_NUM_XCC];
};

/* new for gpu metrics v1.8 */
struct amdgpu_xcp_metrics_v1_2 {
  /* Utilization Instantaneous (%) */
  uint32_t gfx_busy_inst[kRSMI_MAX_NUM_XCC];
  uint16_t jpeg_busy[kRSMI_MAX_NUM_JPEG_ENG_V1];
  uint16_t vcn_busy[kRSMI_MAX_NUM_VCNS];

  /* Utilization Accumulated (%) */
  uint64_t gfx_busy_acc[kRSMI_MAX_NUM_XCC];

  /* Total App Clock Counter Accumulated */
  uint64_t gfx_below_host_limit_ppt_acc[kRSMI_MAX_NUM_XCC];
  uint64_t gfx_below_host_limit_thm_acc[kRSMI_MAX_NUM_XCC];
  uint64_t gfx_low_utilization_acc[kRSMI_MAX_NUM_XCC];
  uint64_t gfx_below_host_limit_total_acc[kRSMI_MAX_NUM_XCC];
};

struct AMDGpuMetricsBase_t {
    virtual ~AMDGpuMetricsBase_t() = default;
};
using AMDGpuMetricsBaseRef = AMDGpuMetricsBase_t&;


struct AMDGpuMetrics_v11_t {
  ~AMDGpuMetrics_v11_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature
  uint16_t m_temperature_edge;
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrgfx;
  uint16_t m_temperature_vrsoc;
  uint16_t m_temperature_vrmem;

  // Utilization
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;    // memory controller
  uint16_t m_average_mm_activity;     // UVD or VCN

  // Power/Energy
  uint16_t m_average_socket_power;
  uint64_t m_energy_accumulator;

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;

  // Average clocks
  uint16_t m_average_gfxclk_frequency;
  uint16_t m_average_socclk_frequency;
  uint16_t m_average_uclk_frequency;
  uint16_t m_average_vclk0_frequency;
  uint16_t m_average_dclk0_frequency;
  uint16_t m_average_vclk1_frequency;
  uint16_t m_average_dclk1_frequency;

  // Current clocks
  uint16_t m_current_gfxclk;
  uint16_t m_current_socclk;
  uint16_t m_current_uclk;
  uint16_t m_current_vclk0;
  uint16_t m_current_dclk0;
  uint16_t m_current_vclk1;
  uint16_t m_current_dclk1;

  // Throttle status
  uint32_t m_throttle_status;

  // Fans
  uint16_t m_current_fan_speed;

  // Link width/speed
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;      // in 0.1 GT/s

  uint16_t m_padding;

  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  uint16_t m_temperature_hbm[kRSMI_MAX_NUM_HBM_INSTANCES];
};

struct AMDGpuMetrics_v12_t {
  ~AMDGpuMetrics_v12_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature
  uint16_t m_temperature_edge;
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrgfx;
  uint16_t m_temperature_vrsoc;
  uint16_t m_temperature_vrmem;

  // Utilization
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;    // memory controller
  uint16_t m_average_mm_activity;     // UVD or VCN

  // Power/Energy
  uint16_t m_average_socket_power;
  uint64_t m_energy_accumulator;      // v1 mod. (32->64)

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;    // v1 mod. (moved from top of struct)

  // Average clocks
  uint16_t m_average_gfxclk_frequency;
  uint16_t m_average_socclk_frequency;
  uint16_t m_average_uclk_frequency;
  uint16_t m_average_vclk0_frequency;
  uint16_t m_average_dclk0_frequency;
  uint16_t m_average_vclk1_frequency;
  uint16_t m_average_dclk1_frequency;

  // Current clocks
  uint16_t m_current_gfxclk;
  uint16_t m_current_socclk;
  uint16_t m_current_uclk;
  uint16_t m_current_vclk0;
  uint16_t m_current_dclk0;
  uint16_t m_current_vclk1;
  uint16_t m_current_dclk1;

  // Throttle status
  uint32_t m_throttle_status;

  // Fans
  uint16_t m_current_fan_speed;

  // Link width/speed
  uint16_t m_pcie_link_width;         // v1 mod.(8->16)
  uint16_t m_pcie_link_speed;         // in 0.1 GT/s; v1 mod. (8->16)

  uint16_t m_padding;                 // new in v1

  uint32_t m_gfx_activity_acc;        // new in v1
  uint32_t m_mem_activity_acc;        // new in v1
  uint16_t m_temperature_hbm[kRSMI_MAX_NUM_HBM_INSTANCES];  // new in v1

  // PMFW attached timestamp (10ns resolution)
  uint64_t m_firmware_timestamp;
};

struct AMDGpuMetrics_v13_t {
  ~AMDGpuMetrics_v13_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature
  uint16_t m_temperature_edge;
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrgfx;
  uint16_t m_temperature_vrsoc;
  uint16_t m_temperature_vrmem;

  // Utilization
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;    // memory controller
  uint16_t m_average_mm_activity;     // UVD or VCN

  // Power/Energy
  uint16_t m_average_socket_power;
  uint64_t m_energy_accumulator;      // v1 mod. (32->64)

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;    // v1 mod. (moved from top of struct)

  // Average clocks
  uint16_t m_average_gfxclk_frequency;
  uint16_t m_average_socclk_frequency;
  uint16_t m_average_uclk_frequency;
  uint16_t m_average_vclk0_frequency;
  uint16_t m_average_dclk0_frequency;
  uint16_t m_average_vclk1_frequency;
  uint16_t m_average_dclk1_frequency;

  // Current clocks
  uint16_t m_current_gfxclk;
  uint16_t m_current_socclk;
  uint16_t m_current_uclk;
  uint16_t m_current_vclk0;
  uint16_t m_current_dclk0;
  uint16_t m_current_vclk1;
  uint16_t m_current_dclk1;

  // Throttle status
  uint32_t m_throttle_status;

  // Fans
  uint16_t m_current_fan_speed;

  // Link width/speed
  uint16_t m_pcie_link_width;         // v1 mod.(8->16)
  uint16_t m_pcie_link_speed;         // in 0.1 GT/s; v1 mod. (8->16)

  uint16_t m_padding;                 // new in v1

  uint32_t m_gfx_activity_acc;        // new in v1
  uint32_t m_mem_activity_acc;        // new in v1
  uint16_t m_temperature_hbm[kRSMI_MAX_NUM_HBM_INSTANCES];  // new in v1

  // PMFW attached timestamp (10ns resolution)
  uint64_t m_firmware_timestamp;

  // Voltage (mV)
  uint16_t m_voltage_soc;
  uint16_t m_voltage_gfx;
  uint16_t m_voltage_mem;

  uint16_t m_padding1;

  // Throttle status
  uint64_t m_indep_throttle_status;
};

struct AMDGpuMetrics_v14_t {
  ~AMDGpuMetrics_v14_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature (Celsius). It will be zero (0) if unsupported.
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  // Power (Watts)
  uint16_t m_current_socket_power;

  // Utilization (%)
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;            // memory controller
  uint16_t m_vcn_activity[kRSMI_MAX_NUM_VCNS];  // VCN instances activity percent (encode/decode)

  // Energy (15.259uJ (2^-16) units)
  uint64_t m_energy_accumulator;

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;

  // Throttle status
  uint32_t m_throttle_status;

  // Clock Lock Status. Each bit corresponds to clock instance
  uint32_t m_gfxclk_lock_status;

  // Link width (number of lanes) and speed (in 0.1 GT/s)
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;  // in 0.1 GT/s

  // XGMI bus width and bitrate (in Gbps)
  uint16_t m_xgmi_link_width;
  uint16_t m_xgmi_link_speed;

  // Utilization Accumulated (%)
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  // PCIE accumulated bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_acc;

  // PCIE instantaneous bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_inst;

  // PCIE L0 to recovery state transition accumulated count
  uint64_t m_pcie_l0_to_recov_count_acc;

  // PCIE replay accumulated count
  uint64_t m_pcie_replay_count_acc;

  // PCIE replay rollover accumulated count
  uint64_t m_pcie_replay_rover_count_acc;

  // XGMI accumulated data transfer size(KiloBytes)
  uint64_t m_xgmi_read_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];
  uint64_t m_xgmi_write_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];

  // PMFW attached timestamp (10ns resolution)
  uint64_t m_firmware_timestamp;

  // Current clocks (Mhz)
  uint16_t m_current_gfxclk[kRSMI_MAX_NUM_GFX_CLKS];
  uint16_t m_current_socclk[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_vclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_dclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_uclk;

  uint16_t m_padding;
};

struct AMDGpuMetrics_v15_t {
  ~AMDGpuMetrics_v15_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature (Celsius). It will be zero (0) if unsupported.
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  // Power (Watts)
  uint16_t m_current_socket_power;

  // Utilization (%)
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;            // memory controller
  uint16_t m_vcn_activity[kRSMI_MAX_NUM_VCNS];  // VCN instances activity percent (encode/decode)
  uint16_t m_jpeg_activity[kRSMI_MAX_JPEG_ENGINES];  // JPEG activity percent (encode/decode)

  // Energy (15.259uJ (2^-16) units)
  uint64_t m_energy_accumulator;

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;

  // Throttle status
  uint32_t m_throttle_status;

  // Clock Lock Status. Each bit corresponds to clock instance
  uint32_t m_gfxclk_lock_status;

  // Link width (number of lanes) and speed (in 0.1 GT/s)
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;  // in 0.1 GT/s

  // XGMI bus width and bitrate (in Gbps)
  uint16_t m_xgmi_link_width;
  uint16_t m_xgmi_link_speed;

  // Utilization Accumulated (%)
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  // PCIE accumulated bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_acc;

  // PCIE instantaneous bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_inst;

  // PCIE L0 to recovery state transition accumulated count
  uint64_t m_pcie_l0_to_recov_count_acc;

  // PCIE replay accumulated count
  uint64_t m_pcie_replay_count_acc;

  // PCIE replay rollover accumulated count
  uint64_t m_pcie_replay_rover_count_acc;

  // PCIE NAK sent accumulated count
  uint32_t m_pcie_nak_sent_count_acc;

  // PCIE NAK received accumulated count
  uint32_t m_pcie_nak_rcvd_count_acc;

  // XGMI accumulated data transfer size(KiloBytes)
  uint64_t m_xgmi_read_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];
  uint64_t m_xgmi_write_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];

  // PMFW attached timestamp (10ns resolution)
  uint64_t m_firmware_timestamp;

  // Current clocks (Mhz)
  uint16_t m_current_gfxclk[kRSMI_MAX_NUM_GFX_CLKS];
  uint16_t m_current_socclk[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_vclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_dclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_uclk;

  uint16_t m_padding;
};
struct AMDGpuMetrics_v16_t {
  ~AMDGpuMetrics_v16_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature (Celsius). It will be zero (0) if unsupported.
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  // Power (Watts)
  uint16_t m_current_socket_power;

  // Utilization (%)
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;            // memory controller

  // Energy (15.259uJ (2^-16) units)
  uint64_t m_energy_accumulator;

  // Driver attached timestamp (in ns)
  uint64_t m_system_clock_counter;

  /*
   * Important: bumped up public to uint64_t due to planned size increase
   * for newer ASICs
   */
  /* Accumulation cycle counter */
  uint32_t m_accumulation_counter;

  /* Accumulated throttler residencies */
  uint32_t m_prochot_residency_acc;
  uint32_t m_ppt_residency_acc;
  uint32_t m_socket_thm_residency_acc;
  uint32_t m_vr_thm_residency_acc;
  uint32_t m_hbm_thm_residency_acc;

  // Clock Lock Status. Each bit corresponds to clock instance
  uint32_t m_gfxclk_lock_status;

  // Link width (number of lanes) and speed (in 0.1 GT/s)
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;  // in 0.1 GT/s

  // XGMI bus width and bitrate (in Gbps)
  uint16_t m_xgmi_link_width;
  uint16_t m_xgmi_link_speed;

  // Utilization Accumulated (%)
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  // PCIE accumulated bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_acc;

  // PCIE instantaneous bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_inst;

  // PCIE L0 to recovery state transition accumulated count
  uint64_t m_pcie_l0_to_recov_count_acc;

  // PCIE replay accumulated count
  uint64_t m_pcie_replay_count_acc;

  // PCIE replay rollover accumulated count
  uint64_t m_pcie_replay_rover_count_acc;

  // PCIE NAK sent accumulated count
  uint32_t m_pcie_nak_sent_count_acc;

  // PCIE NAK received accumulated count
  uint32_t m_pcie_nak_rcvd_count_acc;

  // XGMI accumulated data transfer size(KiloBytes)
  uint64_t m_xgmi_read_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];
  uint64_t m_xgmi_write_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];

  // PMFW attached timestamp (10ns resolution)
  uint64_t m_firmware_timestamp;

  // Current clocks (Mhz)
  uint16_t m_current_gfxclk[kRSMI_MAX_NUM_GFX_CLKS];
  uint16_t m_current_socclk[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_vclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_dclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_uclk;

  /* Number of current partition */
  uint16_t m_num_partition;

  /* XCP (Graphic Cluster Partitions) metrics stats */
  struct amdgpu_xcp_metrics m_xcp_stats[kRSMI_MAX_NUM_XCP];

  /* PCIE other end recovery counter */
  uint32_t m_pcie_lc_perf_other_end_recovery;
};

struct AMDGpuMetrics_v17_t {
  ~AMDGpuMetrics_v17_t() = default;
  struct AMDGpuMetricsHeader_v1_t m_common_header;

  /* Temperature (Celsius) */
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  /* Power (Watts) */
  uint16_t m_current_socket_power;

  /* Utilization (%) */
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;  // memory controller

  /* VRAM max bandwidth at max memory clock (GB/s) */
  uint64_t m_vram_max_bandwidth;  // new for 1.7

  /* Energy (15.259uJ (2^-16) units) */
  uint64_t m_energy_accumulator;

  /* Driver attached timestamp (in ns) */
  uint64_t m_system_clock_counter;

  /* Accumulation cycle counter */
  uint32_t m_accumulation_counter;

  /* Accumulated throttler residencies */
  uint32_t m_prochot_residency_acc;
  uint32_t m_ppt_residency_acc;
  uint32_t m_socket_thm_residency_acc;
  uint32_t m_vr_thm_residency_acc;
  uint32_t m_hbm_thm_residency_acc;

  /* Clock Lock Status. Each bit corresponds to clock instance */
  uint32_t m_gfxclk_lock_status;

  /* Link width (number of lanes) and speed (in 0.1 GT/s) */
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;

  /* XGMI bus width and bitrate (in Gbps) */
  uint16_t m_xgmi_link_width;
  uint16_t m_xgmi_link_speed;

  /* Utilization Accumulated (%) */
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  /*PCIE accumulated bandwidth (GB/sec) */
  uint64_t m_pcie_bandwidth_acc;

  /*PCIE instantaneous bandwidth (GB/sec) */
  uint64_t m_pcie_bandwidth_inst;

  /* PCIE L0 to recovery state transition accumulated count */
  uint64_t m_pcie_l0_to_recov_count_acc;

  /* PCIE replay accumulated count */
  uint64_t m_pcie_replay_count_acc;

  /* PCIE replay rollover accumulated count */
  uint64_t m_pcie_replay_rover_count_acc;

  /* PCIE NAK sent accumulated count */
  uint32_t m_pcie_nak_sent_count_acc;

  /* PCIE NAK received accumulated count */
  uint32_t m_pcie_nak_rcvd_count_acc;

  /* XGMI accumulated data transfer size(KiloBytes) */
  uint64_t m_xgmi_read_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];
  uint64_t m_xgmi_write_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];

  /* XGMI link status(up/down) */
  uint16_t m_xgmi_link_status[kRSMI_MAX_NUM_XGMI_LINKS];  // new for 1.7

  uint16_t m_padding;

  /* PMFW attached timestamp (10ns resolution) */
  uint64_t m_firmware_timestamp;

  /* Current clocks (Mhz) */
  uint16_t m_current_gfxclk[kRSMI_MAX_NUM_GFX_CLKS];
  uint16_t m_current_socclk[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_vclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_dclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_uclk;

  /* Number of current partition */
  uint16_t m_num_partition;

  /* XCP metrics stats */
  struct amdgpu_xcp_metrics_v1_1 m_xcp_stats[kRSMI_MAX_NUM_XCP];

  /* PCIE other end recovery counter */
  uint32_t m_pcie_lc_perf_other_end_recovery;
};

struct AMDGpuMetrics_v18_t {
  ~AMDGpuMetrics_v18_t() = default;
  struct AMDGpuMetricsHeader_v1_t m_common_header;

  /* Temperature (Celsius) */
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  /* Power (Watts) */
  uint16_t m_current_socket_power;

  /* Utilization (%) */
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;  // memory controller

  /* VRAM max bandwidthi (in GB/sec) at max memory clock */
  uint64_t m_mem_max_bandwidth;

  /* Energy (15.259uJ (2^-16) units) */
  uint64_t m_energy_accumulator;

  /* Driver attached timestamp (in ns) */
  uint64_t m_system_clock_counter;

  /* Accumulation cycle counter */
  uint32_t m_accumulation_counter;

  /* Accumulated throttler residencies */
  uint32_t m_prochot_residency_acc;
  uint32_t m_ppt_residency_acc;
  uint32_t m_socket_thm_residency_acc;
  uint32_t m_vr_thm_residency_acc;
  uint32_t m_hbm_thm_residency_acc;

  /* Clock Lock Status. Each bit corresponds to clock instance */
  uint32_t m_gfxclk_lock_status;

  /* Link width (number of lanes) and speed (in 0.1 GT/s) */
  uint16_t m_pcie_link_width;
  uint16_t m_pcie_link_speed;

  /* XGMI bus width and bitrate (in Gbps) */
  uint16_t m_xgmi_link_width;
  uint16_t m_xgmi_link_speed;

  /* Utilization Accumulated (%) */
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  /*PCIE accumulated bandwidth (GB/sec) */
  uint64_t m_pcie_bandwidth_acc;

  /*PCIE instantaneous bandwidth (GB/sec) */
  uint64_t m_pcie_bandwidth_inst;

  /* PCIE L0 to recovery state transition accumulated count */
  uint64_t m_pcie_l0_to_recov_count_acc;

  /* PCIE replay accumulated count */
  uint64_t m_pcie_replay_count_acc;

  /* PCIE replay rollover accumulated count */
  uint64_t m_pcie_replay_rover_count_acc;

  /* PCIE NAK sent  accumulated count */
  uint32_t m_pcie_nak_sent_count_acc;

  /* PCIE NAK received accumulated count */
  uint32_t m_pcie_nak_rcvd_count_acc;

  /* XGMI accumulated data transfer size(KiloBytes) */
  uint64_t m_xgmi_read_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];
  uint64_t m_xgmi_write_data_acc[kRSMI_MAX_NUM_XGMI_LINKS];

  /* XGMI link status(active/inactive) */
  uint16_t m_xgmi_link_status[kRSMI_MAX_NUM_XGMI_LINKS];

  uint16_t m_padding;

  /* PMFW attached timestamp (10ns resolution) */
  uint64_t m_firmware_timestamp;

  /* Current clocks (Mhz) */
  uint16_t m_current_gfxclk[kRSMI_MAX_NUM_GFX_CLKS];
  uint16_t m_current_socclk[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_vclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_dclk0[kRSMI_MAX_NUM_CLKS];
  uint16_t m_current_uclk;

  /* Number of current partition */
  uint16_t m_num_partition;

  /* XCP metrics stats */
  struct amdgpu_xcp_metrics_v1_2 m_xcp_stats[kRSMI_MAX_NUM_XCP];

  /* PCIE other end recovery counter */
  uint32_t m_pcie_lc_perf_other_end_recovery;
};
using AMGpuMetricsLatest_t = AMDGpuMetrics_v18_t;

/**
 *  This is GPU Metrics version that gets to public access.
 *  It is a unique/unified version (joined) of the previous
 *  versions (1.2 to latest 1.4). Data fields not used/relevant
 *  for the current driver version and GPU metrics version will
 *  not be populated, and therefore 0s (zeroes).
 *
 *  If/in case anything new is added to a new version and there is
 *  a requirement to make it publicly available, into a single static
 *  table/form/struct, then it should be added here.
 *
 */
using AMGpuMetricsPublicLatest_t = rsmi_gpu_metrics_t;
using AMGpuMetricsPublicLatestTupl_t = std::tuple<rsmi_status_t, AMGpuMetricsPublicLatest_t>;

using GpuMetricU16Tbl_t = std::vector<uint16_t>;
using GpuMetricU32Tbl_t = std::vector<uint32_t>;
using GpuMetricU64Tbl_t = std::vector<uint64_t>;

using GPUMetricTempHbm_t = decltype(AMDGpuMetrics_v13_t::m_temperature_hbm);
using GPUMetricTempHbmTbl_t = GpuMetricU16Tbl_t;

using GPUMetricVcnActivity_t = decltype(AMDGpuMetrics_v14_t::m_vcn_activity);
using GPUMetricVcnActivityTbl_t = GpuMetricU16Tbl_t;

using GPUMetricJpegActivity_t = decltype(AMDGpuMetrics_v15_t::m_jpeg_activity);
using GPUMetricJpegActivityTbl_t = GpuMetricU16Tbl_t;

using GPUMetricXgmiReadDataAcc_t = decltype(AMDGpuMetrics_v14_t::m_xgmi_read_data_acc);
using GPUMetricXgmiWriteDataAcc_t = decltype(AMDGpuMetrics_v14_t::m_xgmi_write_data_acc);
using GPUMetricXgmiAccTbl_t = GpuMetricU64Tbl_t;

using GPUMetricCurrGfxClk_t = decltype(AMDGpuMetrics_v14_t::m_current_gfxclk);
using GPUMetricCurrGfxClkTbl_t = GpuMetricU16Tbl_t;

using GPUMetricCurrSocClk_t = decltype(AMDGpuMetrics_v14_t::m_current_socclk);
using GPUMetricCurrSocClkTbl_t = GpuMetricU16Tbl_t;

using GPUMetricCurrVClk0_t = decltype(AMDGpuMetrics_v14_t::m_current_vclk0);
using GPUMetricCurrVClkTbl_t = GpuMetricU16Tbl_t;

using GPUMetricCurrDClk0_t = decltype(AMDGpuMetrics_v14_t::m_current_dclk0);
using GPUMetricCurrDClkTbl_t = GpuMetricU16Tbl_t;


////
/************************************************************
  * When a new metric table is released, we have to update: *
    1.  Constants related to the new metrics added (if any);
        (ie: kRSMI_MAX_NUM_XGMI_LINKS)
    2.  Constants related to new version:
        (ie: kRSMI_GPU_METRICS_API_CONTENT_MAJOR_VER_1)
        (ie: kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_x)
        (ie: kRSMI_LATEST_GPU_METRICS_API_CONTENT_MAJOR_VER)
        (ie: kRSMI_LATEST_GPU_METRICS_API_CONTENT_MINOR_VER)
    3.  Check if still use the same existing header or if a new one is needed:
        (ie: AMDGpuMetricsHeader_v1_t)
    4.  Create a new struct representing the new table format
        (ie: AMDGpuMetrics_v13_t -> AMDGpuMetrics_v14_t)
    5.  AMGpuMetricsLatest_t -> Newest AMDGpuMetrics_v1x_t
    6.  AMDGpuMetricVersionFlags_t
        (ie: AMDGpuMetricVersionFlags_t::kGpuMetricV14)
    7.  Create the proper API using granular controls used by
        rsmi_dev_gpu_metrics_info_query() (ie: rsmi_dev_temp_hotspot_get())

    -> Remember to check/update:
      - AMDGpuMetricsUnitType_t
      - amdgpu_metrics_unit_type_translation_table
      - AMDGpuMetrics_v1X_t structure in question
      - populate_metrics_dynamic_tbl()
      - copy_internal_to_external_metrics()
      - init_max_public_gpu_matrics()
*/

using AMDGpuMetricTypeId_t = uint32_t;
using AMDGpuMetricTypeIdSeq_t = uint32_t;
using AMDGpuMetricVersionFlagId_t = uint32_t;

////
/*
  *
  * These are used as Metric class, so Metric Units can be properly grouped.
  * Each Metric Unit (or a set of them) is related to a Metric class.
  *
 */
enum class AMDGpuMetricsClassId_t : AMDGpuMetricTypeId_t {
  kGpuMetricHeader,
  kGpuMetricTemperature,
  kGpuMetricUtilization,
  kGpuMetricPowerEnergy,
  kGpuMetricAverageClock,
  kGpuMetricCurrentClock,
  kGpuMetricThrottleStatus,
  kGpuMetricGfxClkLockStatus,
  kGpuMetricCurrentFanSpeed,
  kGpuMetricLinkWidthSpeed,
  kGpuMetricVoltage,
  kGpuMetricTimestamp,
  kGpuMetricThrottleResidency,
  kGpuMetricPartition,
  kGpuMetricXcpStats,
};
using AMDGpuMetricsClassIdTranslationTbl_t = std::map<AMDGpuMetricsClassId_t, std::string>;

/*
  *
  * These are the Metric units. Each one represents a specific metric we want
  * to either store or retrieve.
  *
  * This also gives a more granular control over to what exactly is needed,
  * helping to generalize metric queries.
  *
  * Each type a new (non-existing metric unit) metric is added, it should be
  * updated here.
  *   - Their names matches (closely, regardless of their version) the name of
  *     the data structure members they represent.
  *
  * All metric units not flagged as v1.4 were either part of the base or
  * added/changed up to v1.3
 */
enum class AMDGpuMetricsUnitType_t : AMDGpuMetricTypeId_t
{
  // kGpuMetricTemperature counters
  kMetricTempEdge,
  kMetricTempHotspot,
  kMetricTempMem,
  kMetricTempVrGfx,
  kMetricTempVrSoc,
  kMetricTempVrMem,
  kMetricTempHbm,

  // kGpuMetricUtilization counters
  kMetricAvgGfxActivity,
  kMetricAvgUmcActivity,
  kMetricAvgMmActivity,
  kMetricGfxActivityAccumulator,
  kMetricMemActivityAccumulator,
  kMetricVcnActivity,                     // v1.4
  kMetricJpegActivity,                    // v1.5

  // kGpuMetricAverageClock counters
  kMetricAvgGfxClockFrequency,
  kMetricAvgSocClockFrequency,
  kMetricAvgUClockFrequency,
  kMetricAvgVClock0Frequency,
  kMetricAvgDClock0Frequency,
  kMetricAvgVClock1Frequency,
  kMetricAvgDClock1Frequency,

  // kGpuMetricCurrentClock counters
  kMetricCurrGfxClock,                    // v1.4: Changed to multi-valued
  kMetricCurrSocClock,                    // v1.4: Changed to multi-valued
  kMetricCurrUClock,
  kMetricCurrVClock0,                     // v1.4: Changed to multi-valued
  kMetricCurrDClock0,                     // v1.4: Changed to multi-valued
  kMetricCurrVClock1,
  kMetricCurrDClock1,

  // kGpuMetricThrottleStatus counters
  kMetricThrottleStatus,
  kMetricIndepThrottleStatus,

  // kGpuMetricGfxClkLockStatus counters
  kMetricGfxClkLockStatus,                // v1.4

  // kGpuMetricCurrentFanSpeed counters
  kMetricCurrFanSpeed,

  // kGpuMetricLinkWidthSpeed counters
  kMetricPcieLinkWidth,
  kMetricPcieLinkSpeed,
  kMetricPcieBandwidthAccumulator,              // v1.4
  kMetricPcieBandwidthInst,                     // v1.4
  kMetricXgmiLinkWidth,                         // v1.4
  kMetricXgmiLinkSpeed,                         // v1.4
  kMetricXgmiReadDataAccumulator,               // v1.4
  kMetricXgmiWriteDataAccumulator,              // v1.4
  kMetricPcieL0RecovCountAccumulator,           // v1.4
  kMetricPcieReplayCountAccumulator,            // v1.4
  kMetricPcieReplayRollOverCountAccumulator,    // v1.4
  kMetricPcieNakSentCountAccumulator,           // v1.5
  kMetricPcieNakReceivedCountAccumulator,       // v1.5

  // kGpuMetricPowerEnergy counters
  kMetricAvgSocketPower,
  kMetricCurrSocketPower,                 // v1.4
  kMetricEnergyAccumulator,               // v1.4

  // kGpuMetricVoltage counters
  kMetricVoltageSoc,                      // v1.3
  kMetricVoltageGfx,                      // v1.3
  kMetricVoltageMem,                      // v1.3

  // kGpuMetricTimestamp counters
  kMetricTSClockCounter,
  kMetricTSFirmware,

  // kMetricAccumulationCounter counters
  kMetricAccumulationCounter,                // v1.6
  kMetricProchotResidencyAccumulator,        // v1.6
  kMetricPPTResidencyAccumulator,            // v1.6
  kMetricSocketThmResidencyAccumulator,      // v1.6
  kMetricVRThmResidencyAccumulator,          // v1.6
  kMetricHBMThmResidencyAccumulator,         // v1.6

  // kGpuMetricPartition
  kGpuMetricNumPartition,                    // v1.6

  // kGpuMetricXcpStats
  kMetricGfxBusyInst,                        // v1.6
  kMetricJpegBusy,                           // v1.6
  kMetricVcnBusy,                            // v1.6
  kMetricGfxBusyAcc,                         // v1.6
  kMetricGfxBelowHostLimitAccumulator,       // v1.7

  kMetricPcieLCPerfOtherEndRecov,            // v1.6

  kMetricVramMaxBandwidth,                   // v1.7
  kMetricXgmiLinkStatus,                     // v1.7

  kMetricGfxBelowHostLimitPptAcc,            // v1.8
  kMetricGfxBelowHostLimitThmAcc,            // v1.8
  kMetricGfxBelowHostLimitTotalAcc,          // v1.8
  kMetricGfxLowUtilitizationAcc,             // v1.8

};
using AMDGpuMetricsUnitTypeTranslationTbl_t = std::map<AMDGpuMetricsUnitType_t, std::string>;

using AMDGpuMetricsDataTypeId_t = uint8_t;
enum class AMDGpuMetricsDataType_t : AMDGpuMetricsDataTypeId_t
{
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
};

struct AMDGpuDynamicMetricsValue_t {
  uint64_t m_value;
  std::string m_info;
  AMDGpuMetricsDataType_t m_original_type;
};
using AMDGpuDynamicMetricTblValues_t = std::vector<AMDGpuDynamicMetricsValue_t>;
using AMDGpuDynamicMetricsTbl_t = std::map<AMDGpuMetricsClassId_t,
  std::map<AMDGpuMetricsUnitType_t, AMDGpuDynamicMetricTblValues_t>>;


/*
  *
  * Note: All supported metric versions are listed here, otherwise unsupported
  *
 */
enum class AMDGpuMetricVersionFlags_t : AMDGpuMetricVersionFlagId_t
{
  kGpuMetricNone = 0x0,
  kGpuMetricV10 = (0x1 << 0),
  kGpuMetricV11 = (0x1 << 1),
  kGpuMetricV12 = (0x1 << 2),
  kGpuMetricV13 = (0x1 << 3),
  kGpuMetricV14 = (0x1 << 4),
  kGpuMetricV15 = (0x1 << 5),
  kGpuMetricV16 = (0x1 << 6),
  kGpuMetricV17 = (0x1 << 7),
  kGpuMetricV18 = (0x1 << 8),  // Added new version flag
};
using AMDGpuMetricVersionTranslationTbl_t = std::map<uint16_t, AMDGpuMetricVersionFlags_t>;
using GpuMetricTypePtr_t = std::shared_ptr<void>;

class GpuMetricsBase_t {
 public:
    virtual ~GpuMetricsBase_t() = default;
    virtual size_t sizeof_metric_table() = 0;
    virtual GpuMetricTypePtr_t get_metrics_table() = 0;
    virtual AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() = 0;
    virtual rsmi_status_t populate_metrics_dynamic_tbl() = 0;
    virtual AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() = 0;
    virtual void set_device_id(uint32_t device_id) { m_device_id = device_id; }
    virtual void set_partition_id(uint32_t partition_id) { m_partition_id = partition_id; }
    virtual AMDGpuDynamicMetricsTbl_t get_metrics_dynamic_tbl() {
      return m_base_metrics_dynamic_tbl;
    }

 protected:
    AMDGpuDynamicMetricsTbl_t m_base_metrics_dynamic_tbl;
    uint64_t m_metrics_timestamp;
    uint32_t m_device_id;
    uint32_t m_partition_id;
};
using GpuMetricsBasePtr = std::shared_ptr<GpuMetricsBase_t>;
using AMDGpuMetricFactories_t = const std::map<AMDGpuMetricVersionFlags_t, GpuMetricsBasePtr>;

class GpuMetricsBase_v11_t final : public GpuMetricsBase_t {
 public:
    virtual ~GpuMetricsBase_v11_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v11_t);
    }

    GpuMetricTypePtr_t get_metrics_table() override {
      if (!m_gpu_metric_ptr) {
        m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v11_t*){});
      }
      assert(m_gpu_metric_ptr != nullptr);
      return m_gpu_metric_ptr;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV11;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;
    AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;


 private:
    AMDGpuMetrics_v11_t m_gpu_metrics_tbl;
    std::shared_ptr<AMDGpuMetrics_v11_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v12_t final : public GpuMetricsBase_t {
 public:
    ~GpuMetricsBase_v12_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v12_t);
    }

    GpuMetricTypePtr_t get_metrics_table() override {
      if (!m_gpu_metric_ptr) {
        m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v12_t*){});
      }
      assert(m_gpu_metric_ptr != nullptr);
      return m_gpu_metric_ptr;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV12;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;
    AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;

 private:
    AMDGpuMetrics_v12_t m_gpu_metrics_tbl;
    std::shared_ptr<AMDGpuMetrics_v12_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v13_t final : public GpuMetricsBase_t {
 public:
    ~GpuMetricsBase_v13_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v13_t);
    }

    GpuMetricTypePtr_t get_metrics_table() override {
      if (!m_gpu_metric_ptr) {
        m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v13_t*){});
      }
      assert(m_gpu_metric_ptr != nullptr);
      return (m_gpu_metric_ptr);
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV13;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;
    AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;


 private:
    AMDGpuMetrics_v13_t m_gpu_metrics_tbl;
    std::shared_ptr<AMDGpuMetrics_v13_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v14_t final : public GpuMetricsBase_t {
 public:
    ~GpuMetricsBase_v14_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v14_t);
    }

    GpuMetricTypePtr_t get_metrics_table() override {
      if (!m_gpu_metric_ptr) {
        m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v14_t*){});
      }
      assert(m_gpu_metric_ptr != nullptr);
      return m_gpu_metric_ptr;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV14;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;
    AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;


 private:
    AMDGpuMetrics_v14_t m_gpu_metrics_tbl;
    std::shared_ptr<AMDGpuMetrics_v14_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v15_t final : public GpuMetricsBase_t {
 public:
    ~GpuMetricsBase_v15_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v15_t);
    }

    GpuMetricTypePtr_t get_metrics_table() override {
      if (!m_gpu_metric_ptr) {
        m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v15_t*){});
      }
      assert(m_gpu_metric_ptr != nullptr);
      return m_gpu_metric_ptr;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV15;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;
    AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;


 private:
    AMDGpuMetrics_v15_t m_gpu_metrics_tbl;
    std::shared_ptr<AMDGpuMetrics_v15_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v16_t final : public GpuMetricsBase_t {
 public:
  ~GpuMetricsBase_v16_t() = default;

  size_t sizeof_metric_table() override {
    return sizeof(AMDGpuMetrics_v16_t);
  }

  GpuMetricTypePtr_t get_metrics_table() override {
    if (!m_gpu_metric_ptr) {
      m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v16_t*){});
    }
    assert(m_gpu_metric_ptr != nullptr);
    return m_gpu_metric_ptr;
  }

  AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
    return AMDGpuMetricVersionFlags_t::kGpuMetricV16;
  }

  rsmi_status_t populate_metrics_dynamic_tbl() override;
  AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;

 private:
  AMDGpuMetrics_v16_t m_gpu_metrics_tbl;
  std::shared_ptr<AMDGpuMetrics_v16_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v17_t final : public GpuMetricsBase_t {
 public:
  ~GpuMetricsBase_v17_t() = default;

  size_t sizeof_metric_table() override {
    return sizeof(AMDGpuMetrics_v17_t);
  }

  GpuMetricTypePtr_t get_metrics_table() override {
    if (!m_gpu_metric_ptr) {
      m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v17_t*){});
    }
    assert(m_gpu_metric_ptr != nullptr);
    return m_gpu_metric_ptr;
  }

  AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
    return AMDGpuMetricVersionFlags_t::kGpuMetricV17;
  }

  rsmi_status_t populate_metrics_dynamic_tbl() override;
  AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;

 private:
  AMDGpuMetrics_v17_t m_gpu_metrics_tbl;
  std::shared_ptr<AMDGpuMetrics_v17_t> m_gpu_metric_ptr;
};

class GpuMetricsBase_v18_t final : public GpuMetricsBase_t {
 public:
  ~GpuMetricsBase_v18_t() = default;

  size_t sizeof_metric_table() override {
    return sizeof(AMDGpuMetrics_v18_t);
  }

  GpuMetricTypePtr_t get_metrics_table() override {
    if (!m_gpu_metric_ptr) {
      m_gpu_metric_ptr.reset(&m_gpu_metrics_tbl, [](AMDGpuMetrics_v18_t*){});
    }
    assert(m_gpu_metric_ptr != nullptr);
    return m_gpu_metric_ptr;
  }

  AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override {
    return AMDGpuMetricVersionFlags_t::kGpuMetricV18;
  }

  rsmi_status_t populate_metrics_dynamic_tbl() override;
  AMGpuMetricsPublicLatestTupl_t copy_internal_to_external_metrics() override;

 private:
  AMDGpuMetrics_v18_t m_gpu_metrics_tbl;
  std::shared_ptr<AMDGpuMetrics_v18_t> m_gpu_metric_ptr;
};

template<typename T>
rsmi_status_t rsmi_dev_gpu_metrics_info_query(uint32_t dv_ind,
                        AMDGpuMetricsUnitType_t metric_counter, T& metric_value);

}  // namespace amd::smi


rsmi_status_t
rsmi_dev_gpu_metrics_header_info_get(uint32_t dv_ind,
                                          metrics_table_header_t& header_value);


#endif  // ROCM_SMI_ROCM_SMI_GPU_METRICS_H_
