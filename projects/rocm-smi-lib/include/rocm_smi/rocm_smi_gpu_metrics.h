/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
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

#ifndef ROCM_SMI_ROCM_SMI_GPU_METRICS_H_
#define ROCM_SMI_ROCM_SMI_GPU_METRICS_H_

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>


/**
 *  All 1.4 and newer GPU metrics are now defined in this header.
 *
 */
namespace amd::smi
{

constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MAJOR_VER_1 = 1;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_1 = 1;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_2 = 2;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_3 = 3;
constexpr uint32_t kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_4 = 4;
constexpr uint32_t kRSMI_LATEST_GPU_METRICS_API_CONTENT_MAJOR_VER = kRSMI_GPU_METRICS_API_CONTENT_MAJOR_VER_1;
constexpr uint32_t kRSMI_LATEST_GPU_METRICS_API_CONTENT_MINON_VER = kRSMI_GPU_METRICS_API_CONTENT_MINOR_VER_4;


//  Note: As gpu metrics are updating
constexpr uint32_t kRSMI_GPU_METRICS_EXPIRATION_SECS = 5;

//  Note: This *must* match NUM_HBM_INSTANCES
constexpr uint32_t kRSMI_MAX_NUM_HBM_INSTANCES = 4;

//  Note: This *must* match NUM_XGMI_LINKS
constexpr uint32_t kRSMI_MAX_NUM_XGMI_LINKS = 8;

//  Note: This *must* match MAX_GFX_CLKS
constexpr uint32_t kRSMI_MAX_NUM_GFX_CLKS = 8;

//  Note: This *must* match MAX_CLKS
constexpr uint32_t kRSMI_MAX_NUM_CLKS = 4;

//  Note: This *must* match NUM_VCN
constexpr uint32_t kRSMI_MAX_NUM_VCN = 4;


struct AMDGpuMetricsHeader_v1_t
{
  uint16_t m_structure_size;
  uint8_t  m_format_revision;
  uint8_t  m_content_revision;
};


struct AMDGpuMetricsBase_t;
using AMDGpuMetricsBaseRef = AMDGpuMetricsBase_t&;
struct AMDGpuMetricsBase_t
{
    virtual ~AMDGpuMetricsBase_t() = default;
};

struct AMDGpuMetrics_v11_t : AMDGpuMetricsBase_t
{
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

struct AMDGpuMetrics_v12_t : AMDGpuMetricsBase_t
{
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

struct AMDGpuMetrics_v13_t : AMDGpuMetricsBase_t
{
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

struct AMDGpuMetrics_v14_t : AMDGpuMetricsBase_t
{
  ~AMDGpuMetrics_v14_t() = default;

  struct AMDGpuMetricsHeader_v1_t m_common_header;

  // Temperature (Celsius). It will be zero (0) if unsupported.
  uint16_t m_temperature_hotspot;
  uint16_t m_temperature_mem;
  uint16_t m_temperature_vrsoc;

  // Power (Watts)
  uint16_t m_curr_socket_power;

  // Utilization (%)
  uint16_t m_average_gfx_activity;
  uint16_t m_average_umc_activity;            // memory controller
  uint16_t m_vcn_activity[kRSMI_MAX_NUM_VCN]; // VCN instances activity percent (encode/decode)

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
  uint16_t m_pcie_link_speed; // in 0.1 GT/s

	// XGMI bus width and bitrate (in Gbps)
  uint16_t  m_xgmi_link_width;
  uint16_t  m_xgmi_link_speed;

  // Utilization Accumulated (%)
  uint32_t m_gfx_activity_acc;
  uint32_t m_mem_activity_acc;

  // PCIE accumulated bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_acc;

	// PCIE instantaneous bandwidth (GB/sec)
  uint64_t m_pcie_bandwidth_inst;

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
using AMGpuMetricsLatest_t = AMDGpuMetrics_v14_t;


using GPUMetricTempHbm_t = decltype(AMDGpuMetrics_v13_t::m_temperature_hbm);
using GPUMetricTempHbmTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_HBM_INSTANCES>;

using GPUMetricVcnActivity_t = decltype(AMDGpuMetrics_v14_t::m_vcn_activity);
using GPUMetricVcnActivityTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_VCN>;

using GPUMetricXgmiReadDataAcc_t = decltype(AMDGpuMetrics_v14_t::m_xgmi_read_data_acc);
using GPUMetricXgmiWriteDataAcc_t = decltype(AMDGpuMetrics_v14_t::m_xgmi_write_data_acc);
using GPUMetricXgmiAccTbl_t = std::array<uint64_t, kRSMI_MAX_NUM_XGMI_LINKS>;

using GPUMetricCurrGfxClk_t = decltype(AMDGpuMetrics_v14_t::m_current_gfxclk);
using GPUMetricCurrGfxClkTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_GFX_CLKS>;

using GPUMetricCurrSocClk_t = decltype(AMDGpuMetrics_v14_t::m_current_socclk);
using GPUMetricCurrSocClkTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_CLKS>;

using GPUMetricCurrVClk0_t = decltype(AMDGpuMetrics_v14_t::m_current_vclk0);
using GPUMetricCurrVClkTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_CLKS>;

using GPUMetricCurrDClk0_t = decltype(AMDGpuMetrics_v14_t::m_current_dclk0);
using GPUMetricCurrDClkTbl_t = std::array<uint16_t, kRSMI_MAX_NUM_CLKS>;


/*
  * When a new metric table is released, we have to update: *
    1.  Constants related to the new metrics added;
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
*/

using AMDGpuMetricTypeId_t = uint32_t;
using AMDGpuMetricTypeIdSeq_t = uint32_t;
using AMDGpuMetricVersionFlagId_t = uint32_t;

enum class AMDGpuMetricsClassId_t : AMDGpuMetricTypeId_t
{
  kGpuMetricHeader = 0,
  kGpuMetricTemperature,
  kGpuMetricUtilization,
  kGpuMetricPowerEnergy,
  kGpuMetricSystemClockCounter,
  kGpuMetricAverageClock,
  kGpuMetricCurrentClock,
  kGpuMetricThrottleStatus,
  kGpuMetricGfxClkLockStatus,
  kGpuMetricCurrentFanSpeed,
  kGpuMetricLinkWidthSpeed,
  kGpuMetricVoltage,
  kGpuMetricTimestamp,
};
using AMDGpuMetricsClassIdTranslationTbl_t = std::map<AMDGpuMetricsClassId_t, std::string>;

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
  kMetricVcnActivity,

  // kGpuMetricAverageClock counters
  kMetricAvgGfxClockFrequency,
  kMetricAvgSocClockFrequency,
  kMetricAvgUClockFrequency,
  kMetricAvgVClock0Frequency,
  kMetricAvgDClock0Frequency,
  kMetricAvgVClock1Frequency,
  kMetricAvgDClock1Frequency,

  // kGpuMetricCurrentClock counters
  kMetricCurrGfxClock,
  kMetricCurrSocClock,
  kMetricCurrUClock,
  kMetricCurrVClock0,
  kMetricCurrDClock0,
  kMetricCurrVClock1,
  kMetricCurrDClock1,

  // kGpuMetricThrottleStatus counters
  kMetricThrottleStatus,
  kMetricIndepThrottleStatus,

  // kGpuMetricGfxClkLockStatus counters
  kMetricGfxClkLockStatus,

  // kGpuMetricCurrentFanSpeed counters
  kMetricCurrFanSpeed,

  // kGpuMetricLinkWidthSpeed counters
  kMetricPcieLinkWidth,
  kMetricPcieLinkSpeed,
  kMetricPcieBandwidthAccumulator,
  kMetricPcieBandwidthInst,
  kMetricXgmiLinkWidth,
  kMetricXgmiLinkSpeed,
  kMetricXgmiReadDataAccumulator,
  kMetricXgmiWriteDataAccumulator,

  // kGpuMetricPowerEnergy counters
  kMetricAvgSocketPower,
  kMetricCurrSocketPower,
  kMetricEnergyAccumulator,

  // kGpuMetricVoltage counters
  kMetricVoltageSoc,
  kMetricVoltageGfx,
  kMetricVoltageMem,

  // kGpuMetricTimestamp counters
  kMetricTSClockCounter,
  kMetricTSFirmware,
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

struct AMDGpuDynamicMetricsValue_t
{
  uint64_t m_value;
  std::string m_info;
  AMDGpuMetricsDataType_t m_original_type;
};
using AMDGpuDynamicMetricTblValues_t = std::vector<AMDGpuDynamicMetricsValue_t>;
using AMDGpuDynamicMetricsTbl_t = std::map<AMDGpuMetricsClassId_t, std::map<AMDGpuMetricsUnitType_t, AMDGpuDynamicMetricTblValues_t>>;

//  Note: All supported metric versions are listed her
//        If not here, they are not supported
enum class AMDGpuMetricVersionFlags_t : AMDGpuMetricVersionFlagId_t
{
  kGpuMetricNone = 0x0,
  kGpuMetricV10 = (0x1 << 0),
  kGpuMetricV11 = (0x1 << 1),
  kGpuMetricV12 = (0x1 << 2),
  kGpuMetricV13 = (0x1 << 3),
  kGpuMetricV14 = (0x1 << 4),
};
using AMDGpuMetricVersionTranslationTbl_t = std::map<uint64_t, AMDGpuMetricVersionFlags_t>;


class GpuMetricsBase_t;
using GpuMetricsBasePtr = std::shared_ptr<GpuMetricsBase_t>;

class GpuMetricsBase_t
{
  public:
    virtual ~GpuMetricsBase_t() = default;
    virtual size_t sizeof_metric_table() = 0;
    virtual AMDGpuMetricsBaseRef get_metrics_table() = 0;
    virtual AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() = 0;
    virtual rsmi_status_t populate_metrics_dynamic_tbl() = 0;

    virtual AMDGpuDynamicMetricsTbl_t get_metrics_dynamic_tbl() {
      return m_metrics_dynamic_tbl;
    }

  protected:
    AMDGpuDynamicMetricsTbl_t m_metrics_dynamic_tbl;
    uint64_t m_metrics_timestamp;

};
using AMDGpuMetricFactories_t = std::map<AMDGpuMetricVersionFlags_t, GpuMetricsBasePtr>;


class GpuMetricsBase_v11_t final : public GpuMetricsBase_t
{
  public:
    ~GpuMetricsBase_v11_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v11_t);
    }

    AMDGpuMetricsBaseRef get_metrics_table() override
    {
      return m_gpu_metrics_tbl;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override
    {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV11;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;


  private:
    AMDGpuMetrics_v11_t m_gpu_metrics_tbl;

};

class GpuMetricsBase_v12_t final : public GpuMetricsBase_t
{
  public:
    ~GpuMetricsBase_v12_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v12_t);
    }

    AMDGpuMetricsBaseRef get_metrics_table() override
    {
      return m_gpu_metrics_tbl;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override
    {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV12;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;


  private:
    AMDGpuMetrics_v12_t m_gpu_metrics_tbl;

};

class GpuMetricsBase_v13_t final : public GpuMetricsBase_t
{
  public:
    ~GpuMetricsBase_v13_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v13_t);
    }

    AMDGpuMetricsBaseRef get_metrics_table() override
    {
      return m_gpu_metrics_tbl;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override
    {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV13;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;


  private:
    AMDGpuMetrics_v13_t m_gpu_metrics_tbl;

};

class GpuMetricsBase_v14_t final : public GpuMetricsBase_t
{
  public:
    ~GpuMetricsBase_v14_t() = default;

    size_t sizeof_metric_table() override {
      return sizeof(AMDGpuMetrics_v14_t);
    }

    AMDGpuMetricsBaseRef get_metrics_table() override
    {
      return m_gpu_metrics_tbl;
    }

    AMDGpuMetricVersionFlags_t get_gpu_metrics_version_used() override
    {
      return AMDGpuMetricVersionFlags_t::kGpuMetricV14;
    }

    rsmi_status_t populate_metrics_dynamic_tbl() override;


  private:
    AMDGpuMetrics_v14_t m_gpu_metrics_tbl;

};

template<typename T>
rsmi_status_t rsmi_dev_gpu_metrics_info_query(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, T& metric_value);

} // namespace amd::smi

#endif // ROCM_SMI_ROCM_SMI_GPU_METRICS_H_

