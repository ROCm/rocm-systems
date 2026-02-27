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

#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "rocm_smi/rocm_smi_dyn_gpu_metrics.h" // Dynamic metrics
#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_logger.h"

#include <dirent.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <map>
#include <regex>  // NOLINT
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cstddef>
#include <variant>

using namespace amd::smi;

#define TRY try {
#define CATCH } catch (...) {return amd::smi::handleException();}


namespace amd::smi
{

std::mutex GpuMetricsBase_t::s_base_tbl_mu;

constexpr uint16_t join_metrics_version(uint8_t format_rev, uint8_t content_rev)
{
  return static_cast<uint16_t>((format_rev << 8 | content_rev));
}

constexpr uint16_t join_metrics_version(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  return join_metrics_version(metrics_header.m_format_revision, metrics_header.m_content_revision);
}

AMDGpuMetricsHeader_v1_t disjoin_metrics_version(uint16_t version)
{
  AMDGpuMetricsHeader_v1_t metrics_header;

  metrics_header.m_format_revision  = static_cast<uint8_t>((version & 0xFF00) >> 8);
  metrics_header.m_content_revision = static_cast<uint8_t>(version & 0x00FF);

  return metrics_header;
}

uint64_t actual_timestamp_in_secs()
{
  using namespace std::chrono;
  return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

auto timestamp_to_time_point(uint64_t timestamp_in_secs)
{
  using namespace std::chrono;
  system_clock::time_point time_point{seconds{timestamp_in_secs}};
  std::time_t timestamp_time = system_clock::to_time_t(time_point);
  return timestamp_time;
}

// header_details:
// - bool: true if header contains partition metrics, false otherwise
// - std::string: file path of the metrics header file
std::string stringfy_metrics_header(const AMDGpuMetricsHeader_v1_t& metrics_header,
                                    bool is_partition_metrics,
                                    const std::string& file_path) {
  std::stringstream metrics_header_info;
  metrics_header_info
    << "{Header Info: "
    << print_unsigned_int(metrics_header.m_format_revision)
    << "."
    << print_unsigned_int(metrics_header.m_content_revision)
    << " Size: "
    << print_unsigned_int(metrics_header.m_structure_size)
    << "}  "
    << "[Format: " << print_unsigned_hex_and_int(metrics_header.m_format_revision)
    << " Revision: " << print_unsigned_hex_and_int(metrics_header.m_content_revision)
    << " Size: " << print_unsigned_hex_and_int(metrics_header.m_structure_size)
    << "]"
    << " | Is Partition Metrics: " << std::boolalpha << is_partition_metrics
    << " | Metric File: " << file_path
    << "\n";

  return metrics_header_info.str();
}

std::string stringfy_metric_header_version(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  std::stringstream metrics_header_info;
  metrics_header_info
    << print_unsigned_int(metrics_header.m_format_revision)
    << "."
    << print_unsigned_int(metrics_header.m_content_revision);

  return metrics_header_info.str();
}


//
//  version 1.0: 256
//  version 1.1: 257
//  version 1.2: 258
//  version 1.3: 259
//  version 1.4: 260
//  version 1.5: 261
//  version 1.6: 262
//  version 1.7: 263
//  version 1.8: 264
//  version 1.9: 265
const AMDGpuMetricVersionTranslationTbl_t amdgpu_metric_version_translation_table {
  {join_metrics_version(1, 0), AMDGpuMetricVersionFlags_t::kGpuMetricV10},
  {join_metrics_version(1, 1), AMDGpuMetricVersionFlags_t::kGpuMetricV11},
  {join_metrics_version(1, 2), AMDGpuMetricVersionFlags_t::kGpuMetricV12},
  {join_metrics_version(1, 3), AMDGpuMetricVersionFlags_t::kGpuMetricV13},
  {join_metrics_version(1, 4), AMDGpuMetricVersionFlags_t::kGpuMetricV14},
  {join_metrics_version(1, 5), AMDGpuMetricVersionFlags_t::kGpuMetricV15},
  {join_metrics_version(1, 6), AMDGpuMetricVersionFlags_t::kGpuMetricV16},
  {join_metrics_version(1, 7), AMDGpuMetricVersionFlags_t::kGpuMetricV17},
  {join_metrics_version(1, 8), AMDGpuMetricVersionFlags_t::kGpuMetricV18},
  {join_metrics_version(1, 9), AMDGpuMetricVersionFlags_t::kGpuMetricDynV19Plus},  // Dynamic GPU Metrics
};

//  version 1.0: 256
//  version 1.1: 257
const AMDGpuMetricVersionTranslationTbl_t amdgpu_partition_metric_version_translation_table {
  {join_metrics_version(1, 0), AMDGpuMetricVersionFlags_t::kGpuXcpMetricV10},
  {join_metrics_version(1, 1), AMDGpuMetricVersionFlags_t::kGpuXcpMetricDynV11Plus},  // Dynamic XCP Metrics
};

/**
 *
*/
const AMDGpuMetricsClassIdTranslationTbl_t amdgpu_metrics_class_id_translation_table
{
  {AMDGpuMetricsClassId_t::kGpuMetricHeader, "Header"},
  {AMDGpuMetricsClassId_t::kGpuMetricTemperature, "Temperature"},
  {AMDGpuMetricsClassId_t::kGpuMetricUtilization, "Utilization"},
  {AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy, "Power/Energy"},
  {AMDGpuMetricsClassId_t::kGpuMetricAverageClock, "Average Clock"},
  {AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, "Current Clock"},
  {AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus, "Throttle"},
  {AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus, "Gfx Clock Lock"},
  {AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed, "Current Fan Speed"},
  {AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, "Link/Bandwidth/Speed"},
  {AMDGpuMetricsClassId_t::kGpuMetricVoltage, "Voltage"},
  {AMDGpuMetricsClassId_t::kGpuMetricTimestamp, "Timestamp"},
  {AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, "Throttler Residency"},
  {AMDGpuMetricsClassId_t::kGpuMetricPartition, "Partition Number"},
  {AMDGpuMetricsClassId_t::kGpuMetricXcpStats, "XCP Stats"},
};

const AMDGpuMetricsUnitTypeTranslationTbl_t amdgpu_metrics_unit_type_translation_table {
  // kGpuMetricTemperature counters
  {AMDGpuMetricsUnitType_t::kMetricTempEdge, "TempEdge"},
  {AMDGpuMetricsUnitType_t::kMetricTempHotspot, "TempHotspot"},
  {AMDGpuMetricsUnitType_t::kMetricTempMem, "TempMem"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrGfx, "TempVrGfx"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrSoc, "TempVrSoc"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrMem, "TempVrMem"},
  {AMDGpuMetricsUnitType_t::kMetricTempHbm, "TempHbm"},

  // kGpuMetricUtilization counters
  {AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity, "AvgGfxActivity"},
  {AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity, "AvgUmcActivity"},
  {AMDGpuMetricsUnitType_t::kMetricAvgMmActivity, "AvgMmActivity"},
  {AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator, "GfxActivityAcc"},
  {AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator, "MemActivityAcc"},
  {AMDGpuMetricsUnitType_t::kMetricVcnActivity, "VcnActivity"},     /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricJpegActivity, "JpegActivity"},   /* v1.5 */

  // kGpuMetricAverageClock counters
  {AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency, "AvgGfxClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency, "AvgSocClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency, "AvgUClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency, "AvgVClock0Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency, "AvgDClock0Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency, "AvgVClock1Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency, "AvgDClock1Frequency"},

  // kGpuMetricCurrentClock counters
  {AMDGpuMetricsUnitType_t::kMetricCurrGfxClock, "CurrGfxClock"},     /* v1.4: Changed to array */
  {AMDGpuMetricsUnitType_t::kMetricCurrSocClock, "CurrSocClock"},     /* v1.4: Changed to array */
  {AMDGpuMetricsUnitType_t::kMetricCurrUClock, "CurrUClock"},
  {AMDGpuMetricsUnitType_t::kMetricCurrVClock0, "CurrVClock0"},       /* v1.4: Changed to array */
  {AMDGpuMetricsUnitType_t::kMetricCurrDClock0, "CurrDClock0"},       /* v1.4: Changed to array */
  {AMDGpuMetricsUnitType_t::kMetricCurrVClock1, "CurrVClock1"},
  {AMDGpuMetricsUnitType_t::kMetricCurrDClock1, "CurrDClock1"},

  // kGpuMetricThrottleStatus counters
  {AMDGpuMetricsUnitType_t::kMetricThrottleStatus, "ThrottleStatus"},
  {AMDGpuMetricsUnitType_t::kMetricIndepThrottleStatus, "IndepThrottleStatus"},

  // kGpuMetricGfxClkLockStatus counters
  {AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus, "GfxClkLockStatus"},     /* v1.4 */

  // kGpuMetricCurrentFanSpeed counters
  {AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed, "CurrFanSpeed"},

  // kGpuMetricLinkWidthSpeed counters
  {AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth, "PcieLinkWidth"},
  {AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed, "PcieLinkSpeed"},
  {AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator, "PcieBandwidthAcc"},     /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst, "PcieBandwidthInst"},           /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth, "XgmiLinkWidth"},                   /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed, "XgmiLinkSpeed"},                   /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator, "XgmiReadDataAcc"},       /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator, "XgmiWriteDataAcc"},     /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator, "PcieL0RecovCountAcc"},                 /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator, "PcieReplayCountAcc"},                   /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator, "PcieReplayRollOverCountAcc"},   /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator, "PcieNakSentCountAcc"},                 /* v1.5 */
  {AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator, "PcieNakRcvdCountAcc"},             /* v1.5 */

  // kGpuMetricPowerEnergy counters
  {AMDGpuMetricsUnitType_t::kMetricAvgSocketPower, "AvgSocketPower"},
  {AMDGpuMetricsUnitType_t::kMetricCurrSocketPower, "CurrSocketPower"},     /* v1.4 */
  {AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator, "EnergyAcc"},

  // kGpuMetricVoltage counters
  {AMDGpuMetricsUnitType_t::kMetricVoltageSoc, "VoltageSoc"},
  {AMDGpuMetricsUnitType_t::kMetricVoltageGfx, "VoltageGfx"},
  {AMDGpuMetricsUnitType_t::kMetricVoltageMem, "VoltageMem"},

  // kGpuMetricTimestamp counters
  {AMDGpuMetricsUnitType_t::kMetricTSClockCounter, "TSClockCounter"},
  {AMDGpuMetricsUnitType_t::kMetricTSFirmware, "TSFirmware"},

  // kGpuMetricThrottleResidency counters
  {AMDGpuMetricsUnitType_t::kMetricAccumulationCounter, "AccumulationCounter"},                     /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricProchotResidencyAccumulator, "ProchotResidencyAccumulator"},     /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricPPTResidencyAccumulator, "PPTResidencyAccumulator"},             /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricSocketThmResidencyAccumulator, "SocketThmResidencyAccumulator"}, /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricVRThmResidencyAccumulator, "VRThmResidencyAccumulator"},         /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricHBMThmResidencyAccumulator, "HBMThmResidencyAccumulator"},       /* v1.6 */

  // kGpuMetricPartition
  {AMDGpuMetricsUnitType_t::kGpuMetricNumPartition, "numPartition"},                                /* v1.6 */

  // kGpuMetricXcpStats
  {AMDGpuMetricsUnitType_t::kMetricGfxBusyInst, "GfxBusyInst"},                                     /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricJpegBusy, "JpegBusy"},                                           /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricVcnBusy, "VcnBusy"},                                             /* v1.6 */
  {AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc, "GfxBusyAcc"},                                       /* v1.6 */

  // kGpuMetricLinkWidthSpeed
  {AMDGpuMetricsUnitType_t::kMetricPcieLCPerfOtherEndRecov, "PcieLCPerfOtherEndRecov"},             /* v1.6 */


  {AMDGpuMetricsUnitType_t::kMetricXgmiLinkStatus, "XgmiLinkStatus"},                               /* v1.7 */
  {AMDGpuMetricsUnitType_t::kMetricVramMaxBandwidth, "VramMaxBandwidth"},                           /* v1.7 */
  {AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitAccumulator,"GfxBelowHostLimitAccumulator"},    /* v1.7 */

  // kGpuMetricXcpStats v1.8
  {AMDGpuMetricsUnitType_t::kMetricGfxLowUtilitizationAcc, "GfxLowUtilitizationAcc"},               /* v1.8 */
  {AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitTotalAcc, "GfxBelowHostLimitTotalAcc"},         /* v1.8 */
  {AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitPptAcc, "GfxBelowHostLimitPptAcc"},             /* v1.8 */
  {AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitThmAcc, "GfxBelowHostLimitThmAcc"},             /* v1.8 */
};


AMDGpuMetricVersionFlags_t translate_header_to_flag_version(
                              const AMDGpuMetricsHeader_v1_t& metrics_header,
                              bool is_partition_metrics,
                              const std::string& file_path) {
  std::ostringstream ss;
  auto version_id(AMDGpuMetricVersionFlags_t::kGpuMetricNone);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  const auto flag_version = join_metrics_version(metrics_header);
  if (!is_partition_metrics) {
    if (auto it = amdgpu_metric_version_translation_table.find(flag_version);
        it != amdgpu_metric_version_translation_table.end()) {
      return it->second;
    }
    if (metrics_header.m_format_revision == 1 &&
        metrics_header.m_content_revision >= 9) {
      return AMDGpuMetricVersionFlags_t::kGpuMetricDynV19Plus;
    }
  } else {
    if (auto it = amdgpu_partition_metric_version_translation_table.find(flag_version);
        it != amdgpu_partition_metric_version_translation_table.end()) {
      return it->second;
    }
    if (metrics_header.m_format_revision == 1 &&
        metrics_header.m_content_revision >= 2) {
      return AMDGpuMetricVersionFlags_t::kGpuXcpMetricDynV11Plus;
    }
  }

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Fail "
              << " | Translation Tbl: " << flag_version
              << " | Metric Version: " << stringfy_metrics_header(metrics_header, is_partition_metrics, file_path)
              << " | Returning = "
              << static_cast<AMDGpuMetricVersionFlagId_t>(version_id)
              << " |";
  LOG_ERROR(ss);
  return version_id;
}

uint16_t translate_flag_to_metric_version(AMDGpuMetricVersionFlags_t version_flag)
{
  std::ostringstream ss;
  auto version_id = uint16_t(0);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  for (const auto& [key, value] : amdgpu_metric_version_translation_table) {
      if (value == version_flag) {
        version_id = key;
        ss << __PRETTY_FUNCTION__
                   << " | ======= end ======= "
                   << " | Success "
                   << " | Version Flag: " << static_cast<AMDGpuMetricVersionFlagId_t>(version_flag)
                   << " | Unified Version: " << version_id
                   << " | Str. Version: " << stringfy_metric_header_version(disjoin_metrics_version(version_id))
                   << " |";
        LOG_TRACE(ss);
        return version_id;
      }
  }

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Fail "
              << " | Version Flag: " << static_cast<AMDGpuMetricVersionFlagId_t>(version_flag)
              << " | Unified Version: " << version_id
              << " | Str. Version: " << stringfy_metric_header_version(disjoin_metrics_version(version_id))
              << " |";
  LOG_TRACE(ss);
  return version_id;
}

// metric_details:
// - bool: true if header contains partition metrics, false otherwise
// - std::string: file path of the metrics header file
rsmi_status_t is_gpu_metrics_version_supported(
                const AMDGpuMetricsHeader_v1_t& metrics_header,
                bool is_partition_metrics) {
  (void)is_partition_metrics;//unused
  const auto flag_version = join_metrics_version(metrics_header);
  if (flag_version == static_cast<uint16_t>(
                        AMDGpuMetricVersionFlags_t::kGpuMetricNone)) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  return RSMI_STATUS_SUCCESS;
}

GpuMetricsBasePtr amdgpu_metrics_factory(AMDGpuMetricVersionFlags_t v,
                                         bool is_partition_metrics,
                                         const std::string& file_path) {

  (void)(file_path);//unused
  if (!is_partition_metrics) {
    switch (v) {
      case AMDGpuMetricVersionFlags_t::kGpuMetricV10: return std::make_shared<GpuMetricsBase_v10_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV11: return std::make_shared<GpuMetricsBase_v11_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV12: return std::make_shared<GpuMetricsBase_v12_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV13: return std::make_shared<GpuMetricsBase_v13_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV14: return std::make_shared<GpuMetricsBase_v14_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV15: return std::make_shared<GpuMetricsBase_v15_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV16: return std::make_shared<GpuMetricsBase_v16_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV17: return std::make_shared<GpuMetricsBase_v17_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricV18: return std::make_shared<GpuMetricsBase_v18_t>();
      case AMDGpuMetricVersionFlags_t::kGpuMetricDynV19Plus: return std::make_shared<GpuMetricsBaseDynamic_t>();
      default: return nullptr;
    }
  } else {
    switch (v) {
      case AMDGpuMetricVersionFlags_t::kGpuXcpMetricV10: return std::make_shared<GpuMetricsBase_v18_t>();
      case AMDGpuMetricVersionFlags_t::kGpuXcpMetricDynV11Plus: return std::make_shared<GpuMetricsBaseDynamic_t>();
      default: return nullptr;
    }
  }
}

template<typename>
constexpr bool is_dependent_false_v = false;

template<typename T>
constexpr T init_max_uint_types()
{
  if constexpr ((std::is_same_v<T, std::uint8_t>)  ||
                (std::is_same_v<T, std::uint16_t>) ||
                (std::is_same_v<T, std::uint32_t>) ||
                (std::is_same_v<T, std::uint64_t>)) {
    return std::numeric_limits<T>::max();
  } else {
    static_assert(is_dependent_false_v<T>, "Error: Type not supported...");
  }
}

AMDGpuMetricsDataType_t dtype_from_attr(details::AMDGpuMetricAttributeType_t t) {
  switch (t) {
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT8: {
      return AMDGpuMetricsDataType_t::kUInt8;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT16: {
      return AMDGpuMetricsDataType_t::kUInt16;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT32: {
      return AMDGpuMetricsDataType_t::kUInt32;
    }
    default: {
      return AMDGpuMetricsDataType_t::kUInt64;
    }
  }
}

template<typename Tp>
constexpr uint64_t safe_way_to_uint64(Tp value) {
    if constexpr (std::is_signed_v<Tp>) {
        using intermediate_type = std::conditional_t<sizeof(Tp) <= sizeof(int64_t), int64_t, std::make_signed_t<Tp>>;
        return static_cast<uint64_t>(static_cast<intermediate_type>(value));
    } else {
        return static_cast<uint64_t>(value);
    }
}

// Existing format_metric_row doesn't take vectors, so overload and write our own
template<typename T, typename A>
AMDGpuDynamicMetricTblValues_t
format_metric_row(const std::vector<T, A>& vec, const std::string& value_title, details::AMDGpuMetricAttributeType_t attr_type)
{
  AMDGpuDynamicMetricTblValues_t out;
  out.reserve(vec.size());

  const auto n = static_cast<uint16_t>(
      std::min<std::size_t>(vec.size(), std::numeric_limits<uint16_t>::max()));

  for (uint16_t idx = 0; idx < n; ++idx) {
    uint64_t u64 = safe_way_to_uint64(vec[idx]);
    AMDGpuDynamicMetricsValue_t amdgpu_dynamic_metric_value_init{};
    amdgpu_dynamic_metric_value_init.m_value         = u64;
    amdgpu_dynamic_metric_value_init.m_info          = value_title + " : " + std::to_string(idx);
    amdgpu_dynamic_metric_value_init.m_original_type = dtype_from_attr(attr_type);
    out.emplace_back(std::move(amdgpu_dynamic_metric_value_init));
  }
  return out;
}

template<class T> struct is_vector : std::false_type {};
template<class U, class A> struct is_vector<std::vector<U,A>> : std::true_type {};

template<typename T>
AMDGpuDynamicMetricTblValues_t format_metric_row(const T& metric, const std::string& value_title) {
  auto multi_values = AMDGpuDynamicMetricTblValues_t{};

  // Determine data type and count inline
  AMDGpuMetricsDataType_t inferred_data_type = AMDGpuMetricsDataType_t::kUInt64;
  uint16_t num_values = 1;

  if constexpr (std::is_array_v<T>) {
    using ElementType = std::remove_extent_t<T>;
    if constexpr (std::is_same_v<ElementType, uint8_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt8;
    } else if constexpr (std::is_same_v<ElementType, uint16_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt16;
    } else if constexpr (std::is_same_v<ElementType, uint32_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt32;
    } else if constexpr (std::is_same_v<ElementType, uint64_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt64;
    }
    num_values = static_cast<uint16_t>(std::extent_v<T>);
  } else {
    // Scalar case
    if constexpr (std::is_same_v<T, uint8_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt16;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      inferred_data_type = AMDGpuMetricsDataType_t::kUInt64;
    }
  }

  // Populate the values
  for (uint16_t idx = 0; idx < num_values; ++idx) {
    uint64_t value = 0;
    if constexpr (std::is_array_v<T>) {
      value = static_cast<uint64_t>(metric[idx]);
    } else {
      value = static_cast<uint64_t>(metric);
    }

    AMDGpuDynamicMetricsValue_t amdgpu_dynamic_metric_value_init{};
    amdgpu_dynamic_metric_value_init.m_value = value;
    amdgpu_dynamic_metric_value_init.m_info = (value_title + " : " + std::to_string(idx));
    amdgpu_dynamic_metric_value_init.m_original_type = inferred_data_type;

    multi_values.emplace_back(amdgpu_dynamic_metric_value_init);
  }

  return multi_values;
}

rsmi_status_t GpuMetricsBaseDynamic_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};

  auto emit = [&](AMDGpuMetricsClassId_t cls, AMDGpuMetricsUnitType_t unit,
                  const char* label,
                  const details::AMDGpuMetricAttributeData_t& row) {

    auto rows = std::visit([&](const auto& x) -> AMDGpuDynamicMetricTblValues_t {
    using S = std::decay_t<decltype(x)>;
    if constexpr (is_vector<S>::value) { // Would like to use is_multivalued() here, but compiler needs well-formed
        return format_metric_row(x, std::string(label), row.m_instance.m_attribute_type);
      } else {
        return format_metric_row(x, std::string(label));
      }
    }, row.m_value);

    m_metrics_dynamic_tbl[cls].insert({unit, std::move(rows)});
  };

  for (const auto& r : m_dyn.get_metric_rows()) {
    switch (r.m_instance.m_attribute_id) {
      // Power energy and temperature
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_HOTSPOT:
        emit(AMDGpuMetricsClassId_t::kGpuMetricTemperature, AMDGpuMetricsUnitType_t::kMetricTempHotspot,
             "temperature_hotspot", r);
        break;
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_MEM:
        emit(AMDGpuMetricsClassId_t::kGpuMetricTemperature, AMDGpuMetricsUnitType_t::kMetricTempMem,
             "temperature_mem", r);
        break;
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_VRSOC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricTemperature, AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
             "temperature_vrsoc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::CURR_SOCKET_POWER:
        emit(AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy, AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
             "curr_socket_power", r);
        break;
      case details::AMDGpuMetricAttributeId_t::ENERGY_ACCUMULATOR:
        emit(AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy, AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
             "energy_acc", r);
        break;

      // Utilization
      case details::AMDGpuMetricAttributeId_t::AVERAGE_GFX_ACTIVITY:
        emit(AMDGpuMetricsClassId_t::kGpuMetricUtilization, AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
             "average_gfx_activity", r);
        break;
      case details::AMDGpuMetricAttributeId_t::AVERAGE_UMC_ACTIVITY:
        emit(AMDGpuMetricsClassId_t::kGpuMetricUtilization, AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
             "average_umc_activity", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_ACTIVITY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricUtilization, AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
             "gfx_activity_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::MEM_ACTIVITY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricUtilization, AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
             "mem_activity_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFXCLK_LOCK_STATUS:
        emit(AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus, AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
             "gfxclk_lock_status", r);
        break;

      // Metric Timestamp
      case details::AMDGpuMetricAttributeId_t::FIRMWARE_TIMESTAMP:
        emit(AMDGpuMetricsClassId_t::kGpuMetricTimestamp, AMDGpuMetricsUnitType_t::kMetricTSFirmware,
             "firmware_timestamp", r);
        break;
      case details::AMDGpuMetricAttributeId_t::SYSTEM_CLOCK_COUNTER:
        emit(AMDGpuMetricsClassId_t::kGpuMetricTimestamp, AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
             "system_clock_counter", r);
        break;

      // Throttle Residency
      case details::AMDGpuMetricAttributeId_t::ACCUMULATION_COUNTER:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricAccumulationCounter,
             "accumulation_counter", r);
        break;

      // Link Width Speed
      case details::AMDGpuMetricAttributeId_t::PCIE_LINK_WIDTH:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
             "pcie_link_width", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_LINK_SPEED:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
             "pcie_link_speed", r);
        break;
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_WIDTH:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
             "xgmi_link_width", r);
        break;
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_SPEED:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
             "xgmi_link_speed", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
             "pcie_bandwidth_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_INST:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
             "pcie_bandwidth_inst", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_L0_TO_RECOV_COUNT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
             "pcie_l0_recov_count_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_COUNT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
             "pcie_replay_count_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_ROVER_COUNT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
             "pcie_replay_rollover_count_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_NAK_SENT_COUNT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator,
             "pcie_nak_sent_count_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PCIE_NAK_RCVD_COUNT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator,
             "pcie_nak_rcvd_count_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::XGMI_READ_DATA_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
             "xgmi_read_data_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::XGMI_WRITE_DATA_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
             "xgmi_write_data_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_STATUS:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricXgmiLinkStatus,
             "xgmi_link_status", r);
        break;

      case details::AMDGpuMetricAttributeId_t::MEM_MAX_BANDWIDTH:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricVramMaxBandwidth,
             "vram_max_bandwidth", r);
        break;

      case details::AMDGpuMetricAttributeId_t::PCIE_LC_PERF_OTHER_END_RECOVERY:
        emit(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, AMDGpuMetricsUnitType_t::kMetricPcieLCPerfOtherEndRecov,
             "pcie_lc_perf_other_end_recovery", r);
        break;

      // Current Clock
      case details::AMDGpuMetricAttributeId_t::CURRENT_GFXCLK:
        emit(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
             "current_gfxclk", r);
        break;
      case details::AMDGpuMetricAttributeId_t::CURRENT_SOCCLK:
        emit(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
             "current_socclk", r);
        break;
      case details::AMDGpuMetricAttributeId_t::CURRENT_VCLK0:
        emit(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
             "current_vclk0", r);
        break;
      case details::AMDGpuMetricAttributeId_t::CURRENT_DCLK0:
        emit(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
             "current_dclk0", r);
        break;
      case details::AMDGpuMetricAttributeId_t::CURRENT_UCLK:
        emit(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrUClock,
             "current_uclk", r);
        break;

      // Throttle Residency
      case details::AMDGpuMetricAttributeId_t::PROCHOT_RESIDENCY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricProchotResidencyAccumulator,
             "prochot_residency_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::PPT_RESIDENCY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricPPTResidencyAccumulator,
             "ppt_residency_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::SOCKET_THM_RESIDENCY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricSocketThmResidencyAccumulator,
             "socket_thm_residency_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::VR_THM_RESIDENCY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricVRThmResidencyAccumulator,
             "vr_thm_residency_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::HBM_THM_RESIDENCY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency, AMDGpuMetricsUnitType_t::kMetricHBMThmResidencyAccumulator,
             "hbm_thm_residency_acc", r);
        break;

      // XCP stats
      case details::AMDGpuMetricAttributeId_t::GFX_BUSY_INST:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBusyInst,
             "xcp_stats->gfx_busy_inst", r);
        break;
      case details::AMDGpuMetricAttributeId_t::JPEG_BUSY:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricJpegBusy,
             "xcp_stats->jpeg_busy", r);
        break;
      case details::AMDGpuMetricAttributeId_t::VCN_BUSY:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricVcnBusy,
             "xcp_stats->vcn_busy", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_BUSY_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc,
             "xcp_stats->gfx_busy_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_PPT_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitPptAcc,
             "xcp_stats->gfx_below_host_limit_ppt_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_THM_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitThmAcc,
             "xcp_stats->gfx_below_host_limit_thm_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_LOW_UTILIZATION_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxLowUtilitizationAcc,
             "xcp_stats->gfx_low_utilization_acc", r);
        break;
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_TOTAL_ACC:
        emit(AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitTotalAcc,
             "xcp_stats->gfx_below_host_limit_total_acc", r);
        break;

      default:
        ss  << __PRETTY_FUNCTION__
            << " UNKNOWN Attribute "
            << static_cast<uint32_t>(r.m_instance.m_attribute_id)
            << " |";
        LOG_ERROR(ss);
        break;
    }
  }

  ss << __PRETTY_FUNCTION__ << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code) << " |";
  LOG_TRACE(ss);

  { std::lock_guard<std::mutex> lk(s_base_tbl_mu);
    // Copy to base class
    this->m_base_metrics_dynamic_tbl = m_metrics_dynamic_tbl;
  }

  return status_code;
}

rsmi_status_t GpuMetricsBase_v18_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};

  // Lambda function to populate the metrics table
  auto populate_metrics_table = [&](AMDGpuMetricsClassId_t class_id,
                                    AMDGpuMetricsUnitType_t unit_type, const auto& metric,
                                    const std::string& metric_name) {
    m_metrics_dynamic_tbl[class_id].insert(
        std::make_pair(unit_type, format_metric_row(metric, metric_name)));
  };

  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v18 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version =
        translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__ << " | ======= info ======= "
       << " | Applying adjustments "
       << " | Metric Version: "
       << stringfy_metric_header_version(disjoin_metrics_version(gpu_metrics_version)) << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution; leave as-is.
    ss << __PRETTY_FUNCTION__
       << " | firmware_timestamp (10ns) = " << m_gpu_metrics_tbl.m_firmware_timestamp;
    LOG_DEBUG(ss);
  };

  ss << __PRETTY_FUNCTION__ << " | ======= info ======= "
     << " | START-> " << (m_is_partition_metrics ? "Partitioned" : "GPU")
     << " metrics for partition: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << m_is_partition_metrics;
  LOG_DEBUG(ss);

  if (!m_is_partition_metrics) {
    run_metric_adjustments_v18();

    // Temperature Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricTemperature,
      AMDGpuMetricsUnitType_t::kMetricTempHotspot, m_gpu_metrics_tbl.m_temperature_hotspot,
      "temperature_hotspot");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricTemperature,
      AMDGpuMetricsUnitType_t::kMetricTempMem,
      m_gpu_metrics_tbl.m_temperature_mem, "temperature_mem");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricTemperature,
      AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
      m_gpu_metrics_tbl.m_temperature_vrsoc, "temperature_vrsoc");

    // Power/Energy Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy,
      AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
      m_gpu_metrics_tbl.m_current_socket_power, "curr_socket_power");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy,
      AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
      m_gpu_metrics_tbl.m_energy_accumulator, "energy_acc");

    // Utilization Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricUtilization,
      AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
      m_gpu_metrics_tbl.m_average_gfx_activity, "average_gfx_activity");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricUtilization,
      AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
      m_gpu_metrics_tbl.m_average_umc_activity, "average_umc_activity");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricUtilization,
      AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
      m_gpu_metrics_tbl.m_gfx_activity_acc, "gfx_activity_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricUtilization,
      AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
      m_gpu_metrics_tbl.m_mem_activity_acc, "mem_activity_acc");

    // GfxLock Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus,
      AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
      m_gpu_metrics_tbl.m_gfxclk_lock_status, "gfxclk_lock_status");

    // Timestamp Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricTimestamp,
      AMDGpuMetricsUnitType_t::kMetricTSFirmware,
      m_gpu_metrics_tbl.m_firmware_timestamp, "firmware_timestamp");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricTimestamp,
      AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
      m_gpu_metrics_tbl.m_system_clock_counter, "system_clock_counter");

    // Link/Width/Speed Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
      m_gpu_metrics_tbl.m_pcie_link_width, "pcie_link_width");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
      m_gpu_metrics_tbl.m_pcie_link_speed, "pcie_link_speed");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
      m_gpu_metrics_tbl.m_xgmi_link_width, "xgmi_link_width");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
      m_gpu_metrics_tbl.m_xgmi_link_speed, "xgmi_link_speed");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
      m_gpu_metrics_tbl.m_pcie_bandwidth_acc, "pcie_bandwidth_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
      m_gpu_metrics_tbl.m_pcie_bandwidth_inst, "pcie_bandwidth_inst");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
      m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc,
      "pcie_l0_recov_count_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
      m_gpu_metrics_tbl.m_pcie_replay_count_acc, "pcie_replay_count_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
      m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc,
      "pcie_replay_rollover_count_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator,
      m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc, "pcie_nak_sent_count_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator,
      m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc, "pcie_nak_rcvd_count_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
      m_gpu_metrics_tbl.m_xgmi_read_data_acc, "[xgmi_read_data_acc]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
      m_gpu_metrics_tbl.m_xgmi_write_data_acc, "[xgmi_write_data_acc]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricXgmiLinkStatus,
      m_gpu_metrics_tbl.m_xgmi_link_status, "[xgmi_link_status]");

    // Current Clock Info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
      m_gpu_metrics_tbl.m_current_gfxclk, "[current_gfxclk]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
      m_gpu_metrics_tbl.m_current_socclk, "[current_socclk]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
      m_gpu_metrics_tbl.m_current_vclk0, "[current_vclk0]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
      m_gpu_metrics_tbl.m_current_dclk0, "[current_dclk0]");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrUClock,
      m_gpu_metrics_tbl.m_current_uclk, "current_uclk");

    // Throttle residency counter
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricAccumulationCounter,
      m_gpu_metrics_tbl.m_accumulation_counter, "accumulation_counter");

    // Accumulated throttler residencies
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricProchotResidencyAccumulator,
      m_gpu_metrics_tbl.m_prochot_residency_acc, "prochot_residency_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricPPTResidencyAccumulator,
      m_gpu_metrics_tbl.m_ppt_residency_acc, "ppt_residency_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricSocketThmResidencyAccumulator,
      m_gpu_metrics_tbl.m_socket_thm_residency_acc, "socket_thm_residency_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricVRThmResidencyAccumulator,
      m_gpu_metrics_tbl.m_vr_thm_residency_acc, "vr_thm_residency_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency,
      AMDGpuMetricsUnitType_t::kMetricHBMThmResidencyAccumulator,
      m_gpu_metrics_tbl.m_hbm_thm_residency_acc, "hbm_thm_residency_acc");

    // Partition info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricPartition,
      AMDGpuMetricsUnitType_t::kGpuMetricNumPartition,
      m_gpu_metrics_tbl.m_num_partition, "num_partition");

    // xcp_stats info
    populate_metrics_table(
        AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBusyInst,
        m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_inst, "xcp_stats->gfx_busy_inst");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricVcnBusy,
      m_gpu_metrics_tbl.m_xcp_stats->vcn_busy, "xcp_stats->vcn_busy");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricJpegBusy,
      m_gpu_metrics_tbl.m_xcp_stats->jpeg_busy, "xcp_stats->jpeg_busy");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc,
      m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_acc, "xcp_stats->gfx_busy_acc");

    // GPU metrics v1.8 xcp_stats info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitTotalAcc,
      m_gpu_metrics_tbl.m_xcp_stats->gfx_below_host_limit_total_acc,
      "xcp_stats->gfx_below_host_limit_total_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitPptAcc,
      m_gpu_metrics_tbl.m_xcp_stats->gfx_below_host_limit_ppt_acc,
      "xcp_stats->gfx_below_host_limit_ppt_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitThmAcc,
      m_gpu_metrics_tbl.m_xcp_stats->gfx_below_host_limit_thm_acc,
      "xcp_stats->gfx_below_host_limit_thm_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxLowUtilitizationAcc,
      m_gpu_metrics_tbl.m_xcp_stats->gfx_low_utilization_acc,
      "xcp_stats->gfx_low_utilization_acc");

    // PCIE other end recovery counter info
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricPcieLCPerfOtherEndRecov,
      m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery,
      "pcie_lc_perf_other_end_recovery");

    // VRAM max bandwidth
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed,
      AMDGpuMetricsUnitType_t::kMetricVramMaxBandwidth,
      m_gpu_metrics_tbl.m_mem_max_bandwidth, "vram_max_bandwidth");
  } else {  // Partition metrics
    // Current clocks
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
      m_gpu_metrics_partition_tbl.m_current_gfxclk,
      "[partition 1.0] current_gfxclk");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricCurrentClock,
      AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
      m_gpu_metrics_partition_tbl.m_current_socclk,
      "[partition 1.0] current_socclk");
    populate_metrics_table(
      AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
      m_gpu_metrics_partition_tbl.m_current_vclk0, "[partition 1.0] current_vclk0");
    populate_metrics_table(
      AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
      m_gpu_metrics_partition_tbl.m_current_dclk0, "[partition 1.0] current_dclk0");
    populate_metrics_table(
      AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, AMDGpuMetricsUnitType_t::kMetricCurrUClock,
      m_gpu_metrics_partition_tbl.m_current_uclk, "[partition 1.0] current_uclk");

    // XCP stats - Utilization
    populate_metrics_table(
      AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBusyInst,
      m_gpu_metrics_partition_tbl.m_gfx_busy_inst, "[partition 1.0] gfx_busy_inst");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricVcnBusy,
      m_gpu_metrics_partition_tbl.m_vcn_busy, "[partition 1.0] vcn_busy");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricJpegBusy,
      m_gpu_metrics_partition_tbl.m_jpeg_busy, "[partition 1.0] jpeg_busy");
    populate_metrics_table(
      AMDGpuMetricsClassId_t::kGpuMetricXcpStats, AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc,
      m_gpu_metrics_partition_tbl.m_gfx_busy_acc, "[partition 1.0] gfx_busy_acc");

    // Total App Clock Counter Accumulated
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitTotalAcc,
      m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_total_acc,
      "[partition 1.0] gfx_below_host_limit_total_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitPptAcc,
      m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_ppt_acc,
      "[partition 1.0] gfx_below_host_limit_ppt_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitThmAcc,
      m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_thm_acc,
      "[partition 1.0] gfx_below_host_limit_thm_acc");
    populate_metrics_table(AMDGpuMetricsClassId_t::kGpuMetricXcpStats,
      AMDGpuMetricsUnitType_t::kMetricGfxLowUtilitizationAcc,
      m_gpu_metrics_partition_tbl.m_gfx_low_utilization_acc,
      "[partition 1.0] gfx_low_utilization_acc");
  }

  ss << __PRETTY_FUNCTION__ << " | ======= end ======= "
     << " | Success "
     << " | B4 copy"
     << " | m_metrics_dynamic_tbl size: " << m_metrics_dynamic_tbl.size()
     << " | m_base_metrics_dynamic_tbl size: "
     << GpuMetricsBase_t::m_base_metrics_dynamic_tbl.size()
     << " | m_partition_id: " << m_partition_id << " | is_partition_metrics: " << std::boolalpha
     << m_is_partition_metrics << " | Returning = " << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);

  {
    std::lock_guard<std::mutex> lk(s_base_tbl_mu);
    // Copy to base class
    this->m_base_metrics_dynamic_tbl = m_metrics_dynamic_tbl;
  }

  return status_code;
}

rsmi_status_t GpuMetricsBase_v17_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v17 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version =
      translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };

  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v17();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                               "temperature_hotspot")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc")));

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_current_socket_power,
                                "curr_socket_power")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc")));

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
                                "gfx_activity_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc")));

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter")));


  // GfxLock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
              format_metric_row(m_gpu_metrics_tbl.m_gfxclk_lock_status,
                                "gfxclk_lock_status")));

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_width,
                                "xgmi_link_width")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_speed,
                                "xgmi_link_speed")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_inst,
                                "pcie_bandwidth_inst")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc,
                                "pcie_l0_recov_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_count_acc,
                                "pcie_replay_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc,
                                "pcie_replay_rollover_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc,
                                "pcie_nak_sent_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc,
                                "pcie_nak_rcvd_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_read_data_acc,
                                "[xgmi_read_data_acc]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_write_data_acc,
                                "[xgmi_write_data_acc]")));
  // new for v1.7
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkStatus,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_status,
                                "[xgmi_link_status]")));
  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "[current_gfxclk]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "[current_socclk]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "[current_vclk0]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "[current_dclk0]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk")));

  /* Accumulation cycle counter */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAccumulationCounter,
              format_metric_row(m_gpu_metrics_tbl.m_accumulation_counter,
                                "accumulation_counter")));

  /* Accumulated throttler residencies */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricProchotResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_prochot_residency_acc,
                                "prochot_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPPTResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_ppt_residency_acc,
                                "ppt_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricSocketThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_socket_thm_residency_acc,
                                "socket_thm_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVRThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_vr_thm_residency_acc,
                                "vr_thm_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricHBMThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_hbm_thm_residency_acc,
                                "hbm_thm_residency_acc")));

  /* Partition info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPartition]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kGpuMetricNumPartition,
              format_metric_row(m_gpu_metrics_tbl.m_num_partition,
                                "num_partition")));

  /* xcp_stats info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxBusyInst,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_inst,
                                "xcp_stats->gfx_busy_inst")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVcnBusy,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->vcn_busy,
                                "xcp_stats->vcn_busy")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricJpegBusy,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->jpeg_busy,
                                "xcp_stats->jpeg_busy")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_acc,
                                "xcp_stats->gfx_busy_acc")));

  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats].insert(
      std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxBelowHostLimitAccumulator,
                     format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->gfx_below_host_limit_acc,
                                       "xcp_stats->gfx_below_host_limit_acc")));

  /* PCIE other end recovery counter info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLCPerfOtherEndRecov,
           format_metric_row(m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery,
                                "pcie_lc_perf_other_end_recovery")));

  /* VRAM max bandwidth at max memory clock */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVramMaxBandwidth,
              format_metric_row(m_gpu_metrics_tbl.m_vram_max_bandwidth,
                                "vram_max_bandwidth")));

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

rsmi_status_t GpuMetricsBase_v16_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v16 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version =
      translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };

  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v16();
  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                               "temperature_hotspot")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc")));

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_current_socket_power,
                                "curr_socket_power")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc")));

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
              "gfx_activity_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc")));

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter")));


  // GfxLock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
              format_metric_row(m_gpu_metrics_tbl.m_gfxclk_lock_status,
                                "gfxclk_lock_status")));

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_width,
                                "xgmi_link_width")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_speed,
                                "xgmi_link_speed")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_inst,
                                "pcie_bandwidth_inst")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc,
                                "pcie_l0_recov_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_count_acc,
                                "pcie_replay_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc,
                                "pcie_replay_rollover_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc,
                                "pcie_nak_sent_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc,
                                "pcie_nak_rcvd_count_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_read_data_acc,
                                "[xgmi_read_data_acc]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_write_data_acc,
                                "[xgmi_write_data_acc]")));

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "[current_gfxclk]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "[current_socclk]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "[current_vclk0]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "[current_dclk0]")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk")));

  /* Accumulation cycle counter */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAccumulationCounter,
              format_metric_row(m_gpu_metrics_tbl.m_accumulation_counter,
                                "accumulation_counter")));

  /* Accumulated throttler residencies */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricProchotResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_prochot_residency_acc,
                                "prochot_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPPTResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_ppt_residency_acc,
                                "ppt_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricSocketThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_socket_thm_residency_acc,
                                "socket_thm_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVRThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_vr_thm_residency_acc,
                                "vr_thm_residency_acc")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleResidency]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricHBMThmResidencyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_hbm_thm_residency_acc,
                                "hbm_thm_residency_acc")));

  /* Partition info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPartition]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kGpuMetricNumPartition,
              format_metric_row(m_gpu_metrics_tbl.m_num_partition,
                                "num_partition")));

  /* xcp_stats info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxBusyInst,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_inst,
                                "xcp_stats->gfx_busy_inst")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVcnBusy,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->vcn_busy,
                                "xcp_stats->vcn_busy")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricJpegBusy,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->jpeg_busy,
                                "xcp_stats->jpeg_busy")));
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricXcpStats]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxBusyAcc,
              format_metric_row(m_gpu_metrics_tbl.m_xcp_stats->gfx_busy_acc,
              "xcp_stats->gfx_busy_acc")));

  /* PCIE other end recovery counter info */
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLCPerfOtherEndRecov,
           format_metric_row(m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery,
          "pcie_lc_perf_other_end_recovery")));

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

rsmi_status_t GpuMetricsBase_v15_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v15 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };


  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v15();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                               "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_current_socket_power,
                                "curr_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVcnActivity,
              format_metric_row(m_gpu_metrics_tbl.m_vcn_activity,
                                "[average_vcn_activity]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricJpegActivity,
              format_metric_row(m_gpu_metrics_tbl.m_jpeg_activity,
                                "[average_jpeg_activity]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
              "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );

  // GfxLock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
              format_metric_row(m_gpu_metrics_tbl.m_gfxclk_lock_status,
                                "gfxclk_lock_status"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_width,
                                "xgmi_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_speed,
                                "xgmi_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_inst,
                                "pcie_bandwidth_inst"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc,
                                "pcie_l0_recov_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_count_acc,
                                "pcie_replay_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc,
                                "pcie_replay_rollover_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakSentCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc,
                                "pcie_nak_sent_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieNakReceivedCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc,
                                "pcie_nak_rcvd_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_read_data_acc,
                                "[xgmi_read_data_acc]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_write_data_acc,
                                "[xgmi_write_data_acc]"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "[current_gfxclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "[current_socclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "[current_vclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "[current_dclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

rsmi_status_t GpuMetricsBase_v14_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v14 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };


  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v14();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                               "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_current_socket_power,
                                "curr_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVcnActivity,
              format_metric_row(m_gpu_metrics_tbl.m_vcn_activity,
                                "[average_vcn_activity]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
              "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );

  // GfxLock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
              format_metric_row(m_gpu_metrics_tbl.m_gfxclk_lock_status,
                                "gfxclk_lock_status"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_width,
                                "xgmi_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_speed,
                                "xgmi_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_inst,
                                "pcie_bandwidth_inst"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieL0RecovCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc,
                                "pcie_l0_recov_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_count_acc,
                                "pcie_replay_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieReplayRollOverCountAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc,
                                "pcie_replay_rollover_count_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_read_data_acc,
                                "[xgmi_read_data_acc]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_write_data_acc,
                                "[xgmi_write_data_acc]"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "[current_gfxclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "[current_socclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "[current_vclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "[current_dclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code)
     << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

rsmi_status_t init_max_public_gpu_matrics(AMGpuMetricsPublicLatest_t& rsmi_gpu_metrics)
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  rsmi_gpu_metrics.temperature_edge = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_edge)>();
  rsmi_gpu_metrics.temperature_hotspot = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_hotspot)>();
  rsmi_gpu_metrics.temperature_mem = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_mem)>();
  rsmi_gpu_metrics.temperature_vrgfx = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_vrgfx)>();
  rsmi_gpu_metrics.temperature_vrsoc = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_vrsoc)>();
  rsmi_gpu_metrics.temperature_vrmem = init_max_uint_types<decltype(rsmi_gpu_metrics.temperature_vrmem)>();
  rsmi_gpu_metrics.average_gfx_activity = init_max_uint_types<decltype(rsmi_gpu_metrics.average_gfx_activity)>();
  rsmi_gpu_metrics.average_umc_activity = init_max_uint_types<decltype(rsmi_gpu_metrics.average_umc_activity)>();
  rsmi_gpu_metrics.average_mm_activity = init_max_uint_types<decltype(rsmi_gpu_metrics.average_mm_activity)>();
  rsmi_gpu_metrics.average_socket_power = init_max_uint_types<decltype(rsmi_gpu_metrics.average_socket_power)>();
  rsmi_gpu_metrics.energy_accumulator = init_max_uint_types<decltype(rsmi_gpu_metrics.energy_accumulator)>();
  rsmi_gpu_metrics.system_clock_counter = init_max_uint_types<decltype(rsmi_gpu_metrics.system_clock_counter)>();
  rsmi_gpu_metrics.average_gfxclk_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_gfxclk_frequency)>();
  rsmi_gpu_metrics.average_socclk_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_socclk_frequency)>();
  rsmi_gpu_metrics.average_uclk_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_uclk_frequency)>();
  rsmi_gpu_metrics.average_vclk0_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_vclk0_frequency)>();
  rsmi_gpu_metrics.average_dclk0_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_dclk0_frequency)>();
  rsmi_gpu_metrics.average_vclk1_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_vclk1_frequency)>();
  rsmi_gpu_metrics.average_dclk1_frequency = init_max_uint_types<decltype(rsmi_gpu_metrics.average_dclk1_frequency)>();
  rsmi_gpu_metrics.current_gfxclk = init_max_uint_types<decltype(rsmi_gpu_metrics.current_gfxclk)>();
  rsmi_gpu_metrics.current_socclk = init_max_uint_types<decltype(rsmi_gpu_metrics.current_socclk)>();
  rsmi_gpu_metrics.current_uclk = init_max_uint_types<decltype(rsmi_gpu_metrics.current_uclk)>();
  rsmi_gpu_metrics.current_vclk0 = init_max_uint_types<decltype(rsmi_gpu_metrics.current_vclk0)>();
  rsmi_gpu_metrics.current_dclk0 = init_max_uint_types<decltype(rsmi_gpu_metrics.current_dclk0)>();
  rsmi_gpu_metrics.current_vclk1 = init_max_uint_types<decltype(rsmi_gpu_metrics.current_vclk1)>();
  rsmi_gpu_metrics.current_dclk1 = init_max_uint_types<decltype(rsmi_gpu_metrics.current_dclk1)>();
  rsmi_gpu_metrics.throttle_status = init_max_uint_types<decltype(rsmi_gpu_metrics.throttle_status)>();
  rsmi_gpu_metrics.current_fan_speed = init_max_uint_types<decltype(rsmi_gpu_metrics.current_fan_speed)>();
  rsmi_gpu_metrics.pcie_link_width = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_link_width)>();
  rsmi_gpu_metrics.pcie_link_speed = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_link_speed)>();
  rsmi_gpu_metrics.gfx_activity_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.gfx_activity_acc)>();
  rsmi_gpu_metrics.mem_activity_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.mem_activity_acc)>();
  rsmi_gpu_metrics.vram_max_bandwidth = init_max_uint_types<decltype(rsmi_gpu_metrics.vram_max_bandwidth)>();

  std::fill(std::begin(rsmi_gpu_metrics.xgmi_link_status),
            std::end(rsmi_gpu_metrics.xgmi_link_status),
            init_max_uint_types<std::uint16_t>());


  std::fill(std::begin(rsmi_gpu_metrics.temperature_hbm),
            std::end(rsmi_gpu_metrics.temperature_hbm),
            init_max_uint_types<std::uint16_t>());

  rsmi_gpu_metrics.firmware_timestamp = init_max_uint_types<decltype(rsmi_gpu_metrics.firmware_timestamp)>();
  rsmi_gpu_metrics.voltage_soc = init_max_uint_types<decltype(rsmi_gpu_metrics.voltage_soc)>();
  rsmi_gpu_metrics.voltage_gfx = init_max_uint_types<decltype(rsmi_gpu_metrics.voltage_gfx)>();
  rsmi_gpu_metrics.voltage_mem = init_max_uint_types<decltype(rsmi_gpu_metrics.voltage_mem)>();
  rsmi_gpu_metrics.indep_throttle_status = init_max_uint_types<decltype(rsmi_gpu_metrics.indep_throttle_status)>();
  rsmi_gpu_metrics.current_socket_power  = init_max_uint_types<decltype(rsmi_gpu_metrics.current_socket_power)>();

  std::fill(std::begin(rsmi_gpu_metrics.vcn_activity),
            std::end(rsmi_gpu_metrics.vcn_activity),
            init_max_uint_types<std::uint16_t>());

  std::fill(std::begin(rsmi_gpu_metrics.jpeg_activity),
            std::end(rsmi_gpu_metrics.jpeg_activity),
            init_max_uint_types<std::uint16_t>());

  rsmi_gpu_metrics.gfxclk_lock_status = init_max_uint_types<decltype(rsmi_gpu_metrics.gfxclk_lock_status)>();
  rsmi_gpu_metrics.xgmi_link_width = init_max_uint_types<decltype(rsmi_gpu_metrics.xgmi_link_width)>();
  rsmi_gpu_metrics.xgmi_link_speed = init_max_uint_types<decltype(rsmi_gpu_metrics.xgmi_link_speed)>();
  rsmi_gpu_metrics.pcie_bandwidth_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_bandwidth_acc)>();
  rsmi_gpu_metrics.pcie_bandwidth_inst = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_bandwidth_inst)>();
  rsmi_gpu_metrics.pcie_l0_to_recov_count_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_l0_to_recov_count_acc)>();
  rsmi_gpu_metrics.pcie_replay_count_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_replay_count_acc)>();
  rsmi_gpu_metrics.pcie_replay_rover_count_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_replay_rover_count_acc)>();

  std::fill(std::begin(rsmi_gpu_metrics.xgmi_read_data_acc),
            std::end(rsmi_gpu_metrics.xgmi_read_data_acc),
            init_max_uint_types<std::uint64_t>());

  std::fill(std::begin(rsmi_gpu_metrics.xgmi_write_data_acc),
            std::end(rsmi_gpu_metrics.xgmi_write_data_acc),
            init_max_uint_types<std::uint64_t>());

  std::fill(std::begin(rsmi_gpu_metrics.current_gfxclks),
            std::end(rsmi_gpu_metrics.current_gfxclks),
            init_max_uint_types<std::uint16_t>());

  std::fill(std::begin(rsmi_gpu_metrics.current_socclks),
            std::end(rsmi_gpu_metrics.current_socclks),
            init_max_uint_types<std::uint16_t>());

  std::fill(std::begin(rsmi_gpu_metrics.current_vclk0s),
            std::end(rsmi_gpu_metrics.current_vclk0s),
            init_max_uint_types<std::uint16_t>());

  std::fill(std::begin(rsmi_gpu_metrics.current_dclk0s),
            std::end(rsmi_gpu_metrics.current_dclk0s),
            init_max_uint_types<std::uint16_t>());

  rsmi_gpu_metrics.pcie_nak_sent_count_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_nak_sent_count_acc)>();
  rsmi_gpu_metrics.pcie_nak_rcvd_count_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_nak_rcvd_count_acc)>();

  rsmi_gpu_metrics.accumulation_counter = init_max_uint_types<decltype(rsmi_gpu_metrics.accumulation_counter)>();
  rsmi_gpu_metrics.prochot_residency_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.prochot_residency_acc)>();
  rsmi_gpu_metrics.ppt_residency_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.ppt_residency_acc)>();
  rsmi_gpu_metrics.socket_thm_residency_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.socket_thm_residency_acc)>();
  rsmi_gpu_metrics.vr_thm_residency_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.vr_thm_residency_acc)>();
  rsmi_gpu_metrics.hbm_thm_residency_acc = init_max_uint_types<decltype(rsmi_gpu_metrics.hbm_thm_residency_acc)>();

  rsmi_gpu_metrics.num_partition = init_max_uint_types<decltype(rsmi_gpu_metrics.num_partition)>();

  rsmi_gpu_metrics.pcie_lc_perf_other_end_recovery =
    init_max_uint_types<decltype(rsmi_gpu_metrics.pcie_lc_perf_other_end_recovery)>();

  for (auto& row : rsmi_gpu_metrics.xcp_stats) {
    std::fill(std::begin(row.gfx_busy_inst), std::end(row.gfx_busy_inst),
              init_max_uint_types<std::uint32_t>());
    std::fill(std::begin(row.jpeg_busy), std::end(row.jpeg_busy),
              init_max_uint_types<std::uint16_t>());
    std::fill(std::begin(row.vcn_busy), std::end(row.vcn_busy),
              init_max_uint_types<std::uint16_t>());
    std::fill(std::begin(row.gfx_busy_acc), std::end(row.gfx_busy_acc),
              init_max_uint_types<std::uint64_t>());
    std::fill(std::begin(row.gfx_below_host_limit_acc), std::end(row.gfx_below_host_limit_acc),
              init_max_uint_types<std::uint64_t>());
    std::fill(std::begin(row.gfx_below_host_limit_ppt_acc), std::end(row.gfx_below_host_limit_ppt_acc),
              init_max_uint_types<std::uint64_t>());
    std::fill(std::begin(row.gfx_below_host_limit_thm_acc), std::end(row.gfx_below_host_limit_thm_acc),
              init_max_uint_types<std::uint64_t>());
    std::fill(std::begin(row.gfx_low_utilization_acc), std::end(row.gfx_low_utilization_acc),
              init_max_uint_types<std::uint64_t>());
    std::fill(std::begin(row.gfx_below_host_limit_total_acc), std::end(row.gfx_below_host_limit_total_acc),
              init_max_uint_types<std::uint64_t>());
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code)
     << " |";
  LOG_TRACE(ss);

  return status_code;
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBaseDynamic_t::copy_internal_to_external_metrics() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  AMGpuMetricsPublicLatest_t out{};
  init_max_public_gpu_matrics(out);

  out.common_header.structure_size = m_header.m_structure_size;
  out.common_header.format_revision = m_header.m_format_revision;
  out.common_header.content_revision = m_header.m_content_revision;

  auto assign_by_type = [&](auto& dst, const details::AMDGpuMetricAttributeData_t& r) {
    using D = std::decay_t<decltype(dst)>;

    std::visit(
        [&](const auto& x) {
          using S = std::decay_t<decltype(x)>;
          if constexpr (std::is_integral_v<S>) {
            dst = static_cast<D>(x);
          }
        },
        r.m_value);
  };

  auto assign_vector = [&](auto& dst, const details::AMDGpuMetricAttributeData_t& r,
                           std::size_t cap) {
    using Dst = std::remove_reference_t<decltype(dst)>;
    using T = std::remove_cv_t<std::remove_extent_t<Dst>>;
    auto v = std::get_if<std::vector<T>>(&r.m_value);
    const std::size_t n = std::min<std::size_t>(v->size(), cap);
    std::copy_n(v->data(), n, dst);
  };

  for (const auto& r : m_dyn.get_metric_rows()) {
    switch (r.m_instance.m_attribute_id) {
      // Temps
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_HOTSPOT:
        assign_by_type(out.temperature_hotspot, r); break;
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_MEM:
        assign_by_type(out.temperature_mem, r); break;
      case details::AMDGpuMetricAttributeId_t::TEMPERATURE_VRSOC:
        assign_by_type(out.temperature_vrsoc, r); break;

      // Power/Energy
      case details::AMDGpuMetricAttributeId_t::CURR_SOCKET_POWER:
        assign_by_type(out.current_socket_power, r); break;
      case details::AMDGpuMetricAttributeId_t::ENERGY_ACCUMULATOR:
        assign_by_type(out.energy_accumulator, r); break;

      // Utilization
      case details::AMDGpuMetricAttributeId_t::AVERAGE_GFX_ACTIVITY:
        assign_by_type(out.average_gfx_activity, r); break;
      case details::AMDGpuMetricAttributeId_t::AVERAGE_UMC_ACTIVITY:
        assign_by_type(out.average_umc_activity, r); break;
      case details::AMDGpuMetricAttributeId_t::GFX_ACTIVITY_ACC:
        assign_by_type(out.gfx_activity_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::MEM_ACTIVITY_ACC:
        assign_by_type(out.mem_activity_acc, r); break;

      // Timestamps / Lock
      case details::AMDGpuMetricAttributeId_t::SYSTEM_CLOCK_COUNTER:
        assign_by_type(out.system_clock_counter, r); break;
      case details::AMDGpuMetricAttributeId_t::FIRMWARE_TIMESTAMP:
        assign_by_type(out.firmware_timestamp, r); break;
      case details::AMDGpuMetricAttributeId_t::GFXCLK_LOCK_STATUS:
        assign_by_type(out.gfxclk_lock_status, r); break;

      // Link width/speed, bandwidth, counts
      case details::AMDGpuMetricAttributeId_t::PCIE_LINK_WIDTH:
        assign_by_type(out.pcie_link_width, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_LINK_SPEED:
        assign_by_type(out.pcie_link_speed, r); break;
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_WIDTH:
        assign_by_type(out.xgmi_link_width, r); break;
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_SPEED:
        assign_by_type(out.xgmi_link_speed, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_ACC:
        assign_by_type(out.pcie_bandwidth_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_BANDWIDTH_INST:
        assign_by_type(out.pcie_bandwidth_inst, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_L0_TO_RECOV_COUNT_ACC:
        assign_by_type(out.pcie_l0_to_recov_count_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_COUNT_ACC:
        assign_by_type(out.pcie_replay_count_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_REPLAY_ROVER_COUNT_ACC:
        assign_by_type(out.pcie_replay_rover_count_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_NAK_SENT_COUNT_ACC:
        assign_by_type(out.pcie_nak_sent_count_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PCIE_NAK_RCVD_COUNT_ACC:
        assign_by_type(out.pcie_nak_rcvd_count_acc, r); break;

      // Residency / counters
      case details::AMDGpuMetricAttributeId_t::ACCUMULATION_COUNTER:
        assign_by_type(out.accumulation_counter, r); break;
      case details::AMDGpuMetricAttributeId_t::PROCHOT_RESIDENCY_ACC:
        assign_by_type(out.prochot_residency_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::PPT_RESIDENCY_ACC:
        assign_by_type(out.ppt_residency_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::SOCKET_THM_RESIDENCY_ACC:
        assign_by_type(out.socket_thm_residency_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::VR_THM_RESIDENCY_ACC:
        assign_by_type(out.vr_thm_residency_acc, r); break;
      case details::AMDGpuMetricAttributeId_t::HBM_THM_RESIDENCY_ACC:
        assign_by_type(out.hbm_thm_residency_acc, r); break;

      // VRAM max bandwidth
      case details::AMDGpuMetricAttributeId_t::MEM_MAX_BANDWIDTH:
        assign_by_type(out.vram_max_bandwidth, r); break;

      // XGMI accumulators / link status (arrays)
      case details::AMDGpuMetricAttributeId_t::XGMI_READ_DATA_ACC: {
        assign_vector(out.xgmi_read_data_acc, r, RSMI_MAX_NUM_XGMI_LINKS); break;
      }
      case details::AMDGpuMetricAttributeId_t::XGMI_WRITE_DATA_ACC: {
        assign_vector(out.xgmi_write_data_acc, r, RSMI_MAX_NUM_XGMI_LINKS); break;
      }
      case details::AMDGpuMetricAttributeId_t::XGMI_LINK_STATUS: {
        assign_vector(out.xgmi_link_status, r, RSMI_MAX_NUM_XGMI_LINKS); break;
      }

      // Current clocks (arrays) + uclk (scalar)
      case details::AMDGpuMetricAttributeId_t::CURRENT_GFXCLK: {
        assign_vector(out.current_gfxclks, r, RSMI_MAX_NUM_GFX_CLKS); break;
      }
      case details::AMDGpuMetricAttributeId_t::CURRENT_SOCCLK: {
        assign_vector(out.current_socclks, r, RSMI_MAX_NUM_CLKS); break;
      }
      case details::AMDGpuMetricAttributeId_t::CURRENT_VCLK0: {
        assign_vector(out.current_vclk0s, r, RSMI_MAX_NUM_CLKS); break;
      }
      case details::AMDGpuMetricAttributeId_t::CURRENT_DCLK0: {
        assign_vector(out.current_dclk0s, r, RSMI_MAX_NUM_CLKS); break;
      }

      case details::AMDGpuMetricAttributeId_t::CURRENT_UCLK:
        assign_by_type(out.current_uclk, r); break;

      case details::AMDGpuMetricAttributeId_t::PCIE_LC_PERF_OTHER_END_RECOVERY:
        assign_by_type(out.pcie_lc_perf_other_end_recovery, r); break;

      // XCP stats
      // Only fill in entry 0
      case details::AMDGpuMetricAttributeId_t::GFX_BUSY_INST: {
        assign_vector(out.xcp_stats[0].gfx_busy_inst, r, RSMI_MAX_NUM_XCC); break;
      }
      case details::AMDGpuMetricAttributeId_t::JPEG_BUSY: {
        assign_vector(out.xcp_stats[0].jpeg_busy, r, RSMI_MAX_NUM_JPEG_ENG_V1); break;
      }
      case details::AMDGpuMetricAttributeId_t::VCN_BUSY: {
        assign_vector(out.xcp_stats[0].vcn_busy, r, RSMI_MAX_NUM_VCNS); break;
      }
      case details::AMDGpuMetricAttributeId_t::GFX_BUSY_ACC: {
        assign_vector(out.xcp_stats[0].gfx_busy_acc, r, RSMI_MAX_NUM_XCC); break;
      }
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_PPT_ACC: {
        assign_vector(out.xcp_stats[0].gfx_below_host_limit_ppt_acc, r, RSMI_MAX_NUM_XCC); break;
      }
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_THM_ACC: {
        assign_vector(out.xcp_stats[0].gfx_below_host_limit_thm_acc, r, RSMI_MAX_NUM_XCC); break;
      }
      case details::AMDGpuMetricAttributeId_t::GFX_LOW_UTILIZATION_ACC: {
        assign_vector(out.xcp_stats[0].gfx_low_utilization_acc, r, RSMI_MAX_NUM_XCC); break;
      }
      case details::AMDGpuMetricAttributeId_t::GFX_BELOW_HOST_LIMIT_TOTAL_ACC: {
        assign_vector(out.xcp_stats[0].gfx_below_host_limit_total_acc, r, RSMI_MAX_NUM_XCC); break;
      }

      default: break;
    }
  }

  out.current_gfxclk = out.current_gfxclks[0];
  out.current_socclk = out.current_socclks[0];
  out.current_vclk0 = out.current_vclk0s[0];
  out.current_vclk1 = out.current_vclk0s[1];
  out.current_dclk0 = out.current_dclk0s[0];
  out.current_dclk1 = out.current_dclk0s[1];

  ss << __PRETTY_FUNCTION__ << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code) << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, out);
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v18_t::copy_internal_to_external_metrics() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Logic below:
    // Default path (::kDevGpuMetrics / !m_is_partition_metrics):
    //             /sys/class/drm/renderDXXX/device/gpu_metrics
    // Partition Path (::kDevGpuMetrics / m_is_partition_metrics):
    //            /sys/class/drm/renderDXXX/device/xcp/xcp_metrics
    // 1. If ::kDevGpuMetrics, copy all data from primary gpu_metrics file.
    // 2. If ::kdevGpuPartitionMetrics, copy data from xcp_metrics file.
    // 3. Provide any backwards compatibility changes

    if (!m_is_partition_metrics) {
      // Header
      metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
      metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
      metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;

      // Temperature
      metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
      metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
      metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;

      // Power
      metrics_public_init.current_socket_power = m_gpu_metrics_tbl.m_current_socket_power;

      // Utilization
      metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
      metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;

      // Power/Energy
      metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

      // Driver attached timestamp (in ns)
      metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

      // Clock Lock Status. Each bit corresponds to clock instance
      metrics_public_init.gfxclk_lock_status = m_gpu_metrics_tbl.m_gfxclk_lock_status;

      // Link width (number of lanes) and speed
      metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
      metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

      // XGMI bus width and bitrate
      metrics_public_init.xgmi_link_width = m_gpu_metrics_tbl.m_xgmi_link_width;
      metrics_public_init.xgmi_link_speed = m_gpu_metrics_tbl.m_xgmi_link_speed;

      // Utilization Accumulated
      metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
      metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

      // PCIE accumulated bandwidth
      metrics_public_init.pcie_bandwidth_acc = m_gpu_metrics_tbl.m_pcie_bandwidth_acc;

      // PCIE instantaneous bandwidth
      metrics_public_init.pcie_bandwidth_inst = m_gpu_metrics_tbl.m_pcie_bandwidth_inst;

      // PCIE L0 to recovery state transition accumulated count
      metrics_public_init.pcie_l0_to_recov_count_acc = m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc;

      // PCIE replay accumulated count
      metrics_public_init.pcie_replay_count_acc = m_gpu_metrics_tbl.m_pcie_replay_count_acc;

      // PCIE replay rollover accumulated count
      metrics_public_init.pcie_replay_rover_count_acc = m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc;

      // PCIE NAK sent accumulated count
      metrics_public_init.pcie_nak_sent_count_acc = m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc;

      // PCIE NAK received accumulated count
      metrics_public_init.pcie_nak_rcvd_count_acc = m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc;

      // Accumulated throttler residencies
      // bumped up public to uint64_t due to planned size increase for newer ASICs
      metrics_public_init.accumulation_counter = m_gpu_metrics_tbl.m_accumulation_counter;
      metrics_public_init.prochot_residency_acc = m_gpu_metrics_tbl.m_prochot_residency_acc;
      metrics_public_init.ppt_residency_acc = m_gpu_metrics_tbl.m_ppt_residency_acc;
      metrics_public_init.socket_thm_residency_acc = m_gpu_metrics_tbl.m_socket_thm_residency_acc;
      metrics_public_init.vr_thm_residency_acc = m_gpu_metrics_tbl.m_vr_thm_residency_acc;
      metrics_public_init.hbm_thm_residency_acc = m_gpu_metrics_tbl.m_hbm_thm_residency_acc;

      /* VRAM max bandwidth at max memory clock */
      metrics_public_init.vram_max_bandwidth = m_gpu_metrics_tbl.m_mem_max_bandwidth;

      // XGMI accumulated data transfer size
      // xgmi_read_data
      const auto xgmi_read_data_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_xgmi_read_data_acc) -
          std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc),
                  xgmi_read_data_num_elems,
                  metrics_public_init.xgmi_read_data_acc);
      // xgmi_write_data
      const auto xgmi_write_data_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_xgmi_write_data_acc) -
          std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc),
                  xgmi_write_data_num_elems,
                  metrics_public_init.xgmi_write_data_acc);

      // xgmi_link_status // new for 1.7
      const auto xgmi_link_status_num_elems = static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_xgmi_link_status) -
          std::begin(m_gpu_metrics_tbl.m_xgmi_link_status));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_link_status),
                  xgmi_link_status_num_elems,
                  metrics_public_init.xgmi_link_status);

      // PMFW attached timestamp (10ns resolution)
      metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

      // Current clocks
      // current_gfxclk
      const auto curr_gfxclk_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_current_gfxclk) -
          std::begin(m_gpu_metrics_tbl.m_current_gfxclk));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_gfxclk),
                  curr_gfxclk_num_elems,
                  metrics_public_init.current_gfxclks);

      // current_socclk
      const auto curr_socclk_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_current_socclk) -
          std::begin(m_gpu_metrics_tbl.m_current_socclk));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_socclk),
                  curr_socclk_num_elems,
                  metrics_public_init.current_socclks);

      // current_vclk0
      const auto curr_vclk0_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_current_vclk0) -
          std::begin(m_gpu_metrics_tbl.m_current_vclk0));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_vclk0),
                  curr_vclk0_num_elems,
                  metrics_public_init.current_vclk0s);

      // current_dclk0
      const auto curr_dclk0_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_tbl.m_current_dclk0) -
          std::begin(m_gpu_metrics_tbl.m_current_dclk0));
      std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_dclk0),
                  curr_dclk0_num_elems,
                  metrics_public_init.current_dclk0s);

      metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;

      metrics_public_init.num_partition = m_gpu_metrics_tbl.m_num_partition;

      metrics_public_init.pcie_lc_perf_other_end_recovery =
        m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery;

      // xcp stats
      auto priv_it = std::begin(m_gpu_metrics_tbl.m_xcp_stats);
      for (auto pub_it = std::begin(metrics_public_init.xcp_stats);
          pub_it != std::end(metrics_public_init.xcp_stats); ++pub_it, ++priv_it) {
        std::copy_n(std::begin(priv_it->gfx_busy_inst), RSMI_MAX_NUM_XCC, pub_it->gfx_busy_inst);
        std::copy_n(std::begin(priv_it->jpeg_busy), RSMI_MAX_NUM_JPEG_ENG_V1, pub_it->jpeg_busy);
        std::copy_n(std::begin(priv_it->vcn_busy), RSMI_MAX_NUM_VCNS, pub_it->vcn_busy);
        std::copy_n(std::begin(priv_it->gfx_busy_acc), RSMI_MAX_NUM_XCC, pub_it->gfx_busy_acc);
        std::copy_n(std::begin(priv_it->gfx_below_host_limit_ppt_acc), RSMI_MAX_NUM_XCC,
                    pub_it->gfx_below_host_limit_ppt_acc);
        std::copy_n(std::begin(priv_it->gfx_below_host_limit_thm_acc), RSMI_MAX_NUM_XCC,
                    pub_it->gfx_below_host_limit_thm_acc);
        std::copy_n(std::begin(priv_it->gfx_low_utilization_acc), RSMI_MAX_NUM_XCC,
                    pub_it->gfx_low_utilization_acc);
        std::copy_n(std::begin(priv_it->gfx_below_host_limit_total_acc), RSMI_MAX_NUM_XCC,
                    pub_it->gfx_below_host_limit_total_acc);
      }
    } else {
      // Partition Data: /sys/class/drm/renderDXXX/device/xcp/xcp_metrics
      // Copy common data from xcp metrics table

      // Header
      metrics_public_init.common_header.structure_size = m_gpu_metrics_partition_tbl.m_common_header.m_structure_size;
      metrics_public_init.common_header.format_revision = m_gpu_metrics_partition_tbl.m_common_header.m_format_revision;
      metrics_public_init.common_header.content_revision = m_gpu_metrics_partition_tbl.m_common_header.m_content_revision;

      // Current clocks
      // current_gfxclk
      const auto curr_gfxclk_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_partition_tbl.m_current_gfxclk) -
          std::begin(m_gpu_metrics_partition_tbl.m_current_gfxclk));
      std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_current_gfxclk),
                  curr_gfxclk_num_elems,
                  metrics_public_init.current_gfxclks);

      // current_socclk
      const auto curr_socclk_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_partition_tbl.m_current_socclk) -
          std::begin(m_gpu_metrics_partition_tbl.m_current_socclk));
      std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_current_socclk),
                  curr_socclk_num_elems,
                  metrics_public_init.current_socclks);

      // current_vclk0
      const auto curr_vclk0_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_partition_tbl.m_current_vclk0) -
          std::begin(m_gpu_metrics_partition_tbl.m_current_vclk0));
      std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_current_vclk0),
                  curr_vclk0_num_elems,
                  metrics_public_init.current_vclk0s);

      // current_dclk0
      const auto curr_dclk0_num_elems =
        static_cast<uint16_t>(
          std::end(m_gpu_metrics_partition_tbl.m_current_dclk0) -
          std::begin(m_gpu_metrics_partition_tbl.m_current_dclk0));
      std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_current_dclk0),
                  curr_dclk0_num_elems,
                  metrics_public_init.current_dclk0s);
      metrics_public_init.current_uclk = m_gpu_metrics_partition_tbl.m_current_uclk;

      // Copy data from partition table to public table
      // Only copy data for xcp #0
      // Partition metrics should default to XCP #0 position, since we can gather only one partition
      // metrics at a time.
      const uint32_t xcp_num = 0;
      uint32_t row = 0;
      for (auto it = std::begin(metrics_public_init.xcp_stats);
           it != std::end(metrics_public_init.xcp_stats); ++it, ++row) {
        if (row == xcp_num) {
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_busy_inst),
                      RSMI_MAX_NUM_XCC, it->gfx_busy_inst);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_jpeg_busy),
                      RSMI_MAX_NUM_JPEG_ENG_V1, it->jpeg_busy);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_vcn_busy),
                      RSMI_MAX_NUM_VCNS, it->vcn_busy);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_busy_acc),
                      RSMI_MAX_NUM_XCC, it->gfx_busy_acc);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_ppt_acc),
                      RSMI_MAX_NUM_XCC, it->gfx_below_host_limit_ppt_acc);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_thm_acc),
                      RSMI_MAX_NUM_XCC, it->gfx_below_host_limit_thm_acc);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_low_utilization_acc),
                      RSMI_MAX_NUM_XCC, it->gfx_low_utilization_acc);
          std::copy_n(std::begin(m_gpu_metrics_partition_tbl.m_gfx_below_host_limit_total_acc),
                      RSMI_MAX_NUM_XCC, it->gfx_below_host_limit_total_acc);
        } else {
          break;  // No need to copy for other rows
        }
      }
    }

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.3/1.4/1.5)
    metrics_public_init.current_gfxclk = metrics_public_init.current_gfxclks[0];

    metrics_public_init.current_socclk = metrics_public_init.current_socclks[0];

    metrics_public_init.current_vclk0 = metrics_public_init.current_vclk0s[0];

    metrics_public_init.current_vclk1 = metrics_public_init.current_vclk0s[1];

    metrics_public_init.current_dclk0 = metrics_public_init.current_dclk0s[0];

    metrics_public_init.current_dclk1 = metrics_public_init.current_dclk0s[1];

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code)
     << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v17_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;


    // Temperature
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;

    // Power
    metrics_public_init.current_socket_power = m_gpu_metrics_tbl.m_current_socket_power;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;

    // Power/Energy
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Clock Lock Status. Each bit corresponds to clock instance
    metrics_public_init.gfxclk_lock_status = m_gpu_metrics_tbl.m_gfxclk_lock_status;

    // Link width (number of lanes) and speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    // XGMI bus width and bitrate
    metrics_public_init.xgmi_link_width = m_gpu_metrics_tbl.m_xgmi_link_width;
    metrics_public_init.xgmi_link_speed = m_gpu_metrics_tbl.m_xgmi_link_speed;

    // Utilization Accumulated
    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // PCIE accumulated bandwidth
    metrics_public_init.pcie_bandwidth_acc = m_gpu_metrics_tbl.m_pcie_bandwidth_acc;

    // PCIE instantaneous bandwidth
    metrics_public_init.pcie_bandwidth_inst = m_gpu_metrics_tbl.m_pcie_bandwidth_inst;

    // PCIE L0 to recovery state transition accumulated count
    metrics_public_init.pcie_l0_to_recov_count_acc = m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc;

    // PCIE replay accumulated count
    metrics_public_init.pcie_replay_count_acc = m_gpu_metrics_tbl.m_pcie_replay_count_acc;

    // PCIE replay rollover accumulated count
    metrics_public_init.pcie_replay_rover_count_acc = m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc;

    // PCIE NAK sent accumulated count
    metrics_public_init.pcie_nak_sent_count_acc = m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc;

    // PCIE NAK received accumulated count
    metrics_public_init.pcie_nak_rcvd_count_acc = m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc;

    // Accumulated throttler residencies
    // bumped up public to uint64_t due to planned size increase for newer ASICs
    metrics_public_init.accumulation_counter = m_gpu_metrics_tbl.m_accumulation_counter;
    metrics_public_init.prochot_residency_acc = m_gpu_metrics_tbl.m_prochot_residency_acc;
    metrics_public_init.ppt_residency_acc = m_gpu_metrics_tbl.m_ppt_residency_acc;
    metrics_public_init.socket_thm_residency_acc = m_gpu_metrics_tbl.m_socket_thm_residency_acc;
    metrics_public_init.vr_thm_residency_acc = m_gpu_metrics_tbl.m_vr_thm_residency_acc;
    metrics_public_init.hbm_thm_residency_acc = m_gpu_metrics_tbl.m_hbm_thm_residency_acc;

    /* VRAM max bandwidth at max memory clock */
    metrics_public_init.vram_max_bandwidth = m_gpu_metrics_tbl.m_vram_max_bandwidth;

    // XGMI accumulated data transfer size
    // xgmi_read_data
    const auto xgmi_read_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_read_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc),
                xgmi_read_data_num_elems,
                metrics_public_init.xgmi_read_data_acc);
    // xgmi_write_data
    const auto xgmi_write_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_write_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc),
                xgmi_write_data_num_elems,
                metrics_public_init.xgmi_write_data_acc);

    // xgmi_link_status // new for 1.7
    const auto xgmi_link_status_num_elems = static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_link_status) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_link_status));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_link_status),
                xgmi_link_status_num_elems,
                metrics_public_init.xgmi_link_status);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    // Current clocks
    // current_gfxclk
    const auto curr_gfxclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_gfxclk) -
        std::begin(m_gpu_metrics_tbl.m_current_gfxclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_gfxclk),
                curr_gfxclk_num_elems,
                metrics_public_init.current_gfxclks);

    // current_socclk
    const auto curr_socclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_socclk) -
        std::begin(m_gpu_metrics_tbl.m_current_socclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_socclk),
                curr_socclk_num_elems,
                metrics_public_init.current_socclks);

    // current_vclk0
    const auto curr_vclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_vclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_vclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_vclk0),
                curr_vclk0_num_elems,
                metrics_public_init.current_vclk0s);

    // current_dclk0
    const auto curr_dclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_dclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_dclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_dclk0),
                curr_dclk0_num_elems,
                metrics_public_init.current_dclk0s);

    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;

    metrics_public_init.num_partition = m_gpu_metrics_tbl.m_num_partition;

    metrics_public_init.pcie_lc_perf_other_end_recovery =
      m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery;

    auto priv_it = std::begin(m_gpu_metrics_tbl.m_xcp_stats);
    for (auto pub_it = std::begin(metrics_public_init.xcp_stats);
         pub_it != std::end(metrics_public_init.xcp_stats);
         ++pub_it, ++priv_it) {
      std::copy_n(std::begin(priv_it->gfx_busy_inst), RSMI_MAX_NUM_XCC,
                  pub_it->gfx_busy_inst);
      std::copy_n(std::begin(priv_it->jpeg_busy), RSMI_MAX_NUM_JPEG_ENGS,
                  pub_it->jpeg_busy);
      std::copy_n(std::begin(priv_it->vcn_busy), RSMI_MAX_NUM_VCNS,
                  pub_it->vcn_busy);
      std::copy_n(std::begin(priv_it->gfx_busy_acc), RSMI_MAX_NUM_XCC,
                  pub_it->gfx_busy_acc);
      std::copy_n(std::begin(priv_it->gfx_below_host_limit_acc), RSMI_MAX_NUM_XCC,
                  pub_it->gfx_below_host_limit_acc);
    }

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.3/1.4/1.5)
    metrics_public_init.current_gfxclk = metrics_public_init.current_gfxclks[0];

    metrics_public_init.current_socclk = metrics_public_init.current_socclks[0];

    metrics_public_init.current_vclk0 = metrics_public_init.current_vclk0s[0];

    metrics_public_init.current_vclk1 = metrics_public_init.current_vclk0s[1];

    metrics_public_init.current_dclk0 = metrics_public_init.current_dclk0s[0];

    metrics_public_init.current_dclk1 = metrics_public_init.current_dclk0s[1];

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code)
     << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);

}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v16_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;


    // Temperature
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;

    // Power
    metrics_public_init.current_socket_power = m_gpu_metrics_tbl.m_current_socket_power;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;

    // Power/Energy
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Clock Lock Status. Each bit corresponds to clock instance
    metrics_public_init.gfxclk_lock_status = m_gpu_metrics_tbl.m_gfxclk_lock_status;

    // Link width (number of lanes) and speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    // XGMI bus width and bitrate
    metrics_public_init.xgmi_link_width = m_gpu_metrics_tbl.m_xgmi_link_width;
    metrics_public_init.xgmi_link_speed = m_gpu_metrics_tbl.m_xgmi_link_speed;

    // Utilization Accumulated
    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // PCIE accumulated bandwidth
    metrics_public_init.pcie_bandwidth_acc = m_gpu_metrics_tbl.m_pcie_bandwidth_acc;

    // PCIE instantaneous bandwidth
    metrics_public_init.pcie_bandwidth_inst = m_gpu_metrics_tbl.m_pcie_bandwidth_inst;

    // PCIE L0 to recovery state transition accumulated count
    metrics_public_init.pcie_l0_to_recov_count_acc = m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc;

    // PCIE replay accumulated count
    metrics_public_init.pcie_replay_count_acc = m_gpu_metrics_tbl.m_pcie_replay_count_acc;

    // PCIE replay rollover accumulated count
    metrics_public_init.pcie_replay_rover_count_acc = m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc;

    // PCIE NAK sent accumulated count
    metrics_public_init.pcie_nak_sent_count_acc = m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc;

    // PCIE NAK received accumulated count
    metrics_public_init.pcie_nak_rcvd_count_acc = m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc;

    // Accumulated throttler residencies
    // bumped up public to uint64_t due to planned size increase for newer ASICs
    metrics_public_init.accumulation_counter = m_gpu_metrics_tbl.m_accumulation_counter;
    metrics_public_init.prochot_residency_acc = m_gpu_metrics_tbl.m_prochot_residency_acc;
    metrics_public_init.ppt_residency_acc = m_gpu_metrics_tbl.m_ppt_residency_acc;
    metrics_public_init.socket_thm_residency_acc = m_gpu_metrics_tbl.m_socket_thm_residency_acc;
    metrics_public_init.vr_thm_residency_acc = m_gpu_metrics_tbl.m_vr_thm_residency_acc;
    metrics_public_init.hbm_thm_residency_acc = m_gpu_metrics_tbl.m_hbm_thm_residency_acc;

    // XGMI accumulated data transfer size
    // xgmi_read_data
    const auto xgmi_read_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_read_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc),
                xgmi_read_data_num_elems,
                metrics_public_init.xgmi_read_data_acc);
    // xgmi_write_data
    const auto xgmi_write_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_write_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc),
                xgmi_write_data_num_elems,
                metrics_public_init.xgmi_write_data_acc);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    // Current clocks
    // current_gfxclk
    const auto curr_gfxclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_gfxclk) -
        std::begin(m_gpu_metrics_tbl.m_current_gfxclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_gfxclk),
                curr_gfxclk_num_elems,
                metrics_public_init.current_gfxclks);

    // current_socclk
    const auto curr_socclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_socclk) -
        std::begin(m_gpu_metrics_tbl.m_current_socclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_socclk),
                curr_socclk_num_elems,
                metrics_public_init.current_socclks);

    // current_vclk0
    const auto curr_vclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_vclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_vclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_vclk0),
                curr_vclk0_num_elems,
                metrics_public_init.current_vclk0s);

    // current_dclk0
    const auto curr_dclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_dclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_dclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_dclk0),
                curr_dclk0_num_elems,
                metrics_public_init.current_dclk0s);

    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;

    metrics_public_init.num_partition = m_gpu_metrics_tbl.m_num_partition;

    metrics_public_init.pcie_lc_perf_other_end_recovery =
      m_gpu_metrics_tbl.m_pcie_lc_perf_other_end_recovery;

    auto priv_it = std::begin(m_gpu_metrics_tbl.m_xcp_stats);
    for (auto pub_it = std::begin(metrics_public_init.xcp_stats);
         pub_it != std::end(metrics_public_init.xcp_stats);
         ++pub_it, ++priv_it) {
      std::copy_n(std::begin(priv_it->gfx_busy_inst), RSMI_MAX_NUM_XCC,
                  pub_it->gfx_busy_inst);
      std::copy_n(std::begin(priv_it->jpeg_busy), RSMI_MAX_NUM_JPEG_ENGS,
                  pub_it->jpeg_busy);
      std::copy_n(std::begin(priv_it->vcn_busy), RSMI_MAX_NUM_VCNS,
                  pub_it->vcn_busy);
      std::copy_n(std::begin(priv_it->gfx_busy_acc), RSMI_MAX_NUM_XCC,
                  pub_it->gfx_busy_acc);
    }

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.3/1.4/1.5)
    metrics_public_init.current_gfxclk = metrics_public_init.current_gfxclks[0];

    metrics_public_init.current_socclk = metrics_public_init.current_socclks[0];

    metrics_public_init.current_vclk0 = metrics_public_init.current_vclk0s[0];

    metrics_public_init.current_vclk1 = metrics_public_init.current_vclk0s[1];

    metrics_public_init.current_dclk0 = metrics_public_init.current_dclk0s[0];

    metrics_public_init.current_dclk1 = metrics_public_init.current_dclk0s[1];

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Returning = " << getRSMIStatusString(status_code)
     << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v15_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;


    // Temperature
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;

    // Power
    metrics_public_init.current_socket_power = m_gpu_metrics_tbl.m_current_socket_power;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;

    // vcn_activity
    const auto vcn_activity_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_vcn_activity) -
        std::begin(m_gpu_metrics_tbl.m_vcn_activity));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_vcn_activity),
                vcn_activity_num_elems,
                metrics_public_init.vcn_activity);

    // jpeg_activity
    const auto jpeg_activity_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_jpeg_activity) -
        std::begin(m_gpu_metrics_tbl.m_jpeg_activity));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_jpeg_activity),
                jpeg_activity_num_elems,
                metrics_public_init.jpeg_activity);

    // Power/Energy
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Throttle status
    metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

    // Clock Lock Status. Each bit corresponds to clock instance
    metrics_public_init.gfxclk_lock_status = m_gpu_metrics_tbl.m_gfxclk_lock_status;

    // Link width (number of lanes) and speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    // XGMI bus width and bitrate
    metrics_public_init.xgmi_link_width = m_gpu_metrics_tbl.m_xgmi_link_width;
    metrics_public_init.xgmi_link_speed = m_gpu_metrics_tbl.m_xgmi_link_speed;

    // Utilization Accumulated
    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // PCIE accumulated bandwidth
    metrics_public_init.pcie_bandwidth_acc = m_gpu_metrics_tbl.m_pcie_bandwidth_acc;

    // PCIE instantaneous bandwidth
    metrics_public_init.pcie_bandwidth_inst = m_gpu_metrics_tbl.m_pcie_bandwidth_inst;

    // PCIE L0 to recovery state transition accumulated count
    metrics_public_init.pcie_l0_to_recov_count_acc = m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc;

    // PCIE replay accumulated count
    metrics_public_init.pcie_replay_count_acc = m_gpu_metrics_tbl.m_pcie_replay_count_acc;

    // PCIE replay rollover accumulated count
    metrics_public_init.pcie_replay_rover_count_acc = m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc;

    // PCIE NAK sent accumulated count
    metrics_public_init.pcie_nak_sent_count_acc = m_gpu_metrics_tbl.m_pcie_nak_sent_count_acc;

    // PCIE NAK received accumulated count
    metrics_public_init.pcie_nak_rcvd_count_acc = m_gpu_metrics_tbl.m_pcie_nak_rcvd_count_acc;

    // XGMI accumulated data transfer size
    // xgmi_read_data
    const auto xgmi_read_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_read_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc),
                xgmi_read_data_num_elems,
                metrics_public_init.xgmi_read_data_acc);
    // xgmi_write_data
    const auto xgmi_write_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_write_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc),
                xgmi_write_data_num_elems,
                metrics_public_init.xgmi_write_data_acc);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    // Current clocks
    // current_gfxclk
    const auto curr_gfxclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_gfxclk) -
        std::begin(m_gpu_metrics_tbl.m_current_gfxclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_gfxclk),
                curr_gfxclk_num_elems,
                metrics_public_init.current_gfxclks);

    // current_socclk
    const auto curr_socclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_socclk) -
        std::begin(m_gpu_metrics_tbl.m_current_socclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_socclk),
                curr_socclk_num_elems,
                metrics_public_init.current_socclks);

    // current_vclk0
    const auto curr_vclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_vclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_vclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_vclk0),
                curr_vclk0_num_elems,
                metrics_public_init.current_vclk0s);

    // current_dclk0
    const auto curr_dclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_dclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_dclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_dclk0),
                curr_dclk0_num_elems,
                metrics_public_init.current_dclk0s);

    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.3)
    metrics_public_init.current_gfxclk = metrics_public_init.current_gfxclks[0];
    // metrics_public_init.average_gfxclk_frequency = metrics_public_init.current_gfxclks[0];

    metrics_public_init.current_socclk = metrics_public_init.current_socclks[0];
    // metrics_public_init.average_socclk_frequency = metrics_public_init.current_socclks[0];

    metrics_public_init.current_vclk0 = metrics_public_init.current_vclk0s[0];
    // metrics_public_init.average_vclk0_frequency = metrics_public_init.current_vclk0s[0];

    metrics_public_init.current_vclk1 = metrics_public_init.current_vclk0s[1];
    // metrics_public_init.average_vclk1_frequency = metrics_public_init.current_vclk0s[1];

    metrics_public_init.current_dclk0 = metrics_public_init.current_dclk0s[0];
    // metrics_public_init.average_dclk0_frequency = metrics_public_init.current_dclk0s[0];

    metrics_public_init.current_dclk1 = metrics_public_init.current_dclk0s[1];
    // metrics_public_init.average_dclk1_frequency = metrics_public_init.current_dclk0s[1];

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v14_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;


    // Temperature
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;

    // Power
    metrics_public_init.current_socket_power = m_gpu_metrics_tbl.m_current_socket_power;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;

    // vcn_activity
    const auto vcn_activity_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_vcn_activity) -
        std::begin(m_gpu_metrics_tbl.m_vcn_activity));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_vcn_activity),
                vcn_activity_num_elems,
                metrics_public_init.vcn_activity);

    // Power/Energy
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Throttle status
    metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

    // Clock Lock Status. Each bit corresponds to clock instance
    metrics_public_init.gfxclk_lock_status = m_gpu_metrics_tbl.m_gfxclk_lock_status;

    // Link width (number of lanes) and speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    // XGMI bus width and bitrate
    metrics_public_init.xgmi_link_width = m_gpu_metrics_tbl.m_xgmi_link_width;
    metrics_public_init.xgmi_link_speed = m_gpu_metrics_tbl.m_xgmi_link_speed;

    // Utilization Accumulated
    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // PCIE accumulated bandwidth
    metrics_public_init.pcie_bandwidth_acc = m_gpu_metrics_tbl.m_pcie_bandwidth_acc;

    // PCIE instantaneous bandwidth
    metrics_public_init.pcie_bandwidth_inst = m_gpu_metrics_tbl.m_pcie_bandwidth_inst;

    // PCIE L0 to recovery state transition accumulated count
    metrics_public_init.pcie_l0_to_recov_count_acc = m_gpu_metrics_tbl.m_pcie_l0_to_recov_count_acc;

    // PCIE replay accumulated count
    metrics_public_init.pcie_replay_count_acc = m_gpu_metrics_tbl.m_pcie_replay_count_acc;

    // PCIE replay rollover accumulated count
    metrics_public_init.pcie_replay_rover_count_acc = m_gpu_metrics_tbl.m_pcie_replay_rover_count_acc;

    // XGMI accumulated data transfer size
    // xgmi_read_data
    const auto xgmi_read_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_read_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_read_data_acc),
                xgmi_read_data_num_elems,
                metrics_public_init.xgmi_read_data_acc);
    // xgmi_write_data
    const auto xgmi_write_data_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_xgmi_write_data_acc) -
        std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_xgmi_write_data_acc),
                xgmi_write_data_num_elems,
                metrics_public_init.xgmi_write_data_acc);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    // Current clocks
    // current_gfxclk
    const auto curr_gfxclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_gfxclk) -
        std::begin(m_gpu_metrics_tbl.m_current_gfxclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_gfxclk),
                curr_gfxclk_num_elems,
                metrics_public_init.current_gfxclks);

    // current_socclk
    const auto curr_socclk_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_socclk) -
        std::begin(m_gpu_metrics_tbl.m_current_socclk));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_socclk),
                curr_socclk_num_elems,
                metrics_public_init.current_socclks);

    // current_vclk0
    const auto curr_vclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_vclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_vclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_vclk0),
                curr_vclk0_num_elems,
                metrics_public_init.current_vclk0s);

    // current_dclk0
    const auto curr_dclk0_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_current_dclk0) -
        std::begin(m_gpu_metrics_tbl.m_current_dclk0));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_current_dclk0),
                curr_dclk0_num_elems,
                metrics_public_init.current_dclk0s);

    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.3)
    metrics_public_init.current_gfxclk = metrics_public_init.current_gfxclks[0];
    // metrics_public_init.average_gfxclk_frequency = metrics_public_init.current_gfxclks[0];

    metrics_public_init.current_socclk = metrics_public_init.current_socclks[0];
    // metrics_public_init.average_socclk_frequency = metrics_public_init.current_socclks[0];

    metrics_public_init.current_vclk0 = metrics_public_init.current_vclk0s[0];
    // metrics_public_init.average_vclk0_frequency = metrics_public_init.current_vclk0s[0];

    metrics_public_init.current_vclk1 = metrics_public_init.current_vclk0s[1];
    // metrics_public_init.average_vclk1_frequency = metrics_public_init.current_vclk0s[1];

    metrics_public_init.current_dclk0 = metrics_public_init.current_dclk0s[0];
    // metrics_public_init.average_dclk0_frequency = metrics_public_init.current_dclk0s[0];

    metrics_public_init.current_dclk1 = metrics_public_init.current_dclk0s[1];
    // metrics_public_init.average_dclk1_frequency = metrics_public_init.current_dclk0s[1];

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}


rsmi_status_t GpuMetricsBase_v13_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v13 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };


  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v13();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempEdge,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_edge,
                                "temperature_edge"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrGfx,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrgfx,
                                "temperature_vrgfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrmem,
                                "temperature_vrmem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHbm,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hbm,
                                "[temperature_hbm]"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_average_socket_power,
                                "average_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgMmActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_mm_activity,
                                "average_mm_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
                                "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Fan Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_current_fan_speed,
                                "current_fan_speed"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricIndepThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_indep_throttle_status,
                                "indep_throttle_status"))
           );

  // Average Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfxclk_frequency,
                                "average_gfxclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_socclk_frequency,
                                "average_socclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_uclk_frequency,
                                "average_uclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk0_frequency,
                                "average_vclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk0_frequency,
                                "average_dclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk1_frequency,
                                "average_vclk1_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk1_frequency,
                                "average_dclk1_frequency"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "current_gfxclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "current_socclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "current_vclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "current_dclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk1,
                                "current_vclk1"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk1,
                                "current_dclk1"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );

  // Voltage Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageSoc,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_soc,
                                "voltage_soc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageGfx,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_gfx,
                                "voltage_gfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageMem,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_mem,
                                "voltage_mem"))
           );

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v13_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;

    // Temperature
    metrics_public_init.temperature_edge = m_gpu_metrics_tbl.m_temperature_edge;
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrgfx = m_gpu_metrics_tbl.m_temperature_vrgfx;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;
    metrics_public_init.temperature_vrmem = m_gpu_metrics_tbl.m_temperature_vrmem;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;
    metrics_public_init.average_mm_activity = m_gpu_metrics_tbl.m_average_mm_activity;

    // Power/Energy
    metrics_public_init.average_socket_power = m_gpu_metrics_tbl.m_average_socket_power;  // 1.3 and 1.4 have the same value
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Average clocks
    metrics_public_init.average_gfxclk_frequency = m_gpu_metrics_tbl.m_average_gfxclk_frequency;
    metrics_public_init.average_socclk_frequency = m_gpu_metrics_tbl.m_average_socclk_frequency;
    metrics_public_init.average_uclk_frequency = m_gpu_metrics_tbl.m_average_uclk_frequency;
    metrics_public_init.average_vclk0_frequency = m_gpu_metrics_tbl.m_average_vclk0_frequency;
    metrics_public_init.average_dclk0_frequency = m_gpu_metrics_tbl.m_average_dclk0_frequency;
    metrics_public_init.average_vclk1_frequency = m_gpu_metrics_tbl.m_average_vclk1_frequency;
    metrics_public_init.average_dclk1_frequency = m_gpu_metrics_tbl.m_average_dclk1_frequency;

    // Current clocks
    metrics_public_init.current_gfxclk = m_gpu_metrics_tbl.m_current_gfxclk;
    metrics_public_init.current_gfxclks[0] = m_gpu_metrics_tbl.m_current_gfxclk;
    metrics_public_init.current_socclk = m_gpu_metrics_tbl.m_current_socclk;
    metrics_public_init.current_socclks[0] = m_gpu_metrics_tbl.m_current_socclk;
    metrics_public_init.current_vclk0 = m_gpu_metrics_tbl.m_current_vclk0;
    metrics_public_init.current_vclk0s[0] = m_gpu_metrics_tbl.m_current_vclk0;
    metrics_public_init.current_dclk0 = m_gpu_metrics_tbl.m_current_dclk0;
    metrics_public_init.current_dclk0s[0] = m_gpu_metrics_tbl.m_current_dclk0;
    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;
    metrics_public_init.current_vclk1 = m_gpu_metrics_tbl.m_current_vclk1;
    metrics_public_init.current_dclk1 = m_gpu_metrics_tbl.m_current_dclk1;

    // Throttle status
    metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

    // Fans
    metrics_public_init.current_fan_speed = m_gpu_metrics_tbl.m_current_fan_speed;

    // Link width/speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // temperature_hbm
    const auto temp_hbm_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_temperature_hbm) -
        std::begin(m_gpu_metrics_tbl.m_temperature_hbm));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_temperature_hbm),
                temp_hbm_num_elems,
                metrics_public_init.temperature_hbm);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    // Voltage (mV)
    metrics_public_init.voltage_soc = m_gpu_metrics_tbl.m_voltage_soc;
    metrics_public_init.voltage_gfx = m_gpu_metrics_tbl.m_voltage_gfx;
    metrics_public_init.voltage_mem = m_gpu_metrics_tbl.m_voltage_mem;

    // Throttle status
    metrics_public_init.indep_throttle_status = m_gpu_metrics_tbl.m_indep_throttle_status;

    //
    // Note:  Forwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.2)
    // metrics_public_init.current_socket_power = metrics_public_init.average_socket_power;
    // average_mm_activity needs to not be UIN16_MAX and
    // metrics_public_init.vcn_activity[0] should also be UIN16_MAX
    if (metrics_public_init.average_mm_activity != UINT16_MAX
        && metrics_public_init.vcn_activity[0] == UINT16_MAX) {
      metrics_public_init.vcn_activity[0] = metrics_public_init.average_mm_activity;
    }

    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

rsmi_status_t GpuMetricsBase_v12_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v12 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);

    // firmware_timestamp is at 10ns resolution
    ss << __PRETTY_FUNCTION__
                << " | ======= Changes ======= "
                << " | {m_firmware_timestamp} from: " << m_gpu_metrics_tbl.m_firmware_timestamp
                << " to: " << (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    m_gpu_metrics_tbl.m_firmware_timestamp = (m_gpu_metrics_tbl.m_firmware_timestamp * 10);
    LOG_DEBUG(ss);
  };


  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v12();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempEdge,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_edge,
                                "temperature_edge"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrGfx,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrgfx,
                                "temperature_vrgfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrmem,
                                "temperature_vrmem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHbm,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hbm,
                                "[temperature_hbm]"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_average_socket_power,
                                "average_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgMmActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_mm_activity,
                                "average_mm_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
                                "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Fan Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_current_fan_speed,
                                "current_fan_speed"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );

  // Average Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfxclk_frequency,
                                "average_gfxclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_socclk_frequency,
                                "average_socclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_uclk_frequency,
                                "average_uclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk0_frequency,
                                "average_vclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk0_frequency,
                                "average_dclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk1_frequency,
                                "average_vclk1_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk1_frequency,
                                "average_dclk1_frequency"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "current_gfxclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "current_socclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "current_vclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "current_dclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk1,
                                "current_vclk1"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk1,
                                "current_dclk1"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v12_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;

    // Temperature
    metrics_public_init.temperature_edge = m_gpu_metrics_tbl.m_temperature_edge;
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrgfx = m_gpu_metrics_tbl.m_temperature_vrgfx;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;
    metrics_public_init.temperature_vrmem = m_gpu_metrics_tbl.m_temperature_vrmem;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;
    metrics_public_init.average_mm_activity = m_gpu_metrics_tbl.m_average_mm_activity;

    // Power/Energy
    metrics_public_init.average_socket_power = m_gpu_metrics_tbl.m_average_socket_power;
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Average clocks
    metrics_public_init.average_gfxclk_frequency = m_gpu_metrics_tbl.m_average_gfxclk_frequency;
    metrics_public_init.average_socclk_frequency = m_gpu_metrics_tbl.m_average_socclk_frequency;
    metrics_public_init.average_uclk_frequency = m_gpu_metrics_tbl.m_average_uclk_frequency;
    metrics_public_init.average_vclk0_frequency = m_gpu_metrics_tbl.m_average_vclk0_frequency;
    metrics_public_init.average_dclk0_frequency = m_gpu_metrics_tbl.m_average_dclk0_frequency;
    metrics_public_init.average_vclk1_frequency = m_gpu_metrics_tbl.m_average_vclk1_frequency;
    metrics_public_init.average_dclk1_frequency = m_gpu_metrics_tbl.m_average_dclk1_frequency;

    // Current clocks
    metrics_public_init.current_gfxclk = m_gpu_metrics_tbl.m_current_gfxclk;
    metrics_public_init.current_socclk = m_gpu_metrics_tbl.m_current_socclk;
    metrics_public_init.current_vclk0 = m_gpu_metrics_tbl.m_current_vclk0;
    metrics_public_init.current_dclk0 = m_gpu_metrics_tbl.m_current_dclk0;
    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;
    metrics_public_init.current_vclk1 = m_gpu_metrics_tbl.m_current_vclk1;
    metrics_public_init.current_dclk1 = m_gpu_metrics_tbl.m_current_dclk1;

    // Throttle status
    metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

    // Fans
    metrics_public_init.current_fan_speed = m_gpu_metrics_tbl.m_current_fan_speed;

    // Link width/speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // temperature_hbm
    const auto temp_hbm_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_temperature_hbm) -
        std::begin(m_gpu_metrics_tbl.m_temperature_hbm));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_temperature_hbm),
                temp_hbm_num_elems,
                metrics_public_init.temperature_hbm);

    // PMFW attached timestamp (10ns resolution)
    metrics_public_init.firmware_timestamp = m_gpu_metrics_tbl.m_firmware_timestamp;

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.1)


    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

rsmi_status_t GpuMetricsBase_v11_t::populate_metrics_dynamic_tbl() {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
  //
  //  Note: Any metric treatment/changes (if any) should happen before they
  //        get written to internal/external tables.
  //
  auto run_metric_adjustments_v11 = [&]() {
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
    ss << __PRETTY_FUNCTION__
                << " | ======= info ======= "
                << " | Applying adjustments "
                << " | Metric Version: " << stringfy_metric_header_version(
                                              disjoin_metrics_version(gpu_metrics_version))
                << " |";
    LOG_TRACE(ss);
  };


  //  Adjustments/Changes specific to this version
  run_metric_adjustments_v11();

  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempEdge,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_edge,
                                "temperature_edge"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrGfx,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrgfx,
                                "temperature_vrgfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrmem,
                                "temperature_vrmem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHbm,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hbm,
                                "[temperature_hbm]"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_average_socket_power,
                                "average_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgMmActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_mm_activity,
                                "average_mm_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
                                "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Fan Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_current_fan_speed,
                                "current_fan_speed"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );

  // Average Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfxclk_frequency,
                                "average_gfxclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_socclk_frequency,
                                "average_socclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_uclk_frequency,
                                "average_uclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk0_frequency,
                                "average_vclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk0_frequency,
                                "average_dclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk1_frequency,
                                "average_vclk1_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk1_frequency,
                                "average_dclk1_frequency"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "current_gfxclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "current_socclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "current_vclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "current_dclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk1,
                                "current_vclk1"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk1,
                                "current_dclk1"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  // Copy to base class
  std::copy(m_metrics_dynamic_tbl.begin(),
            m_metrics_dynamic_tbl.end(),
            std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                          GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

  return status_code;
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v11_t::copy_internal_to_external_metrics()
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  auto copy_data_from_internal_metrics_tbl = [&]() {
    AMGpuMetricsPublicLatest_t metrics_public_init{};

    //
    //  Note: Initializing data members with their max. If field is max,
    //        no data was assigned to it.
    init_max_public_gpu_matrics(metrics_public_init);

    // Header
    metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
    metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
    metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;

    // Temperature
    metrics_public_init.temperature_edge = m_gpu_metrics_tbl.m_temperature_edge;
    metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
    metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
    metrics_public_init.temperature_vrgfx = m_gpu_metrics_tbl.m_temperature_vrgfx;
    metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;
    metrics_public_init.temperature_vrmem = m_gpu_metrics_tbl.m_temperature_vrmem;

    // Utilization
    metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
    metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;
    metrics_public_init.average_mm_activity = m_gpu_metrics_tbl.m_average_mm_activity;

    // Power/Energy
    metrics_public_init.average_socket_power = m_gpu_metrics_tbl.m_average_socket_power;
    metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

    // Driver attached timestamp (in ns)
    metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

    // Average clocks
    metrics_public_init.average_gfxclk_frequency = m_gpu_metrics_tbl.m_average_gfxclk_frequency;
    metrics_public_init.average_socclk_frequency = m_gpu_metrics_tbl.m_average_socclk_frequency;
    metrics_public_init.average_uclk_frequency = m_gpu_metrics_tbl.m_average_uclk_frequency;
    metrics_public_init.average_vclk0_frequency = m_gpu_metrics_tbl.m_average_vclk0_frequency;
    metrics_public_init.average_dclk0_frequency = m_gpu_metrics_tbl.m_average_dclk0_frequency;
    metrics_public_init.average_vclk1_frequency = m_gpu_metrics_tbl.m_average_vclk1_frequency;
    metrics_public_init.average_dclk1_frequency = m_gpu_metrics_tbl.m_average_dclk1_frequency;

    // Current clocks
    metrics_public_init.current_gfxclk = m_gpu_metrics_tbl.m_current_gfxclk;
    metrics_public_init.current_socclk = m_gpu_metrics_tbl.m_current_socclk;
    metrics_public_init.current_vclk0 = m_gpu_metrics_tbl.m_current_vclk0;
    metrics_public_init.current_dclk0 = m_gpu_metrics_tbl.m_current_dclk0;
    metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;
    metrics_public_init.current_vclk1 = m_gpu_metrics_tbl.m_current_vclk1;
    metrics_public_init.current_dclk1 = m_gpu_metrics_tbl.m_current_dclk1;

    // Throttle status
    metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

    // Fans
    metrics_public_init.current_fan_speed = m_gpu_metrics_tbl.m_current_fan_speed;

    // Link width/speed
    metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
    metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;

    metrics_public_init.gfx_activity_acc = m_gpu_metrics_tbl.m_gfx_activity_acc;
    metrics_public_init.mem_activity_acc = m_gpu_metrics_tbl.m_mem_activity_acc;

    // temperature_hbm
    const auto temp_hbm_num_elems =
      static_cast<uint16_t>(
        std::end(m_gpu_metrics_tbl.m_temperature_hbm) -
        std::begin(m_gpu_metrics_tbl.m_temperature_hbm));
    std::copy_n(std::begin(m_gpu_metrics_tbl.m_temperature_hbm),
                temp_hbm_num_elems,
                metrics_public_init.temperature_hbm);

    //
    // Note:  Backwards compatibility -> Handling extra/exception cases
    //        related to earlier versions (1.0)


    return metrics_public_init;
  }();

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Returning = " << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

rsmi_status_t GpuMetricsBase_v10_t::populate_metrics_dynamic_tbl() {
    std::ostringstream ss;
    auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    LOG_TRACE(ss);

    auto m_metrics_dynamic_tbl = AMDGpuDynamicMetricsTbl_t{};
    //
    //  Note: Any metric treatment/changes (if any) should happen before they
    //        get written to internal/external tables.
    //
    auto run_metric_adjustments_v10 = [&]() {
      ss << __PRETTY_FUNCTION__ << " | ======= start =======";
      const auto gpu_metrics_version = translate_flag_to_metric_version(get_gpu_metrics_version_used());
      ss << __PRETTY_FUNCTION__
                  << " | ======= info ======= "
                  << " | Applying adjustments "
                  << " | Metric Version: " << stringfy_metric_header_version(
                                                disjoin_metrics_version(gpu_metrics_version))
                  << " |";
      LOG_TRACE(ss);
    };


    //  Adjustments/Changes specific to this version
    run_metric_adjustments_v10();

    // Timestamp Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
                format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                  "system_clock_counter"))
             );

    // Temperature Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempEdge,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_edge,
                                  "temperature_edge"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                  "temperature_hotspot"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                  "temperature_mem"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrGfx,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_vrgfx,
                                  "temperature_vrgfx"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                  "temperature_vrsoc"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrMem,
                format_metric_row(m_gpu_metrics_tbl.m_temperature_vrmem,
                                  "temperature_vrmem"))
             );

    // Power/Energy Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocketPower,
                format_metric_row(m_gpu_metrics_tbl.m_average_socket_power,
                                  "average_socket_power"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
                format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                  "energy_acc"))
             );

    // Utilization Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
                format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                  "average_gfx_activity"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
                format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                  "average_umc_activity"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgMmActivity,
                format_metric_row(m_gpu_metrics_tbl.m_average_mm_activity,
                                  "average_mm_activity"))
             );


    // Fan Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed,
                format_metric_row(m_gpu_metrics_tbl.m_current_fan_speed,
                                  "current_fan_speed"))
             );

    // Throttle Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
                format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                  "throttle_status"))
             );

    // Average Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_gfxclk_frequency,
                                  "average_gfxclk_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_socclk_frequency,
                                  "average_socclk_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_uclk_frequency,
                                  "average_uclk_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_vclk0_frequency,
                                  "average_vclk0_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_dclk0_frequency,
                                  "average_dclk0_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_vclk1_frequency,
                                  "average_vclk1_frequency"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency,
                format_metric_row(m_gpu_metrics_tbl.m_average_dclk1_frequency,
                                  "average_dclk1_frequency"))
             );

    // CurrentClock Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
                format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                  "current_gfxclk"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
                format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                  "current_socclk"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
                format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                  "current_uclk"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
                format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                  "current_vclk0"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
                format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                  "current_dclk0"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock1,
                format_metric_row(m_gpu_metrics_tbl.m_current_vclk1,
                                  "current_vclk1"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock1,
                format_metric_row(m_gpu_metrics_tbl.m_current_dclk1,
                                  "current_dclk1"))
             );

    // Link/Width/Speed Info
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
                format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                  "pcie_link_width"))
             );
    m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
      .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
                format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                  "pcie_link_speed"))
             );

    ss << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Success "
                << " | Returning = " << getRSMIStatusString(status_code)
                << " |";
    LOG_TRACE(ss);

    // Copy to base class
    std::copy(m_metrics_dynamic_tbl.begin(),
              m_metrics_dynamic_tbl.end(),
              std::inserter(GpuMetricsBase_t::m_base_metrics_dynamic_tbl,
                            GpuMetricsBase_t::m_base_metrics_dynamic_tbl.end()));

    return status_code;
}

AMGpuMetricsPublicLatestTupl_t GpuMetricsBase_v10_t::copy_internal_to_external_metrics() {
    std::ostringstream ss;
    auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
    ss << __PRETTY_FUNCTION__ << " | ======= start =======";
    LOG_TRACE(ss);

    auto copy_data_from_internal_metrics_tbl = [&]() {
      AMGpuMetricsPublicLatest_t metrics_public_init{};

      //
      //  Note: Initializing data members with their max. If field is max,
      //        no data was assigned to it.
      init_max_public_gpu_matrics(metrics_public_init);

      // Header
      metrics_public_init.common_header.structure_size = m_gpu_metrics_tbl.m_common_header.m_structure_size;
      metrics_public_init.common_header.format_revision = m_gpu_metrics_tbl.m_common_header.m_format_revision;
      metrics_public_init.common_header.content_revision = m_gpu_metrics_tbl.m_common_header.m_content_revision;

      // Temperature
      metrics_public_init.temperature_edge = m_gpu_metrics_tbl.m_temperature_edge;
      metrics_public_init.temperature_hotspot = m_gpu_metrics_tbl.m_temperature_hotspot;
      metrics_public_init.temperature_mem = m_gpu_metrics_tbl.m_temperature_mem;
      metrics_public_init.temperature_vrgfx = m_gpu_metrics_tbl.m_temperature_vrgfx;
      metrics_public_init.temperature_vrsoc = m_gpu_metrics_tbl.m_temperature_vrsoc;
      metrics_public_init.temperature_vrmem = m_gpu_metrics_tbl.m_temperature_vrmem;

      // Utilization
      metrics_public_init.average_gfx_activity = m_gpu_metrics_tbl.m_average_gfx_activity;
      metrics_public_init.average_umc_activity = m_gpu_metrics_tbl.m_average_umc_activity;
      metrics_public_init.average_mm_activity = m_gpu_metrics_tbl.m_average_mm_activity;

      // Power/Energy
      metrics_public_init.average_socket_power = m_gpu_metrics_tbl.m_average_socket_power;
      metrics_public_init.energy_accumulator = m_gpu_metrics_tbl.m_energy_accumulator;

      // Driver attached timestamp (in ns)
      metrics_public_init.system_clock_counter = m_gpu_metrics_tbl.m_system_clock_counter;

      // Average clocks
      metrics_public_init.average_gfxclk_frequency = m_gpu_metrics_tbl.m_average_gfxclk_frequency;
      metrics_public_init.average_socclk_frequency = m_gpu_metrics_tbl.m_average_socclk_frequency;
      metrics_public_init.average_uclk_frequency = m_gpu_metrics_tbl.m_average_uclk_frequency;
      metrics_public_init.average_vclk0_frequency = m_gpu_metrics_tbl.m_average_vclk0_frequency;
      metrics_public_init.average_dclk0_frequency = m_gpu_metrics_tbl.m_average_dclk0_frequency;
      metrics_public_init.average_vclk1_frequency = m_gpu_metrics_tbl.m_average_vclk1_frequency;
      metrics_public_init.average_dclk1_frequency = m_gpu_metrics_tbl.m_average_dclk1_frequency;

      // Current clocks
      metrics_public_init.current_gfxclk = m_gpu_metrics_tbl.m_current_gfxclk;
      metrics_public_init.current_socclk = m_gpu_metrics_tbl.m_current_socclk;
      metrics_public_init.current_vclk0 = m_gpu_metrics_tbl.m_current_vclk0;
      metrics_public_init.current_dclk0 = m_gpu_metrics_tbl.m_current_dclk0;
      metrics_public_init.current_uclk = m_gpu_metrics_tbl.m_current_uclk;
      metrics_public_init.current_vclk1 = m_gpu_metrics_tbl.m_current_vclk1;
      metrics_public_init.current_dclk1 = m_gpu_metrics_tbl.m_current_dclk1;

      // Throttle status
      metrics_public_init.throttle_status = m_gpu_metrics_tbl.m_throttle_status;

      // Fans
      metrics_public_init.current_fan_speed = m_gpu_metrics_tbl.m_current_fan_speed;

      // Link width/speed
      metrics_public_init.pcie_link_width = m_gpu_metrics_tbl.m_pcie_link_width;
      metrics_public_init.pcie_link_speed = m_gpu_metrics_tbl.m_pcie_link_speed;


      //
      // Note:  Backwards compatibility -> Handling extra/exception cases
      //        related to earlier versions (1.0)


      return metrics_public_init;
    }();

    ss << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Success "
                << " | Returning = " << getRSMIStatusString(status_code)
                << " |";
    LOG_TRACE(ss);

    return std::make_tuple(status_code, copy_data_from_internal_metrics_tbl);
}

auto Device::dev_read_gpu_metrics_header_data(DevInfoTypes type) -> rsmi_status_t {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  int op_result;
  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);
  op_result = readDevInfo(type, sizeof(AMDGpuMetricsHeader_v1_t),
                          &m_gpu_metrics_header);
  if ((status_code = ErrnoToRsmiStatus(op_result)) !=
      rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Partition ID: " << m_partition_id
       << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
       << " | File Path: " << gpu_metrics_path
       << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: "
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " Could not read Metrics Header: "
       << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }
  ss << __PRETTY_FUNCTION__ << " | Before is_gpu_metrics_version_supported() "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Partition ID: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
     << " | File Path: " << gpu_metrics_path
     << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);
  if ((status_code = is_gpu_metrics_version_supported(m_gpu_metrics_header, is_smi_expecting_partition_metrics())) ==
      rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Partition ID: " << m_partition_id
       << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
       << " | File Path: " << gpu_metrics_path
       << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: gpu metric file version is not supported: "
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " Could not read Metrics Header: "
       << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }
  m_gpu_metrics_updated_timestamp = actual_timestamp_in_secs();

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Partition ID: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
     << " | File Path: " << gpu_metrics_path
     << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);
  return status_code;
}

auto Device::dev_read_gpu_metrics_all_data(DevInfoTypes type) -> rsmi_status_t {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  int op_result;
  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);
  // Default path (::kDevGpuMetrics / !m_is_partition_metrics):
  //             /sys/class/drm/renderDXXX/device/gpu_metrics
  // Partition Path (::kDevGpuMetrics / m_is_partition_metrics):
  //            /sys/class/drm/renderDXXX/device/xcp/xcp_metrics

  //  At this point we should have a valid gpu_metrics pointer, and
  //  we already read the header; setup_gpu_metrics_reading()
  if (!m_gpu_metrics_ptr || (status_code = is_gpu_metrics_version_supported(
    m_gpu_metrics_header, is_smi_expecting_partition_metrics())) == RSMI_STATUS_NOT_SUPPORTED
      ) {
    status_code = RSMI_STATUS_SETTING_UNAVAILABLE;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: Couldn't get a valid metric object. setup_gpu_metrics_reading()"
       << " | m_gpu_metrics_ptr: "
       << (m_gpu_metrics_ptr ? "valid" : "nullptr")
       << " | m_gpu_metrics_header.m_structure_size: "
       << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
       << " | m_gpu_metrics_header.m_format_revision: "
       << print_unsigned_int(m_gpu_metrics_header.m_format_revision)
       << " | m_gpu_metrics_header.m_content_revision: "
       << print_unsigned_int(m_gpu_metrics_header.m_content_revision)
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= P1 Start ======= "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Partition ID: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
     << " | Is Dynamic GPU Metrics Supported: " << std::boolalpha << m_is_dynamic_gpu_metrics_supported
     << " | File Path: " << gpu_metrics_path
     << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_DEBUG(ss);

  if (m_is_dynamic_gpu_metrics_supported) {
    // Parse blob to schema rows AMDGpuDynamicMetrics_t
    AMDGpuDynamicMetrics_t parsed;
    rsmi_status_t st = parsed.parse_from_file(gpu_metrics_path, m_gpu_metrics_header.m_structure_size);

    if (st != RSMI_STATUS_SUCCESS) {
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << index()
         << " | Cause: read_dynamic_gpu_metrics_file()"
         << " | Returning rocmsmi_status = " << getRSMIStatusString(st) << " |";
      LOG_ERROR(ss);
      return rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    }

    // Store AMDGpuDynamicMetrics_t
    auto* dyn = static_cast<GpuMetricsBaseDynamic_t*>(m_gpu_metrics_ptr.get());
    status_code = dyn->set_parsed_dynamic(std::move(parsed));
    if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
      return status_code;
    }

  } else {
  op_result = readDevInfo(type,
                          m_gpu_metrics_header.m_structure_size,
                          m_gpu_metrics_ptr->get_metrics_table().get());

  if ((status_code = ErrnoToRsmiStatus(op_result)) !=
      rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Partition ID: " << m_partition_id
       << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
       << " | File Path: " << gpu_metrics_path
       << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: readDevInfo(kDevGpuMetrics)"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " Could not read Metrics Header: "
       << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
       << " |";
    LOG_ERROR(ss);
    return status_code;
    }
  }

  //  All metric units are pushed in.
  status_code = m_gpu_metrics_ptr->populate_metrics_dynamic_tbl();
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Partition ID: " << m_partition_id
       << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
       << " | File Path: " << gpu_metrics_path
       << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
  }

  m_gpu_metrics_updated_timestamp = actual_timestamp_in_secs();
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Partition ID: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
     << " | File Path: " << gpu_metrics_path
     << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header, is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);
  return status_code;
}

auto Device::setup_gpu_metrics_reading(DevInfoTypes type) -> rsmi_status_t {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  status_code = dev_read_gpu_metrics_header_data(type);
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    return status_code;
  }

  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);
  // Default path (::kDevGpuMetrics / !m_is_partition_metrics):
  //             /sys/class/drm/renderDXXX/device/gpu_metrics
  // Partition Path (::kDevGpuMetrics / m_is_partition_metrics):
  //            /sys/class/drm/renderDXXX/device/xcp/xcp_metrics

  std::string metric_version_str =
      stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path);

  const auto gpu_metrics_flag_version = translate_header_to_flag_version(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path);
  if (gpu_metrics_flag_version == AMDGpuMetricVersionFlags_t::kGpuMetricNone) {
    status_code = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | [Translates to: " << join_metrics_version(dev_get_metrics_header())
       << " ] "
       << " | Cause: Metric version found is not supported!"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  m_is_dynamic_gpu_metrics_supported =
      (static_cast<std::underlying_type_t<AMDGpuMetricVersionFlags_t>>(gpu_metrics_flag_version) >=
        static_cast<std::underlying_type_t<AMDGpuMetricVersionFlags_t>>(
            AMDGpuMetricVersionFlags_t::kGpuMetricDynV19Plus)
            && !is_smi_expecting_partition_metrics()) ||
       (static_cast<std::underlying_type_t<AMDGpuMetricVersionFlags_t>>(gpu_metrics_flag_version) >=
           static_cast<std::underlying_type_t<AMDGpuMetricVersionFlags_t>>(
               AMDGpuMetricVersionFlags_t::kGpuXcpMetricDynV11Plus)
               && is_smi_expecting_partition_metrics());

  m_gpu_metrics_ptr.reset();
  ss << __PRETTY_FUNCTION__
     << " | ======= P1 Start ======= "
     << " | Status: Before amdgpu_metrics_factory() "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Partition ID: " << m_partition_id
     << " | Is Partition Metrics: " << std::boolalpha << is_smi_expecting_partition_metrics()
     << " | Is Dynamic Metrics Supported: " << std::boolalpha << m_is_dynamic_gpu_metrics_supported
     << " | Metric Flag Version: ||" << metric_version_str << "||"
     << " | File Path: " << gpu_metrics_path
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_DEBUG(ss);
  m_gpu_metrics_ptr = amdgpu_metrics_factory(gpu_metrics_flag_version, is_smi_expecting_partition_metrics(), gpu_metrics_path);
  if (!m_gpu_metrics_ptr) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: amdgpu_metrics_factory() couldn't get a valid metric object"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }
  m_gpu_metrics_ptr->set_device_id(get_smi_device_id());
  m_gpu_metrics_ptr->set_partition_id(get_smi_partition_id());
  m_gpu_metrics_ptr->set_is_partition_metrics(is_smi_expecting_partition_metrics());

  // m_gpu_metrics_ptr has the pointer to the proper object type/version.
  status_code = dev_read_gpu_metrics_all_data(type);
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: dev_read_gpu_metrics_all_data() couldn't read gpu metric data!"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | [A] Fabric: [" << &m_gpu_metrics_ptr
     << " ]"
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);
  return status_code;
}


template<AMDGpuMetricsDataType_t data_type>
struct MetricValueCast_t;

template<>
struct MetricValueCast_t<AMDGpuMetricsDataType_t::kUInt8>
{
  using value_type = std::uint8_t;
};
template<>
struct MetricValueCast_t<AMDGpuMetricsDataType_t::kUInt16>
{
  using value_type = std::uint16_t;
};
template<>
struct MetricValueCast_t<AMDGpuMetricsDataType_t::kUInt32>
{
  using value_type = std::uint32_t;
};

template<>
struct MetricValueCast_t<AMDGpuMetricsDataType_t::kUInt64>
{
  using value_type = std::uint64_t;
};

template<AMDGpuMetricsDataType_t dt>
auto get_casted_value(const AMDGpuDynamicMetricsValue_t& metrics_value)
{
    using ValueType_t = typename MetricValueCast_t<dt>::value_type;
    return static_cast<ValueType_t>(metrics_value.m_value);
}


auto Device::dev_log_gpu_metrics(std::ostringstream& outstream_metrics,
                                          DevInfoTypes type) -> rsmi_status_t {
  std::ostringstream ss;
  std::ostringstream tmp_outstream_metrics;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);

  //  If we still don't have a valid gpu_metrics pointer;
  //  meaning, we didn't run any queries, and just want to
  //  print all the gpu metrics content, we need to setup
  //  the environment first.
  status_code = setup_gpu_metrics_reading(type);
  if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || (!m_gpu_metrics_ptr)) {
    // At this point we should have a valid gpu_metrics pointer.
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: Couldn't get a valid metric object"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  //  Header info
  const auto kDoubleLine = std::string("+==============================+");
  const auto kSingleLine = std::string("+------------------------------+");
  auto header_output = [&]() {
    const auto gpu_metrics_header = dev_get_metrics_header();
    const auto timestamp_time = timestamp_to_time_point(m_gpu_metrics_updated_timestamp);
    tmp_outstream_metrics << "\n" << kDoubleLine << "\n";
    tmp_outstream_metrics << "*** GPU Metrics Header: ***";
    tmp_outstream_metrics << "\n";
    tmp_outstream_metrics << "Timestamp: "
                          << " ["
                          << m_gpu_metrics_updated_timestamp
                          << "] "
                          << std::ctime(&timestamp_time);
    tmp_outstream_metrics << "Version: "
                          << print_unsigned_int(gpu_metrics_header.m_format_revision)
                          << "."
                          << print_unsigned_int(gpu_metrics_header.m_content_revision)
                          << " [Flag: "
                          << static_cast<uint32_t>(m_gpu_metrics_ptr->get_gpu_metrics_version_used())
                          << "] "
                          << "\n";
    tmp_outstream_metrics << " ->Device #: " << index() << "\n";
    tmp_outstream_metrics << print_unsigned_hex_and_int(gpu_metrics_header.m_structure_size,   " ->structure_size   ");
    tmp_outstream_metrics << print_unsigned_hex_and_int(gpu_metrics_header.m_format_revision,  " ->format_revision  ");
    tmp_outstream_metrics << print_unsigned_hex_and_int(gpu_metrics_header.m_content_revision, " ->content_revision ");
    tmp_outstream_metrics << "\n" << kSingleLine << "\n";
    return;
  };

  //  Metrics info
  auto table_content_output = [&]() {
    const auto gpu_metrics_tbl = m_gpu_metrics_ptr->get_metrics_dynamic_tbl();
    tmp_outstream_metrics << "\n";
    tmp_outstream_metrics << "*** GPU Metrics Data: *** \n";
    for (const auto& [metric_class, metric_data] : gpu_metrics_tbl) {
      tmp_outstream_metrics << "\n";
      tmp_outstream_metrics << "[ " << amdgpu_metrics_class_id_translation_table.at(metric_class) << " ]" << "\n";

      for (const auto& [metric_unit, metric_values] : metric_data) {
        auto tmp_metric_info = ("[ " + amdgpu_metrics_unit_type_translation_table.at(metric_unit) + " ]");
        for (const auto& metric_value : metric_values) {
          switch (metric_value.m_original_type) {
            case (AMDGpuMetricsDataType_t::kUInt8):
              {
                auto value = get_casted_value<AMDGpuMetricsDataType_t::kUInt8>(metric_value);
                tmp_outstream_metrics << print_unsigned_hex_and_int((value), metric_value.m_info) << " -> " << tmp_metric_info;
              }
              break;
            case (AMDGpuMetricsDataType_t::kUInt16):
              {
                auto value = get_casted_value<AMDGpuMetricsDataType_t::kUInt16>(metric_value);
                tmp_outstream_metrics << print_unsigned_hex_and_int((value), metric_value.m_info) << " -> " << tmp_metric_info;
              }
              break;

            case (AMDGpuMetricsDataType_t::kUInt32):
              {
                auto value = get_casted_value<AMDGpuMetricsDataType_t::kUInt32>(metric_value);
                tmp_outstream_metrics << print_unsigned_hex_and_int((value), metric_value.m_info) << " -> " << tmp_metric_info;
              }
              break;

            case (AMDGpuMetricsDataType_t::kUInt64):
              {
                auto value = get_casted_value<AMDGpuMetricsDataType_t::kUInt64>(metric_value);
                tmp_outstream_metrics << print_unsigned_hex_and_int((value), metric_value.m_info) << " -> " << tmp_metric_info;
              }
              break;

            default:
            tmp_outstream_metrics << "Error: No data type conversion for original type: " << static_cast<AMDGpuMetricsDataTypeId_t>(metric_value.m_original_type) << "\n";
            break;
          }
        }
      }
      tmp_outstream_metrics << "\n\n";
    }
    tmp_outstream_metrics << "\n" << kDoubleLine << "\n";
    return;
  };

  header_output();
  table_content_output();
  outstream_metrics << tmp_outstream_metrics.rdbuf();
  LOG_DEBUG(tmp_outstream_metrics);

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | [B] Fabric: [" << &m_gpu_metrics_ptr
     << " ]"
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);
  return status_code;
}

auto Device::dev_copy_internal_to_external_metrics(DevInfoTypes type)
    -> AMGpuMetricsPublicLatestTupl_t {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);

  if (!m_gpu_metrics_ptr) {
    // At this point we should have a valid gpu_metrics pointer.
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: Couldn't get a valid metric object"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return std::make_tuple(status_code, AMGpuMetricsPublicLatest_t());
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | [C] Fabric: [" << &m_gpu_metrics_ptr
     << " ]"
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_TRACE(ss);

  return m_gpu_metrics_ptr->copy_internal_to_external_metrics();
}

auto Device::run_internal_gpu_metrics_query(AMDGpuMetricsUnitType_t metric_counter,
                                  AMDGpuDynamicMetricTblValues_t& values,
                                  DevInfoTypes type) -> rsmi_status_t {
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  std::string gpu_metrics_path = get_sys_file_path_by_type(type, true);
  // Default path (::kDevGpuMetrics / !m_is_partition_metrics):
  //             /sys/class/drm/renderDXXX/device/gpu_metrics
  // Partition Path (::kDevGpuMetrics / m_is_partition_metrics):
  //            /sys/class/drm/renderDXXX/device/xcp/xcp_metrics
  status_code = setup_gpu_metrics_reading(type);
  if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || (!m_gpu_metrics_ptr)) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << index()
       << " | Type: " << Device::get_type_string(type)
       << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
       << " | Cause: Couldn't get a valid metric object"
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  // Lookup the dynamic table
  ss << __PRETTY_FUNCTION__
     << " | ======= info ======= "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Metric Unit: " << static_cast<AMDGpuMetricTypeId_t>(metric_counter)
     << " |";
  LOG_INFO(ss);
  const auto gpu_metrics_tbl = m_gpu_metrics_ptr->get_metrics_dynamic_tbl();
  for (const auto& [metric_class, metric_data] : gpu_metrics_tbl) {
    for (const auto& [metric_unit, metric_values] : metric_data) {
      if (metric_unit == metric_counter) {
        values = metric_values;
        status_code = rsmi_status_t::RSMI_STATUS_SUCCESS;
        ss << __PRETTY_FUNCTION__
           << " | ======= end ======= "
           << " | Success "
           << " | Device #: " << index()
           << " | Type: " << Device::get_type_string(type)
           << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
           << " | Metric Unit: " << static_cast<AMDGpuMetricTypeId_t>(metric_counter)
           << " | Returning = "
           << getRSMIStatusString(status_code, false)
           << " |";
        LOG_TRACE(ss);
        return status_code;
      }
    }
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Fail "
     << " | Device #: " << index()
     << " | Type: " << Device::get_type_string(type)
     << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header(), is_smi_expecting_partition_metrics(), gpu_metrics_path)
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_ERROR(ss);
  return status_code;
}


template<typename T>
constexpr inline bool is_metric_data_type_supported_v =
    ((std::is_same_v<T, std::uint16_t>) || (std::is_same_v<T, const std::uint16_t>) ||
     (std::is_same_v<T, std::uint32_t>) || (std::is_same_v<T, const std::uint32_t>) ||
     (std::is_same_v<T, std::uint64_t>) || (std::is_same_v<T, const std::uint64_t>));

template<typename>
struct is_std_vector : std::false_type {};

template <typename T, typename... Ts>
struct is_std_vector<std::vector<T, Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename T>
constexpr bool is_std_vector_type_supported_v()
{
    if constexpr (is_std_vector_v<T>) {
        using ValueType_t = typename T::value_type;
        return (is_metric_data_type_supported_v<ValueType_t>);
    }
    return false;
};

template<typename T>
rsmi_status_t rsmi_dev_gpu_metrics_info_query(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, T& metric_value)
{
  std::ostringstream ss;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  static constexpr bool is_supported_vector_type = [&]() {
    if constexpr (is_std_vector_v<T>) {
      if (is_std_vector_type_supported_v<T>()) {
        return true;
      }
    }
    return false;
  }();


  if constexpr ((is_supported_vector_type) || (is_metric_data_type_supported_v<T>)) {
    // Get all stored values for the metric unit/counter
    AMDGpuDynamicMetricTblValues_t tmp_values{};
    GET_DEV_FROM_INDX
    status_code = dev->run_internal_gpu_metrics_query(metric_counter, tmp_values);
    if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || tmp_values.empty()) {
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Metric Version: " << stringfy_metrics_header(dev->dev_get_metrics_header(), false, "N/A")
         << " | Cause: Couldn't find metric/counter requested"
         << " | Metric Type: " << static_cast<uint32_t>(metric_counter)
         << " " << amdgpu_metrics_unit_type_translation_table.at(metric_counter)
         << " | Values: " << tmp_values.size()
         << " | Returning = "
         << getRSMIStatusString(status_code, false)
         << " |";
      LOG_ERROR(ss);
      return status_code;
    }

    if constexpr (is_std_vector_v<T>) {
      using ValueType_t = typename T::value_type;
      ValueType_t tmp_value;

      for (const auto& value : tmp_values) {
        tmp_value = static_cast<ValueType_t>(value.m_value);
        metric_value.push_back(tmp_value);
      }
    }
    else if constexpr (is_metric_data_type_supported_v<T>) {
      T tmp_value(0);
      tmp_value = static_cast<decltype(tmp_value)>(tmp_values[0].m_value);
      metric_value = tmp_value;
    }
  }
  else {
    static_assert(is_dependent_false_v<T>, "Error: Data Type not supported...");
  }

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Device #: " << dv_ind
              << " | Metric Type: " << static_cast<uint32_t>(metric_counter)
              << " | Returning = "
              << getRSMIStatusString(status_code, false)
              << " |";
  LOG_TRACE(ss);
  return status_code;
}


template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint16_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint16_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint32_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint32_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint64_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint64_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<GpuMetricU16Tbl_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, GpuMetricU16Tbl_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<GpuMetricU32Tbl_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, GpuMetricU32Tbl_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<GpuMetricU64Tbl_t>
(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, GpuMetricU64Tbl_t& metric_value);

} //namespace amd::smi

rsmi_status_t
rsmi_dev_gpu_metrics_header_info_get(uint32_t dv_ind, metrics_table_header_t& header_value)
{
  TRY
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  GET_DEV_FROM_INDX
  status_code = dev->dev_read_gpu_metrics_header_data();
  if (status_code == rsmi_status_t::RSMI_STATUS_SUCCESS) {
    auto tmp_header_info = dev->dev_get_metrics_header();
    std::memcpy(&header_value, &tmp_header_info, sizeof(metrics_table_header_t));
  }

  ss << __PRETTY_FUNCTION__
              << " | ======= end ======= "
              << " | Success "
              << " | Device #: " << dv_ind
              << " | Returning = "
              << getRSMIStatusString(status_code)
              << " |";
  LOG_TRACE(ss);

  return status_code;
  CATCH
}

//dev_read_gpu_metrics_header_data

/**
 *  Note: These keep backwards compatibility with previous GPU metrics work
 */
// log current gpu_metrics file content read
// any metrics value can be a nullptr
rsmi_status_t
rsmi_dev_gpu_metrics_info_get(uint32_t dv_ind, rsmi_gpu_metrics_t* smu) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(smu)

  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  thread_local std::ostringstream ostrstream;
  thread_local std::ostringstream ss;

  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  DevInfoTypes type = DevInfoTypes::kDevGpuMetrics;

  assert(smu != nullptr);
  if (smu == nullptr) {
    status_code = rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  dev->set_smi_device_id(dv_ind);
  dev->set_smi_dev_info_type(type);
  uint32_t partition_id = 0;
  auto ret = rsmi_dev_partition_id_get(dv_ind, &partition_id);
  if (ret == RSMI_STATUS_SUCCESS) {
    dev->set_smi_partition_id(partition_id);
  } else {
    dev->set_smi_partition_id(0);
  }

  // check if file exists, report not supported if it does not exist
  std::string file_name = dev->get_sys_file_path_by_type(type, true);
  if (access(file_name.c_str(), F_OK | R_OK) != 0) {
    status_code = RSMI_STATUS_NOT_SUPPORTED;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Cause: File does not exist or is not readable"
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | File: " << file_name
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  dev->dev_log_gpu_metrics(ostrstream);
  const auto [error_code, external_metrics] = dev->dev_copy_internal_to_external_metrics();
  if (error_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Cause: Could not copy internal to external metrics"
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | File: " << file_name
       << " | Returning = "
       << getRSMIStatusString(error_code, false)
       << " |";
    LOG_ERROR(ss);
    return error_code;
  }

  *smu = external_metrics;
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: " << Device::get_type_string(type)
     << " | File: " << file_name
     << " | Returning = "
     << getRSMIStatusString(status_code, false)
     << " |";
  LOG_INFO(ss);

  return status_code;
  CATCH
}

rsmi_status_t
rsmi_dev_gpu_partition_metrics_info_get(uint32_t dv_ind, rsmi_gpu_metrics_t* smu) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(smu)

  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  thread_local std::ostringstream ostrstream;
  thread_local std::ostringstream ss;

  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  DevInfoTypes type = DevInfoTypes::kdevGpuPartitionMetrics;

  assert(smu != nullptr);
  if (smu == nullptr) {
    status_code = rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  dev->set_smi_device_id(dv_ind);
  dev->set_smi_dev_info_type(type);
  uint32_t partition_id = 0;
  auto ret = rsmi_dev_partition_id_get(dv_ind, &partition_id);
  if (ret == RSMI_STATUS_SUCCESS) {
    dev->set_smi_partition_id(partition_id);
  } else {
    dev->set_smi_partition_id(0);
  }

  // check if file exists, report not supported if it does not exist
  std::string file_name = dev->get_sys_file_path_by_type(type, true);
  // Default path (::kDevGpuMetrics / !m_is_partition_metrics):
  //             /sys/class/drm/renderDXXX/device/gpu_metrics
  // Partition Path (::kDevGpuMetrics / m_is_partition_metrics):
  //            /sys/class/drm/renderDXXX/device/xcp/xcp_metrics
  if (access(file_name.c_str(), F_OK | R_OK) != 0) {
    status_code = RSMI_STATUS_NOT_SUPPORTED;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Cause: File does not exist or is not readable"
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | File: " << file_name
       << " | Returning = "
       << getRSMIStatusString(status_code, false)
       << " |";
    LOG_ERROR(ss);
    return status_code;
  }

  dev->dev_log_gpu_metrics(ostrstream, type);
  const auto [error_code, external_metrics]
    = dev->dev_copy_internal_to_external_metrics(type);
  if (error_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Cause: Could not copy internal to external metrics"
       << " | Device #: " << dv_ind
       << " | Type: " << Device::get_type_string(type)
       << " | File: " << file_name
       << " | Returning = "
       << getRSMIStatusString(error_code, false)
       << " |";
    LOG_ERROR(ss);
    return error_code;
  }

  *smu = external_metrics;
  ss << __PRETTY_FUNCTION__
      << " | ======= end ======= "
      << " | Success "
      << " | Device #: " << dv_ind
      << " | Type: " << Device::get_type_string(type)
      << " | File: " << file_name
      << " | Returning = "
      << getRSMIStatusString(status_code, false)
      << " |";
  LOG_INFO(ss);

  return status_code;
  CATCH
}

