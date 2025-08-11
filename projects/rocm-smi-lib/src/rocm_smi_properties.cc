/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2025, Advanced Micro Devices, Inc.
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

#include "rocm_smi/rocm_smi_properties.h"
#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_logger.h"

#include <algorithm>
#include <cassert>
#include <sstream>


//
// Property reinforcement check list
//
// NOTE: This is a *temporary solution* until we get a better approach, likely
//       a driver API that can give us the capabilities of a GPU in question.
//
namespace amd {
namespace smi {

const AMDGpuOpModeList_t amdgpu_opmode_check_list {
  {AMDGpuPropertyOpModeTypes_t::kBareMetal, "Bare Metal"},
  {AMDGpuPropertyOpModeTypes_t::kSrIov, "SR-IOV"},
  {AMDGpuPropertyOpModeTypes_t::kBoth, "Both"},
};

const AMDGpuPropertyTypesOffsetList_t amdgpu_typeoffset_check_list {
  {AMDGpuPropertyTypesOffset_t::kNone, "None"},
  {AMDGpuPropertyTypesOffset_t::kDevInfoTypes, "Device Info Type"},
  {AMDGpuPropertyTypesOffset_t::kMonitorTypes, "Monitor Type"},
  {AMDGpuPropertyTypesOffset_t::kPerfTypes, "Performance Type"},
  {AMDGpuPropertyTypesOffset_t::kClkTypes, "Clock Type"},
  {AMDGpuPropertyTypesOffset_t::kVoltMetricTypes, "Volt Metric Type"},
};


AMDGpuPropertyId_t make_unique_property_id(AMDGpuPropertyTypesOffset_t type_offset, AMDGpuPropertyId_t property_id) {
  return (static_cast<AMDGpuPropertyOffsetType>(type_offset) | (property_id));
}

AMDGpuPropertyId_t unmake_unique_property_id(AMDGpuPropertyId_t property_id) {
  const auto property_type_offset_mask =
      static_cast<AMDGpuPropertyOffsetType>(AMDGpuPropertyTypesOffset_t::kDevInfoTypes) |
      static_cast<AMDGpuPropertyOffsetType>(AMDGpuPropertyTypesOffset_t::kMonitorTypes) |
      static_cast<AMDGpuPropertyOffsetType>(AMDGpuPropertyTypesOffset_t::kPerfTypes) |
      static_cast<AMDGpuPropertyOffsetType>(AMDGpuPropertyTypesOffset_t::kClkTypes)  |
      static_cast<AMDGpuPropertyOffsetType>(AMDGpuPropertyTypesOffset_t::kVoltMetricTypes);

  auto property_type_id = (static_cast<AMDGpuPropertyOffsetType>(property_id) & ~(property_type_offset_mask));

  return property_type_id;
}

AMDGpuPropertyTypesOffset_t operator| (AMDGpuPropertyTypesOffset_t lhs, AMDGpuPropertyTypesOffset_t rhs)
{
  if (lhs == rhs) {
    return lhs;
  }

  return AMDGpuPropertyTypesOffset_t(static_cast<AMDGpuPropertyOffsetType>(lhs) | static_cast<AMDGpuPropertyOffsetType>(rhs));
}

AMDGpuPropertyTypesOffset_t operator& (AMDGpuPropertyTypesOffset_t lhs, AMDGpuPropertyTypesOffset_t rhs)
{
  if (lhs == rhs) {
    return lhs;
  }

  return AMDGpuPropertyTypesOffset_t(static_cast<AMDGpuPropertyOffsetType>(lhs) & static_cast<AMDGpuPropertyOffsetType>(rhs));
}

AMDGpuPropertyOpModeTypes_t operator| (AMDGpuPropertyOpModeTypes_t lhs, AMDGpuPropertyOpModeTypes_t rhs)
{
  if (lhs == rhs) {
    return lhs;
  }

  return AMDGpuPropertyOpModeTypes_t(static_cast<AMDGpuPropertyOpModeType>(lhs) | static_cast<AMDGpuPropertyOpModeType>(rhs));
}

AMDGpuPropertyOpModeTypes_t operator& (AMDGpuPropertyOpModeTypes_t lhs, AMDGpuPropertyOpModeTypes_t rhs)
{
  if (lhs == rhs) {
    return lhs;
  }

  return AMDGpuPropertyOpModeTypes_t(static_cast<AMDGpuPropertyOpModeType>(lhs) & static_cast<AMDGpuPropertyOpModeType>(rhs));
}


//
// Note:  Due to the fact that we have different enum elements with the same
//        number, keying a hash by the number is not an option; ie:
//        - DevInfoTypes::kDevVendorID = 7
//        - MonitorTypes::kMonPowerCapDefault = 7
//        So, we are keying it by a unique key, based on their info types
//
const AMDGpuVerbList_t amdgpu_verb_check_list {
  { AMDGpuVerbTypes_t::kNone, "None" },
  { AMDGpuVerbTypes_t::kSetGpuPciBandwidth, "amdsmi_set_gpu_pci_bandwidth" },
  { AMDGpuVerbTypes_t::kSetPowerCap, "amdsmi_set_power_cap" },
  { AMDGpuVerbTypes_t::kSetGpuPowerProfile, "amdsmi_set_gpu_power_profile" },
  { AMDGpuVerbTypes_t::kSetGpuClkRange, "amdsmi_set_gpu_clk_range" },
  { AMDGpuVerbTypes_t::kSetGpuOdClkInfo, "amdsmi_set_gpu_od_clk_info" },
  { AMDGpuVerbTypes_t::kSetGpuOdVoltInfo, "amdsmi_set_gpu_od_volt_info" },
  { AMDGpuVerbTypes_t::kSetGpuPerfLevelV1, "amdsmi_set_gpu_perf_level_v1" },
  { AMDGpuVerbTypes_t::kSetGpuPerfLevel, "amdsmi_set_gpu_perf_level" },
  { AMDGpuVerbTypes_t::kGetGpuPowerProfilePresets, "amdsmi_get_gpu_power_profile_presets" },
  { AMDGpuVerbTypes_t::kResetGpu, "amdsmi_reset_gpu" },
  { AMDGpuVerbTypes_t::kSetGpuPerfDeterminismMode, "amdsmi_set_gpu_perf_determinism_mode" },
  { AMDGpuVerbTypes_t::kSetGpuFanSpeed, "amdsmi_set_gpu_fan_speed" },
  { AMDGpuVerbTypes_t::kResetGpuFan, "amdsmi_reset_gpu_fan" },
  { AMDGpuVerbTypes_t::kSetClkFreq, "amdsmi_set_clk_freq" },
  { AMDGpuVerbTypes_t::kSetGpuOverdriveLevelV1, "amdsmi_set_gpu_overdrive_level_v1" },
  { AMDGpuVerbTypes_t::kSetGpuOverdriveLevel, "amdsmi_set_gpu_overdrive_level" },
  { AMDGpuVerbTypes_t::kGetGpuFanRpms, "amdsmi_get_gpu_fan_rpms" },
  { AMDGpuVerbTypes_t::kGetGpuFanSpeed, "amdsmi_get_gpu_fan_speed" },
  { AMDGpuVerbTypes_t::kGetGpuFanSpeedMax, "amdsmi_get_gpu_fan_speed_max" },
  { AMDGpuVerbTypes_t::kGetGpuVoltMetric, "amdsmi_get_temp_metric" },
  { AMDGpuVerbTypes_t::kGetGpuOverDriveLevel, "amdsmi_get_gpu_overdrive_level" },
  { AMDGpuVerbTypes_t::kGetGpuOdVoltInfo, "amdsmi_get_gpu_od_volt_info" },
  { AMDGpuVerbTypes_t::kGetGpuOdVoltCurveRegions, "amdsmi_get_gpu_od_volt_curve_regions" }
};

const uint16_t kDevIDAll(0xFFFF);
const uint16_t kDevRevIDAll(0xFFFF);
const AMDGpuPropertyList_t amdgpu_property_reinforcement_list {
  //
  // {"Asic ID", {"Asic Rev. ID", "Unique Property ID", "Property Op.Mode", "Availability Flag"}}
  // DevInfoTypes::kDevPCIEClk = rsmi_dev_pci_bandwidth_get; rsmi_dev_pci_bandwidth_set
  // MonitorTypes::kMonPowerCapDefault = rsmi_dev_power_cap_default_get;
  // DevInfoTypes::kDevPowerProfileMode =
  // rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_MANUAL = rsmi_dev_clk_range_set;
  //

  // AMD All Families
  {kDevIDAll, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonFanCntrlEnable),
            AMDGpuVerbTypes_t::kResetGpuFan,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },

  // AMD Instinct MI210
  {0x740F, {0x02,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPowerProfileMode),
            AMDGpuVerbTypes_t::kSetGpuPowerProfile,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },

  // AMD MIxxx
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPCIEClk),
            AMDGpuVerbTypes_t::kSetGpuPciBandwidth,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonPowerCapDefault),
            AMDGpuVerbTypes_t::kSetPowerCap,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPowerProfileMode),
            AMDGpuVerbTypes_t::kSetGpuPowerProfile,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_MANUAL),
            AMDGpuVerbTypes_t::kSetGpuClkRange,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_MANUAL),
            AMDGpuVerbTypes_t::kSetGpuOdClkInfo,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_MANUAL),
            AMDGpuVerbTypes_t::kSetGpuOdVoltInfo,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_AUTO),
            AMDGpuVerbTypes_t::kSetGpuPerfLevelV1,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_MANUAL),
            AMDGpuVerbTypes_t::kSetGpuPerfLevel,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPowerProfileMode),
            AMDGpuVerbTypes_t::kGetGpuPowerProfilePresets,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kPerfTypes,
                                    rsmi_dev_perf_level::RSMI_DEV_PERF_LEVEL_DETERMINISM),
            AMDGpuVerbTypes_t::kSetGpuPerfDeterminismMode,
            AMDGpuPropertyOpModeTypes_t::kSrIov, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonFanSpeed),
            AMDGpuVerbTypes_t::kSetGpuFanSpeed,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonFanCntrlEnable),
            AMDGpuVerbTypes_t::kResetGpuFan,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kClkTypes,
                                    rsmi_clk_type::RSMI_CLK_TYPE_FIRST),
            AMDGpuVerbTypes_t::kSetClkFreq,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevOverDriveLevel),
            AMDGpuVerbTypes_t::kSetGpuOverdriveLevel,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevOverDriveLevel),
            AMDGpuVerbTypes_t::kSetGpuOverdriveLevelV1,
            AMDGpuPropertyOpModeTypes_t::kBoth, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonFanRPMs),
            AMDGpuVerbTypes_t::kGetGpuFanRpms,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonFanSpeed),
            AMDGpuVerbTypes_t::kGetGpuFanSpeed,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kMonitorTypes,
                                    MonitorTypes::kMonMaxFanSpeed),
            AMDGpuVerbTypes_t::kGetGpuFanSpeedMax,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kVoltMetricTypes,
                                    rsmi_voltage_metric_t::RSMI_VOLT_CURRENT),
            AMDGpuVerbTypes_t::kGetGpuVoltMetric,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevOverDriveLevel),
            AMDGpuVerbTypes_t::kGetGpuOverDriveLevel,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPowerODVoltage),
            AMDGpuVerbTypes_t::kGetGpuOdVoltInfo,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  },
  {0x74A1, {kDevRevIDAll,
            make_unique_property_id(AMDGpuPropertyTypesOffset_t::kDevInfoTypes,
                                    DevInfoTypes::kDevPowerODVoltage),
            AMDGpuVerbTypes_t::kGetGpuOdVoltCurveRegions,
            AMDGpuPropertyOpModeTypes_t::kBareMetal, false }
  }
};


rsmi_status_t validate_property_reinforcement_query(uint32_t dv_ind, AMDGpuVerbTypes_t verb_type, rsmi_status_t actual_error_code)
{
  std::ostringstream osstream;
  osstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  osstream << __PRETTY_FUNCTION__ << "  actual error code: " << actual_error_code << "\n";
  LOG_TRACE(osstream);

  if (actual_error_code == rsmi_status_t::RSMI_STATUS_SUCCESS) {
    return actual_error_code;
  }

  //
  // For property reinforcement query, the possible return values are:
  //  RSMI_STATUS_SUCCESS:
  //    - Property found in the reinforcement table, and it *should exist*
  //  RSMI_STATUS_NOT_SUPPORTED:
  //    - Property found in the reinforcement table, and it *should not* exist
  //  RSMI_STATUS_NO_DATA:
  //    - Could not find the correct dev_id and dev_revision info to build the filter
  //  RSMI_STATUS_UNKNOWN_ERROR:
  //    - The results are initialized with that. If that is returned,
  //      likely the reinforcement table does not contain any entries/rules for the
  //      dev_id in question.
  //
  auto amdgpu_property_query_result_hdlr = [&](const rsmi_status_t query_result) {
    switch (query_result) {
      case (rsmi_status_t::RSMI_STATUS_UNKNOWN_ERROR):
      case (rsmi_status_t::RSMI_STATUS_NO_DATA):
        return rsmi_status_t::RSMI_STATUS_NOT_FOUND;
        break;

      case (rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED):
      case (rsmi_status_t::RSMI_STATUS_SUCCESS):
        return query_result;
        break;

      default:
        return actual_error_code;
        break;
    }
  };

  ///
  GET_DEV_FROM_INDX
  osstream << __PRETTY_FUNCTION__ << "| ======= about to run property query ======="
            << " [query filters: ]"
            << " device: " << dv_ind
            << " property/verb: " << static_cast<AMDGpuVerbId_t>(verb_type) << amdgpu_verb_check_list.at(verb_type);
  auto reinforcement_query_result = dev->check_amdgpu_property_reinforcement_query(dv_ind, verb_type);
  osstream << __PRETTY_FUNCTION__ << "| ======= result from property query ======="
            << " query result: " << reinforcement_query_result;

  reinforcement_query_result = amdgpu_property_query_result_hdlr(reinforcement_query_result);
  osstream << __PRETTY_FUNCTION__ << "| ======= result from property query ======="
            << " query result: " << reinforcement_query_result;

  return reinforcement_query_result;
}

void dump_amdgpu_property_reinforcement_list()
{
  std::ostringstream osstream;
  osstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  if (!amdgpu_property_reinforcement_list.empty()) {
    for (const auto& property : amdgpu_property_reinforcement_list) {
      osstream << __PRETTY_FUNCTION__
               << "  Asic ID: " << property.first
               << "  Asic Rev.ID: " << property.second.m_pci_rev_id
               << "  Property ID: " << property.second.m_property
               << "  Verb ID : " << static_cast<AMDGpuVerbId_t>(property.second.m_verb_id)
               << "  Verb Desc: " << amdgpu_verb_check_list.at(property.second.m_verb_id)
               << "  OpMode: " << static_cast<AMDGpuOpModeType_t>(property.second.m_opmode)
               << "  OpMode Desc: " << amdgpu_opmode_check_list.at(property.second.m_opmode)
               << "  Flag Avail.: " << property.second.m_should_be_available;
    }
    osstream << __PRETTY_FUNCTION__ << "| ======= end =======";
    return;
  }

  osstream << __PRETTY_FUNCTION__ << "amdgpu_property_reinforcement_list is empty";
  LOG_TRACE(osstream);
}


rsmi_status_t Device::check_amdgpu_property_reinforcement_query(uint32_t dev_idx, AMDGpuVerbTypes_t verb_type)
{
  std::ostringstream osstream;
  auto rsmi_status(rsmi_status_t::RSMI_STATUS_UNKNOWN_ERROR);

  auto amdgpu_property_query = [&]() {
    AMDGpuPropertyQuery_t amdgpu_property_query_init{};
    amdgpu_property_query_init.m_asic_id = 0;
    amdgpu_property_query_init.m_pci_rev_id = 0;
    amdgpu_property_query_init.m_dev_idx = dev_idx;
    amdgpu_property_query_init.m_property = 0;
    amdgpu_property_query_init.m_verb_id = verb_type;
    return amdgpu_property_query_init;
  }();

  auto build_asic_id_filters = [&](const AMDGpuPropertyQuery_t& amdgpu_query_validate, bool& is_filter_good) {
    auto tmp_amdgpu_query = amdgpu_query_validate;
    auto id_filter_result(rsmi_status_t::RSMI_STATUS_SUCCESS);
    if (amdgpu_query_validate.m_asic_id == 0) {
      id_filter_result = rsmi_dev_id_get(dev_idx, &tmp_amdgpu_query.m_asic_id);
      if (id_filter_result == rsmi_status_t::RSMI_STATUS_SUCCESS) {
        id_filter_result = rsmi_dev_revision_get(dev_idx, &tmp_amdgpu_query.m_pci_rev_id);
      }
    }
    is_filter_good = (id_filter_result == rsmi_status_t::RSMI_STATUS_SUCCESS);
    return tmp_amdgpu_query;
  };

  // If the original amdgpu_query is missing parts of the filter, such as;
  // asic_id, revision_id, we try to retrieve them based on the dev_idx.
  // the property we are searching for, *must be present* .
  osstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(osstream);

  bool is_proper_query(false);

  // Generic filter for checking properties for all asics and revisions.
  auto amdgpu_property_query_all_asics = amdgpu_property_query;
  amdgpu_property_query_all_asics.m_asic_id = kDevIDAll;
  amdgpu_property_query_all_asics.m_pci_rev_id = kDevRevIDAll;
  auto amdgpu_property_query_result = run_amdgpu_property_reinforcement_query(amdgpu_property_query_all_asics);
  // We found a generic entry for all asics and revisions
  if (amdgpu_property_query_result != rsmi_status_t::RSMI_STATUS_UNKNOWN_ERROR) {
    return amdgpu_property_query_result;
  }

  // If no generic entry, then we query for specific asic and revision ids.
  amdgpu_property_query = build_asic_id_filters(amdgpu_property_query, is_proper_query);
  if (!is_proper_query) {
    rsmi_status = rsmi_status_t::RSMI_STATUS_NO_DATA;
    osstream << __PRETTY_FUNCTION__ << "| ======= end ======="
             << ", Missing Query Filters were not successfully retrieved: "
             << " [query filters: ]"
             << " device: " << dev_idx
             << " asic id: " << amdgpu_property_query.m_asic_id
             << " revision id: " << amdgpu_property_query.m_pci_rev_id
             << " property: "  << amdgpu_property_query.m_property
             << " verb: "  << static_cast<AMDGpuVerbId_t>(amdgpu_property_query.m_verb_id)
             << " proper_query: " << is_proper_query
             << " error: " << rsmi_status;
    LOG_TRACE(osstream);
    return rsmi_status;
  }

  return run_amdgpu_property_reinforcement_query(amdgpu_property_query);
}

rsmi_status_t Device::run_amdgpu_property_reinforcement_query(const AMDGpuPropertyQuery_t& amdgpu_property_query)
{
  std::ostringstream osstream;
  auto rsmi_status(rsmi_status_t::RSMI_STATUS_UNKNOWN_ERROR);

  auto contains = [](const uint16_t asic_id) {
    return (amdgpu_property_reinforcement_list.find(asic_id) != amdgpu_property_reinforcement_list.end());
  };

  // Traverse through all values for a given key
  osstream << __PRETTY_FUNCTION__ << "| ======= start =======" << "\n";
  LOG_TRACE(osstream);
  if (contains(amdgpu_property_query.m_asic_id)) {
    osstream << __PRETTY_FUNCTION__ << "  asic id found in table: " << amdgpu_property_query.m_asic_id << "\n";
    auto itr_begin = amdgpu_property_reinforcement_list.lower_bound(amdgpu_property_query.m_asic_id);
    auto itr_end = amdgpu_property_reinforcement_list.upper_bound(amdgpu_property_query.m_asic_id);
    while (itr_begin != itr_end) {
      // Still same key, and...
      if (itr_begin->first == amdgpu_property_query.m_asic_id) {
        osstream << __PRETTY_FUNCTION__ << "  asic id found: " << itr_begin->first << "\n";
        // Pci_rev_id matches the filter or ALL Revisions
        if ((itr_begin->second.m_pci_rev_id == amdgpu_property_query.m_pci_rev_id) ||
            (itr_begin->second.m_pci_rev_id == kDevRevIDAll)) {
          osstream << __PRETTY_FUNCTION__ << "  asic rev.id found: " << itr_begin->second.m_pci_rev_id << "\n";
          // Do we have the property we are looking for?
          if (((amdgpu_property_query.m_property != 0) &&
               (itr_begin->second.m_property == amdgpu_property_query.m_property)) ||
              ((amdgpu_property_query.m_verb_id != AMDGpuVerbTypes_t::kNone) &&
               (itr_begin->second.m_verb_id == amdgpu_property_query.m_verb_id))) {
            osstream << __PRETTY_FUNCTION__
                     << "  property found: " << itr_begin->second.m_property
                     << "  verb found: " << static_cast<AMDGpuVerbId_t>(itr_begin->second.m_verb_id)
                     << " " << amdgpu_verb_check_list.at(amdgpu_property_query.m_verb_id)
                     << " should_be_available: " << itr_begin->second.m_should_be_available << "\n";
            // and if we do, should we consider it available, or forcefully
            // considered it unavailable
            osstream << __PRETTY_FUNCTION__ << "| ======= validating ======="
                      << ", Property found in the table for this device and flagged as *Not Available* : "
                      << " [query filters: ]"
                      << " device: " << amdgpu_property_query.m_dev_idx
                      << " asic id: " << amdgpu_property_query.m_asic_id
                      << " revision id: " << amdgpu_property_query.m_pci_rev_id
                      << " reinf.tbl.rev. id: " << itr_begin->second.m_pci_rev_id;
            //
            // The property is set in the reinforcement table to 'it should not be available'
            if (!itr_begin->second.m_should_be_available) {
              // If the property is found and set to not available
              // (rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED),
              // it should be all good (rsmi_status_t::RSMI_STATUS_SUCCESS);
              rsmi_status = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
              osstream << __PRETTY_FUNCTION__
                       << " should_be_available: " << itr_begin->second.m_should_be_available
                       << " result: " << rsmi_status << "\n";
              LOG_TRACE(osstream);
              return rsmi_status;
            }
            //
            // The property is set in the reinforcement table to 'it should be available'
            rsmi_status = rsmi_status_t::RSMI_STATUS_SUCCESS;
            osstream << __PRETTY_FUNCTION__
                      << " should_be_available: " << itr_begin->second.m_should_be_available
                      << " result: " << rsmi_status << "\n";
            LOG_TRACE(osstream);
            return rsmi_status;
          }
        }
      }
      itr_begin++;
    }
  }

  osstream << __PRETTY_FUNCTION__ << "| ======= end ======="
            << "Done searching for the Property in reinforcement table for this device: "
            << " device: " << amdgpu_property_query.m_dev_idx
            << " asic id: " << amdgpu_property_query.m_asic_id
            << " revision id: " << amdgpu_property_query.m_pci_rev_id
            << " property id: " << amdgpu_property_query.m_property
            << " error: " << rsmi_status;
  LOG_TRACE(osstream);
  return rsmi_status;
}


}  // namespace smi
}  // namespace amd
