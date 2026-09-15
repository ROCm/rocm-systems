// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_PROPERTIES_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_PROPERTIES_H_

#include <cstdint>
#include <map>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

//
// Property reinforcement check list
//
using AMDGpuPropertyId_t = uint32_t;
using AMDGpuDevIdx_t = uint32_t;
using AMDGpuVerbId_t = uint32_t;
using AMDGpuAsicId_t = uint16_t;
using AMDGpuAsicRevId_t = uint16_t;
using AMDGpuOpModeType_t = uint8_t;

enum class AMDGpuVerbTypes_t : AMDGpuVerbId_t {
  kNone = 0,
  kSetGpuPciBandwidth,
  kSetPowerCap,
  kSetGpuPowerProfile,
  kSetGpuOdClkInfo,
  kSetGpuOdVoltInfo,
  kSetGpuPerfLevelV1,
  kSetGpuPerfLevel,
  kGetGpuPowerProfilePresets,
  kResetGpu,
  kSetGpuPerfDeterminismMode,
  kSetGpuFanSpeed,
  kResetGpuFan,
  kSetClkFreq,
  kSetGpuOverdriveLevelV1,
  kSetGpuOverdriveLevel,
  kGetGpuFanRpms,
  kGetGpuFanSpeed,
  kGetGpuFanSpeedMax,
  kGetGpuVoltMetric,
  kGetGpuOverDriveLevel,
  kGetGpuOdVoltInfo,
  kGetGpuOdVoltCurveRegions,
};
using AMDGpuVerbList_t = std::map<AMDGpuVerbTypes_t, std::string>;

enum class AMDGpuPropertyTypesOffset_t : AMDGpuPropertyId_t {
  kNone = 0,
  kDevInfoTypes = (0x1000 << 0),
  kMonitorTypes = (0x1000 << 1),
  kPerfTypes = (0x1000 << 2),
  kClkTypes = (0x1000 << 3),
  kVoltMetricTypes = (0x1000 << 4),
};

using AMDGpuPropertyOffsetType = std::underlying_type<AMDGpuPropertyTypesOffset_t>::type;
using AMDGpuPropertyTypesOffsetList_t = std::map<AMDGpuPropertyTypesOffset_t, std::string>;
AMDGpuPropertyTypesOffset_t operator|(AMDGpuPropertyTypesOffset_t lhs,
                                      AMDGpuPropertyTypesOffset_t rhs);
AMDGpuPropertyTypesOffset_t operator&(AMDGpuPropertyTypesOffset_t lhs,
                                      AMDGpuPropertyTypesOffset_t rhs);

enum class AMDGpuPropertyOpModeTypes_t : AMDGpuOpModeType_t {
  kBareMetal = (0x1 << 0),
  kSrIov = (0x1 << 1),
  kBoth = (0x1 << 2),
};

using AMDGpuPropertyOpModeType = std::underlying_type<AMDGpuPropertyOpModeTypes_t>::type;
using AMDGpuOpModeList_t = std::map<AMDGpuPropertyOpModeTypes_t, std::string>;
AMDGpuPropertyOpModeTypes_t operator|(AMDGpuPropertyOpModeTypes_t lhs,
                                      AMDGpuPropertyOpModeTypes_t rhs);
AMDGpuPropertyOpModeTypes_t operator&(AMDGpuPropertyOpModeTypes_t lhs,
                                      AMDGpuPropertyOpModeTypes_t rhs);

struct AMDGpuProperties_t {
  AMDGpuAsicRevId_t m_pci_rev_id;
  AMDGpuPropertyId_t m_property;
  AMDGpuVerbTypes_t m_verb_id;
  AMDGpuPropertyOpModeTypes_t m_opmode;
  bool m_should_be_available;
};
using AMDGpuPropertyList_t = std::multimap<AMDGpuAsicId_t, AMDGpuProperties_t>;

struct AMDGpuPropertyQuery_t {
  AMDGpuAsicId_t m_asic_id;
  AMDGpuAsicRevId_t m_pci_rev_id;
  AMDGpuDevIdx_t m_dev_idx;
  AMDGpuPropertyId_t m_property;
  AMDGpuVerbTypes_t m_verb_id;
};

//
AMDGpuPropertyId_t make_unique_property_id(AMDGpuPropertyTypesOffset_t type_offset,
                                           AMDGpuPropertyId_t property_id);
AMDGpuPropertyId_t unmake_unique_property_id(AMDGpuPropertyId_t property_id);

rsmi_status_t validate_property_reinforcement_query(uint32_t dv_ind,
                                                    AMDGpuVerbTypes_t dev_info_type,
                                                    rsmi_status_t actual_error_code);

void dump_amdgpu_property_reinforcement_list();

}  // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_DEVICE_H_
