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

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_PROPERTIES_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_PROPERTIES_H_

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi.h"

#include <cstdint>
#include <map>


namespace amd {
namespace smi {

//
// Property reinforcement check list
//
using AMDGpuPropertyId_t = uint32_t;
using AMDGpuDevIdx_t     = uint32_t;
using AMDGpuVerbId_t     = uint32_t;
using AMDGpuAsicId_t     = uint16_t;
using AMDGpuAsicRevId_t  = uint16_t;
using AMDGpuOpModeType_t = uint8_t;

enum class AMDGpuVerbTypes_t : AMDGpuVerbId_t
{
  kNone = 0,
  kSetGpuPciBandwidth,
  kSetPowerCap,
  kSetGpuPowerProfile,
  kSetGpuClkRange,
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


enum class AMDGpuPropertyTypesOffset_t : AMDGpuPropertyId_t
{
  kNone = 0,
  kDevInfoTypes    = (0x1000 << 0),
  kMonitorTypes    = (0x1000 << 1),
  kPerfTypes       = (0x1000 << 2),
  kClkTypes        = (0x1000 << 3),
  kVoltMetricTypes = (0x1000 << 4),
};

using AMDGpuPropertyOffsetType = std::underlying_type<AMDGpuPropertyTypesOffset_t>::type;
using AMDGpuPropertyTypesOffsetList_t = std::map<AMDGpuPropertyTypesOffset_t, std::string>;
AMDGpuPropertyTypesOffset_t operator| (AMDGpuPropertyTypesOffset_t lhs, AMDGpuPropertyTypesOffset_t rhs);
AMDGpuPropertyTypesOffset_t operator& (AMDGpuPropertyTypesOffset_t lhs, AMDGpuPropertyTypesOffset_t rhs);


enum class AMDGpuPropertyOpModeTypes_t : AMDGpuOpModeType_t
{
  kBareMetal  = (0x1 << 0),
  kSrIov      = (0x1 << 1),
  kBoth       = (0x1 << 2),
};

using AMDGpuPropertyOpModeType = std::underlying_type<AMDGpuPropertyOpModeTypes_t>::type;
using AMDGpuOpModeList_t = std::map<AMDGpuPropertyOpModeTypes_t, std::string>;
AMDGpuPropertyOpModeTypes_t operator| (AMDGpuPropertyOpModeTypes_t lhs, AMDGpuPropertyOpModeTypes_t rhs);
AMDGpuPropertyOpModeTypes_t operator& (AMDGpuPropertyOpModeTypes_t lhs, AMDGpuPropertyOpModeTypes_t rhs);


struct AMDGpuProperties_t
{
  AMDGpuAsicRevId_t m_pci_rev_id;
  AMDGpuPropertyId_t m_property;
  AMDGpuVerbTypes_t m_verb_id;
  AMDGpuPropertyOpModeTypes_t m_opmode;
  bool m_should_be_available;
};
using AMDGpuPropertyList_t = std::multimap<AMDGpuAsicId_t, AMDGpuProperties_t>;

struct AMDGpuPropertyQuery_t
{
  AMDGpuAsicId_t m_asic_id;
  AMDGpuAsicRevId_t m_pci_rev_id;
  AMDGpuDevIdx_t m_dev_idx;
  AMDGpuPropertyId_t m_property;
  AMDGpuVerbTypes_t m_verb_id;
};


//
AMDGpuPropertyId_t make_unique_property_id(AMDGpuPropertyTypesOffset_t type_offset, AMDGpuPropertyId_t property_id);
AMDGpuPropertyId_t unmake_unique_property_id(AMDGpuPropertyId_t property_id);

rsmi_status_t validate_property_reinforcement_query(uint32_t dv_ind,
                                                    AMDGpuVerbTypes_t dev_info_type,
                                                    rsmi_status_t actual_error_code);

void dump_amdgpu_property_reinforcement_list();


}   // namespace smi
}   // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_DEVICE_H_
