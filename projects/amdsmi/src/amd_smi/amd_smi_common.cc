// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_common.h"

#include "amd_smi/amdsmi.h"

namespace {

auto g_amdsmi_init_ref_count = int32_t(0);

}  // namespace

namespace amd::smi {

bool amdsmi_library_initialized() { return (g_amdsmi_init_ref_count > 0); }

void amdsmi_library_init_ref_acquire() { ++g_amdsmi_init_ref_count; }

bool amdsmi_library_init_ref_release() {
  if (g_amdsmi_init_ref_count == 0) {
    return false;
  }

  --g_amdsmi_init_ref_count;
  return (g_amdsmi_init_ref_count == 0);
}

amdsmi_status_t rsmi_to_amdsmi_status(rsmi_status_t status) {
  amdsmi_status_t amdsmi_status = AMDSMI_STATUS_MAP_ERROR;

  // Look for it in the map
  // If found: use the mapped value
  // If not found: return the map error established above
  auto search = amd::smi::rsmi_status_map.find(status);
  if (search != amd::smi::rsmi_status_map.end()) {
    amdsmi_status = search->second;
  }

  return amdsmi_status;
}

amdsmi_vram_type_t vram_type_value(unsigned type) {
  amdsmi_vram_type_t value = AMDSMI_VRAM_TYPE_UNKNOWN;

  auto search = amd::smi::vram_type_map.find(type);
  if (search != amd::smi::vram_type_map.end()) {
    value = search->second;
  }

  return value;
}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t esmi_to_amdsmi_status(esmi_status_t status) {
  amdsmi_status_t amdsmi_status = AMDSMI_STATUS_MAP_ERROR;

  // Look for it in the map
  // If found: use the mapped value
  // If not found: return the map error established above
  auto search = amd::smi::esmi_status_map.find(status);
  if (search != amd::smi::esmi_status_map.end()) {
    amdsmi_status = search->second;
  }

  return amdsmi_status;
}
#endif

amdsmi_status_t ainic_to_amdsmi_status(smi_nic_status_t status) {
  amdsmi_status_t amdsmi_status = AMDSMI_STATUS_MAP_ERROR;

  // Look for it in the map
  // If found: use the mapped value
  // If not found: return the map error established above
  if (auto search_itr = ainic_status_map.find(status); search_itr != ainic_status_map.end()) {
    amdsmi_status = search_itr->second;
  }

  return amdsmi_status;
}

}  // namespace amd::smi
