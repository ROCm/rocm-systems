// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_common.h"

#include <atomic>

#include "amd_smi/amdsmi.h"

namespace {

// Atomic because every API reads it, unlocked, while init/shut_down change it.
std::atomic<int32_t> g_amdsmi_init_ref_count{0};

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

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
amdsmi_status_t rsmi_to_amdsmi_status(rsmi_status_t status) {
  // Add a case for every new rsmi_status_t; a missing case fails the build (-Wswitch).
  // clang-format off
  switch (status) {
    case RSMI_STATUS_SUCCESS:             return AMDSMI_STATUS_SUCCESS;
    case RSMI_STATUS_INVALID_ARGS:        return AMDSMI_STATUS_INVAL;
    case RSMI_STATUS_NOT_SUPPORTED:       return AMDSMI_STATUS_NOT_SUPPORTED;
    case RSMI_STATUS_FILE_ERROR:          return AMDSMI_STATUS_FILE_ERROR;
    case RSMI_STATUS_PERMISSION:          return AMDSMI_STATUS_NO_PERM;
    case RSMI_STATUS_OUT_OF_RESOURCES:    return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case RSMI_STATUS_INTERNAL_EXCEPTION:  return AMDSMI_STATUS_INTERNAL_EXCEPTION;
    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS: return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
    // RSMI_INITIALIZATION_ERROR is an alias of RSMI_STATUS_INIT_ERROR.
    case RSMI_STATUS_INIT_ERROR:          return AMDSMI_STATUS_NOT_INIT;
    case RSMI_STATUS_NOT_YET_IMPLEMENTED: return AMDSMI_STATUS_NOT_YET_IMPLEMENTED;
    case RSMI_STATUS_NOT_FOUND:           return AMDSMI_STATUS_NOT_FOUND;
    case RSMI_STATUS_INSUFFICIENT_SIZE:   return AMDSMI_STATUS_INSUFFICIENT_SIZE;
    case RSMI_STATUS_INTERRUPT:           return AMDSMI_STATUS_INTERRUPT;
    case RSMI_STATUS_UNEXPECTED_SIZE:     return AMDSMI_STATUS_UNEXPECTED_SIZE;
    case RSMI_STATUS_NO_DATA:             return AMDSMI_STATUS_NO_DATA;
    case RSMI_STATUS_UNEXPECTED_DATA:     return AMDSMI_STATUS_UNEXPECTED_DATA;
    case RSMI_STATUS_BUSY:                return AMDSMI_STATUS_BUSY;
    case RSMI_STATUS_REFCOUNT_OVERFLOW:   return AMDSMI_STATUS_REFCOUNT_OVERFLOW;
    case RSMI_STATUS_DIRECTORY_NOT_FOUND: return AMDSMI_STATUS_DIRECTORY_NOT_FOUND;
    case RSMI_STATUS_SETTING_UNAVAILABLE: return AMDSMI_STATUS_SETTING_UNAVAILABLE;
    case RSMI_STATUS_AMDGPU_RESTART_ERR:  return AMDSMI_STATUS_AMDGPU_RESTART_ERR;
    case RSMI_STATUS_DRIVER_NOT_LOADED:   return AMDSMI_STATUS_DRIVER_NOT_LOADED;
    case RSMI_STATUS_IPC_ERROR:           return AMDSMI_STATUS_IPC_ERROR;
    case RSMI_STATUS_UNKNOWN_ERROR:       return AMDSMI_STATUS_UNKNOWN_ERROR;
  }
  // clang-format on
  // A value outside the enum matches no case.
  return AMDSMI_STATUS_MAP_ERROR;
}
#pragma GCC diagnostic pop

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

const char* esmi_status_message(esmi_status_t status) {
  switch (status) {
    case ESMI_HSMP_TIMEOUT:
      return "HSMP message timed out";
    default:
      return esmi_get_err_msg(status);
  }
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
