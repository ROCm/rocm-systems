/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_cuid_interface.h"
#include "inc/hsa_ext_amd.h"
#include "core/inc/agent.h"

#ifdef HSA_ENABLE_AMDCUID_SUPPORT
#include "amd_cuid.h"
#endif

#include <cstdint>
#include <string>

namespace rocr {
namespace core {

#ifdef HSA_ENABLE_AMDCUID_SUPPORT

hsa_status_t CuidInterface::QueryGpuCuid(const std::string& device_node, uint8_t* cuid,
                                         uint32_t* cuid_length) {
  if (!cuid || !cuid_length || device_node.empty()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Retrieve the handle for a GPU device using its system path
  amdcuid_id_t handle{};
  amdcuid_status_t status =
      amdcuid_get_handle_by_dev_path(device_node.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle);
  if (status != AMDCUID_STATUS_SUCCESS) return HSA_STATUS_ERROR;

  // Query the derived CUID using the device handle
  status = amdcuid_query_device_property(handle, AMDCUID_QUERY_DERIVED_CUID, cuid, cuid_length);

  return (status == AMDCUID_STATUS_SUCCESS) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

#else  // HSA_ENABLE_AMDCUID_SUPPORT

hsa_status_t CuidInterface::QueryGpuCuid(const std::string&, uint8_t*, uint32_t) {
  return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
}

#endif  // HSA_ENABLE_AMDCUID_SUPPORT

}  // namespace core
}  // namespace rocr