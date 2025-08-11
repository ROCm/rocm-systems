/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "rdc_lib/impl/RdcPartitionImpl.h"

#include <stdio.h>
#include <string.h>

#include <iostream>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace amd {
namespace rdc {

rdc_status_t RdcPartitionImpl::rdc_instance_profile_get_impl(
    uint32_t entity_index, rdc_instance_resource_type_t resource_type,
    rdc_resource_profile_t* profile) {
  if (profile == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }
  profile->partition_resource = 0;
  profile->num_partitions_share_resource = 0;

  rdc_entity_info_t info = rdc_get_info_from_entity_index(entity_index);

  amdsmi_processor_handle proc_handle;
  // Get processor handle of socket
  amdsmi_status_t ret = get_processor_handle_from_id(info.device_index, &proc_handle);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return RDC_ST_UNKNOWN_ERROR;
  }

  amdsmi_accelerator_partition_profile_config_t config;
  memset(&config, 0, sizeof(config));
  ret = amdsmi_get_gpu_accelerator_partition_profile_config(proc_handle, &config);
  if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
    return RDC_ST_OK;
  } else if (ret != AMDSMI_STATUS_SUCCESS) {
    return RDC_ST_UNKNOWN_ERROR;
  }

  amdsmi_accelerator_partition_profile_t active_profile;
  memset(&active_profile, 0, sizeof(active_profile));
  uint32_t num = 0;  // This is unused
  ret = amdsmi_get_gpu_accelerator_partition_profile(proc_handle, &active_profile, &num);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return RDC_ST_UNKNOWN_ERROR;
  }

  // If physical device, use profile 0 to get all XCC's/Decoders
  uint32_t lookup_id =
      (info.entity_role == RDC_DEVICE_ROLE_PARTITION_INSTANCE) ? active_profile.profile_index : 0;

  // Map rdc resource type to smi
  amdsmi_accelerator_partition_resource_type_t smi_resource;
  switch (resource_type) {
    case RDC_ACCELERATOR_XCC:
      smi_resource = AMDSMI_ACCELERATOR_XCC;
      break;
    case RDC_ACCELERATOR_DECODER:
      smi_resource = AMDSMI_ACCELERATOR_DECODER;
      break;
    default:
      return RDC_ST_NOT_SUPPORTED;
  }

  bool found = false;
  uint32_t total_resource = 0;
  uint32_t resource_share = 0;
  for (uint32_t i = 0; i < AMDSMI_MAX_CP_PROFILE_RESOURCES; i++) {
    const auto& res = config.resource_profiles[i];
    if (res.profile_index == lookup_id && res.resource_type == smi_resource) {
      total_resource = res.partition_resource;
      resource_share = res.num_partitions_share_resource;
      found = true;
      break;
    }
  }
  if (!found) {
    return RDC_ST_UNKNOWN_ERROR;
  }
  profile->partition_resource = total_resource;
  profile->num_partitions_share_resource = resource_share;

  return RDC_ST_OK;
}

rdc_status_t RdcPartitionImpl::rdc_get_num_partition_impl(uint32_t index, uint16_t* num_partition) {
  if (get_num_partition(index, num_partition) != AMDSMI_STATUS_SUCCESS) {
    return RDC_ST_UNKNOWN_ERROR;
  }
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
