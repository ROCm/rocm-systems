/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"

#include <ctime>

#include "amd_smi/amdsmi.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcPartitionImpl.h"
#include "rdc_lib/impl/SmiUtils.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcGroupSettingsImpl::RdcGroupSettingsImpl(const RdcPartitionPtr& partition)
    : partition_(partition) {
  // Add the default job stats fields
  rdc_field_t job_fields[] = {RDC_FI_GPU_MEMORY_USAGE, RDC_FI_POWER_USAGE, RDC_FI_GPU_CLOCK,
                              RDC_FI_GPU_UTIL,         RDC_FI_PCIE_TX,     RDC_FI_PCIE_RX,
                              RDC_FI_PCIE_BANDWIDTH,   RDC_FI_MEM_CLOCK,   RDC_FI_GPU_TEMP};
  char job_field_group[] = "JobStatsFields";
  rdc_field_grp_t fgid = JOB_FIELD_ID;

  rdc_group_field_create(sizeof(job_fields) / sizeof(uint32_t), job_fields, job_field_group, &fgid);
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_gpu_create(const char* group_name,
                                                        rdc_gpu_group_t* p_rdc_group_id) {
  RDC_LOG(RDC_DEBUG, "Create group " << group_name);
  rdc_group_info_t ginfo;
  strncpy_with_null(ginfo.group_name, group_name, RDC_MAX_STR_LENGTH);
  ginfo.count = 0;

  std::lock_guard<std::mutex> guard(group_mutex_);
  if (gpu_group_.size() >= RDC_MAX_NUM_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  gpu_group_.emplace(cur_group_id_, ginfo);
  *p_rdc_group_id = cur_group_id_;
  cur_group_id_++;

  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) {
  std::lock_guard<std::mutex> guard(group_mutex_);
  if (!gpu_group_.erase(p_rdc_group_id)) return RDC_ST_NOT_FOUND;
  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_gpu_add(rdc_gpu_group_t groupId, uint32_t gpu_index) {
  std::lock_guard<std::mutex> guard(group_mutex_);
  auto ite = gpu_group_.find(groupId);
  if (ite == gpu_group_.end()) {
    return RDC_ST_NOT_FOUND;
  }

  rdc_entity_info_t entity_info = rdc_get_info_from_entity_index(gpu_index);

  uint16_t num_partitions = 0;
  rdc_status_t status =
      partition_->rdc_get_num_partition_impl(entity_info.device_index, &num_partitions);
  if (status != RDC_ST_OK) {
    return status;
  }

  if (num_partitions != UINT16_MAX && num_partitions > 1) {
    if (entity_info.entity_role == RDC_DEVICE_ROLE_PARTITION_INSTANCE) {
      if (entity_info.instance_index >= num_partitions) {
        RDC_LOG(RDC_INFO, "Invalid partition instance: GPU "
                              << entity_info.device_index << " supports " << num_partitions
                              << " partitions, but instance index is "
                              << entity_info.instance_index);
        return RDC_ST_BAD_PARAMETER;
      }
    }
  } else {
    if (entity_info.entity_role != RDC_DEVICE_ROLE_PHYSICAL) {
      RDC_LOG(RDC_INFO, "GPU " << entity_info.device_index
                               << " is not partitionable, but a partition instance was provided.");
      return RDC_ST_BAD_PARAMETER;
    }
  }

  // Check whether the index already exists
  for (uint32_t i = 0; i < ite->second.count; i++) {
    if (ite->second.entity_ids[i] == gpu_index) {
      RDC_LOG(RDC_INFO, "Fail to add " << gpu_index << " to GPU group " << groupId
                                       << " as it is already exists");
      return RDC_ST_BAD_PARAMETER;
    }
  }
  if (ite->second.count < RDC_GROUP_MAX_ENTITIES) {
    ite->second.entity_ids[ite->second.count] = gpu_index;
    ite->second.count++;
  } else {
    return RDC_ST_MAX_LIMIT;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
                                                          rdc_group_info_t* p_rdc_group_info) {
  std::lock_guard<std::mutex> guard(group_mutex_);
  auto ite = gpu_group_.find(p_rdc_group_id);
  if (ite != gpu_group_.end()) {
    auto info = ite->second;
    strncpy_with_null(p_rdc_group_info->group_name, info.group_name, RDC_MAX_STR_LENGTH);
    p_rdc_group_info->count = info.count;
    for (uint32_t i = 0; i < info.count; i++) {
      p_rdc_group_info->entity_ids[i] = info.entity_ids[i];
    }
  } else {
    return RDC_ST_NOT_FOUND;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_get_all_ids(rdc_gpu_group_t group_id_list[],
                                                         uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }

  *count = 0;
  std::lock_guard<std::mutex> guard(group_mutex_);
  auto ite = gpu_group_.begin();
  for (; ite != gpu_group_.end(); ite++) {
    if (*count >= RDC_MAX_NUM_GROUPS) {
      return RDC_ST_MAX_LIMIT;
    }
    group_id_list[*count] = ite->first;
    (*count)++;
  }

  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_field_create(uint32_t num_field_ids,
                                                          rdc_field_t* field_ids,
                                                          const char* field_group_name,
                                                          rdc_field_grp_t* rdc_field_group_id) {
  RDC_LOG(RDC_DEBUG, "Create field group " << field_group_name);
  rdc_field_group_info_t finfo;
  finfo.count = num_field_ids;
  strncpy_with_null(finfo.group_name, field_group_name, RDC_MAX_STR_LENGTH);
  if (num_field_ids <= RDC_MAX_FIELD_IDS_PER_FIELD_GROUP) {
    for (uint32_t i = 0; i < num_field_ids; i++) {
      finfo.field_ids[i] = field_ids[i];
    }
  } else {
    return RDC_ST_MAX_LIMIT;
  }

  std::lock_guard<std::mutex> guard(field_group_mutex_);
  if (field_group_.size() >= RDC_MAX_NUM_FIELD_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  field_group_.emplace(cur_field_group_id_, finfo);
  *rdc_field_group_id = cur_field_group_id_;
  cur_field_group_id_++;

  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) {
  if (rdc_field_group_id == JOB_FIELD_ID) {
    RDC_LOG(RDC_INFO, "Cannot delete system JOB_FIELD_ID field group");
    return RDC_ST_BAD_PARAMETER;
  }
  std::lock_guard<std::mutex> guard(field_group_mutex_);
  if (!field_group_.erase(rdc_field_group_id)) return RDC_ST_NOT_FOUND;
  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_field_get_info(
    rdc_field_grp_t rdc_field_group_id, rdc_field_group_info_t* field_group_info) {
  std::lock_guard<std::mutex> guard(field_group_mutex_);
  auto ite = field_group_.find(rdc_field_group_id);
  if (ite != field_group_.end()) {
    auto info = ite->second;
    strncpy_with_null(field_group_info->group_name, info.group_name, RDC_MAX_STR_LENGTH);
    field_group_info->count = info.count;
    for (uint32_t i = 0; i < info.count; i++) {
      field_group_info->field_ids[i] = info.field_ids[i];
    }
  } else {
    return RDC_ST_NOT_FOUND;
  }
  return RDC_ST_OK;
}

rdc_status_t RdcGroupSettingsImpl::rdc_group_field_get_all_ids(
    rdc_field_grp_t field_group_id_list[], uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }

  *count = 0;
  std::lock_guard<std::mutex> guard(field_group_mutex_);
  auto ite = field_group_.begin();
  for (; ite != field_group_.end(); ite++) {
    if (*count >= RDC_MAX_NUM_FIELD_GROUPS) {
      return RDC_ST_MAX_LIMIT;
    }

    // Skip system defined JOB_FIELD_ID
    if (ite->first == JOB_FIELD_ID) continue;

    field_group_id_list[*count] = ite->first;
    (*count)++;
  }

  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
