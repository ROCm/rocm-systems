/*
 * RDC Diagnostics API Test Fixture Implementation
 */

#include "rdc_test_fixture.h"

#include <unistd.h>  // for sleep

#include <cstring>
#include <iostream>

namespace rdc {
namespace test {

void RdcTestFixture::SetUp() {
  rdc_handle_ = nullptr;
  is_initialized_ = false;
  gpu_count_ = 0;
  gpu_indices_.clear();
  created_gpu_groups_.clear();
  created_field_groups_.clear();
  watched_fields_.clear();

  // Initialize RDC
  rdc_status_t result = rdc_init(0);
  ASSERT_EQ(result, RDC_ST_OK) << "Failed to initialize RDC: " << rdc_status_string(result);

  // Start in embedded mode with AUTO operation mode
  // AUTO mode = background threads automatically collect metrics
  result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle_);
  ASSERT_EQ(result, RDC_ST_OK) << "Failed to start embedded mode: " << rdc_status_string(result);
  ASSERT_NE(rdc_handle_, nullptr) << "RDC handle is null";

  is_initialized_ = true;

  // Discover available GPUs
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  result = rdc_device_get_all(rdc_handle_, gpu_index_list, &gpu_count_);

  // It's OK if there are no GPUs (for CI environments)
  if (result == RDC_ST_OK && gpu_count_ > 0) {
    gpu_indices_.assign(gpu_index_list, gpu_index_list + gpu_count_);
    std::cout << "Discovered " << gpu_count_ << " GPU(s)" << std::endl;
  } else if (result != RDC_ST_OK) {
    std::cout << "Warning: GPU discovery failed: " << rdc_status_string(result) << std::endl;
  } else {
    std::cout << "Warning: No GPUs discovered" << std::endl;
  }
}

void RdcTestFixture::TearDown() {
  if (!is_initialized_) {
    return;
  }

  // Unwatch all watched fields
  for (const auto& watch_pair : watched_fields_) {
    rdc_field_unwatch(rdc_handle_, watch_pair.first, watch_pair.second);
  }
  watched_fields_.clear();

  // Cleanup field groups
  CleanupFieldGroups();

  // Cleanup GPU groups
  CleanupGroups();

  // Stop embedded mode
  if (rdc_handle_ != nullptr) {
    rdc_status_t result = rdc_stop_embedded(rdc_handle_);
    if (result != RDC_ST_OK) {
      std::cerr << "Warning: Failed to stop embedded mode: " << rdc_status_string(result)
                << std::endl;
    }
    rdc_handle_ = nullptr;
  }

  // Shutdown RDC
  rdc_status_t result = rdc_shutdown();
  if (result != RDC_ST_OK) {
    std::cerr << "Warning: Failed to shutdown RDC: " << rdc_status_string(result) << std::endl;
  }

  is_initialized_ = false;
}

rdc_status_t RdcTestFixture::CreateGpuGroup(const std::vector<uint32_t>& gpu_indices,
                                            const std::string& group_name,
                                            rdc_gpu_group_t* group_id) {
  if (!is_initialized_ || group_id == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Create empty group
  rdc_status_t result =
      rdc_group_gpu_create(rdc_handle_, RDC_GROUP_EMPTY, group_name.c_str(), group_id);

  if (result != RDC_ST_OK) {
    return result;
  }

  // Track for cleanup
  created_gpu_groups_.push_back(*group_id);

  // Add GPUs to group
  for (uint32_t gpu_index : gpu_indices) {
    result = rdc_group_gpu_add(rdc_handle_, *group_id, gpu_index);
    if (result != RDC_ST_OK) {
      return result;
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcTestFixture::CreateFieldGroup(const std::vector<rdc_field_t>& field_ids,
                                              const std::string& field_group_name,
                                              rdc_field_grp_t* field_group_id) {
  if (!is_initialized_ || field_group_id == nullptr || field_ids.empty()) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Create field group
  rdc_status_t result = rdc_group_field_create(rdc_handle_, field_ids.size(),
                                               const_cast<rdc_field_t*>(field_ids.data()),
                                               field_group_name.c_str(), field_group_id);

  if (result == RDC_ST_OK) {
    // Track for cleanup
    created_field_groups_.push_back(*field_group_id);
  }

  return result;
}

rdc_status_t RdcTestFixture::WatchFields(rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
                                         uint64_t update_freq, double max_keep_age,
                                         uint32_t max_keep_samples) {
  if (!is_initialized_) {
    return RDC_ST_BAD_PARAMETER;
  }

  rdc_status_t result = rdc_field_watch(rdc_handle_, group_id, field_group_id, update_freq,
                                        max_keep_age, max_keep_samples);

  if (result == RDC_ST_OK) {
    // Track for cleanup
    watched_fields_.push_back(std::make_pair(group_id, field_group_id));
  }

  return result;
}

rdc_status_t RdcTestFixture::GetLatestValue(uint32_t gpu_index, rdc_field_t field_id,
                                            rdc_field_value* value) {
  if (!is_initialized_ || value == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  return rdc_field_get_latest_value(rdc_handle_, gpu_index, field_id, value);
}

std::string RdcTestFixture::GetFieldName(rdc_field_t field_id) {
  const char* name = field_id_string(field_id);
  return name ? std::string(name) : std::string("UNKNOWN");
}

std::string RdcTestFixture::GetStatusString(rdc_status_t status) {
  const char* status_str = rdc_status_string(status);
  return status_str ? std::string(status_str) : std::string("UNKNOWN");
}

void RdcTestFixture::CleanupGroups() {
  for (auto group_id : created_gpu_groups_) {
    rdc_status_t result = rdc_group_gpu_destroy(rdc_handle_, group_id);
    if (result != RDC_ST_OK) {
      std::cerr << "Warning: Failed to destroy GPU group " << group_id << ": "
                << rdc_status_string(result) << std::endl;
    }
  }
  created_gpu_groups_.clear();
}

void RdcTestFixture::CleanupFieldGroups() {
  for (auto field_group_id : created_field_groups_) {
    rdc_status_t result = rdc_group_field_destroy(rdc_handle_, field_group_id);
    if (result != RDC_ST_OK) {
      std::cerr << "Warning: Failed to destroy field group " << field_group_id << ": "
                << rdc_status_string(result) << std::endl;
    }
  }
  created_field_groups_.clear();
}

}  // namespace test
}  // namespace rdc
