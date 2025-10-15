/*
 * RDC Diagnostics API Test Fixture
 *
 * Provides a reusable test fixture for RDC API testing in embedded mode.
 * Follows RAII principles and gtest best practices.
 */

#ifndef NEW_TESTS_RDC_TEST_FIXTURE_H_
#define NEW_TESTS_RDC_TEST_FIXTURE_H_

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "rdc/rdc.h"

namespace rdc {
namespace test {

/**
 * @brief Base test fixture for RDC embedded mode testing
 *
 * This fixture handles:
 * - RDC initialization and shutdown
 * - Embedded mode startup/stop
 * - GPU discovery
 * - Group and field group management
 * - Automatic cleanup on test completion
 *
 * Usage:
 *   class MyTest : public RdcTestFixture {
 *     // Test implementation
 *   };
 */
class RdcTestFixture : public ::testing::Test {
 protected:
  // Test fixture setup - runs before each test
  void SetUp() override;

  // Test fixture teardown - runs after each test
  void TearDown() override;

  // Helper methods for common operations

  /**
   * @brief Get the RDC handle for embedded mode
   * @return RDC handle
   */
  rdc_handle_t GetHandle() const { return rdc_handle_; }

  /**
   * @brief Get list of available GPU indices
   * @return Vector of GPU indices
   */
  const std::vector<uint32_t>& GetGpuIndices() const { return gpu_indices_; }

  /**
   * @brief Get the number of GPUs discovered
   * @return GPU count
   */
  uint32_t GetGpuCount() const { return gpu_count_; }

  /**
   * @brief Check if RDC was initialized successfully
   * @return true if initialized
   */
  bool IsInitialized() const { return is_initialized_; }

  /**
   * @brief Create a GPU group with specified GPUs
   * @param gpu_indices GPU indices to add to group
   * @param group_name Name for the group
   * @param group_id Output parameter for created group ID
   * @return RDC status
   */
  rdc_status_t CreateGpuGroup(const std::vector<uint32_t>& gpu_indices,
                              const std::string& group_name, rdc_gpu_group_t* group_id);

  /**
   * @brief Create a field group with specified fields
   * @param field_ids Field IDs to include in group
   * @param field_group_name Name for the field group
   * @param field_group_id Output parameter for created field group ID
   * @return RDC status
   */
  rdc_status_t CreateFieldGroup(const std::vector<rdc_field_t>& field_ids,
                                const std::string& field_group_name,
                                rdc_field_grp_t* field_group_id);

  /**
   * @brief Start watching fields
   * @param group_id GPU group to monitor
   * @param field_group_id Field group to monitor
   * @param update_freq Update frequency in microseconds (default: 1 second)
   * @param max_keep_age Maximum age of data to keep in seconds (default: 60s)
   * @param max_keep_samples Maximum number of samples to keep (default: 10)
   * @return RDC status
   */
  rdc_status_t WatchFields(rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
                           uint64_t update_freq = 1000000, double max_keep_age = 60.0,
                           uint32_t max_keep_samples = 10);

  /**
   * @brief Get latest value for a field
   * @param gpu_index GPU index to query
   * @param field_id Field ID to retrieve
   * @param value Output parameter for field value
   * @return RDC status
   */
  rdc_status_t GetLatestValue(uint32_t gpu_index, rdc_field_t field_id, rdc_field_value* value);

  /**
   * @brief Get field name as string
   * @param field_id Field ID
   * @return Field name string
   */
  std::string GetFieldName(rdc_field_t field_id);

  /**
   * @brief Get status string
   * @param status RDC status code
   * @return Status string
   */
  std::string GetStatusString(rdc_status_t status);

  // Cleanup helpers
  void CleanupGroups();
  void CleanupFieldGroups();

 protected:
  // RDC state
  rdc_handle_t rdc_handle_;
  bool is_initialized_;
  uint32_t gpu_count_;
  std::vector<uint32_t> gpu_indices_;

  // Tracking for cleanup
  std::vector<rdc_gpu_group_t> created_gpu_groups_;
  std::vector<rdc_field_grp_t> created_field_groups_;
  std::vector<std::pair<rdc_gpu_group_t, rdc_field_grp_t>> watched_fields_;
};

}  // namespace test
}  // namespace rdc

#endif  // NEW_TESTS_RDC_TEST_FIXTURE_H_
