/*
 * Category-Based Field Tests
 *
 * Organized tests by field category for targeted testing.
 * Useful for testing specific subsystems.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include "field_metadata.h"
#include "rdc_test_fixture.h"

namespace rdc {
namespace test {

/**
 * @brief Batch test multiple fields from a category
 */
class CategoryBatchTest : public RdcTestFixture,
                          public ::testing::WithParamInterface<FieldCategory> {
 protected:
  // Test all fields in a category
  void TestCategory(FieldCategory category) {
    if (GetGpuCount() == 0) {
      GTEST_SKIP() << "No GPUs available";
    }

    auto fields_in_category = FieldRegistry::Instance().GetFieldsByCategory(category);
    if (fields_in_category.empty()) {
      GTEST_SKIP() << "No fields in category " << GetCategoryName(category);
    }

    std::cout << "Testing " << fields_in_category.size() << " fields in category "
              << GetCategoryName(category) << std::endl;

    uint32_t gpu_index = GetGpuIndices()[0];

    // Create GPU group
    rdc_gpu_group_t group_id;
    std::vector<uint32_t> gpu_indices = {gpu_index};
    rdc_status_t result = CreateGpuGroup(gpu_indices, "batch_group", &group_id);
    ASSERT_EQ(result, RDC_ST_OK);

    // Create field group with all fields in category
    std::vector<rdc_field_t> field_ids;
    for (const auto& field_meta : fields_in_category) {
      field_ids.push_back(field_meta.field_id);
    }

    rdc_field_grp_t field_group_id;
    result = CreateFieldGroup(field_ids, "category_fields", &field_group_id);
    ASSERT_EQ(result, RDC_ST_OK);

    // Start watching all fields
    result = WatchFields(group_id, field_group_id, 500000, 60.0, 10);
    ASSERT_EQ(result, RDC_ST_OK);

    // Wait for collection
    sleep(2);

    // Query and validate each field
    int successful = 0;
    int not_supported = 0;
    int failed = 0;

    for (const auto& field_meta : fields_in_category) {
      rdc_field_value value;
      result = GetLatestValue(gpu_index, field_meta.field_id, &value);

      if (result == RDC_ST_NOT_SUPPORTED || result == RDC_ST_NOT_FOUND) {
        not_supported++;
        continue;
      }

      if (result != RDC_ST_OK) {
        std::cerr << "Failed to query " << field_meta.name << ": " << GetStatusString(result)
                  << std::endl;
        failed++;
        continue;
      }

      std::string error_msg;
      if (!field_meta.Validate(value, error_msg)) {
        std::cerr << "Validation failed for " << field_meta.name << ": " << error_msg << std::endl;
        failed++;
      } else {
        successful++;
      }
    }

    std::cout << "Results: " << successful << " successful, " << not_supported << " not supported, "
              << failed << " failed" << std::endl;

    // Expect at least some fields to be supported
    EXPECT_GT(successful, 0) << "No fields supported in category " << GetCategoryName(category);
  }
};

TEST_P(CategoryBatchTest, BatchTestCategory) { TestCategory(GetParam()); }

// Instantiate for GPU categories
INSTANTIATE_TEST_SUITE_P(GPUCategories, CategoryBatchTest,
                         ::testing::Values(FieldCategory::GPU_IDENTIFICATION,
                                           FieldCategory::GPU_FREQUENCY, FieldCategory::GPU_THERMAL,
                                           FieldCategory::GPU_POWER, FieldCategory::GPU_UTILIZATION,
                                           FieldCategory::GPU_MEMORY),
                         [](const ::testing::TestParamInfo<FieldCategory>& info) {
                           return GetCategoryName(info.param);
                         });

/**
 * @brief Test that tests critical GPU fields quickly
 */
TEST_F(RdcTestFixture, CriticalGpuFields) {
  if (GetGpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  // Critical fields that should always work on any GPU
  std::vector<rdc_field_t> critical_fields = {RDC_FI_GPU_COUNT,   RDC_FI_DEV_NAME,
                                              RDC_FI_GPU_CLOCK,   RDC_FI_GPU_TEMP,
                                              RDC_FI_POWER_USAGE, RDC_FI_GPU_MEMORY_TOTAL};

  uint32_t gpu_index = GetGpuIndices()[0];

  // Setup watching
  rdc_gpu_group_t group_id;
  std::vector<uint32_t> gpu_indices = {gpu_index};
  rdc_status_t result = CreateGpuGroup(gpu_indices, "critical_group", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_field_grp_t field_group_id;
  result = CreateFieldGroup(critical_fields, "critical_fields", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = WatchFields(group_id, field_group_id, 500000, 60.0, 10);
  ASSERT_EQ(result, RDC_ST_OK);

  sleep(2);

  // Verify all critical fields work
  for (auto field_id : critical_fields) {
    rdc_field_value value;
    result = GetLatestValue(gpu_index, field_id, &value);

    auto* field_meta = FieldRegistry::Instance().GetField(field_id);
    ASSERT_NE(field_meta, nullptr) << "Field not in registry";

    EXPECT_EQ(result, RDC_ST_OK) << "Critical field " << field_meta->name
                                 << " failed: " << GetStatusString(result);

    if (result == RDC_ST_OK) {
      std::cout << field_meta->name << ": ";
      if (value.type == INTEGER) {
        std::cout << value.value.l_int;
      } else if (value.type == DOUBLE) {
        std::cout << value.value.dbl;
      } else if (value.type == STRING) {
        std::cout << value.value.str;
      }
      std::cout << std::endl;
    }
  }
}

/**
 * @brief Bulk test - watch many fields simultaneously
 */
TEST_F(RdcTestFixture, BulkFieldWatching) {
  if (GetGpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  // Get first 30 displayable fields
  auto all_fields = FieldRegistry::Instance().GetDisplayableFields();
  if (all_fields.empty()) {
    GTEST_SKIP() << "No fields available";
  }

  size_t field_count = std::min(all_fields.size(), size_t(30));
  std::vector<rdc_field_t> field_ids;
  for (size_t i = 0; i < field_count; ++i) {
    field_ids.push_back(all_fields[i].field_id);
  }

  std::cout << "Testing bulk watching of " << field_count << " fields" << std::endl;

  uint32_t gpu_index = GetGpuIndices()[0];

  // Setup
  rdc_gpu_group_t group_id;
  std::vector<uint32_t> gpu_indices = {gpu_index};
  rdc_status_t result = CreateGpuGroup(gpu_indices, "bulk_group", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_field_grp_t field_group_id;
  result = CreateFieldGroup(field_ids, "bulk_fields", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = WatchFields(group_id, field_group_id, 1000000, 60.0, 10);
  ASSERT_EQ(result, RDC_ST_OK);

  sleep(3);

  // Query all fields
  int successful = 0;
  for (size_t i = 0; i < field_count; ++i) {
    rdc_field_value value;
    result = GetLatestValue(gpu_index, field_ids[i], &value);

    if (result == RDC_ST_OK) {
      successful++;
    }
  }

  std::cout << "Successfully queried " << successful << " out of " << field_count << " fields"
            << std::endl;

  // Expect at least 50% success rate
  EXPECT_GE(successful, static_cast<int>(field_count) / 2) << "Too many fields failed in bulk test";
}

}  // namespace test
}  // namespace rdc
