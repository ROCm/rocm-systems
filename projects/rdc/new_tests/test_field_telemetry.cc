/*
 * Parameterized Field Telemetry Tests
 *
 * Data-driven tests that validate ALL RDC fields using metadata.
 * Add new fields by updating the registry - no new test code needed!
 */

#include <gtest/gtest.h>
#include <unistd.h>  // for sleep

#include "field_metadata.h"
#include "rdc_test_fixture.h"

namespace rdc {
namespace test {

/**
 * @brief Parameterized test fixture for field testing
 */
class FieldTelemetryTest : public RdcTestFixture,
                           public ::testing::WithParamInterface<FieldMetadata> {
 protected:
  // Helper to check if test should be skipped based on field metadata
  bool ShouldSkipField(const FieldMetadata& metadata) {
    // Skip if requires GPU but none available
    if ((metadata.flags & ValidationFlags::REQUIRES_GPU) && GetGpuCount() == 0) {
      return true;
    }

    // Skip if requires multiple GPUs
    if ((metadata.flags & ValidationFlags::REQUIRES_MULTI_GPU) && GetGpuCount() < 2) {
      return true;
    }

    // Skip CPU-only fields if running GPU-only tests
    if (metadata.flags & ValidationFlags::REQUIRES_CPU) {
      // For now, allow CPU fields but expect NOT_SUPPORTED
      return false;
    }

    return false;
  }
};

/**
 * Test that a field can be watched and queried
 */
TEST_P(FieldTelemetryTest, CanWatchAndQuery) {
  const FieldMetadata& field_meta = GetParam();

  if (ShouldSkipField(field_meta)) {
    GTEST_SKIP() << "Field " << field_meta.name << " requires unavailable hardware";
  }

  // Use first GPU for testing
  if (GetGpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  uint32_t gpu_index = GetGpuIndices()[0];

  // Create GPU group
  rdc_gpu_group_t group_id;
  std::vector<uint32_t> gpu_indices = {gpu_index};
  rdc_status_t result = CreateGpuGroup(gpu_indices, "test_group", &group_id);
  ASSERT_EQ(result, RDC_ST_OK) << "Failed to create GPU group";

  // Create field group with this field
  rdc_field_grp_t field_group_id;
  std::vector<rdc_field_t> fields = {field_meta.field_id};
  result = CreateFieldGroup(fields, "field_group", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK) << "Failed to create field group for " << field_meta.name;

  // Start watching (500ms intervals)
  result = WatchFields(group_id, field_group_id, 500000, 60.0, 10);
  ASSERT_EQ(result, RDC_ST_OK) << "Failed to watch field " << field_meta.name;

  // Wait for data collection
  sleep(2);

  // Query the field
  rdc_field_value value;
  result = GetLatestValue(gpu_index, field_meta.field_id, &value);

  // Check if field is supported
  if (field_meta.flags & ValidationFlags::ALLOW_NOT_SUPPORTED) {
    if (result == RDC_ST_NOT_SUPPORTED || result == RDC_ST_NOT_FOUND) {
      GTEST_SKIP() << "Field " << field_meta.name << " not supported on this hardware";
    }
  }

  ASSERT_EQ(result, RDC_ST_OK) << "Failed to get value for field " << field_meta.name << ": "
                               << GetStatusString(result);

  // Validate the field value using metadata
  std::string error_msg;
  bool is_valid = field_meta.Validate(value, error_msg);

  if (!is_valid) {
    if (value.type == INTEGER) {
      FAIL() << "Field " << field_meta.name << " validation failed: " << error_msg
             << " (value: " << value.value.l_int << ")";
    } else if (value.type == DOUBLE) {
      FAIL() << "Field " << field_meta.name << " validation failed: " << error_msg
             << " (value: " << value.value.dbl << ")";
    } else if (value.type == STRING) {
      FAIL() << "Field " << field_meta.name << " validation failed: " << error_msg
             << " (value: " << value.value.str << ")";
    } else {
      FAIL() << "Field " << field_meta.name << " validation failed: " << error_msg;
    }
  }

  // Print field value for debugging
  if (value.type == INTEGER) {
    std::cout << "Field " << field_meta.name << ": " << value.value.l_int << std::endl;
  } else if (value.type == DOUBLE) {
    std::cout << "Field " << field_meta.name << ": " << value.value.dbl << std::endl;
  } else if (value.type == STRING) {
    std::cout << "Field " << field_meta.name << ": " << value.value.str << std::endl;
  }
}

/**
 * Test that a field remains consistent across multiple queries
 */
TEST_P(FieldTelemetryTest, ConsistentAcrossQueries) {
  const FieldMetadata& field_meta = GetParam();

  if (ShouldSkipField(field_meta)) {
    GTEST_SKIP() << "Field " << field_meta.name << " requires unavailable hardware";
  }

  // Skip accumulator fields (they're supposed to change)
  if (field_meta.flags & ValidationFlags::ACCUMULATOR) {
    GTEST_SKIP() << "Field " << field_meta.name << " is an accumulator";
  }

  if (GetGpuCount() == 0) {
    GTEST_SKIP() << "No GPUs available";
  }

  uint32_t gpu_index = GetGpuIndices()[0];

  // Setup field watching
  rdc_gpu_group_t group_id;
  std::vector<uint32_t> gpu_indices = {gpu_index};
  rdc_status_t result = CreateGpuGroup(gpu_indices, "test_group", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_field_grp_t field_group_id;
  std::vector<rdc_field_t> fields = {field_meta.field_id};
  result = CreateFieldGroup(fields, "field_group", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = WatchFields(group_id, field_group_id, 100000, 60.0, 10);
  ASSERT_EQ(result, RDC_ST_OK);

  sleep(1);

  // Query multiple times
  constexpr int NUM_QUERIES = 3;
  std::vector<int64_t> int_values;
  std::vector<double> double_values;

  for (int i = 0; i < NUM_QUERIES; ++i) {
    rdc_field_value value;
    result = GetLatestValue(gpu_index, field_meta.field_id, &value);

    if (field_meta.flags & ValidationFlags::ALLOW_NOT_SUPPORTED) {
      if (result == RDC_ST_NOT_SUPPORTED || result == RDC_ST_NOT_FOUND) {
        GTEST_SKIP() << "Field not supported";
      }
    }

    ASSERT_EQ(result, RDC_ST_OK);

    if (value.type == INTEGER) {
      int_values.push_back(value.value.l_int);
    } else if (value.type == DOUBLE) {
      double_values.push_back(value.value.dbl);
    }

    usleep(100000);  // 100ms between queries
  }

  // Check consistency (allow small variations for dynamic fields)
  if (field_meta.expected_type == INTEGER && !int_values.empty()) {
    int64_t max_val = *std::max_element(int_values.begin(), int_values.end());
    int64_t min_val = *std::min_element(int_values.begin(), int_values.end());

    // Allow up to 20% variation for dynamic fields
    int64_t max_variation = std::abs(max_val) / 5;
    EXPECT_LE(max_val - min_val, max_variation)
        << "Field " << field_meta.name << " varies too much";
  }
}

// Instantiate tests for all fields in the registry
INSTANTIATE_TEST_SUITE_P(AllFields, FieldTelemetryTest,
                         ::testing::ValuesIn(FieldRegistry::Instance().GetAllFields()),
                         [](const ::testing::TestParamInfo<FieldMetadata>& info) {
                           return info.param.name;
                         });

}  // namespace test
}  // namespace rdc
