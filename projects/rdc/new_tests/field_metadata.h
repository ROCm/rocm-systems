/*
 * Field Metadata System for RDC API Tests
 *
 * Provides a data-driven approach to testing all RDC fields.
 * Makes it trivial to add new fields without writing new test code.
 */

#ifndef NEW_TESTS_FIELD_METADATA_H_
#define NEW_TESTS_FIELD_METADATA_H_

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rdc/rdc.h"

namespace rdc {
namespace test {

/**
 * @brief Field categories for organizing tests
 */
enum class FieldCategory {
  GPU_IDENTIFICATION,  // GPU count, device name, IDs, etc.
  GPU_FREQUENCY,       // GPU/memory clocks
  GPU_THERMAL,         // Temperatures
  GPU_POWER,           // Power usage
  GPU_PCIE,            // PCIe metrics
  GPU_UTILIZATION,     // Usage metrics
  GPU_MEMORY,          // Memory usage/activity
  GPU_ECC,             // ECC error counters
  GPU_XGMI,            // XGMI link metrics
  GPU_PROFILER,        // ROC profiler counters
  CPU_BASIC,           // CPU identification
  CPU_FREQUENCY,       // CPU clocks
  CPU_ENERGY,          // CPU power/energy
  CPU_ADVANCED,        // Advanced CPU metrics
  HEALTH,              // Health monitoring
  EVENT,               // Event counters
  NOTIFICATION,        // Async notifications
  UNKNOWN              // Uncategorized
};

/**
 * @brief Field validation flags
 */
enum class ValidationFlags : uint32_t {
  NONE = 0,
  ALLOW_NOT_SUPPORTED = 1 << 0,  // Field may return RDC_ST_NOT_SUPPORTED
  ALLOW_ZERO = 1 << 1,           // Value can be zero
  ALLOW_NEGATIVE = 1 << 2,       // Value can be negative
  REQUIRES_GPU = 1 << 3,         // Requires GPU hardware
  REQUIRES_CPU = 1 << 4,         // Requires CPU hardware
  REQUIRES_EPYC = 1 << 5,        // Requires AMD EPYC CPU
  REQUIRES_MULTI_GPU = 1 << 6,   // Requires multiple GPUs
  STRING_FIELD = 1 << 7,         // Field value is string
  ACCUMULATOR = 1 << 8,          // Counter that accumulates (monotonic)
  PERCENTAGE = 1 << 9            // Value is a percentage (0-100)
};

inline ValidationFlags operator|(ValidationFlags a, ValidationFlags b) {
  return static_cast<ValidationFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool operator&(ValidationFlags a, ValidationFlags b) {
  return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * @brief Validator function type
 * Returns true if valid, false otherwise
 */
using FieldValidator = std::function<bool(const rdc_field_value&, std::string& error_msg)>;

/**
 * @brief Metadata for a single RDC field
 */
struct FieldMetadata {
  rdc_field_t field_id;
  std::string name;
  std::string description;
  FieldCategory category;
  rdc_field_type_t expected_type;
  ValidationFlags flags;
  FieldValidator custom_validator;  // Optional custom validation logic

  // Value range for INTEGER/DOUBLE types (optional)
  int64_t min_value_int = INT64_MIN;
  int64_t max_value_int = INT64_MAX;
  double min_value_double = -INFINITY;
  double max_value_double = INFINITY;

  FieldMetadata(rdc_field_t id, const std::string& field_name, const std::string& desc,
                FieldCategory cat, rdc_field_type_t type,
                ValidationFlags val_flags = ValidationFlags::NONE)
      : field_id(id),
        name(field_name),
        description(desc),
        category(cat),
        expected_type(type),
        flags(val_flags),
        custom_validator(nullptr) {}

  FieldMetadata& WithRange(int64_t min_val, int64_t max_val) {
    min_value_int = min_val;
    max_value_int = max_val;
    return *this;
  }

  FieldMetadata& WithRangeDouble(double min_val, double max_val) {
    min_value_double = min_val;
    max_value_double = max_val;
    return *this;
  }

  FieldMetadata& WithValidator(FieldValidator validator) {
    custom_validator = std::move(validator);
    return *this;
  }

  // Validate a field value
  bool Validate(const rdc_field_value& value, std::string& error_msg) const;
};

/**
 * @brief Central registry of all RDC fields
 */
class FieldRegistry {
 public:
  static FieldRegistry& Instance();

  // Register a field
  void Register(const FieldMetadata& metadata);

  // Get all fields
  const std::vector<FieldMetadata>& GetAllFields() const { return fields_; }

  // Get fields by category
  std::vector<FieldMetadata> GetFieldsByCategory(FieldCategory category) const;

  // Get fields by type
  std::vector<FieldMetadata> GetFieldsByType(rdc_field_type_t type) const;

  // Get field metadata by ID
  const FieldMetadata* GetField(rdc_field_t field_id) const;

  // Get displayable fields (for testing priority)
  std::vector<FieldMetadata> GetDisplayableFields() const;

 private:
  FieldRegistry() = default;
  std::vector<FieldMetadata> fields_;
  bool is_initialized_ = false;
};

/**
 * @brief Category name conversion
 */
const char* GetCategoryName(FieldCategory category);

/**
 * @brief Initialize the field registry with all RDC fields
 */
void InitializeFieldRegistry();

}  // namespace test
}  // namespace rdc

#endif  // NEW_TESTS_FIELD_METADATA_H_
