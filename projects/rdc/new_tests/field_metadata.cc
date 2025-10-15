/*
 * Field Metadata System Implementation
 */

#include "field_metadata.h"

#include <map>
#include <sstream>

namespace rdc {
namespace test {

bool FieldMetadata::Validate(const rdc_field_value& value, std::string& error_msg) const {
  // Check field ID matches
  if (value.field_id != field_id) {
    error_msg = "Field ID mismatch";
    return false;
  }

  // Check type matches
  if (value.type != expected_type) {
    std::ostringstream oss;
    oss << "Expected type " << expected_type << ", got " << value.type;
    error_msg = oss.str();
    return false;
  }

  // Type-specific validation
  if (expected_type == INTEGER) {
    int64_t val = value.value.l_int;

    // Check zero allowed
    if (!(flags & ValidationFlags::ALLOW_ZERO) && val == 0) {
      error_msg = "Zero value not allowed for this field";
      return false;
    }

    // Check negative allowed
    if (!(flags & ValidationFlags::ALLOW_NEGATIVE) && val < 0) {
      error_msg = "Negative value not allowed for this field";
      return false;
    }

    // Check range
    if (val < min_value_int || val > max_value_int) {
      std::ostringstream oss;
      oss << "Value " << val << " outside range [" << min_value_int << ", " << max_value_int << "]";
      error_msg = oss.str();
      return false;
    }

    // Percentage validation
    if (flags & ValidationFlags::PERCENTAGE) {
      if (val < 0 || val > 100) {
        std::ostringstream oss;
        oss << "Percentage value " << val << " outside [0, 100]";
        error_msg = oss.str();
        return false;
      }
    }
  } else if (expected_type == DOUBLE) {
    double val = value.value.dbl;

    // Check range
    if (val < min_value_double || val > max_value_double) {
      std::ostringstream oss;
      oss << "Value " << val << " outside range [" << min_value_double << ", " << max_value_double
          << "]";
      error_msg = oss.str();
      return false;
    }
  } else if (expected_type == STRING) {
    // String validation
    if (!(flags & ValidationFlags::STRING_FIELD)) {
      error_msg = "String field not properly flagged";
      return false;
    }
  }

  // Custom validator
  if (custom_validator) {
    return custom_validator(value, error_msg);
  }

  return true;
}

FieldRegistry& FieldRegistry::Instance() {
  static FieldRegistry instance;
  if (!instance.is_initialized_) {
    instance.is_initialized_ = true;  // Set BEFORE calling init to prevent recursion
    InitializeFieldRegistry();
  }
  return instance;
}

void FieldRegistry::Register(const FieldMetadata& metadata) { fields_.push_back(metadata); }

std::vector<FieldMetadata> FieldRegistry::GetFieldsByCategory(FieldCategory category) const {
  std::vector<FieldMetadata> result;
  for (const auto& field : fields_) {
    if (field.category == category) {
      result.push_back(field);
    }
  }
  return result;
}

std::vector<FieldMetadata> FieldRegistry::GetFieldsByType(rdc_field_type_t type) const {
  std::vector<FieldMetadata> result;
  for (const auto& field : fields_) {
    if (field.expected_type == type) {
      result.push_back(field);
    }
  }
  return result;
}

const FieldMetadata* FieldRegistry::GetField(rdc_field_t field_id) const {
  for (const auto& field : fields_) {
    if (field.field_id == field_id) {
      return &field;
    }
  }
  return nullptr;
}

std::vector<FieldMetadata> FieldRegistry::GetDisplayableFields() const {
  // Return first 50 fields for testing priority (most commonly used)
  size_t count = std::min(fields_.size(), size_t(50));
  return std::vector<FieldMetadata>(fields_.begin(), fields_.begin() + count);
}

const char* GetCategoryName(FieldCategory category) {
  static const std::map<FieldCategory, const char*> names = {
      {FieldCategory::GPU_IDENTIFICATION, "GPU_IDENTIFICATION"},
      {FieldCategory::GPU_FREQUENCY, "GPU_FREQUENCY"},
      {FieldCategory::GPU_THERMAL, "GPU_THERMAL"},
      {FieldCategory::GPU_POWER, "GPU_POWER"},
      {FieldCategory::GPU_PCIE, "GPU_PCIE"},
      {FieldCategory::GPU_UTILIZATION, "GPU_UTILIZATION"},
      {FieldCategory::GPU_MEMORY, "GPU_MEMORY"},
      {FieldCategory::GPU_ECC, "GPU_ECC"},
      {FieldCategory::GPU_XGMI, "GPU_XGMI"},
      {FieldCategory::GPU_PROFILER, "GPU_PROFILER"},
      {FieldCategory::CPU_BASIC, "CPU_BASIC"},
      {FieldCategory::CPU_FREQUENCY, "CPU_FREQUENCY"},
      {FieldCategory::CPU_ENERGY, "CPU_ENERGY"},
      {FieldCategory::CPU_ADVANCED, "CPU_ADVANCED"},
      {FieldCategory::HEALTH, "HEALTH"},
      {FieldCategory::EVENT, "EVENT"},
      {FieldCategory::NOTIFICATION, "NOTIFICATION"},
      {FieldCategory::UNKNOWN, "UNKNOWN"}};

  auto it = names.find(category);
  return (it != names.end()) ? it->second : "UNKNOWN";
}

// Initialize the registry with all important RDC fields
void InitializeFieldRegistry() {
  auto& registry = FieldRegistry::Instance();

  // GPU Identification Fields
  registry.Register(FieldMetadata(RDC_FI_GPU_COUNT, "GPU_COUNT", "GPU count in the system",
                                  FieldCategory::GPU_IDENTIFICATION, INTEGER)
                        .WithRange(0LL, static_cast<int64_t>(RDC_MAX_NUM_DEVICES)));

  registry.Register(FieldMetadata(RDC_FI_DEV_NAME, "DEV_NAME", "Name of the device",
                                  FieldCategory::GPU_IDENTIFICATION, STRING,
                                  ValidationFlags::STRING_FIELD));

  registry.Register(
      FieldMetadata(RDC_FI_OAM_ID, "OAM_ID", "OAM ID of the device",
                    FieldCategory::GPU_IDENTIFICATION, INTEGER,
                    ValidationFlags::ALLOW_NOT_SUPPORTED | ValidationFlags::REQUIRES_GPU)
          .WithRange(0LL, 0xFFFFFFFFLL));  // 0xFFFFFFFF means not supported

  registry.Register(FieldMetadata(RDC_FI_DEV_ID, "DEV_ID", "ID of the device",
                                  FieldCategory::GPU_IDENTIFICATION, INTEGER,
                                  ValidationFlags::REQUIRES_GPU));

  registry.Register(FieldMetadata(RDC_FI_UUID, "UUID", "Unique ID of the device",
                                  FieldCategory::GPU_IDENTIFICATION, STRING,
                                  ValidationFlags::STRING_FIELD | ValidationFlags::REQUIRES_GPU));

  // GPU Frequency Fields
  registry.Register(FieldMetadata(RDC_FI_GPU_CLOCK, "GPU_CLOCK", "Current GPU clock",
                                  FieldCategory::GPU_FREQUENCY, INTEGER,
                                  ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO)
                        .WithRange(0LL, 5000000000LL));  // 0 Hz (idle) to 5 GHz

  registry.Register(FieldMetadata(RDC_FI_MEM_CLOCK, "MEM_CLOCK", "Current Memory clock",
                                  FieldCategory::GPU_FREQUENCY, INTEGER,
                                  ValidationFlags::REQUIRES_GPU)
                        .WithRange(100000LL, 3000000000LL));  // 100 kHz to 3 GHz

  // GPU Thermal Fields
  registry.Register(FieldMetadata(RDC_FI_GPU_TEMP, "GPU_TEMP", "GPU temperature in millidegrees",
                                  FieldCategory::GPU_THERMAL, INTEGER,
                                  ValidationFlags::REQUIRES_GPU)
                        .WithRange(-40000LL, 150000LL));  // -40C to 150C

  registry.Register(
      FieldMetadata(RDC_FI_MEMORY_TEMP, "MEMORY_TEMP", "Memory temperature in millidegrees",
                    FieldCategory::GPU_THERMAL, INTEGER,
                    ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_NOT_SUPPORTED)
          .WithRange(-40000LL, 150000LL));

  // GPU Power Fields
  registry.Register(FieldMetadata(RDC_FI_POWER_USAGE, "POWER_USAGE", "Power usage in microwatts",
                                  FieldCategory::GPU_POWER, INTEGER, ValidationFlags::REQUIRES_GPU)
                        .WithRange(0LL, 1000000000000LL));  // 0 to 1000W

  // GPU PCIe Fields
  registry.Register(FieldMetadata(
      RDC_FI_PCIE_TX, "PCIE_TX", "PCIe Tx utilization", FieldCategory::GPU_PCIE, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_NOT_SUPPORTED |
          ValidationFlags::ALLOW_ZERO | ValidationFlags::ACCUMULATOR));

  registry.Register(FieldMetadata(
      RDC_FI_PCIE_RX, "PCIE_RX", "PCIe Rx utilization", FieldCategory::GPU_PCIE, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_NOT_SUPPORTED |
          ValidationFlags::ALLOW_ZERO | ValidationFlags::ACCUMULATOR));

  // GPU Utilization Fields
  registry.Register(FieldMetadata(
      RDC_FI_GPU_UTIL, "GPU_UTIL", "GPU busy percentage", FieldCategory::GPU_UTILIZATION, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO | ValidationFlags::PERCENTAGE));

  registry.Register(FieldMetadata(
      RDC_FI_GPU_BUSY_PERCENT, "GPU_BUSY_PERCENT", "GPU busy percentage",
      FieldCategory::GPU_UTILIZATION, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO | ValidationFlags::PERCENTAGE));

  // GPU Memory Fields
  registry.Register(FieldMetadata(RDC_FI_GPU_MEMORY_USAGE, "GPU_MEMORY_USAGE",
                                  "Memory usage in bytes", FieldCategory::GPU_MEMORY, INTEGER,
                                  ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO));

  registry.Register(FieldMetadata(RDC_FI_GPU_MEMORY_TOTAL, "GPU_MEMORY_TOTAL", "Total memory",
                                  FieldCategory::GPU_MEMORY, INTEGER, ValidationFlags::REQUIRES_GPU)
                        .WithRange(1LL, 1024LL * 1024 * 1024 * 1024));  // Up to 1TB

  registry.Register(FieldMetadata(
      RDC_FI_GPU_MEMORY_ACTIVITY, "GPU_MEMORY_ACTIVITY", "Memory busy percentage",
      FieldCategory::GPU_MEMORY, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO | ValidationFlags::PERCENTAGE));

  // GPU ECC Fields (sample - there are 60+ ECC fields)
  registry.Register(FieldMetadata(
      RDC_FI_ECC_CORRECT_TOTAL, "ECC_CORRECT_TOTAL", "Total correctable ECC errors",
      FieldCategory::GPU_ECC, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO | ValidationFlags::ACCUMULATOR));

  registry.Register(FieldMetadata(
      RDC_FI_ECC_UNCORRECT_TOTAL, "ECC_UNCORRECT_TOTAL", "Total uncorrectable ECC errors",
      FieldCategory::GPU_ECC, INTEGER,
      ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO | ValidationFlags::ACCUMULATOR));

  // XGMI Fields (sample)
  registry.Register(FieldMetadata(RDC_FI_XGMI_0_READ_KB, "XGMI_0_READ_KB", "XGMI0 read size (KB)",
                                  FieldCategory::GPU_XGMI, INTEGER,
                                  ValidationFlags::REQUIRES_GPU | ValidationFlags::ALLOW_ZERO |
                                      ValidationFlags::ALLOW_NOT_SUPPORTED |
                                      ValidationFlags::ACCUMULATOR));

  // CPU Basic Fields
  registry.Register(
      FieldMetadata(RDC_FI_CPU_SKT_COUNT, "CPU_SKT_COUNT", "CPU socket count",
                    FieldCategory::CPU_BASIC, INTEGER,
                    ValidationFlags::REQUIRES_CPU | ValidationFlags::ALLOW_NOT_SUPPORTED)
          .WithRange(0LL, 16LL));

  registry.Register(FieldMetadata(RDC_FI_CPU_MODEL, "CPU_MODEL", "CPU model name",
                                  FieldCategory::CPU_BASIC, STRING,
                                  ValidationFlags::STRING_FIELD | ValidationFlags::REQUIRES_CPU |
                                      ValidationFlags::ALLOW_NOT_SUPPORTED));

  registry.Register(
      FieldMetadata(RDC_FI_CPU_FAMILY, "CPU_FAMILY", "CPU family ID", FieldCategory::CPU_BASIC,
                    INTEGER, ValidationFlags::REQUIRES_CPU | ValidationFlags::ALLOW_NOT_SUPPORTED));

  // CPU Energy Fields
  registry.Register(
      FieldMetadata(RDC_FI_CPU_SKT_ENERGY, "CPU_SKT_ENERGY", "CPU socket energy (microjoules)",
                    FieldCategory::CPU_ENERGY, INTEGER,
                    ValidationFlags::REQUIRES_EPYC | ValidationFlags::ALLOW_ZERO |
                        ValidationFlags::ALLOW_NOT_SUPPORTED | ValidationFlags::ACCUMULATOR));

  // CPU Frequency Fields
  registry.Register(
      FieldMetadata(RDC_FI_CPU_FCLK_FREQUENCY, "CPU_FCLK_FREQUENCY", "CPU fabric clock (MHz)",
                    FieldCategory::CPU_FREQUENCY, INTEGER,
                    ValidationFlags::REQUIRES_EPYC | ValidationFlags::ALLOW_NOT_SUPPORTED)
          .WithRange(0LL, 10000LL));

  registry.Register(
      FieldMetadata(RDC_FI_CPU_CCLK_LIMIT, "CPU_CCLK_LIMIT", "CPU core clock limit (MHz)",
                    FieldCategory::CPU_FREQUENCY, INTEGER,
                    ValidationFlags::REQUIRES_EPYC | ValidationFlags::ALLOW_NOT_SUPPORTED)
          .WithRange(0LL, 10000LL));
}

}  // namespace test
}  // namespace rdc
