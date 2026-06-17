/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */

/// \file
/// Implementation of centralized environment variable management

#include "common/env_config.h"
#include <cstdlib>
#include <iostream>

namespace rocrtst {

EnvironmentConfig::EnvironmentConfig()
    : initialized_(false),
      pool_size_limit_(0) {
}

EnvironmentConfig& EnvironmentConfig::getInstance() {
  static EnvironmentConfig instance;
  return instance;
}

void EnvironmentConfig::initialize() {
  if (initialized_) {
    return;  // Already initialized
  }

  // Read all environment variables
  hsa_model_topology_ = readEnvString("HSA_MODEL_TOPOLOGY");
  platform_override_ = readEnvString("ROCRTST_PLATFORM_OVERRIDE");
  platform_config_path_ = readEnvString("ROCRTST_PLATFORM_CONFIG");
  test_groups_ = readEnvString("ROCRTST_TEST_GROUPS");
  pool_size_limit_ = readEnvSize("ROCRTST_LIMIT_POOL_SIZE");

  initialized_ = true;

  // Optional: Print configuration for debugging
  #ifdef ROCRTST_DEBUG_ENV
  std::cout << "=== Environment Configuration ===" << std::endl;
  if (!hsa_model_topology_.empty()) {
    std::cout << "HSA_MODEL_TOPOLOGY: " << hsa_model_topology_ << std::endl;
  }
  if (!platform_override_.empty()) {
    std::cout << "ROCRTST_PLATFORM_OVERRIDE: " << platform_override_ << std::endl;
  }
  if (!platform_config_path_.empty()) {
    std::cout << "ROCRTST_PLATFORM_CONFIG: " << platform_config_path_ << std::endl;
  }
  if (!test_groups_.empty()) {
    std::cout << "ROCRTST_TEST_GROUPS: " << test_groups_ << std::endl;
  }
  if (pool_size_limit_ > 0) {
    std::cout << "ROCRTST_LIMIT_POOL_SIZE: " << pool_size_limit_ << std::endl;
  }
  std::cout << "=================================" << std::endl;
  #endif
}

std::string EnvironmentConfig::readEnvString(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

size_t EnvironmentConfig::readEnvSize(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return 0;
  }

  char* end = nullptr;
  unsigned long result = std::strtoul(value, &end, 10);

  // Check for conversion errors
  if (end == value || *end != '\0') {
    std::cerr << "Warning: Invalid value for " << name << ": " << value << std::endl;
    return 0;
  }

  return static_cast<size_t>(result);
}

}  // namespace rocrtst
