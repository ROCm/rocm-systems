/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <errno.h>
#endif

// Generic helper executable to test environment variables with HIP runtime
// AMD-only: Tests GPU_ENABLE_PAL (ROCr vs PAL backend selection)
//
#ifndef __HIP_PLATFORM_AMD__
#error "This test is AMD-only (GPU_ENABLE_PAL controls ROCr vs PAL backend)"
#endif

static int setEnvironmentVariable(const char* name, const char* value) {
#ifdef _WIN32
  return _putenv_s(name, value);
#else
  return setenv(name, value, 1);
#endif
}

static int unsetEnvironmentVariable(const char* name) {
#ifdef _WIN32
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

// Initialize HIP with AMD_LOG_LEVEL=7, then restore original log level
static int initializeHIP() {
  const char* originalLogLevel = getenv("AMD_LOG_LEVEL");
  std::string savedLogLevel;
  if (originalLogLevel) {
    savedLogLevel = originalLogLevel;
  }

  setEnvironmentVariable("AMD_LOG_LEVEL", "7");

  int deviceCount = 0;
  hipError_t err = hipGetDeviceCount(&deviceCount);

  if (!savedLogLevel.empty()) {
    setEnvironmentVariable("AMD_LOG_LEVEL", savedLogLevel.c_str());
  } else {
    unsetEnvironmentVariable("AMD_LOG_LEVEL");
  }

  if (err != hipSuccess) {
    std::cerr << "ERROR: hipGetDeviceCount failed: " << hipGetErrorString(err) << std::endl;
    return 255;
  }
  return 0;
}

// Test GPU_ENABLE_PAL - logs which backend initialized
static int checkGpuEnablePal(const char* value) {
  return initializeHIP();
}

// Test generic environment variable
static int testEnvironmentVariable(const std::string& envName, const char* value) {
  return initializeHIP();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <env_var_name> <env_var_value>" << std::endl;
    return 255;
  }

  const std::string envName = argv[1];
  const char* envValue = argv[2];

  // Handle special "UNSET" marker to test default behavior
  if (strcmp(envValue, "UNSET") == 0) {
    unsetEnvironmentVariable(envName.c_str());
  } else {
    // Set the environment variable
    if (setEnvironmentVariable(envName.c_str(), envValue) != 0) {
      std::cerr << "ERROR: Failed to set " << envName << std::endl;
      return 255;
    }
  }

  // Run the test
  int result;
  if (envName == "GPU_ENABLE_PAL") {
    result = checkGpuEnablePal(envValue);
  } else {
    result = testEnvironmentVariable(envName, envValue);
  }

  return result;
}
