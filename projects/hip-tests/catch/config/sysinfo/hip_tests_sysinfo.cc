/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Probes the HIP devices visible on the build machine and writes the set of
// device features they support as YAML. parse_config.py consumes the output
// when HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE is ON and disables test cases
// whose 'requires' list names a feature that is not available.
//
// Every feature listed in configs/definitions.yaml under 'features' must be
// reported here (as true or false), and nothing else may be reported;
// parse_config.py fails the build otherwise.
//
// A feature is reported as available only when every visible device supports
// it, so a test that requires it can run on whichever device it picks.

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "[hip_tests_sysinfo] ERROR: " << message << std::endl;
  std::exit(1);
}

void Check(hipError_t status, const std::string& what) {
  if (status != hipSuccess) {
    Fail(what + " failed: " + hipGetErrorString(status));
  }
}

bool GetBoolAttribute(hipDeviceAttribute_t attribute, const char* attribute_name, int device) {
  int value = 0;
  Check(hipDeviceGetAttribute(&value, attribute, device), std::string("hipDeviceGetAttribute(") +
                                                              attribute_name + ", device " +
                                                              std::to_string(device) + ")");
  return value != 0;
}

// Returns true if the ':'-separated target feature list of gcnArchName
// (e.g. "gfx942:sramecc+:xnack-") contains the given entry (e.g. "xnack+").
bool HasTargetFeature(const std::string& arch_name, const std::string& target_feature) {
  size_t start = arch_name.find(':');
  while (start != std::string::npos) {
    size_t end = arch_name.find(':', start + 1);
    if (arch_name.compare(start + 1, end - start - 1, target_feature) == 0) {
      return true;
    }
    start = end;
  }
  return false;
}

struct DeviceInfo {
  std::string name;
  std::string arch_name;
};

using DeviceFeatureCheck = std::function<bool(int device, const DeviceInfo& info)>;

#define DEVICE_ATTRIBUTE(attribute)                                                                \
  [](int device, const DeviceInfo&) { return GetBoolAttribute(attribute, #attribute, device); }

// Features that are evaluated per device and must hold for every device.
// Keep in sync with 'features' in configs/definitions.yaml.
const std::vector<std::pair<std::string, DeviceFeatureCheck>> kDeviceFeatures = {
    {"images", DEVICE_ATTRIBUTE(hipDeviceAttributeImageSupport)},
    {"managed_memory", DEVICE_ATTRIBUTE(hipDeviceAttributeManagedMemory)},
    {"concurrent_managed_access", DEVICE_ATTRIBUTE(hipDeviceAttributeConcurrentManagedAccess)},
    {"pageable_memory_access", DEVICE_ATTRIBUTE(hipDeviceAttributePageableMemoryAccess)},
    {"host_native_atomics", DEVICE_ATTRIBUTE(hipDeviceAttributeHostNativeAtomicSupported)},
    {"host_register", DEVICE_ATTRIBUTE(hipDeviceAttributeHostRegisterSupported)},
    {"cooperative_launch", DEVICE_ATTRIBUTE(hipDeviceAttributeCooperativeLaunch)},
    {"cooperative_multi_device_launch",
     DEVICE_ATTRIBUTE(hipDeviceAttributeCooperativeMultiDeviceLaunch)},
    {"virtual_memory", DEVICE_ATTRIBUTE(hipDeviceAttributeVirtualMemoryManagementSupported)},
    {"memory_pools", DEVICE_ATTRIBUTE(hipDeviceAttributeMemoryPoolsSupported)},
    {"large_bar", DEVICE_ATTRIBUTE(hipDeviceAttributeIsLargeBar)},
    {"stream_wait_value", DEVICE_ATTRIBUTE(hipDeviceAttributeCanUseStreamWaitValue)},
    {"xnack",
     [](int, const DeviceInfo& info) { return HasTargetFeature(info.arch_name, "xnack+"); }},
    {"sramecc",
     [](int, const DeviceInfo& info) { return HasTargetFeature(info.arch_name, "sramecc+"); }},
};

#undef DEVICE_ATTRIBUTE

// Returns true if every device can access the memory of every other device.
// Requires at least two devices.
bool AllDevicesArePeers(int device_count) {
  if (device_count < 2) {
    return false;
  }
  for (int device = 0; device < device_count; ++device) {
    for (int peer = 0; peer < device_count; ++peer) {
      if (device == peer) {
        continue;
      }
      int can_access = 0;
      Check(hipDeviceCanAccessPeer(&can_access, device, peer),
            "hipDeviceCanAccessPeer(" + std::to_string(device) + ", " + std::to_string(peer) + ")");
      if (!can_access) {
        return false;
      }
    }
  }
  return true;
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program << " [--output <path>]\n"
            << "Writes the HIP device features of this machine as YAML to <path>, or to\n"
            << "stdout when --output is not given.\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output_path = argv[++i];
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }

  int device_count = 0;
  Check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
  if (device_count == 0) {
    Fail("no HIP devices found; HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE requires a GPU");
  }

  std::vector<DeviceInfo> devices;
  for (int device = 0; device < device_count; ++device) {
    hipDeviceProp_t props;
    Check(hipGetDeviceProperties(&props, device),
          "hipGetDeviceProperties(" + std::to_string(device) + ")");
    devices.push_back({props.name, props.gcnArchName});
  }

  std::vector<std::pair<std::string, bool>> features;
  for (const auto& [feature, check] : kDeviceFeatures) {
    bool available = true;
    for (int device = 0; device < device_count; ++device) {
      available = available && check(device, devices[device]);
    }
    features.emplace_back(feature, available);
  }
  features.emplace_back("multi_gpu", device_count >= 2);
  features.emplace_back("p2p", AllDevicesArePeers(device_count));

  std::ofstream output_file;
  if (!output_path.empty()) {
    output_file.open(output_path);
    if (!output_file) {
      Fail("cannot open '" + output_path + "' for writing");
    }
  }
  std::ostream& out = output_path.empty() ? std::cout : output_file;

  out << "# Generated by hip_tests_sysinfo - DO NOT EDIT\n";
  out << "devices:\n";
  for (const auto& device : devices) {
    out << "  - name: \"" << device.name << "\"\n";
    out << "    arch: \"" << device.arch_name << "\"\n";
  }
  out << "features:\n";
  for (const auto& [feature, available] : features) {
    out << "  " << feature << ": " << (available ? "true" : "false") << "\n";
  }

  out.flush();
  if (!out) {
    Fail("failed to write the device features");
  }
  return 0;
}
