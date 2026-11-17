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
// Only devices that the tests are built for are considered: those that match
// one of the target IDs passed with --targets (e.g. "gfx1201" or
// "gfx942:xnack+"). Other visible devices are ignored. A feature is reported as
// available only when every considered device supports it, so a test that
// requires it can run on whichever of those devices it picks.

#include <hip/hip_runtime.h>

#include <algorithm>
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

// Splits s at every occurrence of delimiter, dropping empty pieces.
std::vector<std::string> Split(const std::string& s, char delimiter) {
  std::vector<std::string> pieces;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(delimiter, start);
    if (end == std::string::npos) {
      end = s.size();
    }
    if (end > start) {
      pieces.push_back(s.substr(start, end - start));
    }
    start = end + 1;
  }
  return pieces;
}

// Returns true if the ':'-separated target feature list of a target ID
// (e.g. "gfx942:sramecc+:xnack-") contains the given entry (e.g. "xnack+").
bool HasTargetFeature(const std::string& target_id, const std::string& target_feature) {
  std::vector<std::string> pieces = Split(target_id, ':');
  return std::find(pieces.begin() + (pieces.empty() ? 0 : 1), pieces.end(), target_feature) !=
         pieces.end();
}

// Returns true if code built for the target ID target (e.g. "gfx942" or
// "gfx942:xnack+") runs on a device whose gcnArchName is arch_name (e.g.
// "gfx942:sramecc+:xnack-"): the processors must be equal, and every target
// feature set in target must be set the same way on the device. SPIR-V code
// runs on every device.
bool TargetMatchesDevice(const std::string& target, const std::string& arch_name) {
  if (target == "amdgcnspirv") {
    return true;
  }
  std::vector<std::string> target_pieces = Split(target, ':');
  std::vector<std::string> device_pieces = Split(arch_name, ':');
  if (target_pieces.empty() || device_pieces.empty() || target_pieces[0] != device_pieces[0]) {
    return false;
  }
  for (size_t i = 1; i < target_pieces.size(); ++i) {
    if (!HasTargetFeature(arch_name, target_pieces[i])) {
      return false;
    }
  }
  return true;
}

struct DeviceInfo {
  int id;
  std::string name;
  std::string arch_name;
  // Whether the tests are built for this device.
  bool selected;
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
bool AllDevicesArePeers(const std::vector<int>& devices) {
  if (devices.size() < 2) {
    return false;
  }
  for (int device : devices) {
    for (int peer : devices) {
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
  std::cerr << "Usage: " << program << " [--targets <id>[,<id>...]] [--output <path>]\n"
            << "Writes the HIP device features of this machine as YAML to <path>, or to\n"
            << "stdout when --output is not given. Only devices matching one of the\n"
            << "target IDs given with --targets are considered; all devices are\n"
            << "considered when --targets is not given or empty.\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_path;
  std::vector<std::string> targets;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output_path = argv[++i];
    } else if (std::strcmp(argv[i], "--targets") == 0 && i + 1 < argc) {
      targets = Split(argv[++i], ',');
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
  std::vector<int> selected_devices;
  for (int device = 0; device < device_count; ++device) {
    hipDeviceProp_t props;
    Check(hipGetDeviceProperties(&props, device),
          "hipGetDeviceProperties(" + std::to_string(device) + ")");
    bool selected = targets.empty() ||
                    std::any_of(targets.begin(), targets.end(), [&](const std::string& target) {
                      return TargetMatchesDevice(target, props.gcnArchName);
                    });
    devices.push_back({device, props.name, props.gcnArchName, selected});
    if (selected) {
      selected_devices.push_back(device);
    }
  }

  if (selected_devices.empty()) {
    std::string message =
        "none of the visible HIP devices matches the targets the tests are "
        "built for. Targets:";
    for (const auto& target : targets) {
      message += " " + target;
    }
    message += ". Devices:";
    for (const auto& device : devices) {
      message += " " + device.arch_name;
    }
    Fail(message);
  }

  std::vector<std::pair<std::string, bool>> features;
  for (const auto& [feature, check] : kDeviceFeatures) {
    bool available = true;
    for (int device : selected_devices) {
      available = available && check(device, devices[device]);
    }
    features.emplace_back(feature, available);
  }
  features.emplace_back("multi_gpu", selected_devices.size() >= 2);
  features.emplace_back("p2p", AllDevicesArePeers(selected_devices));

  std::ofstream output_file;
  if (!output_path.empty()) {
    output_file.open(output_path);
    if (!output_file) {
      Fail("cannot open '" + output_path + "' for writing");
    }
  }
  std::ostream& out = output_path.empty() ? std::cout : output_file;

  out << "# Generated by hip_tests_sysinfo - DO NOT EDIT\n";
  out << "targets: [";
  for (size_t i = 0; i < targets.size(); ++i) {
    out << (i ? ", " : "") << "\"" << targets[i] << "\"";
  }
  out << "]\n";
  out << "# Devices that the tests are not built for are listed with\n";
  out << "# 'selected: false' and do not contribute to the features below.\n";
  out << "devices:\n";
  for (const auto& device : devices) {
    out << "  - id: " << device.id << "\n";
    out << "    name: \"" << device.name << "\"\n";
    out << "    arch: \"" << device.arch_name << "\"\n";
    out << "    selected: " << (device.selected ? "true" : "false") << "\n";
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
