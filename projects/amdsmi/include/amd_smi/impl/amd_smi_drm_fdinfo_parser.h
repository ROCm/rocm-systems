// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Parser for the DRM client usage stats a process reads from its own
// /proc/self/fdinfo/<fd>; see https://docs.kernel.org/gpu/drm-usage-stats.html.
//
// Memory prefers drm-total-<region>, the size of the client's buffers in
// their preferred placement, over the older drm-memory-<region>, which amdgpu
// reports as resident size and so drops when buffers are evicted.
//
// Header-only so fdinfo.cc and the unit tests share one implementation.

#ifndef AMD_SMI_INCLUDE_AMD_SMI_DRM_FDINFO_PARSER_H_
#define AMD_SMI_INCLUDE_AMD_SMI_DRM_FDINFO_PARSER_H_

#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>

namespace amd::smi {

// Usage that one DRM client reports in one fdinfo file.
struct DrmClientUsage {
  std::string driver;
  std::string pdev;
  // drm-client-id, or amdgpu's pasid on kernels without it. Duplicated fds
  // share one client and so report the same key.
  std::string client;
  uint64_t vram = 0;  // bytes
  uint64_t gtt = 0;   // bytes
  uint64_t cpu = 0;   // bytes
  uint64_t gfx_ns = 0;
  uint64_t enc_ns = 0;
};

// Parse "<number>[ <unit>]" as the kernel prints sizes: no unit, KiB, or MiB.
inline bool ParseDrmSize(const std::string& value, uint64_t* bytes) {
  std::istringstream in(value);
  uint64_t number = 0;
  std::string unit;
  if (!(in >> number)) return false;
  in >> unit;
  if (unit.empty()) {
    *bytes = number;
  } else if (unit == "KiB") {
    *bytes = number << 10;
  } else if (unit == "MiB") {
    *bytes = number << 20;
  } else {
    return false;
  }
  return true;
}

inline DrmClientUsage ParseDrmFdinfo(const std::string& text) {
  std::map<std::string, std::string> fields;
  std::istringstream lines(text);
  for (std::string line; std::getline(lines, line);) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const auto start = line.find_first_not_of(" \t", colon + 1);
    fields[line.substr(0, colon)] = start == std::string::npos ? "" : line.substr(start);
  }
  auto get = [&fields](const std::string& key) {
    auto it = fields.find(key);
    return it == fields.end() ? std::string() : it->second;
  };
  auto memory = [&](const std::string& region) {
    uint64_t bytes = 0;
    if (!ParseDrmSize(get("drm-total-" + region), &bytes)) {
      ParseDrmSize(get("drm-memory-" + region), &bytes);
    }
    return bytes;
  };
  auto engine_ns = [&](const std::string& engine) {
    // "<time> ns"
    return static_cast<uint64_t>(std::strtoull(get("drm-engine-" + engine).c_str(), nullptr, 10));
  };

  DrmClientUsage usage;
  usage.driver = get("drm-driver");
  usage.pdev = get("drm-pdev");
  usage.client = get("drm-client-id");
  if (usage.client.empty()) usage.client = get("pasid");
  usage.vram = memory("vram");
  usage.gtt = memory("gtt");
  usage.cpu = memory("cpu");
  usage.gfx_ns = engine_ns("gfx");
  usage.enc_ns = engine_ns("enc");
  return usage;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_DRM_FDINFO_PARSER_H_
