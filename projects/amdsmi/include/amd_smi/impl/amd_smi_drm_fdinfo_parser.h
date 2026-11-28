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

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <sstream>
#include <string>

namespace amd::smi {

// Usage that one DRM client reports in one fdinfo file.
struct DrmClientUsage {
  std::string driver;
  // drm-client-id, or amdgpu's pasid on kernels without it. Duplicated fds
  // share one client and so report the same key.
  std::string client;
  // Inode number of the open file; empty on kernels before 5.14, which do not
  // print fdinfo "ino".
  std::string ino;
  uint64_t vram = 0;  // bytes
  uint64_t gtt = 0;   // bytes
  uint64_t cpu = 0;   // bytes
  uint64_t gfx_ns = 0;
  uint64_t enc_ns = 0;
  // False if a counter was present but not a number the kernel would print.
  bool valid = true;
};

// Parse "<number>[ <unit>]", requiring the whole string to match. Rejects
// signs, overflow and trailing text.
inline bool ParseDrmNumberWithUnit(const std::string& value, uint64_t* number, std::string* unit) {
  if (value.empty() || value[0] < '0' || value[0] > '9') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (errno == ERANGE) return false;
  const std::string rest(end);
  if (rest.empty()) {
    unit->clear();
  } else if (rest[0] == ' ' && rest.find(' ', 1) == std::string::npos) {
    *unit = rest.substr(1);
  } else {
    return false;
  }
  *number = parsed;
  return true;
}

// Parse a size as the kernel prints it: no unit, KiB, or MiB.
inline bool ParseDrmSize(const std::string& value, uint64_t* bytes) {
  uint64_t number = 0;
  std::string unit;
  if (!ParseDrmNumberWithUnit(value, &number, &unit)) return false;
  unsigned shift = 0;
  if (unit == "KiB") {
    shift = 10;
  } else if (unit == "MiB") {
    shift = 20;
  } else if (!unit.empty()) {
    return false;
  }
  if (number > (std::numeric_limits<uint64_t>::max() >> shift)) return false;
  *bytes = number << shift;
  return true;
}

// Parse an engine time, which the kernel prints as "<time> ns".
inline bool ParseDrmEngineTime(const std::string& value, uint64_t* ns) {
  std::string unit;
  return ParseDrmNumberWithUnit(value, ns, &unit) && unit == "ns";
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
  DrmClientUsage usage;
  // An absent counter reads as 0; a present one must parse.
  auto counter = [&](const std::string& key, uint64_t* value,
                     bool (*parse)(const std::string&, uint64_t*)) {
    *value = 0;
    auto it = fields.find(key);
    if (it == fields.end()) return false;
    if (!parse(it->second, value)) {
      *value = 0;
      usage.valid = false;
    }
    return true;
  };
  auto memory = [&](const std::string& region) {
    uint64_t bytes = 0;
    if (!counter("drm-total-" + region, &bytes, ParseDrmSize)) {
      counter("drm-memory-" + region, &bytes, ParseDrmSize);
    }
    return bytes;
  };
  auto engine_ns = [&](const std::string& engine) {
    uint64_t ns = 0;
    counter("drm-engine-" + engine, &ns, ParseDrmEngineTime);
    return ns;
  };

  usage.driver = get("drm-driver");
  usage.client = get("drm-client-id");
  if (usage.client.empty()) usage.client = get("pasid");
  usage.ino = get("ino");
  usage.vram = memory("vram");
  usage.gtt = memory("gtt");
  usage.cpu = memory("cpu");
  usage.gfx_ns = engine_ns("gfx");
  usage.enc_ns = engine_ns("enc");
  return usage;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_DRM_FDINFO_PARSER_H_
