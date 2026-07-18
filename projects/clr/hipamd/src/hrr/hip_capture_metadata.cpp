/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_capture_metadata.h"

#include "amd_comgr/amd_comgr.h"
#include "hip/hip_runtime_api.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace hip {
class Device;
extern std::vector<hip::Device*> g_devices;
extern hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device);
}  // namespace hip

namespace hrr_cap {
namespace metadata {
namespace {

std::string json_escape(const char* s) {
  std::string out;
  if (!s) return out;
  out.reserve(strlen(s));
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p != '\0'; ++p) {
    const unsigned char c = *p;
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"':  out += "\\\""; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
        break;
    }
  }
  return out;
}

std::string json_escape(const std::string& s) { return json_escape(s.c_str()); }

std::string quote(const std::string& s) { return "\"" + json_escape(s) + "\""; }

size_t bounded_strlen(const char* s, size_t max_len) {
  size_t n = 0;
  while (n < max_len && s[n] != '\0') ++n;
  return n;
}

std::string bounded_string(const char* s, size_t max_len) {
  return std::string(s, bounded_strlen(s, max_len));
}

std::string version_string_from_int(int version) {
  if (version <= 0) return "";
  const int major = version / 10000000;
  const int minor = (version / 100000) % 100;
  const int patch = version % 100000;
  if (major <= 0) return std::to_string(version);
  return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

std::string bytes_to_hex(const char* bytes, size_t len) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.resize(len * 2);
  for (size_t i = 0; i < len; ++i) {
    const auto v = static_cast<unsigned char>(bytes[i]);
    result[2 * i] = kHex[v >> 4];
    result[2 * i + 1] = kHex[v & 0xf];
  }
  return result;
}

std::string uuid_to_hex(const hipUUID& uuid) {
  return bytes_to_hex(uuid.bytes, sizeof(uuid.bytes));
}

void append_prop_fields(std::ostringstream& os, const hipDeviceProp_t& prop) {
  os << "      \"properties\": {\n"
     << "        \"name\": " << quote(bounded_string(prop.name, sizeof(prop.name))) << ",\n"
     << "        \"gcn_arch_name\": "
     << quote(bounded_string(prop.gcnArchName, sizeof(prop.gcnArchName))) << ",\n"
     << "        \"total_global_mem\": " << static_cast<unsigned long long>(prop.totalGlobalMem) << ",\n"
     << "        \"multi_processor_count\": " << prop.multiProcessorCount << ",\n"
     << "        \"warp_size\": " << prop.warpSize << ",\n"
     << "        \"max_threads_per_block\": " << prop.maxThreadsPerBlock << ",\n"
     << "        \"clock_rate_khz\": " << prop.clockRate << ",\n"
     << "        \"memory_clock_rate_khz\": " << prop.memoryClockRate << ",\n"
     << "        \"memory_bus_width\": " << prop.memoryBusWidth << ",\n"
     << "        \"l2_cache_size\": " << prop.l2CacheSize << ",\n"
     << "        \"integrated\": " << prop.integrated << ",\n"
     << "        \"managed_memory\": " << prop.managedMemory << ",\n"
     << "        \"memory_pools_supported\": " << prop.memoryPoolsSupported << ",\n"
     << "        \"compute_mode\": " << prop.computeMode << ",\n"
     << "        \"compute_capability\": " << quote(std::to_string(prop.major) + "." +
                                                  std::to_string(prop.minor)) << ",\n"
     << "        \"pci\": " << quote(std::to_string(prop.pciDomainID) + ":" +
                                    std::to_string(prop.pciBusID) + ":" +
                                    std::to_string(prop.pciDeviceID)) << ",\n"
     << "        \"uuid\": " << quote(uuid_to_hex(prop.uuid)) << ",\n"
     << "        \"luid\": " << quote(bytes_to_hex(prop.luid, sizeof(prop.luid))) << ",\n"
     << "        \"luid_device_node_mask\": " << prop.luidDeviceNodeMask << "\n"
     << "      }";
}

std::string collect_comgr_version() {
  size_t major = 0;
  size_t minor = 0;
  amd_comgr_get_version(&major, &minor);
  return std::to_string(static_cast<unsigned long long>(major)) + "." +
         std::to_string(static_cast<unsigned long long>(minor));
}

void append_runtime_json(std::ostringstream& os) {
  os << "{\n";

  const int hip_version = HIP_VERSION;
  os << "    \"hip_runtime_version\": " << quote(version_string_from_int(hip_version)) << ",\n"
     << "    \"comgr_version\": " << quote(collect_comgr_version()) << "\n"
     << "  }";
}

struct DeviceMetadata {
  std::string devices_json;
  int captured_count = 0;
};

DeviceMetadata collect_device_metadata(int device_count) {
  std::ostringstream devices;
  devices << "[";

  bool first_device = true;
  int captured_count = 0;

  for (int device = 0; device < device_count; ++device) {
    hipDeviceProp_t prop{};
    const hipError_t prop_err = hip::ihipGetDeviceProperties(&prop, device);
    if (prop_err != hipSuccess) continue;

    if (!first_device) devices << ",";
    first_device = false;
    ++captured_count;
    devices << "\n"
            << "    {\n"
            << "      \"ordinal\": " << device << ",\n";
    append_prop_fields(devices, prop);
    devices << "\n"
            << "    }";
  }

  devices << "\n"
          << "  ]";
  return {devices.str(), captured_count};
}

}  // namespace

std::string collect_json() {
  std::ostringstream os;
  os << "{\n"
     << "  \"schema_version\": 1,\n"
     << "  \"runtime\": ";
  append_runtime_json(os);
  os << ",\n";

  const int count = static_cast<int>(hip::g_devices.size());
  DeviceMetadata device_metadata = collect_device_metadata(count);
  os << "  \"device_count\": " << count << ",\n";
  os << "  \"captured_device_count\": " << device_metadata.captured_count << ",\n";
  os << "  \"devices\": " << device_metadata.devices_json << "\n"
     << "}";
  return os.str();
}

}  // namespace metadata
}  // namespace hrr_cap
