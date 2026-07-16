/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_capture_metadata.h"

#include "amd_comgr/amd_comgr.h"
#include "hip/hip_runtime_api.h"

#include <cstdio>
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
  for (const unsigned char c : std::string(s)) {
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

std::string hip_error_name(hipError_t err) {
  switch (err) {
    case hipSuccess: return "hipSuccess";
    case hipErrorInvalidValue: return "hipErrorInvalidValue";
    case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
    case hipErrorNoDevice: return "hipErrorNoDevice";
    case hipErrorNotInitialized: return "hipErrorNotInitialized";
    default: return "hipError(" + std::to_string(static_cast<int>(err)) + ")";
  }
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

void append_prop_fields(std::ostringstream& os, const hipDeviceProp_tR0600& prop) {
  os << "      \"properties\": {\n"
     << "        \"name\": " << quote(prop.name) << ",\n"
     << "        \"gcn_arch_name\": " << quote(prop.gcnArchName) << ",\n"
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
     << "        \"compute_capability\": { \"major\": " << prop.major
     << ", \"minor\": " << prop.minor << " },\n"
     << "        \"pci\": { \"domain\": " << prop.pciDomainID
     << ", \"bus\": " << prop.pciBusID
     << ", \"device\": " << prop.pciDeviceID << " },\n"
     << "        \"uuid\": " << quote(uuid_to_hex(prop.uuid)) << ",\n"
     << "        \"luid\": " << quote(bytes_to_hex(prop.luid, sizeof(prop.luid))) << ",\n"
     << "        \"luid_device_node_mask\": " << prop.luidDeviceNodeMask << "\n"
     << "      }";
}

std::string collect_comgr_json() {
  size_t major = 0;
  size_t minor = 0;
  amd_comgr_get_version(&major, &minor);
  std::ostringstream os;
  os << "{\n"
     << "    \"available\": true,\n"
     << "    \"major\": " << static_cast<unsigned long long>(major) << ",\n"
     << "    \"minor\": " << static_cast<unsigned long long>(minor) << ",\n"
     << "    \"source\": \"amd_comgr_get_version\"\n"
     << "  }";
  return os.str();
}

std::string collect_runtime_json() {
  std::ostringstream os;
  os << "{\n";

  const int hip_version = HIP_VERSION;
  os << "    \"hip_runtime_version\": " << quote(version_string_from_int(hip_version)) << ",\n"
     << "    \"hip_driver_version\": " << quote(version_string_from_int(hip_version));

  os << ",\n"
     << "    \"comgr\": " << collect_comgr_json() << "\n"
     << "  }";
  return os.str();
}

std::string collect_devices_json() {
  std::ostringstream os;
  os << "[";

  const int count = static_cast<int>(hip::g_devices.size());

  for (int device = 0; device < count; ++device) {
    if (device > 0) os << ",";
    os << "\n"
       << "    {\n"
       << "      \"ordinal\": " << device << ",\n";

    hipDeviceProp_tR0600 prop{};
    const hipError_t prop_err = hip::ihipGetDeviceProperties(&prop, device);
    if (prop_err == hipSuccess) {
      append_prop_fields(os, prop);
    } else {
      os << "      \"properties_error\": " << quote(hip_error_name(prop_err)) << "\n";
    }

    os << "\n"
       << "    }";
  }

  os << "\n"
     << "  ]";
  return os.str();
}

}  // namespace

std::string Metadata::collect_json() const {
  std::ostringstream os;
  os << "{\n"
     << "  \"schema_version\": 1,\n"
     << "  \"runtime\": " << collect_runtime_json() << ",\n";

  int count = 0;
  count = static_cast<int>(hip::g_devices.size());
  os << "  \"device_count\": " << count << ",\n";

  os << "  \"devices\": " << collect_devices_json() << "\n"
     << "}";
  return os.str();
}

std::string collect_json() {
  return Metadata().collect_json();
}

}  // namespace metadata
}  // namespace hrr_cap
