// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/cuid_device_manager.h"
#include "src/cuid_gpu.h"
#include "src/cuid_util.h"

namespace {
bool same(const amdcuid_id_t& a, const amdcuid_id_t& b) {
  return std::memcmp(a.bytes, b.bytes, 16) == 0;
}

amdcuid_status_t lookup(const std::string& name, amdcuid_id_t& id) {
  if (name.compare(0, 4, "bdf=") == 0)
    return amdcuid_get_handle_by_bdf(name.substr(4).c_str(), AMDCUID_DEVICE_TYPE_GPU, &id);
  if (name.compare(0, 3, "fd=") == 0) {
    const int fd = open(name.substr(3).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return AMDCUID_STATUS_FILE_ERROR;
    const auto status = amdcuid_get_handle_by_fd(fd, AMDCUID_DEVICE_TYPE_GPU, &id);
    close(fd);
    return status;
  }
  return amdcuid_get_handle_by_dev_path(name.c_str(), AMDCUID_DEVICE_TYPE_GPU, &id);
}

bool usable(amdcuid_id_t id) {
  amdcuid_id_t derived{};
  uint32_t length = sizeof(derived);
  return amdcuid_query_device_property(id, AMDCUID_QUERY_DERIVED_CUID, &derived, &length) ==
             AMDCUID_STATUS_SUCCESS &&
         same(id, derived);
}

bool whole_devices(amdcuid_id_t& first, amdcuid_id_t& second) {
  return lookup("bdf=0000:03:00.0", first) == AMDCUID_STATUS_SUCCESS &&
         lookup("bdf=0000:63:00.0", second) == AMDCUID_STATUS_SUCCESS && !same(first, second) &&
         usable(first) && usable(second);
}

template <typename T>
bool query(amdcuid_id_t id, amdcuid_query_t what, T& value) {
  uint32_t length = sizeof(value);
  return amdcuid_query_device_property(id, what, &value, &length) == AMDCUID_STATUS_SUCCESS;
}

// Payload bit 117, the Auxiliary Value Identifier.
bool auxiliary_bit(amdcuid_id_t id) {
  uint8_t raw[16]{};
  CuidUtilities::remove_UUIDv8_bits(&id, raw);
  return raw[14] & 0x20;
}

// A temporary handle has no primary: root is refused with UNSUPPORTED and the
// buffer is left alone; anyone else is refused for privilege first. Every
// other query still answers.
bool no_primary(amdcuid_id_t id) {
  amdcuid_id_t primary;
  std::memset(&primary, 0xA5, sizeof(primary));
  const amdcuid_id_t sentinel = primary;
  uint32_t length = sizeof(primary);
  const auto status =
      amdcuid_query_device_property(id, AMDCUID_QUERY_PRIMARY_CUID, &primary, &length);
  const auto want = geteuid() == 0 ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_PERMISSION_DENIED;
  if (status != want || !same(primary, sentinel)) {
    std::cerr << "primary of a temporary CUID: " << amdcuid_status_to_string(status) << '\n';
    return false;
  }
  amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
  uint16_t vendor = 0, unit = 0;
  char path[256] = {};
  length = sizeof(path);
  if (!query(id, AMDCUID_QUERY_DEVICE_TYPE, type)) return false;
  // A partition's metadata may be unknown to a caller without root, which
  // VENDOR_ID and UNIT_ID then report as UNSUPPORTED, temporary or not.
  const bool metadata = geteuid() == 0 || type != AMDCUID_DEVICE_TYPE_GPU;
  return (type == AMDCUID_DEVICE_TYPE_PLATFORM || !metadata ||
          (query(id, AMDCUID_QUERY_VENDOR_ID, vendor) && query(id, AMDCUID_QUERY_UNIT_ID, unit))) &&
         (type != AMDCUID_DEVICE_TYPE_GPU ||
          amdcuid_query_device_property(id, AMDCUID_QUERY_DEVICE_PATH, path, &length) ==
              AMDCUID_STATUS_SUCCESS);
}

// A whole GPU the driver does not name: temporary, computed here, and the same
// under every spelling.
bool temporary_gpu(const std::vector<std::string>& spellings, amdcuid_id_t& id) {
  for (size_t n = 0; n < spellings.size(); ++n) {
    amdcuid_id_t current{}, derived{};
    bool temporary = false;
    amdcuid_source_t source = AMDCUID_SOURCE_UNKNOWN;
    if (lookup(spellings[n], current) != AMDCUID_STATUS_SUCCESS) return false;
    if (n == 0) id = current;
    if (!same(id, current) || !query(current, AMDCUID_QUERY_DERIVED_CUID, derived) ||
        !same(current, derived) || !auxiliary_bit(current) ||
        !query(current, AMDCUID_QUERY_TEMPORARY_CUID, temporary) || !temporary ||
        !query(current, AMDCUID_QUERY_SOURCE, source) || source != AMDCUID_SOURCE_LIBRARY ||
        !no_primary(current))
      return false;
  }
  return true;
}

// Every CPU, NIC and Platform handle is temporary: there is no node key.
bool components() {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return false;
  std::vector<amdcuid_id_t> handles(count);
  if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return false;
  bool cpu = false, nic = false, platform = false;
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    bool temporary = false;
    amdcuid_source_t source = AMDCUID_SOURCE_UNKNOWN;
    if (!query(handle, AMDCUID_QUERY_DEVICE_TYPE, type)) return false;
    if (type == AMDCUID_DEVICE_TYPE_GPU) continue;
    cpu = cpu || type == AMDCUID_DEVICE_TYPE_CPU;
    nic = nic || type == AMDCUID_DEVICE_TYPE_NIC;
    platform = platform || type == AMDCUID_DEVICE_TYPE_PLATFORM;
    if (!query(handle, AMDCUID_QUERY_TEMPORARY_CUID, temporary) ||
        !query(handle, AMDCUID_QUERY_SOURCE, source) || source != AMDCUID_SOURCE_LIBRARY) {
      std::cerr << "type " << type << ": no temporariness or source\n";
      return false;
    }
    if (!temporary || !auxiliary_bit(handle)) {
      std::cerr << "type " << type << ": not temporary\n";
      return false;
    }
    if (!no_primary(handle)) {
      std::cerr << "type " << type << ": a temporary CUID reported a primary\n";
      return false;
    }
  }
  if (!(cpu && nic && platform))
    std::cerr << "cpu=" << cpu << " nic=" << nic << " platform=" << platform << '\n';
  return cpu && nic && platform;
}

// The two functions of the two-port NIC share a serial number but not an
// identity: each has its own handle, its PCI function as UnitID, and for root
// its own primary.
bool nic_functions() {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return false;
  std::vector<amdcuid_id_t> handles(count);
  if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return false;
  std::vector<std::string> units;
  std::vector<amdcuid_id_t> primaries;
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    if (!query(handle, AMDCUID_QUERY_DEVICE_TYPE, type) || type != AMDCUID_DEVICE_TYPE_NIC)
      continue;
    char bdf[16] = {};
    uint32_t length = sizeof(bdf);
    uint16_t unit = 0xFFFF;
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, bdf, &length) !=
            AMDCUID_STATUS_SUCCESS ||
        !query(handle, AMDCUID_QUERY_UNIT_ID, unit))
      return false;
    units.push_back(std::string(bdf) + "=" + std::to_string(unit));
    bool temporary = true;
    if (!query(handle, AMDCUID_QUERY_TEMPORARY_CUID, temporary)) return false;
    amdcuid_id_t primary{};
    if (geteuid() == 0 && !temporary) {
      if (!query(handle, AMDCUID_QUERY_PRIMARY_CUID, primary)) return false;
      primaries.push_back(primary);
    }
  }
  std::sort(units.begin(), units.end());
  if (units != std::vector<std::string>{"0000:07:00.0=0", "0000:07:00.1=1"}) {
    for (const auto& unit : units) std::cerr << "nic " << unit << '\n';
    return false;
  }
  return primaries.empty() || !same(primaries[0], primaries[1]);
}

// Every GPU handle, one sorted line each: device path, UnitID (or why not),
// whether it is temporary, its source and its CUID.
bool gpu_listing(std::vector<std::string>& lines) {
  lines.clear();
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return false;
  std::vector<amdcuid_id_t> handles(count);
  if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return false;
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    if (!query(handle, AMDCUID_QUERY_DEVICE_TYPE, type)) return false;
    if (type != AMDCUID_DEVICE_TYPE_GPU) continue;
    char path[512] = {};
    uint32_t length = sizeof(path);
    if (amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path, &length) !=
        AMDCUID_STATUS_SUCCESS)
      return false;
    uint16_t unit = 0;
    length = sizeof(unit);
    const auto unit_status =
        amdcuid_query_device_property(handle, AMDCUID_QUERY_UNIT_ID, &unit, &length);
    bool temporary = false;
    amdcuid_source_t source = AMDCUID_SOURCE_UNKNOWN;
    if (!query(handle, AMDCUID_QUERY_TEMPORARY_CUID, temporary) ||
        !query(handle, AMDCUID_QUERY_SOURCE, source) || !usable(handle))
      return false;
    char unit_text[16];
    std::snprintf(unit_text, sizeof(unit_text), "0x%03x", unit);
    lines.push_back(std::string(path) + "\t" +
                    (unit_status == AMDCUID_STATUS_SUCCESS
                         ? std::string(unit_text)
                         : std::string(amdcuid_status_to_string(unit_status))) +
                    "\t" + (temporary ? "temporary" : "permanent") + "\t" +
                    std::to_string(static_cast<int>(source)) + "\t" + amdcuid_id_to_string(handle));
  }
  std::sort(lines.begin(), lines.end());
  return true;
}

std::string partition(unsigned n) {
  return n == 0 ? "/sys/bus/pci/devices/0000:03:00.0/xcp"
                : "/sys/devices/platform/amdgpu_xcp_" + std::to_string(n) + "/xcp";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string action(argv[1]);
  if (action == "reject") {
    for (int n = 2; n < argc; ++n) {
      amdcuid_id_t id{};
      if (lookup(argv[n], id) == AMDCUID_STATUS_SUCCESS) {
        std::cerr << argv[n] << " unexpectedly resolved to " << amdcuid_id_to_string(id) << '\n';
        return 1;
      }
    }
    return 0;
  }
  if (action == "components") {
    return components() ? 0 : 1;
  }
  // The GPU listing, twice with a refresh between, which must agree; then
  // each argument looked up as a path or bdf=.
  if (action == "stock") {
    std::vector<std::string> first, second;
    if (!gpu_listing(first) || amdcuid_refresh() != AMDCUID_STATUS_SUCCESS || !gpu_listing(second))
      return 1;
    if (first != second) {
      std::cerr << "two enumerations disagree\n";
      return 1;
    }
    for (const auto& line : first) std::cout << line << '\n';
    for (int n = 2; n < argc; ++n) {
      amdcuid_id_t id{};
      const auto status = lookup(argv[n], id);
      std::cout << "lookup\t" << argv[n] << "\t"
                << (status == AMDCUID_STATUS_SUCCESS ? amdcuid_id_to_string(id)
                                                     : amdcuid_status_to_string(status))
                << '\n';
    }
    return 0;
  }
  if (action == "nic-functions") {
    return nic_functions() ? 0 : 1;
  }
  if (action == "temporary-gpu") {
    amdcuid_id_t first{}, second{}, id{};
    if (!whole_devices(first, second)) return 1;
    return temporary_gpu({argv + 2, argv + argc}, id) ? 0 : 1;
  }
  if (action == "foreign") {
    amdcuid_id_t first{}, second{}, wrong{};
    if (!whole_devices(first, second)) return 1;
    for (int n = 2; n < argc; ++n) {
      if (lookup(argv[n], wrong) == AMDCUID_STATUS_SUCCESS) {
        std::cerr << argv[n] << " resolved\n";
        return 1;
      }
    }
    uint32_t count = 0;
    if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return 1;
    std::vector<amdcuid_id_t> handles(count);
    if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return 1;
    unsigned gpus = 0;
    for (const auto& handle : handles) {
      amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
      if (query(handle, AMDCUID_QUERY_DEVICE_TYPE, type) && type == AMDCUID_DEVICE_TYPE_GPU) ++gpus;
    }
    return gpus == 2 && CuidDeviceManager::instance().devices().size() == 2 &&
                   whole_devices(first, second)
               ? 0
               : 1;
  }
  if (action == "unnamed-vf") {
    for (int n = 2; n < argc; ++n) {
      amdcuid_gpu_info info{};
      info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
      info.header.fields.gpu.vendor_id = 0x1002;
      info.render_node = argv[n];
      info.unnamed_vf = true;
      CuidGpu vf(info);
      uint64_t fingerprint = 0xDEADBEEF;
      const bool root = geteuid() == 0;
      const auto want =
          root ? AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND : AMDCUID_STATUS_PERMISSION_DENIED;
      if (vf.get_hardware_fingerprint(fingerprint) != want || (root && fingerprint != 0)) return 1;
      amdcuid_primary_id primary{};
      const auto primary_status = vf.get_primary_cuid(primary);
      amdcuid_derived_id derived{};
      const auto derived_status = vf.get_derived_cuid(derived);
      if (primary_status != AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND) return 1;
      if (derived_status != AMDCUID_STATUS_UNSUPPORTED) return 1;
    }
    return 0;
  }
  if (action == "sequence") {
    amdcuid_id_t expected{};
    for (int n = 2; n < argc; ++n) {
      amdcuid_id_t id{}, first{}, second{};
      const auto status = lookup(argv[n], id);
      if (status != AMDCUID_STATUS_SUCCESS) {
        std::cerr << "lookup " << n << ": " << amdcuid_status_to_string(status) << '\n';
        return 1;
      }
      if (n == 2) expected = id;
      if (!same(expected, id) || !usable(id) || !whole_devices(first, second) || !same(id, first))
        return 1;
      const auto devices = CuidDeviceManager::instance().devices();
      if (devices.size() != 2) return 1;
    }
    return 0;
  }
  if (action == "reject-fd") {
    amdcuid_id_t first{}, second{}, bogus{};
    if (!whole_devices(first, second)) return 1;
    if (lookup("fd=/sys/bus/pci/devices/0000:03:00.0/vendor", bogus) !=
        AMDCUID_STATUS_DEVICE_NOT_FOUND)
      return 1;
    return whole_devices(first, second) ? 0 : 1;
  }
  if (action == "not-gpu") {
    amdcuid_id_t first{}, second{}, wrong{};
    if (!whole_devices(first, second)) return 1;
    if (lookup("/sys/bus/pci/devices/0000:05:00.0", wrong) == AMDCUID_STATUS_SUCCESS) return 1;
    return whole_devices(first, second) ? 0 : 1;
  }
  if (action == "wait-reject") {
    amdcuid_id_t first{}, second{};
    if (!whole_devices(first, second)) return 1;
    std::cout << "ready" << std::endl;
    std::string line;
    std::getline(std::cin, line);
    amdcuid_id_t current{}, other{};
    if (argc != 3 || lookup(argv[2], current) != AMDCUID_STATUS_INVALID_FORMAT) return 1;
    return whole_devices(current, other) && same(first, current) &&
                   CuidDeviceManager::instance().devices().size() == 2
               ? 0
               : 1;
  }
  if (action == "unpublished-partitions") {
    amdcuid_id_t first{}, second{}, id{};
    if (!temporary_gpu({"bdf=0000:03:00.0", "/dev/dri/renderD128"}, first)) return 1;
    std::vector<amdcuid_id_t> ids{first};
    for (unsigned n = 0; n < 8; ++n) {
      if (!temporary_gpu({partition(n)}, id)) {
        std::cerr << "partition " << n << ": no temporary CUID\n";
        return 1;
      }
      for (const auto& prior : ids)
        if (same(prior, id)) {
          std::cerr << "partition " << n << ": collides\n";
          return 1;
        }
      ids.push_back(id);
    }
    return whole_devices(id, second) && same(first, id) ? 0 : 1;
  }
  if (action == "partitions" || action == "wait-unpublish") {
    amdcuid_id_t first{}, second{};
    if (!whole_devices(first, second)) return 1;
    std::vector<amdcuid_id_t> ids{first, second};
    for (unsigned n = 0; n < 8; ++n) {
      amdcuid_id_t id{}, alias{};
      if (lookup(partition(n), id) != AMDCUID_STATUS_SUCCESS || !usable(id)) return 1;
      uint16_t unit = 0;
      uint32_t size = sizeof(unit);
      const auto metadata = amdcuid_query_device_property(id, AMDCUID_QUERY_UNIT_ID, &unit, &size);
      if (metadata == AMDCUID_STATUS_SUCCESS) {
        if (unit != CuidUtilities::partition_unit_id(8, n)) return 1;
      } else if (geteuid() == 0 || (argc > 2 && std::string(argv[2]) == "metadata") ||
                 metadata != AMDCUID_STATUS_UNSUPPORTED)
        return 1;
      for (const auto& prior : ids)
        if (same(prior, id)) return 1;
      ids.push_back(id);
      if (n > 0) {
        const auto render = std::to_string(139 + n);
        for (const auto& path : {"/sys/class/drm/renderD" + render, "/dev/dri/renderD" + render,
                                 "/sys/class/drm/card" + std::to_string(10 + n)}) {
          if (lookup(path, alias) != AMDCUID_STATUS_SUCCESS || !same(alias, id)) return 1;
        }
      }
    }
    if (action == "wait-unpublish") {
      std::cout << "ready" << std::endl;
      std::string line;
      std::getline(std::cin, line);
      for (unsigned n = 0; n < 8; ++n) {
        amdcuid_id_t id{};
        if (lookup(partition(n), id) != AMDCUID_STATUS_UNSUPPORTED) return 1;
        if (n &&
            lookup("/dev/dri/renderD" + std::to_string(139 + n), id) != AMDCUID_STATUS_UNSUPPORTED)
          return 1;
      }
    }
    amdcuid_id_t again{}, other{};
    return whole_devices(again, other) && same(first, again) && same(second, other) ? 0 : 1;
  }
  if (action == "gim") {
    uint32_t count = 0;
    if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return 1;
    std::vector<amdcuid_id_t> handles(count);
    if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return 1;
    std::vector<std::string> gpus;
    for (const auto& handle : handles) {
      char bdf[16] = {};
      uint32_t length = sizeof(bdf);
      if (amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, bdf, &length) !=
          AMDCUID_STATUS_SUCCESS)
        return 1;
      uint16_t vendor = 0, device = 0, pci_class = 0;
      uint8_t revision = 0;
      bool temporary = false;
      if (!query(handle, AMDCUID_QUERY_VENDOR_ID, vendor) ||
          !query(handle, AMDCUID_QUERY_DEVICE_ID, device) ||
          !query(handle, AMDCUID_QUERY_REVISION_ID, revision) ||
          !query(handle, AMDCUID_QUERY_PCI_CLASS, pci_class) || !usable(handle) ||
          !query(handle, AMDCUID_QUERY_TEMPORARY_CUID, temporary) || !temporary)
        return 1;
      gpus.push_back(std::string(bdf) + " " + std::to_string(vendor) + ":" +
                     std::to_string(device) + ":" + std::to_string(revision) + ":" +
                     std::to_string(pci_class));
      uint64_t fingerprint = 0;
      if (std::string(bdf) == "0000:a1:00.0" && geteuid() == 0 &&
          (!query(handle, AMDCUID_QUERY_HARDWARE_FINGERPRINT, fingerprint) ||
           fingerprint != 0x1234abcd))
        return 1;
    }
    std::sort(gpus.begin(), gpus.end());
    const std::vector<std::string> want{"0000:03:00.0 4098:29603:1:768",
                                        "0000:63:00.0 4098:29603:1:768",
                                        "0000:a1:00.0 4098:29857:5:768"};
    if (gpus == want) return 0;
    for (const auto& gpu : gpus) std::cerr << "gpu " << gpu << '\n';
    return 1;
  }
  return 2;
}
