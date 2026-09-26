// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "src/cuid_device_manager.h"
#include "src/cuid_gpu.h"
#include "src/cuid_util.h"
#include "src/hmac.h"

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

std::vector<uint8_t> from_hex(const std::string& hex) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2)
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  return out;
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
        !query(current, AMDCUID_QUERY_SOURCE, source) || source != AMDCUID_SOURCE_LIBRARY)
      return false;
  }
  return true;
}

bool key_info(const std::string& want_status, const std::string& fingerprint,
              const std::string& provisioned) {
  amdcuid_key_info_t info{};
  const auto status = amdcuid_get_key_info(&info);
  if (want_status != amdcuid_status_to_string(status)) {
    std::cerr << "key info: " << amdcuid_status_to_string(status) << '\n';
    return false;
  }
  if (status != AMDCUID_STATUS_SUCCESS) return true;
  const auto want = from_hex(fingerprint);
  if (want.size() == sizeof(info.fingerprint) &&
      std::memcmp(want.data(), info.fingerprint, sizeof(info.fingerprint)) == 0 &&
      info.provisioned == std::stoi(provisioned))
    return true;
  std::cerr << "key info: provisioned " << int(info.provisioned) << ", fingerprint";
  for (const auto octet : info.fingerprint) std::cerr << ' ' << int(octet);
  std::cerr << '\n';
  return false;
}

// Every CPU, NIC and Platform handle: temporary without a key; with one,
// derived from its primary under exactly that key.
bool components(const std::string& mode, const std::string& key_hex) {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return false;
  std::vector<amdcuid_id_t> handles(count);
  if (amdcuid_get_all_handles(handles.data(), &count) != AMDCUID_STATUS_SUCCESS) return false;
  const auto key = from_hex(key_hex);
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
    if (temporary != (mode == "temporary") || temporary != auxiliary_bit(handle)) {
      std::cerr << "type " << type << ": temporary=" << temporary << '\n';
      return false;
    }
    if (mode != "keyed") continue;
    amdcuid_primary_id primary{};
    if (!query(handle, AMDCUID_QUERY_PRIMARY_CUID, primary.UUIDv8_representation)) return false;
    if (CuidUtilities::is_constructed(&primary.UUIDv8_representation))
      CuidUtilities::remove_UUIDv8_bits(&primary.UUIDv8_representation, primary.raw_bits);
    else
      std::memcpy(primary.raw_bits, primary.UUIDv8_representation.bytes, sizeof(primary.raw_bits));
    cuid_hmac hmac(reinterpret_cast<const char*>(key.data()), key.size());
    amdcuid_derived_id derived{};
    if (CuidUtilities::generate_derived_cuid(&primary, &derived, &hmac) != AMDCUID_STATUS_SUCCESS ||
        !same(derived.UUIDv8_representation, handle)) {
      std::cerr << "type " << type << ": not derived with the expected key\n";
      return false;
    }
  }
  if (!(cpu && nic && platform))
    std::cerr << "cpu=" << cpu << " nic=" << nic << " platform=" << platform << '\n';
  return cpu && nic && platform;
}

std::string partition(unsigned n) {
  return n == 0 ? "/sys/bus/pci/devices/0000:03:00.0/xcp"
                : "/sys/devices/platform/amdgpu_xcp." + std::to_string(n) + "/xcp";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string action(argv[1]);
  if (action == "rotate") {
    uint8_t key[32];
    for (uint8_t i = 0; i < sizeof(key); ++i) key[i] = static_cast<uint8_t>(0x40 + i);
    const auto status = amdcuid_set_hash_key(key);
    std::cout << amdcuid_status_to_string(status) << std::endl;
    return status == AMDCUID_STATUS_SUCCESS ? 0 : 1;
  }
  if (action == "set-key" && argc == 3) {
    const auto key = from_hex(argv[2]);
    if (key.size() != key_length) return 2;
    const auto status = amdcuid_set_hash_key(key.data());
    if (status != AMDCUID_STATUS_SUCCESS) {
      std::cerr << "set: " << amdcuid_status_to_string(status) << '\n';
      return 1;
    }
    uint8_t want[4 + kKeyVariablePayloadLen];
    CuidUtilities::build_key_variable(key.data(), want);
    std::ifstream in(kKeyVariablePath, std::ios::binary);
    const std::vector<uint8_t> got{std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>()};
    struct stat st{};
    if (got.size() != sizeof(want) || std::memcmp(got.data(), want, sizeof(want)) != 0 ||
        stat(kKeyVariablePath, &st) != 0 || (st.st_mode & 07777) != 0600)
      return 1;
    return 0;
  }
  if (action == "key-info" && argc >= 3) {
    return key_info(argv[2], argc > 3 ? argv[3] : "", argc > 4 ? argv[4] : "0") ? 0 : 1;
  }
  if (action == "components" && argc >= 3) {
    return components(argv[2], argc > 3 ? argv[3] : "") ? 0 : 1;
  }
  if (action == "temporary-gpu") {
    amdcuid_id_t first{}, second{}, id{};
    if (!whole_devices(first, second)) return 1;
    return temporary_gpu({argv + 2, argv + argc}, id) ? 0 : 1;
  }
  if (action == "wait-publish" && argc == 3) {
    amdcuid_id_t before{}, after{}, other{}, expected{};
    if (CuidUtilities::uuid_string_to_uint8(argv[2], expected.bytes) != AMDCUID_STATUS_SUCCESS)
      return 2;
    if (!temporary_gpu({"bdf=0000:03:00.0", "/dev/dri/renderD128"}, before)) return 1;
    std::cout << "ready" << std::endl;
    std::string line;
    std::getline(std::cin, line);
    bool temporary = true;
    amdcuid_source_t source = AMDCUID_SOURCE_UNKNOWN;
    for (const char* spelling : {"bdf=0000:03:00.0", "/dev/dri/renderD128"}) {
      if (lookup(spelling, after) != AMDCUID_STATUS_SUCCESS || !same(after, expected) ||
          auxiliary_bit(after) || !query(after, AMDCUID_QUERY_TEMPORARY_CUID, temporary) ||
          temporary || !query(after, AMDCUID_QUERY_SOURCE, source) ||
          source != AMDCUID_SOURCE_DRIVER)
        return 1;
    }
    return whole_devices(after, other) && same(after, expected) ? 0 : 1;
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
      if (vf.driver_primary_cuid(primary) != AMDCUID_STATUS_UNSUPPORTED) return 1;
      const auto primary_status = vf.get_primary_cuid(primary);
      amdcuid_derived_id derived{};
      const auto derived_status = vf.get_derived_cuid(derived);
      if (!vf.get_info().partition_attr_dir.empty()) {
        if (primary_status != AMDCUID_STATUS_UNSUPPORTED ||
            derived_status != AMDCUID_STATUS_UNSUPPORTED)
          return 1;
      } else if (primary_status != AMDCUID_STATUS_SUCCESS || !(primary.raw_bits[14] & 0x20) ||
                 derived_status != AMDCUID_STATUS_SUCCESS || !(derived.raw_bits[14] & 0x20) ||
                 vf.derived_source() != AMDCUID_SOURCE_LIBRARY) {
        return 1;
      }
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
  if (action == "wait-rotate" || action == "wait-reject" || action == "wait-reject-xcp") {
    amdcuid_id_t first{}, second{};
    if (!whole_devices(first, second)) return 1;
    amdcuid_id_t existing_partition{};
    if (action == "wait-reject-xcp" &&
        lookup(partition(0), existing_partition) != AMDCUID_STATUS_SUCCESS)
      return 1;
    std::cout << "ready" << std::endl;
    std::string line;
    std::getline(std::cin, line);
    amdcuid_id_t current{}, other{};
    if (action == "wait-reject" || action == "wait-reject-xcp") {
      if (argc != 3 || lookup(argv[2], current) != AMDCUID_STATUS_INVALID_FORMAT) return 1;
      if (!whole_devices(current, other) || !same(first, current) ||
          CuidDeviceManager::instance().devices().size() != (action == "wait-reject" ? 2 : 3))
        return 1;
      if (action == "wait-reject-xcp" && (lookup(partition(0), current) != AMDCUID_STATUS_SUCCESS ||
                                          !same(current, existing_partition)))
        return 1;
    } else {
      for (const char* path : {"/dev/dri/renderD128", "/sys/bus/pci/devices/0000:03:00.0",
                               "/sys/class/drm/card0", "/sys/class/drm/renderD128"}) {
        if (lookup(path, current) != AMDCUID_STATUS_SUCCESS || same(first, current) ||
            !usable(current))
          return 1;
        if (!whole_devices(current, other) || same(first, current) || same(second, other)) return 1;
      }
      uint32_t length = sizeof(current);
      if (amdcuid_query_device_property(first, AMDCUID_QUERY_DERIVED_CUID, &current, &length) !=
          AMDCUID_STATUS_DEVICE_NOT_FOUND)
        return 1;
    }
    return 0;
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
      if (geteuid() != 0 && argc > 2 && std::string(argv[2]) == "unknown-metadata" &&
          metadata != AMDCUID_STATUS_UNSUPPORTED)
        return 1;
      if (metadata == AMDCUID_STATUS_SUCCESS) {
        if (unit != (0x40u | n)) return 1;
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
      for (unsigned n = 0; n < 64; ++n) {
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
  return 2;
}
