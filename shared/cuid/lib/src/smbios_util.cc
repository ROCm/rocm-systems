// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smbios_util.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "cuid_util.h"

amdcuid_status_t SmbiosUtil::get_system_uuid(uint8_t* uuid) {
  if (!uuid) {
    return AMDCUID_STATUS_INVALID_ARGUMENT;
  }

  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // attempt to read UUID from sysfs first
  std::string uuid_str = CuidUtilities::read_sysfs_file(std::string(DMI_PATH) + "product_uuid");
  if (!uuid_str.empty()) {
    // Convert the UUID string to uint8_t array
    amdcuid_status_t status = CuidUtilities::uuid_string_to_uint8(uuid_str, uuid);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }
  } else {
    // try and get UUID from SMBIOS table
    amdcuid_status_t status = SmbiosUtil::get_uuid_from_smbios_table(uuid);
    if (status != AMDCUID_STATUS_SUCCESS) {
      return status;
    }
  }

  return AMDCUID_STATUS_SUCCESS;
}

namespace {

// read(2) reports a read error as a failed read; a stream iterator throws it
// out of this status-returning interface.
bool read_file(const std::string& path, std::vector<uint8_t>& out) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  out.clear();
  uint8_t buffer[4096];
  for (;;) {
    const ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) {
      close(fd);
      return false;
    }
    if (n == 0) break;
    out.insert(out.end(), buffer, buffer + n);
  }
  close(fd);
  return true;
}

}  // namespace

amdcuid_status_t SmbiosUtil::parse_entry_point_version(const uint8_t* entry, size_t length,
                                                       uint16_t& version) {
  if (!entry) return AMDCUID_STATUS_INVALID_ARGUMENT;
  // The 32-bit entry point, "_SM_", has the major and minor version at 0x6
  // and 0x7; the 64-bit one, "_SM3_", at 0x7 and 0x8.
  if (length >= 0x8 && std::memcmp(entry, "_SM_", 4) == 0) {
    version = static_cast<uint16_t>(entry[0x6] << 8 | entry[0x7]);
  } else if (length >= 0x9 && std::memcmp(entry, "_SM3_", 5) == 0) {
    version = static_cast<uint16_t>(entry[0x7] << 8 | entry[0x8]);
  } else {
    return AMDCUID_STATUS_SMBIOS_ERROR;
  }
  // The System UUID field arrived in SMBIOS 2.1.
  return version < 0x0201 ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t SmbiosUtil::parse_system_uuid(const uint8_t* table, size_t length,
                                               uint16_t version, uint8_t* uuid) {
  if (!table || !uuid) return AMDCUID_STATUS_INVALID_ARGUMENT;
  constexpr uint8_t kSystemInformation = 1;
  constexpr uint8_t kEndOfTable = 127;
  constexpr size_t kUuidOffset = 0x8;
  constexpr size_t kUuidLength = 16;
  size_t offset = 0;
  while (offset + 4 <= length) {
    const uint8_t type = table[offset];
    const size_t formatted = table[offset + 1];
    if (formatted < 4 || offset + formatted > length) return AMDCUID_STATUS_SMBIOS_ERROR;
    // The string set follows the formatted area and ends with two NULs. Like
    // Linux's dmi_table(), a structure whose string set the table cuts off is
    // not read.
    size_t next = offset + formatted;
    while (next + 1 < length && (table[next] != 0 || table[next + 1] != 0)) ++next;
    if (next + 1 >= length) return AMDCUID_STATUS_SMBIOS_ERROR;
    if (type == kSystemInformation) {
      // A structure too short to hold the field has no UUID, as Linux's
      // dmi_save_uuid() also decides.
      if (formatted < kUuidOffset + kUuidLength) return AMDCUID_STATUS_SMBIOS_ERROR;
      const uint8_t* raw = table + offset + kUuidOffset;
      // From SMBIOS 2.6 the first three fields are little-endian; before it
      // the UUID is stored as written. Linux reads it the same way.
      if (version >= 0x0206) {
        const uint8_t swapped[8] = {raw[3], raw[2], raw[1], raw[0], raw[5], raw[4], raw[7], raw[6]};
        std::memcpy(uuid, swapped, sizeof(swapped));
        std::memcpy(uuid + 8, raw + 8, 8);
      } else {
        std::memcpy(uuid, raw, kUuidLength);
      }
      return AMDCUID_STATUS_SUCCESS;
    }
    if (type == kEndOfTable) break;
    offset = next + 2;
  }
  return AMDCUID_STATUS_SMBIOS_ERROR;
}

amdcuid_status_t SmbiosUtil::get_uuid_from_smbios_table(uint8_t* uuid) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  std::vector<uint8_t> entry, table;
  if (!read_file(std::string(DMI_TABLES_PATH) + "smbios_entry_point", entry) ||
      !read_file(std::string(DMI_TABLES_PATH) + "DMI", table))
    return AMDCUID_STATUS_SMBIOS_ERROR;
  uint16_t version = 0;
  const amdcuid_status_t status = parse_entry_point_version(entry.data(), entry.size(), version);
  if (status != AMDCUID_STATUS_SUCCESS) return status;
  return parse_system_uuid(table.data(), table.size(), version, uuid);
}

// Get system serial number
amdcuid_status_t SmbiosUtil::get_system_serial(std::string& serial) {
  if (geteuid() != 0) {
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }

  // Try multiple sources in order of preference
  const char* serial_files[] = {"product_serial", "board_serial", "chassis_serial"};

  for (const char* filename : serial_files) {
    std::string path = std::string(DMI_PATH) + filename;
    serial = CuidUtilities::read_sysfs_file(path);

    if (!serial.empty()) {
      return AMDCUID_STATUS_SUCCESS;
    }
  }

  return AMDCUID_STATUS_SMBIOS_ERROR;
}
