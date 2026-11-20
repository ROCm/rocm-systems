// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef SMBIOS_UTIL_H
#define SMBIOS_UTIL_H

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "include/amd_cuid.h"

class SmbiosUtil {
 public:
  static amdcuid_status_t get_system_uuid(uint8_t* uuid);
  static amdcuid_status_t get_uuid_from_smbios_table(uint8_t* uuid);
  // The SMBIOS version, major << 8 | minor, from an entry point; UNSUPPORTED
  // before 2.1, which has no System UUID.
  static amdcuid_status_t parse_entry_point_version(const uint8_t* entry, size_t length,
                                                    uint16_t& version);
  // The System UUID of the Type 1 structure of a structure table, in the byte
  // order Linux's product_uuid shows for that SMBIOS version.
  static amdcuid_status_t parse_system_uuid(const uint8_t* table, size_t length, uint16_t version,
                                            uint8_t* uuid);
  static amdcuid_status_t get_system_serial(std::string& serial);

 private:
  static constexpr const char* DMI_PATH = "/sys/class/dmi/id/";
  static constexpr const char* DMI_TABLES_PATH = "/sys/firmware/dmi/tables/";
};

#endif  // SMBIOS_UTIL_H
