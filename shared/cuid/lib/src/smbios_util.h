// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef SMBIOS_UTIL_H
#define SMBIOS_UTIL_H

#include <iostream>

#include "include/amd_cuid.h"

class SmbiosUtil {
 public:
  static amdcuid_status_t get_system_uuid(uint8_t* uuid);
  static amdcuid_status_t get_uuid_from_smbios_table(uint8_t* uuid);
  static amdcuid_status_t get_system_serial(std::string& serial);

 private:
  static constexpr const char* DMI_PATH = "/sys/class/dmi/id/";
  static constexpr const char* DMI_TABLES_PATH = "/sys/firmware/dmi/tables/";
};

#endif  // SMBIOS_UTIL_H
