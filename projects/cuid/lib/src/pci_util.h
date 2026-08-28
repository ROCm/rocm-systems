// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef PCI_UTIL_H
#define PCI_UTIL_H

#include "include/amd_cuid.h"
#include <cstdint>
#include <string>
#include <vector>

class PciUtil {
public:
  static amdcuid_status_t read_pci_config_space(std::string bdf,
                                                uint8_t *buffer,
                                                size_t buffer_size,
                                                uint16_t offset);
  static amdcuid_status_t get_pci_dsn_cap_offset(std::string bdf,
                                                 uint16_t &offset);
  static amdcuid_status_t get_pci_vsec_cap_offset(std::string bdf,
                                                  uint16_t &offset);

  // Endianness conversion utilities
  static uint16_t le16_to_be16(uint16_t value);
  static uint64_t le64_to_be64(uint64_t value);
};

#endif // PCI_UTIL_H
