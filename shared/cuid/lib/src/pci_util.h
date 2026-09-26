// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef PCI_UTIL_H
#define PCI_UTIL_H

#include <cstdint>
#include <string>
#include <vector>

#include "include/amd_cuid.h"

/// Size of a PCIe extended configuration space, in bytes.
constexpr size_t kPciConfigSpaceSize = 4096;

class PciUtil {
 public:
  static amdcuid_status_t read_pci_config_space(std::string bdf, uint8_t* buffer,
                                                size_t buffer_size, uint16_t offset);
  static amdcuid_status_t get_pci_dsn_cap_offset(std::string bdf, uint16_t& offset);
  static amdcuid_status_t get_pci_vsec_cap_offset(std::string bdf, uint16_t& offset);

  // Load a 16-bit little-endian config-space field. Use this instead of
  // casting the buffer to uint16_t*, which violates alignment and aliasing.
  static uint16_t load_le16(const uint8_t bytes[2]) {
    return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                                 (static_cast<uint16_t>(bytes[1]) << 8));
  }

  // Load a 64-bit little-endian config-space field, in practice the PCIe Device
  // Serial Number: its capability holds two dwords, low half first. Do not
  // byte-swap the result. Config space is little-endian and the payload is
  // packed little-endian, and the kernel's pci_get_dsn() assembles the dwords
  // low-first too, so a swap here names the same card with a byte-reversed
  // serial and hence a different CUID than the driver reports.
  static uint64_t load_le64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
      value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
  }
};

#endif  // PCI_UTIL_H
