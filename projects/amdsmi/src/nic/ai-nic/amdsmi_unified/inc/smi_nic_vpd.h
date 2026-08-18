// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_SMI_NIC_VPD_H_
#define AMDSMI_UNIFIED_SMI_NIC_VPD_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace amd::smi::nic::vpd {

// Identity fields carried in a PCI Vital Product Data image. Each stays nullopt
// when the corresponding resource/keyword is absent or malformed.
struct VpdFields {
  std::optional<std::string> product_name;   // Identifier String resource (0x82)
  std::optional<std::string> part_number;    // VPD-R keyword "PN"
  std::optional<std::string> serial_number;  // VPD-R keyword "SN"
};

/**
 * Parses a raw PCI VPD image (the bytes exposed at
 * /sys/bus/pci/devices/<bdf>/vpd) per the PCI Local Bus spec resource-tag
 * layout. Walks the large/small resource tags rather than substring-searching,
 * so keyword lengths are honoured. Robust to truncation: parsing stops at the
 * end tag or the first out-of-bounds length, returning whatever was decoded.
 */
VpdFields parse_pci_vpd(const std::vector<uint8_t>& image);

}  // namespace amd::smi::nic::vpd

#endif  // AMDSMI_UNIFIED_SMI_NIC_VPD_H_
