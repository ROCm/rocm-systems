/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
