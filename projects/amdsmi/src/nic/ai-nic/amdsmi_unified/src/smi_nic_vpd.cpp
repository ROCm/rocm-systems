// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_vpd.h"

#include <cstddef>

namespace amd::smi::nic::vpd {

namespace {

// PCI VPD resource-tag constants (PCI Local Bus spec, 6.4).
constexpr uint8_t kLargeResourceFlag = 0x80;    // bit7 set => large resource
constexpr uint8_t kLargeItemMask = 0x7f;        // large item name (bits 6:0)
constexpr uint8_t kLargeItemIdentifier = 0x02;  // Identifier String (product name)
constexpr uint8_t kLargeItemVpdR = 0x10;        // VPD read-only keyword block
constexpr uint8_t kSmallItemShift = 3;          // small item name (bits 6:3)
constexpr uint8_t kSmallItemMask = 0x0f;
constexpr uint8_t kSmallLenMask = 0x07;  // small length (bits 2:0)
constexpr uint8_t kSmallItemEnd = 0x0f;  // End Tag

// VPD strings are ASCII, right-padded with spaces or NULs; drop that trailing
// filler so callers see the bare value.
std::optional<std::string> trimmed(const uint8_t* data, size_t len) {
  size_t end = len;
  while (end > 0 && (data[end - 1] == ' ' || data[end - 1] == '\0')) {
    --end;
  }
  if (end == 0) {
    return std::nullopt;
  }
  return std::string(reinterpret_cast<const char*>(data), end);
}

// Decodes the { keyword[2], len[1], data[len] } entries inside a VPD-R block.
void parse_vpd_r(const std::vector<uint8_t>& image, size_t start, size_t end, VpdFields& out) {
  size_t j = start;
  while (j + 3 <= end) {
    const char k0 = static_cast<char>(image[j]);
    const char k1 = static_cast<char>(image[j + 1]);
    const size_t klen = image[j + 2];
    if (j + 3 + klen > end) {
      return;  // truncated keyword; stop rather than read past the block
    }
    const uint8_t* kdata = image.data() + j + 3;
    if (k0 == 'P' && k1 == 'N') {
      out.part_number = trimmed(kdata, klen);
    } else if (k0 == 'S' && k1 == 'N') {
      out.serial_number = trimmed(kdata, klen);
    }
    j += 3 + klen;
  }
}

}  // namespace

VpdFields parse_pci_vpd(const std::vector<uint8_t>& image) {
  VpdFields out;
  size_t i = 0;
  while (i < image.size()) {
    const uint8_t tag = image[i];
    if (tag & kLargeResourceFlag) {
      if (i + 3 > image.size()) {
        break;  // incomplete large-resource header
      }
      const uint8_t item = tag & kLargeItemMask;
      const size_t len = image[i + 1] | (static_cast<size_t>(image[i + 2]) << 8);
      const size_t data = i + 3;
      if (data + len > image.size()) {
        break;  // declared length runs past the image
      }
      if (item == kLargeItemIdentifier) {
        out.product_name = trimmed(image.data() + data, len);
      } else if (item == kLargeItemVpdR) {
        parse_vpd_r(image, data, data + len, out);
      }
      i = data + len;
    } else {
      const uint8_t item = (tag >> kSmallItemShift) & kSmallItemMask;
      if (item == kSmallItemEnd) {
        break;
      }
      i += 1 + (tag & kSmallLenMask);
    }
  }
  return out;
}

}  // namespace amd::smi::nic::vpd
