// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Device text sanitising. Firmware, drivers and the kernel supply names and versions that
// reach the debug log and the CLI, so every byte must be printable ASCII.

#ifndef AMDSMI_UNIFIED_SMI_NIC_TEXT_H_
#define AMDSMI_UNIFIED_SMI_NIC_TEXT_H_

#include <cstddef>
#include <cstring>
#include <string>

namespace amd::smi::nic::text {

constexpr char kPrintableAsciiMin = ' ';
constexpr char kPrintableAsciiMax = '~';
constexpr char kNonPrintableReplacement = '?';

// Escape sequences and line breaks from a device would forge log lines or move the terminal
// cursor, and bytes that are not UTF-8 would break decoding in the callers.
inline auto printable_ascii(std::string text) -> std::string {
  for (auto& ch : text) {
    if ((ch < kPrintableAsciiMin) || (ch > kPrintableAsciiMax)) {
      ch = kNonPrintableReplacement;
    }
  }
  return text;
}

// A fixed-size text field from a kernel structure is not guaranteed to be NUL-terminated.
template <size_t N>
auto printable_field(const char (&field)[N]) -> std::string {
  return printable_ascii(std::string(field, strnlen(field, N)));
}

}  // namespace amd::smi::nic::text

#endif  // AMDSMI_UNIFIED_SMI_NIC_TEXT_H_
