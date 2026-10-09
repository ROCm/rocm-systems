// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __SMI_SYSFS_H__
#define __SMI_SYSFS_H__

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

class SmiSysfsReader {
 public:
  using SysfsValue = std::variant<int, std::string>;
  enum class SysfsStatus { Success = 0, FileNotFound, IOError, ParseError };

  static SysfsStatus readAll(const std::string& filepath, std::vector<SysfsValue>& content);
  // A successful read is not written to the debug log when is_success_logged is false;
  // failures are always logged. For high-volume scans such as PCI id discovery.
  static SysfsStatus readLine(const std::string& filepath, SysfsValue& content,
                              bool is_success_logged = true);
  static bool is_readable(const std::string& filepath);

  SmiSysfsReader() = delete;
};

// Numeric value of a sysfs field, base auto-detected. nullopt for text that is not a number
// (e.g. "Unknown speed"), so callers need no exception handling at the C ABI boundary.
inline std::optional<uint64_t> parse_sysfs_uint(const SmiSysfsReader::SysfsValue& value) {
  if (std::holds_alternative<int>(value)) {
    return static_cast<uint64_t>(std::get<int>(value));
  }
  try {
    return static_cast<uint64_t>(std::stoul(std::get<std::string>(value), nullptr, 0));
  } catch (const std::logic_error&) {
    return std::nullopt;
  }
}

#endif  // __SMI_SYSFS_H__
