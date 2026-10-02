// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __SMI_SYSFS_H__
#define __SMI_SYSFS_H__

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

#endif  // __SMI_SYSFS_H__
