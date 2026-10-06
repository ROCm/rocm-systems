/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#pragma once

#include <cctype>
#include <cstddef>
#include <string>

// Splits a recorder log name "<base>.<pid>.<host><ext>" into pid and host. The host may contain dots.
inline bool ParseLogName(const std::string& name, const std::string& base, const std::string& ext, int* pid,
                         std::string* host) {
  const std::string prefix = base + ".";
  if (name.size() <= prefix.size() + ext.size() || name.compare(0, prefix.size(), prefix) != 0) {
    return false;
  }
  if (!ext.empty() && name.compare(name.size() - ext.size(), ext.size(), ext) != 0) {
    return false;
  }
  // replay_log_converter.py writes "<base>.<pid>.<host>.json" next to the binary logs; never read those as binary.
  const std::string json = ".json";
  if (ext.empty() && name.size() > json.size() && name.compare(name.size() - json.size(), json.size(), json) == 0) {
    return false;
  }
  size_t pos = prefix.size();
  long value = 0;
  while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
    value = value * 10 + (name[pos] - '0');
    if (value > 0x7fffffff) {
      return false;
    }
    pos++;
  }
  if (pos == prefix.size() || pos >= name.size() || name[pos] != '.') {
    return false;
  }
  const size_t hostBegin = pos + 1;
  const size_t hostEnd = name.size() - ext.size();
  if (hostEnd <= hostBegin) {
    return false;
  }
  *pid = static_cast<int>(value);
  *host = name.substr(hostBegin, hostEnd - hostBegin);
  return true;
}
