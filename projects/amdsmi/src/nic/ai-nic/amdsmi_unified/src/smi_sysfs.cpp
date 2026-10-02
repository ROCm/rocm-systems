// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_sysfs.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

#include "smi_nic_log.h"

namespace {

// sysfs attributes that hold a MAC address or a GUID derived from it; masked in the debug log.
constexpr const char* kMaskedAttrNames[] = {"address", "node_guid", "sys_image_guid"};

const char* status_name(SmiSysfsReader::SysfsStatus status) {
  switch (status) {
    case SmiSysfsReader::SysfsStatus::Success:
      return "SUCCESS";
    case SmiSysfsReader::SysfsStatus::FileNotFound:
      return "FAIL FileNotFound";
    case SmiSysfsReader::SysfsStatus::IOError:
      return "FAIL IOError";
    case SmiSysfsReader::SysfsStatus::ParseError:
      return "FAIL ParseError";
  }
  return "FAIL Unknown";
}

void log_read(const char* func, const std::string& path, SmiSysfsReader::SysfsStatus status,
              const std::string& content) {
  if (!amd::smi::nic::log::is_enabled()) {
    return;
  }
  std::string msg = "sysfs read " + path + " -> " + status_name(status);
  if (status == SmiSysfsReader::SysfsStatus::Success) {
    const auto name_pos = path.rfind('/');
    const auto attr_name = path.substr((name_pos == std::string::npos) ? 0 : (name_pos + 1));
    const bool is_masked =
        std::any_of(std::begin(kMaskedAttrNames), std::end(kMaskedAttrNames),
                    [&attr_name](const char* name) { return (attr_name == name); });
    msg += " value=" + (is_masked ? amd::smi::nic::log::mask_tail(content) : content);
  }
  amd::smi::nic::log::debug(func, msg);
}

}  // namespace

// `raw`, when non-null, receives the text as read (lines joined by a space), for the debug log.
static SmiSysfsReader::SysfsStatus read_all_impl(const std::string& filepath,
                                                 std::vector<SmiSysfsReader::SysfsValue>& content,
                                                 std::string* raw) {
  std::ifstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    return SmiSysfsReader::SysfsStatus::FileNotFound;
  }

  if (!SmiSysfsReader::is_readable(filepath)) {
    return SmiSysfsReader::SysfsStatus::IOError;
  }

  content.clear();
  while (std::getline(file, line)) {
    if (raw != nullptr) {
      *raw += (raw->empty() ? "" : " ") + line;
    }
    if (line.find(' ') != std::string::npos) {
      content.push_back(line);
      continue;
    }
    std::stringstream ss(line);
    std::string token;
    while (ss >> token) {
      try {
        if (token.find("0x") == 0 || token.find("0X") == 0) {
          int hex_value = std::stoi(token, nullptr, 16);
          content.emplace_back(std::in_place_type<int>, hex_value);
        } else if (std::all_of(token.begin(), token.end(), ::isdigit)) {
          content.emplace_back(std::in_place_type<int>, std::stoi(token));
        } else {
          content.push_back(token);
        }
      } catch (const std::invalid_argument&) {
        return SmiSysfsReader::SysfsStatus::ParseError;
      } catch (const std::out_of_range&) {
        return SmiSysfsReader::SysfsStatus::ParseError;
      }
    }
  }

  return SmiSysfsReader::SysfsStatus::Success;
}

// `raw`, when non-null, receives the line as read, for the debug log.
static SmiSysfsReader::SysfsStatus read_line_impl(const std::string& filepath,
                                                  SmiSysfsReader::SysfsValue& content,
                                                  std::string* raw) {
  std::ifstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    return SmiSysfsReader::SysfsStatus::FileNotFound;
  }

  if (!SmiSysfsReader::is_readable(filepath)) {
    return SmiSysfsReader::SysfsStatus::IOError;
  }

  if (std::getline(file, line)) {
    if (raw != nullptr) {
      *raw = line;
    }
    if (line.find(' ') != std::string::npos) {
      content = line;
      return SmiSysfsReader::SysfsStatus::Success;
    }
    std::stringstream ss(line);
    std::string token;
    if (ss >> token) {
      try {
        if (token.find("0x") == 0 || token.find("0X") == 0) {
          int hex_value = std::stoi(token, nullptr, 16);
          content = hex_value;
        } else if (std::all_of(token.begin(), token.end(), ::isdigit)) {
          content = std::stoi(token);
        } else {
          content = token;
        }
        return SmiSysfsReader::SysfsStatus::Success;
      } catch (...) {
        return SmiSysfsReader::SysfsStatus::ParseError;
      }
    }
  }

  /**
   * Reached only when the file was empty or the line held no token: nothing was
   * written to `content`, so reporting Success would hand the caller its default
   * (int{0}) as if it were a real reading. Signal that no value was available.
   */
  return SmiSysfsReader::SysfsStatus::IOError;
}

SmiSysfsReader::SysfsStatus SmiSysfsReader::readAll(const std::string& filepath,
                                                    std::vector<SysfsValue>& content) {
  std::string raw;
  const auto status =
      read_all_impl(filepath, content, amd::smi::nic::log::is_enabled() ? &raw : nullptr);
  log_read(__PRETTY_FUNCTION__, filepath, status, raw);
  return status;
}

SmiSysfsReader::SysfsStatus SmiSysfsReader::readLine(const std::string& filepath,
                                                     SysfsValue& content, bool is_success_logged) {
  std::string raw;
  const auto status =
      read_line_impl(filepath, content, amd::smi::nic::log::is_enabled() ? &raw : nullptr);
  if (is_success_logged || (status != SysfsStatus::Success)) {
    log_read(__PRETTY_FUNCTION__, filepath, status, raw);
  }
  return status;
}

bool SmiSysfsReader::is_readable(const std::string& filepath) {
  std::ifstream file(filepath);
  return file.good();
}
