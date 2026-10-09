// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_npm.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"

using amd::smi::getRSMIStatusString;

namespace amd::smi {

namespace fs = std::filesystem;

rsmi_status_t read_npm_file(const fs::path& path, std::string& out) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return RSMI_STATUS_FILE_ERROR;
  }
  std::string line;
  if (!std::getline(ifs, line)) {
    return RSMI_STATUS_NO_DATA;
  }
  out = line;
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t get_npm_board_status(const std::string& board_path, bool* enabled) {
  if (enabled == nullptr) return RSMI_STATUS_INVALID_ARGS;
  if (board_path.empty()) return RSMI_STATUS_INVALID_ARGS;

  fs::path bd(board_path);
  if (!fs::exists(bd) || !fs::is_directory(bd)) return RSMI_STATUS_NOT_SUPPORTED;

  std::string s;
  rsmi_status_t r = read_npm_file(bd / "npm_status", s);
  if (r != RSMI_STATUS_SUCCESS) return RSMI_STATUS_NOT_SUPPORTED;

  if (s == "enabled") {
    *enabled = true;
    return RSMI_STATUS_SUCCESS;
  }
  if (s == "disabled") {
    *enabled = false;
    return RSMI_STATUS_SUCCESS;
  }
  return RSMI_STATUS_UNEXPECTED_DATA;
}

static rsmi_status_t read_board_uint64(const std::string& board_path, const char* filename,
                                       uint64_t* value) {
  if (value == nullptr) return RSMI_STATUS_INVALID_ARGS;
  if (board_path.empty()) return RSMI_STATUS_INVALID_ARGS;

  fs::path bd(board_path);
  if (!fs::exists(bd) || !fs::is_directory(bd)) return RSMI_STATUS_NOT_SUPPORTED;

  fs::path p = bd / filename;
  if (!fs::exists(p) || !fs::is_regular_file(p)) return RSMI_STATUS_NOT_SUPPORTED;

  std::string s;
  rsmi_status_t r = read_npm_file(p, s);
  if (r != RSMI_STATUS_SUCCESS) return RSMI_STATUS_NOT_SUPPORTED;

  // std::stoull() accepts a leading '-' and silently negates modulo 2^64
  // (e.g. "-1" successfully parses as UINT64_MAX with idx == s.size()),
  // which is well-defined C++ behavior but would let corrupted/negative
  // sysfs content masquerade as a huge-but-"valid" unsigned value instead of
  // failing to parse. Reject anything that doesn't start with a digit
  // before ever calling std::stoull() so negative/garbage content is always
  // treated as a parse error.
  // Behavior change: negative/malformed sysfs content previously parsed
  // successfully as UINT64_MAX; it now returns RSMI_STATUS_UNEXPECTED_DATA.
  if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0]))) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  try {
    size_t idx = 0;
    unsigned long long v = std::stoull(s, &idx, 10);
    if (idx != s.size()) return RSMI_STATUS_UNEXPECTED_DATA;
    *value = static_cast<uint64_t>(v);
    return RSMI_STATUS_SUCCESS;
  } catch (const std::invalid_argument&) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  } catch (const std::out_of_range&) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
}

rsmi_status_t get_npm_board_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "cur_node_power_limit", limit);
}

rsmi_status_t get_npm_board_max_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "max_node_power_limit", limit);
}

rsmi_status_t set_npm_board_limit(const std::string& board_path, uint64_t limit) {
  if (board_path.empty()) return RSMI_STATUS_INVALID_ARGS;

  fs::path bd(board_path);
  if (!fs::exists(bd) || !fs::is_directory(bd)) return RSMI_STATUS_NOT_SUPPORTED;

  fs::path p = bd / "cur_node_power_limit";
  if (!fs::exists(p) || !fs::is_regular_file(p)) return RSMI_STATUS_NOT_SUPPORTED;

  // Write the numeric limit value as text to the sysfs file. This mirrors the
  // WriteSysfsStr()-based pattern used elsewhere in rocm_smi for other
  // write-capable attributes (e.g. power cap set path), which in turn
  // triggers a Set NPM Limit request to GPU PMFW via the amdgpu driver.
  int ret = WriteSysfsStr(p.string(), std::to_string(limit));
  // If the sysfs file doesn't exist, treat as not supported.
  if (ret == ENOENT) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  return ErrnoToRsmiStatus(ret);
}

rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "baseboard_power_limit", limit);
}

rsmi_status_t get_npm_node_power(const std::string& board_path, uint64_t* power) {
  return read_board_uint64(board_path, "node_power", power);
}

}  // namespace amd::smi
