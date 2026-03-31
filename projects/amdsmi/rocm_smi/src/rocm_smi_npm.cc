/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE file for full license text.
 */

#include "rocm_smi/rocm_smi_npm.h"

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

rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "baseboard_power_limit", limit);
}

}  // namespace amd::smi
