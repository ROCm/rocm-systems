// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_npm.h"

#include <fcntl.h>
#include <unistd.h>

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

rsmi_status_t get_npm_board_max_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "max_node_power_limit", limit);
}

rsmi_status_t validate_npm_board_limit(const std::string& board_path, uint64_t limit) {
  bool enabled = false;
  rsmi_status_t ret = get_npm_board_status(board_path, &enabled);
  if (ret != RSMI_STATUS_SUCCESS) return ret;
  if (!enabled) return RSMI_STATUS_INVALID_ARGS;

  uint64_t max_limit = 0;
  ret = get_npm_board_max_limit(board_path, &max_limit);
  if (ret != RSMI_STATUS_SUCCESS) return ret;
  // UINT64_MAX means the platform max is unavailable (e.g. "-1"), not an unbounded limit.
  if (max_limit == UINT64_MAX) return RSMI_STATUS_UNEXPECTED_DATA;

  if (limit == 0 || limit > max_limit) return RSMI_STATUS_INVALID_ARGS;
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t set_npm_board_limit(const std::string& board_path, uint64_t limit) {
  if (board_path.empty()) return RSMI_STATUS_INVALID_ARGS;

  fs::path bd(board_path);
  if (!fs::exists(bd) || !fs::is_directory(bd)) return RSMI_STATUS_NOT_SUPPORTED;

  fs::path p = bd / "cur_node_power_limit";
  if (!fs::exists(p) || !fs::is_regular_file(p)) return RSMI_STATUS_NOT_SUPPORTED;

  // Not WriteSysfsStr(): it reports every post-open failure as ENOENT, hiding the driver's EPERM.
  const std::string val = std::to_string(limit);
  int err = 0;
  int fd = open(p.c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC);
  if (fd < 0) {
    err = errno;
  } else {
    ssize_t n = write(fd, val.data(), val.size());
    if (n < 0) {
      err = errno;
    } else if (static_cast<size_t>(n) != val.size()) {
      err = EIO;
    }
    close(fd);
  }

  // EPERM here is the driver rejecting a non-permitted guest, not a missing feature.
  if (err == EPERM) return RSMI_STATUS_PERMISSION;
  return ErrnoToRsmiStatus(err);
}

rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit) {
  return read_board_uint64(board_path, "baseboard_power_limit", limit);
}

rsmi_status_t get_npm_node_power(const std::string& board_path, uint64_t* power) {
  return read_board_uint64(board_path, "node_power", power);
}

}  // namespace amd::smi
