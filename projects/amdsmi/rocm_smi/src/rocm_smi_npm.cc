/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "rocm_smi/rocm_smi_npm.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_logger.h"
#include <fstream>
#include <cstring>
#include <cerrno>
#include <iomanip>
#include <sstream>
#include <map>

using amd::smi::getRSMIStatusString;

namespace amd::smi {

namespace fs = std::filesystem;

rsmi_status_t read_npm_file(const fs::path &path, std::string &out) {
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

rsmi_status_t get_npm_board_status(const std::string &board_path, bool *enabled) {
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

rsmi_status_t get_npm_board_limit(const std::string &board_path, uint64_t *limit) {
  if (limit == nullptr) return RSMI_STATUS_INVALID_ARGS;
  if (board_path.empty()) return RSMI_STATUS_INVALID_ARGS;

  fs::path bd(board_path);
  if (!fs::exists(bd) || !fs::is_directory(bd)) return RSMI_STATUS_NOT_SUPPORTED;

  fs::path p = bd / "cur_node_power_limit";
  if (!fs::exists(p) || !fs::is_regular_file(p)) return RSMI_STATUS_NOT_SUPPORTED;

  std::string s;
  rsmi_status_t r = read_npm_file(p, s);
  if (r != RSMI_STATUS_SUCCESS) return RSMI_STATUS_NOT_SUPPORTED;

  try {
    size_t idx = 0;
    unsigned long long v = std::stoull(s, &idx, 10);
    if (idx != s.size()) return RSMI_STATUS_UNEXPECTED_DATA;
    *limit = static_cast<uint64_t>(v);
    return RSMI_STATUS_SUCCESS;
  } catch (...) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
}

}  // end namespace
