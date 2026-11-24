// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_utils.h"

#include <dirent.h>
#include <fcntl.h>
#include <libdrm/amdgpu.h>
#include <libdrm/drm.h>
#include <poll.h>
#include <spawn.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/impl/amd_smi_clk_testing.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_gpu_mutex.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "amd_smi/impl/scoped_fd.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

std::string leftTrim(const std::string& s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("^\\s+"), "");
  }
  return s;
}

std::string rightTrim(const std::string& s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\\s+$"), "");
  }
  return s;
}

std::string removeNewLines(const std::string& s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\n+"), "");
  }
  return s;
}

std::string trim(const std::string& s) {
  if (!s.empty()) {
    // remove new lines -> trim white space at ends
    std::string noNewLines = removeNewLines(s);
    return leftTrim(rightTrim(noNewLines));
  }
  return s;
}

std::string_view trim(std::string_view str) {
  if (str.empty()) {
    return str;
  }

  auto first_itr = std::find_if_not(
      str.begin(), str.end(), [](unsigned char character) { return std::isspace(character); });
  if (first_itr == str.end()) {
    return {};
  }

  auto last_itr = std::find_if_not(str.rbegin(), str.rend(),
                                   [](unsigned char character) { return std::isspace(character); });

  return str.substr(static_cast<size_t>(first_itr - str.begin()),
                    static_cast<size_t>(last_itr.base() - first_itr));
}

// Given original string and string to remove (removeMe)
// Return will provide the resulting modified string with the removed string(s)
std::string removeString(const std::string origStr, const std::string& removeMe) {
  std::string modifiedStr = origStr;
  std::string::size_type l = removeMe.length();
  for (std::string::size_type i = modifiedStr.find(removeMe); i != std::string::npos;
       i = modifiedStr.find(removeMe)) {
    modifiedStr.erase(i, l);
  }
  return modifiedStr;
}

amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len,
                                                std::string newString) {
  char* begin = &buffer[0];
  char* end = &buffer[len];
  std::fill(begin, end, 0);

  // Safer approach - copy directly with length limit
  size_t copy_len = std::min(static_cast<size_t>(len - 1), newString.length());
  if (copy_len > 0) {
    std::memcpy(buffer, newString.c_str(), copy_len);
  }
  buffer[copy_len] = '\0';
  return AMDSMI_STATUS_SUCCESS;
}

int openFileAndModifyBuffer(std::string path, char* buff, size_t sizeOfBuff,
                            bool trim_whitespace = true) {
  bool errorDiscovered = false;
  std::ifstream file(path, std::ifstream::in);
  std::string contents = {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
  smi_clear_char_and_reinitialize(buff, static_cast<uint32_t>(sizeOfBuff), contents);
  if (!file.is_open()) {
    errorDiscovered = true;
  } else {
    if (trim_whitespace) {
      contents = amd::smi::trimAllWhiteSpace(contents);
    }
    // remove all new lines
    contents.erase(std::remove(contents.begin(), contents.end(), '\n'), contents.cend());
  }

  file.close();
  if (!errorDiscovered && file.good() && !file.bad() && !file.fail() && !file.eof() &&
      !contents.empty()) {
    std::strncpy(buff, contents.c_str(), sizeOfBuff - 1);
    buff[sizeOfBuff - 1] = '\0';
    return 0;
  } else {
    return -1;
  }
}

static const uint32_t kAmdGpuId = 0x1002;

static bool isAMDGPU(std::string dev_path) {
  std::string vend_path = dev_path + "/device/vendor";

  if (!amd::smi::FileExists(vend_path.c_str())) {
    return false;
  }

  std::ifstream fs;
  fs.open(vend_path);

  if (!fs.is_open()) {
    return false;
  }

  uint32_t vendor_id;

  fs >> std::hex >> vendor_id;

  fs.close();

  if (vendor_id == kAmdGpuId) {
    return true;
  }
  return false;
}

amdsmi_status_t smi_amdgpu_find_hwmon_dir(amd::smi::AMDSmiGPUDevice* device,
                                          std::string* full_path) {
  if (full_path == nullptr) {
    return AMDSMI_STATUS_API_FAILED;
  }
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  DIR* dh;
  struct dirent* contents;
  std::string device_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path();
  std::string directory_path = device_path + "/device/hwmon/";

  if (!isAMDGPU(device_path)) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  dh = opendir(directory_path.c_str());
  if (!dh) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  /*
     First directory is '.', second directory is '..' and third directory is
     valid directory for reading sysfs node
     */
  while ((contents = readdir(dh)) != NULL) {
    std::string name = contents->d_name;
    if (name.find("hwmon", 0) != std::string::npos) *full_path = directory_path + name;
  }

  closedir(dh);

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_board_info(amd::smi::AMDSmiGPUDevice* device,
                                          amdsmi_board_info_t* info) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string model_number_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" +
                                  device->get_gpu_path() + std::string("/device/product_number");
  std::string product_serial_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" +
                                    device->get_gpu_path() + std::string("/device/serial_number");
  std::string fru_id_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() +
                            std::string("/device/fru_id");
  std::string manufacturer_name_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" +
                                       device->get_gpu_path() + std::string("/device/manufacturer");
  std::string product_name_path = amd::smi::test_hooks::SysfsDrmRoot() + "/" +
                                  device->get_gpu_path() + std::string("/device/product_name");

  auto ret_mod =
      openFileAndModifyBuffer(model_number_path, info->model_number, AMDSMI_MAX_STRING_LENGTH);
  auto ret_ser =
      openFileAndModifyBuffer(product_serial_path, info->product_serial, AMDSMI_MAX_STRING_LENGTH);
  auto ret_fru = openFileAndModifyBuffer(fru_id_path, info->fru_id, AMDSMI_MAX_STRING_LENGTH);
  auto ret_man = openFileAndModifyBuffer(manufacturer_name_path, info->manufacturer_name,
                                         AMDSMI_MAX_STRING_LENGTH);
  auto ret_prod = openFileAndModifyBuffer(product_name_path, info->product_name,
                                          AMDSMI_MAX_STRING_LENGTH, false);

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "[Before correction] " << "Returning status = AMDSMI_STATUS_SUCCESS"
     << " | model_number_path = |" << model_number_path << "|\n"
     << "; info->model_number: |" << info->model_number << "|\n"
     << "; ret_mod = " << ret_mod << "|\n"
     << "\n product_serial_path = |" << product_serial_path << "|\n"
     << "; info->product_serial: |" << info->product_serial << "|\n"
     << "; ret_ser = " << ret_ser << "|\n"
     << "\n fru_id_path = |" << fru_id_path << "|\n"
     << "; info->fru_id: |" << info->fru_id << "|\n"
     << "; ret_fru = " << ret_fru << "|\n"
     << "\n manufacturer_name_path = |" << manufacturer_name_path << "|\n"
     << "; info->manufacturer_name: |" << info->manufacturer_name << "|\n"
     << "; ret_man = " << ret_man << "|\n"
     << "\n product_name_path = |" << product_name_path << "|\n"
     << "; info->product_name: |" << info->product_name << "|" << "; ret_prod = " << ret_prod
     << "|\n";
  LOG_INFO(ss);

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_power_cap(amd::smi::AMDSmiGPUDevice* device, uint32_t sensor_ind,
                                         int* cap) {
  constexpr int DATA_SIZE = 16;
  char val[DATA_SIZE];
  std::string fullpath;
  amdsmi_status_t ret = AMDSMI_STATUS_SUCCESS;

  ret = smi_amdgpu_find_hwmon_dir(device, &fullpath);

  SMIGPUDEVICE_MUTEX(device->get_mutex())

  if (ret) return ret;

  fullpath += "/power" + std::to_string(sensor_ind + 1) + "_cap";
  std::ifstream file(fullpath.c_str(), std::ifstream::in);
  if (!file.is_open()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  file.getline(val, DATA_SIZE);

  if (sscanf(val, "%d", cap) < 0) {
    return AMDSMI_STATUS_API_FAILED;
  }

  return AMDSMI_STATUS_SUCCESS;
}

bool smi_amdgpu_parse_od_clk_range(std::istream& od_stream, amdsmi_clk_type_t domain,
                                   unsigned int* max_freq, unsigned int* min_freq) {
  // Section header (and its GFXCLK/MCLK/FCLK alias) whose levels feed this
  // domain's user-defined range.
  const char* od_header = nullptr;
  const char* alias_header = nullptr;
  switch (domain) {
    case AMDSMI_CLK_TYPE_GFX:
      od_header = "OD_SCLK:";
      alias_header = "GFXCLK:";
      break;
    case AMDSMI_CLK_TYPE_MEM:
      od_header = "OD_MCLK:";
      alias_header = "MCLK:";
      break;
    case AMDSMI_CLK_TYPE_DF:
      od_header = "OD_FCLK:";
      alias_header = "FCLK:";
      break;
    default:
      return false;
  }

  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  bool in_domain = false;
  bool found = false;
  char str[10];
  unsigned int dpm_level, freq;
  for (std::string line; getline(od_stream, line);) {
    // Section headers end with ':'. This domain's header (or its alias) starts
    // capture; any other header ends it, so an adjacent section such as
    // OD_VDDC_CURVE is not folded into the range.
    if (!line.empty() && line.back() == ':') {
      in_domain = line.compare(od_header) == 0 || line.compare(alias_header) == 0;
      continue;
    }
    if (!in_domain) {
      continue;
    }
    if (sscanf(line.c_str(), "%u: %u%9s", &dpm_level, &freq, str) <= 2) {
      continue;  // skip lines that don't conform to the format
    }
    found = true;
    if (freq > max) max = freq;
    if (freq < min) min = freq;
  }

  if (!found || max == 0) {
    return false;
  }
  *max_freq = max;
  *min_freq = min;
  return true;
}

// Finish smi_amdgpu_get_ranges() from an already-open pp_dpm_* stream: fold the
// dpm levels (and the optional "S:" sleep line) into the SmiAmdgpuClkRanges
// output. Split out as a library-local test seam (see amd_smi_clk_testing.h) so
// the folding and the bounds guard can be exercised over in-memory streams.
//
// od_range carries the pp_od_clk_voltage range the caller parsed; when it is not
// present, min/max are derived from the dpm levels instead. A domain with no
// minimum level or no sleep state keeps its UINT_MAX "unavailable" sentinel,
// which callers surface as the unavailable marker; only genuinely out-of-range
// (> INT_MAX and not the sentinel) values are rejected.
amdsmi_status_t smi_amdgpu_parse_dpm_ranges(std::istream& dpm_stream,
                                            const SmiAmdgpuOdClkRange& od_range,
                                            SmiAmdgpuClkRanges& ranges) {
  unsigned int max = od_range.present ? od_range.max : 0;
  unsigned int min = od_range.present ? od_range.min : UINT_MAX;
  unsigned int dpm = 0;
  unsigned int sleep_freq = UINT_MAX;
  unsigned int current_freq = 0;
  char str[10];
  char single_char;
  for (std::string line; getline(dpm_stream, line);) {
    unsigned int dpm_level, freq;

    char firstChar = line[0];
    if (firstChar == 'S') {
      if (sscanf(line.c_str(), "%c: %u%9s", &single_char, &sleep_freq, str) <= 2) {
        return AMDSMI_STATUS_NO_DATA;
      }
    } else {
      /**
       * if the first line contains '*', then
       * we are saving that value as current_freq then checking
       * for other dpm levels if none are found then we
       * set min and max to current_freq as per Driver
       * We then skip to the next line to avoid getting
       * incorrect min value.
       */

      if (sscanf(line.c_str(), "%u: %u%c", &dpm_level, &freq, str) <= 2) {
        return AMDSMI_STATUS_IO;
      }

      char lastChar = line.back();
      if (lastChar == '*') {
        current_freq = freq;
      }

      // Domains without an OD range derive min/max from the dpm levels here.
      if (!od_range.present) {
        max = freq > max ? freq : max;
        min = freq < min ? freq : min;
      }
      dpm = dpm_level > dpm ? dpm_level : dpm;
    }
  }
  if (dpm == 0 && current_freq > 0) {
    // if the dpm level is 0, then the current frequency is the min/max frequency
    max = current_freq;
    min = current_freq;
  }
  // Reject genuinely out-of-range values, but let the UINT_MAX "unavailable"
  // sentinel through: a domain with no minimum level or no sleep state keeps it,
  // and callers (e.g. amdsmi_get_clock_info) surface it as the unavailable marker.
  if ((dpm != UINT_MAX && dpm > static_cast<unsigned int>(INT_MAX)) ||
      (max != UINT_MAX && max > static_cast<unsigned int>(INT_MAX)) ||
      (min != UINT_MAX && min > static_cast<unsigned int>(INT_MAX)) ||
      (sleep_freq != UINT_MAX && sleep_freq > static_cast<unsigned int>(INT_MAX))) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }
  ranges.max_freq = static_cast<int>(max);
  ranges.min_freq = static_cast<int>(min);
  ranges.num_dpm = static_cast<int>(dpm);
  ranges.sleep_state_freq = static_cast<int>(sleep_freq);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_ranges(amd::smi::AMDSmiGPUDevice* device, amdsmi_clk_type_t domain,
                                      int* max_freq, int* min_freq, int* num_dpm,
                                      int* sleep_state_freq) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string fullpath =
      amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() + "/device";

  std::string smclk_min_max_fullpath = "";

  bool use_od_range = false;
  switch (domain) {
    case AMDSMI_CLK_TYPE_GFX:
      smclk_min_max_fullpath = fullpath + "/pp_od_clk_voltage";
      fullpath += "/pp_dpm_sclk";
      use_od_range = true;
      break;
    case AMDSMI_CLK_TYPE_MEM:
      smclk_min_max_fullpath = fullpath + "/pp_od_clk_voltage";
      fullpath += "/pp_dpm_mclk";
      use_od_range = true;
      break;
    case AMDSMI_CLK_TYPE_VCLK0:
      fullpath += "/pp_dpm_vclk";
      break;
    case AMDSMI_CLK_TYPE_VCLK1:
      fullpath += "/pp_dpm_vclk1";
      break;
    case AMDSMI_CLK_TYPE_DCLK0:
      fullpath += "/pp_dpm_dclk";
      break;
    case AMDSMI_CLK_TYPE_DCLK1:
      fullpath += "/pp_dpm_dclk1";
      break;
    case AMDSMI_CLK_TYPE_SOC:
      fullpath += "/pp_dpm_socclk";
      break;
    case AMDSMI_CLK_TYPE_DF:
      smclk_min_max_fullpath = fullpath + "/pp_od_clk_voltage";
      fullpath += "/pp_dpm_fclk";
      use_od_range = true;
      break;
    default:
      return AMDSMI_STATUS_INVAL;
  }

  std::ifstream ranges(fullpath.c_str());
  if (ranges.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  SmiAmdgpuOdClkRange od_range;
  // GFX/MEM/DF expose a user-defined range in pp_od_clk_voltage; when it omits
  // this domain's section (e.g. no OD_FCLK on MI45x) fall back to the pp_dpm_*
  // levels parsed by smi_amdgpu_parse_dpm_ranges().
  if (use_od_range) {
    std::ifstream smclk_ranges(smclk_min_max_fullpath.c_str());
    od_range.present =
        smi_amdgpu_parse_od_clk_range(smclk_ranges, domain, &od_range.max, &od_range.min);
  }
  SmiAmdgpuClkRanges parsed;
  amdsmi_status_t status = smi_amdgpu_parse_dpm_ranges(ranges, od_range, parsed);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return status;
  }
  if (max_freq) *max_freq = parsed.max_freq;
  if (min_freq) *min_freq = parsed.min_freq;
  if (num_dpm) *num_dpm = parsed.num_dpm;
  if (sleep_state_freq) *sleep_state_freq = parsed.sleep_state_freq;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_enabled_blocks(amd::smi::AMDSmiGPUDevice* device,
                                              uint64_t* enabled_blocks) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string fullpath =
      amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() + "/device/ras/features";
  std::ifstream f(fullpath.c_str());
  std::string tmp_str;

  if (f.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  std::string line;
  getline(f, line);

  std::istringstream f1(line);

  f1 >> tmp_str;  // ignore
  f1 >> tmp_str;  // ignore
  f1 >> tmp_str;
  f.close();

  // strtoul() alone can't tell "parsed 0" from "parsed nothing" -- both leave
  // the result at 0 -- so confirm real hex digits were consumed via endptr. A
  // cleanly-parsed 0 is a legitimate "no blocks enabled" mask, not a failure.
  // ULONG_MAX stays a failure sentinel regardless: no real ASIC defines all 64
  // mask bits (see AMDSMI_GPU_BLOCK_LAST), so a full set of 1s means a garbled
  // read, not a plausible value.
  char* endptr = nullptr;
  unsigned long parsed = strtoul(tmp_str.c_str(), &endptr, 16);
  bool parse_failed = endptr == tmp_str.c_str() || *endptr != '\0';
  if (parse_failed || parsed == ULONG_MAX) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | " << fullpath << " line: \"" << line << "\""
       << "; parsed token: \"" << tmp_str << "\""
       << "; reason: " << (parse_failed ? "unparsable token" : "implausible all-bits-set value")
       << "; returning AMDSMI_STATUS_API_FAILED";
    LOG_DEBUG(ss);
    return AMDSMI_STATUS_API_FAILED;
  }

  *enabled_blocks = parsed;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_bad_page_info(amd::smi::AMDSmiGPUDevice* device, uint32_t* num_pages,
                                             amdsmi_retired_page_record_t* info) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string line;
  std::vector<std::string> badPagesVec;

  std::string fullpath = amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() +
                         std::string("/device/ras/gpu_vram_bad_pages");
  std::ifstream fs(fullpath.c_str());

  if (fs.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  while (std::getline(fs, line)) {
    badPagesVec.push_back(line);
  }

  if (badPagesVec.size() == 0) {
    *num_pages = 0;
    return AMDSMI_STATUS_SUCCESS;
  }
  // Remove any *trailing* empty (whitespace) lines
  while (badPagesVec.size() != 0 &&
         badPagesVec.back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    badPagesVec.pop_back();
  }

  *num_pages = static_cast<uint32_t>(badPagesVec.size());

  if (info == nullptr) {
    return AMDSMI_STATUS_SUCCESS;
  }

  char status_code;
  amdsmi_memory_page_status_t tmp_stat;
  std::string junk;

  for (uint32_t i = 0; i < *num_pages; ++i) {
    std::istringstream fs1(badPagesVec[i]);

    fs1 >> std::hex >> info[i].page_address;
    fs1 >> junk;
    fs1 >> std::hex >> info[i].page_size;
    fs1 >> junk;
    fs1 >> status_code;

    switch (status_code) {
      case 'P':
        tmp_stat = AMDSMI_MEM_PAGE_STATUS_PENDING;
        break;
      case 'F':
        tmp_stat = AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE;
        break;
      case 'R':
        tmp_stat = AMDSMI_MEM_PAGE_STATUS_RESERVED;
        break;
      default:
        return AMDSMI_STATUS_API_FAILED;
    }
    info[i].status = tmp_stat;
  }

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_bad_page_threshold(amd::smi::AMDSmiGPUDevice* device,
                                                  uint32_t* threshold) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())

  // TODO: Accessing the node requires root privileges, and its interface may need to be exposed in
  // another path
  uint32_t index = device->get_card_id();
  std::string fullpath =
      "/sys/kernel/debug/dri/" + std::to_string(index) + std::string("/ras/bad_page_cnt_threshold");
  std::ifstream fs(fullpath.c_str());

  if (fs.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  std::string line;
  getline(fs, line);
  if (sscanf(line.c_str(), "%d", threshold) < 0) {
    return AMDSMI_STATUS_API_FAILED;
  }

  fs.close();

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_validate_ras_eeprom(amd::smi::AMDSmiGPUDevice* device) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())

  // TODO: need to expose the corresponding interface to validate the checksum of ras eeprom table.
  // verify fail: return AMDSMI_STATUS_CORRUPTED_EEPROM
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t smi_amdgpu_get_driver_version(amd::smi::AMDSmiGPUDevice* device, int* length,
                                              char* version) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  size_t len;
  if (*length <= 0 || version == nullptr) {
    return AMDSMI_STATUS_INVAL;
  } else {
    len = static_cast<size_t>(*length);
  }

  std::string empty = "";
  std::strncpy(version, empty.c_str(), len - 1);
  openFileAndModifyBuffer("/sys/module/amdgpu/version", version, static_cast<size_t>(len));
  if (version[0] == '\0') {
    std::strncpy(version, "N/A", len - 1);
    version[len - 1] = '\0';
  }

  return status;
}

amdsmi_status_t smi_amdgpu_plx_temp_from_metrics(const amdsmi_gpu_metrics_t& metrics,
                                                 int64_t* temperature) {
  // A max-value reading is the not-applicable sentinel (e.g. an APU with no
  // PLX/VRSOC sensor); surface it as unsupported instead of a bogus reading.
  if (metrics.temperature_vrsoc == std::numeric_limits<uint16_t>::max()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  *temperature = metrics.temperature_vrsoc;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_pcie_speed_from_pcie_type(uint16_t pcie_type, uint32_t* pcie_speed) {
  switch (pcie_type) {
    case 1:
      *pcie_speed = 2500;
      break;
    case 2:
      *pcie_speed = 5000;
      break;
    case 3:
      *pcie_speed = 8000;
      break;
    case 4:
      *pcie_speed = 16000;
      break;
    case 5:
      *pcie_speed = 32000;
      break;
    case 6:
      *pcie_speed = 64000;
      break;
    default:
      return AMDSMI_STATUS_API_FAILED;
  }
  return AMDSMI_STATUS_SUCCESS;
}

bool smi_amdgpu_has_xgmi_host_link(amd::smi::AMDSmiGPUDevice* device) {
  uint64_t kfd_gpu_id = 0;
  const rsmi_status_t status = rsmi_dev_guid_get(device->get_gpu_id(), &kfd_gpu_id);
  const auto& kfd_nodes = amd::smi::RocmSMI::getInstance().kfd_node_map();
  const auto node = status == RSMI_STATUS_SUCCESS ? kfd_nodes.find(kfd_gpu_id) : kfd_nodes.end();
  if (node == kfd_nodes.end()) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | GPU " << device->get_gpu_id() << ": rsmi_dev_guid_get "
       << amd::smi::getRSMIStatusString(status, false);
    if (status == RSMI_STATUS_SUCCESS) ss << ", gpu_id " << kfd_gpu_id << " has no KFD node";
    ss << "; keeping the PCIe link values";
    LOG_INFO(ss);
    return false;
  }
  return node->second->numa_node_type() == amd::smi::IOLINK_TYPE_XGMI;
}

amdsmi_status_t smi_amdgpu_get_market_name_from_dev_id(amd::smi::AMDSmiGPUDevice* device,
                                                       char* market_name) {
  SMIGPUDEVICE_MUTEX(device->get_mutex())
  if (market_name == nullptr || device == nullptr) {
    return AMDSMI_STATUS_ARG_PTR_NULL;
  }
  // initialize the market_name to empty string
  std::string empty = "";
  std::strncpy(market_name, empty.c_str(), AMDSMI_MAX_STRING_LENGTH - 1);

  std::ostringstream ss;
  std::string render_name = device->get_gpu_path();
  std::string path = "/dev/dri/" + render_name;
  if (render_name.empty() || !device->has_libdrm()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  ScopedFD fd(path.c_str(), O_RDWR | O_CLOEXEC);
  if (!fd.valid()) {
    ss << __PRETTY_FUNCTION__ << " | Render Name: " << render_name << "; path: " << path
       << "; fd: " << (fd < 0 ? "less than 0" : std::to_string(fd)) << "\n"
       << "; Returning: " << smi_amdgpu_get_status_string(AMDSMI_STATUS_FILE_ERROR, false) << "\n";
    LOG_INFO(ss);
    return AMDSMI_STATUS_FILE_ERROR;
  }

  amd::smi::AMDSmiLibraryLoader libdrm_amdgpu_;
  amdsmi_status_t status = libdrm_amdgpu_.load(amd::smi::libdrm_amdgpu_sonames());
  if (status != AMDSMI_STATUS_SUCCESS) {
    libdrm_amdgpu_.unload();
    return status;
  }

  // Function pointer typedefs
  typedef int (*amdgpu_device_initialize_t)(int fd, uint32_t* major_version,
                                            uint32_t* minor_version,
                                            amdgpu_device_handle* device_handle);
  typedef int (*amdgpu_device_deinitialize_t)(amdgpu_device_handle device_handle);
  typedef const char* (*amdgpu_get_marketing_name_t)(amdgpu_device_handle device_handle);
  amdgpu_device_initialize_t amdgpu_device_initialize = nullptr;
  amdgpu_device_deinitialize_t amdgpu_device_deinitialize = nullptr;
  amdgpu_get_marketing_name_t amdgpu_get_marketing_name = nullptr;

  status = libdrm_amdgpu_.load_symbol(reinterpret_cast<void**>(&amdgpu_device_deinitialize),
                                      "amdgpu_device_deinitialize");
  if (status != AMDSMI_STATUS_SUCCESS) {
    libdrm_amdgpu_.unload();
    return status;
  }

  status = libdrm_amdgpu_.load_symbol(reinterpret_cast<void**>(&amdgpu_device_initialize),
                                      "amdgpu_device_initialize");
  if (status != AMDSMI_STATUS_SUCCESS) {
    libdrm_amdgpu_.unload();
    return status;
  }

  amdgpu_device_handle device_handle = nullptr;
  uint32_t major_version, minor_version;
  int ret = amdgpu_device_initialize(fd, &major_version, &minor_version, &device_handle);
  if (ret != 0) {
    amdgpu_device_deinitialize(device_handle);
    libdrm_amdgpu_.unload();
    return AMDSMI_STATUS_DRM_ERROR;
  }

  status = libdrm_amdgpu_.load_symbol(reinterpret_cast<void**>(&amdgpu_get_marketing_name),
                                      "amdgpu_get_marketing_name");
  if (status != AMDSMI_STATUS_SUCCESS) {
    amdgpu_device_deinitialize(device_handle);
    libdrm_amdgpu_.unload();
    return status;
  }

  // Get the marketing name using libdrm's API
  const char* name = amdgpu_get_marketing_name(device_handle);
  if (name != nullptr) {
    std::strncpy(market_name, name, AMDSMI_MAX_STRING_LENGTH - 1);
    market_name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    amdgpu_device_deinitialize(device_handle);
    libdrm_amdgpu_.unload();
    return AMDSMI_STATUS_SUCCESS;
  }

  amdgpu_device_deinitialize(device_handle);
  libdrm_amdgpu_.unload();
  ss << __PRETTY_FUNCTION__ << " | path: " << path << "\n"
     << " | fd: " << std::dec << fd << "\n"
     << " | Marketing Name: " << market_name << "\n"
     << " | Returning: " << smi_amdgpu_get_status_string(AMDSMI_STATUS_DRM_ERROR, false) << "\n";
  LOG_INFO(ss);
  return AMDSMI_STATUS_DRM_ERROR;
}

amdsmi_status_t smi_amdgpu_is_gpu_power_management_enabled(amd::smi::AMDSmiGPUDevice* device,
                                                           bool* enabled) {
  if (enabled == nullptr) {
    return AMDSMI_STATUS_API_FAILED;
  }

  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string fullpath = amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() +
                         std::string("/device/pp_features");
  std::ifstream fs(fullpath.c_str());

  if (fs.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  // ANY line must end with "enabled" and have space before it
  const std::regex regex(R"(.*\senabled$)");
  std::string line;
  while (std::getline(fs, line)) {
    // match the whole line against regex, not just substrings
    if (std::regex_match(line, regex)) {
      *enabled = true;
      return AMDSMI_STATUS_SUCCESS;
    }
  }
  *enabled = false;
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_vcn_busy_percent(amd::smi::AMDSmiGPUDevice* device,
                                                uint32_t* vcn_busy_percent) {
  if (vcn_busy_percent == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  SMIGPUDEVICE_MUTEX(device->get_mutex())
  std::string fullpath = amd::smi::test_hooks::SysfsDrmRoot() + "/" + device->get_gpu_path() +
                         "/device/vcn_busy_percent";

  std::ifstream fs(fullpath.c_str());
  if (fs.fail()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  std::string line;
  if (std::getline(fs, line)) {
    try {
      uint32_t line_value = static_cast<uint32_t>(std::stoul(std::string(trim(line))));
      if (line_value > 100) {
        // max of uint32_t is used to indicate the erroneous value
        *vcn_busy_percent = std::numeric_limits<uint32_t>::max();
        return AMDSMI_STATUS_UNEXPECTED_DATA;
      }
      *vcn_busy_percent = line_value;
      return AMDSMI_STATUS_SUCCESS;
    } catch (const std::exception&) {
      *vcn_busy_percent = std::numeric_limits<uint32_t>::max();
      return AMDSMI_STATUS_UNEXPECTED_DATA;
    }
  }

  *vcn_busy_percent = std::numeric_limits<uint32_t>::max();
  return AMDSMI_STATUS_UNEXPECTED_DATA;
}

std::string smi_amdgpu_split_string(std::string str, char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  if (str.empty()) {
    return "";
  }

  while (std::getline(ss, token, delim)) {
    tokens.push_back(token);
    return token;  // return 1st match
  }
  return "";
}

// Split string at delimiter and return strings in vector
std::vector<std::string> split_string(const std::string& line, char delim) {
  std::vector<std::string> out;
  std::size_t start = 0;

  while (start < line.size()) {
    auto pos = line.find(delim, start);
    if (pos == std::string::npos) {
      pos = line.size();
    }
    std::string token = trim(line.substr(start, pos - start));
    if (!token.empty()) {
      out.push_back(token);
    }
    start = pos + 1;
  }
  return out;
}

// wrapper to return string expression of a rsmi_status_t return
// rsmi_status_t ret - return value of RSMI API function
// bool fullStatus - defaults to true, set to false to chop off description
// Returns:
// string - if fullStatus == true, returns full description of return value
//      ex. 'RSMI_STATUS_SUCCESS: The function has been executed successfully.'
// string - if fullStatus == false, returns a minimalized return value
//      ex. 'RSMI_STATUS_SUCCESS'
std::string smi_amdgpu_get_status_string(amdsmi_status_t ret, bool fullStatus = true) {
  const char* err_str;
  amdsmi_status_code_to_string(ret, &err_str);
  if (!fullStatus) {
    return smi_amdgpu_split_string(std::string(err_str), ':');
  }
  return std::string(err_str);
}

std::string smi_read_sysfs_string(const std::string& folder, const std::string& file_name) {
  std::stringstream temp;
  std::string file_path = folder + "/" + file_name;
  std::ifstream file(file_path.c_str(), std::ifstream::in);
  if (!file.is_open()) {
    return "N/A";
  } else {
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        break;
      }
      temp << line;
    }
  }

  return temp.str();
}

// TODO(amdsmi_team): Do we want to include these functions in header?
amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t* device_index) {
  uint32_t socket_count;
  std::vector<amdsmi_socket_handle> sockets;
  std::ostringstream ss;

  if (device_index == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  *device_index = std::numeric_limits<uint32_t>::max();  // set to max value for invalid readings

  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  // allocate memory
  sockets.resize(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t current_device_index = 0;
  for (uint32_t i = 0; i < socket_count; i++) {
    // Get Socket info
    char socket_info[128];
    ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
    ss << __PRETTY_FUNCTION__ << " | Socket " << socket_info << "\n";
    LOG_DEBUG(ss);

    // Get the device count available for the socket.
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, processor_handles.data());
    ss << __PRETTY_FUNCTION__ << " | Processor Count: " << device_count << "\n";
    LOG_DEBUG(ss);

    for (uint32_t j = 0; j < device_count; j++) {
      if (processor_handles[j] == processor_handle) {
        *device_index = current_device_index;
        ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_SUCCESS "
           << "Returning device_index: " << *device_index << "\nSocket #: " << i
           << "; Device #: " << j << "; current_device_index #: " << current_device_index << "\n";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_SUCCESS;
      }
      current_device_index++;
    }
  }
  ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_API_FAILED "
     << "Returning device_index: " << *device_index << "\n";
  LOG_DEBUG(ss);
  return AMDSMI_STATUS_API_FAILED;
}

// TODO(amdsmi_team): Do we want to include these functions in header?
amdsmi_status_t smi_amdgpu_get_device_count(uint32_t* total_num_devices) {
  uint32_t socket_count;
  std::vector<amdsmi_socket_handle> sockets;
  std::ostringstream ss;

  if (total_num_devices == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  // set to max value for invalid readings
  *total_num_devices = std::numeric_limits<uint32_t>::max();

  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  // allocate memory
  sockets.resize(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t device_num = 0;
  for (uint32_t i = 0; i < socket_count; i++) {
    // Get Socket info
    char socket_info[128];
    ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
    ss << __PRETTY_FUNCTION__ << " | Socket " << socket_info << "\n";
    LOG_DEBUG(ss);

    // Get the processor count available for the socket.
    uint32_t processor_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &processor_count, nullptr);

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(processor_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i], &processor_count, processor_handles.data());
    ss << __PRETTY_FUNCTION__ << " | Processor Count: " << processor_count << "\n";
    LOG_DEBUG(ss);

    for (uint32_t j = 0; j < processor_count; j++) {
      device_num++;
    }
  }
  *total_num_devices = device_num;
  ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_SUCCESS "
     << "Returning device_index: " << *total_num_devices << "\n";
  LOG_DEBUG(ss);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_ainic_processor_handle_by_index(
    uint32_t device_index, amdsmi_processor_handle* processor_handle) {
  if (!processor_handle) {
    return AMDSMI_STATUS_INVAL;
  }
  for (const auto& socket : amd::smi::AMDSmiSystem::getInstance().get_sockets()) {
    uint32_t idx = 0;
    for (const auto& processor : socket->get_processors(AMDSMI_PROCESSOR_TYPE_AMD_NIC)) {
      if (device_index == idx) {
        *processor_handle = processor;
        return AMDSMI_STATUS_SUCCESS;
      }
      idx++;
    }
  }
  return AMDSMI_STATUS_API_FAILED;
}

// TODO(amdsmi_team): Do we want to include these functions in header?
amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(
    uint32_t device_index, amdsmi_processor_handle* processor_handle) {
  uint32_t socket_count;
  std::vector<amdsmi_socket_handle> sockets;
  std::ostringstream ss;

  if (processor_handle == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  // allocate memory
  sockets.resize(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t current_device_index = 0;
  for (uint32_t i = 0; i < socket_count; i++) {
    // Get Socket info
    char socket_info[128];
    ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
    ss << __PRETTY_FUNCTION__ << " | Socket " << socket_info << "\n";
    LOG_DEBUG(ss);

    // Get the device count available for the socket.
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, processor_handles.data());
    ss << __PRETTY_FUNCTION__ << " | Processor Count: " << device_count << "\n";
    LOG_DEBUG(ss);

    for (uint32_t j = 0; j < device_count; j++) {
      if (current_device_index == device_index) {
        *processor_handle = processor_handles[j];
        ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_SUCCESS"
           << "\nReturning processor_handle for device_index: " << device_index
           << "\nSocket #: " << i << "; Device #: " << j
           << "; current_device_index #: " << current_device_index
           << "; processor_handle: " << *processor_handle
           << "; processor_handles[j]: " << processor_handles[j] << "\n";
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_SUCCESS;
      }
      current_device_index++;
    }
  }
  ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_API_FAILED "
     << "Could not find matching processor_handle for device_index: " << device_index << "\n";
  LOG_DEBUG(ss);
  return AMDSMI_STATUS_API_FAILED;
}

amdsmi_status_t get_gpu_device_from_handle(amdsmi_processor_handle processor_handle,
                                           amd::smi::AMDSmiGPUDevice** gpudevice) {
  AMDSMI_CHECK_INIT();
  std::ostringstream ss;
  if (processor_handle == nullptr || gpudevice == nullptr) {
    ss << __PRETTY_FUNCTION__ << " | processor_handle is NULL; returning: AMDSMI_STATUS_INVAL";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_INVAL;
  }
  amd::smi::AMDSmiProcessor* device = nullptr;
  amdsmi_status_t r =
      amd::smi::AMDSmiSystem::getInstance().handle_to_processor(processor_handle, &device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
    *gpudevice = static_cast<amd::smi::AMDSmiGPUDevice*>(device);
    return AMDSMI_STATUS_SUCCESS;
  }
  ss << __PRETTY_FUNCTION__ << " | returning AMDSMI_STATUS_NOT_SUPPORTED";
  LOG_ERROR(ss);
  return AMDSMI_STATUS_NOT_SUPPORTED;
}

int read_env_ms(const char* name, int def) {
  if (const char* s = std::getenv(name)) {
    try {
      return std::max(0, std::stoi(s));
    } catch (...) {
      // Ignore error, fallback to passed in def
    }
  }
  return def;
}

uint64_t get_product_serial_number(amdsmi_processor_handle processor_handle) {
  uint64_t serial_number = 0;
  amdsmi_board_info_t board_info = {};
  amdsmi_status_t status = amdsmi_get_gpu_board_info(processor_handle, &board_info);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << "Failed to retrieve product serial number! error: " << static_cast<int>(status);
    LOG_DEBUG(ss);
    return serial_number;
  }
  if (!*board_info.product_serial) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << " Product serial string is empty.";
    LOG_DEBUG(ss);
    return serial_number;
  }
  try {
    serial_number = std::stoull(board_info.product_serial, nullptr, 10);
  } catch (const std::invalid_argument& e) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << " Invalid product serial string. Exception: " << e.what();
    LOG_DEBUG(ss);
    serial_number = 0;
  } catch (const std::out_of_range& e) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
       << " Product serial out of range, Exception: " << e.what();
    LOG_DEBUG(ss);
    serial_number = 0;
  }
  return serial_number;
}

/**
 *  Important points to be pay attention to:
 *      - BDF is a struct in AMDSMI (amdsmi_bdf_t), and a char* in HIP (hipDeviceGetPCIBusId())
 *      - To convert from BDF to string, use: AMDSmiGPUDevice::bdf_to_string()
 *      - For HIP, UUID seems to be the best approach to identify the device, use:
 *        amdsmi_get_processor_handle_from_uuid()

 */

/**
 * Returns a pointer to the raw 16 bytes of the UUID.
 *
 *  Note:
 *      - This is NOT a null-terminated C string. The pointer refers to exactly
 *        HIP_UUID_BYTES_SIZE (16) bytes
 *      - Do not use with strlen(), printf("%s", ...), or APIs that expect a null-terminated string
 */
const char* from_uuid_to_cstring(const hipUUID_t& uuid) noexcept { return uuid.bytes; }

std::optional<amdsmi_bdf_t> from_cstring_to_bdf(const char* bdf_str) noexcept {
  if (!bdf_str) {
    return std::nullopt;
  }

  using uchar_t = unsigned char;
  constexpr auto HEX_BASE = std::int32_t(16);
  auto bdf = amdsmi_bdf_t{};
  bdf.as_uint = 0;

  const auto* ptr_str_bdf = bdf_str;
  /* Try parsing the domain (optional) */
  auto domain = std::uint64_t(0);
  auto [ptr_domain, error_code_domain] =
      std::from_chars(ptr_str_bdf, std::strchr(ptr_str_bdf, '\0'), domain, HEX_BASE);
  if ((error_code_domain == std::errc{}) && (ptr_domain != ptr_str_bdf) && (*ptr_domain == ':')) {
    bdf.bdf.domain_number = static_cast<std::uint64_t>((domain) & ((1ULL << 48) - 1));
    ptr_str_bdf = (ptr_domain + 1); /* If ':' is present, skip it */
  } else {
    ptr_str_bdf = bdf_str;
    bdf.bdf.domain_number = 0;
  }

  /* Try parsing the bus */
  auto bus = std::uint64_t(0);
  auto [ptr_bus, error_code_bus] =
      std::from_chars(ptr_str_bdf, std::strchr(ptr_str_bdf, '\0'), bus, HEX_BASE);
  /* If the bus is not valid (including only 8 bits) return nullopt */
  if (((error_code_bus != std::errc{}) || (ptr_bus == ptr_str_bdf) || (*ptr_bus != ':')) ||
      (bus > std::uint8_t(0xFF))) {
    return std::nullopt;
  }
  bdf.bdf.bus_number = static_cast<std::uint8_t>(bus);
  ptr_str_bdf = (ptr_bus + 1); /* If ':' is present, skip it */

  /* Try parsing the device */
  auto device = std::uint64_t(0);
  auto [ptr_device, error_code_device] =
      std::from_chars(ptr_str_bdf, std::strchr(ptr_str_bdf, '\0'), device, HEX_BASE);
  /* If the device is not valid (including only 5 bits) return nullopt */
  if (((error_code_device != std::errc{}) || (ptr_device == ptr_str_bdf) || (*ptr_device != '.')) ||
      (device > std::uint8_t(0x1F))) {
    return std::nullopt;
  }
  bdf.bdf.device_number = static_cast<std::uint8_t>((device) & ((1ULL << 5) - 1));
  ptr_str_bdf = (ptr_device + 1); /* If '.' is present, skip it */

  /* Try parsing the function */
  auto function = std::uint64_t(0);
  auto [ptr_function, error_code_function] =
      std::from_chars(ptr_str_bdf, std::strchr(ptr_str_bdf, '\0'), function, HEX_BASE);
  /* If the function is not valid (including only 3 bits) return nullopt */
  if (((error_code_function != std::errc{}) || (ptr_function == ptr_str_bdf)) ||
      (function > std::uint8_t(0x7))) {
    return std::nullopt;
  }
  bdf.bdf.function_number = static_cast<std::uint8_t>((function) & ((1ULL << 3) - 1));
  ptr_str_bdf = ptr_function;

  /* Allow trailing whitespace or nothing, but nothing else (optional) */
  while (*ptr_str_bdf != '\0') {
    /* Anything after the function is garbage */
    if (!std::isspace(static_cast<uchar_t>(*ptr_str_bdf))) {
      return std::nullopt;
    }
    ++ptr_str_bdf;
  }

  return bdf;
}

std::optional<hipUUID_t> from_cstring_to_uuid(const char* uuid_str) noexcept {
  if (!uuid_str) {
    return std::nullopt;
  }

  using uchar_t = unsigned char;
  auto hip_uuid = hipUUID_t{};
  auto char_pos = size_t(0);
  auto half_byte = std::size_t(0);

  /*
   *  So when we count half_bytes (nibbles; each valid hex character = one nibble),
   *  we expect exactly 32 of them after skipping all formatting characters (-, {, }, spaces, ...)
   */
  while ((uuid_str[char_pos] != '\0') && (half_byte < HIP_UUID_STRING_FULL_SIZE)) {
    auto character = char(uuid_str[char_pos++]);
    if (character == '-' || character == '{' || character == '}' ||
        std::isspace(static_cast<uchar_t>(character))) {
      continue;
    }

    if (!std::isxdigit(static_cast<uchar_t>(character))) {
      return std::nullopt;
    }

    auto character_value = static_cast<std::uint8_t>(
        (character >= '0' && character <= '9')   ? (character - '0')
        : (character >= 'a' && character <= 'f') ? (10 + (character - 'a'))
                                                 : (10 + (character - 'A')));

    /*
     *  Each half_byte corresponds to one byte in the UUID.
     *  - If the half_byte is even, we are processing the high nibble of the byte
     *  - If the half_byte is odd, we are processing the low nibble of the byte
     */
    constexpr auto HALF_BYTE_IN_BITS = static_cast<std::size_t>(CHAR_BIT / 2);

    /*
     *  (half_byte / 2) is the index of the byte in the UUID
     *  ((half_byte % 2) == 0) is where check for even/odd so high nibble or low nibble
     */
    auto* byte = reinterpret_cast<uchar_t*>(&hip_uuid.bytes[(half_byte / 2)]);
    if ((half_byte % 2) == 0) {
      *byte = static_cast<uchar_t>(character_value << HALF_BYTE_IN_BITS);
    } else {
      *byte |= character_value;
    }

    half_byte++;
  }

  return (half_byte == HIP_UUID_STRING_FULL_SIZE) ? std::optional<hipUUID_t>{hip_uuid}
                                                  : std::nullopt;
}

std::string stringify_bdf(const amdsmi_bdf_t& bdf) {
  std::ostringstream bdf_outstream;
  bdf_outstream << std::setfill('0') << std::hex << std::setw(4) << bdf.domain_number << ":"
                << std::setw(2) << static_cast<uint32_t>(bdf.bus_number) << ":" << std::setw(2)
                << static_cast<uint32_t>(bdf.device_number) << "."
                << static_cast<uint32_t>(bdf.function_number);
  return bdf_outstream.str();
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> parse_bdfid(uint64_t bdfid) {
  uint64_t domain = (bdfid >> 32) & 0xffffffff;
  uint64_t bus = (bdfid >> 8) & 0xff;
  uint64_t device_id = (bdfid >> 3) & 0x1f;
  uint64_t function = bdfid & 0x7;
  return std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>(domain, bus, device_id, function);
}

amdsmi_status_t smi_amdgpu_read_clk_freq_from_pp_dpm(amd::smi::AMDSmiGPUDevice* device,
                                                     const char* pp_dpm_file,
                                                     amdsmi_frequencies_t* f) {
  if (f == nullptr || device == nullptr || pp_dpm_file == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  uint32_t drm_render = device->get_drm_render_minor();
  std::string sysfs_path = amd::smi::test_hooks::SysfsDrmRoot() + "/renderD" +
                           std::to_string(drm_render) + "/device/" + pp_dpm_file;

  std::ifstream file(sysfs_path);
  if (!file.good()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  f->num_supported = 0;
  f->current = 0;
  f->has_deep_sleep = 0;

  std::string line;
  uint32_t level_index = 0;

  while (std::getline(file, line) && level_index < AMDSMI_MAX_NUM_FREQUENCIES) {
    // Parse line format: "0: 200Mhz" or "1: 400Mhz *"
    size_t colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }

    std::string freq_str = line.substr(colon_pos + 1);

    // Check if this is the current level (marked with *)
    bool is_current = (freq_str.find('*') != std::string::npos);
    if (is_current) {
      f->current = level_index;
    }

    // Remove asterisk and surrounding whitespace
    freq_str.erase(std::remove(freq_str.begin(), freq_str.end(), '*'), freq_str.end());
    freq_str.erase(0, freq_str.find_first_not_of(" \t"));
    size_t end = freq_str.find_last_not_of(" \t");
    if (end != std::string::npos) {
      freq_str.erase(end + 1);
    }

    // Parse "200Mhz" / "200 Mhz" / "200MHz"
    uint64_t freq_value = 0;
    char unit = 'M';  // Default to MHz

    size_t unit_pos = freq_str.find_first_not_of("0123456789 ");
    if (unit_pos != std::string::npos) {
      std::string value_str = freq_str.substr(0, unit_pos);
      value_str.erase(std::remove(value_str.begin(), value_str.end(), ' '), value_str.end());
      try {
        freq_value = std::stoull(value_str);
      } catch (...) {
        continue;  // Skip invalid lines
      }
      std::string unit_str = freq_str.substr(unit_pos);
      if (!unit_str.empty()) {
        unit = static_cast<char>(std::toupper(static_cast<unsigned char>(unit_str[0])));
      }
    }

    f->frequency[level_index] = freq_value * amd::smi::get_multiplier_from_char(unit);
    level_index++;
  }

  f->num_supported = level_index;
  return (f->num_supported > 0) ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NOT_SUPPORTED;
}

const char* smi_amdgpu_pp_dpm_filename_for_clk_type(amdsmi_clk_type_t clk_type) {
  switch (clk_type) {
    case AMDSMI_CLK_TYPE_VCLK0:
      return "pp_dpm_vclk";
    case AMDSMI_CLK_TYPE_VCLK1:
      return "pp_dpm_vclk1";
    case AMDSMI_CLK_TYPE_DCLK0:
      return "pp_dpm_dclk";
    case AMDSMI_CLK_TYPE_DCLK1:
      return "pp_dpm_dclk1";
    default:
      return nullptr;
  }
}

amdsmi_status_t smi_amdgpu_is_apu(uint64_t ids_flags, bool* is_apu) {
  if (is_apu == nullptr) return AMDSMI_STATUS_INVAL;
  *is_apu = (ids_flags & AMDGPU_IDS_FLAGS_FUSION) != 0;
  return AMDSMI_STATUS_SUCCESS;
}

void init_asic_info_defaults(amdsmi_asic_info_t* info) {
  if (info == nullptr) {
    return;
  }
  std::memset(info, 0, sizeof(*info));
  info->vendor_id = std::numeric_limits<uint32_t>::max();
  info->subvendor_id = std::numeric_limits<uint32_t>::max();
  info->device_id = std::numeric_limits<uint64_t>::max();
  info->rev_id = std::numeric_limits<uint32_t>::max();
  std::snprintf(info->asic_serial, AMDSMI_MAX_STRING_LENGTH, "ffffffffffffffff");
  info->oam_id = std::numeric_limits<uint32_t>::max();
  info->num_of_compute_units = std::numeric_limits<uint32_t>::max();
  info->target_graphics_version = std::numeric_limits<uint64_t>::max();
  info->subsystem_id = std::numeric_limits<uint32_t>::max();
  info->physical_acc_id = std::numeric_limits<uint32_t>::max();
  info->chip_rev_id = std::numeric_limits<uint32_t>::max();
  info->external_rev_id = std::numeric_limits<uint32_t>::max();
}

auto smi_amdgpu_format_driver_date(std::string_view drm_date) -> std::string {
  if (drm_date == "0") {
    return {};
  }
  if (drm_date.size() == 8) {
    return std::string{drm_date.substr(0, 4)} + "/" + std::string{drm_date.substr(4, 2)} + "/" +
           std::string{drm_date.substr(6, 2)} + " 00:00";
  }
  return std::string{drm_date};
}

namespace {

namespace fs = std::filesystem;

// DKMS registers amdgpu under /var/lib/dkms/amdgpu/<version>/; package
// metadata lives in source/dkms.conf behind the source symlink
constexpr std::string_view kAmdgpuDkmsPackageName = "amdgpu";
constexpr std::string_view kDkmsSourceSymlinkName = "source";
constexpr std::string_view kDkmsKernelPrefix = "kernel-";
constexpr std::string_view kDkmsConfName = "dkms.conf";
constexpr std::string_view kDkmsConfPackageName = "PACKAGE_NAME";
constexpr std::string_view kDkmsConfPackageVersion = "PACKAGE_VERSION";
constexpr std::string_view kAmdgpuKmodPackagePrefix = "amdgpu-kmod-";
// The file an rpm database keeps its packages in: Berkeley DB, ndb, sqlite.
constexpr std::array<std::string_view, 3> kRpmDatabaseFiles = {"Packages", "Packages.db",
                                                               "rpmdb.sqlite"};
constexpr size_t kRpmQueryMaxOutput = 4096;
// A dkms.conf is a few lines; the cap only stops a runaway regular file.
constexpr size_t kDkmsConfMaxSize = (1024U * 1024U);
constexpr size_t kDkmsConfChunkSize = 4096;
constexpr int kNoPidFd = -1;
// Pause between checks for rpm's exit. It starts short: rpm exits right after its output ends.
constexpr auto kRpmExitPollFirst = std::chrono::microseconds{100};
constexpr auto kRpmExitPollMax = std::chrono::microseconds{10000};

struct AmdgpuPackageVersion {
  std::string kernel_version;  // A.B.C
  std::string version;         // A.B.C[.D]-<build>
  std::string build;
};

struct AmdgpuModuleVersion {
  std::string kernel_version;  // A.B.C
  std::string amdgpu_version;  // D, empty for A.B.C
};

using DkmsConf = std::pair<std::string, std::string>;

auto log_driver_package_debug(std::string_view caller, std::string_view message) -> void {
  auto outstream = std::ostringstream{};
  outstream << caller << " | " << message;
  LOG_DEBUG(outstream);
}

// A DKMS PACKAGE_VERSION such as 6.19.14-2370381.24.04 (Ubuntu appends the OS release, other
// distros may stop at the build), or an rpm VERSION-RELEASE such as 6.16.13.000000-2412345.el9.
auto parse_package_version(std::string_view package_version)
    -> std::optional<AmdgpuPackageVersion> {
  static const auto regex = std::regex{R"(^(\d+\.\d+\.\d+)((?:\.\d+)*)-(\d+)(?:\..*)?$)"};

  // Longer text cannot be a version, and std::regex recurses once per character.
  if (package_version.size() >= AMDSMI_MAX_STRING_LENGTH) {
    return std::nullopt;
  }
  const auto text = std::string{package_version};
  auto match = std::smatch{};
  if (!std::regex_match(text, match, regex)) {
    return std::nullopt;
  }
  return AmdgpuPackageVersion{
      match[1].str(), match[1].str() + match[2].str() + "-" + match[3].str(), match[3].str()};
}

// amdgpu-kmod modules report A.B.C; DKMS modules report A.B.C.D.
auto parse_module_version(std::string_view module_version) -> std::optional<AmdgpuModuleVersion> {
  static const auto regex = std::regex{R"(^(\d+\.\d+\.\d+)(?:\.(\d+))?$)"};

  if (module_version.size() >= AMDSMI_MAX_STRING_LENGTH) {
    return std::nullopt;
  }
  const auto text = std::string{module_version};
  auto match = std::smatch{};
  if (!std::regex_match(text, match, regex)) {
    return std::nullopt;
  }
  return AmdgpuModuleVersion{match[1].str(), match[2].str()};
}

// Nothing checks that the module was installed from the package, only that both have the same
// kernel version.
auto is_package_for_module(const std::optional<AmdgpuPackageVersion>& package,
                           const std::optional<AmdgpuModuleVersion>& module) -> bool {
  return package.has_value() && module.has_value() &&
         (package->kernel_version == module->kernel_version);
}

// The one version in `versions` that has the kernel version of `module`. Empty when there is
// none, and when there are several: which of them is loaded is then not known.
auto only_version_of_module(const std::vector<std::string>& versions,
                            const std::optional<AmdgpuModuleVersion>& module) -> std::string {
  auto found = std::string{};
  for (const auto& version : versions) {
    if ((!is_package_for_module(parse_package_version(version), module)) || (version == found)) {
      continue;
    }
    if (!found.empty()) {
      return {};
    }
    found = version;
  }
  return found;
}

auto try_parse_dkms_conf_assignment(std::string_view line, std::string_view key)
    -> std::optional<std::string_view> {
  const auto trimmed_line = trim(line);
  if ((trimmed_line.size() <= (key.size() + 1)) ||
      (trimmed_line.compare(0, key.size(), key) != 0)) {
    return std::nullopt;
  }

  if (trimmed_line[key.size()] != '=') {
    return std::nullopt;
  }

  const auto value_part = trim(trimmed_line.substr((key.size() + 1)));
  if ((value_part.size() < 2) || (value_part.front() != '"') || (value_part.back() != '"')) {
    return std::nullopt;
  }

  return value_part.substr(1, (value_part.size() - 2));
}

auto read_to_end(int fd) -> std::optional<std::string> {
  auto contents = std::string{};
  auto chunk = std::array<char, kDkmsConfChunkSize>{};
  while (true) {
    const auto bytes = read(fd, chunk.data(), chunk.size());
    if (bytes == 0) {
      return contents;
    }
    if (bytes > 0) {
      contents.append(chunk.data(), static_cast<size_t>(bytes));
    } else if (errno != EINTR) {
      return std::nullopt;
    }
  }
}

// O_NONBLOCK keeps a FIFO from blocking the open; the type is checked on the opened descriptor.
auto read_small_regular_file(const fs::path& path) -> std::optional<std::string> {
  const auto fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    return std::nullopt;
  }

  struct stat file_stat = {};
  const auto is_small_file = (fstat(fd, &file_stat) == 0) && S_ISREG(file_stat.st_mode) &&
                             (static_cast<size_t>(file_stat.st_size) <= kDkmsConfMaxSize);
  auto contents = std::optional<std::string>{};
  if (is_small_file) {
    contents = read_to_end(fd);
  }
  close(fd);
  return contents;
}

auto read_dkms_conf(std::string_view caller, const fs::path& conf_path) -> std::optional<DkmsConf> {
  const auto conf_text = read_small_regular_file(conf_path);
  if (!conf_text.has_value()) {
    log_driver_package_debug(caller, "Cannot read " + conf_path.string());
    return std::nullopt;
  }
  auto instream = std::istringstream{*conf_text};

  auto package_name = std::string{};
  auto package_version = std::string{};
  auto line_str = std::string{};
  while (std::getline(instream, line_str)) {
    if ((!line_str.empty()) && (line_str.back() == '\r')) {
      line_str.pop_back();
    }

    if (const auto parsed_name = try_parse_dkms_conf_assignment(line_str, kDkmsConfPackageName)) {
      package_name = std::string{*parsed_name};
    } else if (const auto parsed_version =
                   try_parse_dkms_conf_assignment(line_str, kDkmsConfPackageVersion)) {
      package_version = std::string{*parsed_version};
    }
  }

  if ((package_name.empty()) || (package_version.empty())) {
    log_driver_package_debug(caller,
                             conf_path.string() + " has no PACKAGE_NAME or PACKAGE_VERSION");
    return std::nullopt;
  }
  return std::make_pair(package_name, package_version);
}

auto package_version_from_kernel_symlink(std::string_view caller, const fs::path& dkms_root,
                                         const fs::path& kernel_link)
    -> std::optional<std::string> {
  auto err_code = std::error_code{};
  const auto target = fs::read_symlink(kernel_link, err_code);
  if ((err_code) || (target.empty())) {
    log_driver_package_debug(caller, "No kernel symlink at: " + kernel_link.string());
    return std::nullopt;
  }

  // DKMS writes <version>/<release>/<machine>, relative to the root.
  const auto relative = target.is_relative() ? target : fs::relative(target, dkms_root, err_code);
  if ((err_code) || (relative.empty())) {
    log_driver_package_debug(caller, "Cannot resolve kernel symlink at: " + kernel_link.string());
    return std::nullopt;
  }

  // Also rejects a target outside the DKMS root, whose first component is "..".
  auto version = relative.begin()->string();
  if (!parse_package_version(version).has_value()) {
    log_driver_package_debug(caller, "Kernel symlink at: " + kernel_link.string() +
                                         " does not name a package version: " + target.string());
    return std::nullopt;
  }
  return version;
}

auto is_valid_dkms_package(std::string_view caller, const fs::path& version_dir,
                           std::string_view dir_name) -> bool {
  // DKMS source_tree is configurable, so follow the source symlink instead of assuming /usr/src.
  const auto dkms_conf_path = (version_dir / kDkmsSourceSymlinkName / kDkmsConfName);
  const auto package_info = read_dkms_conf(caller, dkms_conf_path);
  if (!package_info.has_value()) {
    return false;
  }

  const auto& [package_name, package_version] = *package_info;
  if ((package_name != kAmdgpuDkmsPackageName) || (package_version != dir_name)) {
    log_driver_package_debug(caller, dkms_conf_path.string() + " is for " + package_name + " " +
                                         package_version + "; expected " +
                                         std::string{kAmdgpuDkmsPackageName} + " " +
                                         std::string{dir_name});
    return false;
  }
  return true;
}

// Waits for rpm to exit, but not past `until`. Returns what waitpid() said last: the pid once
// rpm was reaped, 0 when it was still running, -1 when there was nothing to reap, as for a
// caller that ignores SIGCHLD.
auto wait_for_rpm(pid_t pid, std::chrono::steady_clock::time_point until, int* status) -> pid_t {
  auto pause = kRpmExitPollFirst;
  for (;;) {
    const auto waited = waitpid(pid, status, WNOHANG);
    if ((waited != 0) || (std::chrono::steady_clock::now() >= until)) {
      return waited;
    }
    std::this_thread::sleep_for(pause);
    pause = std::min(pause * 2, kRpmExitPollMax);
  }
}

// P_PIDFD, which glibc older than 2.36 does not declare.
constexpr auto kWaitIdPidfd = static_cast<idtype_t>(3);

// A pidfd names one process, so a late rpm can be reaped later without a pid-reuse race, also when
// the host ignores SIGCHLD. Returns kNoPidFd before Linux 5.3, which has no pidfd_open.
auto open_pidfd(pid_t pid) -> int {
#ifdef SYS_pidfd_open
  return static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
#else
  static_cast<void>(pid);
  return kNoPidFd;
#endif
}

class PidFd {
 public:
  explicit PidFd(int fd) : fd_{fd} {}
  PidFd(const PidFd&) = delete;
  auto operator=(const PidFd&) -> PidFd& = delete;
  ~PidFd() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }
  auto release() -> int { return std::exchange(fd_, kNoPidFd); }

 private:
  int fd_;
};

// The one rpm that missed its deadline and may still run. Without a pidfd its pid is polled. Its
// stdout read end stays open until it has ended, so a late write is not a SIGPIPE.
struct LateRpm {
  pid_t pid = -1;
  int pidfd = kNoPidFd;
  int output_fd = -1;
};
std::mutex g_late_rpm_mutex;
LateRpm g_late_rpm;

auto close_late_rpm(LateRpm* late) -> void {
  if (late->pidfd >= 0) {
    close(late->pidfd);
  }
  close(late->output_fd);
  *late = LateRpm{};
}

auto keep_late_rpm(LateRpm late) -> void {
  const auto lock = std::lock_guard<std::mutex>{g_late_rpm_mutex};
  if (g_late_rpm.pid >= 0) {
    close_late_rpm(&late);
    return;
  }
  g_late_rpm = late;
}

// True while an rpm that missed its deadline still runs, so that none is started next to it. One
// that has ended is reaped here.
auto is_late_rpm_running() -> bool {
  const auto lock = std::lock_guard<std::mutex>{g_late_rpm_mutex};
  if (g_late_rpm.pid < 0) {
    return false;
  }
  if (g_late_rpm.pidfd < 0) {
    // 0 means still running. The pid, or -1 (ECHILD) when SIGCHLD is ignored, means it ended.
    auto status = 0;
    if (waitpid(g_late_rpm.pid, &status, WNOHANG) == 0) {
      return true;
    }
    close_late_rpm(&g_late_rpm);
    return false;
  }
  // A poll error keeps the rpm tracked: dropping a running one would let a second one start.
  auto poll_fd = pollfd{g_late_rpm.pidfd, POLLIN, 0};
  if (poll(&poll_fd, 1, 0) <= 0) {
    return true;
  }
  // ECHILD when SIGCHLD is ignored or the host reaped it already: nothing is left to reap.
  auto info = siginfo_t{};
  waitid(kWaitIdPidfd, static_cast<id_t>(g_late_rpm.pidfd), &info, WEXITED | WNOHANG);
  close_late_rpm(&g_late_rpm);
  return false;
}

// Keeps the caller's open descriptors out of rpm, which can outlive the call. glibc 2.34 closes
// them in the child, so no thread can open one in between. Returns 0 or an errno value.
#if defined(__GLIBC__) && ((__GLIBC__ > 2) || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 34)))
auto add_close_of_open_descriptors(posix_spawn_file_actions_t* actions) -> int {
  return posix_spawn_file_actions_addclosefrom_np(actions, (STDERR_FILENO + 1));
}
#else
// Older libc: the descriptors are listed here, so one that another thread opens afterwards is
// still inherited.
auto add_close_of_open_descriptors(posix_spawn_file_actions_t* actions) -> int {
  auto err_code = std::error_code{};
  auto entry = fs::directory_iterator{"/proc/self/fd", err_code};
  for (; (!err_code) && (entry != fs::directory_iterator{}); entry.increment(err_code)) {
    const auto name = entry->path().filename().string();
    auto fd = 0;
    const auto parsed = std::from_chars(name.data(), name.data() + name.size(), fd);
    if ((parsed.ec != std::errc{}) || (fd <= STDERR_FILENO)) {
      continue;
    }
    if (const auto add_err = posix_spawn_file_actions_addclose(actions, fd); add_err != 0) {
      return add_err;
    }
  }
  return err_code.value();
}
#endif

// rpm must not inherit a blocked or ignored signal, which would make it survive SIGTERM. Returns 0
// or an errno value, and leaves `attr` initialized only for 0.
auto init_default_signals(posix_spawnattr_t* attr) -> int {
  auto no_signals = sigset_t{};
  auto all_signals = sigset_t{};
  sigemptyset(&no_signals);
  sigfillset(&all_signals);
  auto err = posix_spawnattr_init(attr);
  if (err != 0) {
    return err;
  }
  err = posix_spawnattr_setsigmask(attr, &no_signals);
  if (err == 0) {
    err = posix_spawnattr_setsigdefault(attr, &all_signals);
  }
  if (err == 0) {
    err = posix_spawnattr_setflags(attr, (POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF));
  }
  if (err != 0) {
    posix_spawnattr_destroy(attr);
  }
  return err;
}

// Runs `<rpm_path> <args>` and returns what it printed, or nothing when rpm failed, printed more
// than is kept, or had not finished by the deadline.
auto run_rpm(std::string_view caller, const std::string& rpm_path, std::vector<std::string> args,
             std::chrono::steady_clock::time_point deadline) -> std::optional<std::string> {
  if (std::chrono::steady_clock::now() >= deadline) {
    log_driver_package_debug(caller, "No time left to run rpm for: " + args.back());
    return std::nullopt;
  }
  int pipe_fds[2] = {-1, -1};
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    log_driver_package_debug(caller, std::string{"pipe2 failed: "} + std::strerror(errno));
    return std::nullopt;
  }

  // dup2(fd, fd) leaves close-on-exec set before glibc 2.29, so a write end on stdout would not
  // reach rpm.
  const auto write_end = fcntl(pipe_fds[1], F_DUPFD_CLOEXEC, (STDERR_FILENO + 1));
  close(pipe_fds[1]);
  pipe_fds[1] = write_end;
  if (write_end < 0) {
    close(pipe_fds[0]);
    log_driver_package_debug(caller, std::string{"fcntl failed: "} + std::strerror(errno));
    return std::nullopt;
  }

  auto arg0 = rpm_path;
  auto argv = std::vector<char*>{arg0.data()};
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);
  // rpm gets none of the caller's environment, which can redirect its macros and database. HOME
  // is set because without it rpm reads ~/.rpmmacros relative to the caller's directory.
  auto env0 = std::string{"LC_ALL=C"};
  auto env1 = std::string{"HOME=/"};
  char* const envp[] = {env0.data(), env1.data(), nullptr};

  auto pid = pid_t{-1};
  auto actions = posix_spawn_file_actions_t{};
  auto spawn_err = posix_spawn_file_actions_init(&actions);
  if (spawn_err == 0) {
    spawn_err = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (spawn_err == 0) {
      spawn_err = posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
    }
    if (spawn_err == 0) {
      spawn_err =
          posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    }
    // After the three above: the pipe is among the descriptors it closes.
    if (spawn_err == 0) {
      spawn_err = add_close_of_open_descriptors(&actions);
    }
    auto attr = posix_spawnattr_t{};
    auto has_attr = false;
    if (spawn_err == 0) {
      spawn_err = init_default_signals(&attr);
      has_attr = (spawn_err == 0);
    }
    if (has_attr) {
      spawn_err = posix_spawn(&pid, rpm_path.c_str(), &actions, &attr, argv.data(), envp);
      posix_spawnattr_destroy(&attr);
    }
    posix_spawn_file_actions_destroy(&actions);
  }
  close(pipe_fds[1]);
  if (spawn_err != 0) {
    close(pipe_fds[0]);
    log_driver_package_debug(caller, "Cannot run " + rpm_path + ": " + std::strerror(spawn_err));
    return std::nullopt;
  }
  auto pidfd = PidFd{open_pidfd(pid)};

  // Read to the end of the output so rpm never blocks on a full pipe, but keep only the start.
  auto output = std::string{};
  auto truncated = false;
  auto finished = false;
  while (!finished) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      break;
    }
    auto poll_fd = pollfd{pipe_fds[0], POLLIN, 0};
    const auto wait_ms = std::min<std::chrono::milliseconds::rep>(remaining.count(), INT_MAX);
    const auto ready = poll(&poll_fd, 1, static_cast<int>(wait_ms));
    if ((ready < 0) && (errno == EINTR)) {
      continue;
    }
    if (ready <= 0) {
      break;
    }
    char buffer[512];
    const auto bytes = read(pipe_fds[0], buffer, sizeof(buffer));
    if ((bytes < 0) && (errno == EINTR)) {
      continue;
    }
    if (bytes <= 0) {
      finished = (bytes == 0);
      break;
    }
    const auto kept = std::min(static_cast<size_t>(bytes), kRpmQueryMaxOutput - output.size());
    output.append(buffer, kept);
    truncated = truncated || (kept < static_cast<size_t>(bytes));
  }
  // A late rpm is not killed, which can leave a Berkeley DB rpm database unusable. Its pipe and
  // pidfd are kept until is_late_rpm_running() reaps it.
  auto wait_status = 0;
  const auto waited = wait_for_rpm(pid, deadline, &wait_status);
  if (waited == 0) {
    keep_late_rpm(LateRpm{pid, pidfd.release(), pipe_fds[0]});
  } else {
    close(pipe_fds[0]);
  }
  if ((!finished) || (waited == 0)) {
    log_driver_package_debug(caller, "rpm did not answer in time for: " + args.back());
    return std::nullopt;
  }

  // A caller that ignores SIGCHLD reaps rpm itself, so no exit status is left to read.
  if ((waited == pid) && ((!WIFEXITED(wait_status)) || (WEXITSTATUS(wait_status) != 0))) {
    log_driver_package_debug(caller, "rpm failed for: " + args.back());
    return std::nullopt;
  }
  if (truncated) {
    log_driver_package_debug(caller, "rpm printed more than is kept for: " + args.back());
    return std::nullopt;
  }
  // Without an exit status, empty output cannot tell "not installed" from a failed rpm.
  if ((waited != pid) && output.empty()) {
    log_driver_package_debug(caller, "rpm gave no exit status and no output for: " + args.back());
    return std::nullopt;
  }
  return output;
}

// A query makes rpm create a database where it finds none, so rpm is first asked where its
// database is, and the package is asked for only where that directory holds the packages file.
auto rpm_has_database(std::string_view caller, const std::string& rpm_path,
                      std::chrono::steady_clock::time_point deadline) -> bool {
  const auto output = run_rpm(caller, rpm_path, {"--eval", "%{_dbpath}"}, deadline);
  if (!output.has_value()) {
    return false;
  }

  const auto dbpath = fs::path{trim(*output)};
  const auto has_database =
      dbpath.is_absolute() &&
      std::any_of(kRpmDatabaseFiles.begin(), kRpmDatabaseFiles.end(), [&dbpath](auto file) {
        auto err_code = std::error_code{};
        return fs::is_regular_file(dbpath / file, err_code);
      });
  if (!has_database) {
    log_driver_package_debug(caller, "No rpm database at: " + dbpath.string());
  }
  return has_database;
}

}  // namespace

auto smi_amdgpu_get_active_dkms_version(std::string_view dkms_root, std::string_view release,
                                        std::string_view machine, std::string* active_version)
    -> amdsmi_status_t {
  if (active_version == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  active_version->clear();

  if ((release.empty()) || (machine.empty())) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  const auto root = fs::path{std::string{dkms_root}};
  auto link_name = std::string{kDkmsKernelPrefix};
  link_name.append(release).append("-").append(machine);
  const auto selected = package_version_from_kernel_symlink(__func__, root, root / link_name);
  if ((!selected.has_value()) || (!is_valid_dkms_package(__func__, root / *selected, *selected))) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  *active_version = *selected;
  return AMDSMI_STATUS_SUCCESS;
}

auto smi_amdgpu_get_kmod_versions(std::string_view rpm_path, std::string_view release,
                                  std::vector<std::string>* kmod_versions,
                                  std::chrono::milliseconds timeout) -> amdsmi_status_t {
  if (kmod_versions == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }
  kmod_versions->clear();

  if (release.empty()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  if (is_late_rpm_running()) {
    log_driver_package_debug(__func__, "An earlier rpm is still running");
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  auto package = std::string{kAmdgpuKmodPackagePrefix};
  package.append(release);
  const auto rpm = std::string{rpm_path};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  if (!rpm_has_database(__func__, rpm, deadline)) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  const auto output =
      run_rpm(__func__, rpm, {"-qa", "--qf", "%{VERSION}-%{RELEASE}\\n", package}, deadline);
  if (!output.has_value()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  // rpm prints one line per installed version, or none. One line that is not understood could be
  // the package of the loaded module, so then none is reported.
  auto versions = std::vector<std::string>{};
  auto lines = std::istringstream{*output};
  for (auto line = std::string{}; std::getline(lines, line);) {
    const auto parsed = parse_package_version(line);
    if (!parsed.has_value()) {
      log_driver_package_debug(__func__, package + " has an unexpected version");
      return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    versions.push_back(parsed->version);
  }

  *kmod_versions = std::move(versions);
  return AMDSMI_STATUS_SUCCESS;
}

KmodVersionCache::KmodVersionCache(std::string_view rpm_path, std::chrono::milliseconds retry_after)
    : rpm_path_{rpm_path}, retry_after_{retry_after} {}

auto KmodVersionCache::versions(std::string_view release) -> std::vector<std::string> {
  auto lock = std::unique_lock<std::mutex>{mutex_};
  for (;;) {
    const auto has_answer_for_release = has_answer_ && (release_ == release);
    const auto is_fresh =
        has_answer_for_release &&
        (is_found_ || ((std::chrono::steady_clock::now() - asked_at_) < retry_after_));
    // A stale answer is a failure (empty): it is given at once while another caller asks rpm.
    if (is_fresh || (is_refreshing_ && has_answer_for_release)) {
      return versions_;
    }
    if (!is_refreshing_) {
      break;
    }
    refreshed_.wait(lock);
  }

  // rpm runs without the lock, so a caller that has an answer is not held up by it.
  is_refreshing_ = true;
  lock.unlock();
  struct EndRefresh {
    KmodVersionCache* cache;
    std::unique_lock<std::mutex>* lock;
    ~EndRefresh() {
      if (!lock->owns_lock()) {
        lock->lock();
      }
      cache->is_refreshing_ = false;
      cache->refreshed_.notify_all();
    }
  } const end_refresh{this, &lock};
  auto fresh = std::vector<std::string>{};
  const auto status = smi_amdgpu_get_kmod_versions(rpm_path_, release, &fresh);
  lock.lock();
  is_found_ = (status == AMDSMI_STATUS_SUCCESS);
  versions_ = std::move(fresh);
  release_ = release;
  asked_at_ = std::chrono::steady_clock::now();
  has_answer_ = true;
  return versions_;
}

auto smi_amdgpu_parse_driver_versions(std::string_view module_version,
                                      std::string_view dkms_version, std::string_view kmod_version,
                                      amdsmi_driver_info_v2_t* info) -> amdsmi_status_t {
  if (info == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  const auto module = parse_module_version(module_version);
  const auto dkms = parse_package_version(dkms_version);
  const auto kmod = parse_package_version(kmod_version);

  // DKMS before kmod. A package built from another kernel version is not the loaded module's,
  // so neither it nor its build is reported.
  const AmdgpuPackageVersion* package = nullptr;
  auto package_type = AMDSMI_DRIVER_PACKAGE_NONE;
  if (is_package_for_module(dkms, module)) {
    package = &*dkms;
    package_type = AMDSMI_DRIVER_PACKAGE_DKMS;
  } else if (is_package_for_module(kmod, module)) {
    package = &*kmod;
    package_type = AMDSMI_DRIVER_PACKAGE_KMOD;
  } else if (dkms.has_value() || kmod.has_value()) {
    log_driver_package_debug(__func__, "No installed package has the kernel version of module " +
                                           std::string{module_version});
  }

  // A build that does not fit behind the module version in its field is not reported at all.
  if ((package != nullptr) &&
      ((module_version.size() + 1 + package->build.size()) >= AMDSMI_MAX_STRING_LENGTH)) {
    package = nullptr;
    package_type = AMDSMI_DRIVER_PACKAGE_NONE;
  }

  auto full_version = std::string{};
  if ((!module_version.empty()) && (module_version != "N/A")) {
    full_version = module_version;
  }
  if (package != nullptr) {
    full_version.append("-").append(package->build);
  }

  smi_clear_char_and_reinitialize(info->driver_kernel_version, AMDSMI_MAX_STRING_LENGTH,
                                  module.has_value() ? module->kernel_version : std::string{});
  smi_clear_char_and_reinitialize(info->amdgpu_driver_version, AMDSMI_MAX_STRING_LENGTH,
                                  module.has_value() ? module->amdgpu_version : std::string{});
  smi_clear_char_and_reinitialize(info->driver_build_version, AMDSMI_MAX_STRING_LENGTH,
                                  (package != nullptr) ? package->build : std::string{});
  smi_clear_char_and_reinitialize(info->driver_full_version, AMDSMI_MAX_STRING_LENGTH,
                                  full_version);
  info->driver_package = package_type;

  return AMDSMI_STATUS_SUCCESS;
}

auto smi_amdgpu_get_driver_versions(
    std::string_view module_version, std::string_view dkms_root, const utsname& uts,
    const std::function<std::vector<std::string>(std::string_view)>& kmod_lookup,
    amdsmi_driver_info_v2_t* info) -> amdsmi_status_t {
  auto dkms_version = std::string{};
  auto kmod_version = std::string{};
  // A module without an amdgpu package version cannot match a package. rpm is slow to start, so
  // kmod is looked up only when DKMS is not the module's package.
  const auto module = parse_module_version(module_version);
  if (module.has_value()) {
    smi_amdgpu_get_active_dkms_version(dkms_root, uts.release, uts.machine, &dkms_version);
    if (!is_package_for_module(parse_package_version(dkms_version), module)) {
      kmod_version = only_version_of_module(kmod_lookup(uts.release), module);
    }
  }
  return smi_amdgpu_parse_driver_versions(module_version, dkms_version, kmod_version, info);
}
