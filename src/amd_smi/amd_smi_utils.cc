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

#include <limits.h>
#include <sys/ioctl.h>
#include <libdrm/amdgpu.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <dirent.h>
#include <sys/types.h>

#include <memory>
#include <random>
#include <fstream>
#include <iostream>
#include <regex>
#include <cstdio>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "shared_mutex.h"  // NOLINT
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"

std::string leftTrim(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("^\\s+"), "");
  }
  return s;
}

std::string rightTrim(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\\s+$"), "");
  }
  return s;
}

std::string removeNewLines(const std::string &s) {
  if (!s.empty()) {
    return std::regex_replace(s, std::regex("\n+"), "");
  }
  return s;
}

std::string trim(const std::string &s) {
  if (!s.empty()) {
    // remove new lines -> trim white space at ends
    std::string noNewLines = removeNewLines(s);
    return leftTrim(rightTrim(noNewLines));
  }
  return s;
}

// Given original string and string to remove (removeMe)
// Return will provide the resulting modified string with the removed string(s)
std::string removeString(const std::string origStr,
                         const std::string &removeMe) {
  std::string modifiedStr = origStr;
  std::string::size_type l = removeMe.length();
  for (std::string::size_type i = modifiedStr.find(removeMe);
       i != std::string::npos;
       i = modifiedStr.find(removeMe)) {
    modifiedStr.erase(i, l);
  }
  return modifiedStr;
}

amdsmi_status_t smi_clear_char_and_reinitialize(char buffer[], uint32_t len,
                                                    std::string newString) {
    char *begin = &buffer[0];
    char *end = &buffer[len];
    std::fill(begin, end, 0);

    // Safer approach - copy directly with length limit
    size_t copy_len = std::min(static_cast<size_t>(len - 1), newString.length());
    if (copy_len > 0) {
        std::memcpy(buffer, newString.c_str(), copy_len);
    }
    buffer[copy_len] = '\0';
    return AMDSMI_STATUS_SUCCESS;
}

int openFileAndModifyBuffer(std::string path, char *buff, size_t sizeOfBuff,
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
    if (!errorDiscovered && file.good() && !file.bad() && !file.fail() && !file.eof()
        && !contents.empty()) {
        std::strncpy(buff, contents.c_str(), sizeOfBuff-1);
        buff[sizeOfBuff-1] = '\0';
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

amdsmi_status_t smi_amdgpu_find_hwmon_dir(amd::smi::AMDSmiGPUDevice *device, std::string* full_path)
{
    if (full_path == nullptr) {
        return AMDSMI_STATUS_API_FAILED;
    }
    SMIGPUDEVICE_MUTEX(device->get_mutex())
        DIR *dh;
    struct dirent * contents;
    std::string device_path = "/sys/class/drm/" + device->get_gpu_path();
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
        if (name.find("hwmon", 0) != std::string::npos)
            *full_path = directory_path + name;
    }

    closedir(dh);

    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t smi_amdgpu_get_board_info(amd::smi::AMDSmiGPUDevice* device, amdsmi_board_info_t *info) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())
    std::string model_number_path = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/product_number");
    std::string product_serial_path = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/serial_number");
    std::string fru_id_path = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/fru_id");
    std::string manufacturer_name_path = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/manufacturer");
    std::string product_name_path = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/product_name");

    auto ret_mod = openFileAndModifyBuffer(model_number_path, info->model_number,
                                           AMDSMI_MAX_STRING_LENGTH);
    auto ret_ser = openFileAndModifyBuffer(product_serial_path, info->product_serial,
                                           AMDSMI_MAX_STRING_LENGTH);
    auto ret_fru = openFileAndModifyBuffer(fru_id_path, info->fru_id, AMDSMI_MAX_STRING_LENGTH);
    auto ret_man = openFileAndModifyBuffer(manufacturer_name_path, info->manufacturer_name,
                                           AMDSMI_MAX_STRING_LENGTH);
    auto ret_prod = openFileAndModifyBuffer(product_name_path, info->product_name,
                                            AMDSMI_MAX_STRING_LENGTH, false);

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "[Before correction] "
       << "Returning status = AMDSMI_STATUS_SUCCESS"
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
       << "; info->product_name: |" << info->product_name << "|"
       << "; ret_prod = " << ret_prod << "|\n";
    LOG_INFO(ss);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_power_cap(amd::smi::AMDSmiGPUDevice* device, int *cap)
{
    constexpr int DATA_SIZE = 16;
    char val[DATA_SIZE];
    std::string fullpath;
    amdsmi_status_t ret = AMDSMI_STATUS_SUCCESS;

    ret = smi_amdgpu_find_hwmon_dir(device, &fullpath);

    SMIGPUDEVICE_MUTEX(device->get_mutex())

    if (ret)
        return ret;

    fullpath += "/power1_cap";
    std::ifstream file(fullpath.c_str(), std::ifstream::in);
    if (!file.is_open()) {
        return AMDSMI_STATUS_API_FAILED;
    }

    file.getline(val, DATA_SIZE);

    if (sscanf(val, "%d", cap) < 0) {
        return AMDSMI_STATUS_API_FAILED;
    }


    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_ranges(amd::smi::AMDSmiGPUDevice* device, amdsmi_clk_type_t domain,
        int *max_freq, int *min_freq, int *num_dpm, int *sleep_state_freq)
{
    SMIGPUDEVICE_MUTEX(device->get_mutex())
        std::string fullpath = "/sys/class/drm/" + device->get_gpu_path() + "/device";

    switch (domain) {
        case AMDSMI_CLK_TYPE_GFX:
            fullpath += "/pp_dpm_sclk";
            break;
        case AMDSMI_CLK_TYPE_MEM:
            fullpath += "/pp_dpm_mclk";
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
            fullpath += "/pp_dpm_fclk";
            break;
        default:
            return AMDSMI_STATUS_INVAL;
    }

    std::ifstream ranges(fullpath.c_str());

    if (ranges.fail()) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    unsigned int max, min, dpm, sleep_freq, current_freq;
    char str[10];
    char single_char;
    max = 0;
    min = UINT_MAX;
    dpm = 0;
    sleep_freq = UINT_MAX;
    current_freq = 0;

    for (std::string line; getline(ranges, line);) {
        unsigned int dpm_level, freq;

        char firstChar = line[0];
        if (firstChar == 'S') {
            if (sscanf(line.c_str(), "%c: %d%s", &single_char, &sleep_freq, str) <= 2) {
                ranges.close();
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

            if (sscanf(line.c_str(), "%u: %d%c", &dpm_level, &freq, str) <= 2){
                ranges.close();
                return AMDSMI_STATUS_IO;
            }

            char lastChar = line.back();
            if (lastChar == '*'){
                current_freq = freq;
                continue;
            }

            // not * was detected so check for the min max
            max = freq > max ? freq : max;
            min = freq < min ? freq : min;
            dpm = dpm_level > dpm ? dpm_level : dpm;
        }
    
    }
    if (dpm == 0 && current_freq > 0) {
        // if the dpm level is 0, then the current frequency is the min/max frequency
        max = current_freq;
        min = current_freq;
    }
    if (num_dpm)
        *num_dpm = dpm;
    if (max_freq)
        *max_freq = max;
    if (min_freq)
        *min_freq = min;
    if (sleep_state_freq)
        *sleep_state_freq = sleep_freq;

    ranges.close();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_enabled_blocks(amd::smi::AMDSmiGPUDevice* device, uint64_t *enabled_blocks) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())
    std::string fullpath = "/sys/class/drm/" + device->get_gpu_path() + "/device/ras/features";
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

    *enabled_blocks = strtoul(tmp_str.c_str(), nullptr, 16);
    f.close();

    if (*enabled_blocks == 0 || *enabled_blocks == ULONG_MAX) {
        return AMDSMI_STATUS_API_FAILED;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_bad_page_info(amd::smi::AMDSmiGPUDevice* device,
        uint32_t *num_pages, amdsmi_retired_page_record_t *info) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())
        std::string line;
    std::vector<std::string> badPagesVec;

    std::string fullpath = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/ras/gpu_vram_bad_pages");
    std::ifstream fs(fullpath.c_str());

    if (fs.fail()) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    while (std::getline(fs, line)) {
        badPagesVec.push_back(line);
    }

    if (badPagesVec.size() == 0) {
        num_pages = 0;
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
        uint32_t *threshold) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())

    //TODO: Accessing the node requires root privileges, and its interface may need to be exposed in another path
    uint32_t index = device->get_card_id();
    std::string fullpath = "/sys/kernel/debug/dri/" + std::to_string(index) + std::string("/ras/bad_page_cnt_threshold");
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

    //TODO: need to expose the corresponding interface to validate the checksum of ras eeprom table.
    //verify fail: return AMDSMI_STATUS_CORRUPTED_EEPROM
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t smi_amdgpu_get_ecc_error_count(amd::smi::AMDSmiGPUDevice* device, amdsmi_error_count_t *err_cnt) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())
        char str[10];

    std::string fullpath = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/ras/umc_err_count");
    std::ifstream f(fullpath.c_str());

    if (f.fail()) {
        //fall back to aca file
        fullpath = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/ras/aca_umc");
        f.open(fullpath.c_str());
        if (f.fail()) {
            return AMDSMI_STATUS_NOT_SUPPORTED;
        }
    }

    std::string line;
    getline(f, line);
    sscanf(line.c_str(), "%s%ld", str, &(err_cnt->uncorrectable_count));

    getline(f, line);
    sscanf(line.c_str(), "%s%ld", str, &(err_cnt->correctable_count));

    f.close();

    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t smi_amdgpu_get_driver_version(amd::smi::AMDSmiGPUDevice* device, int *length, char *version) {
    SMIGPUDEVICE_MUTEX(device->get_mutex())
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    size_t len;
    if (*length <= 0 || version == nullptr) {
        return AMDSMI_STATUS_INVAL;
    } else {
        len = static_cast<size_t>(*length);
    }

    std::string empty = "";
    std::strncpy(version, empty.c_str(), len-1);
    openFileAndModifyBuffer("/sys/module/amdgpu/version",
                                      version, static_cast<size_t>(len));
    if (version[0] == '\0') {
        openFileAndModifyBuffer("/proc/version", version, static_cast<size_t>(len));
        if (version[0] == '\0') {
            return AMDSMI_STATUS_IO;
        }
    }
    return status;
}

amdsmi_status_t smi_amdgpu_get_pcie_speed_from_pcie_type(uint16_t pcie_type, uint32_t *pcie_speed)
{
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

amdsmi_status_t smi_amdgpu_get_market_name_from_dev_id(amd::smi::AMDSmiGPUDevice* device, char *market_name) {
    if (market_name == nullptr || device == nullptr) {
        return AMDSMI_STATUS_ARG_PTR_NULL;
    }

    std::ostringstream ss;
    // requires libdrm being active
    if (!device->check_if_drm_is_supported()) {
        ss << __PRETTY_FUNCTION__ << " | DRM is not supported";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    uint32_t major_version, minor_version;
    amdgpu_device_handle device_handle = nullptr;
    std::string render_name = device->get_gpu_path();
    int fd = -1;
    std::string path = "/dev/dri/" + render_name;

    if (render_name != "") {
        fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    } else {
        market_name[0] = '\0';
        close(fd);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    ss << __PRETTY_FUNCTION__ << " | Render Name: "
    << render_name << "; path: " << path << "; fd: " << fd;
    LOG_DEBUG(ss);

    int ret = amdgpu_device_initialize(fd, &major_version, &minor_version, &device_handle);
    if (ret != 0) {
        std::string empty = "";
        std::strncpy(market_name, empty.c_str(), AMDSMI_256_LENGTH - 1);
        amdgpu_device_deinitialize(device_handle);
        close(fd);
        return AMDSMI_STATUS_DRM_ERROR;
    }

    // Get the marketing name using libdrm's API
    const char *name = amdgpu_get_marketing_name(device_handle);
    if (name != nullptr) {
        std::strncpy(market_name, name, AMDSMI_256_LENGTH - 1);
        market_name[AMDSMI_256_LENGTH - 1] = '\0';
        amdgpu_device_deinitialize(device_handle);
        close(fd);
        return AMDSMI_STATUS_SUCCESS;
    }

    amdgpu_device_deinitialize(device_handle);
    close(fd);
    return AMDSMI_STATUS_DRM_ERROR;
}

amdsmi_status_t smi_amdgpu_is_gpu_power_management_enabled(amd::smi::AMDSmiGPUDevice* device,
        bool *enabled) {
    if (enabled == nullptr) {
        return AMDSMI_STATUS_API_FAILED;
    }

    SMIGPUDEVICE_MUTEX(device->get_mutex())
    std::string fullpath = "/sys/class/drm/" + device->get_gpu_path() + std::string("/device/pp_features");
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

// wrapper to return string expression of a rsmi_status_t return
// rsmi_status_t ret - return value of RSMI API function
// bool fullStatus - defaults to true, set to false to chop off description
// Returns:
// string - if fullStatus == true, returns full decription of return value
//      ex. 'RSMI_STATUS_SUCCESS: The function has been executed successfully.'
// string - if fullStatus == false, returns a minimalized return value
//      ex. 'RSMI_STATUS_SUCCESS'
std::string smi_amdgpu_get_status_string(amdsmi_status_t ret, bool fullStatus = true) {
  const char *err_str;
  amdsmi_status_code_to_string(ret, &err_str);
  if (!fullStatus) {
    return smi_amdgpu_split_string(std::string(err_str), ':');
  }
  return std::string(err_str);
}

// TODO(amdsmi_team): Do we want to include these functions in header?
amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t *device_index) {
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
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
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
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, &processor_handles[0]);
    ss << __PRETTY_FUNCTION__ << " | Processor Count: " << device_count << "\n";
    LOG_DEBUG(ss);

    for (uint32_t j = 0; j < device_count; j++) {
      if (processor_handles[j] == processor_handle) {
        *device_index = current_device_index;
        ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_SUCCESS "
           << "Returning device_index: " << *device_index << "\nSocket #: " << i
           << "; Device #: " << j << "; current_device_index #: " << current_device_index
           << "\n";
        // std::cout << ss.str();
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
amdsmi_status_t smi_amdgpu_get_device_count(uint32_t *total_num_devices) {
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
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
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
    ret = amdsmi_get_processor_handles(sockets[i], &processor_count, &processor_handles[0]);
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

// TODO(amdsmi_team): Do we want to include these functions in header?
amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(
                                        uint32_t device_index,
                                        amdsmi_processor_handle *processor_handle) {
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
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
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
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, &processor_handles[0]);
    ss << __PRETTY_FUNCTION__ << " | Processor Count: " << device_count << "\n";
    LOG_DEBUG(ss);

    for (uint32_t j = 0; j < device_count; j++) {
    //   std::cout << "current_device_index: " << current_device_index
    //   << " device_index: " << device_index << std::endl;
      if (current_device_index == device_index) {
        *processor_handle = processor_handles[j];
        ss << __PRETTY_FUNCTION__ << " | AMDSMI_STATUS_SUCCESS"
           << "\nReturning processor_handle for device_index: " << device_index
           << "\nSocket #: " << i << "; Device #: " << j
           << "; current_device_index #: " << current_device_index
           << "; processor_handle: " << *processor_handle
           << "; processor_handles[j]: " << processor_handles[j]
           << "\n";
        // std::cout << ss.str();
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

static std::vector<const amdsmi_cper_hdr_t *>
amdsmi_get_gpu_cper_headers(const char *buffer, size_t buffer_sz) {

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
        << "[CPER] buffer_sz: " << buffer_sz;
    LOG_DEBUG(ss);

    std::vector<const amdsmi_cper_hdr_t *> headers;
    if(!buffer) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] buffer is null";
        LOG_ERROR(ss);
        return headers;
    }
    static constexpr char cper_signature[] = "CPER";
    static constexpr size_t cper_signature_size = sizeof(cper_signature) - 1;
    for(size_t data_idx = 0;
        buffer_sz >= cper_signature_size &&
        data_idx < buffer_sz - cper_signature_size;
        ++data_idx) {

        const amdsmi_cper_hdr_t *hdr = reinterpret_cast<const amdsmi_cper_hdr_t *>(
            &buffer[data_idx]);
        if(hdr->signature[0] != 'C' || hdr->signature[1] != 'P' ||
            hdr->signature[2] != 'E' || hdr->signature[3] != 'R' ) {
            continue;
        }
        if(hdr->signature_end != 0xFFFFFFFF) {
            continue;
        }
        if(hdr->record_length > buffer_sz) {
            continue;
        }
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] add header at data_idx: " << data_idx
            << ", sig: " << hdr->signature[0] << hdr->signature[1] << hdr->signature[2] << hdr->signature[3];
        LOG_DEBUG(ss);
        headers.emplace_back(hdr);
    }
    return headers;
}

struct CperFileCtx {
    amdsmi_status_t status = AMDSMI_STATUS_FILE_ERROR;
    std::unique_ptr<char[]> buffer;
    long file_size = 0;
};

static auto amdsmi_read_cper_file(const std::string &filepath) -> CperFileCtx {

    std::ostringstream ss;

    CperFileCtx ctx;
    ctx.status = AMDSMI_STATUS_FILE_ERROR;
    ctx.file_size = 0;

    struct stat file_stats;
    if (stat(filepath.c_str(), &file_stats) == 0) {
        if (!S_ISREG(file_stats.st_mode)) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] file is not a regular file: "
                << filepath << ", errno: " << errno << "): " << strerror(errno);
            return ctx;
        }
    } else {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] file does not exist: "
            << filepath << ", errno: " << errno << "): " << strerror(errno);
        ctx.status = AMDSMI_STATUS_NOT_SUPPORTED;
        return ctx;
    }

    ctx.file_size = file_stats.st_size;
    ctx.buffer = std::make_unique<char[]>(ctx.file_size);
    int file = open(filepath.c_str(), O_RDONLY);
    if (file == -1) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] failed to open file: "
            << filepath << ", errno:()" << errno << "): " << strerror(errno);
        LOG_ERROR(ss);
        return ctx;
    }
    long bytes_read = read(file, ctx.buffer.get(), ctx.file_size);
    if (bytes_read <= 0) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] failed to read complete file, read only  "
            << bytes_read << " of " << ctx.file_size << " bytes";
        LOG_ERROR(ss);
        return ctx;
    }
    close(file);

    ctx.status = AMDSMI_STATUS_SUCCESS;
    ctx.file_size = bytes_read;
    return ctx;
}
