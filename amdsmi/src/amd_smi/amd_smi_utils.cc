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

#include <climits>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <ctime>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <sstream>

#include "config/amd_smi_config.h"
#include "amd_smi/impl/amd_smi_utils.h"

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

uint32_t smi_brcm_get_value_u32(const std::string &folder, const std::string &file_name) {

  std::string file_path = folder + "/" + file_name;
  std::ifstream file(file_path.c_str(), std::ifstream::in);
  if (!file.is_open()) {
    return 0xFFFF;
  }
  else {
    std::string line;
    getline(file, line);
    return static_cast<uint32_t>(stoi(line));
  }

  return 0;
}

std::string smi_brcm_get_value_string(const std::string &folder, const std::string &file_name) {
  
  std::stringstream temp;
  std::string file_path = folder + "/" + file_name;
  std::ifstream file(file_path.c_str(), std::ifstream::in);
  if (!file.is_open()) {
    return "N/A";
  }
  else {
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

amdsmi_status_t smi_brcm_execute_cmd_get_data(const std::string &command, std::string *data) {
  std::string result;
  char buffer[128];

  // Open a pipe to execute the command
  std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
  if (!pipe) {
    return AMDSMI_STATUS_API_FAILED;
  }

  // Read the output of the command into the buffer
  while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
    result += buffer;
  }
  *data = result;

  return AMDSMI_STATUS_SUCCESS;
}

// TODO(amdsmi_team): Do we want to include these functions in header?
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

struct CperFileCtx {
    amdsmi_status_t status = AMDSMI_STATUS_FILE_ERROR;
    std::unique_ptr<char[]> buffer;
    long file_size = 0;
};


uint64_t get_product_serial_number(amdsmi_processor_handle processor_handle) {
    uint64_t serial_number = 0;
    amdsmi_board_info_t board_info = {};
    amdsmi_status_t status = amdsmi_get_gpu_board_info(processor_handle, &board_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ <<
            "Failed to retrieve product serial number! error: " <<
            static_cast<int>(status);
        return serial_number;
    }
    if (!*board_info.product_serial) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ <<
            " Product serial string is empty.";
        return serial_number;
    }
    try {
        serial_number = std::stoull(board_info.product_serial, nullptr, 10);
    } catch (const std::invalid_argument& e) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ <<
            " Invalid product serial string. Exception: " << e.what();
        serial_number = 0;
    } catch (const std::out_of_range& e) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ <<
            " Product serial out of range, Exception: " << e.what();
        serial_number = 0;
    }
    return serial_number;
}

std::tuple<uint64_t,uint64_t,uint64_t,uint64_t> parse_bdfid(uint64_t bdfid) {
    uint64_t domain = (bdfid >> 32) & 0xffffffff;
    uint64_t bus = (bdfid >> 8) & 0xff;
    uint64_t device_id = (bdfid >> 3) & 0x1f;
    uint64_t function = bdfid & 0x7;
    return std::tuple<uint64_t,uint64_t,uint64_t,uint64_t>(domain, bus, device_id, function);
}

amdsmi_status_t smi_amdgpu_get_device_index(amdsmi_processor_handle processor_handle,
                                            uint32_t *device_index) {
    if (device_index == nullptr) return AMDSMI_STATUS_INVAL;
    *device_index = std::numeric_limits<uint32_t>::max();
    uint32_t socket_count = 0;
    auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    uint32_t current_device_index = 0;
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t device_count = 0;
        amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
        std::vector<amdsmi_processor_handle> handles(device_count);
        amdsmi_get_processor_handles(sockets[i], &device_count, handles.data());
        for (uint32_t j = 0; j < device_count; j++) {
            if (handles[j] == processor_handle) {
                *device_index = current_device_index;
                return AMDSMI_STATUS_SUCCESS;
            }
            current_device_index++;
        }
    }
    return AMDSMI_STATUS_API_FAILED;
}

amdsmi_status_t smi_amdgpu_get_device_count(uint32_t *total_num_devices) {
    if (total_num_devices == nullptr) return AMDSMI_STATUS_INVAL;
    *total_num_devices = 0;
    uint32_t socket_count = 0;
    auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    uint32_t device_num = 0;
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t processor_count = 0;
        amdsmi_get_processor_handles(sockets[i], &processor_count, nullptr);
        device_num += processor_count;
    }
    *total_num_devices = device_num;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t smi_amdgpu_get_processor_handle_by_index(uint32_t device_index,
                                        amdsmi_processor_handle *processor_handle) {
    if (processor_handle == nullptr) return AMDSMI_STATUS_INVAL;
    uint32_t socket_count = 0;
    auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (ret != AMDSMI_STATUS_SUCCESS) return ret;
    uint32_t current_device_index = 0;
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t device_count = 0;
        amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
        std::vector<amdsmi_processor_handle> handles(device_count);
        amdsmi_get_processor_handles(sockets[i], &device_count, handles.data());
        for (uint32_t j = 0; j < device_count; j++) {
            if (current_device_index == device_index) {
                *processor_handle = handles[j];
                return AMDSMI_STATUS_SUCCESS;
            }
            current_device_index++;
        }
    }
    return AMDSMI_STATUS_API_FAILED;
}

namespace amd::smi {

bool is_vm_guest() {
  const std::string hypervisor = "hypervisor";
  std::string line;
  std::ifstream infile("/proc/cpuinfo");
  if (infile.fail()) {
    return false;
  }
  while (std::getline(infile, line)) {
    if (line.find(hypervisor) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool is_sudo_user() {
  auto myUID = getuid();
  auto myPrivledges = geteuid();
  return (myUID == myPrivledges) && (myPrivledges == 0);
}

} // namespace amd::smi
