////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2021-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIESd OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/util/flag.h"
#include "core/util/utils.h"
#include "core/util/os.h"

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <locale>

namespace rocr {
FILE* log_file = stderr;
uint8_t log_flags[8];

void log_printf(const char* file, int line, const char* format, ...) {
    va_list ap;
    std::stringstream str_thrd_id;
    str_thrd_id << std::hex << std::this_thread::get_id();
    va_start(ap, format);
    char message[4096];
    vsnprintf(message, sizeof(message), format, ap);
    va_end(ap);
    fprintf(log_file, ":%-25s:%-4d: %010lld us: [pid:%-5d tid:0x%s] [***rocr***] %s\n",
            file, line, os::ReadAccurateClock()/1000ULL, os::GetProcessId(),
            str_thrd_id.str().c_str(), message);
    fflush(log_file);
}

// split at separators
static std::vector<std::string> split(std::string& str, char sep) {
  std::vector<std::string> ret;
  while (!str.empty()) {
    size_t pos = str.find(sep);
    if (pos == std::string::npos) {
      ret.push_back(str);
      return ret;
    }
    ret.push_back(str.substr(0, pos));
    str.erase(0, pos + 1);
  }
  return ret;
};

// Parse id,id-id,... strings into id lists
static std::vector<uint32_t> get_elements(std::string& str, uint32_t maxElement) {
  std::vector<uint32_t> ret;
  MAKE_NAMED_SCOPE_GUARD(error, [&]() { ret.clear(); });

  std::vector<std::string> ranges = split(str, ',');
  for (auto& str : ranges) {
    auto range = split(str, '-');
    // failure, too many -'s.
    if (range.size() > 2) return ret;

    char* end;
    uint32_t index = strtoul(range[0].c_str(), &end, 10);
    // Invalid syntax - id's must be base 10 digits only.
    if (*end != '\0') return ret;
    if (index <= maxElement) ret.push_back(index);

    if (range.size() == 2) {
      uint32_t secondindex = strtoul(range[1].c_str(), &end, 10);
      if (*end != '\0') return ret;         // bad syntax
      if (secondindex < index) return ret;  // inverted range
      secondindex = Min(secondindex, maxElement);
      for (uint32_t i = index + 1; i < secondindex + 1; i++) ret.push_back(i);
    }
  }

  // Confirm no duplicate ids.
  std::sort(ret.begin(), ret.end());
  if (std::adjacent_find(ret.begin(), ret.end()) != ret.end()) return ret;

  // Good parse, keep result.
  error.Dismiss();
  return ret;
};

/*
Parse env var per the following syntax, all whitespace is ignored:

ID = [0-9][0-9]*                         ex. base 10 numbers
ID_list = (ID | ID-ID)[, (ID | ID-ID)]*  ex. 0,2-4,7
GPU_list = ID_list                       ex. 0,2-4,7
CU_list = 0x[0-F]* | ID_list             ex. 0x337F OR 0,2-4,7
CU_Set = GPU_list : CU_list              ex. 0,2-4,7:0-15,32-47 OR 0,2-4,7:0x337F
HSA_CU_MASK =  CU_Set [; CU_Set]*        ex. 0,2-4,7:0-15,32-47; 3-9:0x337F

GPU indexes are taken post ROCR_VISIBLE_DEVICES reordering.
Listed or bit set CUs will be enabled at queue creation on the associated GPU.
All other CUs on the associated GPUs will be disabled.
CU masks of unlisted GPUs are not restricted.

Repeating a GPU or CU ID is a syntax error.
Parsing stops at the first CU_Set that has a syntax error, that set and all
following sets are ignored.
Specifying a mask with no usable CUs (CU_list is 0x0) is a syntax error.
Users should use ROCR_VISIBLE_DEVICES if they want to exclude use of a
particular GPU.
*/
void Flag::parse_masks(std::string& var, uint32_t maxGpu, uint32_t maxCU) {
  if (var.empty()) return;

  // Remove whitespace
  auto end = std::remove_if(var.begin(), var.end(),
                            [](char c) { return std::isspace<char>(c, std::locale::classic()); });
  var.erase(end, var.end());

  // Switch to uppercase
  for (auto& c : var) c = toupper(c);

  // Iterate over cu sets
  auto sets = split(var, ';');
  for (auto& set : sets) {
    auto parts = split(set, ':');
    if (parts.size() != 2) return;

    // temp storage for cu_set parsing.
    std::vector<uint32_t> gpu_index;
    std::vector<uint32_t> mask;

    // parse cu list first, check for bitmask format
    if (parts[1][1] == 'X') {
      // Confirm hex format and strip prefix
      auto& cu = parts[1];
      if (cu[0] != '0') return;
      cu.erase(0, 2);

      // Ensure all valid hex characters
      for (auto& c : cu) {
        if (!isxdigit(c)) return;
      }

      // Convert to uint32_t, lsb first.
      size_t len = cu.length();
      while (len != 0) {
        size_t trim = Min(len, size_t(8));
        len -= trim;
        auto tmp = cu.substr(len, trim);
        auto chunk = stoul(tmp, nullptr, 16);
        mask.push_back(chunk);
      }

      // Trim dwords beyond maxCUs
      uint32_t maxDwords = maxCU / 32 + 1;
      if (maxDwords < mask.size()) mask.resize(maxDwords);

      // Trim leading zeros
      while (!mask.empty() && mask.back() == 0) mask.pop_back();

      // Mask 0x0 is an error.
      if (mask.empty()) return;

    } else {
      // parse cu lists
      auto cu_indices = get_elements(parts[1], maxCU);
      if (cu_indices.empty()) return;
      uint32_t maxdword = cu_indices.back() / 32 + 1;
      mask.resize(maxdword, 0);
      for (auto id : cu_indices) {
        uint32_t index, offset;
        index = id / 32;
        offset = id % 32;
        mask[index] |= 1ul << offset;
      }
    }

    // parse device list
    gpu_index = get_elements(parts[0], maxGpu);
    if (gpu_index.empty()) return;

    // Ensure that no GPU was repeated across cu_sets
    for (auto id : gpu_index) {
      if (cu_mask_.find(id) != cu_mask_.end()) return;
    }

    // Insert into map
    for (auto id : gpu_index) {
      cu_mask_[id] = mask;
    }
  }
}

}  // namespace rocr
