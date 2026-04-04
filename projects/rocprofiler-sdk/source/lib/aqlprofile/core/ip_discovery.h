// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_IP_DISCOVERY_H_
#define SRC_CORE_IP_DISCOVERY_H_

#include <array>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

#include "util/reg_offsets.h"

using base_addr_segments_t = std::array<uint32_t, HWIP_MAX_SEGMENT>;

// Represents a single entry in the discovery table, containing information about a specific IP
// block.
struct discovery_table_entry_t {
  int die{0};                       // Die index
  base_addr_segments_t segments{};  // Base address segments
  int major{0};                     // Major version of the IP
  int minor{0};                     // Minor version of the IP
  int revision{0};                  // Revision number of the IP
  int instance{0};                  // Instance ID of the IP
  std::string ipname{};             // Name of the IP block
};

using discovery_table_t = std::vector<discovery_table_entry_t>;
discovery_table_t parse_ip_discovery(uint32_t domain, uint32_t bdf);

#endif