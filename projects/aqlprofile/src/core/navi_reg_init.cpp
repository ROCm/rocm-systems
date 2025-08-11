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

#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <shared_mutex>
#include "ip_discovery.h"

#define __maybe_unused __attribute__((__unused__))

#include "linux/registers/sienna_cichlid_ip_offset.h"
#include "util/reg_offsets.h"

#define LOG_VERBOSE 0

namespace {
void LogErrors(std::string msg) {
#if LOG_VERBOSE
  std::cerr << msg << std::endl;
#endif /* LOG_VERBOSE */
}
}  // namespace

const reg_base_offset_table* sienna_cichlid_reg_base_init() {
  static_assert(HWIP_MAX_INSTANCE >= MAX_INSTANCE,
                "HWIP_MAX_INSTANCE must be greater than MAX_INSTANCE");
  static_assert(HWIP_MAX_SEGMENT >= MAX_SEGMENT,
                "HWIP_MAX_SEGMENT must be greater than MAX_SEGMENT");

  static const auto* sienna_cichlid_reg_table = []() {
    auto* reg_table = new reg_base_offset_table();

    // helper lambda to initialize blocks
    auto init_hwip = [&](amd_hw_ip_block_type hwip, const auto& base) {
      for (uint32_t i = 0; i < MAX_INSTANCE; ++i) {
        std::copy(std::begin(base.instance[i].segment), std::end(base.instance[i].segment),
                  std::begin(reg_table->reg_offset[hwip][i]));
      }
    };

    // HW has more IP blocks,  only initialize the blocks needed
    init_hwip(GC_HWIP, GC_BASE);
    init_hwip(ATHUB_HWIP, ATHUB_BASE);
    return reg_table;
  }();

  return sienna_cichlid_reg_table;
}

const reg_base_offset_table* navi_ip_offset_table_discovery_sysfs(uint32_t domain, uint32_t bdf) {
  // Read the drm device properties, which includes all the IP base offsets for a GPU card on the
  // system.
  discovery_table_t table;
  try {
    table = parse_ip_discovery(domain, bdf);
  } catch (const std::exception& e) {
    LogErrors("Error in IP discovery for domain=" + std::to_string(domain) +
              " bdf=" + std::to_string(bdf) + ": \n" + e.what());
    return nullptr;
  }

  // Note: never cleanup, keep in memory to prevent issue with global destructor
  struct reg_base_offset_table* reg_table = new reg_base_offset_table();

  // helper lambda to initialize blocks
  auto init_hwip = [&](amd_hw_ip_block_type hwip, const auto& entry) {
    std::copy(std::begin(entry.segments), std::end(entry.segments),
              std::begin(reg_table->reg_offset[hwip][entry.instance]));
  };

  for (auto& entry : table) {
    if (entry.ipname == "gc") {
      init_hwip(GC_HWIP, entry);
    } else if (entry.ipname == "athub") {
      init_hwip(ATHUB_HWIP, entry);
    }
  }

  return reg_table;
}
