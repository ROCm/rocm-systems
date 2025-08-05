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
#include <mutex>
#include <stdexcept>
#include <shared_mutex>
#include <array>
#include <unordered_map>

#include "util/hsa_rsrc_factory.h"
#include "util/reg_offsets.h"
#include "ip_offset_table_init.h"

// Pair of pcie domain, bdf
using domain_bdf_t = std::pair<uint32_t, uint32_t>;

// Hash function for domain_bdf_t
template <>
struct std::hash<domain_bdf_t> {
  std::size_t operator()(const domain_bdf_t& key) const {
    return std::hash<uint32_t>()(key.first) ^ (std::hash<uint32_t>()(key.second) << 1);
  }
};

// Map from (Domain, BDF) to reg_base_offset_table*
using reg_base_offset_table_cache = std::unordered_map<domain_bdf_t, const reg_base_offset_table*>;

class locked_ip_offset_table_cache {
 public:
  const reg_base_offset_table* get(const AgentInfo* agent_info) {
    {
      std::shared_lock lock{mutex};
      auto it = cache.find(std::make_pair(agent_info->domain, agent_info->bdf_id));
      if (it != cache.end()) return it->second;
    }
    {
      std::string_view gfxip(agent_info->gfxip);
      std::unique_lock lock{mutex};
      const reg_base_offset_table* table = nullptr;

      if (auto gfxip_prefix = gfxip.substr(0, 4); gfxip_prefix == "gfx9")
        table = vega20_reg_base_init();
      else {
        if (auto gfxip_prefix = gfxip.substr(0, 5);
            gfxip_prefix == "gfx10" || gfxip_prefix == "gfx11" || gfxip_prefix == "gfx12") {
          table = navi_ip_offset_table_discovery_sysfs(agent_info->domain, agent_info->bdf_id);
          if (!table) table = sienna_cichlid_reg_base_init();
        }
      }

      if (table) cache.emplace(std::make_pair(agent_info->domain, agent_info->bdf_id), table);
      return table;
    }
  }

  static locked_ip_offset_table_cache& get_instance() {
    // Note: never cleanup, keep in memory to prevent issue with global destructor
    static auto* cache = new locked_ip_offset_table_cache{};
    return *cache;
  }

 private:
  std::shared_mutex mutex;
  reg_base_offset_table_cache cache;
};

// acquire the IP offset table for the device using the domain and bdf_id
const reg_base_offset_table* acquire_ip_offset_table(const AgentInfo* agent_info) {
  auto ip_offset_table = locked_ip_offset_table_cache::get_instance().get(agent_info);
  if (ip_offset_table == nullptr) {
    throw std::runtime_error(
        "Failed to acquire the IP offset table for the device. Possible reasons include:\n"
        "  1. Incorrect or incomplete ROCm setup. Please verify your installation.\n"
        "  2. The device is not supported.\n"
        "  3. An internal error or bug.\n");
  }
  return ip_offset_table;
}