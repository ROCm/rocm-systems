/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hsa_common.h"

#include "src/utils/exception.h"

namespace rocmtools {

namespace hsa_support {

std::mutex agents_map_lock;
std::map<decltype(hsa_agent_t::handle), Agent::AgentInfo> agent_info_map;
Agent::AgentInfo& GetAgentInfo(decltype(hsa_agent_t::handle) handle) {
  std::lock_guard<std::mutex> lock(agents_map_lock);
  if (agent_info_map.find(handle) != agent_info_map.end())
    return agent_info_map.at(handle);
  else
    throw(std::string("Error: Can't find Agent with handle(") + std::to_string(handle) +
          ") in this system");
}
void SetAgentInfo(decltype(hsa_agent_t::handle) handle, const Agent::AgentInfo& agent_info) {
  std::lock_guard<std::mutex> lock(agents_map_lock);
  agent_info_map.emplace(handle, agent_info);
}

hsa_agent_t GetAgentByIndex(int agent_index) {
  std::lock_guard<std::mutex> lock(agents_map_lock);
  for (auto& agent_info : agent_info_map) {
    if (agent_info.second.getIndex() == agent_index) {
      return hsa_agent_t{agent_info.second.getHandle()};
    }
  }
  throw(std::string("Error: Can't find Agent with Index(") + std::to_string(agent_index) +
        ") in this system");
}

CoreApiTable saved_core_api{};
CoreApiTable& GetCoreApiTable() { return saved_core_api; }
void SetCoreApiTable(const CoreApiTable& table) { saved_core_api = table; }

AmdExtTable saved_amd_ext_api{};
AmdExtTable GetAmdExtTable() { return saved_amd_ext_api; }
void SetAmdExtTable(AmdExtTable* table) { saved_amd_ext_api = *table; }

hsa_ven_amd_loader_1_01_pfn_t hsa_loader_api{};
hsa_ven_amd_loader_1_01_pfn_t GetHSALoaderApi() { return hsa_loader_api; }
void SetHSALoaderApi() {
  hsa_status_t status = saved_core_api.hsa_system_get_major_extension_table_fn(
      HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t), &hsa_loader_api);

  if (status != HSA_STATUS_SUCCESS) fatal("hsa_system_get_major_extension_table failed");
}

void ResetMaps() {
  if (hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(false);
      status != HSA_STATUS_SUCCESS)
    assert(!"hsa_amd_profiling_async_copy_enable failed");
  memset(&saved_core_api, '\0', sizeof(saved_core_api));
  memset(&saved_amd_ext_api, '\0', sizeof(saved_amd_ext_api));
  memset(&hsa_loader_api, '\0', sizeof(hsa_loader_api));
}

rocprofiler_timestamp_t GetCurrentTimestampNS() {
  // If the HSA intercept is installed, then use the "original"
  // 'hsa_system_get_info' function to avoid reporting calls for internal use
  // of the HSA API by the tracer.
  auto hsa_system_get_info_fn = saved_core_api.hsa_system_get_info_fn;

  // If the HSA intercept is not installed, use the default
  // 'hsa_system_get_info'.
  if (hsa_system_get_info_fn == nullptr) hsa_system_get_info_fn = hsa_system_get_info;

  uint64_t sysclock;
  if (hsa_status_t status = hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
      status == HSA_STATUS_ERROR_NOT_INITIALIZED)
    return rocprofiler_timestamp_t{0};
  else if (status != HSA_STATUS_SUCCESS)
    assert(!"hsa_system_get_info failed");

  static uint64_t sysclock_period = [&]() {
    uint64_t sysclock_hz = 0;
    if (hsa_status_t status =
            hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
        status != HSA_STATUS_SUCCESS)
      assert(!"hsa_system_get_info failed");

    return (uint64_t)1000000000 / sysclock_hz;
  }();

  return rocprofiler_timestamp_t{sysclock * sysclock_period};
}

}  // namespace hsa_support
}  // namespace rocmtools
