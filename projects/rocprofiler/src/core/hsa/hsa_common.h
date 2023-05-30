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

#ifndef SRC_CORE_HSA_HSA_COMMON_H_
#define SRC_CORE_HSA_HSA_COMMON_H_

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <map>
#include <mutex>

#include "rocprofiler.h"
#include "src/core/hardware/hsa_info.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace rocprofiler {
namespace hsa_support {


std::vector<hsa_agent_t>& GetCPUAgentList();

Agent::AgentInfo& GetAgentInfo(decltype(hsa_agent_t::handle) handle);
void SetAgentInfo(decltype(hsa_agent_t::handle) handle, const Agent::AgentInfo& agent_info);
hsa_agent_t GetAgentByIndex(uint64_t agent_index);

CoreApiTable& GetCoreApiTable();
void SetCoreApiTable(const CoreApiTable& table);

AmdExtTable GetAmdExtTable();
void SetAmdExtTable(AmdExtTable* table);

hsa_ven_amd_loader_1_01_pfn_t GetHSALoaderApi();
void SetHSALoaderApi();

void ResetMaps();

rocprofiler_timestamp_t GetCurrentTimestampNS();

}  // namespace hsa_support
}  // namespace rocprofiler

#endif  // SRC_CORE_HSA_HSA_COMMON_H_
