// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <rocprofiler/agent.h>
#include "lib/common/utility.hpp"

// Construct const and non-const accessor functions
#define CONST_NONCONST_ACCESSOR(RTYPE, NAME, VAL)                                                  \
    const RTYPE& NAME() const { return VAL; }                                                      \
    RTYPE&       NAME() { return VAL; }

namespace rocprofiler
{
namespace hsa
{
static const uint32_t LDS_BLOCK_SIZE = 128 * 4;

// Stores per-agent HSA information such as GPU and Kernel pools
// along with nearest CPU agent and its pool. Links rocprofiler_agent_t
// to its HSA agent. Note this class is only valid when HSA is
// init'd
class AgentCache
{
public:
    AgentCache(rocprofiler_agent_t, size_t index, const ::CoreApiTable&, const AmdExtTable&);

    // Provides const and a non-const accessor functions.
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, cpu_pool, _cpu_pool);
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, kernarg_pool, _kernarg_pool);
    CONST_NONCONST_ACCESSOR(hsa_amd_memory_pool_t, gpu_pool, _gpu_pool);
    CONST_NONCONST_ACCESSOR(rocprofiler_agent_t, agent_t, _agent_t);
    CONST_NONCONST_ACCESSOR(hsa_agent_t, get_agent, _agent);
    CONST_NONCONST_ACCESSOR(hsa_agent_t, near_cpu, _nearest_cpu);

    const std::string& name() const { return _name; }

private:
    // Agent info
    rocprofiler_agent_t _agent_t;
    size_t              _index{0};  // rocprofiler_agent index

    // GPU Agent
    hsa_agent_t _agent{.handle = 0};
    hsa_agent_t _nearest_cpu{.handle = 0};

    // memory pools
    hsa_amd_memory_pool_t _cpu_pool{.handle = 0};
    hsa_amd_memory_pool_t _kernarg_pool{.handle = 0};
    hsa_amd_memory_pool_t _gpu_pool{.handle = 0};

    std::string _name;
};

}  // namespace hsa
}  // namespace rocprofiler
