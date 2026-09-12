// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace agent
{
// Pairing of rocprofiler agent records with HSA agents.
//
// Split out of construct_agent_cache() so the decision can be exercised without
// an HSA runtime, a GPU or a DXG thunk: everything below is a pure function of
// two lists of integers.

// How an HSA agent is keyed to a rocprofiler agent, and what an unmatched HSA
// agent means.
enum class mapping_policy
{
    // Bare metal. Key on rocprofiler's dense logical_node_id, which the KFD
    // enumeration order makes equal to the driver node id. Any HSA agent left
    // without a partner is an internal inconsistency.
    //
    // (The key is historical - see the TODO(aelwazir) in construct_agent_cache()
    // about ROCR reporting real node ids - and is preserved exactly.)
    strict = 0,

    // WSL. Key on the real KMT node id the DXG thunk reported, which the agent
    // records carry in node_id. rocprofiler and HSA read the topology through
    // the same thunk but rocprofiler drops adapters it cannot fully describe,
    // so HSA legitimately sees GPUs rocprofiler does not. That is an
    // unsupported-profiling condition for those GPUs, not a bug.
    wsl,
};

// The fields of a rocprofiler_agent_t this decision depends on.
struct mapping_agent_view
{
    int64_t  logical_node_id = 0;
    uint32_t node_id         = 0;
    bool     is_gpu          = false;
};

// The fields of an hsa_agent_t this decision depends on. `has_node_id` is false
// when HSA_AMD_AGENT_INFO_DRIVER_NODE_ID could not be queried, which leaves the
// agent unpairable under either policy.
struct mapping_hsa_view
{
    uint32_t driver_node_id = 0;
    bool     has_node_id    = false;
    bool     is_gpu         = false;
};

struct agent_mapping_result
{
    struct pair
    {
        size_t rocp_index = 0;
        size_t hsa_index  = 0;
        // Key construct_agent_cache() files this pair under. Equal to the
        // driver node id, which under the strict policy is also the matched
        // agent's logical_node_id.
        uint32_t key = 0;
    };

    std::vector<pair> pairs;

    // HSA agents with no rocprofiler counterpart, split so the caller can say
    // "GPU profiling is unavailable for these" without lumping in a CPU.
    std::vector<uint32_t> unmatched_gpu_node_ids;
    std::vector<uint32_t> unmatched_other_node_ids;

    // HSA agents whose driver node id could not be read at all. Unpairable
    // under every policy and always a hard error.
    size_t unqueryable_count = 0;

    bool complete() const
    {
        return unmatched_gpu_node_ids.empty() && unmatched_other_node_ids.empty() &&
               unqueryable_count == 0;
    }
};

// Pair every HSA agent with at most one rocprofiler agent. Each rocprofiler
// agent is claimed once, so two HSA agents reporting the same driver node id
// leave the second unmatched rather than both aliasing onto one record.
agent_mapping_result
compute_agent_mapping(const std::vector<mapping_agent_view>& rocp_agents,
                      const std::vector<mapping_hsa_view>&   hsa_agents,
                      mapping_policy                         policy);
}  // namespace agent
}  // namespace rocprofiler
