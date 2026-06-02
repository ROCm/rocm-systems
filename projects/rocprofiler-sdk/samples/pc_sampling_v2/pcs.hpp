// MIT License
//
// Copyright (c) 2023-2026 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace client
{
namespace pcs
{
constexpr size_t   BUFFER_SIZE_BYTES   = 4 * 1024 * 1024;  // 4 MB
constexpr size_t   WATERMARK           = (BUFFER_SIZE_BYTES * 3 / 4);
constexpr uint64_t HOST_TRAP_INTERVAL  = 10000;    // 10000 us
constexpr uint64_t STOCHASTIC_INTERVAL = 1048576;  // 2^20 cycles

struct tool_agent_info;
using ext_fields_vec_t     = std::vector<rocprofiler_pc_sampling_snapshot_ext_field_id_t>;
using ext_field_name_map_t = std::map<rocprofiler_pc_sampling_snapshot_ext_field_id_t, std::string>;
using tool_agent_info_vec_t       = std::vector<std::unique_ptr<tool_agent_info>>;
using pc_sampling_buffer_id_vec_t = std::vector<rocprofiler_buffer_id_t>;

struct tool_agent_info
{
    rocprofiler_agent_id_t     agent_id;
    const rocprofiler_agent_t* agent;

    /// Snapshot ext_data fields supported by this GPU agent (queried per-agent).
    std::unique_ptr<ext_fields_vec_t> ext_fields;
    /// Memoized mapping from ext_data field ID to its name string.
    /// Populated once during query_snapshot_ext_fields_for_agent to avoid
    /// repeated name lookups in the buffer callback hot path.
    ext_field_name_map_t ext_field_names;

    /// Most comprehensive PC sampling configuration discovered during the query phase.
    /// Populated by query_most_comprehensive_config_for_agent(); consumed by
    /// configure_pc_sampling_for_agent().
    rocprofiler_pc_sampling_record_kind_t most_comprehensive_record_kind =
        ROCPROFILER_PC_SAMPLING_RECORD_NONE;
    rocprofiler_pc_sampling_api_flags_t most_comprehensive_api_flags =
        ROCPROFILER_PC_SAMPLING_API_FLAG_NONE;
    rocprofiler_pc_sampling_configuration_v2_t most_comprehensive_config = {};
};

// GPU agents supporting some kind of PC sampling.
// Heap-allocated in init() and released in fini() (mirrors get_pc_sampling_buffer_ids()) to avoid
// static-destruction-order issues.
tool_agent_info_vec_t*
get_gpu_agents();

// Must be called first (prior to any other function from this namespace)
void
init();

// Must be called at the end of the `tool_fini`
void
fini();

// Ids of the buffers used as containers for PC sampling records
pc_sampling_buffer_id_vec_t*
get_pc_sampling_buffer_ids();

void
find_all_gpu_agents_supporting_pc_sampling();

/**
 * @brief Discover the most comprehensive PC sampling record version for the agent.
 *
 * Iterates record versions from the most comprehensive (LAST-1) down to V0,
 * querying via the v2 API with PREFER_STOCHASTIC.  The runtime falls back to
 * host-trap automatically if stochastic is not available on the agent.
 *
 * On success the chosen record kind, API flags, and configuration (unit,
 * min/max interval) are memoized in @p agent_info->most_comprehensive_*.
 *
 * @return true if a usable configuration was found.
 */
bool
query_most_comprehensive_config_for_agent(tool_agent_info* agent_info);

/**
 * @brief Query snapshot ext_data fields supported by the agent and store them
 * in agent_info->ext_fields.
 *
 * Implicitly operates on agent_info->most_comprehensive_record_kind, i.e. the most comprehensive
 * record version discovered during the query phase. Must be called after
 * query_most_comprehensive_config_for_agent() has populated that field. Record kinds that carry no
 * snapshot information (host-trap suitable records V0/V1/V3) yield an empty field list, which is
 * handled gracefully.
 */
void
query_snapshot_ext_fields_for_agent(tool_agent_info* agent_info);

/**
 * @brief Configure PC sampling for the agent using the memoized most comprehensive config.
 *
 * Must be called after query_most_comprehensive_config_for_agent() has populated the
 * agent_info->most_comprehensive_* fields.
 */
void
configure_pc_sampling_for_agent(tool_agent_info*         agent_info,
                                rocprofiler_context_id_t context_id,
                                rocprofiler_buffer_id_t  buffer_id);

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t      context_id,
                                 rocprofiler_buffer_id_t       buffer_id,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void*                         data,
                                 uint64_t                      drop_count);
}  // namespace pcs
}  // namespace client
