// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"

#include <fmt/core.h>
#include <hsa/hsa_ext_amd.h>
#include "glog/logging.h"

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto status = (fn);                                                                        \
        if(status != HSA_STATUS_SUCCESS)                                                           \
        {                                                                                          \
            ROCP_FATAL << "HSA Err: " << status << "\n";                                           \
            exit(1);                                                                               \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace aql
{
CounterPacketConstruct::CounterPacketConstruct(rocprofiler_agent_id_t               agent,
                                               const std::vector<counters::Metric>& metrics)
: _agent(agent)
{
    // Validate that the counter exists and construct the block instances
    // for the counter.
    for(const auto& x : metrics)
    {
        auto query_info                = get_query_info(_agent, x);
        _metrics.emplace_back().metric = x;
        uint64_t event_id              = 0;
        if(!x.event().empty()) event_id = std::stoul(x.event(), nullptr);
        ROCP_TRACE << fmt::format("Fetching events for counter {} (id={}, instance_count={}) on "
                                  "agent {} (node-id:{})(name:{})",
                                  x.name(),
                                  event_id,
                                  query_info.instance_count,
                                  agent.handle,
                                  rocprofiler::agent::get_agent(agent)->node_id,
                                  rocprofiler::agent::get_agent(agent)->name);

        for(unsigned block_index = 0; block_index < query_info.instance_count; ++block_index)
        {
            _metrics.back().instances.push_back(
                {.block_index = block_index,
                 .event_id    = static_cast<uint32_t>(event_id & 0xFFFFFFFF),
                 .flags       = aqlprofile_pmc_event_flags_t{x.flags()},
                 .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id)});

            _metrics.back().events.push_back(
                {.block_index = block_index,
                 .event_id    = static_cast<uint32_t>(event_id & 0xFFFFFFFF),
                 .flags       = aqlprofile_pmc_event_flags_t{x.flags()},
                 .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id)});

            bool validate_event_result;

            auto aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent));

            LOG_IF(FATAL,
                   aqlprofile_validate_pmc_event(aql_agent,
                                                 &_metrics.back().events.back(),
                                                 &validate_event_result) != HSA_STATUS_SUCCESS);
            ROCP_FATAL_IF(!validate_event_result)
                << "Invalid Metric: " << block_index << " " << event_id;
            _event_to_metric[_metrics.back().events.back()] = x;
        }
    }
    _events = get_all_events();
}

std::unique_ptr<hsa::CounterAQLPacket>
CounterPacketConstruct::construct_packet(const CoreApiTable& coreapi, const AmdExtTable& ext)
{
    const auto* agent =
        rocprofiler::agent::get_agent_cache(CHECK_NOTNULL(rocprofiler::agent::get_agent(_agent)));
    if(!agent) ROCP_FATAL << "No agent cache for agent id: " << _agent.handle;

    hsa_amd_memory_pool_access_t _access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    ext.hsa_amd_agent_memory_pool_get_info_fn(agent->get_hsa_agent(),
                                              agent->kernarg_pool(),
                                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                              static_cast<void*>(&_access));

    hsa::CounterAQLPacket::CounterMemoryPool pool;

    if(_access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) pool.bIgnoreKernArg = true;

    pool.allocate_fn     = ext.hsa_amd_memory_pool_allocate_fn;
    pool.allow_access_fn = ext.hsa_amd_agents_allow_access_fn;
    pool.free_fn         = ext.hsa_amd_memory_pool_free_fn;
    pool.api_copy_fn     = coreapi.hsa_memory_copy_fn;
    pool.fill_fn         = ext.hsa_amd_memory_fill_fn;

    pool.gpu_agent     = agent->get_hsa_agent();
    pool.cpu_pool_     = agent->cpu_pool();
    pool.kernarg_pool_ = agent->kernarg_pool();

    const auto* aql_agent = rocprofiler::agent::get_aql_agent(agent->get_rocp_agent()->id);
    if(aql_agent == nullptr) throw std::runtime_error("Could not get AQL agent!");

    if(_events.empty()) ROCP_TRACE << "No events for pkt";

    return std::make_unique<hsa::CounterAQLPacket>(*aql_agent, pool, _events);
}

ThreadTraceAQLPacketFactory::ThreadTraceAQLPacketFactory(const hsa::AgentCache&             agent,
                                                         const thread_trace_parameter_pack& params,
                                                         const CoreApiTable&                coreapi,
                                                         const AmdExtTable&                 ext)
{
    this->tracepool                 = hsa::TraceMemoryPool{};
    this->tracepool.allocate_fn     = ext.hsa_amd_memory_pool_allocate_fn;
    this->tracepool.allow_access_fn = ext.hsa_amd_agents_allow_access_fn;
    this->tracepool.free_fn         = ext.hsa_amd_memory_pool_free_fn;
    this->tracepool.api_copy_fn     = coreapi.hsa_memory_copy_fn;
    this->tracepool.gpu_agent       = agent.get_hsa_agent();
    this->tracepool.cpu_pool_       = agent.cpu_pool();
    this->tracepool.gpu_pool_       = agent.gpu_pool();

    uint32_t cu                 = static_cast<uint32_t>(params.target_cu);
    uint32_t shader_engine_mask = static_cast<uint32_t>(params.shader_engine_mask);
    uint32_t simd               = static_cast<uint32_t>(params.simd_select);
    uint32_t buffer_size_lo     = static_cast<uint32_t>(params.buffer_size);
    uint32_t buffer_size_hi     = static_cast<uint32_t>(params.buffer_size >> 32);
    uint32_t perf_ctrl          = static_cast<uint32_t>(params.perfcounter_ctrl);

    aql_params.clear();

    aql_params.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, {cu}});
    aql_params.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, {shader_engine_mask}});
    aql_params.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION, {simd}});
    aql_params.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, {buffer_size_lo}});

    if(buffer_size_hi != 0) aql_params.push_back({static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(
        AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH), {buffer_size_hi}});

    if(perf_ctrl != 0 && !params.perfcounters.empty())
    {
        for(const auto& perf_counter : params.perfcounters)
        {
            aqlprofile_att_parameter_t param{};
            param.parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME;
            param.counter_id     = perf_counter.first;
            param.simd_mask      = perf_counter.second;
            aql_params.push_back(param);
        }

        aqlprofile_att_parameter_t param{};
        param.parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL;
        param.value          = perf_ctrl - 1;
        aql_params.push_back(param);
    }
}

std::unique_ptr<hsa::TraceControlAQLPacket>
ThreadTraceAQLPacketFactory::construct_control_packet()
{
    auto num_params = static_cast<uint32_t>(aql_params.size());
    auto profile    = aqlprofile_att_profile_t{tracepool.gpu_agent, aql_params.data(), num_params};
    auto packet     = std::make_unique<hsa::TraceControlAQLPacket>(this->tracepool, profile);
    packet->clear();
    return packet;
}

std::unique_ptr<hsa::CodeobjMarkerAQLPacket>
ThreadTraceAQLPacketFactory::construct_load_marker_packet(uint64_t id, uint64_t addr, uint64_t size)
{
    return std::make_unique<hsa::CodeobjMarkerAQLPacket>(tracepool, id, addr, size, false, false);
}

std::unique_ptr<hsa::CodeobjMarkerAQLPacket>
ThreadTraceAQLPacketFactory::construct_unload_marker_packet(uint64_t id)
{
    return std::make_unique<hsa::CodeobjMarkerAQLPacket>(tracepool, id, 0, 0, false, true);
}

std::vector<aqlprofile_pmc_event_t>
CounterPacketConstruct::get_all_events() const
{
    std::vector<aqlprofile_pmc_event_t> ret;
    for(const auto& metric : _metrics)
    {
        ret.insert(ret.end(), metric.instances.begin(), metric.instances.end());
    }
    return ret;
}

const counters::Metric*
CounterPacketConstruct::event_to_metric(const aqlprofile_pmc_event_t& event) const
{
    if(const auto* ptr = rocprofiler::common::get_val(_event_to_metric, event))
    {
        return ptr;
    }
    return nullptr;
}

const std::vector<aqlprofile_pmc_event_t>&
CounterPacketConstruct::get_counter_events(const counters::Metric& metric) const
{
    for(const auto& prof_metric : _metrics)
    {
        if(prof_metric.metric.id() == metric.id())
        {
            return prof_metric.events;
        }
    }
    throw std::runtime_error(fmt::format("Cannot Find Events for {}", metric));
}

rocprofiler_status_t
CounterPacketConstruct::can_collect()
{
    // Verify that the counters fit within harrdware limits
    std::map<std::pair<hsa_ven_amd_aqlprofile_block_name_t, uint32_t>, int64_t> counter_count;
    std::map<std::pair<hsa_ven_amd_aqlprofile_block_name_t, uint32_t>, int64_t> max_allowed;
    for(auto& metric : _metrics)
    {
        for(auto& instance : metric.events)
        {
            auto block_pair       = std::make_pair(instance.block_name, instance.block_index);
            auto [iter, inserted] = counter_count.emplace(block_pair, 0);
            iter->second++;
            if(inserted)
            {
                max_allowed.emplace(block_pair, get_block_counters(_agent, instance));
            }
        }
    }

    // Check if the block count > max count
    for(auto& [block_name, count] : counter_count)
    {
        if(auto* max = CHECK_NOTNULL(common::get_val(max_allowed, block_name)); count > *max)
        {
            return ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT;
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace aql
}  // namespace rocprofiler
