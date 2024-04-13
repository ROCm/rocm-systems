// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"

#include <fmt/core.h>
#include <hsa/hsa_ext_amd.h>
#include "glog/logging.h"

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto status = (fn);                                                                        \
        if(status != HSA_STATUS_SUCCESS)                                                           \
        {                                                                                          \
            std::cerr << "HSA Err: " << status << "\n";                                            \
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
        uint32_t event_id              = std::atoi(x.event().c_str());

        ROCP_TRACE << fmt::format(
            "Fetching events for counter {} (id={}, instance_count={}) on agent {} (name:{})",
            x.name(),
            event_id,
            query_info.instance_count,
            agent.handle,
            rocprofiler::agent::get_agent(agent)->name);

        for(unsigned block_index = 0; block_index < query_info.instance_count; ++block_index)
        {
            _metrics.back().instances.push_back(
                {static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id),
                 block_index,
                 event_id});

            _metrics.back().events.push_back(
                {.block_index = block_index,
                 .event_id    = event_id,
                 .flags       = aqlprofile_pmc_event_flags_t{0},
                 .block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id)});

            bool validate_event_result;

            auto aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent));

            LOG_IF(FATAL,
                   aqlprofile_validate_pmc_event(aql_agent,
                                                 &_metrics.back().events.back(),
                                                 &validate_event_result) != HSA_STATUS_SUCCESS);
            LOG_IF(FATAL, !validate_event_result)
                << "Invalid Metric: " << block_index << " " << event_id;
            _event_to_metric[std::make_tuple(
                static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id),
                block_index,
                event_id)] = x;
        }
    }
    // Check that we can collect all of the metrics in a single execution
    // with a single AQL packet
    can_collect();
    _events = get_all_events();
}

std::unique_ptr<hsa::CounterAQLPacket>
CounterPacketConstruct::construct_packet(const AmdExtTable& ext)
{
    auto  pkt_ptr = std::make_unique<hsa::CounterAQLPacket>(ext.hsa_amd_memory_pool_free_fn);
    auto& pkt     = *pkt_ptr;
    if(_events.empty())
    {
        ROCP_TRACE << "No events for pkt";
        return pkt_ptr;
    }
    pkt.empty = false;

    const auto* agent_cache =
        rocprofiler::agent::get_agent_cache(CHECK_NOTNULL(rocprofiler::agent::get_agent(_agent)));
    if(!agent_cache)
    {
        ROCP_FATAL << "No agent cache for agent id: " << _agent.handle;
    }

    pkt.profile = hsa_ven_amd_aqlprofile_profile_t{
        agent_cache->get_hsa_agent(),
        HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,  // SPM?
        _events.data(),
        static_cast<uint32_t>(_events.size()),
        nullptr,
        0u,
        hsa_ven_amd_aqlprofile_descriptor_t{.ptr = nullptr, .size = 0},
        hsa_ven_amd_aqlprofile_descriptor_t{.ptr = nullptr, .size = 0}};
    auto& profile = pkt.profile;

    hsa_amd_memory_pool_access_t _access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    ext.hsa_amd_agent_memory_pool_get_info_fn(agent_cache->get_hsa_agent(),
                                              agent_cache->kernarg_pool(),
                                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                              static_cast<void*>(&_access));
    // Memory is accessable by both the GPU and CPU, unlock the command buffer for
    // sharing.
    if(_access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
    {
        throw std::runtime_error(
            fmt::format("Agent {} does not allow memory pool access for counter collection",
                        agent_cache->get_hsa_agent().handle));
    }

    CHECK_HSA(hsa_ven_amd_aqlprofile_start(&profile, nullptr), "could not generate packet sizes");

    if(profile.command_buffer.size == 0 || profile.output_buffer.size == 0)
    {
        throw std::runtime_error(
            fmt::format("No command or output buffer size set. CMD_BUF={} PROFILE_BUF={}",
                        profile.command_buffer.size,
                        profile.output_buffer.size));
    }

    // Allocate buffers and check the results
    auto alloc_and_check = [&](auto& pool, auto** mem_loc, auto size) -> bool {
        bool   malloced     = false;
        size_t page_aligned = getPageAligned(size);
        if(ext.hsa_amd_memory_pool_allocate_fn(
               pool, page_aligned, 0, static_cast<void**>(mem_loc)) != HSA_STATUS_SUCCESS)
        {
            *mem_loc = malloc(page_aligned);
            malloced = true;
        }
        else
        {
            CHECK(*mem_loc);
            hsa_agent_t agent = agent_cache->get_hsa_agent();
            // Memory is accessable by both the GPU and CPU, unlock the command buffer for
            // sharing.
            LOG_IF(FATAL,
                   ext.hsa_amd_agents_allow_access_fn(1, &agent, nullptr, *mem_loc) !=
                       HSA_STATUS_SUCCESS)
                << "Error: Allowing access to Command Buffer";
        }
        return malloced;
    };

    // Build command and output buffers
    pkt.command_buf_mallocd = alloc_and_check(
        agent_cache->cpu_pool(), &profile.command_buffer.ptr, profile.command_buffer.size);
    pkt.output_buffer_malloced = alloc_and_check(
        agent_cache->kernarg_pool(), &profile.output_buffer.ptr, profile.output_buffer.size);
    memset(profile.output_buffer.ptr, 0x0, profile.output_buffer.size);

    CHECK_HSA(hsa_ven_amd_aqlprofile_start(&profile, &pkt.start), "failed to create start packet");
    CHECK_HSA(hsa_ven_amd_aqlprofile_stop(&profile, &pkt.stop), "failed to create stop packet");
    CHECK_HSA(hsa_ven_amd_aqlprofile_read(&profile, &pkt.read), "failed to create read packet");
    pkt.start.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.stop.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.read.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    ROCP_TRACE << fmt::format("Following Packets Generated (output_buffer={}, output_size={}). "
                              "Start Pkt: {}, Read Pkt: {}, Stop Pkt: {}",
                              profile.output_buffer.ptr,
                              profile.output_buffer.size,
                              pkt.start,
                              pkt.read,
                              pkt.stop);
    return pkt_ptr;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

ThreadTraceAQLPacketFactory::ThreadTraceAQLPacketFactory(
    const hsa::AgentCache&                    agent,
    std::shared_ptr<thread_trace_parameters>& params,
    const CoreApiTable&                       coreapi,
    const AmdExtTable&                        ext)
{
    this->tracepool                  = std::make_shared<hsa::TraceMemoryPool>();
    this->tracepool->allocate_fn     = ext.hsa_amd_memory_pool_allocate_fn;
    this->tracepool->allow_access_fn = ext.hsa_amd_agents_allow_access_fn;
    this->tracepool->free_fn         = ext.hsa_amd_memory_pool_free_fn;
    this->tracepool->api_copy_fn     = coreapi.hsa_memory_copy_fn;
    this->tracepool->gpu_agent       = agent.get_hsa_agent();
    this->tracepool->cpu_pool_       = agent.cpu_pool();
    this->tracepool->gpu_pool_       = agent.gpu_pool();

    this->aql_params.clear();
    auto& p = this->aql_params;
    p.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, params->target_cu});
    p.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, params->shader_engine_mask});
    p.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION, params->simd_select});
    p.push_back({HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, params->buffer_size});

    this->profile = aqlprofile_att_profile_t{agent.get_hsa_agent(), p.data(), p.size()};
}

#pragma GCC diagnostic pop

std::unique_ptr<hsa::TraceAQLPacket>
ThreadTraceAQLPacketFactory::construct_packet()
{
    auto packet = std::make_unique<hsa::TraceAQLPacket>(this->tracepool);
    /*hsa_status_t _status = aqlprofile_att_create_packets(&packet->handle,
                                                         &packet->packets,
                                                         this->profile,
                                                         &hsa::TraceAQLPacket::Alloc,
                                                         &hsa::TraceAQLPacket::Free,
                                                         &hsa::TraceAQLPacket::Copy,
                                                         packet.get());
    CHECK_HSA(_status, "failed to create ATT packet");*/

    packet->before_krn_pkt.clear();
    packet->after_krn_pkt.clear();
    packet->packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet->packets.stop_packet.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    packet->empty = false;
    packet->start = packet->packets.start_packet;
    packet->stop  = packet->packets.stop_packet;
    packet->before_krn_pkt.push_back(packet->start);
    packet->after_krn_pkt.push_back(packet->stop);

    return packet;
}

std::vector<hsa_ven_amd_aqlprofile_event_t>
CounterPacketConstruct::get_all_events() const
{
    std::vector<hsa_ven_amd_aqlprofile_event_t> ret;
    for(const auto& metric : _metrics)
    {
        ret.insert(ret.end(), metric.instances.begin(), metric.instances.end());
    }
    return ret;
}

const counters::Metric*
CounterPacketConstruct::event_to_metric(const hsa_ven_amd_aqlprofile_event_t& event) const
{
    if(const auto* ptr = rocprofiler::common::get_val(
           _event_to_metric,
           std::make_tuple(event.block_name, event.block_index, event.counter_id)))
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

void
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
            throw std::runtime_error(
                fmt::format("Block {} exceeds max number of hardware counters ({} > {})",
                            static_cast<int64_t>(block_name.first),
                            count,
                            *max));
        }
    }
}
}  // namespace aql
}  // namespace rocprofiler
