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

#include <fmt/core.h>
#include <hsa/hsa_ext_amd.h>
#include "glog/logging.h"

namespace rocprofiler
{
namespace aql
{
AQLPacketConstruct::AQLPacketConstruct(const hsa::AgentCache&               agent,
                                       const std::vector<counters::Metric>& metrics)
: _agent(agent)
{
    // Validate that the counter exists and construct the block instances
    // for the counter.
    for(const auto& x : metrics)
    {
        auto query_info                = get_query_info(_agent.get_hsa_agent(), x);
        _metrics.emplace_back().metric = x;
        uint32_t event_id              = std::atoi(x.event().c_str());
        for(unsigned block_index = 0; block_index < query_info.instance_count; ++block_index)
        {
            _metrics.back().instances.push_back(
                {static_cast<hsa_ven_amd_aqlprofile_block_name_t>(query_info.id),
                 block_index,
                 event_id});
            bool validate_event_result;
            LOG_IF(FATAL,
                   hsa_ven_amd_aqlprofile_validate_event(_agent.get_hsa_agent(),
                                                         &_metrics.back().instances.back(),
                                                         &validate_event_result) !=
                       HSA_STATUS_SUCCESS);
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

std::unique_ptr<hsa::AQLPacket>
AQLPacketConstruct::construct_packet(const AmdExtTable& ext) const
{
    const size_t MEM_PAGE_MASK = 0x1000 - 1;
    auto         pkt_ptr       = std::make_unique<hsa::AQLPacket>(ext.hsa_amd_memory_pool_free_fn);
    auto&        pkt           = *pkt_ptr;
    if(_events.empty())
    {
        return pkt_ptr;
    }
    pkt.empty = false;

    pkt.profile = hsa_ven_amd_aqlprofile_profile_t{
        _agent.get_hsa_agent(),
        HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,  // SPM?
        _events.data(),
        static_cast<uint32_t>(_events.size()),
        nullptr,
        0u,
        hsa_ven_amd_aqlprofile_descriptor_t{.ptr = nullptr, .size = 0},
        hsa_ven_amd_aqlprofile_descriptor_t{.ptr = nullptr, .size = 0}};
    auto& profile = pkt.profile;

    hsa_amd_memory_pool_access_t _access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    ext.hsa_amd_agent_memory_pool_get_info_fn(_agent.get_hsa_agent(),
                                              _agent.kernarg_pool(),
                                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                              static_cast<void*>(&_access));
    // Memory is accessable by both the GPU and CPU, unlock the command buffer for
    // sharing.
    if(_access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
    {
        throw std::runtime_error(
            fmt::format("Agent {} does not allow memory pool access for counter collection",
                        _agent.get_hsa_agent().handle));
    }

    auto throw_if_failed = [](auto status, auto& message) {
        if(status != HSA_STATUS_SUCCESS)
        {
            throw std::runtime_error(message);
        }
    };

    throw_if_failed(hsa_ven_amd_aqlprofile_start(&profile, nullptr),
                    "could not generate packet sizes");

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
        size_t page_aligned = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
        if(ext.hsa_amd_memory_pool_allocate_fn(
               pool, page_aligned, 0, static_cast<void**>(mem_loc)) != HSA_STATUS_SUCCESS)
        {
            *mem_loc = malloc(page_aligned);
            malloced = true;
        }
        else
        {
            CHECK(*mem_loc);
            hsa_agent_t agent = _agent.get_hsa_agent();
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
        _agent.cpu_pool(), &profile.command_buffer.ptr, profile.command_buffer.size);
    pkt.output_buffer_malloced = alloc_and_check(
        _agent.kernarg_pool(), &profile.output_buffer.ptr, profile.output_buffer.size);
    memset(profile.output_buffer.ptr, 0x0, profile.output_buffer.size);

    // throw if we do not construct the packets correctly.
    throw_if_failed(hsa_ven_amd_aqlprofile_start(&profile, &pkt.start),
                    "could not generate start packet");
    throw_if_failed(hsa_ven_amd_aqlprofile_stop(&profile, &pkt.stop),
                    "could not generate stop packet");
    throw_if_failed(hsa_ven_amd_aqlprofile_read(&profile, &pkt.read),
                    "could not generate read packet");
    pkt.start.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.stop.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.read.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

    return pkt_ptr;
}

std::vector<hsa_ven_amd_aqlprofile_event_t>
AQLPacketConstruct::get_all_events() const
{
    std::vector<hsa_ven_amd_aqlprofile_event_t> ret;
    for(const auto& metric : _metrics)
    {
        ret.insert(ret.end(), metric.instances.begin(), metric.instances.end());
    }
    return ret;
}

const counters::Metric*
AQLPacketConstruct::event_to_metric(const hsa_ven_amd_aqlprofile_event_t& event) const
{
    if(const auto* ptr = rocprofiler::common::get_val(
           _event_to_metric,
           std::make_tuple(event.block_name, event.block_index, event.counter_id)))
    {
        return ptr;
    }
    return nullptr;
}

void
AQLPacketConstruct::can_collect()
{
    // Verify that the counters fit within harrdware limits
    std::map<std::pair<hsa_ven_amd_aqlprofile_block_name_t, uint32_t>, int64_t> counter_count;
    std::map<std::pair<hsa_ven_amd_aqlprofile_block_name_t, uint32_t>, int64_t> max_allowed;
    for(auto& metric : _metrics)
    {
        for(auto& instance : metric.instances)
        {
            auto block_pair       = std::make_pair(instance.block_name, instance.block_index);
            auto [iter, inserted] = counter_count.emplace(block_pair, 0);
            iter->second++;
            if(inserted)
            {
                max_allowed.emplace(block_pair,
                                    get_block_counters(_agent.get_hsa_agent(), instance));
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
