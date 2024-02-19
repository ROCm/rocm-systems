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

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/dispatch_profile.h>

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

namespace rocprofiler
{
namespace context
{
struct context;
}
namespace counters
{
// Stores counter profiling information such as the agent
// to collect counters on, the metrics to collect, the hw
// counters needed to evaluate the metrics, and the ASTs.
// This profile can be shared among many rocprof contexts.
struct profile_config
{
    const rocprofiler_agent_t*    agent = nullptr;
    std::vector<counters::Metric> metrics{};
    // HW counters that must be collected to compute the above
    // metrics (derived metrics are broken down into hw counters
    // in this vector).
    std::set<counters::Metric> reqired_hw_counters{};
    // Counters that are not hardware based but based on either a
    // static value (such as those in agent)
    std::set<counters::Metric> required_special_counters{};
    // ASTs to evaluate
    std::vector<counters::EvaluateAST> asts{};
    rocprofiler_profile_config_id_t    id{.handle = 0};
    // Packet generator to create AQL packets for insertion
    std::unique_ptr<rocprofiler::aql::AQLPacketConstruct> pkt_generator{nullptr};
    // A packet cache of AQL packets. This allows reuse of AQL packets (preventing costly
    // allocation of new packets/destruction).
    rocprofiler::common::Synchronized<std::vector<std::unique_ptr<rocprofiler::hsa::AQLPacket>>>
        packets{};
};

// Internal counter struct that stores the state needed to handle an intercepted
// HSA kernel packet.
struct counter_callback_info
{
    // User callback
    rocprofiler_profile_counting_dispatch_callback_t user_cb{nullptr};
    // User id
    void* callback_args{nullptr};
    // Link to the context this is associated with
    rocprofiler_context_id_t context{.handle = 0};
    // Link to the internal context this is associated with
    const context::context* internal_context{nullptr};
    // HSA Queue ClientID. This is an ID we get when we insert a callback into the
    // HSA queue interceptor. This ID can be used to disable the callback.
    rocprofiler::hsa::ClientID queue_id{-1};

    // Buffer to use for storing counter data. Used if callback is not set.
    std::optional<rocprofiler_buffer_id_t> buffer;

    rocprofiler_profile_counting_record_callback_t record_callback;
    void*                                          record_callback_args;

    // Facilitates the return of an AQL Packet to the profile config that constructed it.
    rocprofiler::common::Synchronized<
        std::unordered_map<rocprofiler::hsa::AQLPacket*, std::shared_ptr<profile_config>>>
        packet_return_map{};

    static rocprofiler_status_t setup_profile_config(const hsa::AgentCache&,
                                                     std::shared_ptr<profile_config>&);

    rocprofiler_status_t get_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>&,
                                    const hsa::AgentCache&,
                                    std::shared_ptr<profile_config>&);
};

uint64_t
create_counter_profile(std::shared_ptr<rocprofiler::counters::profile_config>&& config);

void
destroy_counter_profile(uint64_t id);

bool
configure_buffered_dispatch(rocprofiler_context_id_t                         context_id,
                            rocprofiler_buffer_id_t                          buffer,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_args);

bool
configure_callback_dispatch(rocprofiler_context_id_t                         context_id,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_data_args,
                            rocprofiler_profile_counting_record_callback_t   record_callback,
                            void*                                            record_callback_args);

void
start_context(const context::context*);

void
stop_context(const context::context*);

std::unique_ptr<rocprofiler::hsa::AQLPacket>
queue_cb(const std::shared_ptr<counter_callback_info>&                   info,
         const hsa::Queue&                                               queue,
         const hsa::rocprofiler_packet&                                  pkt,
         uint64_t                                                        kernel_id,
         const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
         const context::correlation_id*                                  correlation_id);

using ClientID   = int64_t;
using inst_pkt_t = common::container::
    small_vector<std::pair<std::unique_ptr<rocprofiler::hsa::AQLPacket>, ClientID>, 4>;

void
completed_cb(const std::shared_ptr<counter_callback_info>&,
             const hsa::Queue&,
             hsa::rocprofiler_packet,
             const hsa::Queue::queue_info_session_t&,
             inst_pkt_t& pkts);

std::shared_ptr<profile_config> get_profile_config(rocprofiler_profile_config_id_t);
}  // namespace counters
}  // namespace rocprofiler
