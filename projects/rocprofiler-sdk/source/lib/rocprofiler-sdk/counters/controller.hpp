
// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/ioctl.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace counters
{
// Stores counter profiling information such as the agent
// to collect counters on, the metrics to collect, the hw
// counters needed to evaluate the metrics, and the ASTs.
// This profile can be shared among many rocprof contexts.
struct counter_config
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
    rocprofiler_counter_config_id_t    id{.handle = 0};
    // Packet generator to create AQL packets for insertion
    std::unique_ptr<rocprofiler::aql::CounterPacketConstruct> pkt_generator{nullptr};
    // A packet cache of AQL packets. This allows reuse of AQL packets (preventing costly
    // allocation of new packets/destruction).
    common::Synchronized<std::vector<std::unique_ptr<rocprofiler::hsa::AQLPacket>>> packets{};
};

class CounterController
{
public:
    CounterController();

    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    uint64_t add_profile(std::shared_ptr<counter_config>&& config);

    void destroy_profile(uint64_t id);
    // Setup the counter collection service. counter_callback_info is created here
    // to contain the counters that need to be collected (specified in profile_id) and
    // the AQL packet generator for injecting packets. Note: the service is created
    // in the stop state.
    static rocprofiler_status_t configure_dispatch(
        rocprofiler_context_id_t                   context_id,
        rocprofiler_buffer_id_t                    buffer,
        rocprofiler_dispatch_counting_service_cb_t callback,
        void*                                      callback_args,
        rocprofiler_dispatch_counting_record_cb_t  record_callback,
        void*                                      record_callback_args);
    std::shared_ptr<counter_config> get_profile_cfg(rocprofiler_counter_config_id_t id);

    static rocprofiler_status_t configure_agent_collection(
        rocprofiler_context_id_t                 context_id,
        rocprofiler_buffer_id_t                  buffer_id,
        rocprofiler_agent_id_t                   agent_id,
        rocprofiler_device_counting_service_cb_t cb,
        void*                                    user_data);

private:
    common::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<counter_config>>> _configs;
};

CounterController&
get_controller();

rocprofiler_status_t
create_counter_profile(std::shared_ptr<counter_config> config);

void
destroy_counter_profile(uint64_t id);

std::shared_ptr<counter_config>
get_counter_config(rocprofiler_counter_config_id_t id);

}  // namespace counters
}  // namespace rocprofiler
