// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
class AQLPacket;
};
namespace spm
{
/**
 * @brief  SPM counter config contains SPM parameters and counters
 * SPM config is per agent
 * Pkt generator is used to construct the packet, pkt generator can be created before HSA init
 * Has a packet cache to store the AQLpackets for SPM, it is constructed using the pkt generator.
 * Its valid function checks if config has parameters and metrics initialized
 */
struct spm_counter_config
{
    const rocprofiler_agent_t*    agent = nullptr;
    std::vector<counters::Metric> metrics{};

    std::vector<rocprofiler_spm_parameters_t> spm_parameters{};

    rocprofiler_counter_config_id_t id{.handle = 0};
    // A packet cache of AQL packets. This allows reuse of AQL packets (preventing costly
    // allocation of new packets/destruction).
    common::Synchronized<std::vector<std::unique_ptr<rocprofiler::hsa::AQLPacket>>> packets;
};

/**
 * @brief spm_counter_callback_info has the callbacks and user data associated with a context
 * It has a cache of AQLPackets associated with configs which is used in post kernel callback
 *    to retrieve the config information for the given AQLPacket
 *
 */
struct spm_counter_callback_info
{
    rocprofiler_spm_dispatch_counting_service_cb_t user_cb{nullptr};
    void*                                          callback_args{nullptr};
    // Link to the context this is associated with
    rocprofiler_context_id_t context{.handle = 0};
    // HSA Queue ClientID. This is an ID we get when we insert a callback into the
    // HSA queue interceptor. This ID can be used to disable the callback.
    rocprofiler::hsa::ClientID queue_id{-1};
    // Buffer to use for storing counter data. Used if callback is not set.
    std::optional<rocprofiler_buffer_id_t> buffer;
    // Link to the internal context this is associated with
    // Internal context is used as a key to obtain external correlation id in pre kernel call
    const context::context*                       internal_context{nullptr};
    rocprofiler_spm_dispatch_counting_record_cb_t record_callback{nullptr};
    void*                                         record_callback_args{nullptr};
    common::Synchronized<
        std::unordered_map<rocprofiler::hsa::AQLPacket*, std::shared_ptr<spm_counter_config>>>
        packet_return_map{};
};

rocprofiler_status_t
get_spm_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>&, std::shared_ptr<spm_counter_config>&);

rocprofiler_status_t
create_spm_counter_profile(std::shared_ptr<spm_counter_config> config);

rocprofiler_status_t
destroy_spm_counter_profile(rocprofiler_counter_config_id_t id);

std::shared_ptr<spm_counter_config>
get_spm_counter_config(rocprofiler_counter_config_id_t id);

rocprofiler_status_t
configure_callback_spm_dispatch(rocprofiler_context_id_t                       context_id,
                                rocprofiler_spm_dispatch_counting_service_cb_t callback,
                                void*                                          callback_data_args,
                                rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
                                void* record_callback_args);

bool
is_spm_explicitly_enabled();

/*
 * start dispatch SPM context
 */
rocprofiler_status_t
start_context(const context::context*);

/*
 * stop dispatch SPM context
 */
void
stop_context(const context::context*);

}  // namespace spm
}  // namespace rocprofiler
