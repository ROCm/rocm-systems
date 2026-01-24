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

#pragma once

#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
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

    uint64_t sample_freq = 0;
    uint64_t buffer_size = 0;
    uint64_t timeout     = 0;

    rocprofiler_spm_counter_config_id_t id{.handle = 0};
    // Packet generator to create AQL packets for insertion
    std::unique_ptr<rocprofiler::aql::SPMPacketConstruct> pkt_generator{nullptr};
    // A packet cache of AQL packets. This allows reuse of AQL packets (preventing costly
    // allocation of new packets/destruction).
    
    bool valid() const
    {
        return sample_freq != 0 && buffer_size != 0 && timeout != 0 && !metrics.empty();
    }
};

struct spm_callback_data
{
    bool                                             is_profiling{false};
    rocprofiler_spm_dispatch_counting_record_cb_t    record_cb{};
    rocprofiler_spm_dispatch_counting_service_data_t dispatch_data{};
    rocprofiler_user_data_t*                         user_data;
    void*                                            record_callback_args{};
    bool                                             config_switch{false};
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
    // Link to the internal context this is associated with
    // Internal context is used as a key to obtain external correlation id in pre kernel call
    const context::context*                       internal_context;
    rocprofiler_spm_dispatch_counting_record_cb_t record_callback;
    void*                                         record_callback_args;
    static rocprofiler_status_t setup_spm_counter_config(std::shared_ptr<spm_counter_config>&);

    common::Synchronized<
        std::unordered_map<rocprofiler::hsa::AQLPacket*, std::shared_ptr<spm_counter_config>>>
                                packet_return_map{};
    
};

struct enqueue_dispatch_config_state
{
    rocprofiler_spm_counter_config_id_t          config_id;
    std::unique_ptr<rocprofiler::hsa::SPMPacket> spm_packet;

    enqueue_dispatch_config_state(rocprofiler_spm_counter_config_id_t          id,
                                  std::unique_ptr<rocprofiler::hsa::SPMPacket> packet)
    : config_id(id)
    , spm_packet(std::move(packet))
    {}
};

/**
 *This is a singleton class with lazy initialization
 */
class SpmCounterController
{
public:
    SpmCounterController() = default;
    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    void spm_add_profile(std::shared_ptr<spm_counter_config>&& config);

    void spm_destroy_profile(uint64_t id);
    // Setup the SPM counter collection service. spm_counter_callback_info is created here

    std::shared_ptr<spm_counter_config> get_profile_cfg(rocprofiler_spm_counter_config_id_t id);

    void state_map_fini();
    
    common::Synchronized<
        std::unordered_map<uint64_t, std::deque<std::unique_ptr<enqueue_dispatch_config_state>>>>
        _agent_state_map;
    common::Synchronized<std::unordered_map<uint64_t, std::deque<std::unique_ptr<spm_callback_data>>>>
      _callback_data;

private:
    // Cache to contain the map of config id handle to spm counter config
    common::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<spm_counter_config>>>
                                                           _configs;
};

SpmCounterController&
spm_get_controller();

rocprofiler_status_t
create_spm_counter_profile(std::shared_ptr<spm_counter_config> config);

void
destroy_spm_counter_profile(uint64_t id);

std::shared_ptr<spm_counter_config>
get_spm_counter_config(rocprofiler_spm_counter_config_id_t id);

rocprofiler_status_t
configure_callback_spm_dispatch(rocprofiler_context_id_t                       context_id,
                                rocprofiler_spm_dispatch_counting_service_cb_t callback,
                                void*                                          callback_data_args,
                                rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
                                void* record_callback_args);
/*
 * start dispatch SPM context
 */
void
start_context(const context::context*);

/*
 * stop dispatch SPM context
 */
void
stop_context(const context::context*);

void
state_map_fini();

}  // namespace spm
}  // namespace rocprofiler
