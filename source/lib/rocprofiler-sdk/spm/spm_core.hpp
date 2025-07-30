// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/internalqueue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"

#include <rocprofiler-sdk/experimental/spm/capture.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
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

namespace SPM
{
struct spm_parameter_pack
{
    uint64_t sample_freq = DEFAULT_SAMPLE_FREQUENCY;
    uint64_t buffer_size = DEFAULT_BUFFER_SIZE;
    uint64_t timeout     = DEFAULT_TIMEOUT_MS;

    std::vector<rocprofiler::counters::Metric> metrics{};

    rocprofiler_spm_data_callback_t data_fn{};
    rocprofiler_user_data_t         user_data{};

    rocprofiler_spm_dispatch_callback_t dispatch_fn{};
    void*                               config_userdata{nullptr};

    static constexpr size_t DEFAULT_SAMPLE_FREQUENCY = 640000;     // 640 KHz
    static constexpr size_t DEFAULT_BUFFER_SIZE      = 0x3000000;  // 48 MB
    static constexpr size_t DEFAULT_TIMEOUT_MS       = 50;         // 100ms

    bool valid() const
    {
        return sample_freq != 0 && buffer_size != 0 && timeout != 0 && !metrics.empty();
    }
};

class SPMQueue : public hsa::internal_queue::Queue
{
public:
    using Signal = hsa::internal_queue::Signal;
    class StopSignal
    {
    public:
        StopSignal(hsa::SPMPacket* pkt, std::unique_ptr<Signal>&& _signal)
        : packet(pkt)
        , signal(std::move(_signal)){};
        ~StopSignal();

        hsa::SPMPacket* const   packet;
        std::unique_ptr<Signal> signal{};
    };

    SPMQueue(spm_parameter_pack, const hsa::AgentCache&);
    ~SPMQueue() override;

    std::unique_ptr<Signal>     start();
    std::unique_ptr<StopSignal> stop();

    const spm_parameter_pack params;
    std::mutex               mut{};

    std::unique_ptr<hsa::SPMPacket> packet{nullptr};
};

class SPMAgentManager
{
public:
    SPMAgentManager()  = default;
    ~SPMAgentManager() = default;

    rocprofiler_status_t start_context();
    void                 stop_context();

    void resource_init();
    void resource_deinit();

    bool add_agent(rocprofiler_agent_id_t id, spm_parameter_pack _params)
    {
        if(has_agent(id)) return false;
        std::unique_lock<std::mutex> lk(agent_mut);
        params[id] = std::move(_params);
        return true;
    }

    bool has_agent(rocprofiler_agent_id_t id)
    {
        std::unique_lock<std::mutex> lk(agent_mut);
        return params.find(id) != params.end();
    }

    std::map<rocprofiler_agent_id_t, std::unique_ptr<SPMQueue>> queues{};
    std::map<rocprofiler_agent_id_t, spm_parameter_pack>        params{};

    std::mutex agent_mut;
};

class SPMDispatchFactory
{
public:
    SPMDispatchFactory(spm_parameter_pack _params, const hsa::AgentCache&);
    ~SPMDispatchFactory();

    const spm_parameter_pack params;
    std::mutex               mut{};
    std::condition_variable  cv{};

    std::unique_ptr<hsa::SPMPacket> packet{nullptr};
};

class SPMDispatchManager
{
    using AQLPacketPtr = std::unique_ptr<hsa::AQLPacket>;
    using inst_pkt_t   = common::container::small_vector<std::pair<AQLPacketPtr, int64_t>, 4>;

public:
    SPMDispatchManager()  = default;
    ~SPMDispatchManager() = default;

    void start_context();
    void stop_context();

    void resource_init();
    void resource_deinit();

    bool add_agent(rocprofiler_agent_id_t id, spm_parameter_pack _params)
    {
        if(has_agent(id)) return false;
        auto lk = std::unique_lock{agent_mut};
        timeout_ms += _params.timeout;
        params[id] = std::move(_params);
        return true;
    }

    bool has_agent(rocprofiler_agent_id_t id)
    {
        auto lk = std::unique_lock{agent_mut};
        return params.find(id) != params.end();
    }

    std::vector<std::pair<rocprofiler_agent_id_t, std::shared_ptr<SPMDispatchFactory>>> factories{};
    std::map<rocprofiler_agent_id_t, spm_parameter_pack>                                params{};

    hsa::Queue::pkt_and_serialize_t pre_kernel_call(const hsa::Queue&              queue,
                                                    uint64_t                       kernel_id,
                                                    rocprofiler_dispatch_id_t      dispatch_id,
                                                    rocprofiler_user_data_t*       user_data,
                                                    const context::correlation_id* corr_id);

    void post_kernel_call(inst_pkt_t& aql, const hsa::queue_info_session& session);

    uint64_t                    timeout_ms{0};
    std::condition_variable_any cv;
    std::atomic<int>            pending_dispatches{0};
    std::shared_mutex           agent_mut{};
    std::atomic<bool>           bActiveCtx{false};
};

void
initialize(HsaApiTable* table);

void
finalize();

}  // namespace SPM
}  // namespace rocprofiler
