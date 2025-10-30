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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/counters/tests/metrics_test_helpers.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"
#include "lib/rocprofiler-sdk/thread_trace/threading.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace rocprofiler
{
namespace thread_trace
{
HsaApiTable table{};

void
test_init()
{
    auto init = []() -> bool {
        if(hsa_init() != HSA_STATUS_SUCCESS) abort();

        table.amd_ext_ = &counters::test_constants::get_ext_table();
        table.core_    = &counters::test_constants::get_api_table();
        rocprofiler::hsa::copy_table(table.core_, 0);
        rocprofiler::hsa::copy_table(table.amd_ext_, 0);
        agent::construct_agent_cache(&table);
        hsa::get_queue_controller()->init(counters::test_constants::get_api_table(), counters::test_constants::get_ext_table());
        
        registration::init_logging();
        registration::set_init_status(-1);

        return true;
    };
    [[maybe_unused]] static bool run_once = init();
}

class QueueMock: public HsaATTQueue
{
public:
    QueueMock(const hsa::AgentCache& agent): HsaATTQueue(agent, BUFFER_SIZE) {}
    virtual void Submit(hsa_ext_amd_aql_pm4_packet_t*, Signal*) const {};

    static void copy_data_mock(void* dst, const void* src, hsa_agent_t, hsa_agent_t, size_t size, Signal*)
    {
        std::memcpy(dst, src, size);
    }

    static constexpr size_t BUFFER_SIZE = 1u << 20;
};

using query_status_t = std::function<std::optional<hsa::sqtt_buffer_status_t>(void)>;

class MockPackets: public hsa::SQTTBufferingPackets
{
public:
    MockPackets(aqlprofile_handle_t handle, query_status_t _query): hsa::SQTTBufferingPackets(handle), query_fn(_query) {};

    std::optional<hsa::sqtt_buffer_status_t> query_buffer_status() override { return query_fn(); };
    query_status_t query_fn;
};

struct consumer_producer_t
{
    std::thread consumer{};
    std::thread producer{};
    std::shared_ptr<std::atomic<bool>> flag{};
};

consumer_producer_t
start_threads(rocprofiler_thread_trace_shader_data_callback_t cb_fn, query_status_t query_fn, rocprofiler_user_data_t userdata)
{
    // Build a synthetic queue + packet stack that mimics the runtime so we can
    // exercise the producer/consumer pairing without a real GPU.
    const hsa::AgentCache* agent = nullptr;
    {
        auto& agents = hsa::get_queue_controller()->get_supported_agents();

        for(const auto& [_, _agent] : agents)
        {
            auto* rocp = _agent.get_rocp_agent();
            if (rocp && rocp->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                agent = &_agent;
                break;
            }
        }
    }
    if (agent == nullptr) abort();

    auto running_flag = std::make_shared<std::atomic<bool>>(true);

    auto params = thread_trace_parameter_pack{};
    params.triple_buffering  = true;
    params.buffer_size       = QueueMock::BUFFER_SIZE;
    params.shader_cb_fn      = cb_fn;
    params.callback_userdata = userdata;

    auto factory = std::make_unique<aql::ThreadTraceAQLPacketFactory>(*agent, params, *table.core_, *table.amd_ext_);
    auto control_packet = factory->construct_control_packet();
    auto buffer_packet  = std::make_unique<MockPackets>(control_packet->GetHandle(), query_fn);

    auto worker_data            = std::make_shared<triple_buffer_shared_data_t>();
    worker_data->queue          = std::make_shared<QueueMock>(*agent);

    auto producer_data = triple_buffer_producer_data_t{};
    producer_data.producer_running = running_flag;
    producer_data.start_pkt_signal = std::make_shared<Signal>();
    producer_data.control_packet   = std::move(control_packet);
    producer_data.copy_data_fn     = QueueMock::copy_data_mock;
    producer_data.shared           = worker_data;
    producer_data.buffer_packet    = std::move(buffer_packet);

    auto consumer_data = triple_buffer_consumer_data_t{};
    consumer_data.callback_fn = params.shader_cb_fn;
    consumer_data.userdata    = params.callback_userdata;
    consumer_data.shared      = worker_data;
    
    consumer_producer_t ret{};
    ret.producer = std::thread{producer_loop, std::move(producer_data)};
    ret.consumer = std::thread{consumer_loop, std::move(consumer_data)};
    ret.flag     = running_flag;

    return ret;
}

}  // namespace thread_trace
}  // namespace rocprofiler


/*
typedef void (*rocprofiler_thread_trace_shader_data_callback_t)(
    rocprofiler_agent_id_t                       agent,
    int64_t                                      shader_engine_id,
    void*                                        data,
    size_t                                       data_size,
    rocprofiler_thread_trace_shader_data_flags_t flags,
    rocprofiler_user_data_t                      userdata);
*/

TEST(thread_trace, init_shutdown)
{
    rocprofiler::thread_trace::test_init();

    // Sanity check: threads should spin and exit cleanly when the running flag flips.
    auto empty_cb = [](
        rocprofiler_agent_id_t,
        int64_t,
        void*,
        size_t,
        rocprofiler_thread_trace_shader_data_flags_t,
        rocprofiler_user_data_t
    ) {};

    auto always_null = [] () { return std::nullopt; };

    auto userdata = rocprofiler_user_data_t{};
    auto threads  = rocprofiler::thread_trace::start_threads(empty_cb, always_null, userdata);
    threads.flag->store(false);
    threads.consumer.join();
    threads.producer.join();
}

TEST(thread_trace, status_query)
{
    rocprofiler::thread_trace::test_init();

    // Ensure the producer polls even when the GPU reports "nothing to copy".
    auto empty_cb = [](
        rocprofiler_agent_id_t,
        int64_t,
        void*,
        size_t,
        rocprofiler_thread_trace_shader_data_flags_t,
        rocprofiler_user_data_t
    ) {};

    auto status_called = std::atomic<bool>{false};
    auto always_null = [&]() { status_called = true; return std::nullopt; };

    auto userdata = rocprofiler_user_data_t{};
    auto threads  = rocprofiler::thread_trace::start_threads(empty_cb, always_null, userdata);
    
    while (!status_called) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    threads.flag->store(false);
    threads.consumer.join();
    threads.producer.join();
}

TEST(thread_trace, multiple_calls)
{
    rocprofiler::thread_trace::test_init();
    const size_t BUFFER_SIZE = rocprofiler::thread_trace::QueueMock::BUFFER_SIZE;

    auto data_received = std::atomic<size_t>{0};

    // Accumulate the payload sizes so we can verify every buffer reported by the
    // mock GPU eventually reaches the consumer.
    auto fetch_cb = [](
        rocprofiler_agent_id_t,
        int64_t,
        void*,
        size_t data_size,
        rocprofiler_thread_trace_shader_data_flags_t,
        rocprofiler_user_data_t userdata
    ){
        static_cast<std::atomic<size_t>*>(userdata.ptr)->fetch_add(data_size);
    };

    auto input_buffer = std::vector<size_t>();
    input_buffer.resize(BUFFER_SIZE/sizeof(size_t));

    auto status_called = std::atomic<int>{0};
    auto return_synced = [&]() -> std::optional<rocprofiler::hsa::sqtt_buffer_status_t>
    {
        // Throttle the mock producer so it never outruns the consumer in this scenario.
        if (status_called * BUFFER_SIZE > data_received) return std::nullopt;
        status_called.fetch_add(1);

        auto status = rocprofiler::hsa::sqtt_buffer_status_t{};
        status.data = input_buffer.data();
        status.size = BUFFER_SIZE;
        return status;
    };

    auto userdata = rocprofiler_user_data_t{.ptr = &data_received};
    auto threads  = rocprofiler::thread_trace::start_threads(fetch_cb, return_synced, userdata);
    
    while (status_called < 1000) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    threads.flag->store(false);
    threads.consumer.join();
    threads.producer.join();

    ASSERT_EQ(data_received.load(), status_called.load() * BUFFER_SIZE);
}

TEST(thread_trace, data_integrity)
{
    rocprofiler::thread_trace::test_init();
    const size_t BUFFER_SIZE = rocprofiler::thread_trace::QueueMock::BUFFER_SIZE;

    auto output_buffer = std::vector<size_t>();
    output_buffer.reserve(128*BUFFER_SIZE/sizeof(size_t));

    // Capture every word of the buffer so we can assert exact ordering after many copies.
    auto fetch_cb = [](
        rocprofiler_agent_id_t,
        int64_t,
        void* datain,
        size_t data_size,
        rocprofiler_thread_trace_shader_data_flags_t,
        rocprofiler_user_data_t userdata
    ){
        auto* buf  = static_cast<std::vector<size_t>*>(userdata.ptr);
        auto* data = static_cast<size_t*>(datain);

        for (size_t i=0; i<data_size/sizeof(size_t); i++)
        {
            buf->emplace_back(data[i]);
        }
    };

    auto input_buffer = std::vector<size_t>();
    input_buffer.resize(BUFFER_SIZE/sizeof(size_t));

    auto status_called = std::atomic<int>{0};
    auto return_synced = [&]() -> std::optional<rocprofiler::hsa::sqtt_buffer_status_t>
    {
        auto called = status_called.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        for (size_t i=0; i<input_buffer.size(); i++)
            input_buffer[i] = i + called * input_buffer.size();

        auto status = rocprofiler::hsa::sqtt_buffer_status_t{};
        status.data = input_buffer.data();
        status.size = BUFFER_SIZE;
        return status;
    };

    auto userdata = rocprofiler_user_data_t{.ptr = &output_buffer};
    auto threads  = rocprofiler::thread_trace::start_threads(fetch_cb, return_synced, userdata);
    
    while (status_called < 100) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    threads.flag->store(false);
    threads.consumer.join();
    threads.producer.join();

    ASSERT_EQ(output_buffer.size() * sizeof(size_t), status_called.load() * BUFFER_SIZE);

    for (size_t i=0; i<output_buffer.size(); i++)
    {
        ASSERT_EQ(i, output_buffer[i]);
    }
}

TEST(thread_trace, slow_cpu)
{
    rocprofiler::thread_trace::test_init();
    const size_t BUFFER_SIZE = rocprofiler::thread_trace::QueueMock::BUFFER_SIZE;

    auto interrupt_received = std::atomic<bool>{false};

    // Simulate a user callback that cannot keep up; the producer should flag a
    // CPU buffer stall so the consumer can drain and exit.
    auto fetch_cb = [](
        rocprofiler_agent_id_t,
        int64_t,
        void*,
        size_t,
        rocprofiler_thread_trace_shader_data_flags_t flags,
        rocprofiler_user_data_t userdata
    ){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL)
            static_cast<std::atomic<bool>*>(userdata.ptr)->store(true);
    };

    auto input_buffer = std::vector<size_t>();
    input_buffer.resize(BUFFER_SIZE/sizeof(size_t));

    auto status_called = std::atomic<int>{0};
    auto return_synced = [&]() -> std::optional<rocprofiler::hsa::sqtt_buffer_status_t>
    {
        // Always return a full buffer to force the producer into the "CPU slow" branch.
        status_called.fetch_add(1);
        auto status = rocprofiler::hsa::sqtt_buffer_status_t{};
        status.data = input_buffer.data();
        status.size = BUFFER_SIZE;
        return status;
    };

    auto userdata = rocprofiler_user_data_t{.ptr = &interrupt_received};
    auto threads  = rocprofiler::thread_trace::start_threads(fetch_cb, return_synced, userdata);
    
    while (!interrupt_received) std::this_thread::sleep_for(std::chrono::milliseconds(1));

    threads.flag->store(false);
    threads.consumer.join();
    threads.producer.join();

    ASSERT_EQ(interrupt_received.load(), true);
}

TEST(thread_trace, slow_gpu)
{
    rocprofiler::thread_trace::test_init();
    // TODO: Implement
}
