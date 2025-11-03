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

// Utilities that wrap HSA primitives used by thread trace triple buffering.
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        ROCP_FATAL_IF(_status != HSA_STATUS_SUCCESS) << "HSA Err: " << _status;                    \
    }

namespace rocprofiler
{
namespace thread_trace
{
constexpr size_t QUEUE_SIZE = 256;  // Small dedicated queue for SQTT control traffic

Signal::Signal(hsa_ext_amd_aql_pm4_packet_t* packet)
: Signal()
{
    packet->completion_signal = signal;
    reset();
}

Signal::Signal()
{
    auto* ext = CHECK_NOTNULL(hsa::get_amd_ext_table());

    ext->hsa_amd_signal_create_fn(0, 0, nullptr, 0, &signal);
}

Signal::~Signal()
{
    WaitOn();
    hsa::get_core_table()->hsa_signal_destroy_fn(signal);
}

void
Signal::WaitOn() const
{
    auto wait_fn = hsa::get_core_table()->hsa_signal_wait_scacquire_fn;
    while(wait_fn(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
        sched_yield();
}

void
Signal::reset()
{
    CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_store_screlease_fn(signal, 1);
}

HsaATTQueue::HsaATTQueue(const hsa::AgentCache& agent, size_t triple_buffer_size)
: agent_id(CHECK_NOTNULL(agent.get_rocp_agent())->id)
, buffer_size(triple_buffer_size)
, hsa_agent(agent.get_hsa_agent())
, near_cpu(agent.near_cpu())
{
    ROCP_TRACE << "Constructing Async queue.";

    auto* core = CHECK_NOTNULL(hsa::get_core_table());
    auto* ext  = CHECK_NOTNULL(hsa::get_amd_ext_table());

    auto status = core->hsa_queue_create_fn(hsa_agent,
                                            QUEUE_SIZE,
                                            HSA_QUEUE_TYPE_SINGLE,
                                            nullptr,
                                            nullptr,
                                            UINT32_MAX,
                                            UINT32_MAX,
                                            &this->queue);

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create thread trace async queue";

    if(triple_buffer_size)
    {
        for(auto& memory : triple_buffer_memory)
        {
            CHECK_HSA(ext->hsa_amd_memory_pool_allocate_fn(
                          agent.cpu_pool(), triple_buffer_size, 0, &memory),
                      "failed to allocate contiguous memory");
            CHECK_HSA(ext->hsa_amd_agents_allow_access_fn(1, &near_cpu, nullptr, memory),
                      "failed to allow cpu access");
            CHECK_HSA(ext->hsa_amd_agents_allow_access_fn(1, &hsa_agent, nullptr, memory),
                      "failed to allow gpu access");
        }
    }

    ROCP_TRACE << "Done constructing Async queue.";
}

HsaATTQueue::~HsaATTQueue()
{
    ROCP_TRACE << "Destroying Async Queue...";
    hsa::get_core_table()->hsa_queue_destroy_fn(this->queue);

    for(auto memory : triple_buffer_memory)
        hsa::get_amd_ext_table()->hsa_amd_memory_pool_free_fn(memory);
}

void
HsaATTQueue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet, Signal* completion) const
{
    auto* core = CHECK_NOTNULL(hsa::get_core_table());

    std::unique_ptr<Signal> signal{};
    const uint64_t          write_idx = core->hsa_queue_add_write_index_relaxed_fn(queue, 1);

    size_t index = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    if(completion)
    {
        completion->reset();
        reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(queue_slot)->completion_signal =
            completion->getSignal();
    }
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    core->hsa_signal_store_screlease_fn(queue->doorbell_signal, write_idx);
}

std::unique_ptr<Signal>
HsaATTQueue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet, bool bWait) const
{
    auto signal = std::unique_ptr<Signal>{nullptr};
    if(bWait) signal = std::make_unique<Signal>();

    Submit(packet, signal.get());
    return signal;
}

};  // namespace thread_trace
};  // namespace rocprofiler
