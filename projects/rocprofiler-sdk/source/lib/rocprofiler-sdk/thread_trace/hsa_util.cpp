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
        ROCP_FATAL_IF(_status != HSA_STATUS_SUCCESS) << message << ": " << _status;                \
    }

namespace rocprofiler
{
namespace thread_trace
{
constexpr size_t QUEUE_SIZE = 512;  // Small dedicated queue for SQTT control traffic

// --- signal free functions ---

hsa_signal_t
signal_create()
{
    auto  sig = hsa_signal_t{};
    auto* ext = CHECK_NOTNULL(hsa::get_amd_ext_table());
    CHECK_HSA(ext->hsa_amd_signal_create_fn(0, 0, nullptr, 0, &sig), "failed to create signal");
    return sig;
}

hsa_signal_t
signal_create(hsa_ext_amd_aql_pm4_packet_t* packet)
{
    auto sig                  = signal_create();
    packet->completion_signal = sig;
    signal_reset(sig);
    return sig;
}

void
signal_destroy(hsa_signal_t sig)
{
    signal_wait(sig);
    auto _status = CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_destroy_fn(sig);
    ROCP_CI_LOG_IF(WARNING, _status != HSA_STATUS_SUCCESS) << "Failed: " << _status;
}

void
signal_wait(hsa_signal_t sig)
{
    auto wait_fn = hsa::get_core_table()->hsa_signal_wait_scacquire_fn;
    while(wait_fn(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
        sched_yield();
}

void
signal_reset(hsa_signal_t sig)
{
    CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_store_screlease_fn(sig, 1);
}

void
signal_deleter_t::operator()(hsa_signal_t* s) const
{
    if(s)
    {
        signal_destroy(*s);
        delete s;
    }
}

signal_ptr_t
make_signal()
{
    auto* s = new hsa_signal_t{signal_create()};
    return signal_ptr_t{s};
}

signal_ptr_t
make_signal(hsa_ext_amd_aql_pm4_packet_t* packet)
{
    auto* s = new hsa_signal_t{signal_create(packet)};
    return signal_ptr_t{s};
}

// --- att_queue_t free functions ---

namespace
{
void
default_submit(const att_queue_t& q, hsa_ext_amd_aql_pm4_packet_t* packet, hsa_signal_t* completion)
{
    ROCP_TRACE << "Submit packet";
    auto* core = CHECK_NOTNULL(hsa::get_core_table());

    // NOTE: This does not check for queue-full. With QUEUE_SIZE=256 and bursts of
    // up to 6 packets per buffer swap, the producer can in theory overrun the queue
    // if the GPU stalls. In practice the GPU consumes packets fast enough relative
    // to the producer's ~2ms polling cadence. If queue overrun is observed, add a
    // load_read_index check + wait here.
    const uint64_t write_idx = core->hsa_queue_add_write_index_relaxed_fn(q.hsa_queue, 1);

    size_t index = (write_idx % q.hsa_queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* queue_slot = reinterpret_cast<uint32_t*>(size_t(q.hsa_queue->base_address) + index);

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    if(completion)
    {
        signal_reset(*completion);
        reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(queue_slot)->completion_signal =
            *completion;
    }
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    core->hsa_signal_store_screlease_fn(q.hsa_queue->doorbell_signal, write_idx);
}
}  // namespace

att_queue_t
att_queue_create(const hsa::AgentCache& agent, size_t triple_buffer_size)
{
    ROCP_TRACE << "Constructing Async queue.";

    auto q        = att_queue_t{};
    q.agent_id    = CHECK_NOTNULL(agent.get_rocp_agent())->id;
    q.buffer_size = triple_buffer_size;
    q.hsa_agent   = agent.get_hsa_agent();
    q.near_cpu    = agent.near_cpu();
    q.submit_fn   = default_submit;

    auto* core = CHECK_NOTNULL(hsa::get_core_table());
    auto* ext  = CHECK_NOTNULL(hsa::get_amd_ext_table());

    // MULTI is required because submissions arrive from multiple producers:
    //   - the producer_loop thread (triple buffering)
    //   - HSA loader threads driving load_codeobj/unload_codeobj callbacks
    // The default_submit path uses an atomic write-index increment and a release
    // store of the packet header, which is safe under MULTI. Doorbell stores from
    // racing producers may be written out of order, but the HSA runtime tolerates
    // monotonic-or-greater doorbell values, so the worst case is a slightly
    // delayed wakeup on the GPU side.
    auto status = core->hsa_queue_create_fn(q.hsa_agent,
                                            QUEUE_SIZE,
                                            HSA_QUEUE_TYPE_MULTI,
                                            nullptr,
                                            nullptr,
                                            UINT32_MAX,
                                            UINT32_MAX,
                                            &q.hsa_queue);

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create thread trace async queue";

    if(triple_buffer_size != 0)
    {
        for(auto& memory : q.triple_buffer_memory)
        {
            CHECK_HSA(ext->hsa_amd_memory_pool_allocate_fn(
                          agent.cpu_pool(), triple_buffer_size, 0, &memory),
                      "failed to allocate contiguous memory");
            CHECK_HSA(ext->hsa_amd_agents_allow_access_fn(1, &q.near_cpu, nullptr, memory),
                      "failed to allow cpu access");
            CHECK_HSA(ext->hsa_amd_agents_allow_access_fn(1, &q.hsa_agent, nullptr, memory),
                      "failed to allow gpu access");
        }
    }

    ROCP_TRACE << "Done constructing Async queue.";
    return q;
}

void
att_queue_destroy(att_queue_t& q)
{
    ROCP_TRACE << "Destroying Async Queue...";
    auto _queue_status = hsa::get_core_table()->hsa_queue_destroy_fn(q.hsa_queue);
    ROCP_WARNING_IF(_queue_status != HSA_STATUS_SUCCESS)
        << "Failed to destroy queue: " << _queue_status;

    for(auto* memory : q.triple_buffer_memory)
    {
        if(memory == nullptr) continue;
        auto _mem_status = hsa::get_amd_ext_table()->hsa_amd_memory_pool_free_fn(memory);
        ROCP_WARNING_IF(_mem_status != HSA_STATUS_SUCCESS)
            << "Failed to free memory pool: " << _mem_status;
    }
    q.hsa_queue = nullptr;
}

void
att_queue_submit(const att_queue_t&            q,
                 hsa_ext_amd_aql_pm4_packet_t* packet,
                 hsa_signal_t*                 completion)
{
    q.submit_fn(q, packet, completion);
}

signal_ptr_t
att_queue_submit(const att_queue_t& q, hsa_ext_amd_aql_pm4_packet_t* packet, bool wait)
{
    signal_ptr_t sig{nullptr};
    if(wait) sig = make_signal();

    att_queue_submit(q, packet, sig.get());
    return sig;
}

void
att_queue_deleter_t::operator()(att_queue_t* q) const
{
    if(q)
    {
        att_queue_destroy(*q);
        delete q;
    }
}

att_queue_ptr_t
make_att_queue(const hsa::AgentCache& agent, size_t triple_buffer_size)
{
    auto* q = new att_queue_t{att_queue_create(agent, triple_buffer_size)};
    return att_queue_ptr_t{q};
}

};  // namespace thread_trace
};  // namespace rocprofiler
