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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa_ven_amd_aqlprofile.h>
#include "lib/rocprofiler-sdk/hsa/internalqueue.hpp"

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            ROCP_ERROR << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace hsa
{
namespace internal_queue
{
CoreApiTable&
get_core()
{
    static CoreApiTable api{};
    return api;
}

AmdExtTable&
get_ext()
{
    static AmdExtTable api{};
    return api;
}

void
initialize(HsaApiTable* table)
{
    assert(table->core_ && table->amd_ext_);
    get_core() = *table->core_;
    get_ext()  = *table->amd_ext_;
}

Signal::Signal(void* packet)
{
    get_ext().hsa_amd_signal_create_fn(0, 0, nullptr, 0, &signal);
    reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(packet)->completion_signal = signal;
    get_core().hsa_signal_store_screlease_fn(signal, 1);
}

Signal::Signal(Signal&& other)
{
    released            = other.released.exchange(true);
    signal              = other.signal;
    other.signal.handle = 0;
}

Signal::~Signal()
{
    WaitOn();
    if(signal.handle != 0) get_core().hsa_signal_destroy_fn(signal);
}

void
Signal::WaitOn() const
{
    if(signal.handle == 0) return;

    auto* wait_fn = get_core().hsa_signal_wait_scacquire_fn;
    while(wait_fn(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
    {}
}

Queue::Queue(hsa_agent_t agent, size_t queue_size)
{
    auto status = get_core().hsa_queue_create_fn(agent,
                                                 queue_size,
                                                 HSA_QUEUE_TYPE_SINGLE,
                                                 nullptr,
                                                 nullptr,
                                                 UINT32_MAX,
                                                 UINT32_MAX,
                                                 &this->queue);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "Failed to create thread trace async queue";
        this->queue = nullptr;
    }
}

Queue::~Queue()
{
    if(this->queue) get_core().hsa_queue_destroy_fn(this->queue);
}

std::unique_ptr<Signal>
Queue::Submit(void* packet, bool bWait)
{
    const uint64_t write_idx = get_core().hsa_queue_add_write_index_relaxed_fn(queue, 1);

    size_t index = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));

    auto signal =
        bWait
            ? std::make_unique<Signal>(reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(queue_slot))
            : nullptr;
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_seq_cst);
    get_core().hsa_signal_store_screlease_fn(queue->doorbell_signal, write_idx);

    return signal;
}
}  // namespace internal_queue
}  // namespace hsa
}  // namespace rocprofiler
