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

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <cstring>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

doorbell_map_t&
get_doorbell_map()
{
    static auto*& _v = common::static_object<doorbell_map_t>::construct();
    return *_v;
}

queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue)
{
    queue_state_ptr_t result = {};
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second;
        }
    });
    return result;
}

queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    queue_state_ptr_t result = {};
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second.lock();
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    auto state = lookup_queue_state(queue);
    if(!state) return;

    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
}

void
unregister_doorbell(hsa_signal_t doorbell)
{
    get_doorbell_map().wlock([&](auto& doorbell_map) { doorbell_map.erase(doorbell.handle); });
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    return state->virtual_wptr.fetch_add(value * stride, std::memory_order_relaxed);
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    auto     prev   = state->virtual_wptr.load(std::memory_order_relaxed);
    if(stride > 1)
    {
        value = prev + ((value - prev) * stride);
    }
    state->virtual_wptr.store(value, std::memory_order_relaxed);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    if(stride > 1)
    {
        value = expected + ((value - expected) * stride);
    }
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    return state->virtual_wptr.load(std::memory_order_relaxed);
}

void
sync_metadata_impl(QueueState* compute_state,
                   const hsa_kernel_dispatch_packet_t* /*pkt*/,
                   uint64_t /*dest_pos*/)
{
    auto* meta = compute_state->metadata_state;
    if(!meta) return;

    uint64_t meta_dest    = meta->next_submit_pos;
    meta->next_submit_pos = meta_dest + 1 + compute_state->k_factor;
    __atomic_store_n(meta->real_wdid, meta->next_submit_pos, __ATOMIC_RELEASE);
}

namespace
{
thread_local QueueState* tls_state      = nullptr;
thread_local uint64_t    tls_submit_pos = 0;
thread_local uint32_t    tls_pkt_size   = 64;

void
ring_buffer_writer(const void* pkts, uint64_t pkt_count)
{
    auto*       state    = tls_state;
    auto        pkt_size = tls_pkt_size;
    const auto* src      = static_cast<const char*>(pkts);
    for(uint64_t i = 0; i < pkt_count; i++)
    {
        auto        slot = tls_submit_pos & state->ring_mask;
        auto*       dst  = static_cast<char*>(state->ring_buf) + (slot * pkt_size);
        const auto* s    = src + i * pkt_size;
        if(dst != s) memcpy(dst, s, pkt_size);
        tls_submit_pos++;
    }
}
}  // namespace

void
process_doorbell_impl(queue_state_ptr_t    state,
                      hsa_signal_value_t   value,
                      const doorbell_fn_t& ring_doorbell)
{
    if(!state) return;

    auto* state_ptr = state.get();
    std::unique_lock<std::mutex> lock{state_ptr->gate_lock};

    const uint64_t scan_pos = state_ptr->next_scan_pos;
    const uint64_t scan_end = state_ptr->virtual_wptr.load(std::memory_order_acquire);
    const uint64_t stride   = 1 + state_ptr->k_factor;

    if(scan_pos >= scan_end)
    {
        ring_doorbell(state_ptr->doorbell_signal, value);
        return;
    }

    const uint64_t total     = scan_end - scan_pos;
    const uint64_t pkt_count = total / stride;
    if(pkt_count == 0)
    {
        state_ptr->next_scan_pos = scan_end;
        ring_doorbell(state_ptr->doorbell_signal, value);
        return;
    }

    if((total % stride) != 0)
    {
        ROCP_WARNING << "Dropped partial strided write window in queue-intercept. queue="
                     << state_ptr->hsa_queue << ", scan_pos=" << scan_pos
                     << ", scan_end=" << scan_end << ", stride=" << stride;
    }

    std::vector<char> source_snapshot(pkt_count * state_ptr->pkt_size);
    for(uint64_t i = 0; i < pkt_count; ++i)
    {
        auto* src = static_cast<const char*>(state_ptr->ring_buf) +
                    (((scan_pos + i) & state_ptr->ring_mask) * state_ptr->pkt_size);
        memcpy(source_snapshot.data() + (i * state_ptr->pkt_size), src, state_ptr->pkt_size);
    }

    tls_state      = state_ptr;
    tls_submit_pos = state_ptr->next_submit_pos;
    tls_pkt_size   = state_ptr->pkt_size;

    auto*        qc    = get_queue_controller();
    const Queue* queue = (qc && state_ptr->hsa_queue) ? qc->get_queue(*state_ptr->hsa_queue) : nullptr;

    if(state_ptr->k_factor == 0)
    {
        if(queue)
        {
            queue->invoke_write_interceptor(source_snapshot.data(), pkt_count, ring_buffer_writer);
        }
        else
        {
            ring_buffer_writer(source_snapshot.data(), pkt_count);
        }
    }
    else
    {
        for(uint64_t i = 0; i < pkt_count; ++i)
        {
            auto*    pkt          = source_snapshot.data() + (i * state_ptr->pkt_size);
            uint64_t start_submit = tls_submit_pos;

            if(queue)
            {
                queue->invoke_write_interceptor(pkt, 1, ring_buffer_writer);
            }
            else
            {
                ring_buffer_writer(pkt, 1);
            }

            const uint64_t used = tls_submit_pos - start_submit;
            if(used > stride)
            {
                ROCP_WARNING << "WriteInterceptor packet expansion exceeded reserved stride. queue="
                             << state_ptr->hsa_queue << ", used=" << used
                             << ", stride=" << stride;
            }

            for(uint64_t n = used; n < stride; ++n)
            {
                auto  slot = tls_submit_pos & state_ptr->ring_mask;
                auto* dst  = static_cast<char*>(state_ptr->ring_buf) + (slot * state_ptr->pkt_size);
                memset(dst, 0, state_ptr->pkt_size);
                *reinterpret_cast<uint16_t*>(dst) =
                    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);
                ++tls_submit_pos;
            }
        }
    }

    state_ptr->next_scan_pos   = scan_end;
    state_ptr->next_submit_pos = tls_submit_pos;

    if(state_ptr->metadata_state)
    {
        for(uint64_t i = 0; i < pkt_count; ++i)
            sync_metadata_impl(state_ptr, nullptr, 0);
    }

    auto doorbell_val = static_cast<hsa_signal_value_t>(state_ptr->next_submit_pos - 1);
    auto real_rdid    = __atomic_load_n(state_ptr->real_rdid, __ATOMIC_ACQUIRE);
    auto ring_used    = (state_ptr->next_submit_pos - real_rdid);
    if(ring_used > state_ptr->ring_size)
    {
        ROCP_WARNING << "Queue-intercept observed ring usage beyond ring size. queue="
                     << state_ptr->hsa_queue << ", ring_used=" << ring_used
                     << ", ring_size=" << state_ptr->ring_size;
    }

    __atomic_store_n(state_ptr->real_wdid, state_ptr->next_submit_pos, __ATOMIC_RELEASE);
    ring_doorbell(state_ptr->doorbell_signal, doorbell_val);
}

void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   uint64_t           k_factor)
{
    auto     state         = std::make_shared<QueueState>();
    uint64_t current_wdid  = __atomic_load_n(wdid_addr, __ATOMIC_ACQUIRE);
    state->ring_buf        = queue->base_address;
    state->ring_size       = queue->size;
    state->ring_mask       = queue->size - 1;
    state->real_wdid       = wdid_addr;
    state->real_rdid       = rdid_addr;
    state->hsa_queue       = queue;
    state->doorbell_signal = queue->doorbell_signal;
    state->k_factor        = k_factor;
    state->virtual_wptr.store(current_wdid, std::memory_order_relaxed);
    state->next_scan_pos   = current_wdid;
    state->next_submit_pos = current_wdid;

    get_queue_registry().wlock([&](auto& map) { map[queue] = state; });
    get_doorbell_map().wlock([&](auto& map) { map[queue->doorbell_signal.handle] = state; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    hsa_signal_t doorbell = {0};
    queue_state_ptr_t doomed = {};
    get_queue_registry().wlock([&](auto& map) {
        auto it = map.find(queue);
        if(it == map.end()) return;
        doomed = it->second;
        if(doomed)
        {
            doorbell = doomed->doorbell_signal;
        }
        map.erase(it);
    });
    if(doorbell.handle != 0) unregister_doorbell(doorbell);
}

namespace
{
bool s_intercept_installed = false;

// Saved next-in-chain function pointers (tracing functors or raw HSA, depending on
// when install_intercept is called). Our wrappers chain through these for untracked
// queues and for the final doorbell ring on tracked queues.
CoreApiTable s_next_table = {};

// --- add_write_index wrappers (4) ---

uint64_t
wrap_add_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);
}

uint64_t
wrap_add_write_index_scacq_screl(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);
}

uint64_t
wrap_add_write_index_scacquire(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);
}

uint64_t
wrap_add_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s.get(), v);
    return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);
}

// --- store_write_index wrappers (2) ---

void
wrap_store_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
}

void
wrap_store_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    auto s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s.get(), v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
}

// --- cas_write_index wrappers (4) ---

uint64_t
wrap_cas_write_index_relaxed(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacq_screl(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacquire(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_screlease(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s.get(), expected, value);
    return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);
}

// --- load_write_index wrappers (2) ---

uint64_t
wrap_load_write_index_relaxed(const hsa_queue_t* q)
{
    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get());
    return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);
}

uint64_t
wrap_load_write_index_scacquire(const hsa_queue_t* q)
{
    auto s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s.get());
    return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);
}

// --- signal_store wrappers (2) ---

void
wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(std::move(s), val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_relaxed_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_store_relaxed_fn(sig, val);
}

void
wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(std::move(s), val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_screlease_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_store_screlease_fn(sig, val);
}

}  // namespace

bool
is_intercepting_inline()
{
    return s_intercept_installed;
}

void
install_intercept(CoreApiTable& core_table)
{
    // Save current table entries as our next-in-chain (tracing functors when called
    // after update_table, or raw HSA functions otherwise)
    s_next_table = core_table;

    core_table.hsa_queue_add_write_index_relaxed_fn     = wrap_add_write_index_relaxed;
    core_table.hsa_queue_add_write_index_scacq_screl_fn = wrap_add_write_index_scacq_screl;
    core_table.hsa_queue_add_write_index_scacquire_fn   = wrap_add_write_index_scacquire;
    core_table.hsa_queue_add_write_index_screlease_fn   = wrap_add_write_index_screlease;

    core_table.hsa_queue_store_write_index_relaxed_fn   = wrap_store_write_index_relaxed;
    core_table.hsa_queue_store_write_index_screlease_fn = wrap_store_write_index_screlease;

    core_table.hsa_queue_cas_write_index_relaxed_fn     = wrap_cas_write_index_relaxed;
    core_table.hsa_queue_cas_write_index_scacq_screl_fn = wrap_cas_write_index_scacq_screl;
    core_table.hsa_queue_cas_write_index_scacquire_fn   = wrap_cas_write_index_scacquire;
    core_table.hsa_queue_cas_write_index_screlease_fn   = wrap_cas_write_index_screlease;

    core_table.hsa_queue_load_write_index_relaxed_fn   = wrap_load_write_index_relaxed;
    core_table.hsa_queue_load_write_index_scacquire_fn = wrap_load_write_index_scacquire;

    core_table.hsa_signal_store_relaxed_fn   = wrap_signal_store_relaxed;
    core_table.hsa_signal_store_screlease_fn = wrap_signal_store_screlease;

    s_intercept_installed = true;
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
