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

QueueState*
lookup_queue_state(const hsa_queue_t* queue)
{
    QueueState* result = nullptr;
    get_queue_registry().rlock([&](const auto& registry) {
        auto it = registry.find(queue);
        if(it != registry.end())
        {
            result = it->second.get();
        }
    });
    return result;
}

QueueState*
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    QueueState* result = nullptr;
    get_doorbell_map().rlock([&](const auto& doorbell_map) {
        auto it = doorbell_map.find(signal.handle);
        if(it != doorbell_map.end())
        {
            result = it->second;
        }
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    QueueState* state = lookup_queue_state(queue);
    if(state)
    {
        get_doorbell_map().wlock(
            [&](auto& doorbell_map) { doorbell_map[doorbell.handle] = state; });
    }
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
    auto     prev   = state->virtual_wptr.fetch_add(value * stride, std::memory_order_relaxed);
    ROCP_TRACE << "add_write_index: queue=" << state->hsa_queue << " +=" << value
               << " stride=" << stride << " prev=" << prev << " new=" << (prev + value * stride);
    return prev;
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    auto prev = state->virtual_wptr.load(std::memory_order_relaxed);
    state->virtual_wptr.store(value, std::memory_order_relaxed);
    ROCP_TRACE << "store_write_index: queue=" << state->hsa_queue << " prev=" << prev
               << " new=" << value;
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    ROCP_TRACE << "cas_write_index: queue=" << state->hsa_queue << " expected=" << expected
               << " value=" << value << " prev=" << prev
               << (prev == expected ? " (swapped)" : " (failed)");
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    auto v = state->virtual_wptr.load(std::memory_order_relaxed);
    ROCP_TRACE << "load_write_index: queue=" << state->hsa_queue << " val=" << v;
    return v;
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
    ROCP_TRACE << "ring_buffer_writer: pkt_count=" << pkt_count << " submit_pos=" << tls_submit_pos
               << " k_factor=" << state->k_factor;
    for(uint64_t i = 0; i < pkt_count; i++)
    {
        auto        slot = tls_submit_pos & state->ring_mask;
        auto*       dst  = static_cast<char*>(state->ring_buf) + (slot * pkt_size);
        const auto* s    = src + i * pkt_size;
        if(dst != s) memcpy(dst, s, pkt_size);
        ROCP_TRACE << "  pkt[" << i << "] -> slot=" << slot << " submit_pos=" << tls_submit_pos;
        tls_submit_pos++;
    }
    ROCP_TRACE << "ring_buffer_writer done: final submit_pos=" << tls_submit_pos;
}
}  // namespace

void
process_doorbell_impl(QueueState*          state,
                      hsa_signal_value_t   value,
                      const doorbell_fn_t& ring_doorbell)
{
    std::lock_guard<std::mutex> lock(state->gate_lock);

    uint64_t scan_end = state->virtual_wptr.load(std::memory_order_acquire);
    uint64_t scan_pos = state->next_scan_pos;

    if(scan_pos >= scan_end)
    {
        ROCP_TRACE << "doorbell: no new packets (scan_pos=" << scan_pos << ")";
        ring_doorbell(state->doorbell_signal, value);
        return;
    }

    uint64_t stride    = 1 + state->k_factor;
    uint64_t pkt_count = (scan_end - scan_pos) / stride;
    ROCP_TRACE << "doorbell: processing " << pkt_count << " packets [" << scan_pos << ".."
               << scan_end << ") stride=" << stride;

    // DIAG: H2 — detect stride mismatch where app advanced virtual_wptr by non-stride amount
    if(pkt_count == 0 && scan_end > scan_pos)
    {
        ROCP_FATAL << "DIAG-H2: pkt_count=0 but pending data! scan_pos=" << scan_pos
                   << " scan_end=" << scan_end << " stride=" << stride
                   << " remainder=" << ((scan_end - scan_pos) % stride)
                   << " gap=" << (scan_end - scan_pos) << " queue=" << state->hsa_queue;
    }

    // Set up TLS for ring_buffer_writer
    tls_state      = state;
    tls_submit_pos = state->next_submit_pos;
    tls_pkt_size   = state->pkt_size;

    // Look up Queue* to invoke WriteInterceptor callback chain
    auto*        qc    = get_queue_controller();
    const Queue* queue = (qc && state->hsa_queue) ? qc->get_queue(*state->hsa_queue) : nullptr;

    if(state->k_factor == 0)
    {
        // k_factor=0: pass entire batch to WriteInterceptor at once.
        // WriteInterceptor may expand packets (e.g. adding completion signals,
        // barrier packets for kernel tracing). With stride=1 there are no reserved
        // expansion slots, so we let ring_buffer_writer write sequentially and
        // update real_wdid to cover all output packets.
        auto* pkt =
            static_cast<char*>(state->ring_buf) + ((scan_pos & state->ring_mask) * state->pkt_size);
        if(queue)
        {
            queue->invoke_write_interceptor(pkt, pkt_count, ring_buffer_writer);
        }
        else
        {
            ring_buffer_writer(pkt, pkt_count);
        }
    }
    else
    {
        // k_factor>0: process each packet individually with stride padding.
        // Each app packet occupies 1 slot, the remaining stride-1 slots are
        // filled with barrier_and (or used by WriteInterceptor for profiling).
        for(uint64_t i = 0; i < pkt_count; i++)
        {
            uint64_t pkt_pos = scan_pos + i * stride;
            auto*    pkt     = static_cast<char*>(state->ring_buf) +
                        ((pkt_pos & state->ring_mask) * state->pkt_size);
            uint64_t start_submit = tls_submit_pos;
            if(queue)
            {
                queue->invoke_write_interceptor(pkt, 1, ring_buffer_writer);
            }
            else
            {
                ring_buffer_writer(pkt, 1);
            }

            uint64_t used = tls_submit_pos - start_submit;

            // DIAG: H3 — detect expansion exceeding stride
            if(used > stride)
            {
                ROCP_FATAL << "DIAG-H3: used(" << used << ") > stride(" << stride
                           << ") — spillover corruption! pkt_pos=" << pkt_pos
                           << " queue=" << state->hsa_queue;
            }

            if(used < stride)
            {
                uint64_t remaining = stride - used;
                for(uint64_t k = 0; k < remaining; k++)
                {
                    auto  kslot = tls_submit_pos & state->ring_mask;
                    auto* kdst  = static_cast<char*>(state->ring_buf) + (kslot * state->pkt_size);
                    memset(kdst, 0, state->pkt_size);
                    *reinterpret_cast<uint16_t*>(kdst) =
                        (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);
                    tls_submit_pos++;
                }
            }
        }
    }

    state->next_scan_pos   = scan_end;
    state->next_submit_pos = tls_submit_pos;
    ROCP_TRACE << "doorbell: after interceptor submit_pos=" << state->next_submit_pos;

    // Sync paired metadata queue (one sync per application packet)
    if(state->metadata_state)
    {
        for(uint64_t i = 0; i < pkt_count; i++)
        {
            uint64_t    pkt_pos = scan_pos + i * stride;
            const auto* pkt     = reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(
                static_cast<char*>(state->ring_buf) +
                ((pkt_pos & state->ring_mask) * state->pkt_size));
            sync_metadata_impl(state, pkt, 0);
        }
    }

    // DIAG: H4 — detect ring overflow
    {
        auto diag_rdid = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
        auto diag_used = state->next_submit_pos - diag_rdid;
        if(diag_used > state->ring_size)
        {
            ROCP_FATAL << "DIAG-H4: ring overflow! submit_pos=" << state->next_submit_pos
                       << " rdid=" << diag_rdid << " used=" << diag_used
                       << " ring_size=" << state->ring_size << " queue=" << state->hsa_queue;
        }
    }

    auto doorbell_val = static_cast<hsa_signal_value_t>(state->next_submit_pos - 1);
    auto real_rdid    = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
    ROCP_TRACE << "doorbell: submitting real_wdid=" << state->next_submit_pos
               << " doorbell_val=" << doorbell_val << " real_rdid=" << real_rdid
               << " doorbell=" << state->doorbell_signal.handle
               << " ring_used=" << (state->next_submit_pos - real_rdid)
               << " ring_size=" << state->ring_size;
    __atomic_store_n(state->real_wdid, state->next_submit_pos, __ATOMIC_RELEASE);
    ring_doorbell(state->doorbell_signal, doorbell_val);
    ROCP_TRACE << "doorbell: ring_doorbell returned";
}

void
create_queue_state(const hsa_queue_t* queue,
                   volatile uint64_t* wdid_addr,
                   volatile uint64_t* rdid_addr,
                   uint64_t           k_factor)
{
    auto     state         = std::make_unique<QueueState>();
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

    ROCP_INFO << "create_queue_state: queue=" << queue << " ring_size=" << queue->size
              << " k_factor=" << k_factor << " doorbell=" << queue->doorbell_signal.handle
              << " initial_wdid=" << current_wdid;

    auto* raw_ptr = state.get();
    get_queue_registry().wlock([&](auto& map) { map[queue] = std::move(state); });
    get_doorbell_map().wlock([&](auto& map) { map[queue->doorbell_signal.handle] = raw_ptr; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    ROCP_INFO << "destroy_queue_state: queue=" << queue;
    hsa_signal_t doorbell = {0};
    get_queue_registry().wlock([&](auto& map) {
        auto it = map.find(queue);
        if(it == map.end()) return;
        doorbell = it->second->doorbell_signal;
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
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    ROCP_TRACE << "add_write_index_relaxed PASSTHROUGH: queue=" << q << " v=" << v;
    return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);
}

uint64_t
wrap_add_write_index_scacq_screl(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);
}

uint64_t
wrap_add_write_index_scacquire(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);
}

uint64_t
wrap_add_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    return s_next_table.hsa_queue_add_write_index_screlease_fn(q, v);
}

// --- store_write_index wrappers (2) ---

void
wrap_store_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s, v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_relaxed_fn(q, v);
}

void
wrap_store_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s)
    {
        store_write_index_impl(s, v);
        return;
    }
    s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
}

// --- cas_write_index wrappers (4) ---

uint64_t
wrap_cas_write_index_relaxed(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacq_screl(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacquire(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_screlease(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);
}

// --- load_write_index wrappers (2) ---

uint64_t
wrap_load_write_index_relaxed(const hsa_queue_t* q)
{
    auto* s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s);
    return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);
}

uint64_t
wrap_load_write_index_scacquire(const hsa_queue_t* q)
{
    auto* s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s);
    return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);
}

// --- signal_store wrappers (2) ---

void
wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        ROCP_TRACE << "doorbell_relaxed: sig=" << sig.handle << " val=" << val;
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_relaxed_fn(db, v);
        });
        return;
    }
    s_next_table.hsa_signal_store_relaxed_fn(sig, val);
}

void
wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        ROCP_TRACE << "doorbell_screlease: sig=" << sig.handle << " val=" << val;
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_screlease_fn(db, v);
        });
        return;
    }
    ROCP_TRACE << "signal_store_screlease passthrough: sig=" << sig.handle << " val=" << val;
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
    ROCP_INFO << "inline queue intercept installed (14 API wrappers)";
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
