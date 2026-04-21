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

#include <chrono>
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
        get_doorbell_map().wlock([&](auto& doorbell_map) {
            doorbell_map[doorbell.handle] = state;
            ROCP_INFO << "[DIAG-HG-DOORBELL-REGISTER] queue=" << queue
                      << " doorbell=" << doorbell.handle << " state=" << state
                      << " map_size=" << doorbell_map.size();
        });
    }
    else
    {
        ROCP_WARNING << "[DIAG-HG-DOORBELL-REGISTER] missing queue state for queue=" << queue
                     << " doorbell=" << doorbell.handle;
    }
}

void
unregister_doorbell(hsa_signal_t doorbell)
{
    get_doorbell_map().wlock([&](auto& doorbell_map) {
        auto erased = doorbell_map.erase(doorbell.handle);
        ROCP_INFO << "[DIAG-HG-DOORBELL-UNREGISTER] doorbell=" << doorbell.handle
                  << " erased=" << erased << " map_size=" << doorbell_map.size();
    });
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    auto     prev   = state->virtual_wptr.fetch_add(value * stride, std::memory_order_relaxed);
    auto count = state->add_wptr_count.fetch_add(1, std::memory_order_relaxed) + 1;
    ROCP_INFO << "[DIAG-HG-WPTR-ADD] count=" << count << " queue=" << state->hsa_queue
              << " add_value=" << value << " stride=" << stride << " prev=" << prev
              << " new=" << (prev + value * stride);
    return prev;
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    auto     prev   = state->virtual_wptr.load(std::memory_order_relaxed);
    auto     raw    = value;
    if(stride > 1)
    {
        uint64_t delta = value - prev;
        value          = prev + delta * stride;
        ROCP_INFO << "[DIAG-HG-WPTR-STORE-TRANSLATE] queue=" << state->hsa_queue
                  << " prev=" << prev << " raw_value=" << raw << " delta=" << delta
                  << " stride=" << stride << " translated_value=" << value;
    }
    state->virtual_wptr.store(value, std::memory_order_relaxed);
    auto count = state->store_wptr_count.fetch_add(1, std::memory_order_relaxed) + 1;
    ROCP_INFO << "[DIAG-HG-WPTR-STORE] count=" << count << " queue=" << state->hsa_queue
              << " prev=" << prev << " raw_value=" << raw << " stored_value=" << value
              << " stride=" << stride;
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t stride = 1 + state->k_factor;
    uint64_t raw    = value;
    if(stride > 1)
    {
        uint64_t delta = value - expected;
        value          = expected + delta * stride;
        ROCP_INFO << "[DIAG-HG-WPTR-CAS-TRANSLATE] queue=" << state->hsa_queue
                  << " expected=" << expected << " raw_value=" << raw << " delta=" << delta
                  << " stride=" << stride << " translated_value=" << value;
    }
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    auto count = state->cas_wptr_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if(prev != expected)
    {
        auto fails = state->cas_fail_count.fetch_add(1, std::memory_order_relaxed) + 1;
        ROCP_WARNING << "[DIAG-HG-WPTR-CAS-FAIL] count=" << count << " fail_count=" << fails
                     << " queue=" << state->hsa_queue << " expected=" << expected
                     << " raw_value=" << raw << " translated_value=" << value
                     << " observed_prev=" << prev << " stride=" << stride;
    }
    else
    {
        ROCP_INFO << "[DIAG-HG-WPTR-CAS] count=" << count << " queue=" << state->hsa_queue
                  << " expected=" << expected << " raw_value=" << raw
                  << " translated_value=" << value << " prev=" << prev << " stride=" << stride;
    }
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    auto v = state->virtual_wptr.load(std::memory_order_relaxed);
    ROCP_INFO << "[DIAG-HG-WPTR-LOAD] queue=" << state->hsa_queue << " value=" << v;
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
    auto decode_packet_type = [](uint16_t header) -> uint16_t {
        constexpr auto type_mask = static_cast<uint16_t>(
            ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u) << HSA_PACKET_HEADER_TYPE);
        return static_cast<uint16_t>((header & type_mask) >> HSA_PACKET_HEADER_TYPE);
    };

    auto read_packet_header = [state](uint64_t pos) -> uint16_t {
        auto* pkt = static_cast<const char*>(state->ring_buf) +
                    ((pos & state->ring_mask) * state->pkt_size);
        return *reinterpret_cast<const uint16_t*>(pkt);
    };

    auto lock_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock{state->gate_lock};
    auto lock_wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - lock_start)
                            .count();
    if(lock_wait_ns > 1000000)
    {
        ROCP_WARNING << "[DIAG-HG-GATELOCK-WAIT] queue=" << state->hsa_queue
                     << " wait_ns=" << lock_wait_ns;
    }
    else
    {
        ROCP_INFO << "[DIAG-HG-GATELOCK-WAIT] queue=" << state->hsa_queue
                  << " wait_ns=" << lock_wait_ns;
    }

    auto db_count = state->doorbell_count.fetch_add(1, std::memory_order_relaxed) + 1;

    uint64_t scan_end = state->virtual_wptr.load(std::memory_order_acquire);
    uint64_t scan_pos = state->next_scan_pos;
    uint64_t stride   = 1 + state->k_factor;
    auto     real_wdid = __atomic_load_n(state->real_wdid, __ATOMIC_ACQUIRE);
    auto     real_rdid = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
    ROCP_INFO << "[DIAG-HG-DOORBELL-ENTRY] count=" << db_count << " queue=" << state->hsa_queue
              << " signal=" << state->doorbell_signal.handle << " value=" << value
              << " scan_pos=" << scan_pos << " scan_end=" << scan_end
              << " real_wdid=" << real_wdid << " real_rdid=" << real_rdid
              << " next_submit_pos=" << state->next_submit_pos
              << " destroying=" << state->destroying.load(std::memory_order_relaxed);

    auto previous_doorbell = state->last_doorbell_value.exchange(value, std::memory_order_relaxed);
    auto doorbell_delta =
        (previous_doorbell >= 0) ? (static_cast<int64_t>(value) - previous_doorbell) : 0;
    auto virtual_advance = static_cast<int64_t>(scan_end) - static_cast<int64_t>(scan_pos);
    auto logical_advance =
        static_cast<int64_t>((stride > 0) ? ((scan_end - scan_pos) / stride) : (scan_end - scan_pos));
    auto expected_doorbell =
        (scan_end > 0) ? static_cast<hsa_signal_value_t>(scan_end - 1) : hsa_signal_value_t{-1};
    auto matches_expected = (scan_end > 0 && value == expected_doorbell);
    ROCP_INFO << "[DIAG-HG-DOORBELL-CHECK] queue=" << state->hsa_queue
              << " prev_value=" << previous_doorbell << " curr_value=" << value
              << " delta=" << doorbell_delta << " stride=" << stride
              << " scan_pos=" << scan_pos << " scan_end=" << scan_end
              << " virtual_advance=" << virtual_advance << " logical_advance=" << logical_advance
              << " expected_curr_value=" << expected_doorbell
              << " matches_expected=" << matches_expected;

    if(scan_pos >= scan_end)
    {
        ROCP_INFO << "[DIAG-HG-DOORBELL-NOOP] queue=" << state->hsa_queue
                  << " scan_pos=" << scan_pos << " scan_end=" << scan_end;
        ring_doorbell(state->doorbell_signal, value);
        return;
    }

    uint64_t pkt_count = (scan_end - scan_pos) / stride;
    uint64_t remainder = (scan_end - scan_pos) % stride;
    ROCP_INFO << "[DIAG-HG-PKTCOUNT] queue=" << state->hsa_queue << " scan_pos=" << scan_pos
              << " scan_end=" << scan_end << " stride=" << stride << " pkt_count=" << pkt_count
              << " remainder=" << remainder;
    if(remainder != 0)
    {
        auto rem_count = state->remainder_count.fetch_add(1, std::memory_order_relaxed) + 1;
        ROCP_ERROR << "[DIAG-HG-REMAINDER-DROP] queue=" << state->hsa_queue
                   << " remainder=" << remainder << " count=" << rem_count
                   << " scan_range=[" << scan_pos << "," << scan_end << ")";
    }

    // Set up TLS for ring_buffer_writer
    tls_state      = state;
    tls_submit_pos = state->next_submit_pos;
    tls_pkt_size   = state->pkt_size;

    // Look up Queue* to invoke WriteInterceptor callback chain
    auto*        qc    = get_queue_controller();
    const Queue* queue = (qc && state->hsa_queue) ? qc->get_queue(*state->hsa_queue) : nullptr;
    ROCP_INFO << "[DIAG-HG-QUEUE-LOOKUP] queue=" << state->hsa_queue
              << " lookup_success=" << (queue != nullptr);

    if(state->k_factor == 0)
    {
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
        for(uint64_t i = 0; i < pkt_count; i++)
        {
            uint64_t pkt_pos = scan_pos + i * stride;
            uint64_t contig_pos = scan_pos + i;

            auto stride_header = read_packet_header(pkt_pos);
            auto stride_type   = decode_packet_type(stride_header);
            auto contig_header = read_packet_header(contig_pos);
            auto contig_type   = decode_packet_type(contig_header);
            ROCP_INFO << "[DIAG-HG-SLOT-COMPARE] queue=" << state->hsa_queue
                      << " logical_idx=" << i << " stride=" << stride << " stride_pos=" << pkt_pos
                      << " stride_slot=" << (pkt_pos & state->ring_mask)
                      << " stride_type=" << stride_type << " stride_header=" << stride_header
                      << " contig_pos=" << contig_pos
                      << " contig_slot=" << (contig_pos & state->ring_mask)
                      << " contig_type=" << contig_type << " contig_header=" << contig_header;

            for(uint64_t rel = 1; rel < stride; ++rel)
            {
                auto reserved_pos    = pkt_pos + rel;
                auto reserved_header = read_packet_header(reserved_pos);
                auto reserved_type   = decode_packet_type(reserved_header);
                auto is_suspicious   = (reserved_header != 0) &&
                                     (reserved_type != HSA_PACKET_TYPE_INVALID) &&
                                     (reserved_type != HSA_PACKET_TYPE_BARRIER_AND);
                if(is_suspicious)
                {
                    ROCP_WARNING << "[DIAG-HG-RESERVED-SLOT-WRITTEN] queue=" << state->hsa_queue
                                 << " logical_idx=" << i << " base_pkt_pos=" << pkt_pos
                                 << " reserved_rel=" << rel << " reserved_pos=" << reserved_pos
                                 << " reserved_slot=" << (reserved_pos & state->ring_mask)
                                 << " reserved_type=" << reserved_type
                                 << " reserved_header=" << reserved_header;
                }
                if(reserved_type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
                {
                    ROCP_ERROR << "[DIAG-HG-RESERVED-SLOT-KERNEL] queue=" << state->hsa_queue
                               << " logical_idx=" << i << " base_pkt_pos=" << pkt_pos
                               << " reserved_rel=" << rel << " reserved_pos=" << reserved_pos
                               << " reserved_slot=" << (reserved_pos & state->ring_mask)
                               << " reserved_header=" << reserved_header;
                }
            }

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
            ROCP_INFO << "[DIAG-HG-WI-USED] queue=" << state->hsa_queue << " pkt_pos=" << pkt_pos
                      << " used=" << used << " stride=" << stride
                      << " start_submit=" << start_submit << " end_submit=" << tls_submit_pos;

            if(used > stride)
            {
                auto over_count = state->over_stride_count.fetch_add(1, std::memory_order_relaxed) + 1;
                ROCP_ERROR << "[DIAG-HG-OVER-STRIDE] queue=" << state->hsa_queue
                           << " pkt_pos=" << pkt_pos << " used=" << used
                           << " stride=" << stride << " count=" << over_count;
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
    ROCP_INFO << "[DIAG-HG-DOORBELL-AFTER-WRITE] queue=" << state->hsa_queue
              << " next_scan_pos=" << state->next_scan_pos
              << " next_submit_pos=" << state->next_submit_pos;

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

    auto doorbell_val = static_cast<hsa_signal_value_t>(state->next_submit_pos - 1);
    real_rdid         = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);
    auto ring_used    = (state->next_submit_pos - real_rdid);
    ROCP_INFO << "[DIAG-HG-DOORBELL-SUBMIT] queue=" << state->hsa_queue
              << " real_wdid_submit=" << state->next_submit_pos << " doorbell_val=" << doorbell_val
              << " real_rdid=" << real_rdid << " doorbell=" << state->doorbell_signal.handle
              << " ring_used=" << ring_used << " ring_size=" << state->ring_size;
    if(ring_used > state->ring_size)
    {
        ROCP_ERROR << "[DIAG-HG-RING-OVERFLOW-RISK] queue=" << state->hsa_queue
                   << " ring_used=" << ring_used << " ring_size=" << state->ring_size;
    }
    __atomic_store_n(state->real_wdid, state->next_submit_pos, __ATOMIC_RELEASE);
    ring_doorbell(state->doorbell_signal, doorbell_val);
    ROCP_INFO << "[DIAG-HG-DOORBELL-EXIT] queue=" << state->hsa_queue
              << " next_scan_pos=" << state->next_scan_pos
              << " next_submit_pos=" << state->next_submit_pos;
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

    ROCP_INFO << "[DIAG-HG-QSTATE-CREATE] queue=" << queue << " ring_buf=" << queue->base_address
              << " ring_size=" << queue->size << " ring_mask=" << (queue->size - 1)
              << " k_factor=" << k_factor << " doorbell=" << queue->doorbell_signal.handle
              << " real_wdid_ptr=" << wdid_addr << " real_rdid_ptr=" << rdid_addr
              << " initial_wdid=" << current_wdid;

    auto* raw_ptr = state.get();
    get_queue_registry().wlock([&](auto& map) {
        map[queue] = std::move(state);
        ROCP_INFO << "[DIAG-HG-QSTATE-CREATE-MAP] queue_registry_size=" << map.size();
    });
    get_doorbell_map().wlock([&](auto& map) {
        map[queue->doorbell_signal.handle] = raw_ptr;
        ROCP_INFO << "[DIAG-HG-QSTATE-CREATE-MAP] doorbell_map_size=" << map.size();
    });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    ROCP_INFO << "[DIAG-HG-QSTATE-DESTROY-BEGIN] queue=" << queue;
    hsa_signal_t doorbell = {0};
    get_queue_registry().rlock([&](const auto& map) {
        auto it = map.find(queue);
        if(it != map.end() && it->second)
        {
            it->second->destroying.store(true, std::memory_order_relaxed);
            ROCP_INFO << "[DIAG-HG-QSTATE-DESTROY-FLAG] queue=" << queue << " state="
                      << it->second.get() << " doorbell=" << it->second->doorbell_signal.handle;
        }
    });
    get_queue_registry().wlock([&](auto& map) {
        ROCP_INFO << "[DIAG-HG-QSTATE-DESTROY-MAP] queue_registry_size_before=" << map.size();
        auto it = map.find(queue);
        if(it == map.end()) return;
        doorbell = it->second->doorbell_signal;
        map.erase(it);
        ROCP_INFO << "[DIAG-HG-QSTATE-DESTROY-MAP] queue_registry_size_after=" << map.size();
    });
    if(doorbell.handle != 0) unregister_doorbell(doorbell);
    get_doorbell_map().rlock([&](const auto& map) {
        ROCP_INFO << "[DIAG-HG-QSTATE-DESTROY-END] queue=" << queue
                  << " doorbell_map_size=" << map.size();
    });
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
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=add_relaxed queue=" << q << " value=" << v;
    return s_next_table.hsa_queue_add_write_index_relaxed_fn(q, v);
}

uint64_t
wrap_add_write_index_scacq_screl(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=add_scacq_screl queue=" << q
                 << " value=" << v;
    return s_next_table.hsa_queue_add_write_index_scacq_screl_fn(q, v);
}

uint64_t
wrap_add_write_index_scacquire(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=add_scacquire queue=" << q
                 << " value=" << v;
    return s_next_table.hsa_queue_add_write_index_scacquire_fn(q, v);
}

uint64_t
wrap_add_write_index_screlease(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=add_screlease queue=" << q
                 << " value=" << v;
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
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=store_relaxed queue=" << q
                 << " value=" << v;
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
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=store_screlease queue=" << q
                 << " value=" << v;
    s_next_table.hsa_queue_store_write_index_screlease_fn(q, v);
}

// --- cas_write_index wrappers (4) ---

uint64_t
wrap_cas_write_index_relaxed(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=cas_relaxed queue=" << q
                 << " expected=" << expected << " value=" << value;
    return s_next_table.hsa_queue_cas_write_index_relaxed_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacq_screl(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=cas_scacq_screl queue=" << q
                 << " expected=" << expected << " value=" << value;
    return s_next_table.hsa_queue_cas_write_index_scacq_screl_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_scacquire(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=cas_scacquire queue=" << q
                 << " expected=" << expected << " value=" << value;
    return s_next_table.hsa_queue_cas_write_index_scacquire_fn(q, expected, value);
}

uint64_t
wrap_cas_write_index_screlease(const hsa_queue_t* q, uint64_t expected, uint64_t value)
{
    auto* s = lookup_queue_state(q);
    if(s) return cas_write_index_impl(s, expected, value);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=cas_screlease queue=" << q
                 << " expected=" << expected << " value=" << value;
    return s_next_table.hsa_queue_cas_write_index_screlease_fn(q, expected, value);
}

// --- load_write_index wrappers (2) ---

uint64_t
wrap_load_write_index_relaxed(const hsa_queue_t* q)
{
    auto* s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=load_relaxed queue=" << q;
    return s_next_table.hsa_queue_load_write_index_relaxed_fn(q);
}

uint64_t
wrap_load_write_index_scacquire(const hsa_queue_t* q)
{
    auto* s = lookup_queue_state(q);
    if(s) return load_write_index_impl(s);
    ROCP_WARNING << "[DIAG-HG-WPTR-PASSTHROUGH] op=load_scacquire queue=" << q;
    return s_next_table.hsa_queue_load_write_index_scacquire_fn(q);
}

// --- signal_store wrappers (2) ---

void
wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        ROCP_INFO << "[DIAG-HG-DOORBELL-RELAXED] sig=" << sig.handle << " val=" << val
                  << " state=" << s;
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_relaxed_fn(db, v);
        });
        return;
    }
    ROCP_WARNING << "[DIAG-HG-DOORBELL-LOOKUP-MISS] relaxed sig=" << sig.handle << " val=" << val;
    s_next_table.hsa_signal_store_relaxed_fn(sig, val);
}

void
wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        ROCP_INFO << "[DIAG-HG-DOORBELL-SCRELEASE] sig=" << sig.handle << " val=" << val
                  << " state=" << s;
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            s_next_table.hsa_signal_store_screlease_fn(db, v);
        });
        return;
    }
    ROCP_WARNING << "[DIAG-HG-DOORBELL-LOOKUP-MISS] screlease sig=" << sig.handle
                 << " val=" << val;
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
    ROCP_INFO << "[DIAG-HG-INLINE-INSTALL] inline queue intercept installed (14 API wrappers)";
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
