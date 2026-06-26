// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// SDK-level HSA queue interposition: wraps hsa_queue_*_write_index_* and
// hsa_signal_store_* to virtualize the queue write pointer. Producer threads
// advance QueueState::virtual_wptr; the real write_dispatch_id only advances
// at doorbell time after process_doorbell_impl runs the WriteInterceptor chain.
// Tracing-only; the gate in registration.cpp forces the legacy
// hsa_amd_queue_intercept_create path whenever a context registers
// dispatch_counter_collection, dispatch_thread_trace, or pc_sampler.
// See queue_interposition.hpp for the API.

#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/common/container/pool.hpp"
#include "lib/common/container/pool_object.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <pthread.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_interposition
{
namespace
{
// NOTE:
//  - "installed" is for checking whether HSA functions have been passed
//  - "active" is for controlling whether wrappers are intercepting or passing through
//  - "dynamic" is for whether to allow dynamic discovery of queues whose creation was not
//      observed/intercepted. E.g., during attachment, we want to toggle this on.
auto s_intercept_installed = std::atomic<bool>{false};  // installed (may not be active)
auto s_intercept_active    = std::atomic<bool>{false};  // actively intercepting
auto s_intercept_dynamic   = std::atomic<bool>{false};  // dynamically add queue states

bool
should_bypass_inline_intercept()
{
    return (!s_intercept_installed.load(std::memory_order_acquire) ||
            !s_intercept_active.load(std::memory_order_acquire) ||
            registration::get_fini_status() != 0 ||
            // TODO: debug and enable queue interposition for attachment
            registration::supports_attachment());
}

auto*&
get_original_table()
{
    static CoreApiTable* _v = nullptr;
    return _v;
}

// Saved next-in-chain function pointers (tracing functors or raw HSA, depending on
// when install_intercept is called). Our wrappers chain through these for untracked
// queues and for the final doorbell ring on tracked queues.
auto*
get_next_table()
{
    static auto*& _v = common::static_object<CoreApiTable>::construct();
    return _v;
}
}  // namespace

queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

queue_state_ptr_t
lookup_queue_state(const hsa_queue_t* queue, bool create_if_missing)
{
    auto _state = get_queue_registry().rlock([&](const auto& registry) -> queue_state_ptr_t {
        if(auto it = registry.find(queue); it != registry.end()) return it->second;
        return queue_state_ptr_t{};
    });

    // if create_if_missing is true, create a new state. this is for dynamic discovery of queues.
    if(!_state && create_if_missing)
    {
        return create_queue_state(queue, true);
    }

    return _state;
}

queue_state_ptr_t
lookup_queue_state_by_doorbell(hsa_signal_t signal, bool create_if_missing)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto* _amd_signal = reinterpret_cast<amd_signal_t*>(signal.handle);

    if(!_amd_signal) return queue_state_ptr_t{};

    // Only doorbell-kind signals carry a valid queue_ptr (it aliases reserved2 otherwise).
    if(_amd_signal->kind != AMD_SIGNAL_KIND_DOORBELL &&
       _amd_signal->kind != AMD_SIGNAL_KIND_LEGACY_DOORBELL)
        return queue_state_ptr_t{};

    if(_amd_signal->queue_ptr)
        return lookup_queue_state(reinterpret_cast<const hsa_queue_t*>(_amd_signal->queue_ptr),
                                  create_if_missing);

    return queue_state_ptr_t{};
}

uint64_t
add_write_index_impl(QueueState* state, uint64_t value, std::memory_order order)
{
    return state->virtual_wptr.fetch_add(value, order);
}

void
store_write_index_impl(QueueState* state, uint64_t value, std::memory_order order)
{
    state->virtual_wptr.store(value, order);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value, std::memory_order order)
{
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, order);
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state, std::memory_order order)
{
    return state->virtual_wptr.load(order);
}

namespace
{
// CPU pause hint for short spin-waits (cheaper than yield/sleep, no added latency).
inline void
cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Per-thread handoff from process_doorbell_impl() to ring_buffer_writer().
struct doorbell_tls_t
{
    QueueState*          state                     = nullptr;
    uint64_t             submit_pos                = 0;
    uint32_t             pkt_size                  = 64;
    const doorbell_fn_t* ring_doorbell             = nullptr;
    uint64_t             last_published_submit_pos = 0;
};

doorbell_tls_t&
get_doorbell_tls()
{
    static thread_local auto _v = doorbell_tls_t{};
    return _v;
}

inline void
publish_submitted_packets(QueueState* state, uint64_t submit_pos)
{
    auto& tls = get_doorbell_tls();
    if(!tls.ring_doorbell || submit_pos <= tls.last_published_submit_pos || submit_pos == 0) return;

    // submit_pos must never regress below what we already published (corruption); fatal in CI.
    ROCP_CI_LOG_IF(WARNING, submit_pos < tls.last_published_submit_pos)
        << "publish_submitted_packets: submit_pos (" << submit_pos
        << ") regressed below last_published_submit_pos (" << tls.last_published_submit_pos << ")";

    __atomic_store_n(state->real_wdid, submit_pos, __ATOMIC_RELEASE);
    (*tls.ring_doorbell)(state->doorbell_signal, static_cast<hsa_signal_value_t>(submit_pos - 1));
    tls.last_published_submit_pos = submit_pos;
}

// Ring the doorbell with the last index we have actually submitted (next_submit_pos - 1),
// never the application's virtualized value, which may point past it and make the GPU
// consume unpublished ring slots.
inline void
ring_published_doorbell(QueueState* state, const doorbell_fn_t& ring_doorbell)
{
    const uint64_t published = state->next_submit_pos;
    if(published == 0) return;
    ring_doorbell(state->doorbell_signal, static_cast<hsa_signal_value_t>(published - 1));
}

inline void
wait_for_free_slot(QueueState* state, uint64_t submit_pos)
{
    while(true)
    {
        auto real_rdid = __atomic_load_n(state->real_rdid, __ATOMIC_ACQUIRE);

        // Guard the unsigned subtraction: if real_rdid has reached or passed our write
        // position the ring has free space. Otherwise (submit_pos - real_rdid) would
        // underflow and spin forever while holding gate_lock.
        if(real_rdid >= submit_pos || (submit_pos - real_rdid) < state->ring_size)
        {
            return;
        }

        // If the producer is blocked on a full ring and has already written
        // packets beyond the last visible write index, publish progress so the
        // consumer can observe and drain them.
        publish_submitted_packets(state, submit_pos);
        cpu_relax();
    }
}

void
ring_buffer_writer(const void* pkts, uint64_t pkt_count)
{
    auto&       tls      = get_doorbell_tls();
    auto*       state    = tls.state;
    auto        pkt_size = tls.pkt_size;
    const auto* src      = static_cast<const char*>(pkts);
    for(uint64_t i = 0; i < pkt_count; i++)
    {
        wait_for_free_slot(state, tls.submit_pos);
        auto        slot = tls.submit_pos & state->ring_mask;
        auto*       dst  = static_cast<char*>(state->ring_buf) + (slot * pkt_size);
        const auto* s    = src + i * pkt_size;
        if(dst != s)
        {
            constexpr auto header_size = sizeof(uint16_t);
            if(pkt_size > header_size)
            {
                ::memcpy(dst + header_size, s + header_size, pkt_size - header_size);
                uint16_t header = 0;
                ::memcpy(&header, s, header_size);
                __atomic_store_n(reinterpret_cast<uint16_t*>(dst), header, __ATOMIC_RELEASE);
            }
            else
            {
                ::memcpy(dst, s, pkt_size);
            }
        }
        tls.submit_pos++;
    }
}

auto
async_signal_handler_exists()
{
    return common::static_object<internal_threading::task_group_t>::get();
}

internal_threading::task_group_t*
get_async_signal_handler()
{
    using task_group_t           = internal_threading::task_group_t;
    using create_task_group_fn_t = task_group_t* (*) (void*, size_t);

    // default to 4 threads if neither GPU_MAX_HW_QUEUES or ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS
    // is set, since the async signal handler is primarily intended for handling queue completion
    // signals and a typical GPU may have on the order of 4 hardware queues. Note: GPU_MAX_HW_QUEUES
    // is a ROCr/HSA environment variable. If GPU_MAX_HW_QUEUES is set but
    // ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS is not set, we will use the value of
    // GPU_MAX_HW_QUEUES to determine the number of threads for the async signal handler. If
    // ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS is set, it will take precedence over
    // GPU_MAX_HW_QUEUES.
    static auto*& _v =
        common::static_object<internal_threading::task_group_t>::construct_via_function(
            static_cast<create_task_group_fn_t>(&internal_threading::create_task_group),
            common::get_env("ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS",
                            common::get_env("GPU_MAX_HW_QUEUES", 4)));

    return _v;
}

bool
context_filter(const context::context* ctx)
{
    return (ctx->is_tracing_one_of(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
}

template <typename Integral>
Integral
bit_extract(Integral x, int first, int last)
{
    static_assert(std::is_integral<Integral>::value, "Integral type required");

    auto&& bit_mask = [](int _first, int _last) {
        ROCP_FATAL_IF(!(_last >= _first)) << fmt::format(
            "[queue::bit_extract::bit_mask] -> invalid argument. last (={}) is not >= first (={})",
            _last,
            _first);

        size_t num_bits = _last - _first + 1;
        return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                                   /* num_bits exceed the size of Integral */
                                                   : ((Integral{1} << num_bits) - 1))
               << _first;
    };

    return (x >> first) & bit_mask(0, last - first);
}

void
async_signal_handler(hsa_signal_t                            completion_signal,
                     hsa_signal_value_t                      starting_value,
                     std::shared_ptr<queue_info_session_t>&& session)
{
    constexpr auto timeout_hint =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds{10});

    auto signal_value = starting_value;
    auto niterations  = uint64_t{0};

    // Stop only on completion or finalization; never run cleanup while the kernel is live.
    while(true)
    {
        signal_value = get_core_table()->hsa_signal_wait_relaxed_fn(completion_signal,
                                                                    HSA_SIGNAL_CONDITION_LT,
                                                                    starting_value,
                                                                    timeout_hint.count(),
                                                                    HSA_WAIT_STATE_ACTIVE);

        if(signal_value < starting_value) break;         // kernel completed
        if(registration::get_fini_status() != 0) break;  // tearing down: run cleanup path
        ++niterations;

        // Surface long-running waits for diagnostics without giving up the wait.
        constexpr auto warn_interval = (1UL << 20);
        if(niterations % warn_interval == 0)
            ROCP_WARNING << fmt::format(
                "Async signal handler still waiting on signal {{.handle={}}} after {} iterations "
                "(value={}, starting_value={})",
                completion_signal.handle,
                niterations,
                signal_value,
                starting_value);
    }

    ROCP_INFO << fmt::format("Async signal handler invoked for signal {{.handle={}}} with "
                             "value {} (original value={}, iterations={})",
                             completion_signal.handle,
                             signal_value,
                             starting_value,
                             niterations);

    if(auto delay_us = common::get_env("ROCPROFILER_TEST_INLINE_ASYNC_DELAY_US", 0); delay_us > 0)
    {
        std::this_thread::sleep_for(std::chrono::microseconds{delay_us});
    }

    for(auto& packet : session->packet_data)
    {
        auto dispatch_time = kernel_dispatch::get_dispatch_time(*session, packet);
        kernel_dispatch::dispatch_complete(*session, packet, dispatch_time);

        // if the completion signal was from the pool, we just release it back to the pool for
        // reuse.
        if(packet.pooled_signal)
        {
            Queue::release_signal(packet.pooled_signal);
        }
        else
        {
            // if the signal was not from the pool, we need to decrement the signal value to clean
            // up the signal for the application
            get_core_table()->hsa_signal_subtract_relaxed_fn(packet.completion_signal, 1);
        }

        // we need to decrement this reference count at the end of the functions
        auto* _corr_id = session->correlation_id;
        if(_corr_id)
        {
            ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
                << "reference counter for correlation id " << _corr_id->internal << " from thread "
                << _corr_id->thread_idx << " has no reference count";
            _corr_id->sub_kern_count();
            _corr_id->sub_ref_count();
        }
    }
}

// Local kernel-dispatch tracing path: swaps in pooled completion signals,
// runs KERNEL_DISPATCH_ENQUEUE tracer hooks, and enqueues a completion-signal
// waiter on the async signal handler pool. Strict 1:1 packet forwarding; does
// not insert PM4 packets. Distinct from Queue::WriteInterceptor (legacy path).
void
write_interceptor(Queue*                                queue,
                  const void*                           packets,
                  uint64_t                              pkt_count,
                  hsa_amd_queue_intercept_packet_writer writer)
{
    using callback_record_t = packet_data_t::callback_record_t;
    using packet_vector_t   = common::container::small_vector<rocprofiler_packet, 512>;

    if(registration::get_fini_status() > 0)
    {
        writer(packets, pkt_count);
        return;
    }

    ROCP_INFO << fmt::format("write_interceptor called with pkt_count={}", pkt_count);

    auto _contexts = context::get_active_contexts(context_filter);

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 || _contexts.empty())
    {
        writer(packets, pkt_count);
        return;
    }

    // unique sequence id for the dispatch (global across all queues, matches SDK contract)
    static auto sequence_counter = std::atomic<rocprofiler_dispatch_id_t>{0};

    const auto* packets_arr          = static_cast<const rocprofiler_packet*>(packets);
    auto        num_dispatch_packets = size_t{0};
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            ++num_dispatch_packets;
        }
    }

    if(num_dispatch_packets == 0)
    {
        writer(packets, pkt_count);
        return;
    }

    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               tracing_data_v);

    // all packets should have the same correlation id so we can just look at the first one to
    // get the correlation id for the entire batch of packets
    auto*                    corr_id      = context::get_latest_correlation_id();
    context::correlation_id* _corr_id_pop = nullptr;

    // Allocate a correlation id if we have at least one dispatch packet and we don't have a
    // correlation id already. There will not be a correlation id if there is no API tracing but
    // it was requested by tools to always provide one.
    if(!corr_id)
    {
        constexpr auto ref_count = 1;
        corr_id                  = context::correlation_tracing_service::construct(ref_count);
        _corr_id_pop             = corr_id;
    }

    // During finalization, correlation tracing service will not construct a correlation id so
    // just write packet through without tracing
    if(!corr_id)
    {
        writer(packets, pkt_count);
        return;
    }

    // if we constructed a correlation id, this decrements the reference count after the
    // underlying function returns
    auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop]() {
        if(_corr_id_pop)
        {
            context::pop_latest_correlation_id(_corr_id_pop);
            _corr_id_pop->sub_ref_count();
        }
    }};

    using packet_writer_fn_t = std::function<void(packet_vector_t &&)>;

    auto process_packet_batch = [&queue, &corr_id, tracing_data_v](
                                    const rocprofiler_packet* _packets,
                                    uint64_t                  _num_packets,
                                    const packet_writer_fn_t& _writer) {
        static constexpr auto null_signal = hsa_signal_t{.handle = 0};

        auto transformed_packets = packet_vector_t{};

        auto thr_id           = (corr_id) ? corr_id->thread_idx : common::get_tid();
        auto internal_corr_id = (corr_id) ? corr_id->internal : 0;
        auto ancestor_corr_id = (corr_id) ? corr_id->ancestor : 0;

        using packet_data_array_t = queue_info_session_t::packet_data_array_t;

        auto _info_session = queue_info_session_t{.queue          = *queue,
                                                  .tid            = thr_id,
                                                  .enqueue_ts     = common::timestamp_ns(),
                                                  .correlation_id = corr_id,
                                                  .packet_data    = packet_data_array_t{}};

        // Searching across all the packets given during this write
        for(size_t i = 0; i < _num_packets; ++i)
        {
            const auto& original_packet = _packets[i].kernel_dispatch;
            auto        packet_type =
                bit_extract(original_packet.header,
                            HSA_PACKET_HEADER_TYPE,
                            HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
            if(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            {
                transformed_packets.emplace_back(_packets[i]);
                continue;
            }

            // increase the reference count to denote that this correlation id is being used in a
            // kernel
            corr_id->add_ref_count();
            corr_id->add_kern_count();

            auto _packet_data = packet_data_t{};

            // make a copy of the tracing data
            _packet_data.tracing_data = tracing_data_v;

            tracing::populate_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
                ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                internal_corr_id);

            const uint64_t kernel_id = code_object::get_kernel_id(original_packet.kernel_object);
            const auto     original_completion_signal = original_packet.completion_signal;
            const auto     existing_completion_signal = (original_completion_signal != null_signal);

            // Copy kernel pkt, copy is to allow for signal to be modified
            _packet_data.kernel_packet = _packets[i];
            // create a reference for short hand access
            auto& kernel_packet     = _packet_data.kernel_packet;
            auto& completion_signal = _packet_data.kernel_packet.kernel_dispatch.completion_signal;

            auto create_signal = [](auto* signal) -> common::container::pool_object<signal_t>* {
                if(auto* pool = get_signal_pool(); pool && signal->handle == 0)
                {
                    auto& _signal = pool->acquire(construct_hsa_signal, 0, 0, nullptr, 0);
                    ROCP_FATAL_IF(!_signal.in_use())
                        << "Acquired signal from pool that is not in use";
                    ROCP_FATAL_IF(_signal.get().value == null_signal)
                        << "Acquired signal from pool that has invalid handle";
                    *CHECK_NOTNULL(signal) = _signal.get().value;
                    return &_signal;
                }
                return nullptr;
            };

            // No barrier packet: borrow a pooled signal if needed, then bump value by 1.
            if(!existing_completion_signal)
                _packet_data.pooled_signal = create_signal(&completion_signal);

            get_core_table()->hsa_signal_add_scacq_screl_fn(completion_signal, 1);

            // set the completion signal to the kernel packet
            _packet_data.completion_signal = completion_signal;

            // computes the "size" based on the offset of reserved_padding field
            constexpr auto kernel_dispatch_info_rt_size =
                common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

            static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                          "failed to compute size field based on offset of reserved_padding field");

            auto dispatch_id             = ++sequence_counter;
            _packet_data.callback_record = callback_record_t{
                sizeof(callback_record_t),
                rocprofiler_timestamp_t{0},
                rocprofiler_timestamp_t{0},
                rocprofiler_kernel_dispatch_info_t{
                    .size                 = kernel_dispatch_info_rt_size,
                    .agent_id             = queue->get_agent().get_rocp_agent()->id,
                    .queue_id             = queue->get_id(),
                    .kernel_id            = kernel_id,
                    .dispatch_id          = dispatch_id,
                    .private_segment_size = kernel_packet.kernel_dispatch.private_segment_size,
                    .group_segment_size   = kernel_packet.kernel_dispatch.group_segment_size,
                    .workgroup_size =
                        rocprofiler_dim3_t{kernel_packet.kernel_dispatch.workgroup_size_x,
                                           kernel_packet.kernel_dispatch.workgroup_size_y,
                                           kernel_packet.kernel_dispatch.workgroup_size_z},
                    .grid_size = rocprofiler_dim3_t{kernel_packet.kernel_dispatch.grid_size_x,
                                                    kernel_packet.kernel_dispatch.grid_size_y,
                                                    kernel_packet.kernel_dispatch.grid_size_z},
                    .reserved_padding = {0}}};

            {
                auto tracer_data = _packet_data.callback_record;
                tracing::execute_phase_enter_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    thr_id,
                    internal_corr_id,
                    _packet_data.tracing_data.external_correlation_ids,
                    ancestor_corr_id,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
            }

            // map all the external correlation ids (after enqueue enter phase) for all the contexts
            // captured by the info session
            tracing::update_external_correlation_ids(
                _packet_data.tracing_data.external_correlation_ids,
                thr_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

            // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
            // along with an ID of the client we got the packet from (this will be returned via
            // completed_cb_t)

            // emplace the kernel packet
            transformed_packets.emplace_back(kernel_packet);

            ROCP_FATAL_IF(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
                << "get_kernel_id below might need to be updated";

            {
                auto tracer_data = _packet_data.callback_record;
                tracing::execute_phase_exit_callbacks(
                    _packet_data.tracing_data.callback_contexts,
                    _packet_data.tracing_data.external_correlation_ids,
                    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                    ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                    tracer_data);
            }

            _info_session.packet_data.emplace_back(std::move(_packet_data));
        }

        if(!_info_session.packet_data.empty())
        {
            auto last_completion_signal = _info_session.packet_data.back().completion_signal;

            ROCP_FATAL_IF(last_completion_signal == null_signal)
                << "invalid completion signal in the last packet of the batch";

            auto current_signal_value =
                get_core_table()->hsa_signal_load_scacquire_fn(last_completion_signal);

            ROCP_INFO << fmt::format(
                "  Enqueued batch with completion signal {{.handle={}}} with value {}",
                last_completion_signal.handle,
                current_signal_value);

            auto _shared_info_session =
                std::make_shared<queue_info_session_t>(std::move(_info_session));
            get_async_signal_handler()->async(
                [_signal_v          = last_completion_signal,
                 _expected_signal_v = current_signal_value,
                 _session_v         = std::move(_shared_info_session)]() mutable {
                    async_signal_handler(_signal_v, _expected_signal_v, std::move(_session_v));
                });
        }

        _writer(std::move(transformed_packets));
    };

    ROCP_TRACE_IF(pkt_count > 1) << fmt::format(
        "[{}] Batching packets. Number of packets = {}", __FUNCTION__, pkt_count);

    process_packet_batch(packets_arr, pkt_count, [&writer](packet_vector_t&& _packets) {
        writer(_packets.data(), _packets.size());
    });
}
}  // namespace

void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell)
{
    if(!state) return;

    auto* state_ptr = state.get();

    // gate_lock serializes doorbell processing; producers never take it, so no deadlock.
    std::unique_lock<std::mutex> lock{state_ptr->gate_lock};

    const uint64_t scan_pos = state_ptr->next_scan_pos;

    const uint64_t wptr_end = state_ptr->virtual_wptr.load(std::memory_order_acquire);

    if(scan_pos >= wptr_end)
    {
        // Already scanned through virtual_wptr, so `value` is <= what we have submitted and
        // cannot advertise unpublished slots; forward it (and never drop the doorbell).
        ring_doorbell(state_ptr->doorbell_signal, value);
        return;
    }

    static thread_local auto snapshot_storage = std::vector<char>{};
    const uint64_t           max_bytes        = (wptr_end - scan_pos) * state_ptr->pkt_size;
    if(snapshot_storage.size() < max_bytes) snapshot_storage.resize(max_bytes);
    char* const source_snapshot = snapshot_storage.data();

    uint64_t drained = 0;
    for(uint64_t pos = scan_pos; pos < wptr_end; ++pos)
    {
        const auto  ring_slot = pos & state_ptr->ring_mask;
        char* const slot_base =
            static_cast<char*>(state_ptr->ring_buf) + (ring_slot * state_ptr->pkt_size);
        auto* const hdr_ptr = reinterpret_cast<volatile uint16_t*>(slot_base);

        if((__atomic_load_n(hdr_ptr, __ATOMIC_ACQUIRE) & 0xFFu) ==
           static_cast<unsigned>(HSA_PACKET_TYPE_INVALID))
            break;

        ::memcpy(source_snapshot + (drained * state_ptr->pkt_size), slot_base, state_ptr->pkt_size);
        __atomic_store_n(hdr_ptr, static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID), __ATOMIC_RELEASE);
        ++drained;
    }

    if(drained == 0)
    {
        // The next slot is claimed but not yet written by its producer, so there is
        // nothing to publish now; that producer's own later doorbell will drain it.
        // Re-ring only the last published index, not the virtual value.
        ring_published_doorbell(state_ptr, ring_doorbell);
        return;
    }

    const uint64_t pkt_count = drained;
    const uint64_t scan_end  = scan_pos + drained;

    ROCP_INFO << fmt::format("{} :: pkt_count={} (scan_pos={}, scan_end={})",
                             __FUNCTION__,
                             pkt_count,
                             scan_pos,
                             scan_end);

    auto& tls                     = get_doorbell_tls();
    tls.state                     = state_ptr;
    tls.submit_pos                = state_ptr->next_submit_pos;
    tls.pkt_size                  = state_ptr->pkt_size;
    tls.ring_doorbell             = &ring_doorbell;
    tls.last_published_submit_pos = state_ptr->next_submit_pos;
    uint64_t start_submit_pos     = tls.submit_pos;

    auto*        qc = get_queue_controller();
    const Queue* queue =
        (qc && state_ptr->hsa_queue) ? qc->get_queue(*state_ptr->hsa_queue) : nullptr;

    if(queue)
    {
        // call local write_interceptor directly instead of heavyweight
        // Queue::invoke_write_interceptor
        write_interceptor(
            const_cast<Queue*>(queue), source_snapshot, pkt_count, ring_buffer_writer);
    }
    else
    {
        ring_buffer_writer(source_snapshot, pkt_count);
    }

    uint64_t written = tls.submit_pos - start_submit_pos;
    if(written != pkt_count)
    {
        ROCP_WARNING << "Write-interceptor changed packet count. "
                     << "queue=" << state_ptr->hsa_queue << ", input_pkt_count=" << pkt_count
                     << ", written_pkt_count=" << written;
    }

    state_ptr->next_scan_pos   = scan_end;
    state_ptr->next_submit_pos = tls.submit_pos;

    auto real_rdid = __atomic_load_n(state_ptr->real_rdid, __ATOMIC_ACQUIRE);
    auto ring_used = (state_ptr->next_submit_pos - real_rdid);
    if(ring_used > state_ptr->ring_size)
    {
        ROCP_WARNING << "Queue-intercept observed ring usage beyond ring size. queue="
                     << state_ptr->hsa_queue << ", ring_used=" << ring_used
                     << ", ring_size=" << state_ptr->ring_size << ", scan_pos=" << scan_pos
                     << ", scan_end=" << scan_end
                     << ", next_submit_pos=" << state_ptr->next_submit_pos;
    }

    publish_submitted_packets(state_ptr, state_ptr->next_submit_pos);

    tls.ring_doorbell             = nullptr;
    tls.last_published_submit_pos = 0;
    tls.state                     = nullptr;
}

std::shared_ptr<QueueState>
create_queue_state(const hsa_queue_t* queue, bool overwrite)
{
    if(!queue) return nullptr;

    // this is needed for OpenMP target offload which, unlike HIP, does not automatically enable
    // profiler for queues it creates.
    if(get_amd_ext_table() && get_amd_ext_table()->hsa_amd_profiling_set_profiler_enabled_fn)
    {
        ROCP_HSA_TABLE_CALL(WARNING,
                            get_amd_ext_table()->hsa_amd_profiling_set_profiler_enabled_fn(
                                const_cast<hsa_queue_t*>(queue), true))
            << fmt::format("Could not enable profiler for hsa_queue_t{{.id={}}}", queue->id);
    }

    if(!overwrite)
    {
        if(auto existing = lookup_queue_state(queue, false)) return existing;
    }

    auto*              amd_queue = reinterpret_cast<amd_queue_t*>(const_cast<hsa_queue_t*>(queue));
    auto               state     = std::make_shared<QueueState>();
    volatile uint64_t* wdid_addr = &amd_queue->write_dispatch_id;
    volatile uint64_t* rdid_addr = &amd_queue->read_dispatch_id;
    uint64_t           current_wdid = __atomic_load_n(wdid_addr, __ATOMIC_ACQUIRE);
    state->ring_buf                 = queue->base_address;
    state->ring_size                = queue->size;
    state->ring_mask                = queue->size - 1;
    state->real_wdid                = wdid_addr;
    state->real_rdid                = rdid_addr;
    state->hsa_queue                = queue;
    state->doorbell_signal          = queue->doorbell_signal;
    state->virtual_wptr.store(current_wdid, std::memory_order_relaxed);
    state->next_scan_pos   = current_wdid;
    state->next_submit_pos = current_wdid;

    return get_queue_registry().wlock([&](auto& map) {
        map[queue] = state;
        return state;
    });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    get_queue_registry().wlock(
        [&](auto& map, const auto* _queue_v) {
            auto itr = map.find(_queue_v);
            if(itr != map.end()) map.erase(itr);
        },
        queue);
}

namespace
{
namespace impl
{
// The 16 wrappers differ only by HSA suffix + memory order; generated via macros below.

// add_write_index: uint64_t(const hsa_queue_t*, uint64_t)
#define ROCP_QUEUE_ADD_WRITE_INDEX(SUFFIX, ORDER)                                                  \
    uint64_t queue_add_write_index_##SUFFIX(const hsa_queue_t* q, uint64_t v)                      \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_add_write_index_##SUFFIX##_fn(q, v);                \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return add_write_index_impl(s.get(), v, ORDER);                                        \
        return get_next_table()->hsa_queue_add_write_index_##SUFFIX##_fn(q, v);                    \
    }

ROCP_QUEUE_ADD_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_ADD_WRITE_INDEX(scacq_screl, std::memory_order_acq_rel)
ROCP_QUEUE_ADD_WRITE_INDEX(scacquire, std::memory_order_acquire)
ROCP_QUEUE_ADD_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_ADD_WRITE_INDEX

// store_write_index: void(const hsa_queue_t*, uint64_t)
#define ROCP_QUEUE_STORE_WRITE_INDEX(SUFFIX, ORDER)                                                \
    void queue_store_write_index_##SUFFIX(const hsa_queue_t* q, uint64_t v)                        \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
        {                                                                                          \
            get_next_table()->hsa_queue_store_write_index_##SUFFIX##_fn(q, v);                     \
            return;                                                                                \
        }                                                                                          \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
        {                                                                                          \
            store_write_index_impl(s.get(), v, ORDER);                                             \
            return;                                                                                \
        }                                                                                          \
        get_next_table()->hsa_queue_store_write_index_##SUFFIX##_fn(q, v);                         \
    }

ROCP_QUEUE_STORE_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_STORE_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_STORE_WRITE_INDEX

// cas_write_index: uint64_t(const hsa_queue_t*, uint64_t expected, uint64_t value)
#define ROCP_QUEUE_CAS_WRITE_INDEX(SUFFIX, ORDER)                                                  \
    uint64_t queue_cas_write_index_##SUFFIX(                                                       \
        const hsa_queue_t* q, uint64_t expected, uint64_t value)                                   \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_cas_write_index_##SUFFIX##_fn(q, expected, value);  \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return cas_write_index_impl(s.get(), expected, value, ORDER);                          \
        return get_next_table()->hsa_queue_cas_write_index_##SUFFIX##_fn(q, expected, value);      \
    }

ROCP_QUEUE_CAS_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_CAS_WRITE_INDEX(scacq_screl, std::memory_order_acq_rel)
ROCP_QUEUE_CAS_WRITE_INDEX(scacquire, std::memory_order_acquire)
ROCP_QUEUE_CAS_WRITE_INDEX(screlease, std::memory_order_release)

#undef ROCP_QUEUE_CAS_WRITE_INDEX

// load_write_index: uint64_t(const hsa_queue_t*)
#define ROCP_QUEUE_LOAD_WRITE_INDEX(SUFFIX, ORDER)                                                 \
    uint64_t queue_load_write_index_##SUFFIX(const hsa_queue_t* q)                                 \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
            return get_next_table()->hsa_queue_load_write_index_##SUFFIX##_fn(q);                  \
        if(auto s = lookup_queue_state(q, s_intercept_dynamic.load(std::memory_order_acquire)); s) \
            return load_write_index_impl(s.get(), ORDER);                                          \
        return get_next_table()->hsa_queue_load_write_index_##SUFFIX##_fn(q);                      \
    }

ROCP_QUEUE_LOAD_WRITE_INDEX(relaxed, std::memory_order_relaxed)
ROCP_QUEUE_LOAD_WRITE_INDEX(scacquire, std::memory_order_acquire)

#undef ROCP_QUEUE_LOAD_WRITE_INDEX

// signal stores: void(hsa_signal_t, hsa_signal_value_t); NAME selects hsa_signal_<NAME>_fn.
#define ROCP_SIGNAL_STORE(NAME)                                                                    \
    void signal_##NAME(hsa_signal_t sig, hsa_signal_value_t val)                                   \
    {                                                                                              \
        if(should_bypass_inline_intercept())                                                       \
        {                                                                                          \
            get_next_table()->hsa_signal_##NAME##_fn(sig, val);                                    \
            return;                                                                                \
        }                                                                                          \
        /* it is too late to create queue state at this point so do not create if missing. */      \
        constexpr auto create_if_missing = false;                                                  \
        if(auto s = lookup_queue_state_by_doorbell(sig, create_if_missing); s)                     \
        {                                                                                          \
            process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {              \
                get_next_table()->hsa_signal_##NAME##_fn(db, v);                                   \
            });                                                                                    \
            return;                                                                                \
        }                                                                                          \
        get_next_table()->hsa_signal_##NAME##_fn(sig, val);                                        \
    }

ROCP_SIGNAL_STORE(store_relaxed)
ROCP_SIGNAL_STORE(store_screlease)
ROCP_SIGNAL_STORE(silent_store_relaxed)
ROCP_SIGNAL_STORE(silent_store_screlease)

#undef ROCP_SIGNAL_STORE
}  // namespace impl
}  // namespace

bool
supports_queue_interposition()
{
    return s_intercept_installed.load(std::memory_order_acquire);
}

void
interposition_sync()
{
    if(async_signal_handler_exists())  // query without constructing
    {
        constexpr auto async_only = true;
        if(auto* tg = get_async_signal_handler(); tg)
        {
            tg->join(async_only);
        }
    }
}

void
interposition_init(CoreApiTable* core_table, bool enabled)
{
    ROCP_INFO << "[queue-intercept] inline intercept path ENGAGED (tracing-only, no expansion)";

    // save a pointer to the original
    get_original_table() = core_table;

    // Save current table entries as our next-in-chain (tracing functors when called
    // after update_table, or raw HSA functions otherwise)
    *get_next_table() = *core_table;

    // Dynamic queue discovery: when enabled, the write-index wrappers create QueueState on
    // first encounter for queues we did not observe at hsa_queue_create. Enabled only when
    // attachment is not supported; in attachment mode this has been observed to deadlock.
    // TODO(rocprofiler-sdk): root-cause the attachment-mode deadlock so it can be enabled there.
    s_intercept_dynamic.store(!registration::supports_attachment(), std::memory_order_release);

    // mark that intercept has been installed
    s_intercept_installed.store(true, std::memory_order_release);

    core_table->hsa_queue_add_write_index_relaxed_fn     = impl::queue_add_write_index_relaxed;
    core_table->hsa_queue_add_write_index_scacq_screl_fn = impl::queue_add_write_index_scacq_screl;
    core_table->hsa_queue_add_write_index_scacquire_fn   = impl::queue_add_write_index_scacquire;
    core_table->hsa_queue_add_write_index_screlease_fn   = impl::queue_add_write_index_screlease;

    core_table->hsa_queue_store_write_index_relaxed_fn   = impl::queue_store_write_index_relaxed;
    core_table->hsa_queue_store_write_index_screlease_fn = impl::queue_store_write_index_screlease;

    core_table->hsa_queue_cas_write_index_relaxed_fn     = impl::queue_cas_write_index_relaxed;
    core_table->hsa_queue_cas_write_index_scacq_screl_fn = impl::queue_cas_write_index_scacq_screl;
    core_table->hsa_queue_cas_write_index_scacquire_fn   = impl::queue_cas_write_index_scacquire;
    core_table->hsa_queue_cas_write_index_screlease_fn   = impl::queue_cas_write_index_screlease;

    core_table->hsa_queue_load_write_index_relaxed_fn   = impl::queue_load_write_index_relaxed;
    core_table->hsa_queue_load_write_index_scacquire_fn = impl::queue_load_write_index_scacquire;

    core_table->hsa_signal_store_relaxed_fn          = impl::signal_store_relaxed;
    core_table->hsa_signal_store_screlease_fn        = impl::signal_store_screlease;
    core_table->hsa_signal_silent_store_relaxed_fn   = impl::signal_silent_store_relaxed;
    core_table->hsa_signal_silent_store_screlease_fn = impl::signal_silent_store_screlease;

    // mark that intercept has been activated
    s_intercept_active.store(enabled, std::memory_order_release);
}

void
interposition_fini()
{
    // disable dynamic discovery of queues
    s_intercept_dynamic.store(false, std::memory_order_release);

    // disable active interception
    s_intercept_active.store(false, std::memory_order_release);

    // wait for any in-flight signal handlers to complete and clean up the signal pool
    interposition_sync();

    // clean up signal pool
    signal_pool_fini();

    get_queue_registry().wlock([](auto& map) { map.clear(); });
}
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
