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
#include "lib/common/container/static_vector.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <pthread.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <mutex>
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
auto s_active_queue_interposition_consumers = std::atomic<uint32_t>{0};

// NOTE:
//  - "installed" is for checking whether HSA functions have been passed
//  - "active" is for controlling whether wrappers are intercepting or passing through
//  - "dynamic" is for whether to allow dynamic discovery of queues whose creation was not
//      observed/intercepted. E.g., during attachment, we want to toggle this on.
auto s_intercept_installed = std::atomic<bool>{false};  // installed (may not be active)
auto s_intercept_active    = std::atomic<bool>{false};  // actively intercepting
auto s_intercept_dynamic   = std::atomic<bool>{false};  // dynamically add queue states

bool
has_active_queue_interposition_consumers()
{
    return s_active_queue_interposition_consumers.load(std::memory_order_relaxed) > 0;
}

bool
should_bypass_inline_intercept()
{
    return (!s_intercept_installed.load(std::memory_order_acquire) ||
            !s_intercept_active.load(std::memory_order_acquire) ||
            registration::get_fini_status() != 0 ||
            // TODO: debug and enable queue interposition for attachment
            registration::supports_attachment() || !has_active_queue_interposition_consumers());
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
        auto _created = create_queue_state(queue, true);
        // F29: a queue discovered dynamically (never seen at hsa_queue_create) was
        // never windowed, so first_owner can no longer be trusted anywhere -- the
        // documented "owner we never windowed" invariant. Disable signal-less
        // process-wide. Called with no hub/registry lock held (create returned).
        if(_created && kfd::signal_less_feature_enabled() && !kfd::signal_less_child_stale())
            kfd::signal_less_disable_permanently();
        return _created;
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

// One in-flight completion wait: a batch's completion signal, the value it was
// enqueued at (the signal drops below this once the kernel completes), and the
// session whose completion bookkeeping runs when it does.
struct pending_completion
{
    hsa_signal_t                          completion_signal = {};
    hsa_signal_value_t                    starting_value    = 0;
    std::shared_ptr<queue_info_session_t> session           = {};
};

using pending_completion_vector_t = std::vector<pending_completion>;

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
    const auto doorbell_idx = static_cast<hsa_signal_value_t>(submit_pos - 1);
    (*tls.ring_doorbell)(state->doorbell_signal, doorbell_idx);
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

}  // namespace

namespace
{
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

// Shared state for the single completion-monitor thread. Enqueue threads push new
// waits onto `incoming` (the only cross-thread field, hence Synchronized) and bump
// `wake_signal`, which is appended to the monitor's wait array so
// hsa_amd_signal_wait_any returns and the monitor can pick them up. `active` is the
// monitor's live watch set, touched only by the monitor thread; `active_scratch` is a
// reused compaction buffer swapped with `active` each pass to avoid reallocation;
// wake_signal, thread, and running are set once at startup.
struct completion_monitor
{
    common::Synchronized<std::vector<pending_completion>> incoming       = {};
    std::vector<pending_completion>                       active         = {};
    std::vector<pending_completion>                       active_scratch = {};
    hsa_signal_t                                          wake_signal    = {};
    std::thread                                           thread         = {};
    std::atomic<bool>                                     running        = {false};
    // Registered-but-not-yet-completed waits (inbox + active). Lets other threads
    // observe quiescence without touching the monitor-private `active` vector.
    std::atomic<uint64_t> inflight = {0};
    // Count of producers currently in the doorbell path that may still call
    // register_completion. The monitor must not exit its loop (and abandon the
    // inbox) while this is nonzero, or a late register_completion could strand an
    // entry in `incoming` after the thread is gone -- hanging drain_completion_monitor.
    std::atomic<uint64_t> registering = {0};
};

completion_monitor&
get_completion_monitor()
{
    static auto*& _v = common::static_object<completion_monitor>::construct();
    return *_v;
}

// Hand a completed-batch wait to the monitor: enqueue it on the shared inbox and
// bump the wake signal so the monitor picks it up. Single point of registration;
// a multi-monitor variant would select a monitor here.
void
register_completion(pending_completion&& pending)
{
    auto& mon = get_completion_monitor();
    mon.inflight.fetch_add(1, std::memory_order_relaxed);
    mon.incoming.wlock([&pending](auto& inbox) { inbox.emplace_back(std::move(pending)); });
    get_core_table()->hsa_signal_store_screlease_fn(mon.wake_signal, 1);
}

// Completion bookkeeping for one batch, run after its completion signal drops.
// Emits dispatch-complete records, releases/decrements the completion signal, and
// drops the correlation-id references. Waiting is done elsewhere (the monitor);
// this only runs once the kernel is known complete (or during teardown cleanup).
void
run_completion(const std::shared_ptr<queue_info_session_t>& session)
{
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

// The no-signal finalizer. Runs on a task-group worker, or on the thread
// flushing the retry owner -- never on the reader thread or under a hub lock.
//
// There is deliberately no HSA fallback: a signal-less dispatch never had an SDK
// signal and the app may already have destroyed its own, so a convert/sanity
// failure emits no record but still retires the correlation id.
void
complete_signal_less_dispatch(kfd::signal_less_hub_t::proven&& proven)
{
    auto&       _payload    = proven.payload;
    const auto* _rocp_agent = agent::get_agent(_payload.agent_id);
    auto        _hsa_agent  = agent::get_hsa_agent(_rocp_agent);

    auto _convert = [&_hsa_agent](uint64_t ticks, uint64_t* out) {
        if(!_hsa_agent) return false;
        const auto* _ext = get_amd_ext_table();
        if(!_ext || !_ext->hsa_amd_profiling_convert_tick_to_system_domain_fn) return false;
        return _ext->hsa_amd_profiling_convert_tick_to_system_domain_fn(*_hsa_agent, ticks, out) ==
               HSA_STATUS_SUCCESS;
    };

    auto _emit = [&_payload](uint64_t start_ns, uint64_t end_ns) {
        kernel_dispatch::emit_kernel_dispatch_record(_payload.tracing_data,
                                                     _payload.callback_record,
                                                     _payload.correlation_id,
                                                     _payload.tid,
                                                     start_ns,
                                                     end_ns);
    };

    // Retires exactly once whatever the outcome, and even if a client callback
    // throws (run_complete_signal_less_dispatch arms this from a scope destructor).
    auto _retire = [&_payload]() {
        auto* _corr_id = _payload.correlation_id;
        if(!_corr_id) return;
        ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
            << "reference counter for correlation id " << _corr_id->internal
            << " has no reference count";
        _corr_id->sub_kern_count();
        _corr_id->sub_ref_count();
    };

    const uint64_t _now = common::timestamp_ns();

    auto       _detail  = kfd::finalize_detail{};
    const auto _outcome = kfd::run_complete_signal_less_dispatch(proven.start_ticks,
                                                                 proven.end_ticks,
                                                                 _payload.enqueue_ts,
                                                                 _now,
                                                                 _convert,
                                                                 _emit,
                                                                 _retire,
                                                                 &_detail);

    if(_outcome == kfd::finalize_outcome::result_ready)
    {
        kfd::note_signal_less(kfd::signal_less_counter::finalizer_emitted);
        return;
    }

    kfd::note_signal_less(kfd::signal_less_counter::finalizer_no_timing);

    // Rate-limited: the first few are the diagnostic, a steady stream must not
    // flood the log.
    static auto _warned = std::atomic<int>{0};
    if(_warned.fetch_add(1, std::memory_order_relaxed) < 10)
    {
        ROCP_INFO << fmt::format(
            "KFD dispatch-log: no timing for dispatch (reason={}, gpu={} slot={} idx={})",
            kfd::finalize_reason_name(_detail.reason),
            proven.key.gpu_id,
            proven.key.doorbell_slot,
            proven.key.dispatch_idx_low32);
    }
}

bool
submit_to_task_group(kfd::signal_less_hub_t::proven& proven)
{
    // task_group_t::async takes a copy-constructible std::function but the payload
    // is move-only, so it travels in a shared_ptr the wrapper keeps. On refusal
    // (fork-stale or fini) the primitive returns before consuming the task, so the
    // payload is restored intact and the caller defers it -- packaging it after a
    // refusable call would strand a moved-from payload and never retire its id.
    auto _held = std::make_shared<kfd::signal_less_hub_t::proven>(std::move(proven));
    if(!submit_inline_async([_held]() { complete_signal_less_dispatch(std::move(*_held)); },
                            /*refuse_during_fini=*/true))
    {
        proven = std::move(*_held);
        return false;
    }
    return true;
}

bool
is_dispatch_packet(const rocprofiler_packet& pkt)
{
    auto _type = bit_extract(pkt.kernel_dispatch.header,
                             HSA_PACKET_HEADER_TYPE,
                             HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
    if(_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) return true;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
    if(_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
        return pkt.ext_kernel_dispatch.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
#endif
    return false;
}

// Per-BATCH signal-less eligibility, decided ONCE before any packet is touched.
// It must be final up front: a batch that has skipped its completion signals
// cannot be moved back onto the signal path, so even "will the hub accept these
// keys" is answered here. If ANY packet fails, the WHOLE batch keeps the signal
// path.
//
// keys_out is indexed BY PACKET INDEX, so registration uses the exact keys the
// hub validated here rather than re-deriving them from a second doorbell lookup
// that could observe a different generation.
bool
signal_less_batch_eligible(Queue*                                            queue,
                           const rocprofiler_packet*                         packets,
                           uint64_t                                          num_packets,
                           uint64_t                                          base_pkt_index,
                           std::vector<std::optional<kfd::correlation_key>>* keys_out,
                           kfd::window_ptr*                                  window_out)
{
    keys_out->clear();
    if(window_out) *window_out = {};

    // Cheapest gate first: with the feature off this is one relaxed load.
    if(kfd::signal_less_child_stale()) return false;
    if(!kfd::signal_less_feature_enabled()) return false;

    const auto _gpu_id = queue->get_agent().get_rocp_agent()->gpu_id;
    if(!kfd::ensure_reader_session(static_cast<uint32_t>(_gpu_id))) return false;
    const auto _gpu = static_cast<uint32_t>(_gpu_id);

    // Resolve the window open for this queue: rlock only, no clock, no bind.
    // No window (SDMA, poisoned/overlapped slot, disabled) -> signal path.
    auto _w = kfd::doorbell_map().resolve(_gpu, queue->get_id());
    if(!_w) return false;
    const uint32_t _slot = (*_w)->slot;

    keys_out->assign(num_packets, std::nullopt);
    auto _flat = std::vector<kfd::correlation_key>{};
    for(uint64_t i = 0; i < num_packets; ++i)
    {
        if(!is_dispatch_packet(packets[i])) continue;
        auto _key = kfd::correlation_key{
            _slot, static_cast<uint32_t>((base_pkt_index + i) & 0xFFFFFFFFULL), _gpu};
        (*keys_out)[i] = _key;
        _flat.emplace_back(_key);
    }
    if(_flat.empty())
    {
        keys_out->clear();
        return false;
    }

    // The admission latch replaces the deleted is_closing() hub call: a
    // plain bool read while this thread already holds gate_lock, via the
    // doorbell_tls handoff process_doorbell_impl set before calling the interceptor.
    auto*      _st       = get_doorbell_tls().state;
    const bool _eligible = kfd::owner_registry().slot_uniquely_owned(_gpu, _slot) &&
                           kfd::signal_less_hub().can_register_batch(_flat) &&
                           !(_st != nullptr && _st->admission_closed);

    if(_eligible)
    {
        if(window_out) *window_out = *_w;
    }
    else
    {
        keys_out->clear();
    }
    return _eligible;
}

// Move any newly-registered waits from the shared inbox into the monitor's private
// active set. Called only by the monitor thread.
void
drain_incoming(completion_monitor& mon)
{
    mon.incoming.wlock([&mon](auto& inbox) {
        for(auto& pending : inbox)
            mon.active.emplace_back(std::move(pending));
        inbox.clear();
    });
}

// Body of the single completion-monitor thread. Waits on the whole set of in-flight
// completion signals at once (plus a wake signal appended at the end) via
// hsa_amd_signal_wait_any, so the number of signals watched is decoupled from the
// number of threads. When any completion signal drops below its starting value, its
// batch's completion bookkeeping runs and the entry is removed; the wake signal lets
// producers add new waits or request shutdown. Exits once running is cleared and no
// waits remain.
void
completion_monitor_loop(completion_monitor& mon)
{
    // hsa_amd_signal_wait_any takes parallel arrays; rebuilt from `active` (+ wake
    // signal) each iteration. The wake signal always occupies the final slot.
    auto signals = std::vector<hsa_signal_t>{};
    auto conds   = std::vector<hsa_signal_condition_t>{};
    auto values  = std::vector<hsa_signal_value_t>{};

    constexpr auto timeout_hint = std::chrono::nanoseconds{std::chrono::milliseconds{100}};

    while(true)
    {
        drain_incoming(mon);

        // Reset the wake signal before waiting: any producer bump that happened while
        // we were processing is folded into this pass by the drain above, and a bump
        // that arrives after this store re-satisfies the wait so we never miss it.
        get_core_table()->hsa_signal_store_screlease_fn(mon.wake_signal, 0);

        // One more drain closes the window between the pre-wait drain and the reset.
        drain_incoming(mon);

        // Break on shutdown regardless of whether `active` is empty: pending
        // completion signals may never transition during finalization, so this
        // check must not be gated on an empty active set or the loop could spin
        // without ever reaching the teardown path that releases them. Finalization
        // can begin before the monitor is explicitly stopped, so honor it too.
        if(!mon.running.load(std::memory_order_acquire) || registration::get_fini_status() != 0)
            break;

        if(mon.active.empty())
        {
            // Nothing to watch; block on the wake signal alone until a producer
            // registers work or requests shutdown.
            get_core_table()->hsa_signal_wait_relaxed_fn(mon.wake_signal,
                                                         HSA_SIGNAL_CONDITION_NE,
                                                         0,
                                                         timeout_hint.count(),
                                                         HSA_WAIT_STATE_BLOCKED);
            continue;
        }

        signals.clear();
        conds.clear();
        values.clear();
        for(const auto& pending : mon.active)
        {
            signals.emplace_back(pending.completion_signal);
            conds.emplace_back(HSA_SIGNAL_CONDITION_LT);
            values.emplace_back(pending.starting_value);
        }
        // Wake signal occupies the final slot, so the satisfying index is commonly
        // nonzero when a producer bumps it to interrupt the wait.
        signals.emplace_back(mon.wake_signal);
        conds.emplace_back(HSA_SIGNAL_CONDITION_NE);
        values.emplace_back(0);

        // The satisfying index/value are unused: rather than trust the single index
        // wait_any reports, the active set is rescanned below for all completed waits.
        [[maybe_unused]] auto satisfying_value = hsa_signal_value_t{0};
        get_amd_ext_table()->hsa_amd_signal_wait_any_fn(static_cast<uint32_t>(signals.size()),
                                                        signals.data(),
                                                        conds.data(),
                                                        values.data(),
                                                        timeout_hint.count(),
                                                        HSA_WAIT_STATE_BLOCKED,
                                                        &satisfying_value);

        // Regardless of which signal woke us, scan the active set for completed waits:
        // wait_any reports only one index, but several may have dropped at once.
        // Compact survivors into the reused scratch buffer, then swap so neither
        // buffer is reallocated across iterations.
        mon.active_scratch.clear();
        mon.active_scratch.reserve(mon.active.size());
        for(auto& pending : mon.active)
        {
            auto value = get_core_table()->hsa_signal_load_relaxed_fn(pending.completion_signal);
            if(value < pending.starting_value)
            {
                run_completion(pending.session);
                mon.inflight.fetch_sub(1, std::memory_order_relaxed);
            }
            else
                mon.active_scratch.emplace_back(std::move(pending));
        }
        mon.active.swap(mon.active_scratch);

        // Release the swapped-out entries (completed sessions moved-from above, plus
        // any still-live shared_ptrs from a prior iteration) rather than holding them
        // until the next swap.
        mon.active_scratch.clear();
    }

    // Teardown: a producer already inside process_doorbell_impl when the loop broke
    // can still call register_completion after this point. Wait for all such in-flight
    // producers to finish (registering == 0), draining the inbox as they land, so no
    // entry is stranded in `incoming` after the thread exits (which would leak its
    // signal/correlation-id refs and hang drain_completion_monitor on inflight > 0).
    // New producers cannot start once finalization has set fini_status, so the
    // registering count only falls and this loop always finishes.
    constexpr auto drain_interval = std::chrono::microseconds{100};
    auto           drain_iters    = uint64_t{0};
    do
    {
        drain_incoming(mon);
        for(auto& pending : mon.active)
        {
            run_completion(pending.session);
            mon.inflight.fetch_sub(1, std::memory_order_relaxed);
        }
        mon.active.clear();

        if(mon.registering.load(std::memory_order_acquire) == 0) break;

        // Producers between the entry gate and their registering-- are bounded and
        // never block on the monitor, so this must finish. Surface a diagnostic if
        // it does not, rather than hang silently.
        ++drain_iters;
        constexpr auto warn_interval = uint64_t{10000};  // ~1s at 100us
        if(drain_iters % warn_interval == 0)
            ROCP_WARNING << fmt::format(
                "Completion monitor still waiting for {} in-flight producer(s) to finish during "
                "teardown after {} ms",
                mon.registering.load(std::memory_order_acquire),
                (drain_iters * drain_interval.count()) / 1000);

        std::this_thread::sleep_for(drain_interval);
    } while(true);

    // Final drain: catch any entry registered between the last drain and registering
    // reaching zero.
    drain_incoming(mon);
    for(auto& pending : mon.active)
    {
        run_completion(pending.session);
        mon.inflight.fetch_sub(1, std::memory_order_relaxed);
    }
    mon.active.clear();
}

// Create the wake signal and launch the monitor thread. Idempotent: a second call
// while already running is a no-op.
void
start_completion_monitor()
{
    auto& mon = get_completion_monitor();
    if(mon.running.exchange(true)) return;

    auto status = get_amd_ext_table()->hsa_amd_signal_create_fn(0, 0, nullptr, 0, &mon.wake_signal);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS)
        << "failed to create completion-monitor wake signal";

    mon.thread = std::thread{[&mon]() { completion_monitor_loop(mon); }};
}

// Wait for all in-flight completions to drain without stopping the monitor. Safe to
// call while interception is still active (e.g. on context stop).
void
drain_completion_monitor()
{
    auto& mon = get_completion_monitor();
    if(!mon.running.load(std::memory_order_acquire)) return;

    // Poll the in-flight counter (registered minus completed) until it reaches zero.
    // The monitor makes progress independently; this only observes quiescence without
    // touching its private active set.
    constexpr auto poll_interval = std::chrono::milliseconds{1};
    while(mon.inflight.load(std::memory_order_acquire) > 0)
    {
        get_core_table()->hsa_signal_store_screlease_fn(mon.wake_signal, 1);
        std::this_thread::sleep_for(poll_interval);
    }
}

// Stop the monitor: clear running, wake it so it observes the flag, and join. After
// this returns the monitor thread is dead and its active set is safe to access.
void
stop_completion_monitor()
{
    auto& mon = get_completion_monitor();
    if(!mon.running.exchange(false)) return;

    get_core_table()->hsa_signal_store_screlease_fn(mon.wake_signal, 1);
    if(mon.thread.joinable()) mon.thread.join();

    get_core_table()->hsa_signal_destroy_fn(mon.wake_signal);
    mon.wake_signal = {};
}

// Local kernel-dispatch tracing path: swaps in pooled completion signals,
// runs KERNEL_DISPATCH_ENQUEUE tracer hooks, and prepares a completion-signal
// wait for the monitor thread. Strict 1:1 packet forwarding; does
// not insert PM4 packets. Distinct from Queue::WriteInterceptor (legacy path).
void
write_interceptor(Queue*                                queue,
                  const void*                           packets,
                  uint64_t                              pkt_count,
                  hsa_amd_queue_intercept_packet_writer writer,
                  pending_completion_vector_t*          deferred_completions,
                  uint64_t                              base_pkt_index)
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
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
        else if(packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
        {
            const auto& ext_packet = packets_arr[i].ext_kernel_dispatch;
            if(ext_packet.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
            {
                ++num_dispatch_packets;
            }
        }
#endif
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

    auto process_packet_batch = [&queue, &corr_id, tracing_data_v, deferred_completions](
                                    const rocprofiler_packet* _packets,
                                    uint64_t                  _num_packets,
                                    uint64_t                  _base_pkt_index,
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

        // Decided once for the whole batch, before any packet is modified. All
        // packets in a batch share one queue, hence one owner_window.
        auto            _signal_less_keys   = std::vector<std::optional<kfd::correlation_key>>{};
        kfd::window_ptr _signal_less_window = {};
        // Non-const: D7 clears it on the register_batch refusal path so the post-loop
        // signal-path block runs and builds the async waiter for the fallback.
        bool _signal_less_batch = signal_less_batch_eligible(queue,
                                                             _packets,
                                                             _num_packets,
                                                             _base_pkt_index,
                                                             &_signal_less_keys,
                                                             &_signal_less_window);
        auto _signal_less_regs  = std::vector<kfd::signal_less_hub_t::registration>{};

        // Parallel to _info_session.packet_data: for each dispatch packet, the index
        // in transformed_packets of its SUBMITTED packet plus whether it is the ext
        // form. transformed_packets also holds pass-through and barrier/interrupt
        // packets, so packet_data[k] != transformed_packets[k]; the D7 fallback needs
        // both to write the replayed signal into the right submitted packet field.
        struct dispatch_pkt_ref
        {
            size_t transformed_index = 0;
            bool   is_ext            = false;
        };
        auto _dispatch_pkt_index = std::vector<dispatch_pkt_ref>{};

        auto create_signal = [](auto* signal) -> common::container::pool_object<signal_t>* {
            if(auto* pool = get_signal_pool(); pool && signal->handle == 0)
            {
                auto& _signal = pool->acquire(construct_hsa_signal, 0, 0, nullptr, 0);
                ROCP_FATAL_IF(!_signal.in_use()) << "Acquired signal from pool that is not in use";
                ROCP_FATAL_IF(_signal.get().value == null_signal)
                    << "Acquired signal from pool that has invalid handle";
                *CHECK_NOTNULL(signal) = _signal.get().value;
                return &_signal;
            }
            return nullptr;
        };

        // The three signal-path steps, factored so the normal !_signal_less_batch
        // branch and the D7 refusal fallback cannot drift: borrow a pooled signal if
        // the app supplied none, bump its value by 1, and record it on the packet.
        // Returns the completion signal written into pd.kernel_packet.
        auto apply_signal_path = [&create_signal](packet_data_t& pd, bool is_ext) -> hsa_signal_t {
            (void) is_ext;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
            auto& _cs = is_ext ? pd.kernel_packet.ext_kernel_dispatch.completion_signal
                               : pd.kernel_packet.kernel_dispatch.completion_signal;
#else
            auto& _cs = pd.kernel_packet.kernel_dispatch.completion_signal;
#endif
            if(_cs == null_signal) pd.pooled_signal = create_signal(&_cs);
            get_core_table()->hsa_signal_add_scacq_screl_fn(_cs, 1);
            pd.completion_signal = _cs;
            return _cs;
        };

        // Searching across all the packets given during this write
        for(size_t i = 0; i < _num_packets; ++i)
        {
            const auto& original_packet = _packets[i].kernel_dispatch;
            auto        packet_type =
                bit_extract(original_packet.header,
                            HSA_PACKET_HEADER_TYPE,
                            HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
            bool is_kernel_dispatch     = (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH);
            bool is_ext_kernel_dispatch = false;
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
            if(packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
            {
                const auto& ext_packet = _packets[i].ext_kernel_dispatch;
                if(ext_packet.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
                    is_ext_kernel_dispatch = true;
            }
#endif
            if(!is_kernel_dispatch && !is_ext_kernel_dispatch)
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

            // Lambda to extract packet info regardless of packet type
            auto extract_packet_info = [](const rocprofiler_packet& pkt, bool is_ext) {
                struct packet_info
                {
                    hsa_signal_t       completion_signal;
                    uint64_t           kernel_object;
                    uint32_t           private_segment_size;
                    uint32_t           group_segment_size;
                    rocprofiler_dim3_t workgroup_size;
                    rocprofiler_dim3_t grid_size;
                };
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
                if(is_ext)
                {
                    const auto& e = pkt.ext_kernel_dispatch;
                    return packet_info{e.completion_signal,
                                       e.kernel_object,
                                       e.private_segment_size,
                                       e.group_segment_size,
                                       {e.workgroup_size_x, e.workgroup_size_y, e.workgroup_size_z},
                                       {static_cast<uint32_t>(e.cluster_count_x) *
                                            static_cast<uint32_t>(e.cluster_size_x) *
                                            static_cast<uint32_t>(e.workgroup_size_x),
                                        static_cast<uint32_t>(e.cluster_count_y) *
                                            static_cast<uint32_t>(e.cluster_size_y) *
                                            static_cast<uint32_t>(e.workgroup_size_y),
                                        static_cast<uint32_t>(e.cluster_count_z) *
                                            static_cast<uint32_t>(e.cluster_size_z) *
                                            static_cast<uint32_t>(e.workgroup_size_z)}};
                }
#else
                (void) is_ext;
#endif
                {
                    const auto& s = pkt.kernel_dispatch;
                    return packet_info{s.completion_signal,
                                       s.kernel_object,
                                       s.private_segment_size,
                                       s.group_segment_size,
                                       {s.workgroup_size_x, s.workgroup_size_y, s.workgroup_size_z},
                                       {s.grid_size_x, s.grid_size_y, s.grid_size_z}};
                }
            };

            const auto     pkt_info  = extract_packet_info(_packets[i], is_ext_kernel_dispatch);
            const uint64_t kernel_id = code_object::get_kernel_id(pkt_info.kernel_object);

            // Copy kernel pkt, copy is to allow for signal to be modified
            _packet_data.kernel_packet = _packets[i];
            // create a reference for short hand access
            auto& kernel_packet = _packet_data.kernel_packet;

            if(!_signal_less_batch)
            {
                // No barrier packet: borrow a pooled signal if needed, bump by 1, record it.
                apply_signal_path(_packet_data, is_ext_kernel_dispatch);
            }

            // computes the "size" based on the offset of reserved_padding field
            constexpr auto kernel_dispatch_info_rt_size =
                common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

            static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                          "failed to compute size field based on offset of reserved_padding field");

            auto dispatch_id = ++sequence_counter;
            _packet_data.callback_record =
                callback_record_t{sizeof(callback_record_t),
                                  rocprofiler_timestamp_t{0},
                                  rocprofiler_timestamp_t{0},
                                  rocprofiler_kernel_dispatch_info_t{
                                      .size        = kernel_dispatch_info_rt_size,
                                      .agent_id    = queue->get_agent().get_rocp_agent()->id,
                                      .queue_id    = queue->get_id(),
                                      .kernel_id   = kernel_id,
                                      .dispatch_id = dispatch_id,
                                      .private_segment_size = pkt_info.private_segment_size,
                                      .group_segment_size   = pkt_info.group_segment_size,
                                      .workgroup_size       = pkt_info.workgroup_size,
                                      .grid_size            = pkt_info.grid_size,
                                      .reserved_padding     = {0}}};

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

            if(_signal_less_batch && _signal_less_keys[i].has_value())
            {
                auto _reg           = kfd::signal_less_hub_t::registration{};
                _reg.key            = *_signal_less_keys[i];
                _reg.correlation_id = internal_corr_id;
                _reg.window         = _signal_less_window;

                auto& _pl           = _reg.payload;
                _pl.callback_record = _packet_data.callback_record;
                _pl.tracing_data    = _packet_data.tracing_data;
                // The reference this payload inherits was taken by the
                // add_ref_count()/add_kern_count() above; the finalizer releases it.
                _pl.correlation_id = corr_id;
                _pl.tid            = thr_id;
                _pl.agent_id       = queue->get_agent().get_rocp_agent()->id;
                _pl.enqueue_ts     = _info_session.enqueue_ts;

                _signal_less_regs.emplace_back(std::move(_reg));
            }

            // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
            // along with an ID of the client we got the packet from (this will be returned via
            // completed_cb_t)

            // emplace the kernel packet; record its submitted index (and ext-ness) so
            // the D7 fallback can write a replayed signal into the right submitted slot.
            _dispatch_pkt_index.push_back(
                dispatch_pkt_ref{transformed_packets.size(), is_ext_kernel_dispatch});
            transformed_packets.emplace_back(kernel_packet);

            ROCP_FATAL_IF(!is_kernel_dispatch && !is_ext_kernel_dispatch)
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

        auto last_completion_signal = null_signal;
        auto current_signal_value   = hsa_signal_value_t{0};
        auto _shared_info_session   = std::shared_ptr<queue_info_session_t>{};

        // Register the whole batch BEFORE the writer publishes any packet, so a
        // firmware record can never arrive for a dispatch the hub has not seen.
        // Cap-evicted closed-window entries leave via _evicted, released here AFTER
        // m_mu is dropped (the hub's no-destroy-under-lock contract).
        const auto _signal_less_count = _signal_less_regs.size();
        auto       _evicted           = std::vector<kfd::signal_less_hub_t::leaked>{};
        if(_signal_less_batch &&
           !kfd::signal_less_hub().register_batch(std::move(_signal_less_regs), _evicted))
        {
            // D7: reachable on a slot quarantined between eligibility and here, or the
            // per-GPU cap exceeded with no eligible victim (D9). The batch skipped its
            // signal instrumentation, so replay exactly the !_signal_less_batch signal
            // path per dispatch and fall through to the normal signal-path completion.
            // No id retirement here: register_batch did not consume _signal_less_regs on
            // refusal, its payload correlation_id* is non-owning, and the signal path's
            // completion handler performs the matching releases -- retiring here would
            // double-release. Telemetry only.
            kfd::note_signal_less(kfd::signal_less_counter::register_refused, _signal_less_count);
            for(size_t k = 0; k < _info_session.packet_data.size(); ++k)
            {
                auto&      _pd  = _info_session.packet_data[k];
                const auto _sig = apply_signal_path(_pd, _dispatch_pkt_index[k].is_ext);
                // Mirror the replayed signal into the SUBMITTED packet, whose index in
                // transformed_packets differs from k (pass-through/barrier packets).
                auto& _tp = transformed_packets[_dispatch_pkt_index[k].transformed_index];
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
                if(_dispatch_pkt_index[k].is_ext)
                    _tp.ext_kernel_dispatch.completion_signal = _sig;
                else
                    _tp.kernel_dispatch.completion_signal = _sig;
#else
                _tp.kernel_dispatch.completion_signal = _sig;
#endif
            }
            // Clear BEFORE the post-loop block so it builds the async waiter unchanged.
            _signal_less_batch = false;
            ROCP_WARNING << "KFD dispatch-log: signal-less batch registration refused; falling "
                            "back to the signal path for these dispatches";
        }
        else if(_signal_less_batch)
        {
            kfd::note_signal_less(kfd::signal_less_counter::entry_registered, _signal_less_count);
        }

        // Cap eviction is loud accepted coverage loss: count it, warn (rate-limited by
        // the census cadence), and latch losses (INV-L1) before any finalize can see
        // the ledgered ids. Off-lock; _evicted's payloads release at scope end.
        if(!_evicted.empty())
        {
            kfd::note_signal_less(kfd::signal_less_counter::cap_evicted, _evicted.size());
            kfd::note_signal_less_losses();
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log: per-GPU hub cap evicted {} closed-window entry(ies); those "
                "dispatches emit no record",
                _evicted.size());
        }

        // A signal-less batch has no completion signal to wait on.
        if(!_info_session.packet_data.empty() && !_signal_less_batch)
        {
            last_completion_signal = _info_session.packet_data.back().completion_signal;

            ROCP_FATAL_IF(last_completion_signal == null_signal)
                << "invalid completion signal in the last packet of the batch";

            current_signal_value =
                get_core_table()->hsa_signal_load_scacquire_fn(last_completion_signal);

            ROCP_INFO << fmt::format(
                "  Enqueued batch with completion signal {{.handle={}}} with value {}",
                last_completion_signal.handle,
                current_signal_value);

            _shared_info_session = std::make_shared<queue_info_session_t>(std::move(_info_session));
        }

        // Copy packets into the real queue before creating the completion wait. The caller
        // defers registration until after it publishes the final doorbell.
        _writer(std::move(transformed_packets));

        if(_shared_info_session)
        {
            auto pending = pending_completion{
                last_completion_signal, current_signal_value, std::move(_shared_info_session)};

            if(deferred_completions)
                deferred_completions->emplace_back(std::move(pending));
            else
                register_completion(std::move(pending));
        }
    };

    ROCP_TRACE_IF(pkt_count > 1) << fmt::format(
        "[{}] Batching packets. Number of packets = {}", __FUNCTION__, pkt_count);

    process_packet_batch(
        packets_arr, pkt_count, base_pkt_index, [&writer](packet_vector_t&& _packets) {
            writer(_packets.data(), _packets.size());
        });
}
}  // namespace

// Precondition (F1): caller holds state.drain_mu. gate_lock still orders the
// admission_closed store against the publishing critical sections.
uint64_t
close_admission_and_snapshot_locked(QueueState& state)
{
    auto lk                = std::lock_guard<std::mutex>{state.gate_lock};
    state.admission_closed = true;  // SW-2: no later batch can register
    return state.next_submit_pos;   // snapshot, ordered by this same lock
}

// Precondition (F1): caller holds state.drain_mu, so real_rdid (which points into
// the runtime's amd_queue_t) cannot be freed by a concurrent destroy under the load.
bool
wait_queue_hw_drained_locked(QueueState& state, uint64_t submit_pos, uint64_t deadline_ns)
{
    if(!state.real_rdid) return true;

    while(!hw_queue_drained(__atomic_load_n(state.real_rdid, __ATOMIC_ACQUIRE), submit_pos))
    {
        if(kfd::steady_now_ns() >= deadline_ns) return false;
        std::this_thread::sleep_for(std::chrono::microseconds{200});
    }
    return true;
}

void
fence_all_queue_gates()
{
    // Copy the states out from under the registry lock FIRST: taking a queue's
    // gate_lock while holding it would invert the established order.
    auto _states = std::vector<queue_state_ptr_t>{};
    get_queue_registry().rlock([&_states](const auto& map) {
        _states.reserve(map.size());
        for(const auto& itr : map)
            if(itr.second) _states.emplace_back(itr.second);
    });

    for(const auto& _state : _states)
    {
        auto lk = std::lock_guard<std::mutex>{_state->gate_lock};
    }
}

void
drain_all_queues_hw(uint64_t deadline_ns)
{
    // F1 teardown drain, race-free against concurrent hsa_queue_destroy. Snapshot
    // the states under the registry lock, release it, then per state take drain_mu
    // and skip any queue destroy already invalidated (rdid_valid==false) -- that
    // queue ran its own drain. Lock order: registry lock -> (released) -> drain_mu.
    auto _states = std::vector<queue_state_ptr_t>{};
    get_queue_registry().rlock([&_states](const auto& map) {
        _states.reserve(map.size());
        for(const auto& itr : map)
            if(itr.second) _states.emplace_back(itr.second);
    });

    for(const auto& _state : _states)
    {
        auto _lk = std::unique_lock<std::mutex>{_state->drain_mu};
        if(!_state->rdid_valid) continue;  // destroyed under the same lock; skip
        const uint64_t _P = close_admission_and_snapshot_locked(*_state);
        wait_queue_hw_drained_locked(*_state, _P, deadline_ns);
    }
}

void
process_doorbell_impl(const queue_state_ptr_t& state,
                      hsa_signal_value_t       value,
                      const doorbell_fn_t&     ring_doorbell)
{
    if(!state) return;

    // Mark this producer as potentially about to register a completion. The monitor
    // will not exit (abandoning the inbox) while any producer is in this window, so a
    // register_completion at the tail of this function can never land after the monitor
    // thread has gone. Decremented on every exit path.
    auto& mon = get_completion_monitor();
    mon.registering.fetch_add(1, std::memory_order_acq_rel);
    auto _registering_dtor = common::scope_destructor{
        [&mon]() { mon.registering.fetch_sub(1, std::memory_order_acq_rel); }};

    auto* state_ptr            = state.get();
    auto  deferred_completions = pending_completion_vector_t{};

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

    constexpr size_t kSnapshotMaxPkts = 16;
    const uint64_t   max_pkts         = wptr_end - scan_pos;
    const auto       pkt_size         = state_ptr->pkt_size;

    using snapshot_pkt_t = std::array<char, 64>;
    common::container::static_vector<snapshot_pkt_t, kSnapshotMaxPkts> snapshot;
    std::vector<char>                                                  overflow_snapshot;
    char*                                                              source_snapshot = nullptr;

    if(max_pkts > kSnapshotMaxPkts)
    {
        overflow_snapshot.resize(max_pkts * pkt_size);
        source_snapshot = overflow_snapshot.data();
    }

    uint64_t drained = 0;
    for(uint64_t pos = scan_pos; pos < wptr_end; ++pos)
    {
        const auto  ring_slot = pos & state_ptr->ring_mask;
        char* const slot_base = static_cast<char*>(state_ptr->ring_buf) + (ring_slot * pkt_size);
        auto* const hdr_ptr   = reinterpret_cast<volatile uint16_t*>(slot_base);

        if((__atomic_load_n(hdr_ptr, __ATOMIC_ACQUIRE) & 0xFFu) ==
           static_cast<unsigned>(HSA_PACKET_TYPE_INVALID))
            break;

        char* dst = nullptr;
        if(source_snapshot)
        {
            dst = source_snapshot + (drained * pkt_size);
        }
        else
        {
            dst = snapshot.emplace_back().data();
        }
        ::memcpy(dst, slot_base, pkt_size);
        __atomic_store_n(hdr_ptr, static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID), __ATOMIC_RELEASE);
        ++drained;
    }

    if(!source_snapshot) source_snapshot = reinterpret_cast<char*>(snapshot.data());

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
        write_interceptor(const_cast<Queue*>(queue),
                          source_snapshot,
                          pkt_count,
                          ring_buffer_writer,
                          &deferred_completions,
                          start_submit_pos);
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

    // Register completion waits only after the final doorbell is visible
    // so the monitor can never wait on unpublished packets.
    lock.unlock();

    for(auto& pending : deferred_completions)
        register_completion(std::move(pending));
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
    // Close the interlock's init end: set AFTER real_rdid, BEFORE publication, so no
    // observer reaches a state with rdid_valid true but real_rdid null. The wlock
    // release below orders this plain-bool write for every reader.
    state->rdid_valid = true;

    // Get-or-create UNDER the final wlock: the pre-check above is a separate rlock, so
    // two concurrent dynamic-discovery lookups of the same queue could both miss and
    // both publish, splitting D8's drain_mu across two live states (real_rdid UAF).
    // Re-checking here keeps exactly one live QueueState per queue; a loser discards
    // its just-built state harmlessly (refcount drops).
    return get_queue_registry().wlock([&](auto& map) {
        auto it = map.find(queue);
        if(it != map.end() && it->second) return it->second;
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

namespace
{
void
resync_queue_shadow_state(QueueState* state)
{
    if(!state || !state->real_wdid) return;

    const uint64_t wdid = __atomic_load_n(state->real_wdid, __ATOMIC_ACQUIRE);
    state->virtual_wptr.store(wdid, std::memory_order_release);
    state->next_scan_pos   = wdid;
    state->next_submit_pos = wdid;
}

void
resync_all_queue_shadow_states()
{
    get_queue_registry().rlock([](const auto& registry) {
        for(const auto& entry : registry)
            resync_queue_shadow_state(entry.second.get());
    });
}
}  // namespace

void
notify_queue_interposition_consumer_context_started(const context::context* ctx)
{
    if(!context_needs_queue_interposition_tracing(ctx)) return;

    const auto prev = s_active_queue_interposition_consumers.load(std::memory_order_acquire);
    if(prev == 0 && s_intercept_installed.load(std::memory_order_acquire))
        resync_all_queue_shadow_states();

    s_active_queue_interposition_consumers.fetch_add(1, std::memory_order_release);
}

void
notify_queue_interposition_consumer_context_stopped(const context::context* ctx)
{
    if(!context_needs_queue_interposition_tracing(ctx)) return;
    auto cur = s_active_queue_interposition_consumers.load(std::memory_order_relaxed);
    while(cur > 0)
    {
        if(s_active_queue_interposition_consumers.compare_exchange_weak(
               cur, cur - 1, std::memory_order_release, std::memory_order_relaxed))
        {
            return;
        }
    }
}

void
interposition_sync()
{
    drain_completion_monitor();
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

    // launch the completion-monitor thread that waits on in-flight completion signals
    start_completion_monitor();

    // mark that intercept has been activated
    s_intercept_active.store(enabled, std::memory_order_release);

    // Inline intercept is the only path that produces a KFD correlation key, so
    // probe dispatch-log here (not in generic queue_init()). Master opt-in gate:
    // the KFD dispatch-log feature does nothing unless signal-less is enabled.
    // Best-effort.
    if(kfd::signal_less_feature_enabled()) kfd::init_kfd_profiler();
}

void
interposition_fini()
{
    // disable dynamic discovery of queues
    s_intercept_dynamic.store(false, std::memory_order_release);

    // disable active interception
    s_intercept_active.store(false, std::memory_order_release);

    // A fork child that never exec'd still runs this at exit, and everything below
    // it is unsafe there: the registry wlock and the signal pool's lock may have
    // been held by a thread that did not survive the fork, so acquiring them hangs
    // the child, and destroying inherited HSA signals reaches into a runtime the
    // child does not own. The child abandoned all of this state (D6) and is on its
    // way out, so skip it. The atomic stores above are kept: they are safe and make
    // the child's interception inert.
    if(internal_threading::fork_stale()) return;

    // drain in-flight completions, then stop and join the monitor thread. Stopping
    // before signal_pool_fini ensures the monitor has released all pooled signals and
    // is no longer touching the pool.
    interposition_sync();
    stop_completion_monitor();

    // clean up signal pool
    signal_pool_fini();

    get_queue_registry().wlock([](auto& map) { map.clear(); });
}
}  // namespace queue_interposition
}  // namespace hsa

// Bridge for the KFD layer (declared in kfd/signal_less.hpp). Defined here so
// kfd never needs the HSA interposition headers, and called directly -- both
// sides are in the same object library.
namespace kfd
{
bool
submit_complete_signal_less_dispatch(signal_less_hub_t::proven& p)
{
    return hsa::queue_interposition::submit_to_task_group(p);
}

void
finalize_complete_signal_less_dispatch(signal_less_hub_t::proven&& p)
{
    hsa::queue_interposition::complete_signal_less_dispatch(std::move(p));
}

void
drain_signal_less_interceptor()
{
    hsa::queue_interposition::fence_all_queue_gates();
}

void
drain_signal_less_queues_hw(uint64_t deadline_ns)
{
    hsa::queue_interposition::drain_all_queues_hw(deadline_ns);
}

void
join_signal_less_tasks()
{
    hsa::queue_interposition::interposition_sync();
}
}  // namespace kfd
}  // namespace rocprofiler
