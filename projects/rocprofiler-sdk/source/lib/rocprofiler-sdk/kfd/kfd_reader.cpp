// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"
#include "lib/rocprofiler-sdk/kfd/env_parse.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/poll_reader.hpp"
#include "lib/rocprofiler-sdk/kfd/record_pipe.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/kfd/stream_geometry.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

// Active (profiler ABI v6, stream ABI v3) dispatch-log UAPI. Must be the ONLY kfd
// ioctl header in this TU (it conflicts with lib/rocprofiler-sdk/details/kfd_ioctl.h).
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace rocprofiler
{
namespace kfd
{
namespace
{
constexpr int kEventfdFlags = EFD_CLOEXEC | EFD_NONBLOCK;

// Deep enough that a brief processor stall does not force the copier to drop.
constexpr size_t kBatchPipeDepth = 16;

// Upper bound on armed GPUs. One session per stream-capable KFD node -- and a
// partitioned GPU exposes one node PER compute partition, so the bound must cover
// the densest current configuration: 8 physical GPUs in CPX mode (8 partitions
// each) = 64 nodes. Nodes are armed in ascending gpu_id order, so if a future
// system still exceeds this, the survivors are the lowest gpu_ids deterministically
// rather than hash-order. Sizing this exactly to the discovered node count would
// remove the cap entirely but requires reader_state::sessions to become a
// once-sized heap buffer (dlog_session holds a std::atomic, so it must be
// constructed in place, never moved) reserved before the reader thread starts.
constexpr size_t kMaxSessions = 64;

// Aging cadences. A deposited result is normally taken within milliseconds; an
// unmatched start belongs to a dispatch whose eop never arrived.
constexpr uint64_t kLoggingIntervalNs        = 1'000'000'000ull;  // 1 s
constexpr uint64_t kProcessorEvictIntervalNs = 1'000'000'000ull;  // 1 s

// F3: a retained START may belong to a still-running long kernel, so the eviction
// age MUST exceed the longest real dispatch or a live kernel's START is aged out
// before its EOP arrives (emitting it start-unknown). The old 5 s bound aged out
// ordinary multi-second kernels; default to 30 min and let it be tuned in seconds
// via ROCPROFILER_KFD_DISPATCH_LOG_START_MAX_AGE_SEC. Eviction still bounds
// pending_starts growth for genuinely orphaned STARTs (queue died mid-dispatch),
// now paired with the F7 per-GPU hub cap for the fast-churn case. Coordinated with
// F7: the age is the slow backstop, the hub cap the hard bound.
constexpr uint64_t kStartMaxAgeSecDefault = 1800;  // 30 min

uint64_t
start_max_age_ns()
{
    // Read the env once: this runs every processor batch and common::get_env
    // scans environ, which is not thread-safe against concurrent env mutation.
    static const uint64_t _cached = []() {
        const uint64_t _sec = common::get_env<uint64_t>(
            "ROCPROFILER_KFD_DISPATCH_LOG_START_MAX_AGE_SEC", uint64_t{kStartMaxAgeSecDefault});
        // Clamp: 0 (or an unparsed 0) would evict everything immediately; an absurd
        // value could overflow the ns multiply. Bound to [1 s, 7 days].
        constexpr uint64_t _min_sec = 1;
        constexpr uint64_t _max_sec = 7ull * 24 * 3600;
        uint64_t           _use     = _sec;
        if(_sec < _min_sec)
            _use = kStartMaxAgeSecDefault;
        else if(_sec > _max_sec)
            _use = _max_sec;
        return _use * 1'000'000'000ull;
    }();
    return _cached;
}

// Hard size cap on a GPU's retained-START map. start_max_age_ns() above is only
// the slow age backstop, so this is the bound that actually holds under a
// sustained EOP loss -- exactly the role D9's cap plays for the hub, sized the
// same way and for the same reason (the map is per-GPU, and a workload's genuine
// in-flight peak is hundreds of thousands of dispatches). The [1, 4194304] range
// and the reject-and-default parse are D8/D9's shared contract; read once, forced
// at reader start so no worker thread races a concurrent setenv.
size_t
pending_starts_cap()
{
    static const size_t _cached = []() -> size_t {
        constexpr long _dflt = 2'000'000;
        auto           _v =
            env_long_in_range("ROCPROFILER_KFD_DISPATCH_LOG_MAX_PENDING_STARTS", 1, 4'194'304);
        return static_cast<size_t>(_v.value_or(_dflt));
    }();
    return _cached;
}

// Overflow depth at which the processor is clearly not keeping up. Not a cap:
// dropping here would lose records the ring already gave us.
constexpr size_t kOverflowWarnDepth = 256;

// poll() timeout from the environment, validated by poll_reader.hpp: a
// non-integer, empty, zero, or out-of-range value falls back to the default
// rather than becoming a busy-poll or an infinite block.
int
poll_timeout_ms()
{
    auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_POLL_TIMEOUT_MS");
    if(!_v) return kPollTimeoutMsDefault;

    const int _ms = poll_timeout_ms_from_str(*_v);
    if(_ms <= 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: ignoring invalid ROCPROFILER_KFD_DISPATCH_LOG_POLL_TIMEOUT_MS='{}' "
            "(expected an integer 1-{}); using {} ms",
            *_v,
            static_cast<uint64_t>(INT_MAX),
            kPollTimeoutMsDefault);
        return kPollTimeoutMsDefault;
    }
    return _ms;
}

// Owned exclusively by the processor thread, so none of it needs a lock.
struct processor_state
{
    std::unordered_map<uint32_t, pair_state> by_gpu = {};

    pair_state& for_gpu(uint32_t gpu_id)
    {
        auto [it, ins] = by_gpu.try_emplace(gpu_id);
        if(ins) it->second.max_pending_starts = pending_starts_cap();
        return it->second;
    }
};

// Validated before any sizing math uses it.
uint64_t
ring_bytes()
{
    auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB");
    if(!_v) return kDlogDefaultRingBytes;

    uint64_t _want = dlog_ring_bytes_from_kb_str(*_v);
    if(_want == 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: ignoring invalid ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB='{}' "
            "(expected an integer 1-{}); using {} KB",
            *_v,
            kDlogMaxRingKb,
            kDlogDefaultRingBytes / 1024);
        return kDlogDefaultRingBytes;
    }

    uint64_t _bytes = dlog_snap_ring_bytes(_want);
    ROCP_WARNING_IF(_bytes != _want) << fmt::format(
        "KFD dispatch-log: the driver only accepts ring sizes of 80*2^k bytes up to {} KB; "
        "using {} KB instead of the requested {} KB",
        kDlogMaxRingBytes / 1024,
        _bytes / 1024,
        _want / 1024);
    return _bytes;
}

size_t
page_size()
{
    // Constant for the process lifetime; resolve the syscall once.
    static const size_t _cached = [] {
        long p = sysconf(_SC_PAGESIZE);
        return p > 0 ? static_cast<size_t>(p) : size_t{4096};
    }();
    return _cached;
}

// One dispatch-log data-ring session for a single GPU. The KFD owns the GTT
// backing; the SDK holds only the stream fd and its RAW_MMAP mapping.
struct dlog_session
{
    uint32_t gpu_id    = 0;
    int      stream_fd = -1;
    void*    smap      = MAP_FAILED;
    size_t   smap_len  = 0;

    kfd_dlog_stream_info info = {};

    // Touched ONLY by the ring-copier thread.
    ring_cursors cursors = {};
    // Batches copied out of the ring that had no free pipe slot. Reading the ring
    // is what keeps the firmware from lapping us, so it must never be skipped.
    std::deque<record_batch> overflow        = {};
    bool                     overflow_warned = false;

    // Per-session, so one GPU failing to arm never disables another.
    std::atomic<bool> ready = {false};

    // Set by the reader after a terminal (POLLHUP) stream has been drained: it is
    // then removed from the active poll set so a sticky HUP cannot spin poll().
    // Reader thread only; no cross-thread reader depends on it.
    bool quarantined = false;

    // Edge trigger for the recoverable-reset (POLLERR) warning: logged once when
    // the stream enters the error state, cleared when a real poll of it shows no
    // error, so a sticky POLLERR does not spam the log. Reader thread only.
    bool pollerr_active = false;
};

struct reader_state
{
    std::thread       thread = {};
    std::atomic<bool> stop   = {false};
    // Set by the reader once it has published everything it will ever publish.
    // `stop` only asks it to wind down; the reader still publishes a final batch
    // afterwards, so the processor must key its exit off this instead.
    std::atomic<bool> reader_done = {false};
    // Records copied out of the ring but not yet visible in the pipe: reserved
    // per region inside copy_pipes() before the first memcpy, discharged per batch
    // at publish(). The processor's closed-window GC reads this (acquire) before
    // pipe.empty() so it never expires a window whose EOP is already in the
    // reader's hand but not yet published. Reader is the sole writer. Covers only
    // COPIED EOPs (F4); EOPs still in the ring remain the accepted R10 residual.
    std::atomic<uint64_t> reader_copied_unpublished = {0};
    // Read by nudge_reader() (from wait_for_reader_quiesce, ~500 Hz) while
    // stop_reader() writes them; atomic so those overlap without a data race and a
    // nudge never writes into a recycled fd.
    std::atomic<int>  wake_fd = {-1};
    std::atomic<bool> running = {false};

    int kfd_fd = -1;

    // Fixed array, not a container, so the atfork child handler can walk it with
    // plain indexing -- no allocation, no iterator invalidation.
    std::array<dlog_session, kMaxSessions> sessions      = {};
    std::atomic<size_t>                    session_count = {0};

    // Set up on the app thread, torn down on the finalize thread.
    std::mutex setup_mu = {};
    // True once ANY session is armed: lets the reader skip the whole poll/drain
    // path until a session exists, without walking the array every iteration.
    std::atomic<bool> any_session_ready = {false};
    // Bumped whenever the set of pollable stream fds changes -- a session is
    // published or the reader quarantines a terminal one. The reader rebuilds its
    // pollfds only when this differs from the generation it last built against.
    std::atomic<uint64_t> pollset_generation = {0};
    // Latched when the reader can never serve a session, as opposed to a single
    // GPU failing to arm.
    std::atomic<bool> reader_unavailable = {false};
    // Bumped once per completed drain pass. A sync point waits for it to advance
    // twice, which proves a whole pass ran after the request.
    std::atomic<uint64_t> drain_epoch = {0};

    // Copier is the sole producer, processor the sole consumer.
    record_pipe<kBatchPipeDepth> pipe             = {};
    std::thread                  processor_thread = {};
    std::atomic<uint64_t>        overflow_peak    = {0};

    reader_state() = default;
    ~reader_state();

    reader_state(const reader_state&) = delete;
    reader_state& operator=(const reader_state&) = delete;
};

// gpu_id -> (overruns, untrusted_records, overflow_peak) last reported, so an
// unchanged tuple stays quiet.
using overrun_report_map = std::unordered_map<uint32_t, std::array<uint64_t, 3>>;

overrun_report_map&
overrun_reported()
{
    static auto*& _v = common::static_object<overrun_report_map>::construct();
    return *_v;
}

reader_state&
state()
{
    static auto*& _v = common::static_object<reader_state>::construct();
    return *_v;
}

// Tear a session down: unmap the RAW_MMAP mapping and close the stream fd. The
// KFD owns the GTT backing, so there is nothing else to release from the SDK.
void
teardown_session(dlog_session* s);

// One-call, KFD-owned stream setup: OPEN_STREAM against the target process + GPU
// with the requested ring size and RAW_MMAP, then validate the returned geometry
// EXACTLY against the request before mapping it. Any mismatch is rejected rather
// than silently accepted.
//
// `permanent` distinguishes a retryable failure (device not ready yet) from an
// ABI/geometry disagreement that retrying cannot fix.
bool
setup_session(int kfd, uint32_t gpu_id, dlog_session* s, bool* permanent = nullptr)
{
    if(permanent) *permanent = false;
    s->gpu_id = gpu_id;

    // buffer_size is a uint32 ioctl field; ring_bytes() is bounded to fit it.
    static_assert(kDlogMaxRingBytes <= 0xFFFFFFFFull,
                  "dlog ring size must fit the uint32 buffer_size ioctl field");
    const uint64_t buf_bytes = ring_bytes();

    // From here on resources are acquired, so every failure return must unwind.
    bool                     success = false;
    common::scope_destructor cleanup{[&]() {
        if(!success) teardown_session(s);
    }};

    const uint32_t target_pid = static_cast<uint32_t>(getpid());

    auto open             = kfd_ioctl_profiler_args{};
    open.op               = KFD_IOC_PROFILER_DLOG;
    open.dlog.dlog_op     = KFD_IOC_PROFILER_DLOG_OPEN_STREAM;
    open.dlog.gpu_id      = gpu_id;
    open.dlog.target_pid  = target_pid;
    open.dlog.flags       = KFD_DLOG_OPEN_F_RAW_MMAP;
    open.dlog.buffer_size = static_cast<uint32_t>(buf_bytes);
    open.dlog.stream_fd   = -1;
    if(ioctl(kfd, AMDKFD_IOC_PROFILER, &open) != 0 || open.dlog.stream_fd < 0)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: OPEN_STREAM failed (errno={})", errno);
        return false;
    }
    s->stream_fd = open.dlog.stream_fd;

    auto sinfo = kfd_dlog_stream_args{};
    sinfo.op   = KFD_DLOG_STREAM_OP_INFO;
    if(ioctl(s->stream_fd, KFD_DLOG_STREAM_IOC, &sinfo) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log: STREAM_OP_INFO failed";
        return false;
    }
    s->info = sinfo.info;

    // Exact-contract check: the kernel must return the ABI, record size, the ring
    // size we asked for, and route to the GPU + process we requested. A driver
    // that altered the geometry, or a mis-routed stream that would silently stamp
    // records with the wrong gpu_id, is a contract violation -- not a value we
    // quietly accept.
    if(s->info.abi_version != KFD_DLOG_STREAM_ABI_VERSION ||
       s->info.fw_record_size != KFD_DISPATCH_LOG_FW_RECORD_BYTES ||
       s->info.buffer_size != buf_bytes || s->info.gpu_id != gpu_id ||
       s->info.target_pid != target_pid)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: stream contract mismatch (abi_version={} fw_record_size={} "
            "buffer_size={} requested={} gpu_id={}/{} target_pid={}/{})",
            s->info.abi_version,
            s->info.fw_record_size,
            s->info.buffer_size,
            buf_bytes,
            s->info.gpu_id,
            gpu_id,
            s->info.target_pid,
            target_pid);
        if(permanent) *permanent = true;  // ABI disagreement: retrying cannot help
        return false;
    }

    // Exact-geometry check against the kernel's canonical BO layout, in one pure
    // validator so the same logic is unit-tested without a live stream. It rejects
    // an unsupported region layout copy_pipes() could not drain and requires every
    // offset, the derived record count, the page-aligned mmap size, and region
    // disjointness to match the ABI exactly.
    const auto _geom = validate_stream_geometry(stream_geometry{s->info.num_regions,
                                                                s->info.region_record_count,
                                                                s->info.buffer_size,
                                                                s->info.mmap_size,
                                                                s->info.records_offset,
                                                                s->info.wptr_offset,
                                                                s->info.rptr_offset},
                                                buf_bytes,
                                                page_size());
    if(!_geom.ok)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: stream geometry mismatch ({}) (num_regions={} "
            "region_record_count={} buffer_size={} requested={} mmap_size={} records_offset={} "
            "wptr_offset={} rptr_offset={})",
            geometry_reason_name(_geom.reason),
            s->info.num_regions,
            s->info.region_record_count,
            s->info.buffer_size,
            buf_bytes,
            s->info.mmap_size,
            s->info.records_offset,
            s->info.wptr_offset,
            s->info.rptr_offset);
        if(permanent) *permanent = true;  // ABI disagreement: retrying cannot help
        return false;
    }
    s->smap_len = static_cast<size_t>(_geom.mmap_len);

    s->smap = mmap(nullptr, s->smap_len, PROT_READ | PROT_WRITE, MAP_SHARED, s->stream_fd, 0);
    if(s->smap == MAP_FAILED)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: mmap stream failed (errno={} mmap_size={} smap_len={} "
            "num_regions={} region_records={})",
            errno,
            s->info.mmap_size,
            s->smap_len,
            s->info.num_regions,
            s->info.region_record_count);
        return false;
    }

    ROCP_INFO << fmt::format(
        "KFD dispatch-log: session ready gpu_id={} ring_bytes={} num_regions={} region_records={} "
        "rec_bytes={}",
        gpu_id,
        buf_bytes,
        s->info.num_regions,
        s->info.region_record_count,
        s->info.fw_record_size);
    success = true;  // disarm cleanup: the session is fully built
    return true;
}

void
teardown_session(dlog_session* s)
{
    // Reset every per-session drain cursor so a slot reused after stop/restart
    // starts from a clean state. Leaving cursors set would carry rptr_init=true
    // and a stale rptr[] into the new (zeroed) stream, and copy_pipes() would then
    // skip the first-drain resync and drain nothing -- silent total data loss.
    s->cursors = {};
    // The reader drains every overflow batch into the pipe before publishing
    // reader_done, so a non-empty deque here means a copied batch is about to be
    // discarded -- exactly the data loss that drain guards against.
    ROCP_CI_LOG_IF(WARNING, !s->overflow.empty()) << fmt::format(
        "KFD dispatch-log: tearing down gpu_id={} with {} undelivered overflow batch(es); "
        "records copied from the ring were dropped before the processor saw them",
        s->gpu_id,
        s->overflow.size());
    s->overflow.clear();
    s->overflow_warned = false;
    s->quarantined     = false;
    s->pollerr_active  = false;

    if(s->smap != MAP_FAILED)
    {
        munmap(s->smap, s->smap_len);
        s->smap = MAP_FAILED;
    }
    if(s->stream_fd >= 0)
    {
        ::close(s->stream_fd);
        s->stream_fd = -1;
    }
}

// Reader thread only.
void
note_overflow_depth(reader_state& st, dlog_session& s)
{
    const auto _depth = s.overflow.size();
    if(_depth > st.overflow_peak.load(std::memory_order_relaxed))
        st.overflow_peak.store(_depth, std::memory_order_relaxed);

    if(_depth <= kOverflowWarnDepth || s.overflow_warned) return;
    s.overflow_warned = true;
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log (gpu_id={}): the record processor is not keeping up -- {} batch(es) "
        "queued. Records are still being read, so nothing is lost, but memory grows until it "
        "catches up.",
        s.gpu_id,
        _depth);
}

// STAGE 1, reader thread: copy the ring into a batch and get out. This is the
// ONLY code touching the volatile mapping, and every microsecond here widens
// the window for the firmware to lap us -- so no lock, no map, no pairing.
//
// The ring is drained on EVERY pass, even when the pipe is full: copy_pipes()
// advances rptr, so skipping the read is what lets the firmware lap us. Batches
// with nowhere to go queue in the session's overflow and enter the pipe, in
// order, as slots free.
uint64_t
copy_records(reader_state& st)
{
    uint64_t     _total = 0;
    const size_t _n     = st.session_count.load(std::memory_order_acquire);

    // The rings are independent, so one GPU lapping cannot stall another. A
    // quarantined session has already been drained past its terminal state and
    // its mapping may be inert, so it is skipped here as well as in the poll set.
    for(size_t i = 0; i < _n; ++i)
    {
        auto& _s = st.sessions[i];
        if(!_s.ready.load(std::memory_order_acquire) || _s.quarantined) continue;

        auto* base     = static_cast<uint8_t*>(_s.smap);
        auto* recs     = base + _s.info.records_offset;
        auto* wptr_arr = reinterpret_cast<volatile uint64_t*>(base + _s.info.wptr_offset);
        auto* rptr_arr = reinterpret_cast<volatile uint64_t*>(base + _s.info.rptr_offset);

        // Only take a slot when nothing is queued ahead of this batch, or it would
        // reach the processor before the older ones.
        record_batch* _batch  = _s.overflow.empty() ? st.pipe.acquire() : nullptr;
        const bool    _direct = (_batch != nullptr);
        if(!_direct) _batch = &_s.overflow.emplace_back();

        _batch->now_ns     = common::timestamp_ns();
        _batch->gpu_id     = _s.gpu_id;
        const auto _copied = copy_pipes(recs,
                                        _s.info.num_regions,
                                        _s.info.region_record_count,
                                        wptr_arr,
                                        rptr_arr,
                                        _s.cursors,
                                        _batch->records,
                                        &st.reader_copied_unpublished);

        if(_direct)
        {
            // Release-store inside publish() orders the copy above before the
            // processor can observe the batch. Discharge the reservation this
            // batch made (_copied == _batch->records.size() on the direct path)
            // right after publish, so a copied EOP is counted continuously from
            // copy_pipes' reserve until the processor can see it.
            if(_copied > 0)
            {
                st.pipe.publish();
                st.reader_copied_unpublished.fetch_sub(_copied, std::memory_order_release);
            }
        }
        else
        {
            if(_copied == 0) _s.overflow.pop_back();
            // Records parked in overflow stay reserved until their spill publishes
            // them, so window (1) is covered by the same counter.
            spill_overflow_to_pipe(st.pipe, _s.overflow, &st.reader_copied_unpublished);
            note_overflow_depth(st, _s);
        }
        _total += _copied;
    }
    return _total;
}

// Terminal-path drain-completeness check for ONE session: does any region still
// hold records the reader has not consumed? This is NOT the steady-state readiness
// authority (that is the kernel's level-triggered poll) -- it exists only so the
// terminal handler can defer quarantine until a HUP stream is level-dry, so a
// stream that reported POLLIN|POLLHUP together is not quarantined with records
// still pending. Reader thread only.
bool
session_has_pending(const dlog_session& s)
{
    if(s.smap == MAP_FAILED) return false;
    const auto* base     = static_cast<const uint8_t*>(s.smap);
    const auto* wptr_arr = reinterpret_cast<const volatile uint64_t*>(base + s.info.wptr_offset);
    const auto* rptr_arr = reinterpret_cast<const volatile uint64_t*>(base + s.info.rptr_offset);

    // setup_session() rejects num_regions > kMaxRegions permanently, so a live
    // session's count is always in range -- no clamp needed here.
    for(uint32_t r = 0; r < s.info.num_regions; ++r)
    {
        const uint64_t w = __atomic_load_n(&wptr_arr[r], __ATOMIC_ACQUIRE);
        const uint64_t c = __atomic_load_n(&rptr_arr[r], __ATOMIC_ACQUIRE);
        if(w != c) return true;
    }
    return false;
}

// STAGE 2, processor thread: pairs start/eop and drives the hub's time-as-
// generation resolution. All the lock-taking work lives here, off
// the ring-reading path. It runs no client callback itself; hand_off_proven()
// submits to the task group. Exactly one hub lock per record, no map, no clock.
uint64_t
process_batch(processor_state& proc, const record_batch& batch)
{
    const uint32_t _gpu     = batch.gpu_id;
    auto&          _pairing = proc.for_gpu(_gpu);

    // evict stale retained STARTs BEFORE pairing, watermarked by THIS
    // batch's own copy time (never the wall clock), throttled per GPU.
    // Ordering it ahead of pairing keeps a START the watermark has condemned from
    // binding this batch's recycled EOP first; watermarking by batch.now_ns means a
    // backlogged processor ages nothing. starts_evicted is an R1 source counter.
    if(batch.now_ns - _pairing.last_evict_ns >= kProcessorEvictIntervalNs)
    {
        _pairing.starts_evicted += _pairing.evict_stale(batch.now_ns, start_max_age_ns());
        _pairing.last_evict_ns = batch.now_ns;
    }

    return pair_records(
        batch.records.data(),
        batch.records.size(),
        _pairing,
        batch.now_ns,
        [_gpu](const drained_record& rec) {
            // Torn records were dropped at the top of pair_records, so
            // every record here is trusted; no drain_loss_free gate remains.
            // gpu_id stamped from the ring this record came from: a record can only
            // ever match a dispatch enqueued on the same GPU. No generation.
            auto key =
                correlation_key{doorbell_off_to_page_slot(rec.doorbell_off), rec.dispatch_id, _gpu};
            auto _start = rec.start_known ? std::optional<uint64_t>{rec.start_ticks} : std::nullopt;

            // Signal-less: this EOP IS the completion event. The hub selects the
            // entry whose window contains the START tick.
            if(auto _proven = signal_less_hub().record_kernel_end(key, _start, rec.end_ticks))
            {
                note_signal_less(signal_less_counter::eop_proven);
                hand_off_proven(std::move(*_proven));
                return;
            }
            note_signal_less(signal_less_counter::eop_unmatched);
        });
}

// Two distinct conditions, reported separately so a growing processor backlog is
// never mislabelled as a ring lap. A ring overrun is a real, bounded data loss:
// the firmware lapped records the reader had not copied. A backlog peak is not a
// loss: those batches were copied out of the ring and are delivered to the
// processor as slots free -- it only signals the processor briefly fell behind.
// Each is rate-limited on its own last-reported value. Reader thread only, so the
// statics need no lock.
void
report_overrun(uint32_t gpu_id, const ring_cursors& c, uint64_t overflow_peak)
{
    // Per GPU: one ring lapping (or one GPU's processor backlog) says nothing
    // about another's. .first tracks laps, .second tracks the backlog peak.
    auto& _prev = overrun_reported()[gpu_id];

    // An untrusted record is dropped downstream, so it is real coverage loss even
    // when nothing was lapped (exactly-full, overruns==0) -- report it too.
    if((c.overruns > 0 || c.untrusted_records > 0) &&
       (c.overruns != _prev[0] || c.untrusted_records != _prev[1]))
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log (gpu_id={}): ring overrun -- {} lap(s) so far, at least {} "
            "record(s) lost, {} record(s) untrusted (dropped). Those dispatches have no "
            "dispatch-log timestamps; everything else is unaffected and collection continues. "
            "Raise the ring with ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB if this repeats.",
            gpu_id,
            c.overruns,
            c.lost_records,
            c.untrusted_records);

    if(overflow_peak > 0 && overflow_peak != _prev[2])
        ROCP_INFO << fmt::format(
            "KFD dispatch-log (gpu_id={}): processor backlog peaked at {} batch(es); the "
            "processor briefly fell behind. No records are lost -- backlogged batches are "
            "delivered as it catches up.",
            gpu_id,
            overflow_peak);

    _prev = {c.overruns, c.untrusted_records, overflow_peak};
}

// KFD_DLOG_STREAM_OP_STATUS is DIAGNOSTICS ONLY: the kernel's counters are logged
// but never gate a data decision (wptr is the design's sole overrun authority).
// The status word is used only to describe a terminal stream when quarantining
// it, never to decide whether to drain.
void
log_stream_status(const dlog_session& s)
{
    if(s.stream_fd < 0) return;
    auto args = kfd_dlog_stream_args{};
    args.op   = KFD_DLOG_STREAM_OP_STATUS;
    if(ioctl(s.stream_fd, KFD_DLOG_STREAM_IOC, &args) != 0) return;
    ROCP_INFO << fmt::format("KFD dispatch-log status: flags=0x{:x} target_exit_count={}",
                             args.status.status,
                             args.status.target_exit_count);
}

// Read the STREAM_OP_STATUS flags word (F11 FATAL detection). Returns 0 on a dead
// fd or a failed ioctl: 0 has no bits set, so the FATAL test fails and the caller
// falls back to the recoverable-reset handling -- safe, because a genuinely fatal
// stream also stops producing and will re-assert POLLERR (or HUP) on the next pass.
// Unlike log_stream_status this returns the word so a data decision can be made:
// FATAL is the one status bit that gates handling, per the driver contract.
uint64_t
read_stream_status_flags(const dlog_session& s)
{
    if(s.stream_fd < 0) return 0;
    auto args = kfd_dlog_stream_args{};
    args.op   = KFD_DLOG_STREAM_OP_STATUS;
    if(ioctl(s.stream_fd, KFD_DLOG_STREAM_IOC, &args) != 0) return 0;
    return args.status.status;
}

void
stop_reader()
{
    auto& st = state();

    // setup_mu is taken BEFORE reading running: reading it first (as before) is a
    // no-op that misses a concurrent establish_session() mid-arm before it publishes
    // running=true, which would then create reader threads and publish a session
    // after teardown. establish_session() re-checks get_fini_status() and
    // reader_unavailable UNDER this same lock, so latching reader_unavailable here
    // makes the in-flight arm refuse. The whole terminating sequence runs in this one
    // critical section; setup_mu is a leaf (no worker thread acquires it), so holding
    // it across the joins cannot deadlock. stop+wake precede the joins so the join is
    // bounded rather than hanging while holding setup_mu.
    auto _lk = std::lock_guard<std::mutex>{st.setup_mu};
    if(!st.running.load(std::memory_order_relaxed)) return;

    // Latch unavailable before the join so no dispatch can acquire a correlation
    // key against a reader that is on its way out (cleared again below only on a
    // non-finalize stop).
    st.reader_unavailable.store(true, std::memory_order_release);
    st.stop.store(true, std::memory_order_release);
    const int _wake = st.wake_fd.load(std::memory_order_acquire);
    if(_wake >= 0)
    {
        uint64_t one = 1;
        // Best-effort wake BEFORE any join, so the reader observes stop and exits
        // rather than the join hanging while setup_mu is held.
        [[maybe_unused]] auto _rc = ::write(_wake, &one, sizeof(one));
    }
    if(st.thread.joinable()) st.thread.join();
    // Reader first: it is the sole producer, so once it is joined the pipe can
    // only shrink. The processor then drains what is left and exits.
    if(st.processor_thread.joinable()) st.processor_thread.join();

    const size_t _n = st.session_count.load(std::memory_order_acquire);
    for(size_t i = 0; i < _n; ++i)
    {
        st.sessions[i].ready.store(false, std::memory_order_release);
        teardown_session(&st.sessions[i]);
    }
    st.session_count.store(0, std::memory_order_release);
    st.any_session_ready.store(false, std::memory_order_release);
    // Finalize-path stop: leave reader_unavailable LATCHED so a dispatch that
    // slipped past establish_session()'s pre-lock checks cannot re-arm the reader
    // under setup_mu after us. A non-finalize stop (a client detach that may
    // re-attach) clears it so a later establish_session() can restart the reader.
    if(registration::get_fini_status() == 0)
        st.reader_unavailable.store(false, std::memory_order_release);

    if(st.kfd_fd >= 0)
    {
        ::close(st.kfd_fd);
        st.kfd_fd = -1;
    }
    // The wake eventfd is intentionally left open (§3.8): it is created once per
    // process and never closed, so nudge_reader()'s load/write is always safe and a
    // later start reuses it. Any residual count is harmlessly drained by the next
    // reader's first control-fd read.
    st.running.store(false, std::memory_order_release);
    ROCP_INFO << "KFD dispatch-log reader: stopped";
}

reader_state::~reader_state()
{
    // F13 part 2: in a fork child (D6's unconditional fork generation) the reader
    // thread does not exist and setup_mu / the session containers may be mid-mutation
    // by a vanished thread. Skip stop_reader()'s join+teardown -- the same treatment
    // D6 gives TaskGroup (skip the dangerous thread work; the child abandoned this
    // state via disable_reader_in_child). Value members destruct as they do there.
    if(internal_threading::fork_stale()) return;
    stop_reader();
}

// pthread_atfork child handler. Only the forking thread survives, so this does
// atomic stores and fd/mapping drops only -- no lock, no allocation, no join.
void
disable_reader_in_child()
{
    // Also latches kfd_dispatch_log_available() false, so the child never takes
    // the DoorbellMap lock a vanished thread may have been holding.
    disable_kfd_dispatch_log();

    signal_less_abandon_in_child();

    auto& st = state();
    st.any_session_ready.store(false, std::memory_order_relaxed);
    // Latch unavailable: setup_mu may have been held by a thread that did not
    // survive the fork, so a child dispatch must not reach it.
    st.reader_unavailable.store(true, std::memory_order_relaxed);
    st.running.store(false, std::memory_order_relaxed);  // stop_reader()/dtor become no-ops
    // Not detach(): that would leave the handle believing a thread exists.
    new(&st.thread) std::thread{};
    new(&st.processor_thread) std::thread{};
    // Drop the reference WITHOUT closing (§3.8): the persistent wake eventfd's number
    // must never be recycled under the parent's nudge_reader; the child sets it -1 so
    // its own nudge is inert and a later child reader creates a fresh one.
    st.wake_fd.store(-1, std::memory_order_relaxed);
    // F13: close the inherited fds/maps. After fork the child has its own fd table,
    // so close()/munmap() here release the child's copies WITHOUT touching the
    // parent's descriptors (the old "corrupts the parent" comment was wrong). Both
    // are async-signal-safe, which the atfork child handler requires.
    if(st.kfd_fd >= 0) ::close(st.kfd_fd);
    st.kfd_fd       = -1;
    const size_t _n = st.session_count.load(std::memory_order_relaxed);
    for(size_t i = 0; i < _n && i < kMaxSessions; ++i)
    {
        st.sessions[i].ready.store(false, std::memory_order_relaxed);
        if(st.sessions[i].smap != MAP_FAILED) munmap(st.sessions[i].smap, st.sessions[i].smap_len);
        st.sessions[i].smap = MAP_FAILED;
        if(st.sessions[i].stream_fd >= 0) ::close(st.sessions[i].stream_fd);
        st.sessions[i].stream_fd = -1;
        // Same reuse hazard as teardown_session(): session_count is zeroed here, so
        // any later re-arm in the child reuses these slots. Reset the drain cursors
        // so a reused slot cannot inherit a stale rptr and drain nothing.
        st.sessions[i].cursors = {};
    }
    st.session_count.store(0, std::memory_order_relaxed);
}

// Sole consumer of the pipe; owns all pairing state, so it needs no lock for it.
void
processor_loop()
{
    auto& st   = state();
    auto  proc = processor_state{};

    uint64_t last_gc_ns = common::timestamp_ns();

    while(true)
    {
        // GC closed-window entries on the periodic tick, but ONLY when the pipe is
        // empty: an EOP that would resolve a closed window may be sitting in a
        // backlog behind us, and GC'ing its window first would expire it as a leak
        // and then drop the EOP as start-unknown. Gating on an empty pipe means we
        // only expire windows once no queued batch can still close them. The idle
        // GPU (a queue destroyed after its work finished, the dominant case) still
        // GCs promptly -- its pipe drains to empty and stays there, so the
        // close_grace_ns bound is honored. Residual: a GPU whose pipe never empties
        // (a continuous backlog) defers GC indefinitely, but a continuous backlog
        // means the processor is actively resolving windows, so nothing is leaking.
        // Leaked payloads are ledgered; dropping them here is off the lock.
        // In-flight load FIRST, then pipe.empty(): reversing them is unsound. A
        // zero read here (acquire) means every record the reader took out of the
        // ring was published BEFORE this load, and the acquire pairs with the
        // reader's release so the following pipe.empty() is guaranteed to see it.
        // Covers only copied-but-unpublished EOPs (F4); an EOP still in the ring is
        // the accepted R10 residual and is not gated here.
        // Sequenced into named locals rather than passed inline: the order of
        // evaluation of function arguments is UNSPECIFIED, so the required
        // "in-flight load, THEN pipe.empty()" ordering cannot be expressed by
        // argument position.
        const uint64_t _gc_now      = common::timestamp_ns();
        const uint64_t _unpublished = st.reader_copied_unpublished.load(std::memory_order_acquire);
        const bool     _pipe_empty  = st.pipe.empty();
        if(gc_gate_open(_unpublished, _pipe_empty) &&
           _gc_now - last_gc_ns >= kProcessorEvictIntervalNs)
        {
            last_gc_ns = _gc_now;
            auto _gc   = signal_less_hub().gc_closed_windows(steady_now_ns());
            if(_gc.second.dispatches > 0) note_signal_less_losses();
        }

        auto* _batch = st.pipe.peek();
        if(!_batch)
        {
            if(st.reader_done.load(std::memory_order_acquire))
            {
                // Acquire above pairs with the reader's release: everything it
                // published is visible now, so re-check before leaving. The batch
                // could have been published between the peek and the load.
                if(st.pipe.peek() == nullptr) break;
                continue;
            }
            // Nothing to do. Sleeping here costs nothing on the ring: the reader
            // keeps copying regardless of what this thread is doing.
            std::this_thread::sleep_for(std::chrono::microseconds{200});
            continue;
        }

        process_batch(proc, *_batch);
        st.pipe.pop();

        // Retained-start eviction and recycled-slot disambiguation are both
        // stream-driven inside process_batch (eviction at its top, watermarked by
        // each batch's own copy time; ambiguous same-key starts dropped during
        // pairing), so there is no per-loop purge or wall-clock timer here.
    }

    // Drain whatever the reader published on its way out.
    while(auto* _batch = st.pipe.peek())
    {
        process_batch(proc, *_batch);
        st.pipe.pop();
    }

    for(const auto& _gpu_itr : proc.by_gpu)
    {
        const auto& _p = _gpu_itr.second;
        if(_p.eops_seen == 0) continue;

        ROCP_WARNING << fmt::format(
            "KFD dispatch-log pairing census (gpu_id={}): {} START record(s) drained, {} EOP "
            "record(s) drained, {} EOP(s) unmatched, {} START(s) overwritten on a live key, {} "
            "ambiguous EOP(s) dropped, {} equal-tick EOP(s) dropped, {} stale EOP(s) dropped, {} "
            "START(s) evicted stale, {} START(s) evicted by the size cap, {} START(s) still "
            "retained at exit",
            _gpu_itr.first,
            _p.starts_seen,
            _p.eops_seen,
            _p.unmatched_eops,
            _p.starts_overwritten,
            _p.ambiguous_pairs,
            _p.equal_tick_drops,
            _p.stale_eop_drops,
            _p.starts_evicted,
            _p.starts_cap_evicted,
            _p.pending_starts.size());
    }
}

// Reader-thread only (the overflow deques are reader-owned): true when no live
// session still has copied-but-unpublished batches queued in its overflow.
bool
all_live_overflow_empty(const reader_state& st)
{
    for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
        if(!st.sessions[i].overflow.empty()) return false;
    return true;
}

// Bounded per-wake drain: copy every live session up to kMaxDrainPassesPerWake
// times, stopping early once a pass copies nothing. The cap is what stops a
// continuously advancing producer from livelocking the reader inside one wake;
// the fd stays readable, so poll() re-triggers to resume where this left off.
// Returns records seen.
uint64_t
drain_sessions_bounded(reader_state& st)
{
    uint64_t seen = 0;
    for(int _pass = 0; _pass < kMaxDrainPassesPerWake; ++_pass)
    {
        const uint64_t _copied = copy_records(st);
        seen += _copied;
        // Stage 1: advance drain_epoch only when every live session's overflow
        // is drained into the pipe, so the fence's +2 proves pre-request records
        // were PUBLISHED into the pipe, not merely copied into an overflow deque.
        if(all_live_overflow_empty(st)) st.drain_epoch.fetch_add(1, std::memory_order_release);
        if(_copied == 0) break;
    }
    for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
        report_overrun(st.sessions[i].gpu_id,
                       st.sessions[i].cursors,
                       st.overflow_peak.load(std::memory_order_relaxed));
    return seen;
}

// Rebuild the pollfds: the control eventfd first, then one entry per live,
// non-quarantined stream fd. Poll on POLLIN so the kernel wakes us on records or
// terminal state; a HUP is delivered regardless. `slot_of[k]` maps pollfd k>=1
// back to its session index. Returns the pollset_generation this was built for.
uint64_t
build_pollfds(reader_state& st, std::vector<pollfd>& fds, std::vector<size_t>& slot_of)
{
    const uint64_t _gen = st.pollset_generation.load(std::memory_order_acquire);
    fds.clear();
    slot_of.clear();
    // The control/wake eventfd is poll slot kControlPollSlot; stream fds follow.
    // any_stream_pollin() and the terminal scan rely on this exact ordering.
    fds.push_back(
        pollfd{.fd = st.wake_fd.load(std::memory_order_acquire), .events = POLLIN, .revents = 0});

    // Snapshot each session's poll attributes, then let the pure helper pick the
    // members: the fd skip and the pollable predicate are the same logic the
    // poll-set unit tests exercise.
    const size_t _n   = st.session_count.load(std::memory_order_acquire);
    const size_t _cap = _n < kMaxSessions ? _n : kMaxSessions;

    std::array<poll_session_view, kMaxSessions> _views = {};
    for(size_t i = 0; i < _cap; ++i)
    {
        auto& _s  = st.sessions[i];
        _views[i] = poll_session_view{_s.stream_fd >= 0,
                                      _s.ready.load(std::memory_order_acquire),
                                      _s.quarantined,
                                      _s.smap != MAP_FAILED};
    }
    build_poll_slots(_views.data(), _cap, slot_of);

    for(size_t _slot : slot_of)
        fds.push_back(pollfd{.fd = st.sessions[_slot].stream_fd, .events = POLLIN, .revents = 0});
    return _gen;
}

void
reader_loop()
{
    auto& st = state();

    const int _timeout_ms = poll_timeout_ms();
    ROCP_INFO << fmt::format("KFD dispatch-log reader: poll timeout {} ms", _timeout_ms);

    // Rebuilt only when pollset_generation changes; steady state reuses these.
    auto     fds         = std::vector<pollfd>{};
    auto     slot_of     = std::vector<size_t>{};
    uint64_t built_gen   = build_pollfds(st, fds, slot_of);
    uint64_t total_seen  = 0;
    uint64_t last_log_ns = common::timestamp_ns();

    // Sticky-EPOLLIN spin backstop: consecutive pre-timeout wakes that copied
    // nothing. Past the threshold the reader stops trusting the stream-fd level
    // trigger (polls only the control fd) until a copy makes progress, so an
    // incoherent rptr becomes a diagnosable warning, not a 100% CPU hang.
    int  empty_wakes     = 0;
    bool backstop_warned = false;

    while(!st.stop.load(std::memory_order_acquire))
    {
        if(st.pollset_generation.load(std::memory_order_acquire) != built_gen)
            built_gen = build_pollfds(st, fds, slot_of);

        for(auto& _pfd : fds)
            _pfd.revents = 0;

        // When the backstop is tripped, poll only the control eventfd so a sticky
        // stream EPOLLIN cannot return poll() immediately; the timeout still fires
        // the periodic full scan and a nudge/stop still wakes us.
        const bool   _backstop = empty_wake_backstop_tripped(empty_wakes);
        const nfds_t _nfds     = _backstop ? nfds_t{1} : static_cast<nfds_t>(fds.size());
        int          rc        = ::poll(fds.data(), _nfds, _timeout_ms);
        if(rc < 0 && errno != EINTR)
        {
            // Nothing will drain the ring again, so publish that terminally: this is
            // the same permanent-unavailable state setup failure uses. Do NOT
            // clear any_session_ready here -- records may still be unread, and the
            // fence keys success off `running` (liveness), never readiness, so this
            // path must reach the deadline and abandon rather than falsely succeed.
            // kfd_dispatch_log_available() is deliberately left alone so queue destroy
            // still retires doorbell-map entries.
            st.reader_unavailable.store(true, std::memory_order_release);
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log reader: poll failed (errno={}), reader exiting; dispatch-log is "
                "now disabled for this process, all dispatches use HSA timestamps",
                errno);
            break;
        }

        // Drain the control eventfd if it woke us. Its only job is to break poll();
        // a lifecycle nudge, a new session, or stop all use it.
        if((fds[kControlPollSlot].revents & POLLIN) != 0)
        {
            uint64_t v = 0;
            while(::read(fds[kControlPollSlot].fd, &v, sizeof(v)) == sizeof(v))
            {}
        }

        // A wake or timeout both trigger a full scan: the kernel's level trigger,
        // not the poll revents, is the readiness authority, so a timeout (the
        // sparse-tail and lost-interrupt watchdog) drains exactly like an interrupt
        // would. Hitting the pass cap with records still pending needs no self-nudge:
        // the stream fd stays readable (wptr != rptr), so poll() re-triggers on its
        // own -- the backstop below bounds the case where it re-triggers but the
        // drain still copies nothing.

        if(st.any_session_ready.load(std::memory_order_acquire))
        {
            const uint64_t _copied = drain_sessions_bounded(st);
            total_seen += _copied;

            // Feed the spin backstop. A stream that reported POLLIN or POLLERR but
            // copied nothing is unproductive; a control nudge is not. The latch holds
            // through a tripped (control-fd-only) timeout and clears only on real copy
            // progress or a genuine idle timeout while the stream was still polled --
            // so a sticky POLLERR neither busy-polls nor re-bursts (the earlier
            // rc==0 reset un-tripped it every timeout).
            const bool _unproductive =
                wake_is_unproductive(any_stream_pollin(fds.data(), fds.size()),
                                     any_stream_error(fds.data(), fds.size()),
                                     _copied);
            empty_wakes =
                next_empty_wakes(empty_wakes, _unproductive, _copied > 0, rc == 0, _backstop);

            // Warn exactly once per tripped episode: clear the latch when the backstop
            // is not tripped, warn once when it first is.
            if(!empty_wake_backstop_tripped(empty_wakes))
            {
                backstop_warned = false;
            }
            else if(!backstop_warned)
            {
                ROCP_WARNING << "KFD dispatch-log reader: level readiness reported but the drain "
                                "copied nothing repeatedly; falling back to a timeout-only wait to "
                                "avoid a busy spin. Records may still be collected on the periodic "
                                "scan.";
                backstop_warned = true;
            }
        }
        else if(all_live_overflow_empty(st))
        {
            // No session is armed, so the ready-set is trivially drained: advance
            // drain_epoch every pass so a fence blocked on drain progress is not made
            // to burn its whole timeout waiting for a +2 that drain_sessions_bounded
            // (only reached when a session IS ready) would otherwise never post. The
            // overflow guard still holds: a copied-but-unpublished batch must reach
            // the pipe before the epoch may advance.
            st.drain_epoch.fetch_add(1, std::memory_order_release);
        }

        // Terminal streams. POLLHUP/POLLNVAL are terminal: the stream will never
        // produce again, so it is drained first (above), then -- only once it is
        // level-dry -- logged and quarantined out of the poll set via a generation
        // bump, so no final record is dropped and a sticky HUP cannot spin poll().
        // POLLERR alone is a recoverable reset: the stream node survives (the reset
        // zeroed wptr[], which copy_pipes() reconciles), so it is logged but NOT
        // quarantined. A HUP still pending records defers to the next pass -- the fd
        // stays readable, so poll() re-triggers.
        bool _quarantined_any = false;
        for(size_t k = kFirstStreamPollSlot; k < fds.size(); ++k)
        {
            const terminal_action _act = classify_terminal_revents(fds[k].revents);
            auto&                 _s   = st.sessions[slot_of[k - kFirstStreamPollSlot]];
            if(_s.quarantined) continue;

            // F11: POLLERR alone is usually a recoverable reset, but a GPU reset
            // raises the FATAL latch (EPOLLERR without HUP -- fatal and terminal are
            // independent latches in the driver) after which the stream never produces
            // again ("no live consumer, do not arm"). Read STREAM_OP_STATUS to tell
            // them apart; FATAL is terminal.
            //
            // Also probed with NO revents at all while the empty-wake backstop is
            // tripped: the backstop polls only the control fd and zeroes the stream
            // revents, so classify_terminal_revents() reports `none` forever and a
            // stream that went fatal underneath it would never be re-examined -- the
            // session would stay `ready`, handing out correlation keys nothing can
            // ever deliver. Restricted to the backstop's poll TIMEOUT so the extra
            // ioctl runs at the poll-timeout cadence, not per wake.
            const bool _probe_fatal =
                (_act == terminal_action::keep_live) || (_backstop && rc == 0);
            if(_probe_fatal && (read_stream_status_flags(_s) & KFD_DLOG_STATUS_FATAL) != 0)
            {
                // Drain-first: defer while records remain (the top-of-loop drain
                // already ran this pass; firmware produces nothing more after
                // FATAL, so this converges) so no already-published record is lost.
                if(session_has_pending(_s) || !_s.overflow.empty()) continue;
                log_stream_status(_s);
                ROCP_WARNING << fmt::format(
                    "KFD dispatch-log: gpu_id={} stream FATAL (revents=0x{:x}); GPU reset, "
                    "draining, quarantining, and disabling signal-less process-wide",
                    _s.gpu_id,
                    static_cast<unsigned>(fds[k].revents));
                // Clear ready before quarantine so find_session()/establish_session()
                // stop handing out keys for a stream that can never deliver them.
                _s.ready.store(false, std::memory_order_release);
                _s.quarantined   = true;
                _quarantined_any = true;
                // Smallest safe containment: the reset invalidated the clock domain
                // this stream's windows were opened against, so drop the whole
                // signal-less path process-wide (hub disable latch drains+ledgers
                // live entries) rather than risk mis-windowing against a reset clock.
                //
                // Ring and overflow are dry above, but the batches already PUBLISHED
                // into the pipe are not: this session's final EOP, and the EOPs of
                // every healthy GPU queued behind it, are still waiting on the
                // processor. Draining the hub first would ledger their entries and
                // leave those records unmatched. So latch admission closed, let the
                // processor consume what is published (the same gate the closed-window
                // GC uses -- copied-unpublished first, then pipe.empty), and only then
                // ledger the residual. Bounded by the close budget so one wedged
                // processor cannot stall the ring for the other GPUs; on timeout the
                // disable proceeds exactly as it did before.
                signal_less_disable_latch().store(true, std::memory_order_release);
                const uint64_t _pipe_deadline = common::timestamp_ns() + close_drain_budget_ns();
                while(!gc_gate_open(st.reader_copied_unpublished.load(std::memory_order_acquire),
                                    st.pipe.empty()) &&
                      common::timestamp_ns() < _pipe_deadline &&
                      !st.stop.load(std::memory_order_acquire))
                    std::this_thread::sleep_for(std::chrono::microseconds{200});
                signal_less_disable_permanently();
                continue;
            }

            if(_act == terminal_action::none)
            {
                // A real poll of this stream showing no error clears the edge trigger
                // so a later POLLERR logs again. A tripped backstop polls only the
                // control fd (stream revents were zeroed), so that is not recovery.
                if(!_backstop) _s.pollerr_active = false;
                continue;
            }

            if(_act == terminal_action::keep_live)
            {
                // Recoverable reset: describe it ONCE per episode -- POLLERR is
                // returned by poll() every pass, so an unlatched warning would spin
                // the log -- then keep the stream live. The wake already fed the
                // empty-wake backstop above, which bounds the CPU cost.
                if(!_s.pollerr_active)
                {
                    log_stream_status(_s);
                    ROCP_WARNING << fmt::format(
                        "KFD dispatch-log: gpu_id={} stream POLLERR (revents=0x{:x}); treating as "
                        "a recoverable reset, stream kept live",
                        _s.gpu_id,
                        static_cast<unsigned>(fds[k].revents));
                    _s.pollerr_active = true;
                }
                continue;
            }

            // Drain-first-then-quarantine: defer while records remain so terminal
            // records are not discarded at the bounded-pass cap. session_has_pending
            // only inspects the ring; a batch already copied into this session's
            // reader-owned overflow deque but not yet spilled to the pipe would be
            // stranded by quarantine (copy_records skips !ready sessions), so defer
            // while overflow is non-empty too.
            if(session_has_pending(_s) || !_s.overflow.empty()) continue;

            log_stream_status(_s);
            ROCP_INFO << fmt::format(
                "KFD dispatch-log: gpu_id={} stream terminal (revents=0x{:x}); drained and "
                "quarantined",
                _s.gpu_id,
                static_cast<unsigned>(fds[k].revents));
            // Quarantine is terminal: clear ready BEFORE marking it quarantined so
            // find_session()/establish_session() stop handing out correlation keys
            // for a session that can never deliver them.
            _s.ready.store(false, std::memory_order_release);
            _s.quarantined   = true;
            _quarantined_any = true;
        }
        if(_quarantined_any) st.pollset_generation.fetch_add(1, std::memory_order_release);

        const uint64_t _now = common::timestamp_ns();
        if(_now - last_log_ns >= kLoggingIntervalNs)
        {
            for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
                log_stream_status(st.sessions[i]);
            last_log_ns = _now;
        }
    }

    // Final copy to catch late records; the processor drains the pipe after us.
    if(st.any_session_ready.load(std::memory_order_acquire))
    {
        total_seen += copy_records(st);
        for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
        {
            report_overrun(st.sessions[i].gpu_id,
                           st.sessions[i].cursors,
                           st.overflow_peak.load(std::memory_order_relaxed));
            log_stream_status(st.sessions[i]);
        }
    }

    // Hand off every remaining overflow batch before declaring done. copy_records()
    // only spills as many as the pipe had free slots, so a pipe that was full at
    // stop leaves overflow batches -- already taken from the ring, so gone from the
    // KFD's view -- that the processor would otherwise never see (it reads only the
    // pipe) before teardown_session() clears the deque. The processor is joined
    // AFTER this thread, so it keeps consuming and freeing slots; wait for slots to
    // free rather than busy-spinning. reader_done is stored only once all deques are
    // empty, so the processor's acquire load observes every batch first.
    {
        const size_t _n = st.session_count.load(std::memory_order_acquire);
        bool         _pending;
        do
        {
            _pending = false;
            for(size_t i = 0; i < _n; ++i)
            {
                if(spill_overflow_to_pipe(
                       st.pipe, st.sessions[i].overflow, &st.reader_copied_unpublished))
                    continue;
                _pending = true;
            }
            if(_pending) std::this_thread::sleep_for(std::chrono::microseconds{200});
        } while(_pending);
    }

    // Release: pairs with the processor's acquire so it cannot decide the producer
    // is finished before the final batch above is visible to it.
    st.reader_done.store(true, std::memory_order_release);

    ROCP_INFO << fmt::format("KFD dispatch-log reader: loop exited, total pairs seen = {}",
                             total_seen);

    // Silent unless signal-less is active, so the default path logs exactly what
    // it did before. The chain itself is summarised once, by teardown.
    uint64_t _ring_overruns  = 0;
    uint64_t _ring_lost      = 0;
    uint64_t _ring_untrusted = 0;
    for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
    {
        _ring_overruns += st.sessions[i].cursors.overruns;
        _ring_lost += st.sessions[i].cursors.lost_records;
        _ring_untrusted += st.sessions[i].cursors.untrusted_records;
    }

    ROCP_WARNING_IF(signal_less_feature_enabled() &&
                    (_ring_overruns > 0 || _ring_lost > 0 || _ring_untrusted > 0))
        << fmt::format(
               "KFD dispatch-log ring: {} lap(s), {} record(s) lost to laps, {} record(s) "
               "untrusted (dropped), processor backlog peaked at {} batch(es) -- a lost or "
               "untrusted record is a lost START, which shows up as a start-unknown no-timing",
               _ring_overruns,
               _ring_lost,
               _ring_untrusted,
               st.overflow_peak.load(std::memory_order_relaxed));
}
}  // namespace

bool
start_kfd_reader()
{
    auto& st = state();
    if(st.running.load(std::memory_order_acquire)) return true;

    // Any return below that leaves st.running false is a failure: drop the fds and
    // latch the reader unavailable, so dispatches short-circuit in
    // ensure_reader_session() instead of locking setup_mu for a reader that will
    // never have a session.
    common::scope_destructor cleanup{[&st]() {
        if(st.running.load(std::memory_order_acquire)) return;
        st.reader_unavailable.store(true, std::memory_order_release);
        if(st.kfd_fd >= 0) ::close(st.kfd_fd);
        st.kfd_fd = -1;
        // The wake eventfd is deliberately NOT closed here (§3.8): it lives for the
        // process lifetime so its number can never be recycled under nudge_reader().
    }};

    // Create the wake eventfd once per process and never close it (§3.8). A closed
    // fd number can be recycled by any open/socket/accept between nudge_reader()'s
    // load and its write, sending the wake into an unrelated descriptor exactly when
    // fds churn at teardown. A single process-lifetime EFD_CLOEXEC descriptor makes
    // the load/write unconditionally safe with no synchronization.
    if(st.wake_fd.load(std::memory_order_acquire) < 0)
    {
        const int _wake_fd = eventfd(0, kEventfdFlags);
        if(_wake_fd < 0)
        {
            ROCP_WARNING << "KFD dispatch-log reader: eventfd creation failed, reader not started";
            return false;
        }
        st.wake_fd.store(_wake_fd, std::memory_order_release);
    }

    st.kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(st.kfd_fd < 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: /dev/kfd open failed, reader not started";
        return false;
    }

    // Registered before the thread exists: a fork in between would inherit a reader
    // thread with no handler to abandon it. Registered exactly once per process:
    // pthread_atfork stacks handlers, so a stop/restart cycle would otherwise pile up
    // a duplicate child handler on every restart. call_once leaves the flag unset if
    // the lambda throws, so a transient registration failure can still be retried on a
    // later start rather than being latched.
    static std::once_flag _atfork_once;
    try
    {
        std::call_once(_atfork_once, []() {
            if(int _rc = pthread_atfork(nullptr, nullptr, disable_reader_in_child); _rc != 0)
                throw std::system_error{_rc, std::generic_category(), "pthread_atfork"};
        });
    } catch(const std::exception&)
    {
        ROCP_WARNING << "KFD dispatch-log reader: pthread_atfork failed, reader not started";
        return false;
    }

    st.stop.store(false, std::memory_order_release);
    st.reader_done.store(false, std::memory_order_release);
    // Reset before any worker thread exists: teardown_session() drops a prior
    // reader's unspilled overflow batches whose reservations never published, so a
    // stale residual here would keep the GC gate closed for this whole instance.
    st.reader_copied_unpublished.store(0, std::memory_order_release);

    // Force-construct the singletons the reader/processor hot paths touch here,
    // BEFORE the worker threads exist, so a first-use allocation cannot stall the
    // ring read and turn into an overrun. state() was already constructed at the
    // top of this function; overrun_reported() is the other one on the drain path.
    // Doing it from this one thread before the workers start also means their init
    // needs no cross-thread barrier -- construction strictly happens-before either
    // worker runs. (The downstream signal-less singletons the processor reaches on
    // a proven completion are constructed lazily, but they run off the pipe, not
    // the ring, so a first-use allocation there cannot cause an overrun.)
    overrun_reported();

    // Same reasoning for the env-backed knobs: each caches its read in a
    // function-local static on first use, and common::get_env* scans `environ`,
    // which races a concurrent setenv. Populate them all HERE, on the one
    // serialized start path under setup_mu with no worker thread yet in existence,
    // instead of letting a processor/producer/teardown thread do it lazily.
    start_max_age_ns();
    pending_starts_cap();
    signal_less_prime_env();

    // Processor first: it must be ready to consume before the reader can publish,
    // otherwise the first batches are dropped for no reason.
    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    try
    {
        st.processor_thread = std::thread{processor_loop};
    } catch(const std::system_error& e)
    {
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log reader: processor thread creation failed ({}), reader not started",
            e.what());
        return false;
    }
    internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    try
    {
        st.thread = std::thread{reader_loop};
    } catch(const std::system_error& e)
    {
        // init_kfd_profiler() promises never to throw: the scope guard drops the fds
        // and leaves the dispatch-log unavailable, so dispatches use HSA timestamps.
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
        // The processor is already running; the reader that would have published
        // to it never will, so declare the producer finished or it would wait for
        // a completion that can never arrive.
        st.stop.store(true, std::memory_order_release);
        st.reader_done.store(true, std::memory_order_release);
        if(st.processor_thread.joinable()) st.processor_thread.join();
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log reader: thread creation failed ({}), reader not started", e.what());
        return false;
    }
    internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    st.running.store(true, std::memory_order_release);
    ROCP_INFO << "KFD dispatch-log reader: started";
    return true;
}

void
stop_kfd_reader()
{
    stop_reader();
}

bool
kfd_reader_state_constructed()
{
    // Non-constructing peek (::get(), not state()'s ::construct()).
    return common::static_object<reader_state>::get() != nullptr;
}

bool
reader_unavailable()
{
    // Non-constructing: an unconstructed reader was never armed, so it is not
    // "unavailable" in the sense N1 cares about (nothing to quiesce).
    auto* _st = common::static_object<reader_state>::get();
    return _st != nullptr && _st->reader_unavailable.load(std::memory_order_acquire);
}

bool
wait_for_reader_quiesce(uint64_t timeout_ns)
{
    auto& st = state();
    // Liveness only: the reader never started -> no records exist and
    // none will, so this is an immediate no-op success. NOT any_session_ready --
    // the fatal poll path leaves records unread, and that case must reach the
    // deadline and abandon, not falsely succeed.
    if(!st.running.load(std::memory_order_acquire)) return true;

    const uint64_t _deadline = common::timestamp_ns() + timeout_ns;

    // Nudge before waiting: with the 10 ms poll timeout the reader could otherwise
    // sit idle until it fires, delaying finalization by up to a full interval.
    nudge_reader();

    // STAGE 1 (reader): drain_epoch advances by 2 -- advanced only when every live
    // session's overflow is empty, so +2 proves every pre-request record was
    // copied AND published into the pipe. Two advances: the pass in flight when we
    // asked may have already read past our records, so the second is the first
    // provably complete one.
    const uint64_t _start = st.drain_epoch.load(std::memory_order_acquire);
    while(st.drain_epoch.load(std::memory_order_acquire) < _start + 2)
    {
        // Re-check liveness, not just on entry: stop_reader() clears running only
        // after joining the reader and processor, so a reader that stops mid-wait
        // has already drained.
        if(!st.running.load(std::memory_order_acquire)) return true;
        if(common::timestamp_ns() >= _deadline)
        {
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log fence: Stage 1 made no qualifying progress (drain_epoch delta "
                "{}){}",
                st.drain_epoch.load(std::memory_order_acquire) - _start,
                st.reader_unavailable.load(std::memory_order_acquire) ? ", reader unavailable"
                                                                      : "");
            return false;
        }
        nudge_reader();
        std::this_thread::sleep_for(std::chrono::microseconds{200});
    }

    // STAGE 2 (processor): poll pipe.empty() until true -- every batch published
    // before the request has been popped and fully processed. The
    // pipe's acquire loads make this safe from this thread; the processor is not
    // blocked on a wake, so nothing is nudged.
    while(!st.pipe.empty())
    {
        if(!st.running.load(std::memory_order_acquire)) return true;
        if(common::timestamp_ns() >= _deadline)
        {
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log fence: Stage 2 did not reach an empty pipe{}",
                st.reader_unavailable.load(std::memory_order_acquire) ? ", reader unavailable"
                                                                      : "");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds{200});
    }
    return true;
}

void
nudge_reader()
{
    auto& st = state();
    if(!st.running.load(std::memory_order_acquire)) return;
    const int _wake = st.wake_fd.load(std::memory_order_acquire);
    if(_wake < 0) return;
    uint64_t one = 1;
    // The eventfd is process-lifetime and never closed (§3.8), so the number cannot
    // be recycled between this load and write. Best-effort: a full counter (EAGAIN)
    // just means a wake is already pending, which is all we want.
    [[maybe_unused]] auto _rc = ::write(_wake, &one, sizeof(one));
}

namespace
{
// Returns the armed session for this GPU, or nullptr. Read without a lock:
// session_count and each `ready` flag are release-stored after the session is
// fully built, so an acquirer sees either a complete session or none.
dlog_session*
find_session(reader_state& st, uint32_t gpu_id)
{
    const size_t _n = st.session_count.load(std::memory_order_acquire);
    for(size_t i = 0; i < _n; ++i)
        if(st.sessions[i].gpu_id == gpu_id) return &st.sessions[i];
    return nullptr;
}

// latch_retryable: at first dispatch the device is certainly usable, so any
// failure is permanent and latching stops every later dispatch repeating the
// OPEN_STREAM ioctl. At init the device may just not be ready, so a retryable
// failure must not disable the feature process-wide.
bool
establish_session(uint32_t gpu_id, bool latch_retryable)
{
    // Finalization in progress: refuse to (re)arm the reader. A dispatch racing the
    // teardown window must not re-establish a session after stop_reader() ran, so it
    // cannot re-arm a reader that finalize is tearing down.
    if(registration::get_fini_status() != 0) return false;

    // Capability, not liveness: the reader may not be running yet, and starting it
    // is this function's job. disable_kfd_dispatch_log() clears this too, so a
    // forked child never starts one.
    if(!kfd_dispatch_log_supported() || !gpu_supports_dispatch_log(gpu_id)) return false;

    auto& st = state();
    // A dead or stopping reader disables every GPU at once.
    if(st.reader_unavailable.load(std::memory_order_acquire)) return false;

    if(auto* _existing = find_session(st, gpu_id))
        return _existing->ready.load(std::memory_order_acquire);

    auto lk = std::lock_guard<std::mutex>{st.setup_mu};
    // Re-check finalization UNDER setup_mu: the pre-lock get_fini_status() check
    // above can pass and finalization then begin before we take the lock. stop_reader()
    // sets reader_unavailable and tears down sessions while holding this same lock, so
    // re-checking both here closes the window where we would setup_session() into an
    // array stop_reader() is concurrently clearing.
    if(registration::get_fini_status() != 0) return false;
    // Re-check under the lock: another thread may have armed this GPU meanwhile.
    if(auto* _existing = find_session(st, gpu_id))
        return _existing->ready.load(std::memory_order_relaxed);
    if(st.reader_unavailable.load(std::memory_order_relaxed)) return false;
    // Deferred start: the reader exists only once something actually asks for a
    // session, so a process with no kernel-dispatch consumer creates no SDK
    // thread. setup_mu makes this the one serialized start point; start_kfd_reader
    // is a no-op once running.
    if(!start_kfd_reader()) return false;
    note_kfd_reader_started();

    const size_t _slot = st.session_count.load(std::memory_order_relaxed);
    if(_slot >= kMaxSessions)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: gpu_id={} not armed, already holding {} sessions", gpu_id, _slot);
        return false;
    }

    auto& _s         = st.sessions[_slot];
    bool  _permanent = false;
    if(!setup_session(st.kfd_fd, gpu_id, &_s, &_permanent))
    {
        if(!_permanent && !latch_retryable)
        {
            // Too early, most likely: leave the door open for the first dispatch
            // on THIS GPU to try again rather than disabling it.
            ROCP_INFO << fmt::format(
                "KFD dispatch-log: gpu_id={} session not established at configuration; will retry "
                "on the first dispatch",
                gpu_id);
            return false;
        }

        // THIS GPU only: the slot is claimed but never published ready, so it is
        // skipped by the poll/copy paths; another GPU's session is unaffected.
        _s.gpu_id = gpu_id;
        st.session_count.store(_slot + 1, std::memory_order_release);
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: gpu_id={} supports dispatch-log but session setup failed; that GPU "
            "now uses HSA timestamps",
            gpu_id);
        return false;
    }

    // Publish: `ready` last, so an acquiring reader never sees a half-built
    // session, and session_count after it so the copier's walk finds it complete.
    _s.ready.store(true, std::memory_order_release);
    st.session_count.store(_slot + 1, std::memory_order_release);
    st.any_session_ready.store(true, std::memory_order_release);
    // A new stream fd exists, so the reader must add it to its poll set.
    st.pollset_generation.fetch_add(1, std::memory_order_release);

    // Break the reader out of its poll so the new session's stream fd joins the
    // poll set and the first dispatches are drained immediately.
    nudge_reader();
    return true;
}
}  // namespace

bool
ensure_reader_session(uint32_t gpu_id)
{
    // First-dispatch path: the device is usable by now, so a failure here is
    // permanent for this GPU and latches, exactly as before.
    return establish_session(gpu_id, /*latch_retryable=*/true);
}

bool
arm_reader_session_early(uint32_t gpu_id)
{
    // Arm before any kernel can dispatch, so the ring is live under the first one.
    return establish_session(gpu_id, /*latch_retryable=*/false);
}
}  // namespace kfd
}  // namespace rocprofiler
