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
#include "lib/rocprofiler-sdk/kfd/record_processor.hpp"
#include "lib/rocprofiler-sdk/kfd/record_stream.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/kfd/stream_geometry.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

// Active (profiler ABI v6, stream ABI v4) dispatch-log UAPI. Must be the ONLY kfd
// ioctl header in this TU (it conflicts with lib/rocprofiler-sdk/details/kfd_ioctl.h).
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <mutex>
#include <new>
#include <optional>
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
constexpr uint64_t kLoggingIntervalNs = 1'000'000'000ull;  // 1 s

// A retained START may belong to a still-running long kernel, so the eviction
// age MUST exceed the longest real dispatch or a live kernel's START is aged out
// before its EOP arrives (emitting it start-unknown). Default to 30 min, tunable
// in seconds via ROCPROFILER_KFD_DISPATCH_LOG_START_MAX_AGE_SEC. Eviction still
// bounds pending_starts growth for genuinely orphaned STARTs (queue died
// mid-dispatch); the age is the slow backstop, pending_starts_cap() below the
// hard bound.
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
// sustained EOP loss -- the same role and sizing as the hub's cap (the map is
// per-GPU, and a workload's genuine in-flight peak is hundreds of thousands of
// dispatches). Read once, forced at reader start so no worker thread races a
// concurrent setenv.
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
struct reader_state;
struct processor_state
{
    std::unordered_map<uint32_t, pair_state> by_gpu = {};

    // Per-stream decode backlog, indexed by session slot like
    // reader_state::sessions.
    std::array<stream_backlog, kMaxSessions> backlogs = {};

    // Drained EOPs awaiting one batched hub resolution; reused across units.
    std::vector<signal_less_hub_t::kernel_end> ends = {};

    // Hub keys of the EOPs every GPU is holding back for lookahead. GC must not
    // reap their entries: the next batch retires them.
    signal_less_hub_t::key_set held_eop_keys() const
    {
        auto keys = signal_less_hub_t::key_set{};
        for(const auto& [gpu, pairing] : by_gpu)
        {
            for(const auto& held : pairing.deferred_records)
            {
                const auto& rec = held.rec;
                if(rec.record_type != kRecEop || rec.doorbell_off == 0) continue;
                keys.insert(correlation_key{
                    doorbell_off_to_page_slot(rec.doorbell_off), rec.dispatch_id, gpu});
            }
        }
        return keys;
    }

    uint64_t oldest_deferred_epoch() const
    {
        uint64_t oldest = UINT64_MAX;
        for(const auto& [gpu, pairing] : by_gpu)
            oldest = std::min(oldest, pairing.deferred_epoch);
        return oldest;
    }

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
    // Reader -> processor SPSC chunk chain: the reader appends each pass
    // and publishes at a marker, the processor copies out and returns chunks.
    // Preallocated at arm time, reset after the joins, abandoned in a fork child.
    record_stream    stream    = {};
    reader_alloc_log alloc_log = {};
    // A non-empty pass was published, so the next pass publishes even when it
    // copies nothing: that empty segment is the processor's lookahead marker.
    bool pairing_flush_pending = false;

    // Lap-exposure telemetry (ABI v4 cannot detect a lap, only its approach):
    // the widest interval between two drains of this ring, and whether the
    // near-lap warning has fired.
    uint64_t last_copy_ns    = 0;
    uint64_t max_copy_gap_ns = 0;
    bool     lap_risk_warned = false;

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

    // Once-per-episode latch for the notify-count read() fault warning, cleared by
    // the next successful read. Separate from pollerr_active so neither fault
    // suppresses or re-arms the other's warning. Reader thread only.
    bool notify_read_err_logged = false;
};

struct reader_state
{
    std::thread       thread = {};
    std::atomic<bool> stop   = {false};
    // Set by the reader once it has published everything it will ever publish.
    // `stop` only asks it to wind down; the reader still publishes a final pass
    // afterwards, so the processor must key its exit off this instead.
    std::atomic<bool> reader_done = {false};
    // True while the reader is inside a drain pass. The processor's
    // closed-window GC loads this FIRST (seq_cst) in its gate, so a publish
    // visible to the gate is never left behind by an in-flight pass. Reader is
    // the sole writer.
    std::atomic<bool>     reader_pass_open         = {false};
    std::atomic<uint64_t> processor_deferred_epoch = {UINT64_MAX};
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

    std::thread processor_thread = {};

    reader_state() = default;
    ~reader_state();

    reader_state(const reader_state&) = delete;
    reader_state& operator=(const reader_state&) = delete;
};

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
                                                                s->info.wptr_offset},
                                                buf_bytes,
                                                page_size());
    if(!_geom.ok)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: stream geometry mismatch ({}) (num_regions={} "
            "region_record_count={} buffer_size={} requested={} mmap_size={} records_offset={} "
            "wptr_offset={})",
            geometry_reason_name(_geom.reason),
            s->info.num_regions,
            s->info.region_record_count,
            s->info.buffer_size,
            buf_bytes,
            s->info.mmap_size,
            s->info.records_offset,
            s->info.wptr_offset);
        if(permanent) *permanent = true;  // ABI disagreement: retrying cannot help
        return false;
    }
    s->smap_len = static_cast<size_t>(_geom.mmap_len);

    // ABI v4: the BO is mapped READ-ONLY (the kernel rejects PROT_WRITE). The read
    // cursor lives in the session's local ring_cursors, never in the mapping.
    s->smap = mmap(nullptr, s->smap_len, PROT_READ, MAP_SHARED, s->stream_fd, 0);
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

    // Prime the read cursor SYNCHRONOUSLY here, before the session is published as
    // ready, so any backlog present at attach is discarded on the setup thread and
    // never by the reader thread's first lazy pass. A fast workload that finishes
    // before the reader's first drain would otherwise have all its records counted
    // as backlog and dropped. Records produced AFTER this snapshot drain normally.
    {
        const auto* wptr_base = static_cast<const uint8_t*>(s->smap) + s->info.wptr_offset;
        for(uint32_t r = 0; r < s->info.num_regions; ++r)
            s->cursors.rptr[r] = load_wptr_low32(wptr_base, r);
        s->cursors.rptr_init = true;
    }

    // Arm-time preallocation: the current chunk plus the pass + gap reserves,
    // every page pre-touched HERE so the reader's hot path never faults. A
    // failure takes the existing setup-failure path: the scope destructor
    // above runs teardown_session(), whose stream.reset() frees the partial
    // preallocation, and this GPU falls back to HSA timestamps.
    if(!s->stream.preallocate(dlog_spare_target(s->info.buffer_size, s->info.num_regions)))
        return false;

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
    // starts from a clean state. Leaving cursors set would carry rptr_init=true and
    // a stale local rptr[] into the new stream, and copy_pipes() would then skip the
    // first-drain prime (rptr:=wptr) and mis-count against a stale cursor.
    s->cursors = {};
    // The reader publishes everything it copied before reader_done, so
    // published != processed here means copied records were dropped before the
    // processor saw them.
    ROCP_CI_LOG_IF(WARNING, s->stream.published() != s->stream.processed()) << fmt::format(
        "KFD dispatch-log: tearing down gpu_id={} with {} published record(s) not yet "
        "processed; records copied from the ring were dropped before the processor saw them",
        s->gpu_id,
        s->stream.published() - s->stream.processed());
    // Never silent: a reader allocation means a sizing defect or a slow consumer.
    const auto _allocs     = s->alloc_log.reader_allocs.load(std::memory_order_relaxed);
    const auto _min_spares = s->alloc_log.min_spares.load(std::memory_order_relaxed);
    ROCP_WARNING_IF(_allocs > 0) << fmt::format(
        "KFD dispatch-log (gpu_id={}): reader allocated {} chunk(s); min spares {}",
        s->gpu_id,
        _allocs,
        _min_spares);
    // Both workers are joined by the time this runs on the stop path, and the
    // session was never published on the setup-failure path, so no worker can
    // be touching the stream here.
    s->stream.reset();
    s->alloc_log.reader_allocs.store(0, std::memory_order_relaxed);
    s->alloc_log.min_spares.store(SIZE_MAX, std::memory_order_relaxed);
    s->pairing_flush_pending  = false;
    s->lap_risk_warned        = false;
    s->last_copy_ns           = 0;
    s->max_copy_gap_ns        = 0;
    s->quarantined            = false;
    s->pollerr_active         = false;
    s->notify_read_err_logged = false;

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

// Reader thread only. ABI v4 cannot detect a producer lap, only its approach:
// track the widest interval between drains of this ring, and warn once when a
// region was found more than half full -- the reader came within a factor of
// two of losing a full region of records.
void
note_lap_exposure(dlog_session& s, uint64_t now_ns)
{
    if(s.last_copy_ns != 0 && now_ns > s.last_copy_ns)
        s.max_copy_gap_ns = std::max(s.max_copy_gap_ns, now_ns - s.last_copy_ns);
    s.last_copy_ns = now_ns;

    const uint64_t _capacity = s.info.region_record_count;
    if(s.lap_risk_warned || _capacity == 0 || s.cursors.max_unread <= _capacity / 2) return;
    s.lap_risk_warned = true;
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log (gpu_id={}): a ring region held {} of {} records when drained. A "
        "longer reader delay would overwrite records undetectably; ring is {} KB (max {} KB, "
        "ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB).",
        s.gpu_id,
        s.cursors.max_unread,
        _capacity,
        _capacity * s.info.num_regions * kFwRecBytes / 1024,
        kDlogMaxRingBytes / 1024);
}

// STAGE 1, reader thread: copy each live ring into its session's chunk stream
// and get out. This is the ONLY code touching the volatile mapping, and every
// microsecond here widens the window for the firmware to lap us -- so no lock,
// no pairing. The pass body (snapshot, marker, one publish) is dlog_pass;
// the ring is drained on EVERY pass, and a pass that needs more chunks than are
// spare allocates them on the reader (counted in alloc_log), where a failure
// throws out of reader_loop and terminates, exactly as base's bad_alloc did.
uint64_t
copy_records(reader_state& st)
{
    uint64_t     _total = 0;
    const size_t _n     = st.session_count.load(std::memory_order_acquire);

    // Seq_cst, loaded FIRST by the processor's GC gate: true before the
    // first snapshot of the pass, false after its last publish.
    st.reader_pass_open.store(true, std::memory_order_seq_cst);

    // The rings are independent, so one GPU lapping cannot stall another. A
    // quarantined session has already been drained past its terminal state and
    // its mapping may be inert, so it is skipped here as well as in the poll set.
    for(size_t i = 0; i < _n; ++i)
    {
        auto& _s = st.sessions[i];
        if(!_s.ready.load(std::memory_order_acquire) || _s.quarantined) continue;

        auto* base      = static_cast<uint8_t*>(_s.smap);
        auto* recs      = base + _s.info.records_offset;
        auto* wptr_base = base + _s.info.wptr_offset;

        const uint64_t _now = common::timestamp_ns();
        _total +=
            dlog_pass(recs, wptr_base, _s, st.drain_epoch.load(std::memory_order_relaxed), _now);
        note_lap_exposure(_s, _now);
    }
    st.reader_pass_open.store(false, std::memory_order_seq_cst);
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
    const auto* base      = static_cast<const uint8_t*>(s.smap);
    const auto* wptr_base = base + s.info.wptr_offset;

    // ABI v4: the read cursor is consumer-private (s.cursors), not in the BO. A
    // region has pending records when the firmware wptr differs from our local
    // rptr. Before the first drain rptr_init is false and rptr[] is 0; treat that
    // as "not yet primed", so a fresh (zeroed wptr) stream reads as not pending.
    for(uint32_t r = 0; r < s.info.num_regions; ++r)
    {
        const uint32_t w = load_wptr_low32(wptr_base, r);
        const uint32_t c = s.cursors.rptr[r];
        if(w != c) return true;
    }
    return false;
}

// KFD_DLOG_STREAM_OP_STATUS is DIAGNOSTICS ONLY: the kernel's counters are logged
// but never gate a data decision (wptr, the firmware's wrapping per-pipe cursor, is
// the sole drain target). ABI v4 cannot detect overruns -- an overwritten
// record is indistinguishable from a fresh one -- so the status word is used only to
// describe a terminal stream when quarantining it, never to decide whether to drain.
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

// Read the STREAM_OP_STATUS flags word (FATAL detection). Returns 0 on a dead
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
    // The wake eventfd is intentionally left open: it is created once per
    // process and never closed, so nudge_reader()'s load/write is always safe and a
    // later start reuses it. Any residual count is harmlessly drained by the next
    // reader's first control-fd read.
    st.running.store(false, std::memory_order_release);
    ROCP_INFO << "KFD dispatch-log reader: stopped";
}

reader_state::~reader_state()
{
    // In a fork child the reader thread does not exist and setup_mu / the
    // session containers may be mid-mutation by a vanished thread. Skip
    // stop_reader()'s join+teardown -- the same treatment the fork gives
    // TaskGroup (skip the dangerous thread work; the child abandoned this state
    // via disable_reader_in_child). Value members destruct as they do there.
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
    // Drop the reference WITHOUT closing: the persistent wake eventfd's number
    // must never be recycled under the parent's nudge_reader; the child sets it -1 so
    // its own nudge is inert and a later child reader creates a fresh one.
    st.wake_fd.store(-1, std::memory_order_relaxed);
    // Close the inherited fds/maps. After fork the child has its own fd table,
    // so close()/munmap() here release the child's copies WITHOUT touching the
    // parent's descriptors. Both are async-signal-safe, which the atfork child
    // handler requires.
    if(st.kfd_fd >= 0) ::close(st.kfd_fd);
    st.kfd_fd = -1;
    // Abandon EVERY fixed-array slot's chunk stream, not just the published
    // ones: establish_session() builds slot session_count (preallocate / rollback
    // unmaps) BEFORE publishing the increment, so a fork mid-setup leaves an
    // unpublished slot with live stream state. abandon_in_child() only resets
    // indices -- no munmap, no chunk access: the mappings are inherited and
    // intentionally leak (CoW; the child usually execs).
    for(auto& _s : st.sessions)
        _s.stream.abandon_in_child();
    const size_t _n = st.session_count.load(std::memory_order_relaxed);
    for(size_t i = 0; i < _n && i < kMaxSessions; ++i)
    {
        st.sessions[i].ready.store(false, std::memory_order_relaxed);
        if(st.sessions[i].smap != MAP_FAILED) munmap(st.sessions[i].smap, st.sessions[i].smap_len);
        st.sessions[i].smap = MAP_FAILED;
        if(st.sessions[i].stream_fd >= 0) ::close(st.sessions[i].stream_fd);
        st.sessions[i].stream_fd = -1;
        // Same reuse hazard as teardown_session(): session_count is zeroed here,
        // so any later re-arm in the child reuses these slots. Reset the drain
        // cursors so a reused slot cannot inherit a stale rptr and drain nothing.
        st.sessions[i].cursors               = {};
        st.sessions[i].pairing_flush_pending = false;
        st.sessions[i].lap_risk_warned       = false;
        st.sessions[i].last_copy_ns          = 0;
        st.sessions[i].max_copy_gap_ns       = 0;
    }
    st.session_count.store(0, std::memory_order_relaxed);
}

// Sole consumer of the streams; owns all pairing state, so it needs no lock for
// it. Processes one marker-delimited segment at a time (one base batch), with
// service points between units, in the idle loop, and before each hub
// acquisition.
void
processor_loop()
{
    auto& st   = state();
    auto  proc = processor_state{};
    st.processor_deferred_epoch.store(UINT64_MAX, std::memory_order_release);

    uint64_t last_gc_ns = common::timestamp_ns();
    size_t   _rr        = 0;

    // Drained EOPs are resolved against the hub in runs of up to
    // kCompletionRunLength under ONE hub lock each, and each run's proven
    // completions go to the task group as one task: per-kernel lock and task
    // traffic would contend with every enqueuing thread. The buffer is reused
    // across units.
    constexpr size_t kCompletionRunLength = 256;
    auto&            _ends                = proc.ends;

    // The service point: a copy-out of every armed stream's published runs.
    auto _service = [&]() {
        const size_t _n = st.session_count.load(std::memory_order_acquire);
        copy_out_all(st.sessions, proc.backlogs, _n);
    };

    // The hub resolution for one flushed run. The service runs BEFORE the hub
    // mutex is taken, so the sort of a maximal segment and this wait are
    // always separated.
    auto _flush_ends = [&]() {
        if(_ends.empty()) return;
        _service();
        auto _completed = signal_less_batch_t{};
        _completed.reserve(_ends.size());
        const size_t _proved = signal_less_hub().record_kernel_ends(_ends, _completed);
        if(_proved != 0) note_signal_less(signal_less_counter::eop_proven, _proved);
        if(_proved != _ends.size())
            note_signal_less(signal_less_counter::eop_unmatched, _ends.size() - _proved);
        _ends.clear();
        hand_off_proven(std::move(_completed));
    };

    // One segment = one base batch. gpu_id is stamped from the ring the
    // record came from: a record can only ever match a dispatch enqueued on the
    // same GPU. No generation. Signal-less: this EOP IS the completion event.
    auto _process_one = [&](size_t _slot, stream_segment& _seg) {
        _ends.clear();
        const uint32_t _gpu = st.sessions[_slot].gpu_id;
        process_unit(
            st.sessions[_slot].stream,
            proc.backlogs[_slot],
            proc.for_gpu(_gpu),
            _seg,
            start_max_age_ns(),
            [&](const drained_record& rec) {
                _ends.push_back(signal_less_hub_t::kernel_end{
                    correlation_key{
                        doorbell_off_to_page_slot(rec.doorbell_off), rec.dispatch_id, _gpu},
                    rec.start_known ? std::optional<uint64_t>{rec.start_ticks} : std::nullopt,
                    rec.end_ticks});
                if(_ends.size() == kCompletionRunLength) _flush_ends();
            },
            _flush_ends,
            [&]() {
                // Published after the final flush above and before the stream's
                // processed position advances: a fence that observes processed
                // >= end_pos also observes which lookahead records are still
                // held.
                st.processor_deferred_epoch.store(proc.oldest_deferred_epoch(),
                                                  std::memory_order_release);
            });
    };

    while(true)
    {
        // GC closed-window entries on the periodic tick, but ONLY when the
        // reader is between passes and every stream is fully copied and
        // decoded: an EOP that would resolve a closed window may be sitting in
        // a backlog behind us, and GC'ing its window first would expire it as a
        // leak and then drop the EOP as start-unknown. EOPs the processor
        // holds for lookahead are excluded by key (held_eop_keys), not by
        // suspending GC: under continuous traffic something is always held,
        // and a held EOP is retired by the very next segment.
        const uint64_t _gc_now = common::timestamp_ns();
        // Closed-window GC stays gated (gc_gate_open_pass) and its timer
        // advances only when GC actually runs, so it fires at the first
        // open-gate moment past the interval even under continuous traffic.
        if(gc_gate_open_pass(st.reader_pass_open, st.sessions, proc.backlogs, st.session_count) &&
           _gc_now - last_gc_ns >= kProcessorEvictIntervalNs)
        {
            last_gc_ns       = _gc_now;
            const auto _keep = proc.held_eop_keys();
            auto       _gc   = signal_less_hub().gc_closed_windows(steady_now_ns(), &_keep);
            if(_gc.second.dispatches > 0) note_signal_less_losses();
        }

        const size_t _n = st.session_count.load(std::memory_order_acquire);
        auto         _u = next_unit(proc.backlogs, _n, _rr);
        if(!_u.found)
        {
            if(st.reader_done.load(std::memory_order_acquire))
            {
                // Acquire pairs with the reader's release: everything it
                // published is visible now, but the closed deques may not hold
                // it yet -- the exit drain below copies out and re-checks.
                break;
            }
            // Nothing to do. Sleeping here costs nothing on the ring: the
            // reader keeps copying regardless of what this thread is doing.
            _service();
            std::this_thread::sleep_for(std::chrono::microseconds{200});
            continue;
        }

        _process_one(_u.slot, _u.seg);
        _service();  // between units

        // Retained-start eviction stays segment-driven (inside process_unit,
        // watermarked by the segment's seen_at_ns); the 1 s tick above only
        // GCs closed windows.
    }

    // Drain whatever the reader published on its way out, then process every
    // closed segment.
    {
        const size_t _n = st.session_count.load(std::memory_order_acquire);
        drain_for_exit(st.sessions, proc.backlogs, _n, _process_one);
    }

    // Base per-GPU final flush: an empty segment with a fresh now_ns and epoch 0
    // releases whatever each GPU still holds for lookahead. One session per
    // gpu_id, so walking the slots covers every GPU.
    for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
    {
        const auto _itr = proc.by_gpu.find(st.sessions[i].gpu_id);
        if(_itr == proc.by_gpu.end() || _itr->second.deferred_records.empty()) continue;
        stream_segment _final = {};
        _final.seen_at_ns     = common::timestamp_ns();
        _final.end_pos        = st.sessions[i].stream.taken();
        _process_one(i, _final);
    }

    for(const auto& _gpu_itr : proc.by_gpu)
    {
        const auto& _p = _gpu_itr.second;
        if(_p.eops_seen == 0) continue;

        ROCP_WARNING << fmt::format(
            "KFD dispatch-log pairing census (gpu_id={}): {} START record(s) drained, {} EOP "
            "record(s) drained, {} EOP(s) unmatched, {} START(s) overwritten on a live key, {} "
            "equal-tick EOP(s) dropped, {} stale EOP(s) dropped, {} START(s) evicted stale, {} "
            "START(s) evicted by the size cap, {} START(s) still retained at exit",
            _gpu_itr.first,
            _p.starts_seen,
            _p.eops_seen,
            _p.unmatched_eops,
            _p.starts_overwritten,
            _p.equal_tick_drops,
            _p.stale_eop_drops,
            _p.starts_evicted,
            _p.starts_cap_evicted,
            _p.pending_starts.size());
        ROCP_INFO_IF(_p.starts_out_of_order > 0) << fmt::format(
            "KFD dispatch-log (gpu_id={}): {} late or duplicate START record(s) ignored",
            _gpu_itr.first,
            _p.starts_out_of_order);
    }
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
        // Records are published at the end of the pass that copied them, so the
        // epoch advances after EVERY pass: the fence's +2 proves a whole
        // pass ran after the request.
        st.drain_epoch.fetch_add(1, std::memory_order_release);
        if(_copied == 0) break;
    }
    return seen;
}

// Consume the kernel's per-stream notify count after a POLLIN wake. The kernel
// reports EPOLLIN level-triggered while notify_count != 0 and only read() clears
// it, so a wake that is not answered by a read() re-reports EPOLLIN forever and
// spins the reader. read(fd,&u64,8) returns 8 and zeroes the count, or -1/EAGAIN
// when it is already 0 (a benign race: another consumer path, or our own prior
// read, drained it); the count VALUE is discarded because wptr, not the count, is
// the drain target. Any other errno is a real fd fault: log it once (matching the
// poll-error path) so it is not swallowed, and let the terminal-revents scan below
// quarantine the stream on the HUP/ERR the same fault will raise.
// `notify_read_err_logged` latches the warning so a persistently faulting fd does
// not spam the log every pass.
void
read_stream_notify_count(dlog_session& s)
{
    uint64_t _count = 0;
    for(;;)
    {
        const ssize_t _n        = ::read(s.stream_fd, &_count, sizeof(_count));
        const bool    _consumed = _n == static_cast<ssize_t>(sizeof(_count));
        if(_consumed || (_n < 0 && errno == EAGAIN))  // EAGAIN: already 0
        {
            s.notify_read_err_logged = false;
            return;
        }
        if(_n < 0 && errno == EINTR) continue;
        // A short read or an unexpected errno is a real fault: warn once per episode.
        if(!s.notify_read_err_logged)
        {
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log: gpu_id={} notify-count read failed (rc={} errno={}); the "
                "terminal-revents scan will quarantine the stream if the fault persists",
                s.gpu_id,
                _n,
                _n < 0 ? errno : 0);
            s.notify_read_err_logged = true;
        }
        return;
    }
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

        // Consume each woken stream's kernel notify count BEFORE draining. The
        // kernel's EPOLLIN is level-triggered while notify_count != 0 and only the
        // read() clears it, so skipping this leaves poll() ready forever and spins
        // the reader; doing it before the drain re-arms readiness for a notify that
        // lands mid-drain. HUP/ERR-only wakes carry no POLLIN and are left to the
        // terminal-revents scan below -- reading them would only churn EAGAIN or,
        // on a torn-down fd, noise. A backstop-tripped pass polls only the control
        // fd, so the stream revents are zero and this loop is a no-op there.
        for(size_t k = kFirstStreamPollSlot; k < fds.size(); ++k)
        {
            if(!stream_wake_wants_notify_read(fds[k].revents)) continue;
            read_stream_notify_count(st.sessions[slot_of[k - kFirstStreamPollSlot]]);
        }

        // A wake or timeout both trigger a full scan: the drain always targets the
        // observed wptr, so a timeout (the sparse-tail and lost-interrupt watchdog)
        // picks up records below one notify interval exactly like an interrupt would.
        // Hitting the pass cap with records still pending needs no self-nudge: any
        // notify not yet consumed keeps the kernel's EPOLLIN asserted, so poll() re-
        // triggers on its own -- the backstop below bounds the case where it re-
        // triggers (count still set) but the drain copies nothing.

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
        else
        {
            // No session is armed, so the ready-set is trivially drained: advance
            // drain_epoch every pass so a fence blocked on drain progress is not made
            // to burn its whole timeout waiting for a +2 that drain_sessions_bounded
            // (only reached when a session IS ready) would otherwise never post.
            // Every publish ends at its pass's marker, so the epoch may advance
            // unconditionally.
            st.drain_epoch.fetch_add(1, std::memory_order_release);
        }

        // Terminal streams. POLLHUP/POLLNVAL are terminal: the stream will never
        // produce again, so it is drained first (above), then -- only once it is
        // level-dry -- logged and quarantined out of the poll set via a generation
        // bump, so no final record is dropped and a sticky HUP cannot spin poll().
        // POLLERR alone is a recoverable reset: the stream node survives, so it is
        // logged but NOT quarantined. A HUP still pending records defers to the next
        // pass -- the fd stays readable, so poll() re-triggers.
        bool _quarantined_any = false;
        for(size_t k = kFirstStreamPollSlot; k < fds.size(); ++k)
        {
            const terminal_action _act = classify_terminal_revents(fds[k].revents);
            auto&                 _s   = st.sessions[slot_of[k - kFirstStreamPollSlot]];
            if(_s.quarantined) continue;

            // POLLERR alone is usually a recoverable reset, but a GPU reset
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
                // FATAL, so this converges) so no already-published record is
                // lost.
                if(session_has_pending(_s) || _s.pairing_flush_pending) continue;
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
                // The ring is dry, but the records already published into the
                // stream are not: this session's final EOP, and the EOPs of every
                // healthy GPU queued behind it, are still waiting on the processor.
                // Draining the hub first would ledger their entries and leave those
                // records unmatched. So latch admission closed, let the processor
                // consume what is published, and only then ledger the residual,
                // bounded by the close budget so one wedged processor cannot stall
                // the ring for the other GPUs.
                //
                // "Consumed" is everything copied through drain pass `_through`:
                // the processor must copy it out, pair it up to the captured
                // target, and release any of it held for lookahead. Healthy GPUs
                // keep producing meanwhile, but that newer traffic lies past the
                // target and cannot stretch the wait.
                signal_less_disable_latch().store(true, std::memory_order_release);
                const uint64_t _wait_deadline = common::timestamp_ns() + close_drain_budget_ns();
                const uint64_t _through       = st.drain_epoch.load(std::memory_order_acquire);
                // Stage-2 target over streams, captured once after the first
                // in-wait copy_records.
                std::array<uint64_t, kMaxSessions> _target       = {};
                size_t                             _target_count = 0;
                bool                               _captured     = false;
                while(common::timestamp_ns() < _wait_deadline &&
                      !st.stop.load(std::memory_order_acquire))
                {
                    copy_records(st);
                    if(!_captured)
                    {
                        _target_count = st.session_count.load(std::memory_order_acquire);
                        for(size_t i = 0; i < _target_count; ++i)
                            if(st.sessions[i].stream.armed())
                                _target[i] = st.sessions[i].stream.published();
                        _captured = true;
                    }
                    if(processor_drained_through(st.sessions,
                                                 _target_count,
                                                 _target.data(),
                                                 st.processor_deferred_epoch,
                                                 _through))
                        break;
                    std::this_thread::sleep_for(std::chrono::microseconds{200});
                }
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
            // only inspects the ring; pairing_flush_pending covers a pass that
            // already published this session's lookahead marker, so quarantine
            // waits until the final IDLE_FLUSH is published too: the
            // processor then releases the held set when it pairs that segment.
            if(session_has_pending(_s) || _s.pairing_flush_pending) continue;

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

    // Final bounded drain includes the lookahead marker for a sparse tail.
    if(st.any_session_ready.load(std::memory_order_acquire))
    {
        total_seen += drain_sessions_bounded(st);
        for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
        {
            const auto& _s = st.sessions[i];
            log_stream_status(_s);
            // Lap exposure for the run: how full a region got and how long the
            // reader went between drains. A peak near region_records means the run
            // was close to an undetectable overrun even if nothing was lost.
            ROCP_INFO_IF(_s.last_copy_ns != 0) << fmt::format(
                "KFD dispatch-log (gpu_id={}): ring drain exposure: peak {} of {} records "
                "unread in one region, longest interval between drains {:.3f} ms",
                _s.gpu_id,
                _s.cursors.max_unread,
                _s.info.region_record_count,
                static_cast<double>(_s.max_copy_gap_ns) / 1e6);
        }
    }

    // Release: pairs with the processor's acquire so it cannot decide the producer
    // is finished before the final publishes above are visible. Every publish
    // already ends at its pass's marker, so there is nothing left to spill: the
    // processor drains the streams after it observes this.
    st.reader_done.store(true, std::memory_order_release);

    ROCP_INFO << fmt::format("KFD dispatch-log reader: loop exited, total pairs seen = {}",
                             total_seen);
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
        // The wake eventfd is deliberately NOT closed here: it lives for the
        // process lifetime so its number can never be recycled under nudge_reader().
    }};

    // Create the wake eventfd once per process and never close it. A closed
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

    // Force-construct the singletons the reader/processor hot paths touch here,
    // BEFORE the worker threads exist, so a first-use allocation cannot stall the
    // ring read and turn into an overrun. state() was already constructed at the
    // top of this function. Doing it from this one thread before the workers start
    // also means their init needs no cross-thread barrier -- construction strictly
    // happens-before either worker runs. (The downstream signal-less singletons
    // the processor reaches on a proven completion are constructed lazily, but
    // they run off the ring, so a first-use allocation there cannot cause an
    // overrun.)

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
    // "unavailable" in the sense of having something to quiesce.
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

    // STAGE 1 (reader): drain_epoch advances by 2 -- advanced after every whole
    // pass, and each pass publishes at a marker, so +2 proves every pre-request
    // record was published into its stream. Two advances: the pass in flight when
    // we asked may have already read past our records, so the second is the first
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

    // STAGE 2 (processor): every segment published by now has been copied out,
    // paired and released, and no record from a pre-fence drain pass is still
    // held for lookahead. Stage 1 advanced only between whole passes, so every
    // pre-request record was published before this snapshot. The target is fixed
    // here, so newer traffic -- even a stream that never drains -- cannot
    // prolong the fence.
    std::array<uint64_t, kMaxSessions> _target = {};
    const size_t                       _tcount = st.session_count.load(std::memory_order_acquire);
    for(size_t i = 0; i < _tcount; ++i)
        if(st.sessions[i].stream.armed()) _target[i] = st.sessions[i].stream.published();
    while(!processor_drained_through(
        st.sessions, _tcount, _target.data(), st.processor_deferred_epoch, _start + 2))
    {
        if(!st.running.load(std::memory_order_acquire)) return true;
        if(common::timestamp_ns() >= _deadline)
        {
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log fence: Stage 2 did not drain the streams and pending EOPs{}",
                st.reader_unavailable.load(std::memory_order_acquire) ? ", reader unavailable"
                                                                      : "");
            return false;
        }
        nudge_reader();
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
    // The eventfd is process-lifetime and never closed, so the number cannot
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
