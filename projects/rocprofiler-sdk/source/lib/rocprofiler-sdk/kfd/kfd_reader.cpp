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
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/record_pipe.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"

// Active (v5) dispatch-log profiler ABI. Must be the ONLY kfd ioctl header in
// this translation unit (it conflicts with lib/rocprofiler-sdk/details/kfd_ioctl.h).
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
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

// Upper bound on armed GPUs. One session per GPU that supports dispatch-log.
constexpr size_t kMaxSessions = 8;

// Aging cadences. A deposited result is normally taken within milliseconds; an
// unmatched start belongs to a dispatch whose eop never arrived.
constexpr uint64_t kEvictIntervalNs          = 1'000'000'000ull;  // 1 s
constexpr uint64_t kProcessorEvictIntervalNs = 1'000'000'000ull;  // 1 s
constexpr uint64_t kStartMaxAgeNs            = 5'000'000'000ull;  // 5 s

// Owned exclusively by the processor thread, so none of it needs a lock.
struct processor_state
{
    std::unordered_map<uint32_t, pair_state> by_gpu = {};

    pair_state& for_gpu(uint32_t gpu_id) { return by_gpu[gpu_id]; }
};

// Validated before any sizing math uses it.
uint64_t
ring_bytes()
{
    auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB");
    if(!_v) return kDlogMinRingBytes;

    uint64_t _want = dlog_ring_bytes_from_kb_str(*_v);
    if(_want == 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: ignoring invalid ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB='{}' "
            "(expected an integer 1-{}); using {} KB",
            *_v,
            kDlogMaxRingKb,
            kDlogMinRingBytes / 1024);
        return kDlogMinRingBytes;
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

static_assert(kFwRecBytes == KFD_DISPATCH_LOG_FW_RECORD_BYTES,
              "dlog_drain.hpp fw record size must match the UAPI");

size_t
page_size()
{
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<size_t>(p) : 4096u;
}
size_t
round_up_page(size_t x)
{
    size_t p = page_size();
    return (x + p - 1) & ~(p - 1);
}

// One dispatch-log data-ring session for a single GPU.
struct dlog_session
{
    uint32_t gpu_id       = 0;
    uint64_t buffer_va    = 0;  // GPU VA of the KFD allocation
    uint64_t alloc_handle = 0;  // KFD alloc handle (for unmap/free)
    size_t   alloc_size   = 0;
    int      stream_fd    = -1;
    void*    smap         = MAP_FAILED;
    size_t   smap_len     = 0;

    kfd_dlog_stream_info info = {};

    // Touched ONLY by the ring-copier thread.
    ring_cursors cursors = {};

    // Per-session, so one GPU failing to arm never disables another.
    std::atomic<bool> ready  = {false};
    std::atomic<bool> failed = {false};
};

struct reader_state
{
    std::thread       thread  = {};
    std::atomic<bool> stop    = {false};
    int               wake_fd = -1;
    bool              running = false;

    int kfd_fd = -1;

    // Fixed array, not a container, so the atfork child handler can walk it with
    // plain indexing -- no allocation, no iterator invalidation.
    std::array<dlog_session, kMaxSessions> sessions      = {};
    std::atomic<size_t>                    session_count = {0};

    // Set up on the app thread, torn down on the finalize thread.
    std::mutex setup_mu = {};
    // True once ANY session is armed: lets the reader pick its poll cadence
    // without walking the array on every iteration.
    std::atomic<bool> any_session_ready = {false};
    // Latched when the reader can never serve a session, as opposed to a single
    // GPU failing to arm.
    std::atomic<bool> reader_unavailable = {false};
    // Bumped once per completed drain pass. A sync point waits for it to advance
    // twice, which proves a whole pass ran after the request.
    std::atomic<uint64_t> drain_epoch = {0};

    // Copier is the sole producer, processor the sole consumer.
    record_pipe<kBatchPipeDepth> pipe             = {};
    std::thread                  processor_thread = {};
    std::atomic<uint64_t>        batches_dropped  = {0};

    reader_state() = default;
    ~reader_state();

    reader_state(const reader_state&) = delete;
    reader_state& operator=(const reader_state&) = delete;
};

std::mutex&
purge_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

std::vector<std::pair<uint32_t, uint32_t>>&
purge_requests()
{
    static auto _v = std::vector<std::pair<uint32_t, uint32_t>>{};  // (gpu_id, slot)
    return _v;
}

reader_state&
state()
{
    static auto*& _v = common::static_object<reader_state>::construct();
    return *_v;
}

// Look up the gpuvm aperture for a gpu_id (needed to place the GTT allocation).
bool
get_gpuvm_aperture(int kfd, uint32_t gpu_id, uint64_t* base, uint64_t* limit)
{
    auto count = kfd_ioctl_get_process_apertures_new_args{};
    if(ioctl(kfd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &count) != 0) return false;
    if(count.num_of_nodes == 0 || count.num_of_nodes > 1024) return false;

    auto aps  = std::vector<kfd_process_device_apertures>(count.num_of_nodes);
    auto args = kfd_ioctl_get_process_apertures_new_args{};
    args.kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(aps.data());
    args.num_of_nodes                     = count.num_of_nodes;
    if(ioctl(kfd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &args) != 0) return false;

    for(uint32_t i = 0; i < args.num_of_nodes; ++i)
        if(aps[i].gpu_id == gpu_id)
        {
            *base  = aps[i].gpuvm_base;
            *limit = aps[i].gpuvm_limit;
            return *base < *limit;
        }
    return false;
}

// Allocate + map a GTT buffer, register it for dispatch-log, open a RAW_MMAP
// stream, and mmap the layout.
//
// Raw KFD alloc, not an HSA memory pool: DLOG_REGISTER_BUFFER accepts a pool
// buffer but mmap() on the resulting stream_fd then fails EOPNOTSUPP, since the
// driver only maps stream buffers it owns. HSA would force READ_RECORDS mode (a
// kernel copy per drain), losing the zero-copy read the feature exists for.
void
teardown_session(int kfd, dlog_session* s);

// Whether a failed setup can be retried later: anything depending on device
// readiness can, an ABI or geometry mismatch cannot.
bool
setup_session(int kfd, uint32_t gpu_id, dlog_session* s, bool* permanent = nullptr)
{
    if(permanent) *permanent = false;
    s->gpu_id = gpu_id;

    const uint64_t buf_bytes  = ring_bytes();
    const uint64_t arr_bytes  = ((8ull * sizeof(uint64_t)) + 7) & ~7ull;  // upper bound
    const uint64_t signal_off = buf_bytes + arr_bytes * 2;
    s->alloc_size             = round_up_page(static_cast<size_t>(signal_off + arr_bytes));

    uint64_t gpuvm_base  = 0;
    uint64_t gpuvm_limit = 0;
    if(!get_gpuvm_aperture(kfd, gpu_id, &gpuvm_base, &gpuvm_limit))
    {
        ROCP_WARNING << "KFD dispatch-log: gpuvm aperture lookup failed";
        return false;
    }

    auto     alloc  = kfd_ioctl_alloc_memory_of_gpu_args{};
    bool     ok     = false;
    uint64_t stride = round_up_page(s->alloc_size + (8u << 20));
    for(uint32_t i = 0; i < 128 && !ok; ++i)
    {
        uint64_t cand = (gpuvm_base > (64ull << 30) ? gpuvm_base : (64ull << 30)) + stride * i;
        cand          = (cand + 15ull) & ~15ull;
        if(cand < gpuvm_base || cand + s->alloc_size - 1 > gpuvm_limit) continue;
        alloc         = kfd_ioctl_alloc_memory_of_gpu_args{};
        alloc.va_addr = cand;
        alloc.size    = s->alloc_size;
        alloc.gpu_id  = gpu_id;
        alloc.flags   = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        if(ioctl(kfd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc) == 0) ok = true;
    }
    if(!ok)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: GTT buffer alloc failed (errno={})", errno);
        return false;
    }
    s->buffer_va    = alloc.va_addr;
    s->alloc_handle = alloc.handle;

    // From here on resources are acquired, so every failure return must unwind.
    bool                     success = false;
    common::scope_destructor cleanup{[&]() {
        if(!success) teardown_session(kfd, s);
    }};

    auto map                 = kfd_ioctl_map_memory_to_gpu_args{};
    map.handle               = alloc.handle;
    map.device_ids_array_ptr = reinterpret_cast<uint64_t>(&s->gpu_id);
    map.n_devices            = 1;
    if(ioctl(kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log: map memory to gpu failed";
        return false;
    }

    // buffer_size below is a uint32 field; ring_bytes() is bounded to fit it.
    static_assert(kDlogMaxRingBytes <= 0xFFFFFFFFull,
                  "dlog ring size must fit the uint32 buffer_size ioctl field");
    auto reg                 = kfd_ioctl_profiler_args{};
    reg.op                   = KFD_IOC_PROFILER_DLOG;
    reg.dlog.dlog_op         = KFD_IOC_PROFILER_DLOG_REGISTER_BUFFER;
    reg.dlog.gpu_id          = gpu_id;
    reg.dlog.reg.buffer_size = static_cast<uint32_t>(buf_bytes);
    reg.dlog.reg.buffer_addr = s->buffer_va;
    if(ioctl(kfd, AMDKFD_IOC_PROFILER, &reg) != 0)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: REGISTER_BUFFER failed (errno={})", errno);
        return false;
    }

    auto open                 = kfd_ioctl_profiler_args{};
    open.op                   = KFD_IOC_PROFILER_DLOG;
    open.dlog.dlog_op         = KFD_IOC_PROFILER_DLOG_OPEN_STREAM;
    open.dlog.gpu_id          = gpu_id;
    open.dlog.open.target_pid = static_cast<uint32_t>(getpid());
    open.dlog.open.flags      = KFD_DLOG_OPEN_F_RAW_MMAP;
    open.dlog.open.stream_fd  = -1;
    if(ioctl(kfd, AMDKFD_IOC_PROFILER, &open) != 0 || open.dlog.open.stream_fd < 0)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: OPEN_STREAM failed (errno={})", errno);
        return false;
    }
    s->stream_fd = open.dlog.open.stream_fd;

    auto sinfo = kfd_dlog_stream_args{};
    sinfo.op   = KFD_DLOG_STREAM_OP_INFO;
    if(ioctl(s->stream_fd, KFD_DLOG_STREAM_IOC, &sinfo) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log: STREAM_OP_INFO failed";
        return false;
    }
    s->info = sinfo.info;

    // Reject any geometry copy_pipes() would silently refuse to drain.
    const uint32_t rrc = s->info.region_record_count;
    if(s->info.fw_record_size != kFwRecBytes || s->info.num_regions == 0 ||
       s->info.num_regions > kMaxRegions || rrc == 0 || (rrc & (rrc - 1)) != 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: unsupported stream geometry (fw_record_size={} num_regions={} "
            "region_record_count={})",
            s->info.fw_record_size,
            s->info.num_regions,
            rrc);
        if(permanent) *permanent = true;  // ABI disagreement: retrying cannot help
        return false;
    }

    s->smap_len              = round_up_page(s->info.mmap_size);
    const uint64_t rec_bytes = static_cast<uint64_t>(s->info.num_regions) * rrc * kFwRecBytes;
    const uint64_t ptr_bytes = static_cast<uint64_t>(s->info.num_regions) * sizeof(uint64_t);
    auto           fits      = [&](uint64_t off, uint64_t bytes) {
        return off <= s->info.mmap_size && bytes <= s->info.mmap_size - off;
    };
    if(s->smap_len < s->info.mmap_size || !fits(s->info.records_offset, rec_bytes) ||
       !fits(s->info.wptr_offset, ptr_bytes) || !fits(s->info.rptr_offset, ptr_bytes))
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: stream layout does not fit the mapping (mmap_size={} "
            "records_offset={} wptr_offset={} rptr_offset={})",
            s->info.mmap_size,
            s->info.records_offset,
            s->info.wptr_offset,
            s->info.rptr_offset);
        if(permanent) *permanent = true;  // ABI disagreement: retrying cannot help
        return false;
    }

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
teardown_session(int kfd, dlog_session* s)
{
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
    if(s->buffer_va != 0)
    {
        auto unreg         = kfd_ioctl_profiler_args{};
        unreg.op           = KFD_IOC_PROFILER_DLOG;
        unreg.dlog.dlog_op = KFD_IOC_PROFILER_DLOG_UNREGISTER_BUFFER;
        unreg.dlog.gpu_id  = s->gpu_id;
        ioctl(kfd, AMDKFD_IOC_PROFILER, &unreg);

        auto unmap                 = kfd_ioctl_unmap_memory_from_gpu_args{};
        unmap.handle               = s->alloc_handle;
        unmap.device_ids_array_ptr = reinterpret_cast<uint64_t>(&s->gpu_id);
        unmap.n_devices            = 1;
        ioctl(kfd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);

        auto freea   = kfd_ioctl_free_memory_of_gpu_args{};
        freea.handle = s->alloc_handle;
        ioctl(kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &freea);
        s->buffer_va = 0;
    }
}

// STAGE 1, reader thread: copy the ring into a batch and get out. This is the
// ONLY code touching the volatile mapping, and every microsecond here widens
// the window for the firmware to lap us -- so no lock, no map, no pairing. If
// the processor is a full pipe behind we DROP the batch rather than block,
// since blocking the ring read causes the very overruns this avoids.
uint64_t
copy_records(reader_state& st)
{
    uint64_t     _total = 0;
    const size_t _n     = st.session_count.load(std::memory_order_acquire);

    // The rings are independent, so one GPU lapping cannot stall another.
    for(size_t i = 0; i < _n; ++i)
    {
        auto& _s = st.sessions[i];
        if(!_s.ready.load(std::memory_order_acquire)) continue;

        auto* base     = static_cast<uint8_t*>(_s.smap);
        auto* recs     = base + _s.info.records_offset;
        auto* wptr_arr = reinterpret_cast<volatile uint64_t*>(base + _s.info.wptr_offset);
        auto* rptr_arr = reinterpret_cast<volatile uint64_t*>(base + _s.info.rptr_offset);

        auto* _batch = st.pipe.acquire();
        if(!_batch)
        {
            // Do NOT read the ring: leaving the records for the next pass loses less
            // than copying them somewhere we cannot publish.
            st.batches_dropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        _batch->now_ns     = common::timestamp_ns();
        _batch->gpu_id     = _s.gpu_id;
        const auto _copied = copy_pipes(recs,
                                        _s.info.num_regions,
                                        _s.info.region_record_count,
                                        wptr_arr,
                                        rptr_arr,
                                        _s.cursors,
                                        _batch->records);

        // Release-store inside publish() orders the copy above before the
        // processor can observe the batch.
        if(_copied > 0) st.pipe.publish();
        _total += _copied;
    }
    return _total;
}

// STAGE 2, processor thread: pairs start/eop, resolves the generation, drives
// the hub. All the lock-taking work lives here, off the ring-reading path. It
// runs no client callback itself; hand_off_proven() submits to the task group.
uint64_t
process_batch(processor_state& proc, const record_batch& batch)
{
    const uint32_t _gpu     = batch.gpu_id;
    auto&          _pairing = proc.for_gpu(_gpu);

    return pair_records(
        batch.records.data(),
        batch.records.size(),
        _pairing,
        batch.now_ns,
        [_gpu](const drained_record& rec) {
            uint32_t slot = doorbell_off_to_page_slot(rec.doorbell_off);
            uint32_t gen  = doorbell_map().get_generation(_gpu, slot);
            // gpu_id stamped from the ring this record came from: a record can
            // only ever match a dispatch enqueued on the same GPU.
            auto key = correlation_key{slot, rec.dispatch_id, gen, _gpu};

            // Signal-less: this EOP IS the completion event.
            if(rec.start_known) signal_less_hub().note_start(key, rec.start_ticks);
            if(auto _proven = signal_less_hub().prove_eop(key, rec.end_ticks, rec.loss_free))
            {
                note_signal_less(signal_less_counter::eop_proven);
                hand_off_proven(std::move(*_proven));
                return;
            }
            note_signal_less(signal_less_counter::eop_unmatched);
        });
}

// The ring lapped us, so those records are gone -- but what we DID copy is still
// valid, so this reports and carries on rather than poisoning the session: an
// overrun is a bounded, already-known loss, not a reason to lose the feature
// process-wide. Rate-limited per escalating episode. Reader thread only, so
// the statics need no lock.
void
report_overrun(uint32_t gpu_id, const ring_cursors& c, uint64_t batches_dropped)
{
    // Per GPU: one ring lapping says nothing about another's.
    static std::unordered_map<uint32_t, std::pair<uint64_t, uint64_t>> _reported;
    auto&                                                              _prev = _reported[gpu_id];
    if(c.overruns == _prev.first && batches_dropped == _prev.second) return;
    _prev = {c.overruns, batches_dropped};

    ROCP_WARNING << fmt::format(
        "KFD dispatch-log (gpu_id={}): ring overrun -- {} lap(s) so far, at least {} record(s) "
        "lost, {} "
        "batch(es) dropped because the processor fell behind. Those dispatches have no "
        "dispatch-log timestamps; everything else is unaffected and collection continues. Raise "
        "the ring with ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB if this repeats.",
        gpu_id,
        c.overruns,
        c.lost_records,
        batches_dropped);
}

// KFD_DLOG_STREAM_OP_STATUS is DIAGNOSTICS ONLY: the kernel's counters are logged
// but never gate a decision (wptr is the design's sole overrun authority).
void
log_stream_status(const dlog_session& s)
{
    if(s.stream_fd < 0) return;
    auto args = kfd_dlog_stream_args{};
    args.op   = KFD_DLOG_STREAM_OP_STATUS;
    if(ioctl(s.stream_fd, KFD_DLOG_STREAM_IOC, &args) != 0) return;
    ROCP_INFO << fmt::format(
        "KFD dispatch-log status: flags=0x{:x} records_read={} source_overruns={} copy_faults={}",
        args.status.status,
        args.status.records_read,
        args.status.source_overruns,
        args.status.copy_faults);
}

void
stop_reader()
{
    auto& st = state();
    if(!st.running) return;

    // Latch unavailable before the join so no dispatch can acquire a correlation
    // key against a reader that is on its way out (cleared again below).
    st.reader_unavailable.store(true, std::memory_order_release);
    st.stop.store(true, std::memory_order_release);
    if(st.wake_fd >= 0)
    {
        uint64_t one = 1;
        auto     _   = ::write(st.wake_fd, &one, sizeof(one));
        (void) _;
    }
    if(st.thread.joinable()) st.thread.join();
    // Reader first: it is the sole producer, so once it is joined the pipe can
    // only shrink. The processor then drains what is left and exits.
    if(st.processor_thread.joinable()) st.processor_thread.join();

    // Teardown runs without taking setup_mu on purpose: it is only safe because the
    // caller (registration::finalize) tears down queue interception
    // (queue_controller_fini) BEFORE kfd::finalize(), so no interceptor thread can
    // still be inside ensure_reader_session()/setup_session() by the time we get
    // here, and finalize itself is std::call_once (single-threaded). If that ordering
    // ever changes, this teardown must take st.setup_mu to serialize against a
    // concurrent setup_session(). The reader thread is already stopped+joined above.
    const size_t _n = st.session_count.load(std::memory_order_acquire);
    for(size_t i = 0; i < _n; ++i)
    {
        st.sessions[i].ready.store(false, std::memory_order_release);
        teardown_session(st.kfd_fd, &st.sessions[i]);
        st.sessions[i].failed.store(false, std::memory_order_release);
    }
    st.session_count.store(0, std::memory_order_release);
    st.any_session_ready.store(false, std::memory_order_release);
    st.reader_unavailable.store(false, std::memory_order_release);
    if(st.kfd_fd >= 0)
    {
        ::close(st.kfd_fd);
        st.kfd_fd = -1;
    }
    if(st.wake_fd >= 0)
    {
        ::close(st.wake_fd);
        st.wake_fd = -1;
    }
    st.running = false;
    ROCP_INFO << "KFD dispatch-log reader: stopped";
}

reader_state::~reader_state() { stop_reader(); }

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
    st.running = false;  // makes stop_reader() and ~reader_state() no-ops
    // Not detach(): that would leave the handle believing a thread exists.
    new(&st.thread) std::thread{};
    new(&st.processor_thread) std::thread{};
    st.wake_fd = -1;
    st.kfd_fd  = -1;
    // Dropping ownership without freeing: the parent still owns them, and a free
    // here would corrupt its state.
    const size_t _n = st.session_count.load(std::memory_order_relaxed);
    for(size_t i = 0; i < _n && i < kMaxSessions; ++i)
    {
        st.sessions[i].ready.store(false, std::memory_order_relaxed);
        st.sessions[i].smap      = MAP_FAILED;
        st.sessions[i].stream_fd = -1;
        st.sessions[i].buffer_va = 0;
    }
    st.session_count.store(0, std::memory_order_relaxed);
}

// Sole consumer of the pipe; owns all pairing state, so it needs no lock for it.
void
processor_loop()
{
    auto& st   = state();
    auto  proc = processor_state{};

    uint64_t last_evict_ns = common::timestamp_ns();

    while(true)
    {
        auto* _batch = st.pipe.peek();
        if(!_batch)
        {
            if(st.stop.load(std::memory_order_acquire)) break;
            // Nothing to do. Sleeping here costs nothing on the ring: the reader
            // keeps copying regardless of what this thread is doing.
            std::this_thread::sleep_for(std::chrono::microseconds{200});
            continue;
        }

        process_batch(proc, *_batch);
        st.pipe.pop();

        // The pairing state is ours, so the purge happens here rather than on the
        // destroying thread.
        {
            auto _reqs = std::vector<std::pair<uint32_t, uint32_t>>{};
            {
                auto lk = std::lock_guard<std::mutex>{purge_mutex()};
                _reqs.swap(purge_requests());
            }
            for(const auto& _req : _reqs)
            {
                // A GPU that has not paired anything yet has no entry, and there
                // is nothing to erase for it.
                auto _it = proc.by_gpu.find(_req.first);
                if(_it != proc.by_gpu.end())
                    _it->second.erase_slot(_req.second, kDoorbellSlotsPerPage);
            }
        }

        const uint64_t _now = common::timestamp_ns();
        if(_now - last_evict_ns >= kProcessorEvictIntervalNs)
        {
            for(auto& _itr : proc.by_gpu)
                _itr.second.evict_stale(_now, kStartMaxAgeNs);
            last_evict_ns = _now;
        }
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
            "START(s) still retained at exit",
            _gpu_itr.first,
            _p.starts_seen,
            _p.eops_seen,
            _p.unmatched_eops,
            _p.starts_overwritten,
            _p.pending_starts.size());
    }
}

void
reader_loop()
{
    auto& st = state();

    auto     wake          = pollfd{.fd = st.wake_fd, .events = POLLIN, .revents = 0};
    uint64_t total_seen    = 0;
    uint64_t last_evict_ns = common::timestamp_ns();

    while(!st.stop.load(std::memory_order_acquire))
    {
        // 1 ms while a session is live; records must be drained faster than the
        // firmware can lap the ring.
        const int _timeout_ms = st.any_session_ready.load(std::memory_order_acquire) ? 1 : 100;
        int       rc          = ::poll(&wake, 1, _timeout_ms);
        if(rc < 0 && errno != EINTR)
        {
            // Nothing will drain the ring again, so publish that terminally: this is
            // the same permanent-unavailable state setup failure uses.
            // kfd_dispatch_log_available() is deliberately left alone so queue destroy
            // still retires doorbell-map entries.
            st.any_session_ready.store(false, std::memory_order_release);
            st.reader_unavailable.store(true, std::memory_order_release);
            ROCP_WARNING << fmt::format(
                "KFD dispatch-log reader: poll failed (errno={}), reader exiting; dispatch-log is "
                "now disabled for this process, all dispatches use HSA timestamps",
                errno);
            break;
        }
        if(wake.revents & POLLIN)
        {
            uint64_t v = 0;
            while(::read(st.wake_fd, &v, sizeof(v)) == sizeof(v))
            {}
        }

        if(st.any_session_ready.load(std::memory_order_acquire))
        {
            uint64_t now_ns = common::timestamp_ns();

            // The whole hot path: copy and publish. Everything else the reader
            // used to do now happens on the processor thread.
            total_seen += copy_records(st);
            for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
                report_overrun(st.sessions[i].gpu_id,
                               st.sessions[i].cursors,
                               st.batches_dropped.load(std::memory_order_relaxed));

            st.drain_epoch.fetch_add(1, std::memory_order_release);

            if(now_ns - last_evict_ns >= kEvictIntervalNs)
            {
                for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n;
                    ++i)
                    log_stream_status(st.sessions[i]);
                last_evict_ns = now_ns;
            }
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
                           st.batches_dropped.load(std::memory_order_relaxed));
            log_stream_status(st.sessions[i]);
        }
    }

    ROCP_INFO << fmt::format("KFD dispatch-log reader: loop exited, total pairs seen = {}",
                             total_seen);

    // Signal-less chain, reported from the reader too so the break point is
    // visible even if teardown does not run. Silent unless the feature is active.
    uint64_t _ring_overruns = 0;
    uint64_t _ring_lost     = 0;
    for(size_t i = 0, _n = st.session_count.load(std::memory_order_acquire); i < _n; ++i)
    {
        _ring_overruns += st.sessions[i].cursors.overruns;
        _ring_lost += st.sessions[i].cursors.lost_records;
    }

    // The signal-less chain itself is summarised once, by teardown.
    ROCP_WARNING_IF(_ring_overruns > 0 || _ring_lost > 0) << fmt::format(
        "KFD dispatch-log ring: {} lap(s), {} record(s) lost to laps, {} batch(es) dropped -- a "
        "lost record is a lost START, which shows up as a start-unknown no-timing",
        _ring_overruns,
        _ring_lost,
        st.batches_dropped.load(std::memory_order_relaxed));
}
}  // namespace

bool
start_kfd_reader()
{
    auto& st = state();
    if(st.running) return true;

    // Any return below that leaves st.running false is a failure: drop the fds and
    // latch the reader unavailable, so dispatches short-circuit in
    // ensure_reader_session() instead of locking setup_mu for a reader that will
    // never have a session.
    common::scope_destructor cleanup{[&st]() {
        if(st.running) return;
        st.reader_unavailable.store(true, std::memory_order_release);
        if(st.kfd_fd >= 0) ::close(st.kfd_fd);
        if(st.wake_fd >= 0) ::close(st.wake_fd);
        st.kfd_fd  = -1;
        st.wake_fd = -1;
    }};

    st.wake_fd = eventfd(0, kEventfdFlags);
    if(st.wake_fd < 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: eventfd creation failed, reader not started";
        return false;
    }

    st.kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(st.kfd_fd < 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: /dev/kfd open failed, reader not started";
        return false;
    }

    // Registered before the thread exists: a fork in between would inherit a reader
    // thread with no handler to abandon it.
    if(pthread_atfork(nullptr, nullptr, disable_reader_in_child) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: pthread_atfork failed, reader not started";
        return false;
    }

    st.stop.store(false, std::memory_order_release);

    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    try
    {
        // Processor first: it must be ready to consume before the reader can
        // publish, otherwise the first batches are dropped for no reason.
        st.processor_thread = std::thread{processor_loop};
        st.thread           = std::thread{reader_loop};
    } catch(const std::system_error& e)
    {
        // init_kfd_profiler() promises never to throw: the scope guard drops the fds
        // and leaves the dispatch-log unavailable, so dispatches use HSA timestamps.
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
        // If the reader failed after the processor started, stop the processor too
        // rather than leaving an orphan waiting on a producer that will never run.
        st.stop.store(true, std::memory_order_release);
        if(st.processor_thread.joinable()) st.processor_thread.join();
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log reader: thread creation failed ({}), reader not started", e.what());
        return false;
    }
    internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    st.running = true;
    ROCP_INFO << "KFD dispatch-log reader: started";
    return true;
}

void
stop_kfd_reader()
{
    stop_reader();
}

bool
wait_for_reader_drain_barrier(uint64_t timeout_ns)
{
    auto& st = state();
    if(!st.running || !st.any_session_ready.load(std::memory_order_acquire)) return true;

    const uint64_t _start    = st.drain_epoch.load(std::memory_order_acquire);
    const uint64_t _deadline = common::timestamp_ns() + timeout_ns;

    // Two advances: the pass in flight when we asked may have already read past
    // our records, so the second one is the first that is provably complete.
    while(st.drain_epoch.load(std::memory_order_acquire) < _start + 2)
    {
        if(common::timestamp_ns() >= _deadline)
        {
            ROCP_WARNING << "KFD dispatch-log: timed out waiting for a reader drain barrier; "
                            "continuing without it";
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
    if(!st.running || st.wake_fd < 0) return;
    uint64_t one = 1;
    auto     _   = ::write(st.wake_fd, &one, sizeof(one));
    (void) _;
}

void
request_reader_slot_purge(uint32_t gpu_id, uint32_t doorbell_slot)
{
    auto lk = std::lock_guard<std::mutex>{purge_mutex()};
    purge_requests().emplace_back(gpu_id, doorbell_slot);
}

namespace
{
// latch_retryable: at first dispatch the device is certainly usable, so any
// failure is permanent and latching stops every later dispatch repeating 128
// alloc ioctls. At init the device may just not be ready, so a retryable
// failure must not disable the feature process-wide.
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

bool
establish_session(uint32_t gpu_id, bool latch_retryable)
{
    if(!kfd_dispatch_log_available() || !gpu_supports_dispatch_log(gpu_id)) return false;

    auto& st = state();
    // A dead or stopping reader disables every GPU at once.
    if(st.reader_unavailable.load(std::memory_order_acquire)) return false;

    if(auto* _existing = find_session(st, gpu_id))
        return _existing->ready.load(std::memory_order_acquire);

    auto lk = std::lock_guard<std::mutex>{st.setup_mu};
    // Re-check under the lock: another thread may have armed this GPU meanwhile.
    if(auto* _existing = find_session(st, gpu_id))
        return _existing->ready.load(std::memory_order_relaxed);
    if(st.reader_unavailable.load(std::memory_order_relaxed)) return false;
    if(st.kfd_fd < 0) return false;  // reader not started

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

        // Latched for THIS GPU only: another GPU's session is unaffected.
        _s.gpu_id = gpu_id;
        _s.failed.store(true, std::memory_order_release);
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

    // Break the reader out of its coarse pre-session poll so the first dispatches
    // are drained immediately instead of up to 100 ms later.
    uint64_t one = 1;
    auto     _   = ::write(st.wake_fd, &one, sizeof(one));
    (void) _;
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
