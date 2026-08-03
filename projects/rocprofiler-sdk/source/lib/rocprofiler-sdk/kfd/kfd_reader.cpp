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

// Active (v5) dispatch-log profiler ABI. Must be the ONLY kfd ioctl header in
// this translation unit (it conflicts with lib/rocprofiler-sdk/details/kfd_ioctl.h).
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
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

// Dispatch-log ring size in bytes. Validated before any sizing math uses it:
// dlog_ring_bytes_from_mb_str() accepts only [1, kDlogMaxRingMb] MB, so the value
// always fits the uint32 buffer_size ioctl field.
uint64_t
ring_bytes()
{
    auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_SIZE_MB");
    if(!_v) return kDlogDefaultRingMb * 1024 * 1024;

    uint64_t _bytes = dlog_ring_bytes_from_mb_str(*_v);
    if(_bytes == 0)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: ignoring invalid ROCPROFILER_KFD_DISPATCH_LOG_SIZE_MB='{}' "
            "(expected an integer 1-{}); using {} MB",
            *_v,
            kDlogMaxRingMb,
            kDlogDefaultRingMb);
        return kDlogDefaultRingMb * 1024 * 1024;
    }
    return _bytes;
}

// fw_record, the 20-byte record layout, the kRec* type constants, kFwRecBytes, the
// drain_state cursor bookkeeping, and the drain_pipes() logic all live in the
// header-only dlog_drain.hpp so the drain is unit-testable without a GPU.
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

// One dispatch-log data-ring session for a single GPU. The reader allocates a
// GTT buffer, registers it, opens a RAW_MMAP stream against its own pid, and
// mmaps the layout.
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

    // Per-pipe drain cursors + unmatched-start bookkeeping. Touched only by the
    // reader thread, so no lock is needed. The drain logic itself lives in the
    // header-only drain_pipes() so it can be unit-tested without a GPU.
    drain_state drain = {};
};

// Reader thread state. Single instance via static_object (ordered teardown). Its
// destructor stops+joins the thread so a joinable std::thread is never
// destroyed (would call std::terminate). Mirrors poll_kfd_t in kfd.cpp.
struct reader_state
{
    std::thread       thread  = {};
    std::atomic<bool> stop    = {false};
    int               wake_fd = -1;
    bool              running = false;

    int          kfd_fd  = -1;
    dlog_session session = {};

    // Session is set up on the app thread (ensure_reader_session, via
    // create_queue) and drained on the reader thread. setup_mu serializes setup;
    // session_ready publishes the completed session to the reader (acquire/release
    // so the reader sees a fully-built session before it drains).
    std::mutex        setup_mu      = {};
    std::atomic<bool> session_ready = {false};
    // Latched when the reader can never serve a session: setup_session() failed
    // (no aperture, alloc, ABI, geometry -- all permanent, and retrying would
    // repeat 128 alloc ioctls plus a warning on every dispatch), the reader failed
    // to start, or we are in a forked child. Checked before setup_mu is taken, so a
    // latched reader never makes a dispatch block on that mutex. Cleared only by
    // stop_reader().
    std::atomic<bool> setup_failed = {false};

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
// stream, and mmap the layout. The create_queue trigger guarantees the device is
// acquired before this runs, so the raw KFD allocation does not race init.
//
// ALLOCATION DESIGN (why raw KFD alloc, not HSA memory pools):
// The dispatch-log RAW_MMAP consumption path requires a buffer the KFD driver
// itself allocated. An HSA-pool buffer (hsa_amd_memory_pool_allocate) is accepted
// by DLOG_REGISTER_BUFFER, but mmap() on the resulting stream_fd fails with
// EOPNOTSUPP -- the driver only maps stream buffers it owns. Using HSA would
// therefore force the READ_RECORDS consumption mode (a kernel-mediated copy per
// drain), abandoning the zero-copy read that is the feature's whole overhead
// advantage. Hence the raw-KFD + RAW_MMAP path.
//
// Forward declaration: setup_session arms a scope_destructor that unwinds
// partially-acquired resources via teardown_session on any failure return after
// the GTT allocation succeeds (disarmed only on the success path).
void
teardown_session(int kfd, dlog_session* s);

bool
setup_session(int kfd, uint32_t gpu_id, dlog_session* s)
{
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

    // From here on, resources are acquired (GTT alloc, then map/register/stream/
    // mmap). Any early return must unwind them. Arm a scope guard that tears down
    // the partially-built session unless we reach the success path and disarm it.
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
    static_assert(kDlogMaxRingMb * 1024 * 1024 <= 0xFFFFFFFFull,
                  "dlog ring size must fit the uint32 buffer_size ioctl field");
    auto reg                       = kfd_ioctl_profiler_args{};
    reg.op                         = KFD_IOC_PROFILER_DLOG;
    reg.dlog.dlog_op               = KFD_IOC_PROFILER_DLOG_REGISTER_BUFFER;
    reg.dlog.gpu_id                = gpu_id;
    reg.dlog.reg.buffer_size       = static_cast<uint32_t>(buf_bytes);
    reg.dlog.reg.buffer_addr       = s->buffer_va;
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

    // Reject any geometry drain_pipes() would silently refuse to drain (record
    // size, region count beyond the cursor storage, non-power-of-two slot count),
    // so setup fails loudly instead of reporting ready and draining nothing.
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
        return false;
    }

    // The offsets below are driver-supplied and are used to form pointers that are
    // read (records, wptr) AND written (rptr), so every array must lie inside the
    // mapping. Subtract instead of add so the bounds check cannot overflow, and
    // reject a mmap_size whose page round-up wrapped.
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
        auto unreg           = kfd_ioctl_profiler_args{};
        unreg.op             = KFD_IOC_PROFILER_DLOG;
        unreg.dlog.dlog_op   = KFD_IOC_PROFILER_DLOG_UNREGISTER_BUFFER;
        unreg.dlog.gpu_id    = s->gpu_id;
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

// Drain new firmware records via the shared, unit-tested drain_pipes() core. This
// wrapper supplies the reader-specific bits: the mmap'd ring pointers, the host
// clock for start aging, and a paired (start,end) deposit into ResultsMap. The key
// uses the PAGE-RELATIVE doorbell slot (doorbell_off_to_page_slot) so it matches
// the value the capture side computes from the queue's doorbell pointer; the
// doorbell->queue binding is established directly at capture, so no per-record bind
// happens here. See dlog_drain.hpp for the per-pipe ring geometry.
uint64_t
drain_records(dlog_session* s)
{
    auto* base     = static_cast<uint8_t*>(s->smap);
    auto* recs     = base + s->info.records_offset;
    auto* wptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.wptr_offset);
    auto* rptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.rptr_offset);

    return drain_pipes(
        recs,
        s->info.num_regions,
        s->info.region_record_count,
        wptr_arr,
        rptr_arr,
        s->drain,
        common::timestamp_ns(),
        [](uint32_t doorbell_off, uint32_t dispatch_id, uint64_t start_ticks, uint64_t end_ticks) {
            uint32_t slot = doorbell_off_to_page_slot(doorbell_off);
            uint32_t gen  = doorbell_map().get_generation(slot);
            results_map().deposit(
                correlation_key{slot, dispatch_id, gen},
                kfd_timing_result{start_ticks, end_ticks, common::timestamp_ns()});
        });
}

// Loud one-shot overrun report. Reader thread only, so the local static needs no
// synchronization. Detection/telemetry only: an overrun already dropped records
// by the time it is observed, and Phase 0 does not change what completes.
void
warn_on_overrun(const drain_state& d)
{
    static bool warned = false;
    if(d.overruns == 0 || warned) return;
    warned = true;
    ROCP_WARNING << fmt::format(
        "KFD dispatch-log: ring OVERRUN ({} occurrence(s), at least {} record(s) dropped) -- the "
        "firmware lapped the reader, so those dispatches have no dispatch-log timestamps. Raise "
        "the ring with ROCPROFILER_KFD_DISPATCH_LOG_SIZE_MB (currently {} MB max).",
        d.overruns,
        d.lost_records,
        kDlogMaxRingMb);
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
    st.setup_failed.store(true, std::memory_order_release);
    st.stop.store(true, std::memory_order_release);
    if(st.wake_fd >= 0)
    {
        uint64_t one = 1;
        auto     _   = ::write(st.wake_fd, &one, sizeof(one));
        (void) _;
    }
    if(st.thread.joinable()) st.thread.join();

    // Teardown runs without taking setup_mu on purpose: it is only safe because the
    // caller (registration::finalize) tears down queue interception
    // (queue_controller_fini) BEFORE kfd::finalize(), so no interceptor thread can
    // still be inside ensure_reader_session()/setup_session() by the time we get
    // here, and finalize itself is std::call_once (single-threaded). If that ordering
    // ever changes, this teardown must take st.setup_mu to serialize against a
    // concurrent setup_session(). The reader thread is already stopped+joined above.
    teardown_session(st.kfd_fd, &st.session);
    st.session_ready.store(false, std::memory_order_release);
    st.setup_failed.store(false, std::memory_order_release);
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

// pthread_atfork child handler. Only the forking thread survives into the child,
// so the reader is gone while its joinable std::thread handle, fds and mapping are
// inherited. Permanently disable the reader in the child (dispatches there use HSA
// timestamps); there is deliberately no restart path. Async-signal-safe: scalar
// state reset only -- never join, close, munmap, ioctl or allocate here.
void
disable_reader_in_child()
{
    // Also latches kfd_dispatch_log_available() false, so the child never takes
    // the DoorbellMap lock a vanished thread may have been holding.
    disable_kfd_dispatch_log();

    auto& st = state();
    st.session_ready.store(false, std::memory_order_relaxed);
    // Latch unavailable: setup_mu may have been held by a thread that did not
    // survive the fork, so a child dispatch must not reach it.
    st.setup_failed.store(true, std::memory_order_relaxed);
    st.running = false;  // makes stop_reader() and ~reader_state() no-ops
    // Abandon the handle of the vanished thread. Not detach(): that would touch a
    // pthread descriptor fork has already reclaimed in the child.
    new(&st.thread) std::thread{};
    st.wake_fd = -1;
    st.kfd_fd  = -1;
    // Drop ownership of the inherited stream fd / mapping / GTT allocation so the
    // child can never tear down kernel objects that belong to the parent.
    st.session.smap      = MAP_FAILED;
    st.session.stream_fd = -1;
    st.session.buffer_va = 0;
}

void
reader_loop()
{
    auto& st = state();

    // The session is established lazily by ensure_reader_session() on the
    // app/queue-creation thread (which guarantees the agent cache is ready). Here
    // we simply drain whatever has been published.
    auto     wake          = pollfd{.fd = st.wake_fd, .events = POLLIN, .revents = 0};
    uint64_t total_seen    = 0;
    uint64_t last_evict_ns = common::timestamp_ns();

    // Age out unpaired starts and un-taken results at most this often (not every
    // 1 ms poll). A deposited result is normally taken within milliseconds by the
    // completion path; anything older than kResultMaxAgeNs was never claimed (the
    // dispatch completed via HSA fallback, or the firmware record was misattributed
    // to a doorbell no dispatch keyed on) and must be reclaimed to bound memory.
    constexpr uint64_t kEvictIntervalNs = 1'000'000'000ull;  // 1 s
    constexpr uint64_t kStartMaxAgeNs   = 5'000'000'000ull;  // 5 s
    constexpr uint64_t kResultMaxAgeNs  = 5'000'000'000ull;  // 5 s

    while(!st.stop.load(std::memory_order_acquire))
    {
        // Poll cadence: 1 ms while a session is live (records must be drained
        // promptly), but a coarse 100 ms before any session is published so the
        // reader is not a 1000-wakeup/sec idle spinner during the (possibly long)
        // window before the first GPU queue is created. stop_reader() writes wake_fd
        // to break either wait immediately.
        const int _timeout_ms = st.session_ready.load(std::memory_order_acquire) ? 1 : 100;
        int       rc          = ::poll(&wake, 1, _timeout_ms);
        if(rc < 0 && errno != EINTR)
        {
            // Reader dead: nothing will drain the ring again, so publish that
            // terminally. Clearing session_ready + latching setup_failed is the
            // same permanent-unavailable state setup failure uses, and makes
            // ensure_reader_session() refuse to hand out new correlation keys;
            // dispatches already in flight simply miss and use HSA timestamps.
            // kfd_dispatch_log_available() is deliberately left alone so queue
            // destroy still retires doorbell-map entries.
            st.session_ready.store(false, std::memory_order_release);
            st.setup_failed.store(true, std::memory_order_release);
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

        if(st.session_ready.load(std::memory_order_acquire))
        {
            uint64_t now_ns = common::timestamp_ns();

            total_seen += drain_records(&st.session);
            warn_on_overrun(st.session.drain);

            if(now_ns - last_evict_ns >= kEvictIntervalNs)
            {
                st.session.drain.evict_stale(now_ns, kStartMaxAgeNs);
                results_map().evict_stale(now_ns, kResultMaxAgeNs);
                log_stream_status(st.session);
                last_evict_ns = now_ns;
            }
        }
    }

    // Final drain to catch late records.
    if(st.session_ready.load(std::memory_order_acquire))
    {
        total_seen += drain_records(&st.session);
        warn_on_overrun(st.session.drain);
        log_stream_status(st.session);
    }

    // Misses are dispatches whose completion callback ran before the reader had
    // deposited their firmware record; they silently used HSA timestamps.
    const auto _stats = results_map().stats();
    ROCP_INFO << fmt::format(
        "KFD dispatch-log reader: loop exited, total pairs seen = {}, completion-path "
        "lookups: {} hit / {} miss",
        total_seen,
        _stats.hits,
        _stats.misses);
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
        st.setup_failed.store(true, std::memory_order_release);
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

    // Must be registered before the thread exists: a fork in between would leave the
    // child with a joinable handle to a vanished thread and no handler to abandon it
    // (std::terminate from ~reader_state).
    if(pthread_atfork(nullptr, nullptr, disable_reader_in_child) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: pthread_atfork failed, reader not started";
        return false;
    }

    st.stop.store(false, std::memory_order_release);

    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    try
    {
        st.thread = std::thread{reader_loop};
    }
    catch(const std::system_error& e)
    {
        // init_kfd_profiler() promises never to throw: the scope guard drops the fds
        // and leaves the dispatch-log unavailable, so dispatches use HSA timestamps.
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
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
ensure_reader_session(uint32_t gpu_id)
{
    if(!kfd_dispatch_log_available() || !gpu_supports_dispatch_log(gpu_id)) return false;

    auto& st = state();
    // setup_failed is checked FIRST so it overrides a still-published session: a
    // dead or stopping reader latches it while session_ready may still be set.
    if(st.setup_failed.load(std::memory_order_acquire)) return false;
    // The session's gpu_id is written before session_ready is released and never
    // mutated afterwards, so this acquire-load makes it safe to read unlocked.
    if(st.session_ready.load(std::memory_order_acquire)) return st.session.gpu_id == gpu_id;

    auto lk = std::lock_guard<std::mutex>{st.setup_mu};
    if(st.session_ready.load(std::memory_order_relaxed))
        return st.session.gpu_id == gpu_id;  // lost the race; someone set it up
    // Threads that all passed the unlocked check before the first failure land here
    // one at a time; without this recheck each would repeat the failed setup.
    if(st.setup_failed.load(std::memory_order_relaxed)) return false;
    if(st.kfd_fd < 0) return false;  // reader not started

    // One session for the first supported GPU. (This gpu_id is supported per the
    // guard above.)
    if(!setup_session(st.kfd_fd, gpu_id, &st.session))
    {
        st.setup_failed.store(true, std::memory_order_release);
        // The GPU advertised dispatch-log support but the session could not be
        // established (specific cause logged by setup_session above). This latch is
        // permanent, so warn once that the whole process now uses HSA timestamps.
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: gpu_id={} supports dispatch-log but session setup failed; "
            "dispatch-log is now disabled for this process, all dispatches use HSA timestamps",
            gpu_id);
        return false;
    }
    st.session_ready.store(true, std::memory_order_release);

    // Break the reader out of its coarse pre-session poll so the first dispatches
    // are drained immediately instead of up to 100 ms later.
    uint64_t one = 1;
    auto     _   = ::write(st.wake_fd, &one, sizeof(one));
    (void) _;
    return true;
}
}  // namespace kfd
}  // namespace rocprofiler
