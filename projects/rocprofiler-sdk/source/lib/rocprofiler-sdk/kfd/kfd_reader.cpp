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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <poll.h>
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
constexpr int      kEventfdFlags = EFD_CLOEXEC | EFD_NONBLOCK;
constexpr uint32_t kBufferKb     = 80;  // dlog ring size; matches reference harness default

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

// One dispatch-log data-ring session for a single GPU. The reader (as the
// in-process target+profiler in Mode 1) allocates a GTT buffer, registers it,
// opens a RAW_MMAP stream against its own pid, and mmaps the layout.
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
// advantage. This raw-KFD + RAW_MMAP combination matches the validated reference
// (test-framework dlog_stream_test.cpp). See design notes and
// ~/kfd_probe/verify_hsa_alloc.cpp for the EOPNOTSUPP evidence.
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

    const uint64_t buf_bytes  = static_cast<uint64_t>(kBufferKb) * 1024u;
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

    auto reg                      = kfd_ioctl_profiler_args{};
    reg.op                        = KFD_IOC_PROFILER_DLOG_REGISTER_BUFFER;
    reg.dlog_register.gpu_id      = gpu_id;
    reg.dlog_register.buffer_size = static_cast<uint32_t>(buf_bytes);
    reg.dlog_register.buffer_addr = s->buffer_va;
    if(ioctl(kfd, AMDKFD_IOC_PROFILER, &reg) != 0)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: REGISTER_BUFFER failed (errno={})", errno);
        return false;
    }

    auto open                 = kfd_ioctl_profiler_args{};
    open.op                   = KFD_IOC_PROFILER_DLOG_OPEN_STREAM;
    open.dlog_open.gpu_id     = gpu_id;
    open.dlog_open.target_pid = static_cast<uint32_t>(getpid());
    open.dlog_open.flags      = KFD_DLOG_OPEN_F_RAW_MMAP;
    open.dlog_open.stream_fd  = -1;
    if(ioctl(kfd, AMDKFD_IOC_PROFILER, &open) != 0 || open.dlog_open.stream_fd < 0)
    {
        ROCP_WARNING << fmt::format("KFD dispatch-log: OPEN_STREAM failed (errno={})", errno);
        return false;
    }
    s->stream_fd = open.dlog_open.stream_fd;

    if(ioctl(s->stream_fd, KFD_DLOG_STREAM_IOC_INFO, &s->info) != 0)
    {
        ROCP_WARNING << "KFD dispatch-log: STREAM_IOC_INFO failed";
        return false;
    }

    if(s->info.fw_record_size != kFwRecBytes || s->info.num_regions == 0 || s->info.num_regions > 8)
    {
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: unsupported stream geometry (fw_record_size={} num_regions={})",
            s->info.fw_record_size,
            s->info.num_regions);
        return false;
    }

    s->smap_len = round_up_page(s->info.mmap_size);
    s->smap     = mmap(nullptr, s->smap_len, PROT_READ | PROT_WRITE, MAP_SHARED, s->stream_fd, 0);
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
        "KFD dispatch-log: session ready gpu_id={} num_regions={} region_records={} rec_bytes={}",
        gpu_id,
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
    if(s->buffer_va != 0 && kfd >= 0)
    {
        auto unreg                   = kfd_ioctl_profiler_args{};
        unreg.op                     = KFD_IOC_PROFILER_DLOG_UNREGISTER_BUFFER;
        unreg.dlog_unregister.gpu_id = s->gpu_id;
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

// Age out pending dispatch_start records that never received a matching eop
// (queue died mid-dispatch, ring overwrite). Reader-thread-only, no lock. Mirrors
// ResultsMap::evict_stale. now_ns is passed in for determinism/testability.
void
evict_stale_starts(dlog_session* s, uint64_t now_ns, uint64_t max_age_ns)
{
    s->drain.evict_stale(now_ns, max_age_ns);
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
    if(s->smap == MAP_FAILED) return 0;

    auto* base     = static_cast<uint8_t*>(s->smap);
    auto* recs     = base + s->info.records_offset;
    auto* wptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.wptr_offset);
    auto* rptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.rptr_offset);

    {
        // DBGWPTR: does the ring advance at all? Print wptr[]/rptr[] when any wptr moves.
        static uint64_t s_last[8] = {0};
        const uint32_t  np        = s->info.num_regions <= 8 ? s->info.num_regions : 8;
        bool            moved     = false;
        for(uint32_t p = 0; p < np; ++p)
            if(wptr_arr[p] != s_last[p]) moved = true;
        if(moved)
        {
            std::string line;
            for(uint32_t p = 0; p < np; ++p)
            {
                line += fmt::format("p{}:w={},r={} ", p, wptr_arr[p], rptr_arr[p]);
                s_last[p] = wptr_arr[p];
            }
            ROCP_WARNING << "DBGWPTR " << line;
        }
    }

    return drain_pipes(
        recs,
        s->info.num_regions,
        s->info.region_record_count,
        wptr_arr,
        rptr_arr,
        s->drain,
        common::timestamp_ns(),
        [](uint32_t, uint32_t) {
            // Binding is done at capture (page-relative doorbell slot), so there is
            // nothing to do per raw record here.
        },
        [](uint32_t doorbell_off, uint32_t dispatch_id, uint64_t start_ticks, uint64_t end_ticks) {
            uint32_t slot = doorbell_off_to_page_slot(doorbell_off);
            uint32_t gen  = doorbell_map().get_generation(slot);
            results_map().deposit(
                correlation_key{slot, dispatch_id, gen},
                kfd_timing_result{start_ticks, end_ticks, common::timestamp_ns()});
        });
}

void
stop_reader()
{
    auto& st = state();
    if(!st.running) return;

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
            ROCP_WARNING << "KFD dispatch-log reader: poll failed, exiting reader loop";
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

            if(now_ns - last_evict_ns >= kEvictIntervalNs)
            {
                evict_stale_starts(&st.session, now_ns, kStartMaxAgeNs);
                results_map().evict_stale(now_ns, kResultMaxAgeNs);
                last_evict_ns = now_ns;
            }
        }
    }

    // Final drain to catch late records.
    if(st.session_ready.load(std::memory_order_acquire)) total_seen += drain_records(&st.session);

    ROCP_INFO << fmt::format("KFD dispatch-log reader: loop exited, total records seen = {}",
                             total_seen);
}
}  // namespace

void
start_kfd_reader()
{
    auto& st = state();
    if(st.running) return;

    st.wake_fd = eventfd(0, kEventfdFlags);
    if(st.wake_fd < 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: eventfd creation failed, reader not started";
        return;
    }

    st.kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(st.kfd_fd < 0)
    {
        ROCP_WARNING << "KFD dispatch-log reader: /dev/kfd open failed, reader not started";
        ::close(st.wake_fd);
        st.wake_fd = -1;
        return;
    }

    st.stop.store(false, std::memory_order_release);

    internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
    st.thread = std::thread{reader_loop};
    internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    st.running = true;
    ROCP_INFO << "KFD dispatch-log reader: started";
}

void
stop_kfd_reader()
{
    stop_reader();
}

void
ensure_reader_session(uint32_t gpu_id)
{
    if(!kfd_dispatch_log_available() || !gpu_supports_dispatch_log(gpu_id)) return;

    auto& st = state();
    if(st.session_ready.load(std::memory_order_acquire)) return;  // fast path: already up

    auto lk = std::lock_guard<std::mutex>{st.setup_mu};
    if(st.session_ready.load(std::memory_order_relaxed))
        return;                // lost the race; someone set it up
    if(st.kfd_fd < 0) return;  // reader not started

    // Step 2 scope: one session for the first supported GPU. (This gpu_id is
    // supported per the guard above.)
    if(setup_session(st.kfd_fd, gpu_id, &st.session))
        st.session_ready.store(true, std::memory_order_release);
}
}  // namespace kfd
}  // namespace rocprofiler
