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
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"

// Active (v5) dispatch-log profiler ABI. Must be the ONLY kfd ioctl header in
// this translation unit (it conflicts with lib/rocprofiler-sdk/details/kfd_ioctl.h).
#include "lib/rocprofiler-sdk/kfd/kfd_dlog_uapi.h"

#include <fmt/core.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
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
constexpr uint32_t kFwRecBytes   = KFD_DISPATCH_LOG_FW_RECORD_BYTES;  // 20

// Firmware record type values (from the dispatch_log_format .ksy enum).
constexpr uint32_t kRecPadding = 0;
constexpr uint32_t kRecStart   = 1;  // dispatch_start
constexpr uint32_t kRecEop     = 2;  // end-of-pipe (completion)

// The 20-byte firmware record, laid out per the dispatch_log_format descriptor.
// Not a UAPI type; overlaid onto the mmap'd ring bytes.
struct fw_record
{
    uint32_t ts_lo;         // bytes 0-3:   low 32 bits of GPU timestamp
    uint32_t ts_hi;         // bytes 4-7:   high 32 bits
    uint32_t record_type;   // bytes 8-11:  0 padding, 1 dispatch_start, 2 eop
    uint32_t dispatch_id;   // bytes 12-15: low 32 bits of HSA queue write index
    uint32_t doorbell_off;  // bytes 16-19: queue identity (demux key)
};
static_assert(sizeof(fw_record) == KFD_DISPATCH_LOG_FW_RECORD_BYTES,
              "fw_record must match the 20-byte firmware record layout");

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

    kfd_dlog_stream_info info      = {};
    uint64_t             rptr[8]   = {};     // consumer read pos per region (up to 8)
    bool                 rptr_init = false;  // sync rptr to wptr on first drain

    // dispatch_start records awaiting their matching eop, keyed by
    // (doorbell_off << 32 | dispatch_id) -> start GPU ticks. Touched only by the
    // reader thread, so no lock is needed.
    // TODO: can grow unbounded if starts never receive a matching eop (queue dies
    // mid-dispatch, ring overwrite). Age out stale entries periodically, like
    // ResultsMap::evict_stale.
    struct pending_start
    {
        uint64_t start_ticks = 0;  // GPU ticks from the dispatch_start record
        uint64_t seen_at_ns  = 0;  // host CLOCK_BOOTTIME when recorded, for aging
    };
    std::unordered_map<uint64_t, pending_start> pending_starts = {};
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
bool
setup_session(int kfd, uint32_t gpu_id, dlog_session* s)
{
    s->gpu_id = gpu_id;

    const uint64_t buf_bytes  = static_cast<uint64_t>(kBufferKb) * 1024u;
    const uint64_t arr_bytes  = ((8ull * sizeof(uint64_t)) + 7) & ~7ull;  // upper bound
    const uint64_t signal_off = buf_bytes + arr_bytes * 2;
    s->alloc_size             = round_up_page(static_cast<size_t>(signal_off + arr_bytes));

    uint64_t gpuvm_base = 0, gpuvm_limit = 0;
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
    for(auto it = s->pending_starts.begin(); it != s->pending_starts.end();)
    {
        if(now_ns > it->second.seen_at_ns && now_ns - it->second.seen_at_ns > max_age_ns)
            it = s->pending_starts.erase(it);
        else
            ++it;
    }
}

// Drain new firmware records from every region: parse each 20-byte record, and
// pair dispatch_start with its matching eop by (doorbell_off, dispatch_id). A
// completed pair yields (start_ticks, end_ticks) for that dispatch. Returns the
// number of completed pairs observed this call. (Depositing pairs into the
// ResultsMap requires the doorbell generation and lands in step 5.)
uint64_t
drain_records(dlog_session* s)
{
    if(s->smap == MAP_FAILED) return 0;

    auto*          base     = static_cast<uint8_t*>(s->smap);
    auto*          wptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.wptr_offset);
    auto*          rptr_arr = reinterpret_cast<volatile uint64_t*>(base + s->info.rptr_offset);
    const uint32_t nreg     = s->info.num_regions;
    const uint32_t slots    = s->info.region_record_count;

    // On the first drain, sync each region's read cursor to the current write
    // pointer so we do not replay records that predate the reader (mirrors the
    // reference dmabuf_drain_init).
    if(!s->rptr_init)
    {
        for(uint32_t r = 0; r < nreg && r < 8; ++r)
        {
            uint64_t w = __atomic_load_n(&wptr_arr[r], __ATOMIC_ACQUIRE);
            s->rptr[r] = w;
            __atomic_store_n(&rptr_arr[r], w, __ATOMIC_RELEASE);
        }
        s->rptr_init = true;
        return 0;
    }

    uint64_t seen       = 0;
    uint64_t dbg_starts = 0, dbg_eops = 0, dbg_unmatched_eop = 0;
    for(uint32_t r = 0; r < nreg && r < 8; ++r)
    {
        uint64_t w    = __atomic_load_n(&wptr_arr[r], __ATOMIC_ACQUIRE);
        uint64_t scan = s->rptr[r];

        // Nothing new (or a spurious backwards wptr): skip. This guard is what
        // prevents an unbounded loop when w <= scan.
        if(w <= scan) continue;

        // Overrun: firmware advanced more than the ring holds. Recover to
        // w - slots + 1 (the +1 keeps us strictly behind the producer on a
        // power-of-two ring, where w - slots aliases the slot w itself).
        if(w - scan > slots) scan = w - slots + 1;

        while(scan != w)
        {
            uint64_t       slot      = scan & (slots - 1);
            const uint8_t* rec_bytes = base + s->info.records_offset +
                                       (static_cast<uint64_t>(r) * slots + slot) * kFwRecBytes;
            auto rec = fw_record{};
            std::memcpy(&rec, rec_bytes, sizeof(rec));
            ++scan;

            // Padding / empty slot: skip (per the .ksy consumer rules).
            if(rec.record_type == kRecPadding || rec.doorbell_off == 0) continue;

            const uint64_t ts =
                static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
            const uint64_t key = (static_cast<uint64_t>(rec.doorbell_off) << 32) |
                                 static_cast<uint64_t>(rec.dispatch_id);

            // Empirical doorbell binding: the first record for an unbound doorbell
            // binds it to the queue that enqueued this dispatch_id (recorded by
            // capture). Cheap no-op once bound.
            doorbell_map().bind_from_record(rec.doorbell_off, rec.dispatch_id);

            if(rec.record_type == kRecStart)
            {
                // MUST overwrite (not emplace): dispatch_id is only the low 32 bits
                // of the write index, so a key can recur (32-bit wrap or doorbell
                // reuse). A collision means the prior start is stale -- its eop was
                // lost -- and this new start is the live one. Emplace/first-wins
                // would later pair this key's eop with the STALE start_ticks,
                // producing a garbage (cross-dispatch) duration.
                s->pending_starts[key] = dlog_session::pending_start{ts, common::timestamp_ns()};
                ++dbg_starts;
            }
            else if(rec.record_type == kRecEop)
            {
                auto it = s->pending_starts.find(key);
                if(it != s->pending_starts.end())
                {
                    // Completed pair: (start_ticks, end_ticks) for this dispatch.
                    // start ticks = it->second, end ticks = ts.
                    // (Deposit into ResultsMap happens in step 5, once the
                    // doorbell generation is available for the correlation_key.)
                    uint64_t start_ticks = it->second.start_ticks;
                    s->pending_starts.erase(it);
                    ++seen;
                    ++dbg_eops;

                    // Deposit the paired raw ticks into the ResultsMap keyed by the
                    // full correlation_key. get_dispatch_time() takes it at
                    // completion. Generation comes from the (now-bound) doorbell.
                    uint32_t gen = doorbell_map().get_generation(rec.doorbell_off);
                    results_map().deposit(
                        correlation_key{rec.doorbell_off, rec.dispatch_id, gen},
                        kfd_timing_result{start_ticks, ts, common::timestamp_ns()});
                    ROCP_TRACE << fmt::format(
                        "KFD dlog PAIR: doorbell={} dispatch_id={} start_ticks={} "
                        "end_ticks={} dur_ticks={}",
                        rec.doorbell_off,
                        rec.dispatch_id,
                        start_ticks,
                        ts,
                        (ts > start_ticks ? ts - start_ticks : 0));
                }
                else
                    ++dbg_unmatched_eop;
                // eop with no matching start (ring wrap / start predates us):
                // drop it; that dispatch will fall back to HSA timestamps.
            }
        }

        s->rptr[r] = scan;
        __atomic_store_n(&rptr_arr[r], scan, __ATOMIC_RELEASE);
    }
    if(dbg_starts || dbg_eops || dbg_unmatched_eop)
        ROCP_TRACE << fmt::format(
            "KFD dispatch-log drain: starts={} paired_eops={} unmatched_eops={} pending={}",
            dbg_starts,
            dbg_eops,
            dbg_unmatched_eop,
            s->pending_starts.size());
    return seen;
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

    // Reader thread has exited: safe to tear down the session and fds here.
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

    // Age out unpaired starts at most this often (not every 1 ms poll).
    constexpr uint64_t kEvictIntervalNs = 1'000'000'000ull;  // 1 s
    constexpr uint64_t kStartMaxAgeNs   = 5'000'000'000ull;  // 5 s

    while(!st.stop.load(std::memory_order_acquire))
    {
        int rc = ::poll(&wake, 1, 1 /* ms */);
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
            total_seen += drain_records(&st.session);

            uint64_t now_ns = common::timestamp_ns();
            if(now_ns - last_evict_ns >= kEvictIntervalNs)
            {
                evict_stale_starts(&st.session, now_ns, kStartMaxAgeNs);
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
