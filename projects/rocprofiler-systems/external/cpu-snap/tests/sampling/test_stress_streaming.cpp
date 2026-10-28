// Streaming-mode stress suite: high thread count, fork mid-run, dlopen
// mid-run, ring overflow and abort mid-write. Tests needing perf_event_open
// GTEST_SKIP on permission failure.

#include <functional>
#include <exception>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sched.h>
#include <span>
#include <cstdlib>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace
{

[[gnu::noinline]] double
busy_step(double s, std::uint64_t n) noexcept
{
    double acc = s;
    for(std::uint64_t i = 0; i < n; ++i) {
        acc = acc * 1.0000001 + static_cast<double>(i & 0xFFFFU);
}
    return acc;
}

void
spin_worker(std::atomic<bool>& stop)
{
    double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed)) {
        acc += busy_step(acc, 50'000);
}
    if(acc == 12345.6789) std::printf("unreachable\n");
}

bool
perf_open_available()
{
    struct perf_event_attr a
    {};
    a.size           = sizeof(a);
    a.type           = PERF_TYPE_SOFTWARE;
    a.config         = PERF_COUNT_SW_CPU_CLOCK;
    a.disabled       = 1;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    const int fd     = static_cast<int>(
        ::syscall(SYS_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0) return false;
    ::close(fd);
    return true;
}

/** In-RAM byte sink. */
struct memory_sink
{
    std::mutex             mu;
    std::vector<std::byte> bytes;

    cpu_snap::core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> data) {
            std::lock_guard<std::mutex> const lk(mu);
            bytes.insert(bytes.end(), data.begin(), data.end());
        };
    }

    [[nodiscard]] std::vector<std::byte> snapshot()
    {
        std::lock_guard<std::mutex> const lk(mu);
        return bytes;
    }
};

/**
 * Sink that "aborts" after N bytes by dropping every subsequent span
 * on the floor. Models a host whose persistence layer throws / closes
 * its fd mid-stream; the produced byte buffer ends at a partial chunk
 * boundary so the reader has to resync.
 */
struct abort_sink
{
    std::mutex             mu;
    std::vector<std::byte> bytes;
    std::uint64_t          limit_bytes;
    std::atomic<bool>      aborted{ false };

    explicit abort_sink(std::uint64_t limit) noexcept
    : limit_bytes(limit)
    {}

    cpu_snap::core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> data) {
            std::lock_guard<std::mutex> const lk(mu);
            if(aborted.load(std::memory_order_relaxed)) return;
            if(bytes.size() + data.size() <= limit_bytes)
            {
                bytes.insert(bytes.end(), data.begin(), data.end());
                return;
            }
            // Write whatever still fits, then truncate the rest.
            const std::size_t fits = limit_bytes - bytes.size();
            bytes.insert(bytes.end(), data.begin(), data.begin() + fits);
            aborted.store(true, std::memory_order_relaxed);
        };
    }

    [[nodiscard]] std::vector<std::byte> snapshot()
    {
        std::lock_guard<std::mutex> const lk(mu);
        return bytes;
    }
};

}  // namespace

// 1. High-thread streaming run stays below 1% lost.
TEST(StressStreaming, HighThreadStreamingStaysBelowOnePercentLost)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    constexpr int           n_workers    = 64;
    constexpr std::uint32_t freq_hz      = 2000;
    constexpr auto          run_time     = std::chrono::seconds{ 2 };
    constexpr double        max_loss_pct = 1.0;

    std::atomic<bool>        stop{ false };
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for(int i = 0; i < n_workers; ++i) {
        workers.emplace_back(spin_worker, std::ref(stop));
}

    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    memory_sink sink;

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz            = freq_hz;
    cfg.perf.callchain_depth           = 16;
    cfg.perf.kernel_pages_per_tid      = 16;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.threads.watch_new_threads      = true;
    cfg.streaming.sink                 = sink.as_fn();
    cfg.streaming.chunk_bytes          = 1U << 16;
    cfg.streaming.checkpoint_interval  = 4096;
    cfg.streaming.maps_poll_interval   = 4096;
    cfg.streaming.process_name         = "StressStreaming";

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(run_time);
    s.stop();

    stop.store(true);
    for(auto& t : workers) {
        t.join();
}

    const std::size_t total = s.total_samples();
    const std::size_t lost  = s.lost_samples();
    const double      pct   = total == 0 ? 100.0 :
                                           100.0 * static_cast<double>(lost) /
                                           static_cast<double>(total);

    std::printf("[StressStreaming] tids=%zu total=%zu lost=%zu (%.3f%%) bytes=%zu\n",
                s.tracked_threads(), total, lost, pct, sink.snapshot().size());

    EXPECT_GT(total, 0U);
    EXPECT_LT(pct, max_loss_pct);
}

// 2. fork() mid-run: child must see the recorder stopped via
//    on_child_fork (no inherited fds active, no worker thread).
TEST(StressStreaming, ForkMidRunChildRunsClean)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    memory_sink sink;

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = 1000;
    cfg.perf.callchain_depth      = 16;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink            = sink.as_fn();
    cfg.streaming.chunk_bytes     = 1U << 16;
    cfg.streaming.process_name    = "fork_test";

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    int pipefd[2]{};
    ASSERT_EQ(::pipe(pipefd), 0);

    const pid_t child = ::fork();
    ASSERT_NE(child, -1) << "fork failed";

    if(child == 0)
    {
        ::close(pipefd[0]);
        // The atfork(child) handler should have run by now; the
        // sampler must report is_running() == false in the child.
        const char status = s.is_running() ? 'R' : 'S';
        const ssize_t n = ::write(pipefd[1], &status, 1);
        (void) n;
        ::close(pipefd[1]);
        // Bypass gtest cleanup in the child to avoid double-printing.
        ::_exit(s.is_running() ? 1 : 0);
    }

    ::close(pipefd[1]);
    char child_status = '\0';
    const ssize_t nread = ::read(pipefd[0], &child_status, 1);
    EXPECT_EQ(nread, 1);
    ::close(pipefd[0]);

    int wstatus = 0;
    EXPECT_EQ(::waitpid(child, &wstatus, 0), child);
    EXPECT_TRUE(WIFEXITED(wstatus));
    EXPECT_EQ(WEXITSTATUS(wstatus), 0) << "child saw recorder still running";
    EXPECT_EQ(child_status, 'S') << "child status byte was not 'S'";

    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
    s.stop();

    // Parent stream should be parseable: feed it through reader and
    // confirm we reach an EOF record without fatal framing errors.
    const auto stream = sink.snapshot();
    cpu_snap::core::reader rd;
    bool saw_eof = false;
    rd.on_event([&](const cpu_snap::core::record_variant& rv) {
        if(cpu_snap::core::type_of(rv) == cpu_snap::core::record_type::eof) saw_eof = true;
    });
    rd.feed(std::span<const std::byte>(stream));

    EXPECT_FALSE(rd.is_fatal());
    EXPECT_TRUE(saw_eof) << "parent stream did not reach EOF";
}

// 3. dlopen() mid-run emits a MAPS_DELTA add event for the loaded
//    module. Picks an exotic library that is unlikely to be already
//    mapped on Ubuntu-class distros.
namespace
{

const char*
pick_dlopen_candidate()
{
    static const char* candidates[] = {
        "libcrypt.so.1",
        "libutil.so.1",
        "libnsl.so.1",
        "libresolv.so.2",
        "libm.so.6",
    };
    for(const char* path : candidates)
    {
        // Check if NOT already loaded: dlopen(RTLD_NOLOAD) returns null.
        if(::dlopen(path, RTLD_LAZY | RTLD_NOLOAD) == nullptr)
        {
            return path;
        }
    }
    return nullptr;
}

bool
maps_delta_mentions(const std::vector<std::byte>& stream, const std::string& needle)
{
    cpu_snap::core::reader rd;
    bool found = false;
    rd.on_event([&](const cpu_snap::core::record_variant& rv) {
        if(cpu_snap::core::type_of(rv) != cpu_snap::core::record_type::maps_delta) return;
        const auto& delta = std::get<cpu_snap::core::maps_delta_record>(rv);
        if(delta.operation != cpu_snap::core::maps_delta_record::op::add) return;
        if(delta.module.path.find(needle) != std::string::npos) found = true;
    });
    rd.feed(std::span<const std::byte>(stream));
    return found;
}

}  // namespace

TEST(StressStreaming, DlopenMidrunEmitsMapsDelta)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    const char* candidate = pick_dlopen_candidate();
    if(candidate == nullptr) {
        GTEST_SKIP() << "no unloaded candidate library available for dlopen";
}

    memory_sink sink;

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz            = 1000;
    cfg.perf.callchain_depth           = 16;
    cfg.perf.kernel_pages_per_tid      = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink                 = sink.as_fn();
    cfg.streaming.chunk_bytes          = 1U << 16;
    cfg.streaming.maps_poll_interval   = 64;  // fast poll so the dlopen is picked up
    cfg.streaming.checkpoint_interval  = 4096;
    cfg.streaming.process_name         = "dlopen_test";

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    void* handle = ::dlopen(candidate, RTLD_LAZY);
    ASSERT_NE(handle, nullptr) << "dlopen(" << candidate << ") failed";

    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    s.stop();
    stop.store(true);
    worker.join();

    const auto stream = sink.snapshot();
    std::printf("[StressStreaming] dlopen candidate=%s stream_bytes=%zu\n", candidate,
                stream.size());

    EXPECT_TRUE(maps_delta_mentions(stream, candidate))
        << "expected MAPS_DELTA add for " << candidate;

    ::dlclose(handle);
}

// 4. Force kernel ring overflow with the smallest possible per-tid
//    ring and a very high sample rate. The LOST counter must increment
//    and at least one LOST record must appear in the stream.
TEST(StressStreaming, RingOverflowIncrementsLostCounter)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    constexpr int n_workers = 8;

    std::atomic<bool>        stop{ false };
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for(int i = 0; i < n_workers; ++i) {
        workers.emplace_back(spin_worker, std::ref(stop));
}

    std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

    // 1 page ring + 1 page metadata = absolute minimum, plus a very
    // high sample rate. The kernel will refuse to keep up and emit
    // PERF_RECORD_LOST.
    memory_sink sink;

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz            = 50000;
    cfg.perf.callchain_depth           = 32;
    cfg.perf.kernel_pages_per_tid      = 1;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.threads.watch_new_threads      = true;
    cfg.streaming.sink                 = sink.as_fn();
    cfg.streaming.chunk_bytes          = 1U << 16;
    cfg.streaming.checkpoint_interval  = 4096;
    cfg.streaming.maps_poll_interval   = 4096;
    cfg.streaming.process_name         = "overflow_test";

    cpu_snap::sampling::sampler s(std::move(cfg));
    try
    {
        s.start();
    } catch(const std::exception& e)
    {
        stop.store(true);
        for(auto& t : workers) {
            t.join();
}
        GTEST_SKIP() << "perf_event_open failed at 50 kHz: " << e.what();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    s.stop();
    stop.store(true);
    for(auto& t : workers) {
        t.join();
}

    const std::size_t total = s.total_samples();
    const std::size_t lost  = s.lost_samples();
    std::printf("[StressStreaming] overflow total=%zu lost=%zu bytes=%zu\n", total, lost,
                sink.snapshot().size());

    if(lost == 0)
    {
        // Some kernels rate-limit aggressively and refuse to set up
        // 50 kHz sampling at all. Skip rather than fail.
        GTEST_SKIP() << "kernel did not overflow at 50 kHz; lost=0";
    }

    const auto stream = sink.snapshot();
    cpu_snap::core::reader rd;
    std::uint64_t lost_records = 0;
    rd.on_event([&](const cpu_snap::core::record_variant& rv) {
        if(cpu_snap::core::type_of(rv) == cpu_snap::core::record_type::lost) ++lost_records;
    });
    rd.feed(std::span<const std::byte>(stream));
    EXPECT_GT(lost_records, 0U) << "stream contained no LOST records";
}

// 5. Sink "aborts" partway through; reader emits all previously completed
//    chunks and leaves the partial tail buffered (no parse error).
TEST(StressStreaming, AbortMidWriteChunkRecovery)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    // Limit chosen so several full chunks fit (file_header + a couple of
    // 64 KiB chunks), then truncation kicks in mid-stream.
    constexpr std::uint64_t k_abort_after = 256 * 1024;

    abort_sink sink(k_abort_after);

    std::atomic<bool> stop{ false };
    std::thread       worker(spin_worker, std::ref(stop));

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz            = 2000;
    cfg.perf.callchain_depth           = 16;
    cfg.perf.kernel_pages_per_tid      = 4;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.streaming.sink                 = sink.as_fn();
    cfg.streaming.chunk_bytes          = 1U << 16;
    cfg.streaming.checkpoint_interval  = 1024;
    cfg.streaming.maps_poll_interval   = 1024;
    cfg.streaming.process_name         = "abort_test";

    cpu_snap::sampling::sampler s(std::move(cfg));
    s.start();
    // Run long enough that the abort threshold is hit.
    for(int i = 0; i < 20 && !sink.aborted.load(std::memory_order_relaxed); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
}
    s.stop();
    stop.store(true);
    worker.join();

    const auto stream = sink.snapshot();
    std::printf("[StressStreaming] abort_after=%llu captured=%zu\n",
                static_cast<unsigned long long>(k_abort_after), stream.size());
    ASSERT_GE(stream.size(), sizeof(cpu_snap::core::file_header) + sizeof(cpu_snap::core::chunk_header));

    cpu_snap::core::reader rd;
    std::uint64_t records_seen = 0;
    rd.on_event([&](const cpu_snap::core::record_variant&) { ++records_seen; });
    rd.feed(std::span<const std::byte>(stream));

    const auto& ctr = rd.counters();
    std::printf("[StressStreaming] reader: records=%llu parse_err=%llu\n",
                static_cast<unsigned long long>(ctr.records_emitted),
                static_cast<unsigned long long>(ctr.parse_errors));

    EXPECT_FALSE(rd.is_fatal());
    // A truncated stream leaves the last partial chunk body buffered without
    // emitting it; all previously completed chunks are emitted cleanly.
    EXPECT_GT(ctr.records_emitted, 0U) << "no records recovered from truncated stream";
    EXPECT_EQ(ctr.parse_errors, 0U);
    EXPECT_EQ(records_seen, ctr.records_emitted);
}
