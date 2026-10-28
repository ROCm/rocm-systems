// Stress test: many worker threads, high sample frequency. Asserts
// that PERF_RECORD_LOST stays below a small fraction of total
// captured samples. Skipped when perf_event_open is not permitted.

#include <cstdio>
#include <functional>
#include <cstddef>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace
{

[[gnu::noinline]] double
busy(double s, std::uint64_t n) noexcept
{
    double acc = s;
    for(std::uint64_t i = 0; i < n; ++i) {
        acc = acc * 1.0000001 + static_cast<double>(i & 0xFFFF);
}
    return acc;
}

void
worker(std::atomic<bool>& stop)
{
    double acc = 0.0;
    while(!stop.load(std::memory_order_relaxed)) {
        acc += busy(acc, 50'000);
}
    if(acc == 12345.6789) std::printf("unreachable\n");
}

bool
perf_open_available()
{
    perf_event_attr a{};
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

}  // namespace

TEST(PerfSamplerStress, ManyThreadsHighFreqLowDrops)
{
    if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";

    constexpr int           n_workers    = 32;
    constexpr std::uint32_t freq_hz      = 2000;
    constexpr auto          run_time     = std::chrono::seconds{ 2 };
    constexpr double        max_loss_pct = 1.0;  // <1 %

    std::atomic<bool>        stop{ false };
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for(int i = 0; i < n_workers; ++i) {
        workers.emplace_back(worker, std::ref(stop));
}

    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    constexpr std::size_t k_sink_max_bytes = 256U * 1024U * 1024U;
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = freq_hz;
    cfg.perf.callchain_depth      = 16;
    cfg.perf.kernel_pages_per_tid = 16;
    cfg.perf.unwind = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.threads.watch_new_threads = true;
    cfg.streaming.sink            = std::ref(sink);

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
    const double      pct =
        total == 0 ? 100.0
                        : 100.0 * static_cast<double>(lost) / static_cast<double>(total);

    std::printf("[stress] tids=%zu, total=%zu, lost=%zu (%.3f%%)\n", s.tracked_threads(),
                total, lost, pct);

    EXPECT_GT(total, 0U);
    EXPECT_LT(pct, max_loss_pct);
}
