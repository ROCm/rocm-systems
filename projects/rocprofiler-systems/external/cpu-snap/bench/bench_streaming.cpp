// bench_streaming.cpp
//
// Heavy-scale workload-overhead benchmark for the v1 streaming
// pipeline.
//
// Spawns N worker threads each running a deterministic compute loop
// with a known stack depth, drives a cpu_snap::sampling::sampler in
// streaming mode at the configured sample rate, and reports:
//
//   * wall-clock duration of the workload phase
//   * sum of CLOCK_THREAD_CPUTIME_ID across workers (on-CPU time lost
//     to sample IRQ + kernel callchain walk)
//   * PMU cycles + instructions on each worker (self-attached
//     perf_event_open group, same pattern as bench_overhead)
//   * sampler counters: total_samples, lost_samples, tracked_threads
//   * bytes written to the host sink
//
// Two modes
//   --short (default, SAMPLING_BENCH_FULL unset): 30 s at the heavy
//     scale (5 kHz x 256 threads). Extrapolates to the 1-hour target
//     via samples/sec * 3600 + bytes/sec * 3600 math; prints
//     PASS/FAIL on the workload-overhead and file-size budgets.
//   --full / SAMPLING_BENCH_FULL=1: runs the literal 1-hour soak.
//     Skipped by ctest's default run; reproduce locally with
//     SAMPLING_BENCH_FULL=1 bench_streaming.
//
// Usage
//   bench_streaming [--threads=N] [--freq=HZ] [--duration=S]
//                   [--depth=N] [--pages=N] [--null-sink] [--full]
//                   [--out=PATH] [--help]

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/uio.h>
#include <thread>
#include <utility>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

#include "bench_common.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace
{

using bench_common::hw_counters;
using bench_common::ns_diff;
using bench_common::open_self_counters;
using bench_common::read_counter;
using bench_common::workload_step;

// File sink. Opens once with O_APPEND | O_CREAT and writes spans via
// writev. The sink callback runs serialized on the worker thread, so
// no per-call locking is required for the fd; we still keep one tally
// (bytes_written) under an atomic so the main thread can read it.
class file_sink
{
public:
    file_sink() = default;

    bool open(const std::string& path)
    {
        m_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if(m_fd < 0)
        {
            std::fprintf(stderr, "bench_streaming: open(%s) failed: %s\n", path.c_str(),
                         std::strerror(errno));
            return false;
        }
        return true;
    }

    ~file_sink() { close(); }

    file_sink(const file_sink&)            = delete;
    file_sink& operator=(const file_sink&) = delete;
    file_sink(file_sink&&)                 = delete;
    file_sink& operator=(file_sink&&)      = delete;

    void close() noexcept
    {
        if(m_fd >= 0)
        {
            ::close(m_fd);
            m_fd = -1;
        }
    }

    [[nodiscard]] cpu_snap::core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> bytes) {
            if(m_fd < 0 || bytes.empty()) return;
            iovec const io{ const_cast<std::byte*>(bytes.data()), bytes.size() };
            ssize_t const n = ::writev(m_fd, &io, 1);
            if(n > 0) {
                m_bytes_written.fetch_add(static_cast<std::uint64_t>(n),
                                          std::memory_order_relaxed);
}
        };
    }

    [[nodiscard]] std::uint64_t bytes_written() const noexcept
    {
        return m_bytes_written.load(std::memory_order_relaxed);
    }

private:
    int                        m_fd{ -1 };
    std::atomic<std::uint64_t> m_bytes_written{ 0 };
};

/** Sink that counts but discards bytes; used for overhead-only runs. */
class null_sink
{
public:
    [[nodiscard]] cpu_snap::core::sink_fn as_fn()
    {
        return [this](std::span<const std::byte> bytes) {
            m_bytes_written.fetch_add(bytes.size(), std::memory_order_relaxed);
        };
    }
    [[nodiscard]] std::uint64_t bytes_written() const noexcept
    {
        return m_bytes_written.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::uint64_t> m_bytes_written{ 0 };
};

// Per-worker aggregates returned from each thread.
struct worker_result
{
    double        cputime_ns = 0.0;
    std::uint64_t cycles     = 0;
    std::uint64_t insns      = 0;
    double        sink       = 0.0;
};

void
worker_main(std::atomic<bool>& stop, std::uint64_t inner, worker_result& out)
{
    const hw_counters hc = open_self_counters();
    timespec          t_cpu_0{};
    ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_cpu_0);
    if(hc.ok)
    {
        ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    double acc = 1.0;
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += workload_step(acc, inner);
    }
    asm volatile("" ::"r"(acc) : "memory");
    out.sink = acc;

    if(hc.ok) ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    timespec t_cpu_1{};
    ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_cpu_1);

    out.cputime_ns = ns_diff(t_cpu_1, t_cpu_0);
    if(hc.ok)
    {
        out.cycles = read_counter(hc.fd_cycles);
        out.insns  = read_counter(hc.fd_insns);
        ::close(hc.fd_insns);
        ::close(hc.fd_cycles);
    }
}

// Aggregated result of one run (one (freq, duration, threads) tuple).
struct run_summary
{
    std::uint32_t freq_hz          = 0;
    int           threads          = 0;
    double        wall_s           = 0.0;
    double        cputime_total_s  = 0.0;
    std::uint64_t cycles_total     = 0;
    std::uint64_t insns_total      = 0;
    std::uint64_t samples_total    = 0;
    std::uint64_t samples_lost     = 0;
    std::uint64_t bytes_written    = 0;
    std::size_t   tracked_threads  = 0;
};

run_summary
one_run(std::uint32_t freq_hz, int n_threads, double duration_s, std::uint64_t inner,
        std::uint32_t depth, std::uint32_t pages, cpu_snap::core::sink_fn sink,
        std::uint64_t (*bytes_fn)())
{
    std::atomic<bool>             stop{ false };
    std::vector<worker_result>    results(static_cast<std::size_t>(n_threads));
    std::vector<std::thread>      workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for(int i = 0; i < n_threads; ++i) {
        workers.emplace_back(worker_main, std::ref(stop), inner, std::ref(results[i]));
}

    // Let workers warm up + come on CPU before starting the sampler.
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    std::unique_ptr<cpu_snap::sampling::sampler> rec;
    if(freq_hz > 0)
    {
        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = freq_hz,
                           .callchain_depth      = depth,
                           .kernel_pages_per_tid = pages,
                           .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
            .threads   = { .watch_new_threads   = true,
                           .include_main_thread = false },
            .streaming = { .sink                = std::move(sink),
                           .chunk_bytes         = 1U << 20,  // 1 MiB chunks per design
                           .maps_poll_interval  = 65536,
                           .checkpoint_interval = 65536,
                           .process_name        = "bench_streaming" },
        };
        rec = std::make_unique<cpu_snap::sampling::sampler>(std::move(cfg));
        rec->start();
    }

    timespec t_wall_0{};
    ::clock_gettime(CLOCK_MONOTONIC, &t_wall_0);

    std::this_thread::sleep_for(std::chrono::duration<double>(duration_s));

    stop.store(true, std::memory_order_relaxed);
    for(auto& t : workers) {
        t.join();
}

    timespec t_wall_1{};
    ::clock_gettime(CLOCK_MONOTONIC, &t_wall_1);

    run_summary s{};
    s.freq_hz = freq_hz;
    s.threads = n_threads;
    s.wall_s  = ns_diff(t_wall_1, t_wall_0) / 1e9;
    for(const auto& r : results)
    {
        s.cputime_total_s += r.cputime_ns / 1e9;
        s.cycles_total += r.cycles;
        s.insns_total += r.insns;
    }
    if(rec)
    {
        rec->stop();
        s.samples_total   = rec->total_samples();
        s.samples_lost    = rec->lost_samples();
        s.tracked_threads = rec->tracked_threads();
    }
    s.bytes_written = bytes_fn ? bytes_fn() : 0;
    return s;
}

// CLI parsing.
struct cli
{
    int           threads     = 256;
    std::uint32_t freq_hz     = 5000;
    double        duration_s  = 30.0;
    std::uint64_t inner       = 200'000;
    std::uint32_t depth       = 32;
    std::uint32_t pages       = 8;
    bool          null_sink   = false;
    bool          full        = false;
    std::string   out_path    = "/tmp/callstack_bench_streaming.cs";
    bool          show_help   = false;
};

void
print_help(const char* prog)
{
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "Heavy-scale streaming-pipeline benchmark. Prints a one-line\n"
        "table per run + PASS/FAIL on the workload-overhead and file\n"
        "size budgets from architecture-plan.md §M13.\n"
        "\n"
        "Options:\n"
        "  --threads=N      worker thread count           (default 256)\n"
        "  --freq=HZ        sample rate per thread        (default 5000)\n"
        "  --duration=S     workload duration in seconds  (default 30, full 3600)\n"
        "  --depth=N        callchain depth               (default 32)\n"
        "  --pages=N        kernel_pages_per_tid          (default 8)\n"
        "  --null-sink      drop bytes on the floor instead of writing to a file\n"
        "                   (use this for pure overhead measurement)\n"
        "  --out=PATH       output file when not using --null-sink\n"
        "                   (default /tmp/callstack_bench_streaming.cs)\n"
        "  --full           run the 1-hour soak (acceptance criterion 1).\n"
        "                   Equivalent to SAMPLING_BENCH_FULL=1 in the env.\n"
        "  --help, -h       this message\n"
        "\n"
        "Reproduce the 1-hour soak from the architecture plan:\n"
        "  SAMPLING_BENCH_FULL=1 %s --threads=256 --freq=5000\n",
        prog, prog);
}

cli
parse_cli(int argc, char** argv)
{
    cli c;
    for(int i = 1; i < argc; ++i)
    {
        std::string_view a(argv[i]);
        auto             eat = [&](std::string_view key) -> std::string_view {
            if(a.starts_with(key)) return a.substr(key.size());
            return {};
        };
        if(a == "--help" || a == "-h") { c.show_help = true; continue; }
        if(a == "--null-sink")         { c.null_sink = true; continue; }
        if(a == "--full")              { c.full = true; continue; }
        if(auto v = eat("--threads=");  !v.empty()) { c.threads    = std::stoi(std::string(v));
        } else if(auto v = eat("--freq=");      !v.empty()) { c.freq_hz    = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if(auto v = eat("--duration=");  !v.empty()) { c.duration_s = std::stod(std::string(v));
        } else if(auto v = eat("--depth=");     !v.empty()) { c.depth      = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if(auto v = eat("--pages=");     !v.empty()) { c.pages      = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if(auto v = eat("--out=");       !v.empty()) { c.out_path   = std::string(v);
}
    }
    if(const char* env = std::getenv("SAMPLING_BENCH_FULL");
       env && env[0] != '\0' && env[0] != '0') {
        c.full = true;
}
    if(c.full && c.duration_s < 3600.0) c.duration_s = 3600.0;
    return c;
}

// Acceptance budgets. Short mode extrapolates: the 500 MB ceiling
// applies to the full 1-hour stream, so a 30 s run is permitted up to
// 500 MB * 30/3600.
constexpr double        k_full_soak_seconds   = 3600.0;
constexpr std::uint64_t k_full_size_ceiling   = 500ULL * 1024ULL * 1024ULL;
constexpr double        k_max_workload_ovh    = 5.0;

void
print_table_header()
{
    std::printf("\n%-8s %-7s %-9s %-9s %-9s %-12s %-9s %-9s %-10s %-7s\n",
                "freq_hz", "thr", "wall_s", "cpu_s", "ovh%", "samples", "lost", "trk",
                "bytes", "MB");
}

void
print_run_row(const run_summary& s, const run_summary& baseline)
{
    const double baseline_cpu = baseline.cputime_total_s > 0.0
                                    ? baseline.cputime_total_s
                                    : 1.0;
    const double ovh_pct = (s.cputime_total_s / baseline_cpu - 1.0) * 100.0;
    std::printf("%-8u %-7d %-9.3f %-9.3f %+8.2f  %-12llu %-9llu %-9zu %-10llu %-7.2f\n",
                s.freq_hz, s.threads, s.wall_s, s.cputime_total_s, ovh_pct,
                static_cast<unsigned long long>(s.samples_total),
                static_cast<unsigned long long>(s.samples_lost),
                s.tracked_threads,
                static_cast<unsigned long long>(s.bytes_written),
                static_cast<double>(s.bytes_written) / (1024.0 * 1024.0));
}

}  // namespace

int
main(int argc, char** argv)
{
    try
    {
    const cli c = parse_cli(argc, argv);
    if(c.show_help)
    {
        print_help(argv[0]);
        return 0;
    }

    std::printf("# bench_streaming\n");
    std::printf("# threads=%d freq=%u Hz duration=%.1f s depth=%u pages=%u %s\n", c.threads,
                c.freq_hz, c.duration_s, c.depth, c.pages,
                c.null_sink ? "(null-sink)" : c.out_path.c_str());
    std::printf("# mode=%s soak_ceiling=%llu bytes\n", c.full ? "FULL_SOAK" : "SHORT",
                static_cast<unsigned long long>(k_full_size_ceiling));

    print_table_header();

    // Baseline: sampler OFF (freq=0). Same threads + workload so the
    // overhead column means "extra on-CPU time the sampler caused".
    file_sink const baseline_file;
    null_sink baseline_null;
    cpu_snap::core::sink_fn baseline_sink = baseline_null.as_fn();
    auto baseline_bytes_fn = +[]() -> std::uint64_t { return 0; };
    const run_summary baseline =
        one_run(0, c.threads, c.duration_s, c.inner, c.depth, c.pages,
                std::move(baseline_sink), baseline_bytes_fn);
    print_run_row(baseline, baseline);

    // Streamed run. File or null sink based on flag.
    file_sink streamed_file;
    null_sink streamed_null;
    cpu_snap::core::sink_fn streamed_sink;
    std::uint64_t (*streamed_bytes_fn)() = nullptr;
    static file_sink* g_file_sink = nullptr;
    static null_sink* g_null_sink = nullptr;
    if(c.null_sink)
    {
        g_null_sink       = &streamed_null;
        streamed_sink     = streamed_null.as_fn();
        streamed_bytes_fn = +[]() -> std::uint64_t {
            return g_null_sink ? g_null_sink->bytes_written() : 0;
        };
    }
    else
    {
        if(!streamed_file.open(c.out_path)) return 2;
        g_file_sink       = &streamed_file;
        streamed_sink     = streamed_file.as_fn();
        streamed_bytes_fn = +[]() -> std::uint64_t {
            return g_file_sink ? g_file_sink->bytes_written() : 0;
        };
    }

    const run_summary streamed =
        one_run(c.freq_hz, c.threads, c.duration_s, c.inner, c.depth, c.pages,
                std::move(streamed_sink), streamed_bytes_fn);
    print_run_row(streamed, baseline);

    // Acceptance evaluation. Overhead is measured against baseline
    // on-CPU time (wall-time differs little between runs because all
    // threads stay busy; on-CPU time is the per-thread tax).
    const double baseline_cpu = baseline.cputime_total_s > 0.0
                                    ? baseline.cputime_total_s
                                    : 1.0;
    const double overhead_pct = (streamed.cputime_total_s / baseline_cpu - 1.0) * 100.0;

    const double scale = (c.duration_s > 0.0)
                             ? (k_full_soak_seconds / c.duration_s)
                             : 1.0;
    const auto projected_bytes = static_cast<std::uint64_t>(
        static_cast<double>(streamed.bytes_written) * scale);
    const std::uint64_t allowed_bytes = c.full
                                            ? k_full_size_ceiling
                                            : static_cast<std::uint64_t>(
                                                  static_cast<double>(k_full_size_ceiling) /
                                                  scale);

    const bool ok_overhead = overhead_pct < k_max_workload_ovh;
    const bool ok_size     = streamed.bytes_written < allowed_bytes;

    std::printf("\n# acceptance (architecture-plan.md §M13 / criterion 1)\n");
    std::printf("workload_overhead_pct = %+0.2f  ceiling = %.2f  => %s\n", overhead_pct,
                k_max_workload_ovh, ok_overhead ? "PASS" : "FAIL");
    std::printf("bytes_written         = %llu (%.2f MB)\n",
                static_cast<unsigned long long>(streamed.bytes_written),
                static_cast<double>(streamed.bytes_written) / (1024.0 * 1024.0));
    std::printf("projected_1hr_bytes   = %llu (%.2f MB)  ceiling = %llu (%.2f MB) => %s\n",
                static_cast<unsigned long long>(projected_bytes),
                static_cast<double>(projected_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(k_full_size_ceiling),
                static_cast<double>(k_full_size_ceiling) / (1024.0 * 1024.0),
                ok_size ? "PASS" : "FAIL");
    std::printf("samples_per_sec       = %.0f  lost/total = %.4f%%\n",
                streamed.wall_s > 0.0
                    ? static_cast<double>(streamed.samples_total) / streamed.wall_s
                    : 0.0,
                streamed.samples_total > 0
                    ? 100.0 * static_cast<double>(streamed.samples_lost) /
                          static_cast<double>(streamed.samples_total)
                    : 0.0);
    if(!c.full)
    {
        std::printf("# short mode; reproduce full-soak with: SAMPLING_BENCH_FULL=1 %s\n",
                    argv[0]);
    }

    return (ok_overhead && ok_size) ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n", argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::fprintf(stderr, "%s: unknown exception\n", argc > 0 ? argv[0] : "(unknown)");
        return EXIT_FAILURE;
    }
}
