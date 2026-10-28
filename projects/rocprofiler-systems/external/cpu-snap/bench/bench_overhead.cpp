// bench_overhead.cpp
//
// Measure the run-time overhead cpu_snap::sampling::sampler imposes
// on application code. The "overhead" we care about is what the
// target thread loses to the kernel sample IRQ + kernel-side stack
// walk + ring-buffer write. Postprocessing is intentionally NOT
// measured: it runs after the app and does not steal app CPU time.
//
// Methodology
//   Run a fixed-iteration workload (a nested call chain with no I/O,
//   no syscalls, no allocations) and time it with:
//     CLOCK_MONOTONIC          wall-clock duration
//     CLOCK_THREAD_CPUTIME_ID  on-CPU time for the worker thread
//     PMU cycles + instructions, via perf_event_open on the worker
//   First do a warmup pass, then a baseline pass with sampler OFF.
//   Then sweep sample_freq_hz across a few values, repeating the
//   same workload, printing absolute numbers and overhead vs
//   baseline.
//
// Usage
//   bench_overhead [--iters=N] [--inner=N] [--runs=N] [--warmup=N]
//                  [--depth=N] [--pages=N] [--stack-bytes=N] [--freqs=1000,5000,10000]
//                  [--backend=perf|signal|both] [--unwind=fp|dwarf|both]
//
// For each freq the harness measures a baseline (no sampler) plus one
// pass per selected backend, and reports each backend's wall time,
// overhead vs baseline, sample counts, and overhead normalized per
// 1000 captured samples (so the two backends compare even when they
// collect slightly different sample totals).

#include <bits/time.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

#include "bench_common.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"

namespace {

using bench_common::hw_counters;
using bench_common::ns_diff;
using bench_common::open_self_counters;
using bench_common::read_counter;
using bench_common::workload_step;

/**
 * @brief Which capture backend a measured pass exercises.
 *
 * baseline runs the workload with no sampler at all (the reference point for
 * overhead). perf and signal each build a sampler::config selecting the matching
 * backend_kind.
 */
enum class backend_choice : std::uint8_t
{
    baseline,
    perf,
    signal,
};

// Single timed run.
struct run_result
{
    double wall_ns      = 0.0;
    double cputime_ns   = 0.0;
    std::uint64_t cycles = 0;
    std::uint64_t insns  = 0;
    std::size_t   samples_captured = 0;
    std::size_t   total_samples    = 0;
    std::size_t   lost_samples     = 0;
};

run_result one_run(backend_choice backend,
                   cpu_snap::sampling::unwind_mode unwind,
                   std::uint64_t iters,
                   std::uint64_t inner,
                   std::uint32_t sample_freq_hz,
                   std::uint32_t callchain_depth,
                   std::uint32_t kernel_pages,
                   std::uint32_t stack_bytes)
{
    using cpu_snap::sampling::backend_kind;

    // Sink must outlive the recorder (recorder dtor drains in-flight
    // callbacks). Bounded generously: these are short bench runs, so this
    // is headroom, not a real limit.
    constexpr std::size_t k_bench_sink_max_bytes = 512U * 1024U * 1024U;
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_bench_sink_max_bytes });

    // Open recorder unless this is the baseline (no sampler) pass.
    std::unique_ptr<cpu_snap::sampling::sampler> rec;
    if (backend != backend_choice::baseline && sample_freq_hz > 0) {
        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = sample_freq_hz,
                           .callchain_depth      = callchain_depth,
                           .kernel_pages_per_tid = kernel_pages,
                           .dwarf_stack_bytes    = stack_bytes,
                           .unwind               = unwind },
            .threads   = { .watch_new_threads   = false,  // benchmark only this thread
                           .include_main_thread = true },
            .streaming = { .sink = std::ref(sink), .process_name = "bench_overhead" },
            .backend   = backend == backend_choice::signal ? backend_kind::signal
                                                           : backend_kind::perf,
            .signal    = { .sample_freq_hz       = sample_freq_hz,
                           // SIGRTMIN + 4: default RT offset, avoids SIGPROF clash.
                           .signal_rt_offset     = 4,
                           // Same user-stack snapshot size perf uses in dwarf mode.
                           .stack_snapshot_bytes = stack_bytes,
                           // Headroom for the whole run so the ring never overflows:
                           // freq * 4, floored so short low-freq runs still fit.
                           .ring_capacity        = sample_freq_hz * 4 < 1024
                                                       ? 1024 : sample_freq_hz * 4,
                           .start_delay_ns       = 0 },
        };
        rec = std::make_unique<cpu_snap::sampling::sampler>(std::move(cfg));
        rec->start();
    }

    // HW counters on self (this thread).
    const hw_counters hc = open_self_counters();

    timespec t_wall_0{};
    timespec t_cpu_0{};
    ::clock_gettime(CLOCK_MONOTONIC,         &t_wall_0);
    ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_cpu_0);
    if (hc.ok) {
        ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    // ----- workload -----
    double acc = 1.0;
    for (std::uint64_t k = 0; k < iters; ++k) {
        acc += workload_step(acc, inner);
    }
    // Prevent the optimizer from eliding the result.
    asm volatile("" :: "r"(acc) : "memory");

    if (hc.ok) {
        ::ioctl(hc.fd_cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
    timespec t_wall_1{};
    timespec t_cpu_1{};
    ::clock_gettime(CLOCK_MONOTONIC,         &t_wall_1);
    ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t_cpu_1);

    run_result r{};
    r.wall_ns    = ns_diff(t_wall_1, t_wall_0);
    r.cputime_ns = ns_diff(t_cpu_1,  t_cpu_0);
    if (hc.ok) {
        r.cycles = read_counter(hc.fd_cycles);
        r.insns  = read_counter(hc.fd_insns);
        ::close(hc.fd_insns);
        ::close(hc.fd_cycles);
    }
    if (rec) {
        rec->stop();
        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.bytes());
        r.samples_captured = 0;
        for (const auto& [tid, vec] : reconstructor.samples()) r.samples_captured += vec.size();
        r.total_samples = rec->total_samples();
        r.lost_samples  = rec->lost_samples();
    }
    return r;
}

/**
 * @brief Which backends the sweep measures.
 *
 * perf / signal measure that one backend; both measures each in turn against a
 * shared baseline so the two columns are directly comparable.
 */
enum class backend_mode : std::uint8_t
{
    perf,
    signal,
    both,
};

// Argument parsing.
struct cli
{
    std::uint64_t iters  = 80;        // outer reps of workload_step
    std::uint64_t inner  = 200'000;   // leaf iters per call
    int           runs   = 5;
    int           warmup = 2;
    std::uint32_t depth  = 32;
    std::uint32_t pages  = 0;  // 0 = sized from the DWARF snapshot
    std::uint32_t stack_bytes = cpu_snap::sampling::sampler::perf_attr_options{}.dwarf_stack_bytes;
    backend_mode  backend = backend_mode::both;
    std::vector<cpu_snap::sampling::unwind_mode> unwinds = {
        cpu_snap::sampling::unwind_mode::frame_pointer,
        cpu_snap::sampling::unwind_mode::dwarf,
    };
    std::vector<std::uint32_t> freqs = {1000, 5000, 10000};
};

std::vector<std::uint32_t> parse_csv(std::string_view s)
{
    std::vector<std::uint32_t> out;
    while (!s.empty()) {
        const auto comma = s.find(',');
        const auto tok = s.substr(0, comma);
        if (!tok.empty()) {
            out.push_back(static_cast<std::uint32_t>(
                std::stoul(std::string(tok))));
}
        if (comma == std::string_view::npos) break;
        s.remove_prefix(comma + 1);
    }
    return out;
}

cli parse_cli(int argc, char** argv)
{
    cli c;
    for (int i = 1; i < argc; ++i) {
        std::string_view a(argv[i]);
        auto eat = [&](std::string_view key) -> std::string_view {
            if (a.starts_with(key)) return a.substr(key.size());
            return {};
        };
        if (auto v = eat("--iters=");  !v.empty()) { c.iters  = std::stoull(std::string(v));
        } else if (auto v = eat("--inner=");  !v.empty()) { c.inner  = std::stoull(std::string(v));
        } else if (auto v = eat("--runs=");   !v.empty()) { c.runs   = std::stoi(std::string(v));
        } else if (auto v = eat("--warmup="); !v.empty()) { c.warmup = std::stoi(std::string(v));
        } else if (auto v = eat("--depth=");  !v.empty()) { c.depth  = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if (auto v = eat("--stack-bytes="); !v.empty()) { c.stack_bytes = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if (auto v = eat("--pages=");  !v.empty()) { c.pages  = static_cast<std::uint32_t>(std::stoul(std::string(v)));
        } else if (auto v = eat("--freqs=");  !v.empty()) { c.freqs  = parse_csv(v);
        } else if (auto v = eat("--unwind="); !v.empty()) {
            using cpu_snap::sampling::unwind_mode;
            if (v == "fp")         { c.unwinds = { unwind_mode::frame_pointer }; }
            else if (v == "dwarf") { c.unwinds = { unwind_mode::dwarf }; }
            else if (v == "both")  { c.unwinds = { unwind_mode::frame_pointer, unwind_mode::dwarf }; }
            else {
                throw std::invalid_argument(
                    "unknown --unwind (expected fp|dwarf|both): " + std::string(v));
            }
        } else if (auto v = eat("--backend="); !v.empty()) {
            if (v == "perf")        { c.backend = backend_mode::perf; }
            else if (v == "signal") { c.backend = backend_mode::signal; }
            else if (v == "both")   { c.backend = backend_mode::both; }
            else {
                throw std::invalid_argument(
                    "unknown --backend (expected perf|signal|both): " + std::string(v));
            }
}
    }
    return c;
}

struct agg
{
    double mean      = 0.0;
    double stddev    = 0.0;
};

agg mean_stddev(const std::vector<double>& xs)
{
    agg a;
    if (xs.empty()) return a;
    a.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    double s2 = 0.0;
    for (double const v : xs) s2 += (v - a.mean) * (v - a.mean);
    a.stddev = std::sqrt(s2 / static_cast<double>(xs.size()));
    return a;
}

/** @brief Aggregated results for one (backend, freq) measurement segment. */
struct segment_stats
{
    agg           wall_ms{};
    agg           cputime_ms{};
    double        mean_captured = 0.0;  // samples landed in per-tid vectors
    double        mean_total    = 0.0;  // kernel-reported total (incl. lost)
    double        mean_lost     = 0.0;  // PERF_RECORD_LOST count
};

/**
 * @brief Warm up, then run @p runs measured passes for one backend at one freq.
 * @param backend Which capture backend (or baseline) to exercise.
 * @param c Parsed CLI knobs (iters/inner/runs/warmup/depth/pages).
 * @param freq Sampling frequency in Hz; ignored for baseline.
 * @return Aggregated wall/cputime and mean sample counts over the measured runs.
 */
segment_stats run_segment(backend_choice backend, cpu_snap::sampling::unwind_mode unwind,
                          const cli& c, std::uint32_t freq)
{
    for (int w = 0; w < c.warmup; ++w) {
        (void) one_run(backend, unwind, c.iters, c.inner, freq, c.depth, c.pages, c.stack_bytes);
    }

    std::vector<double> wall;
    std::vector<double> cpu;
    double captured_sum = 0.0;
    double total_sum    = 0.0;
    double lost_sum     = 0.0;
    for (int r = 0; r < c.runs; ++r) {
        const auto rr = one_run(backend, unwind, c.iters, c.inner, freq, c.depth, c.pages, c.stack_bytes);
        wall.push_back(rr.wall_ns);
        cpu.push_back(rr.cputime_ns);
        captured_sum += static_cast<double>(rr.samples_captured);
        total_sum    += static_cast<double>(rr.total_samples);
        lost_sum     += static_cast<double>(rr.lost_samples);
    }

    const auto n = static_cast<double>(c.runs);
    segment_stats s;
    s.wall_ms       = mean_stddev(wall);
    s.cputime_ms    = mean_stddev(cpu);
    s.mean_captured = captured_sum / n;
    s.mean_total    = total_sum / n;
    s.mean_lost     = lost_sum / n;
    s.wall_ms.mean    /= 1e6;
    s.wall_ms.stddev  /= 1e6;
    s.cputime_ms.mean   /= 1e6;
    s.cputime_ms.stddev /= 1e6;
    return s;
}

/**
 * @brief Emit one backend's comparison row for a freq.
 * @param label Backend column label ("perf" / "signal").
 * @param base Baseline (no sampler) stats at this freq, for the overhead ratios.
 * @param s Aggregated stats for this backend at this freq.
 */
void print_backend_row(const std::string& label, const segment_stats& base,
                       const segment_stats& s)
{
    const auto ovh_pct = [](double with, double without) {
        return without > 0.0 ? ((with / without) - 1.0) * 100.0 : 0.0;
    };
    // Normalize overhead by collected sample count: us added per 1000 samples.
    // Lets perf and signal compare even when they land different sample counts.
    const auto ovh_us_per_1k = [&s](double with, double without) {
        return s.mean_captured > 0.0
            ? ((with - without) * 1e3 / s.mean_captured) * 1e3 : 0.0;
    };
    std::printf("  %-12s %-7.2f+-%-5.2f %+8.2f %-7.2f+-%-5.2f %+8.2f %-12.0f %-10.0f %-12.0f"
                " %+10.2f %+10.2f\n",
                label.c_str(),
                s.wall_ms.mean, s.wall_ms.stddev,
                ovh_pct(s.wall_ms.mean, base.wall_ms.mean),
                s.cputime_ms.mean, s.cputime_ms.stddev,
                ovh_pct(s.cputime_ms.mean, base.cputime_ms.mean),
                s.mean_captured, s.mean_total, s.mean_lost,
                ovh_us_per_1k(s.wall_ms.mean, base.wall_ms.mean),
                ovh_us_per_1k(s.cputime_ms.mean, base.cputime_ms.mean));
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
    const cli c = parse_cli(argc, argv);

    const bool do_perf   = c.backend != backend_mode::signal;
    const bool do_signal = c.backend != backend_mode::perf;
    const char* backend_str = "both";
    if (c.backend == backend_mode::perf)   backend_str = "perf";
    if (c.backend == backend_mode::signal) backend_str = "signal";

    std::printf("# bench_overhead\n");
    std::printf("# iters=%lu inner=%lu runs=%d warmup=%d depth=%u pages=%u stack_bytes=%u backend=%s\n",
                c.iters, c.inner, c.runs, c.warmup, c.depth, c.pages, c.stack_bytes, backend_str);
    std::printf("# columns: wall_ms = mean +- stddev over runs; ovh%% vs baseline;\n");
    std::printf("#          cpu_ms = on-CPU time of the worker thread (excludes sampler threads);\n");
    std::printf("#          samples = captured; total = kernel-reported; lost = dropped;\n");
    std::printf("#          ovh/1k = microseconds of overhead per 1000 captured samples\n");

    for (std::uint32_t const f : c.freqs) {
        // Fresh baseline per freq: the workload is identical, but re-measuring
        // each segment keeps the comparison robust to drift / CPU contention.
        const segment_stats base =
            run_segment(backend_choice::baseline, cpu_snap::sampling::unwind_mode::auto_best, c, f);

        std::printf("\nfreq=%u Hz   baseline wall=%.2f +- %.2f ms  cpu=%.2f +- %.2f ms\n",
                    f, base.wall_ms.mean, base.wall_ms.stddev,
                    base.cputime_ms.mean, base.cputime_ms.stddev);
        std::printf("  %-12s %-14s %-8s %-14s %-8s %-12s %-10s %-12s %-10s %-10s\n",
                    "backend", "wall_ms", "ovh%", "cpu_ms", "cpu_ovh%", "samples",
                    "total", "lost", "ovh/1k(us)", "cpu/1k(us)");

        // Per-(backend, unwind) isolation: a combination that fails to open (perf
        // denied, the signal timer rejected by the kernel, or a backend that
        // cannot provide the requested unwind) must not abort the sweep. Report
        // it and carry on so the other rows + baseline still print.
        const auto measure = [&](backend_choice bc, const char* bname) {
            for (const auto um : c.unwinds) {
                const std::string label =
                    std::string(bname) + '/' +
                    (um == cpu_snap::sampling::unwind_mode::dwarf ? "dwarf" : "fp");
                try {
                    const segment_stats s = run_segment(bc, um, c, f);
                    print_backend_row(label, base, s);
                } catch (const std::exception& e) {
                    std::printf("  %-12s FAILED: %s\n", label.c_str(), e.what());
                }
            }
        };
        if (do_perf)   { measure(backend_choice::perf, "perf"); }
        if (do_signal) { measure(backend_choice::signal, "signal"); }
    }
    return 0;
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
