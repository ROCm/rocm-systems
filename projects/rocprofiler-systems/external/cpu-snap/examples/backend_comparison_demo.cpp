// backend_comparison_demo.cpp
//
// Runs the same workload under the perf and signal backends, writes
// <outdir>/backend_cmp_perf.pftrace and <outdir>/backend_cmp_signal.pftrace,
// then prints a comparison table.
//
// Usage: ./backend_comparison_demo [outdir]   (default /tmp)
//
// The perf backend is skipped when perf_event_paranoid forbids it. The signal
// backend needs no privileges and always uses DWARF unwinding.

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>

#include <unistd.h>

#include "backend_select.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "perfetto_proto_writer.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "symbol_preload.hpp"
#include "workload.hpp"

namespace
{

constexpr std::uint32_t k_sample_freq_hz        = 1000;
constexpr auto          k_workload_duration     = std::chrono::milliseconds(500);
constexpr std::size_t   k_memory_sink_max_bytes = 256U * 1024U * 1024U;

struct run_result
{
    bool         available   = false;
    std::size_t  samples     = 0;
    std::size_t  lost        = 0;
    std::int64_t duration_ms = 0;
    std::string  trace_path;
    std::string  error;
};

run_result
run_one(cpu_snap::sampling::backend_kind bk, const std::string& trace_path)
{
    run_result r;
    r.trace_path = trace_path;

    // Sink must outlive the recorder (recorder dtor drains in-flight callbacks).
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_memory_sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg{
        .perf      = { .sample_freq_hz       = k_sample_freq_hz,
                       .callchain_depth      = 32,
                       .kernel_pages_per_tid = 8,
                       .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
        .threads   = { .watch_new_threads    = false },
        .streaming = { .sink                 = std::ref(sink),
                       .process_name         = "backend_comparison_demo" },
    };
    cfg.backend = bk;
    if(bk == cpu_snap::sampling::backend_kind::signal)
    {
        cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
    }

    try
    {
        cpu_snap::sampling::sampler rec(cfg);
        rec.start();

        const auto t0       = std::chrono::steady_clock::now();
        const auto deadline = t0 + k_workload_duration;
        double     workload_sink = 1.0;
        while(std::chrono::steady_clock::now() < deadline)
        {
            workload_sink = cs_examples::run_workload(workload_sink, 10'000);
        }
        const auto t1 = std::chrono::steady_clock::now();
        (void) workload_sink;

        rec.stop();

        r.available   = true;
        r.samples     = rec.total_samples();
        r.lost        = rec.lost_samples();
        r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t1 - t0)
                            .count();

        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.bytes());
        const auto& per_tid = reconstructor.samples();

        cpu_snap::processing::symbols::symbolizer sym(
            cpu_snap::processing::symbols::symbolizer_config{
                .include_binary_name = true,
            });

        cs_examples::preload_symbols_from(sym, per_tid);

        cs_examples::perfetto::perfetto_proto_writer writer(trace_path);
        writer.set_process(::getpid(), "backend_comparison_demo");
        for(const auto& [tid, ignored] : per_tid)
        {
            writer.add_thread(static_cast<int>(tid), "main");
        }

        cpu_snap::processing::timeline::slice_emitter_config scfg;
        scfg.resolver = [&sym](std::uint64_t ip) {
            const auto label = sym.resolve(ip);
            if(!label.empty())
            {
                return std::string(label);
            }
            return cpu_snap::core::format_hex(ip);
        };
        scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
        cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
        emitter.process(per_tid,
                        [&](const cpu_snap::processing::timeline::emitted_slice& s) {
                            writer.write_slice(s);
                        });
        writer.finalize();
    }
    catch(const std::exception& ex)
    {
        r.available = false;
        r.error     = ex.what();
    }

    return r;
}

void
print_table(const run_result& perf_r, const run_result& signal_r)
{
    constexpr int k_col_backend  = 8;
    constexpr int k_col_unwind   = 13;
    constexpr int k_col_samples  = 8;
    constexpr int k_col_lost     = 5;
    constexpr int k_col_duration = 12;

    std::printf("\n%-*s  %-*s  %-*s  %-*s  %-*s  %s\n",
                k_col_backend,  "backend",
                k_col_unwind,   "unwind",
                k_col_samples,  "samples",
                k_col_lost,     "lost",
                k_col_duration, "duration_ms",
                "trace");
    std::printf("--------  -------------  --------  -----  "
                "------------  -----------------------------------\n");

    const auto print_row = [&](const char* name, const char* unwind,
                                const run_result& r) {
        if(r.available)
        {
            std::printf("%-*s  %-*s  %*zu  %*zu  %*lld  %s\n",
                        k_col_backend,  name,
                        k_col_unwind,   unwind,
                        k_col_samples,  r.samples,
                        k_col_lost,     r.lost,
                        k_col_duration, static_cast<long long>(r.duration_ms),
                        r.trace_path.c_str());
        }
        else
        {
            std::printf("%-*s  %-*s  %s\n",
                        k_col_backend, name,
                        k_col_unwind,  unwind,
                        r.error.empty() ? "unavailable" : r.error.c_str());
        }
    };

    print_row("perf",   "auto_best", perf_r);
    print_row("signal", "dwarf",     signal_r);
    std::printf("\n");
}

}  // namespace

int
main(int argc, char** argv)
{
    try
    {
        const std::string outdir =
            (argc > 1 && argv[1][0] != '-') ? argv[1] : "/tmp";

        const std::string perf_path   = outdir + "/backend_cmp_perf.pftrace";
        const std::string signal_path = outdir + "/backend_cmp_signal.pftrace";

        std::printf("cpu-snap backend comparison demo\n");
        std::printf("  workload: %lldms per backend, freq: %u Hz\n\n",
                    static_cast<long long>(k_workload_duration.count()),
                    k_sample_freq_hz);

        std::printf("running perf backend...\n");
        const run_result perf_r =
            run_one(cpu_snap::sampling::backend_kind::perf, perf_path);

        std::printf("running signal backend...\n");
        const run_result signal_r =
            run_one(cpu_snap::sampling::backend_kind::signal, signal_path);

        print_table(perf_r, signal_r);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n",
                     argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
}
