// single_thread_callstack_demo.cpp
//
// Minimal usage of cpu_snap::sampling::sampler: one thread, a short recursive
// workload, sampler routes bytes into a core::memory_sink,
// core::sample_reconstructor decodes them, Perfetto trace output.
//
// Usage: ./single_thread_callstack_demo [/tmp/out.pftrace] [--backend auto|perf|signal]
//   The flag may appear anywhere; --backend signal needs no perf_event_open
//   privilege, so it works where perf is gated by perf_event_paranoid.

#include <algorithm>
#include <array>
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

namespace
{

[[gnu::noinline]] double
leaf(double x) noexcept
{
    return x * 1.0000001 + 1.0;
}

[[gnu::noinline]] double
recurse(int depth, double acc) noexcept
{
    if(depth == 0) return leaf(acc);
    const double next = leaf(acc);
    return recurse(depth - 1, next > 1e20 ? next * 1e-20 : next);
}

[[gnu::noinline]] double
workload(std::uint64_t iters) noexcept
{
    double acc = 1.0;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = recurse(8, acc);
        if(acc > 1e30 || acc < -1e30) acc = 1.0;
    }
    return acc;
}

}  // namespace

int
main(int argc, char** argv)
{
    try
    {
        const cs_examples::backend_selection sel = cs_examples::parse_backend(argc, argv);

        const std::string out_path = (sel.output_path != nullptr)
                                         ? sel.output_path
                                         : "/tmp/single_thread_callstack.pftrace";

        constexpr std::uint32_t k_sample_freq_hz        = 1000;
        constexpr std::size_t   k_memory_sink_max_bytes = 256U * 1024U * 1024U;

        // Sink must outlive the recorder (recorder dtor drains in-flight callbacks).
        cpu_snap::core::memory_sink sink(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_memory_sink_max_bytes });

        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = k_sample_freq_hz,
                           .callchain_depth      = 32,
                           .kernel_pages_per_tid = 8,
                           .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
            .threads   = { .watch_new_threads = false },
            .streaming = { .sink = std::ref(sink), .process_name = "single_thread_callstack_demo" },
        };
        cfg.backend = sel.backend;
        if(sel.backend != cpu_snap::sampling::backend_kind::perf)
        {
            cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
        }

        cpu_snap::sampling::sampler rec(cfg);
        sel.print(rec);
        rec.start();
        const double sink_val = workload(2'000'000);
        rec.stop();
        std::printf("workload sink: %g, total_samples=%zu, captured %zu bytes "
                    "(dropped %zu chunks)\n",
                    sink_val, rec.total_samples(), sink.bytes().size(), sink.dropped_chunks());

        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.bytes());
        const auto& per_tid = reconstructor.samples();
        std::printf("decoded %zu tids (parse_errors=%lu)\n", per_tid.size(),
                    static_cast<unsigned long>(reconstructor.counters().parse_errors));

        cpu_snap::processing::symbols::symbolizer sym(
            cpu_snap::processing::symbols::symbolizer_config{
                .include_binary_name = true,
            });

        cs_examples::perfetto::perfetto_proto_writer writer(out_path);
        writer.set_process(::getpid(), "single_thread_callstack_demo");

        cs_examples::preload_symbols_from(sym, per_tid);

        for(const auto& [tid, _] : per_tid)
        {
            writer.add_thread(static_cast<int>(tid), "main");
        }

        cpu_snap::processing::timeline::slice_emitter_config scfg;
        scfg.resolver = [&sym](std::uint64_t ip) {
            const auto label = sym.resolve(ip);
            if(!label.empty()) return std::string(label);
            return cpu_snap::core::format_hex(ip);
        };
        scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
        cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
        emitter.process(per_tid, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
            writer.write_slice(s);
        });
        writer.finalize();

        std::printf("wrote %s\n", out_path.c_str());
        return 0;
    } catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n", argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
}
