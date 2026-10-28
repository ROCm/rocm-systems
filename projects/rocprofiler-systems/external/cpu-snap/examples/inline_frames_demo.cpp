// inline_frames_demo.cpp
//
// Demonstrates symbolizer inline-frame expansion: a chain of always_inline
// helpers folds into one physical frame, and resolve_inline() stacks each
// level under the outermost caller in the Perfetto trace.
//
// Usage: ./inline_frames_demo [/tmp/out.pftrace] [--backend auto|perf|signal]
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
#include <unordered_set>

#include <unistd.h>

#include "backend_select.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "perfetto_proto_writer.hpp"
#include "symbol_preload.hpp"

namespace
{

[[gnu::always_inline]] inline double
inline_leaf(double x) noexcept
{
    return x * 1.0000001 + 1.0;
}

[[gnu::always_inline]] inline double
inline_mid_a(double x) noexcept
{
    return inline_leaf(x) + 0.5;
}

[[gnu::always_inline]] inline double
inline_mid_b(double x) noexcept
{
    return inline_mid_a(x) * 0.9999999;
}

[[gnu::always_inline]] inline double
inline_mid_c(double x) noexcept
{
    return inline_mid_b(x) + 0.1;
}

[[gnu::noinline]] double
outer_workload(std::uint64_t iters) noexcept
{
    double acc = 1.0;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = inline_mid_c(acc);
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
        const cs_examples::backend_selection sel =
            cs_examples::parse_backend(argc, argv);

        const std::string out_path =
            (sel.output_path != nullptr) ? sel.output_path : "/tmp/inline_frames.pftrace";

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
            .threads   = { .watch_new_threads    = false },
            .streaming = { .sink                 = std::ref(sink),
                           .process_name         = "inline_frames_demo" },
        };
        cfg.backend = sel.backend;
        if(sel.backend != cpu_snap::sampling::backend_kind::perf)
        {
            cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
        }

        cpu_snap::sampling::sampler rec(cfg);
        sel.print(rec);
        rec.start();
        const double sink_val = outer_workload(300'000'000);
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
                .inline_frames       = true,
            });

        cs_examples::preload_symbols_from(sym, per_tid);

        // Unique IPs for the inline-chain depth report below (a distinct
        // pass from preload: it needs the deduped IP set itself, not just
        // the symbolizer cache preload triggers).
        std::unordered_set<std::uint64_t> seen;
        for(const auto& [tid, samples] : per_tid)
        {
            for(const auto& s : samples)
            {
                for(std::uint32_t i = 0; i < s.frames.size(); ++i)
                {
                    seen.insert(s.frames[i]);
                }
            }
        }

        // Report observed inline-chain depth as a quick health check.
        std::size_t max_chain = 0;
        std::size_t total_chain_frames = 0;
        std::size_t resolved_ips       = 0;
        for(std::uint64_t ip : seen)
        {
            const auto chain = sym.resolve_inline(ip);
            if(!chain.empty())
            {
                ++resolved_ips;
                total_chain_frames += chain.size();
                if(chain.size() > max_chain) max_chain = chain.size();
            }
        }
        std::printf("unique IPs=%zu, resolved=%zu, max inline depth=%zu, "
                    "avg inline depth=%.2f\n",
                    seen.size(), resolved_ips, max_chain,
                    resolved_ips
                        ? static_cast<double>(total_chain_frames) / static_cast<double>(resolved_ips)
                        : 0.0);

        cs_examples::perfetto::perfetto_proto_writer writer(out_path);
        writer.set_process(::getpid(), "inline_frames_demo");

        for(const auto& [tid, samples] : per_tid)
        {
            (void) samples;
            writer.add_thread(static_cast<int>(tid), "main");
        }

        cpu_snap::processing::timeline::slice_emitter_config scfg;
        scfg.inline_resolver = [&sym](std::uint64_t ip) {
            auto chain = sym.resolve_inline(ip);
            if(chain.empty())
            {
                chain.emplace_back(cpu_snap::core::format_hex(ip));
            }
            return chain;
        };
        cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
        emitter.process(per_tid,
                        [&](const cpu_snap::processing::timeline::emitted_slice& s) {
                            writer.write_slice(s);
                        });
        writer.finalize();

        std::printf("wrote %s; open in https://ui.perfetto.dev/ to see "
                    "the inlined call chain under each sample\n",
                    out_path.c_str());
        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s: %s\n", argc > 0 ? argv[0] : "(unknown)", e.what());
        return EXIT_FAILURE;
    }
}
