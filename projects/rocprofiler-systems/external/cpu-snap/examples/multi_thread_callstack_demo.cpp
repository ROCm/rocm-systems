// multi_thread_callstack_demo.cpp
//
// Samples a 4-thread compute workload, streams the wire format into a
// bounded core::memory_sink, replays it with core::sample_reconstructor and
// writes one Perfetto trace with per-thread callstack slices.
//
// Usage: ./multi_thread_callstack_demo [out.pftrace] [--backend auto|perf|signal]
//   --backend selects the sampler's capture backend. --backend signal needs
//   no perf_event_open privilege, so it works where perf is gated by
//   perf_event_paranoid.

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#include "backend_select.hpp"
#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "name_filter.hpp"
#include "perfetto_proto_writer.hpp"
#include "symbol_preload.hpp"
#include "workload.hpp"
#include "cpu_snap/processing/symbols/sample_postprocessor.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"

namespace
{

constexpr std::uint32_t k_sample_freq_hz        = 4000;
constexpr std::uint32_t k_callchain_depth       = 32;
constexpr std::uint32_t k_kernel_pages          = 8;
constexpr int           k_n_workers             = 4;
constexpr auto          k_run_duration          = std::chrono::milliseconds{ 1000 };
constexpr std::size_t   k_memory_sink_max_bytes = 256U * 1024U * 1024U;

void
worker_main(int idx, std::atomic<bool>& stop)
{
    std::array<char, 16> namebuf{};
    std::snprintf(namebuf.data(), namebuf.size(), "worker-%d", idx);
    ::pthread_setname_np(::pthread_self(), namebuf.data());

    auto acc = static_cast<double>(idx);
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += cs_examples::run_workload(acc, 200'000);
    }
    if(acc == 12345.6789)
    {
        std::printf("unreachable\n");
    }
}

/**
 * Run the 4-thread workload while the recorder is active.
 * Sleeps the run duration, stops the recorder, joins the workers,
 * and prints the phase-1 summary.
 */
void
run_workload_phase(cpu_snap::sampling::sampler& call_rec, const cpu_snap::core::memory_sink& call_sink)
{
    std::atomic<bool>        stop_workers{ false };
    std::vector<std::thread> workers;
    workers.reserve(k_n_workers);
    for(int i = 0; i < k_n_workers; ++i)
    {
        workers.emplace_back(worker_main, i, std::ref(stop_workers));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

    call_rec.start();
    std::printf("recorder attached: callstack tracks %zu threads\n",
                call_rec.tracked_threads());

    std::this_thread::sleep_for(k_run_duration);

    call_rec.stop();
    stop_workers.store(true);
    for(auto& worker : workers)
    {
        worker.join();
    }

    std::printf("PHASE 1 done: callstack total=%zu lost=%zu | callstack stream=%zu B "
                "(dropped %zu chunks)\n",
                call_rec.total_samples(), call_rec.lost_samples(), call_sink.bytes().size(),
                call_sink.dropped_chunks());
}

/**
 * Register a thread track per tid, then shape the timeline through a
 * single slice_emitter (resolver wired from @p sym) whose output flows
 * straight into the writer sink. Returns the number of threads seen.
 */
std::size_t
emit_callstack_tracks(cs_examples::perfetto::perfetto_proto_writer& writer,
                      cpu_snap::processing::symbols::symbolizer&       sym,
                      const std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>& per_tid_samples,
                      const std::function<void(pid_t)>&                              ensure_thread)
{
    for(const auto& [tid, vec] : per_tid_samples)
    {
        (void) vec;
        ensure_thread(tid);
    }

    cpu_snap::processing::timeline::slice_emitter_config scfg;
    scfg.resolver = [&sym](std::uint64_t ip) -> std::string {
        const auto label = sym.resolve(ip);
        if(!label.empty() && cs_examples::name_looks_resolved(label))
        {
            return std::string(label);
        }
        return cpu_snap::core::format_hex(ip);
    };
    scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
    cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
    emitter.process(per_tid_samples,
                    [&](const cpu_snap::processing::timeline::emitted_slice& s) {
                        writer.write_slice(s);
                    });
    return per_tid_samples.size();
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
            (sel.output_path != nullptr) ? sel.output_path
                                         : "multi_thread_callstack_demo.pftrace";

        std::printf("PHASE 1 (online): callstack recording for %lld ms\n",
                    static_cast<long long>(k_run_duration.count()));

        // LIFETIME INVARIANT: call_sink MUST be declared before the
        // recorder. The recorder config captures it via std::ref (see
        // .sink = std::ref(...) below); the recorder's dtor runs first
        // and joins its worker thread, which drains all in-flight sink
        // callbacks BEFORE this sink local goes out of scope. Reordering
        // this declaration (or moving it into a narrower scope) breaks
        // the invariant silently.
        cpu_snap::core::memory_sink call_sink(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_memory_sink_max_bytes });

        cpu_snap::sampling::sampler::config call_cfg{
            .perf      = { .sample_freq_hz       = k_sample_freq_hz,
                           .callchain_depth      = k_callchain_depth,
                           .kernel_pages_per_tid = k_kernel_pages,
                           .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
            .threads   = { .watch_new_threads = true },
            .streaming = { .sink         = std::ref(call_sink),
                           .process_name = "multi_thread_callstack_demo" },
        };
        call_cfg.backend = sel.backend;
        if(sel.backend != cpu_snap::sampling::backend_kind::perf)
        {
            call_cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
        }

        cpu_snap::sampling::sampler call_rec(call_cfg);
        sel.print(call_rec);

        run_workload_phase(call_rec, call_sink);

        std::printf("PHASE 2 (offline): replay + perfetto trace -> %s\n", out_path.c_str());

        cs_examples::perfetto::perfetto_proto_writer writer{ out_path };
        writer.set_process(::getpid(), "multi_thread_callstack_demo");

        std::unordered_set<pid_t> registered_threads;
        auto ensure_thread = [&](pid_t tid) {
            if(registered_threads.insert(tid).second)
            {
                writer.add_thread(static_cast<int>(tid), cpu_snap::sampling::utils::read_thread_comm(tid));
            }
        };

        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(call_sink.bytes());
        const auto&      per_tid_samples  = reconstructor.samples();
        const std::size_t call_parse_errors =
            static_cast<std::size_t>(reconstructor.counters().parse_errors);

        cpu_snap::processing::symbols::symbolizer sym{ cpu_snap::processing::symbols::symbolizer_config{
            .include_binary_name = true,
        } };
        cs_examples::preload_symbols_from(sym, per_tid_samples);

        const std::size_t slice_threads =
            emit_callstack_tracks(writer, sym, per_tid_samples, ensure_thread);

        writer.finalize();

        std::printf("PHASE 2 done: %zu slice-threads (callstack parse_errors=%zu)\n",
                    slice_threads, call_parse_errors);
        std::printf("\nperfetto trace : %s\nopen in        : https://ui.perfetto.dev/\n",
                    out_path.c_str());
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
