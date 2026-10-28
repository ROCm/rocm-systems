// multi_thread_sample_demo.cpp
//
// Two-phase demo: the sampler records raw stack samples online without any
// symbol resolution; sample_postprocessor then resolves them via addr2line
// offline and writes a Perfetto trace with per-thread nested slices.
//
// Usage: ./multi_thread_sample_demo [out.pftrace] [--backend auto|perf|signal]
//   The flag may appear anywhere; --backend signal needs no perf_event_open
//   privilege, so it works where perf is gated by perf_event_paranoid.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#include "backend_select.hpp"
#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "name_filter.hpp"
#include "perfetto_proto_writer.hpp"
#include "symbol_preload.hpp"
#include "workload.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"
#include "cpu_snap/processing/symbols/sample_postprocessor.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"

namespace
{

constexpr std::uint32_t k_sample_freq_hz  = 200;
constexpr std::uint32_t k_callchain_depth = 32;
constexpr std::uint32_t k_kernel_pages    = 8;
constexpr int           k_n_workers       = 4;
constexpr auto          k_run_duration    = std::chrono::seconds{ 3 };

void
worker_main(int idx, std::atomic<bool>& stop)
{
    char namebuf[16];
    std::snprintf(namebuf, sizeof(namebuf), "worker-%d", idx);
    ::pthread_setname_np(::pthread_self(), namebuf);

    auto acc = static_cast<double>(idx);
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += cs_examples::run_workload(acc, 200'000);
    }
    if(acc == 12345.6789) std::printf("unreachable\n");
}

/** Spawn the worker threads. Caller is responsible for join after stop. */
void
spawn_workers(std::vector<std::thread>& workers, std::atomic<bool>& stop_flag)
{
    workers.reserve(k_n_workers);
    for(int i = 0; i < k_n_workers; ++i)
    {
        workers.emplace_back(worker_main, i, std::ref(stop_flag));
    }
}

/**
 * Snapshot tid -> /proc/self/task/<tid>/comm for every currently-live
 * thread. Must be done before joining workers; once a thread exits,
 * its /proc entry vanishes.
 */
std::unordered_map<pid_t, std::string>
snapshot_thread_names()
{
    std::unordered_map<pid_t, std::string> tid_names;
    for(const pid_t tid : cpu_snap::sampling::utils::list_self_thread_tids())
    {
        tid_names.emplace(tid, cpu_snap::sampling::utils::read_thread_comm(tid));
    }
    return tid_names;
}

/**
 * Build a sample_postprocessor configured to resolve via `sym`, drop
 * hex-only frames, and honor the CALLSTACK_DROP_LIBRARY_FRAMES env knob.
 */
cpu_snap::processing::symbols::sample_postprocessor
make_postprocessor(cpu_snap::processing::symbols::symbolizer& sym)
{
    cpu_snap::processing::symbols::sample_postprocessor_config pcfg;
    pcfg.resolver = [&sym](std::uint64_t ip) {
        return std::string(sym.resolve(ip));
    };
    pcfg.frame_predicate = [](std::uint64_t /*ip*/, std::string_view name) {
        return cs_examples::name_looks_resolved(name);
    };
    if(const char* env = std::getenv("CALLSTACK_DROP_LIBRARY_FRAMES");
       env != nullptr && env[0] == '1')
    {
        pcfg.drop_library_frames = true;
    }
    return cpu_snap::processing::symbols::sample_postprocessor(std::move(pcfg));
}

/**
 * Run the sample_postprocessor over the decoded per-tid raw samples. For
 * each processed sample it (a) registers the thread track in the writer
 * if new, (b) mirrors the processed frames into a raw_sample stored
 * in per_tid_for_slices for later emission via the slice_emitter.
 */
void
postprocess_and_register(cpu_snap::processing::symbols::sample_postprocessor&                              processor,
                         const std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>& raw_per_tid,
                         cs_examples::perfetto::perfetto_proto_writer&                      writer,
                         const std::unordered_map<pid_t, std::string>&                 tid_names,
                         std::unordered_set<pid_t>&                                    registered,
                         std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>& per_tid_for_slices)
{
    auto sink = [&](const cpu_snap::processing::symbols::processed_sample& s) {
        if(registered.insert(s.tid).second)
        {
            auto it = tid_names.find(s.tid);
            writer.add_thread(static_cast<int>(s.tid),
                              it != tid_names.end() ? it->second
                                                    : "tid-" + std::to_string(s.tid));
        }

        cpu_snap::core::raw_sample mirror{};
        mirror.tid          = s.tid;
        mirror.timestamp_ns = s.timestamp_ns;
        mirror.ip           = s.ip;
        mirror.frames.reserve(s.frames.size());
        for(const auto& frame : s.frames)
        {
            mirror.frames.push_back(frame.ip);
        }
        per_tid_for_slices[s.tid].push_back(mirror);
    };

    processor.process_all(raw_per_tid, sink);
}

/**
 * Build slice_emitter_config matching the post-processor's resolver,
 * honoring the same env knob for library-frame dropping. @p library_maps
 * is used only when the env knob is set; it must outlive the returned
 * config's use (i.e. the emitter's process() call).
 */
cpu_snap::processing::timeline::slice_emitter_config
make_slice_emitter_config(cpu_snap::processing::symbols::symbolizer& sym,
                          cpu_snap::processing::symbols::proc_maps&  library_maps)
{
    cpu_snap::processing::timeline::slice_emitter_config scfg;
    if(const char* env = std::getenv("CALLSTACK_DROP_LIBRARY_FRAMES");
       env != nullptr && env[0] == '1')
    {
        scfg.library_frame_filter = cpu_snap::processing::symbols::make_main_exe_only_filter(library_maps);
    }
    scfg.resolver = [&sym](std::uint64_t ip) {
        return std::string(sym.resolve(ip));
    };
    scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
    return scfg;
}

/**
 * Drive slice_emitter once over the captured per-tid samples, feeding
 * each emitted_slice straight into the writer. The slice_emitter is the
 * sole owner of timeline shaping; the writer is a pure sink. Returns the
 * number of slices emitted.
 */
std::size_t
emit_thread_slices(cs_examples::perfetto::perfetto_proto_writer&                                 writer,
                   cpu_snap::processing::symbols::symbolizer&                                       sym,
                   cpu_snap::processing::symbols::proc_maps&                                        library_maps,
                   const std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>& per_tid_for_slices)
{
    cpu_snap::processing::timeline::slice_emitter emitter(make_slice_emitter_config(sym, library_maps));
    return emitter.process(
        per_tid_for_slices, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
            writer.write_slice(s);
        });
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
                                         : "multi_thread_sample_demo.pftrace";

        std::printf("phase 1 (online recording): sample_freq=%u Hz, "
                    "callchain_depth=%u, kernel_pages_per_tid=%u\n",
                    k_sample_freq_hz, k_callchain_depth, k_kernel_pages);

        std::atomic<bool>        stop_workers{ false };
        std::vector<std::thread> workers;
        spawn_workers(workers, stop_workers);

        std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });

        constexpr std::size_t k_memory_sink_max_bytes = 256U * 1024U * 1024U;

        // Sink must outlive the recorder (recorder dtor drains in-flight callbacks).
        cpu_snap::core::memory_sink sink(
            cpu_snap::core::memory_sink::config{ .max_bytes = k_memory_sink_max_bytes });

        cpu_snap::sampling::sampler::config cfg{
            .perf      = { .sample_freq_hz       = k_sample_freq_hz,
                           .callchain_depth      = k_callchain_depth,
                           .kernel_pages_per_tid = k_kernel_pages,
                           .unwind               = cpu_snap::sampling::unwind_mode::frame_pointer },
            .threads   = { .watch_new_threads = true },
            .streaming = { .sink = std::ref(sink), .process_name = "multi_thread_sample_demo" },
        };
        cfg.backend = sel.backend;
        if(sel.backend != cpu_snap::sampling::backend_kind::perf)
        {
            cfg.signal = cs_examples::make_signal_options(k_sample_freq_hz);
        }

        cpu_snap::sampling::sampler recorder(cfg);
        sel.print(recorder);
        recorder.start();
        std::printf("recorder attached to %zu threads\n", recorder.tracked_threads());

        std::this_thread::sleep_for(k_run_duration);

        // Snapshot must happen before workers exit (/proc/self/task entries vanish).
        const auto tid_names = snapshot_thread_names();

        recorder.stop();
        stop_workers.store(true, std::memory_order_relaxed);
        for(auto& t : workers) t.join();

        cpu_snap::core::sample_reconstructor reconstructor;
        reconstructor.feed(sink.bytes());
        const auto& raw_per_tid = reconstructor.samples();

        std::printf("phase 1 done: %zu raw samples across %zu threads (lost=%zu, "
                    "captured %zu bytes, dropped %zu chunks)\n",
                    recorder.total_samples(), raw_per_tid.size(), recorder.lost_samples(),
                    sink.bytes().size(), sink.dropped_chunks());

        // Default symbolizer formatting: "function(args) [binary]". Set
        // include_binary_name=false to drop the suffix entirely.
        cpu_snap::processing::symbols::symbolizer sym(cpu_snap::processing::symbols::symbolizer_config{
            .include_binary_name = true,
        });
        cs_examples::preload_symbols_from(sym, raw_per_tid);

        cs_examples::perfetto::perfetto_proto_writer writer(out_path);
        writer.set_process(static_cast<int>(::getpid()), "multi_thread_sample_demo");

        std::unordered_set<pid_t>                                     registered;
        std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>> per_tid_for_slices;

        cpu_snap::processing::symbols::sample_postprocessor processor = make_postprocessor(sym);
        postprocess_and_register(processor, raw_per_tid, writer, tid_names,
                                 registered, per_tid_for_slices);

        std::printf("phase 2 done: delivered=%zu, dropped_frames=%zu, "
                    "unique_symbols=%zu\n",
                    processor.delivered_samples(), processor.dropped_frames(), processor.unique_symbols());

        // Nested slices per tid for the timeline-view flame graph. The
        // slice_emitter owns timeline shaping (its resolver matches `sym`
        // so names align with the post-processor) and the writer is a
        // pure sink for the emitted spans. library_maps backs the
        // library_frame_filter callback when CALLSTACK_DROP_LIBRARY_FRAMES=1;
        // it must outlive emit_thread_slices's use of it.
        cpu_snap::processing::symbols::proc_maps library_maps;
        const std::size_t total_slices =
            emit_thread_slices(writer, sym, library_maps, per_tid_for_slices);
        std::printf("slice_emitter: %zu slices\n", total_slices);
        writer.finalize();

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
