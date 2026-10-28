// autoload_sampler.cpp
//
// Builds as libcallstack_autoload.so:
//   $ LD_PRELOAD=/path/to/libcallstack_autoload.so /path/to/your_binary
//
// A constructor installs the sampler before main() and rescans for new
// threads; a destructor stops it and writes a Perfetto trace.
//
// Environment variables (all optional)
//   CALLSTACK_TRACE                 output path     (default: /tmp/callstack_autoload.pftrace)
//                                   %p = pid, %e = process name (comm). Every process of a
//                                   tree that loads this library writes its own trace; a process
//                                   with no samples writes nothing.
//   CALLSTACK_FREQ_HZ               sample frequency (default: 200)
//   CALLSTACK_DEPTH                 max frames/sample (default: core::k_default_callchain_depth)
//   CALLSTACK_PAGES                 kernel ring pages per tid, pow2 (default: 0 = automatic)
//   CALLSTACK_BACKEND               auto|perf|signal (default: auto = perf when permitted, else signal)
//   CALLSTACK_UNWIND                auto|fp|dwarf (default: auto = frame pointers on perf, DWARF on signal)
//   CALLSTACK_STACK_BYTES           user-stack snapshot per sample for DWARF unwinding
//                                   (perf dwarf_stack_bytes / signal stack_snapshot_bytes; default: 32768)
//   CALLSTACK_MERGE                 1|0  - merge frames across consecutive samples (default: 1)
//   CALLSTACK_FILTER_STDLIB         1|0  - drop C++ standard library frames (default: 0)
//   CALLSTACK_RESCAN_MS             /proc/self/task rescan period   (default: 100)
//   CALLSTACK_PROCESS_NAME          label in the Perfetto trace     (default: /proc/self/comm)
//   CALLSTACK_INCLUDE_BINARY_NAME   1|0  - include [binary] suffix on names (default: 1)
//   CALLSTACK_MEMORY_SINK_MB        bound on the in-memory capture buffer (default: 512)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "backend_select.hpp"
#include "name_filter.hpp"
#include "perfetto_proto_writer.hpp"
#include "symbol_preload.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/processing/symbols/sample_postprocessor.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"

namespace
{

constexpr std::uint32_t k_default_freq_hz         = 200;
constexpr std::uint32_t k_default_depth           = cpu_snap::core::k_default_callchain_depth;
constexpr std::uint32_t k_default_pages           = 0;  // sampler sizes the ring
constexpr int           k_default_rescan_ms       = 100;
constexpr const char*   k_default_out_path        = "/tmp/callstack_autoload.pftrace";
constexpr std::uint32_t k_default_memory_sink_mb  = 512;

// Owned by std::unique_ptr so the dtor function below resets it
// explicitly; we do not rely on the static destruction order
// (extern "C" __attribute__((destructor)) runs before the global
// dtors, so the reset happens deterministically). g_sink must outlive
// g_recorder (the recorder's dtor drains in-flight sink callbacks), so
// callstack_autoload_fini() resets g_recorder before g_sink.
std::unique_ptr<cpu_snap::core::memory_sink>    g_sink;
std::unique_ptr<cpu_snap::sampling::sampler>    g_recorder;
std::string                            g_out_path;
std::unordered_map<pid_t, std::string> g_tid_names;

/**
 * Read an unsigned 32-bit value from an environment variable.
 *
 * @param name     Environment variable name (null-terminated).
 * @param fallback Returned when the variable is unset, empty, or fails
 *                 to parse as an unsigned integer.
 * @return         Parsed value, otherwise @p fallback.
 *
 * @note Parsing uses @c std::stoul; values exceeding @c uint32_t are
 *       truncated by the static_cast.
 */
std::uint32_t
env_u32(const char* name, std::uint32_t fallback) noexcept
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe): env vars read once at startup, before any worker threads spawn
    if(const char* str = std::getenv(name); str != nullptr)
    {
        try
        {
            return static_cast<std::uint32_t>(std::stoul(str));
        } catch(...)  // NOLINT(bugprone-empty-catch): parse failure -> fallback
        {}
    }
    return fallback;
}

/**
 * Read a string value from an environment variable.
 *
 * @param name     Environment variable name (null-terminated).
 * @param fallback Returned when the variable is unset.
 * @return         Owned copy of the variable's value, otherwise an
 *                 owned copy of @p fallback.
 */
std::string
env_str(const char* name, std::string_view fallback)
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe): env vars read once at startup, before any worker threads spawn
    if(const char* str = std::getenv(name); str != nullptr)
    {
        return str;
    }
    return std::string(fallback);
}

/**
 * Read a boolean value from an environment variable.
 *
 * Recognises true:  "1" "true" "TRUE" "on" "ON" "yes" "YES".
 * Recognises false: "0" "false" "FALSE" "off" "OFF" "no" "NO".
 * Anything else falls through to @p fallback.
 *
 * @param name     Environment variable name (null-terminated).
 * @param fallback Returned when the variable is unset or its value is
 *                 not in the recognised set.
 * @return         Parsed boolean, otherwise @p fallback.
 */
bool
env_bool(const char* name, bool fallback) noexcept
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe): env vars read once at startup, before any worker threads spawn
    if(const char* str = std::getenv(name); str != nullptr)
    {
        const std::string_view view(str);
        if(view == "1" || view == "true" || view == "TRUE" || view == "on" ||
           view == "ON" || view == "yes" || view == "YES")
        {
            return true;
        }
        if(view == "0" || view == "false" || view == "FALSE" || view == "off" ||
           view == "OFF" || view == "no" || view == "NO")
        {
            return false;
        }
    }
    return fallback;
}

std::string
read_self_comm()
{
    auto name = cpu_snap::sampling::utils::read_proc_self_comm();
    if(name.empty()) name = "callstack_target";
    return name;
}

/** Replace every @p token in @p text with @p value. */
void
replace_all(std::string& text, std::string_view token, const std::string& value)
{
    for(std::size_t pos = text.find(token); pos != std::string::npos; pos = text.find(token, pos))
    {
        text.replace(pos, token.size(), value);
        pos += value.size();
    }
}

/** Expand %p (pid) and %e (process name) in an output path. */
std::string
expand_output_path(std::string path)
{
    replace_all(path, "%p", std::to_string(::getpid()));
    replace_all(path, "%e", read_self_comm());
    return path;
}

cpu_snap::sampling::unwind_mode
env_unwind_mode()
{
    const std::string value = env_str("CALLSTACK_UNWIND", "auto");
    if(value == "fp" || value == "frame_pointer") return cpu_snap::sampling::unwind_mode::frame_pointer;
    if(value == "dwarf") return cpu_snap::sampling::unwind_mode::dwarf;
    return cpu_snap::sampling::unwind_mode::auto_best;
}

/** Snapshot tid -> /proc/<pid>/task/<tid>/comm while threads are still alive. */
std::unordered_map<pid_t, std::string>
snapshot_thread_names()
{
    std::unordered_map<pid_t, std::string> names;
    for(const pid_t tid : cpu_snap::sampling::utils::list_self_thread_tids())
    {
        names.emplace(tid, cpu_snap::sampling::utils::read_thread_comm(tid));
    }
    return names;
}

/**
 * Run the postprocessor over raw samples, returning per-tid raw_sample
 * vectors with frames coalesced by resolved function name.
 */
std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>>
postprocess_samples_to_filtered(
    const auto& raw, cpu_snap::processing::symbols::symbolizer& sym,
    cpu_snap::processing::symbols::sample_postprocessor& proc)
{
    std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>> filtered;
    auto sink = [&](const cpu_snap::processing::symbols::processed_sample& s) {
        cpu_snap::core::raw_sample mirror{};
        mirror.tid          = s.tid;
        mirror.timestamp_ns = s.timestamp_ns;
        mirror.ip           = s.ip;
        mirror.frames.reserve(s.frames.size());
        for(const auto& frame : s.frames)
        {
            mirror.frames.push_back(frame.ip);
        }
        filtered[s.tid].push_back(mirror);
    };
    proc.process_all(raw, sink);

    // Collapse same-function frames so the slice_emitter shapes one
    // continuous slice per function instead of one per sample.
    cpu_snap::processing::symbols::coalesce_frames_by_name(
        filtered,
        [&sym](std::uint64_t ip) { return std::string(sym.resolve(ip)); });
    return filtered;
}

}  // namespace

// CONSTRUCTOR: runs before main(). Open per-tid perf events on every
// thread that exists at load time and start the rescan worker.
extern "C" __attribute__((constructor)) void
callstack_autoload_init()
{
    // cpu-snap-run --no-children: only the process with this pid samples itself.
    if(const auto only_pid = env_u32("CALLSTACK_ONLY_PID", 0);
       only_pid != 0 && only_pid != static_cast<std::uint32_t>(::getpid()))
    {
        return;
    }

    // Every process that loads us samples itself: the launcher, the exec'd
    // target and any child. Our own helpers (addr2line) start only after the
    // destructor has unset LD_PRELOAD, so they never re-enter this function.
    g_out_path = expand_output_path(env_str("CALLSTACK_TRACE", k_default_out_path));

    const auto sample_freq_hz = env_u32("CALLSTACK_FREQ_HZ", k_default_freq_hz);
    const auto stack_bytes    = env_u32("CALLSTACK_STACK_BYTES",
                                        cpu_snap::sampling::sampler::perf_attr_options{}.dwarf_stack_bytes);
    const std::size_t sink_max_bytes =
        static_cast<std::size_t>(env_u32("CALLSTACK_MEMORY_SINK_MB", k_default_memory_sink_mb)) *
        1024U * 1024U;
    g_sink = std::make_unique<cpu_snap::core::memory_sink>(
        cpu_snap::core::memory_sink::config{ .max_bytes = sink_max_bytes });

    cpu_snap::sampling::sampler::config cfg{
        .perf      = { .sample_freq_hz       = sample_freq_hz,
                       .callchain_depth      = std::min<std::uint32_t>(
                           env_u32("CALLSTACK_DEPTH", k_default_depth),
                           cpu_snap::core::k_max_callchain_depth),
                       .kernel_pages_per_tid = env_u32("CALLSTACK_PAGES",
                                                       k_default_pages),
                       .dwarf_stack_bytes    = stack_bytes,
                       .unwind               = env_unwind_mode() },
        .threads   = { .rescan_period =
                           std::chrono::milliseconds{ static_cast<int>(env_u32(
                               "CALLSTACK_RESCAN_MS",
                               static_cast<std::uint32_t>(k_default_rescan_ms))) },
                       .watch_new_threads = true },
        .streaming = { .sink         = std::ref(*g_sink),
                       .process_name = env_str("CALLSTACK_PROCESS_NAME", "") },
    };

    const auto backend = cs_examples::backend_from_name(env_str("CALLSTACK_BACKEND", "auto"));
    cfg.backend        = backend;
    if(backend != cpu_snap::sampling::backend_kind::perf)
    {
        cfg.signal                      = cs_examples::make_signal_options(sample_freq_hz);
        cfg.signal.stack_snapshot_bytes = stack_bytes;
    }

    const auto diag_freq_hz       = cfg.perf.sample_freq_hz;
    const auto diag_depth         = cfg.perf.callchain_depth;
    const auto diag_pages_per_tid = cfg.perf.kernel_pages_per_tid;

    try
    {
        g_recorder = std::make_unique<cpu_snap::sampling::sampler>(std::move(cfg));
        g_recorder->start();
    } catch(const std::exception& e)
    {
        std::fprintf(stderr, "[callstack_autoload] failed to start sampler: %s\n",
                     e.what());
        g_recorder.reset();
        return;
    }

    if(const auto& reason = g_recorder->selected_backend().reason; !reason.empty())
    {
        std::fprintf(stderr, "[callstack_autoload] %s\n", reason.c_str());
    }
    const auto& caps     = g_recorder->caps();
    const char* mode_str = "?";
    switch(g_recorder->resolved_unwind_mode())
    {
        case cpu_snap::sampling::unwind_mode::frame_pointer:
            mode_str = "frame_pointer";
            break;
        case cpu_snap::sampling::unwind_mode::dwarf: mode_str = "dwarf"; break;
        case cpu_snap::sampling::unwind_mode::auto_best: mode_str = "auto"; break;
    }
    std::fprintf(stderr, "[callstack_autoload] caps=%s\n", caps.to_string().c_str());
    std::fprintf(stderr,
                 "[callstack_autoload] started: freq=%u Hz, depth=%u, "
                 "pages_per_tid=%u, unwind=%s, trace=%s, threads=%zu\n",
                 diag_freq_hz, diag_depth, diag_pages_per_tid,
                 mode_str, g_out_path.c_str(), g_recorder->tracked_threads());
}

// DESTRUCTOR: runs after main() returns. Snapshot thread names while
// /proc/self/task is still populated for our own pid, stop the
// recorder, post-process raw samples, write the Perfetto trace.
extern "C" __attribute__((destructor)) void
callstack_autoload_fini()
{
    if(!g_recorder) return;

    // Snapshot tid -> name BEFORE stop(): the worker thread is still
    // alive here, but /proc/self/task entries for our own threads also
    // remain until the threads themselves exit.
    g_tid_names = snapshot_thread_names();

    g_recorder->stop();

    // Prevent addr2line and any other spawned helper from inheriting
    // our LD_PRELOAD (would otherwise re-enter callstack_autoload_init).
    ::unsetenv("LD_PRELOAD");

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(g_sink->bytes());
    const auto& raw   = reconstructor.samples();
    std::size_t total = 0;
    for(const auto& [tid, vec] : raw)
    {
        (void) tid;
        total += vec.size();
    }

    if(total == 0)
    {
        // Launchers and short helpers: nothing to show, leave no file behind.
        g_recorder.reset();
        g_sink.reset();
        return;
    }

    const std::string proc_name =
        env_str("CALLSTACK_PROCESS_NAME", read_self_comm().c_str());

    cpu_snap::processing::symbols::symbolizer_config sym_cfg;
    sym_cfg.include_binary_name = env_bool("CALLSTACK_INCLUDE_BINARY_NAME", true);
    cpu_snap::processing::symbols::symbolizer sym{ sym_cfg };
    cs_examples::preload_symbols_from(sym, raw);

    cs_examples::perfetto::perfetto_proto_writer writer(g_out_path);
    writer.set_process(static_cast<int>(::getpid()), proc_name);

    for(const auto& [tid, vec] : raw)
    {
        (void) vec;
        auto it = g_tid_names.find(tid);
        writer.add_thread(static_cast<int>(tid),
                          it != g_tid_names.end() ? it->second
                                                  : "tid-" + std::to_string(tid));
    }

    cpu_snap::processing::symbols::sample_postprocessor_config pcfg;
    pcfg.resolver = [&sym](std::uint64_t ip) { return std::string(sym.resolve(ip)); };
    const bool filter_stdlib = env_bool("CALLSTACK_FILTER_STDLIB", false);
    pcfg.frame_predicate = [filter_stdlib](std::uint64_t, std::string_view name) {
        return cs_examples::name_looks_resolved(name) &&
               !(filter_stdlib && cpu_snap::processing::symbols::is_stdlib_label(name));
    };
    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(pcfg));

    auto filtered = postprocess_samples_to_filtered(raw, sym, proc);

    cpu_snap::processing::timeline::slice_emitter_config scfg;
    scfg.merge_samples  = env_bool("CALLSTACK_MERGE", true);
    scfg.resolver  = [&sym](std::uint64_t ip) { return std::string(sym.resolve(ip)); };
    scfg.symbol_identity = [&sym](std::uint64_t ip) { return sym.symbol_identity(ip); };
    cpu_snap::processing::timeline::slice_emitter emitter(std::move(scfg));
    emitter.process(filtered, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
        writer.write_slice(s);
    });
    writer.finalize();

    std::fprintf(stderr,
                 "[callstack_autoload] wrote %s: %zu raw samples, "
                 "delivered=%zu, dropped_frames=%zu, unique_symbols=%zu, "
                 "captured %zu bytes (dropped %zu chunks)\n",
                 g_out_path.c_str(), total, proc.delivered_samples(),
                 proc.dropped_frames(), proc.unique_symbols(), g_sink->bytes().size(),
                 g_sink->dropped_chunks());

    // g_recorder first: its dtor may still drain in-flight sink callbacks,
    // so g_sink must stay alive until g_recorder is gone.
    g_recorder.reset();
    g_sink.reset();
}
