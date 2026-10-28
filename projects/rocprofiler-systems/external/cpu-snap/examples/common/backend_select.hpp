#pragma once

/**
 * @file backend_select.hpp
 * @brief Tiny argv helper letting every demo pick the capture backend
 *        (auto, perf or signal) at runtime via `--backend auto|perf|signal`.
 *
 * NOT part of the cpu_snap library; the library selects the backend via
 * sampler::config::backend. This header only parses one command-line flag
 * and builds a sensible signal_attr_options for a given sample frequency so
 * the perf and signal runs of a demo stay comparable.
 *
 * The signal backend uses per-thread timers + a realtime signal and needs
 * NO perf_event_open privilege, so `--backend signal` works in environments
 * where perf is gated by perf_event_paranoid.
 */

#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>

#include "cpu_snap/sampling/sampler.hpp"

namespace cs_examples
{

/**
 * @brief Result of parsing the backend selection flag from argv.
 *
 * Carries the chosen backend plus the leftover positional output-path
 * argument (the first non-flag argv entry), so callers keep their existing
 * `./demo [out.pftrace]` behavior unchanged while gaining `--backend`.
 */
struct backend_selection
{
    /** Capture backend the user selected (default automatic: perf, else signal). */
    cpu_snap::sampling::backend_kind backend = cpu_snap::sampling::backend_kind::automatic;

    /**
     * First positional (non-flag) argument, or nullptr when none was given.
     * Demos use this as the output trace path, falling back to their own
     * default when it is null.
     */
    const char* output_path = nullptr;

    /**
     * Print the backend the constructed sampler actually uses (demo startup
     * banner) and, when automatic selection fell back, the reason on stderr.
     */
    void print(const cpu_snap::sampling::sampler& rec) const
    {
        const auto& choice = rec.selected_backend();
        std::printf("backend: %s\n",
                    choice.kind == cpu_snap::sampling::backend_kind::signal ? "signal" : "perf");
        if(!choice.reason.empty())
        {
            std::fprintf(stderr, "cpu-snap: %s\n", choice.reason.c_str());
        }
    }
};

[[nodiscard]] inline cpu_snap::sampling::backend_kind
backend_from_name(std::string_view value) noexcept
{
    if(value == "signal") return cpu_snap::sampling::backend_kind::signal;
    if(value == "perf") return cpu_snap::sampling::backend_kind::perf;
    return cpu_snap::sampling::backend_kind::automatic;
}

/**
 * @brief Parse argv for `--backend auto|perf|signal` and the positional out-path.
 *
 * The flag may appear anywhere in argv. `--backend=NAME` is also accepted.
 * An unrecognized value falls back to automatic. The first
 * argument that is not a flag (does not start with '-') and is not the value
 * consumed by `--backend` is returned as @ref backend_selection::output_path.
 *
 * @param argc Argument count as received by main().
 * @param argv Argument vector as received by main().
 * @return Parsed selection (defaults: automatic backend, null output path).
 */
[[nodiscard]] inline backend_selection
parse_backend(int argc, char** argv) noexcept
{
    backend_selection sel;
    for(int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if(arg == "--backend")
        {
            if(i + 1 < argc)
            {
                const std::string_view value = argv[++i];
                sel.backend = backend_from_name(value);
            }
            continue;
        }
        if(arg.starts_with("--backend="))
        {
            const std::string_view value = arg.substr(std::string_view("--backend=").size());
            sel.backend = backend_from_name(value);
            continue;
        }
        if(!arg.empty() && arg.front() == '-')
        {
            continue;  // ignore unknown flags, leave positional parsing intact
        }
        if(sel.output_path == nullptr)
        {
            sel.output_path = argv[i];
        }
    }
    return sel;
}

/**
 * @brief Build signal_attr_options matched to a target sample frequency.
 *
 * Demos pass the SAME frequency they use for the perf backend so the perf
 * and signal runs produce comparable sample counts. Stack snapshot size,
 * ring capacity, and the realtime-signal offset use library-sensible
 * defaults; start delay is zero so sampling begins immediately.
 *
 * @param sample_freq_hz Target per-thread sample rate in Hertz.
 * @return Populated signal_attr_options.
 */
[[nodiscard]] inline cpu_snap::sampling::sampler::signal_attr_options
make_signal_options(std::uint32_t sample_freq_hz) noexcept
{
    return cpu_snap::sampling::sampler::signal_attr_options{
        .sample_freq_hz       = sample_freq_hz,
        .signal_rt_offset     = 4,
        .stack_snapshot_bytes = 8192,  // 8 KiB, 8-byte aligned
        .ring_capacity        = 256,
        .start_delay_ns       = 0,
    };
}

}  // namespace cs_examples
