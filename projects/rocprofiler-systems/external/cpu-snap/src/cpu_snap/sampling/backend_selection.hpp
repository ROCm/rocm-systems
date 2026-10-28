#pragma once

/**
 * @file backend_selection.hpp
 * @brief Resolve backend_kind::automatic to a concrete capture backend.
 *
 * Header-only. C++20.
 */

#include <cerrno>
#include <cstring>
#include <string>
#include <system_error>

#include <sys/syscall.h>
#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/backend_kind.hpp"
#include "cpu_snap/sampling/sampler_capabilities.hpp"
#include "cpu_snap/sampling/triggers/perf/open_error.hpp"

namespace cpu_snap::sampling
{

using perf_probe_fn = perf_probe_result (*)(bool exclude_kernel);

/** @brief The concrete backend picked by resolve_backend(). */
struct backend_choice
{
    /** Never backend_kind::automatic. */
    backend_kind kind = backend_kind::perf;
    /** Non-empty only when automatic fell back to signal; says why. */
    std::string reason;
};

/**
 * @brief Pick the capture backend for @p requested.
 *
 * signal: signal, @p probe is not called. perf: perf, or throws when
 * @p probe reports perf_event_open unavailable. automatic: perf when the
 * probe succeeds, otherwise signal with a reason.
 *
 * @param exclude_kernel Forwarded to @p probe.
 * @throws cpu_snap::environment_error (code() == the probe errno) When @p requested is perf and the probe fails.
 */
[[nodiscard]] inline backend_choice
resolve_backend(backend_kind requested, bool exclude_kernel, perf_probe_fn probe)
{
    if(requested == backend_kind::signal)
    {
        return { backend_kind::signal, {} };
    }

    const perf_probe_result result = probe(exclude_kernel);
    if(result.available)
    {
        return { backend_kind::perf, {} };
    }

    const auto paranoid = triggers::perf::read_perf_event_paranoid();
    if(requested == backend_kind::perf)
    {
        const auto tid = static_cast<pid_t>(::syscall(SYS_gettid));
        throw environment_error(
            std::error_code(result.err, std::generic_category()),
            triggers::perf::describe_perf_open_failure(tid, result.err, paranoid));
    }

    std::string reason = "perf_event_open failed (";
    reason += std::strerror(result.err);
    if(paranoid.has_value())
    {
        reason += ", perf_event_paranoid=" + std::to_string(*paranoid);
    }
    reason += "); using the signal backend";
    return { backend_kind::signal, std::move(reason) };
}

/**
 * @brief One-line, human-readable description of @p choice for logging.
 *
 * A fallback reason is returned as is (it already names the backend used).
 */
[[nodiscard]] inline std::string
describe_backend_choice(const backend_choice& choice)
{
    if(!choice.reason.empty()) return choice.reason;
    return choice.kind == backend_kind::perf ? "using the perf backend"
                                             : "using the signal backend";
}

}  // namespace cpu_snap::sampling
