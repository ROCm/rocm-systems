#pragma once

/**
 * @file triggers/perf/open_error.hpp
 * @brief Actionable diagnostics for a failed perf_event_open.
 *
 * Header-only. C++20.
 */

#include <cerrno>
#include <cstdio>
#include <optional>
#include <string>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): pid_t (include-cleaner mis-attributes)

namespace cpu_snap::sampling::triggers::perf
{

/**
 * @brief Current value of /proc/sys/kernel/perf_event_paranoid.
 * @return The value, or nullopt when the file cannot be read or parsed.
 */
[[nodiscard]] inline std::optional<int> read_perf_event_paranoid()
{
    std::FILE* file = std::fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if(file == nullptr)
    {
        return std::nullopt;
    }
    int         value  = 0;
    const bool  parsed = std::fscanf(file, "%d", &value) == 1;
    std::fclose(file);
    return parsed ? std::optional<int>{ value } : std::nullopt;
}

/**
 * @brief Context string for the environment_error thrown when
 *        perf_event_open(@p tid) fails with @p err.
 *
 * For EACCES / EPERM the text names perf_event_paranoid (with its current
 * value when known) and the unprivileged signal backend; other errors get
 * only the call description.
 *
 * @param tid      Thread the event was opened for.
 * @param err      errno from the failed call.
 * @param paranoid Current perf_event_paranoid value, if known.
 */
[[nodiscard]] inline std::string
describe_perf_open_failure(pid_t tid, int err, std::optional<int> paranoid)
{
    std::string text = "perf_event_open(tid=" + std::to_string(tid) + ")";
    if(err != EACCES && err != EPERM)
    {
        return text;
    }
    text += " denied by /proc/sys/kernel/perf_event_paranoid";
    if(paranoid.has_value())
    {
        text += " (currently " + std::to_string(*paranoid) + ")";
    }
    text +=
        "; lower it to 2 or less (1 or less to include kernel frames) or grant "
        "CAP_PERFMON, or use backend_kind::signal, which needs no privileges";
    return text;
}

}  // namespace cpu_snap::sampling::triggers::perf
