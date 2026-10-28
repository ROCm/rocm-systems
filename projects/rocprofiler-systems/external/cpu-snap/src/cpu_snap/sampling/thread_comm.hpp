#pragma once

/**
 * @file thread_comm.hpp
 * @brief Thread / process identity helpers backed by /proc/self.
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>  // NOLINT(misc-include-cleaner): provides pid_t below
#include <unistd.h>

namespace cpu_snap::sampling::utils
{

/**
 * Read the kernel-assigned name (comm) of a thread.
 *
 * @param tid Linux thread id.
 * @return    The contents of @c /proc/self/task/<tid>/comm trimmed of
 *            the trailing newline, or @c "tid-<N>" when the file is
 *            missing (thread exited) or empty.
 */
inline std::string
read_thread_comm(pid_t tid)  // NOLINT(misc-include-cleaner): pid_t from <sys/types.h>
{
    std::ostringstream path;
    path << "/proc/self/task/" << tid << "/comm";
    std::ifstream ifs(path.str());
    std::string   name;
    if(ifs)
    {
        std::getline(ifs, name);
    }
    if(name.empty())
    {
        name = "tid-" + std::to_string(tid);
    }
    return name;
}

/**
 * Read the kernel-assigned name (comm) of the current process.
 *
 * @return Contents of @c /proc/self/comm with trailing newline/CR
 *         trimmed, or an empty string on read failure.
 */
[[nodiscard]] inline std::string
read_proc_self_comm() noexcept
{
    try
    {
        std::string out;
        if(FILE* file_ptr = std::fopen("/proc/self/comm", "r"))
        {
            std::array<char, 64> buf{};
            if(std::fgets(buf.data(), buf.size(), file_ptr) != nullptr)
            {
                out = buf.data();
                while(!out.empty() && (out.back() == '\n' || out.back() == '\r'))
                {
                    out.pop_back();
                }
            }
            (void) std::fclose(file_ptr);
        }
        return out;
    } catch(...)  // best-effort /proc read: an absent thread name is not an error
    {
        return {};
    }
}

/**
 * Extract the process start time (field 22, clock ticks since boot) from the
 * contents of a /proc/<pid>/task/<tid>/stat file.
 *
 * @param stat Raw stat line; the comm field may contain spaces and parentheses.
 * @return The start time, or std::nullopt when the line is malformed.
 */
[[nodiscard]] inline std::optional<std::uint64_t>
parse_stat_start_ticks(std::string_view stat) noexcept
{
    const auto close = stat.rfind(')');
    if(close == std::string_view::npos) return std::nullopt;

    constexpr int k_fields_before_start_time = 19;
    std::string_view rest                    = stat.substr(close + 1);
    for(int i = 0; i < k_fields_before_start_time; ++i)
    {
        rest.remove_prefix(std::min(rest.find_first_not_of(' '), rest.size()));
        const auto end = rest.find(' ');
        if(end == std::string_view::npos) return std::nullopt;
        rest.remove_prefix(end);
    }
    rest.remove_prefix(std::min(rest.find_first_not_of(' '), rest.size()));

    std::uint64_t value = 0;
    const auto [ptr, ec] = std::from_chars(rest.data(), rest.data() + rest.size(), value);
    if(ec != std::errc{}) return std::nullopt;
    return value;
}

/**
 * Read the start time of a thread of this process.
 *
 * Together with the tid it identifies a thread uniquely even after the
 * kernel recycles the tid.
 *
 * @param tid Linux thread id.
 * @return Start time in clock ticks since boot, or std::nullopt when the
 *         thread is gone or its stat file is unreadable.
 */
[[nodiscard]] inline std::optional<std::uint64_t>
read_thread_start_ticks(pid_t tid)  // NOLINT(misc-include-cleaner): pid_t from <sys/types.h>
{
    std::array<char, 40> path{};
    std::snprintf(path.data(), path.size(), "/proc/self/task/%d/stat", static_cast<int>(tid));
    const int fd = ::open(path.data(), O_RDONLY | O_CLOEXEC);
    if(fd < 0) return std::nullopt;
    std::array<char, 512> buf{};
    const ssize_t         got = ::read(fd, buf.data(), buf.size());
    ::close(fd);
    if(got <= 0) return std::nullopt;
    const std::string_view line(buf.data(), static_cast<std::size_t>(got));
    return parse_stat_start_ticks(line);
}

/**
 * Enumerate every numeric task entry under @p task_dir.
 *
 * @param task_dir Directory of per-thread entries (normally /proc/self/task).
 * @return The tids found (non-numeric entries are skipped), or std::nullopt
 *         when the directory cannot be opened or iterated to the end, in
 *         which case a partial listing must not be trusted.
 */
[[nodiscard]] inline std::optional<std::vector<pid_t>>
try_list_thread_tids(const std::filesystem::path& task_dir)
{
    std::error_code                     ec;
    std::filesystem::directory_iterator it(task_dir, ec);
    if(ec) return std::nullopt;

    std::vector<pid_t> out;
    for(const std::filesystem::directory_iterator end; it != end; it.increment(ec))
    {
        if(ec) return std::nullopt;
        const auto name = it->path().filename().string();
        try
        {
            out.push_back(static_cast<pid_t>(std::stoi(name)));
        } catch(...)  // NOLINT(bugprone-empty-catch): non-numeric task names skipped
        {
        }
    }
    if(ec) return std::nullopt;
    return out;
}

/**
 * Enumerate every thread id under /proc/self/task.
 *
 * @return The currently-live tids, or std::nullopt when the listing failed
 *         (e.g. EMFILE).
 */
[[nodiscard]] inline std::optional<std::vector<pid_t>>
try_list_self_thread_tids()
{
    return try_list_thread_tids("/proc/self/task");
}

/**
 * Enumerate every thread id under /proc/self/task.
 *
 * @return Vector of currently-live tids in this process; empty when the
 *         listing failed. Callers that must tell failure from an empty
 *         process use try_list_self_thread_tids().
 */
[[nodiscard]] inline std::vector<pid_t>
list_self_thread_tids()
{
    return try_list_self_thread_tids().value_or(std::vector<pid_t>{});
}

}  // namespace cpu_snap::sampling::utils
