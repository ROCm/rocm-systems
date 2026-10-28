#pragma once

/**
 * @file self_exe.hpp
 * @brief Path of the running executable, resolved from /proc/self/exe.
 */

#include <array>
#include <cstddef>
#include <string>

#include <sys/types.h>  // NOLINT(misc-include-cleaner): provides ssize_t for ::readlink
#include <unistd.h>

namespace cpu_snap::core
{

/**
 * @brief Resolve /proc/self/exe via readlink.
 *
 * Paths longer than 4095 bytes are truncated.
 * @return Absolute path of the running executable, or empty on failure
 *         (e.g. inside a chroot that strips /proc).
 */
[[nodiscard]] inline std::string self_exe_path()
{
    constexpr std::size_t k_path_max = 4096;  // NOLINT(readability-identifier-naming)

    std::array<char, k_path_max> buf{};
    // NOLINTNEXTLINE(misc-include-cleaner): ssize_t from <sys/types.h>
    const ssize_t bytes_read = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if(bytes_read <= 0)
    {
        return {};
    }
    return { buf.data(), static_cast<std::size_t>(bytes_read) };
}

}  // namespace cpu_snap::core
