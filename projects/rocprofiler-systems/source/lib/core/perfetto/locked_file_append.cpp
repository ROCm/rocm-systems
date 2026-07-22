// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/locked_file_append.hpp"

#include "logger/debug.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/file.h>
#include <system_error>
#include <unistd.h>

namespace rocprofsys::core
{
locked_append_status
append_with_file_lock(const std::string& filename, const char* data, std::size_t size)
{
    // TIME_OUTPUT=ON puts the merged file under a timestamped subdirectory
    // that no other code may have created yet on this process.
    const auto parent = std::filesystem::path{ filename }.parent_path();
    if(!parent.empty())
    {
        std::error_code ec{};
        std::filesystem::create_directories(parent, ec);
        if(ec)
        {
            LOG_ERROR("append_with_file_lock: could not create directory '{}': {}",
                      parent.string(), ec.message());
            return locked_append_status::open_failed;
        }
    }

    int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if(fd < 0) return locked_append_status::open_failed;

    if(::flock(fd, LOCK_EX) != 0)
    {
        const int err = errno;
        ::close(fd);
        LOG_ERROR("append_with_file_lock: flock(LOCK_EX) failed on '{}': {}", filename,
                  std::strerror(err));
        return locked_append_status::lock_failed;
    }

    auto        status = locked_append_status::success;
    const char* ptr    = data;
    std::size_t remain = size;
    while(remain > 0)
    {
        const ssize_t n = ::write(fd, ptr, remain);
        if(n < 0)
        {
            const int err = errno;
            if(err == EINTR) continue;
            LOG_ERROR("append_with_file_lock: write to '{}' failed: {}", filename,
                      std::strerror(err));
            status = locked_append_status::write_failed;
            break;
        }
        ptr += n;
        remain -= static_cast<std::size_t>(n);
    }

    ::flock(fd, LOCK_UN);
    ::close(fd);
    return status;
}
}  // namespace rocprofsys::core
