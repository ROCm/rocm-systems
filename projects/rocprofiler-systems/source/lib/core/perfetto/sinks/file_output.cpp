// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/sinks/file_output.hpp"

#include "core/output_file_registry.hpp"
#include "logger/debug.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace rocprofsys::core
{
namespace
{
// TIME_OUTPUT=ON puts the output under a timestamped subdirectory that no other
// code may have created yet on this process.
[[nodiscard]] bool
ensure_parent_directory(const std::string& filename)
{
    const auto parent = std::filesystem::path{ filename }.parent_path();
    if(parent.empty())
    {
        return true;
    }

    std::error_code dir_ec{};
    std::filesystem::create_directories(parent, dir_ec);
    if(dir_ec)
    {
        LOG_ERROR("could not create directory '{}': {}", parent.string(),
                  dir_ec.message());
        return false;
    }
    return true;
}
}  // namespace

bool
write_proto_to(const std::string& filename, const char* data, std::size_t size,
               output_file_registry& registry)
{
    if(!ensure_parent_directory(filename))
    {
        return false;
    }

    std::ofstream ofs{ filename, std::ios::out | std::ios::binary };
    if(!ofs.is_open() || !ofs.good())
    {
        return false;
    }

    ofs.write(data, static_cast<std::streamsize>(size));
    registry.register_file(filename, output_format::perfetto);
    return true;
}

locked_append_status
append_with_file_lock(const std::string& filename, const char* data, std::size_t size)
{
    if(!ensure_parent_directory(filename))
    {
        return locked_append_status::open_failed;
    }

    const int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
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
