// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "system.hpp"

#include "common/path.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <unistd.h>

namespace rocprofsys::rocsys
{
int
default_system::set_env(const char* key, const char* value) const
{
    if(::setenv(key, value, 1) == 0)
    {
        return 0;
    }
    return errno == 0 ? -1 : errno;
}

int
default_system::unset_env(const char* key) const
{
    if(::unsetenv(key) == 0)
    {
        return 0;
    }
    return errno == 0 ? -1 : errno;
}

bool
default_system::exists(const char* path) const
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

std::string
default_system::executable_directory() const
{
    std::error_code error;
    const auto      canon = std::filesystem::canonical("/proc/self/exe", error);
    if(error)
    {
        return {};
    }
    return common::path::parent_path(canon.string());
}

std::vector<std::string>
default_system::environ() const
{
    std::vector<std::string> entries;
    for(char** entry = ::environ; entry != nullptr && *entry != nullptr; ++entry)
    {
        entries.emplace_back(*entry);
    }
    return entries;
}

int
default_system::run_tool(int argc, char** argv, common_utils::tool_mode mode) const
{
    return common_utils::run_tool(argc, argv, mode);
}

int
default_system::exec(const char* path, char** argv, char** envp) const
{
    ::execve(path, argv, envp);
    return errno == 0 ? EIO : errno;
}

void
default_system::write_error(std::string_view message) const
{
    std::cerr << message;
}

}  // namespace rocprofsys::rocsys
