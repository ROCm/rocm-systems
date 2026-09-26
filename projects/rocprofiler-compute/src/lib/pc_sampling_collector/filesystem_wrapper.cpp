// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "filesystem_wrapper.h"

using namespace rocprofiler_compute_tool;

filesystem_wrapper_t::ptr filesystem_wrapper_t::create()
{
    return std::make_shared<filesystem_wrapper_impl_t>();
}

std::filesystem::path filesystem_wrapper_impl_t::absolute(const std::filesystem::path& path,
                                                          std::error_code&             error)
{
    return std::filesystem::absolute(path, error);
}

std::filesystem::file_status filesystem_wrapper_impl_t::status(const std::filesystem::path& path,
                                                               std::error_code&             error)
{
    return std::filesystem::status(path, error);
}

bool filesystem_wrapper_impl_t::create_directories(const std::filesystem::path& path,
                                                   std::error_code&             error)
{
    return std::filesystem::create_directories(path, error);
}

bool filesystem_wrapper_impl_t::copy_file(const std::filesystem::path&  source,
                                          const std::filesystem::path&  destination,
                                          std::filesystem::copy_options options,
                                          std::error_code&              error)
{
    return std::filesystem::copy_file(source, destination, options, error);
}

bool filesystem_wrapper_impl_t::exists(const std::filesystem::file_status& status)
{
    return std::filesystem::exists(status);
}

bool filesystem_wrapper_impl_t::is_regular_file(const std::filesystem::file_status& status)
{
    return std::filesystem::is_regular_file(status);
}

bool filesystem_wrapper_impl_t::has_parent_path(const std::filesystem::path& path)
{
    return path.has_parent_path();
}

std::filesystem::path filesystem_wrapper_impl_t::parent_path(const std::filesystem::path& path)
{
    return path.parent_path();
}

std::filesystem::path filesystem_wrapper_impl_t::weakly_canonical(const std::filesystem::path& path,
                                                                  std::error_code& error)
{
    return std::filesystem::weakly_canonical(path, error);
}

std::filesystem::path filesystem_wrapper_impl_t::relative_path(const std::filesystem::path& path)
{
    return path.relative_path();
}
