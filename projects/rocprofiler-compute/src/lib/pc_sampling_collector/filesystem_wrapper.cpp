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
