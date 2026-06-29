// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshotter.h"

#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

namespace
{
bool is_copyable(filesystem_wrapper_t&        filesystem,
                 const std::filesystem::path& source_path,
                 std::filesystem::path&       absolute_source_path)
{
    if (source_path.empty())
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping empty file: "
                  << source_path << '\n';
        return false;
    }

    std::error_code error;
    absolute_source_path = filesystem.absolute(source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: "
                  << source_path << ": " << error.message() << '\n';
        return false;
    }

    const auto source_status = filesystem.status(absolute_source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: "
                  << absolute_source_path << ": " << error.message() << '\n';
        return false;
    }

    if (!std::filesystem::exists(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping missing file: "
                  << absolute_source_path << '\n';
        return false;
    }

    if (!std::filesystem::is_regular_file(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping non-regular file: "
                  << absolute_source_path << '\n';
        return false;
    }

    return true;
}

std::filesystem::path get_destination_path(const std::filesystem::path& source_path,
                                           const std::filesystem::path& destination_root)
{
    return destination_root / source_path.relative_path();
}

bool create_destination_parent_directory(filesystem_wrapper_t&        filesystem,
                                         const std::filesystem::path& destination_path)
{
    if (!destination_path.has_parent_path())
        return true;

    std::error_code error;
    filesystem.create_directories(destination_path.parent_path(), error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to create destination "
                  << "directory "
                  << destination_path.parent_path() << ": " << error.message() << '\n';
        return false;
    }

    return true;
}

void copy_source(filesystem_wrapper_t&        filesystem,
                 const std::filesystem::path& source_path,
                 const std::filesystem::path& destination_path)
{
    if (!create_destination_parent_directory(filesystem, destination_path))
        return;

    std::error_code error;
    filesystem.copy_file(
        source_path, destination_path, std::filesystem::copy_options::overwrite_existing, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to copy "
                  << source_path << " to " << destination_path << ": " << error.message() << '\n';
        return;
    }
}
}  // namespace

source_snapshotter_impl_t::source_snapshotter_impl_t()
    : source_snapshotter_impl_t(filesystem_wrapper_t::create())
{
}

source_snapshotter_impl_t::source_snapshotter_impl_t(filesystem_wrapper_t::ptr filesystem)
    : m_filesystem(std::move(filesystem))
{
}

void source_snapshotter_impl_t::snapshot(const std::set<std::filesystem::path>& source_paths,
                                         const std::filesystem::path&           destination_root)
{
    for (const auto& source_path : source_paths)
    {
        std::filesystem::path absolute_source_path;
        if (!is_copyable(*m_filesystem, source_path, absolute_source_path))
            continue;

        const auto destination_path = get_destination_path(absolute_source_path, destination_root);
        copy_source(*m_filesystem, absolute_source_path, destination_path);
    }
}
