// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshotter.h"

#include <iostream>

using namespace rocprofiler_compute_tool;

namespace
{
bool is_copyable(const std::filesystem::path& source_path)
{
    if (source_path.empty())
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping empty file: "
                  << source_path << '\n';
        return false;
    }

    std::error_code error;
    const auto      source_status = std::filesystem::status(source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: "
                  << source_path << ": " << error.message() << '\n';
        return false;
    }

    if (!std::filesystem::exists(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping missing file: "
                  << source_path << '\n';
        return false;
    }

    if (!std::filesystem::is_regular_file(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping non-regular file: "
                  << source_path << '\n';
        return false;
    }

    return true;
}

std::filesystem::path get_destination_path(const std::filesystem::path& source_path,
                                                    const std::filesystem::path& destination_root)
{
    return destination_root / source_path.relative_path();
}

bool create_destination_parent_directory(const std::filesystem::path& destination_path)
{
    if (!destination_path.has_parent_path())
        return true;

    std::error_code error;
    std::filesystem::create_directories(destination_path.parent_path(), error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to create destination "
                  << "directory "
                  << destination_path.parent_path() << ": " << error.message() << '\n';
        return false;
    }

    return true;
}

void copy_source(const std::filesystem::path& source_path,
                 const std::filesystem::path& destination_path)
{
    if (!create_destination_parent_directory(destination_path))
        return;

    std::error_code error;
    std::filesystem::copy_file(
        source_path, destination_path, std::filesystem::copy_options::overwrite_existing, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to copy "
                  << source_path << " to " << destination_path << ": " << error.message() << '\n';
        return;
    }
}
}  // namespace

void source_snapshotter_impl_t::snapshot(const std::set<std::filesystem::path>& source_paths,
                                         const std::filesystem::path&           destination_root)
{
    for (const auto& source_path : source_paths)
    {
        if (!is_copyable(source_path))
            continue;

        const auto destination_path = get_destination_path(source_path, destination_root);
        copy_source(source_path, destination_path);
    }
}
