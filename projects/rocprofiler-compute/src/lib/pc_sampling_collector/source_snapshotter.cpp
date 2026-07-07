// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshotter.h"

#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

source_snapshotter_t::ptr source_snapshotter_t::create()
{
    return std::make_shared<source_snapshotter_impl_t>();
}

std::optional<std::filesystem::path> source_snapshotter_impl_t::get_destination_path(
    const std::filesystem::path& absolute_source_path,
    const std::filesystem::path& destination_root) const
{
    std::error_code error;
    const auto canonical_source_path = m_filesystem->weakly_canonical(absolute_source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: " << absolute_source_path
                  << ": failed to get canonical source path: " << error.message() << '\n';
        return std::nullopt;
    }

    return destination_root / m_filesystem->relative_path(canonical_source_path);
}

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
        if (!is_copyable(source_path, absolute_source_path))
            continue;

        const auto destination_path = get_destination_path(absolute_source_path, destination_root);
        if (!destination_path)
            continue;

        copy_source(source_path, *destination_path);
    }
}

bool source_snapshotter_impl_t::is_copyable(const std::filesystem::path& source_path,
                                            std::filesystem::path&       absolute_source_path)
{
    if (source_path.empty())
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping empty file: " << source_path
                  << '\n';
        return false;
    }

    std::error_code error;
    absolute_source_path = m_filesystem->absolute(source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: " << source_path
                  << ": " << error.message() << '\n';
        return false;
    }

    const auto source_status = m_filesystem->status(absolute_source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: " << absolute_source_path
                  << ": " << error.message() << '\n';
        return false;
    }

    if (!m_filesystem->exists(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping missing file: "
                  << absolute_source_path << '\n';
        return false;
    }

    if (!m_filesystem->is_regular_file(source_status))
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping non-regular file: "
                  << absolute_source_path << '\n';
        return false;
    }

    return true;
}

bool source_snapshotter_impl_t::create_destination_parent_directory(const std::filesystem::path& destination_path)
{
    if (!m_filesystem->has_parent_path(destination_path))
        return true;

    const auto      parent_path = m_filesystem->parent_path(destination_path);
    std::error_code error;
    m_filesystem->create_directories(parent_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to create destination "
                  << "directory " << parent_path << ": " << error.message() << '\n';
        return false;
    }

    return true;
}

void source_snapshotter_impl_t::copy_source(const std::filesystem::path& source_path,
                                            const std::filesystem::path& destination_path)
{
    if (!create_destination_parent_directory(destination_path))
        return;

    std::error_code error;
    m_filesystem->copy_file(source_path,
                            destination_path,
                            std::filesystem::copy_options::overwrite_existing,
                            error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to copy " << source_path
                  << " to " << destination_path << ": " << error.message() << '\n';
        return;
    }
}
