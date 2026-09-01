// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshotter.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr auto kSourcePathsKey = "source_paths";
}  // namespace

source_snapshotter_t::ptr source_snapshotter_t::create()
{
    return std::make_shared<source_snapshotter_impl_t>();
}

std::string rocprofiler_compute_tool::serialize_source_path_map(const source_path_map_t& source_path_map)
{
    auto source_paths = nlohmann::json::object();
    for (const auto& [raw_path, canonical_path] : source_path_map)
        source_paths[raw_path.string()] = canonical_path.string();

    return nlohmann::json{{kSourcePathsKey, std::move(source_paths)}}.dump();
}

std::optional<std::filesystem::path> source_snapshotter_impl_t::get_canonical_source_path(
    const std::filesystem::path& absolute_source_path) const
{
    std::error_code error;
    const auto canonical_source_path = m_filesystem->weakly_canonical(absolute_source_path, error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Skipping file: " << absolute_source_path
                  << ": failed to get canonical source path: " << error.message() << '\n';
        return std::nullopt;
    }

    return canonical_source_path;
}

source_snapshotter_impl_t::source_snapshotter_impl_t()
    : source_snapshotter_impl_t(filesystem_wrapper_t::create())
{
}

source_snapshotter_impl_t::source_snapshotter_impl_t(filesystem_wrapper_t::ptr filesystem)
    : m_filesystem(std::move(filesystem))
{
}

source_path_map_t source_snapshotter_impl_t::snapshot(const std::set<std::filesystem::path>& source_paths,
                                                      const std::filesystem::path& destination_root)
{
    source_path_map_t source_path_map;
    for (const auto& source_path : source_paths)
    {
        std::filesystem::path absolute_source_path;
        if (!is_copyable(source_path, absolute_source_path))
            continue;

        const auto canonical_source_path = get_canonical_source_path(absolute_source_path);
        if (!canonical_source_path)
            continue;

        const auto destination_path =
            destination_root / m_filesystem->relative_path(*canonical_source_path);
        if (!copy_source(source_path, destination_path))
            continue;

        source_path_map.emplace(source_path, *canonical_source_path);
    }

    return source_path_map;
}

void source_snapshotter_impl_t::write_source_path_map(const source_path_map_t&     source_path_map,
                                                      const std::filesystem::path& output_file_path)
{
    if (source_path_map.empty())
        return;

    if (!create_destination_parent_directory(output_file_path))
        return;

    std::ofstream output_file(output_file_path, std::ios::out);
    if (!output_file.is_open())
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to open source map file: "
                  << output_file_path << '\n';
        return;
    }

    output_file << serialize_source_path_map(source_path_map);
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

bool source_snapshotter_impl_t::copy_source(const std::filesystem::path& source_path,
                                            const std::filesystem::path& destination_path)
{
    if (!create_destination_parent_directory(destination_path))
        return false;

    std::error_code error;
    m_filesystem->copy_file(source_path,
                            destination_path,
                            std::filesystem::copy_options::overwrite_existing,
                            error);
    if (error)
    {
        std::clog << "[rocprofiler-compute] [source_snapshotter] Failed to copy " << source_path
                  << " to " << destination_path << ": " << error.message() << '\n';
        return false;
    }

    return true;
}
