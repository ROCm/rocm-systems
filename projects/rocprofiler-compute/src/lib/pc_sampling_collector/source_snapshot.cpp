// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshot.h"

#include <iostream>
#include <system_error>

using namespace rocprofiler_compute_tool;

std::optional<std::string> rocprofiler_compute_tool::parse_source_path(const std::string& comment)
{
    const auto pos = comment.rfind(':');
    if (pos == std::string::npos || pos == 0)
    {
        return std::nullopt;
    }

    std::string path = comment.substr(0, pos);
    if (path.empty())
    {
        return std::nullopt;
    }
    return path;
}

void rocprofiler_compute_tool::copy_source_files(const std::vector<std::string>& source_paths,
                                                 const std::filesystem::path&    dest_root)
{
    for (const auto& src : source_paths)
    {
        std::error_code ec;

        // Snapshot only real, regular files. symlink_status does not follow links,
        // so a symbolic link is not chased into an unrelated target.
        const auto status = std::filesystem::symlink_status(src, ec);
        if (ec || !std::filesystem::is_regular_file(status))
        {
            std::clog << "[rocprofiler-compute] [" << __FUNCTION__
                      << "] Skipping missing, unreadable, or non-regular source file: " << src << "\n";
            continue;
        }

        // Compose the destination beneath dest_root using the relative path so that an
        // absolute src does not collapse the join (operator/ replaces when RHS is absolute).
        // lexically_normal collapses any ".." segments so a crafted comment path cannot
        // escape dest_root; skip the file if it still resolves outside dest_root.
        const std::filesystem::path rel = std::filesystem::path(src).relative_path().lexically_normal();
        if (rel.empty() || *rel.begin() == "..")
        {
            std::clog << "[rocprofiler-compute] [" << __FUNCTION__
                      << "] Skipping source file with path escaping snapshot dir: " << src << "\n";
            continue;
        }
        const std::filesystem::path dest = dest_root / rel;

        std::filesystem::create_directories(dest.parent_path(), ec);
        if (ec)
        {
            std::clog << "[rocprofiler-compute] [" << __FUNCTION__
                      << "] Failed to create directory for source file: " << src
                      << ", error: " << ec.message() << "\n";
            continue;
        }

        // skip_symlinks: the symlink_status guard above already rejected a
        // symlinked final component, but pass it here too so copy_file never
        // dereferences a link (defence against a check->copy swap / symlinked
        // intermediate component).
        std::filesystem::copy_file(src,
                                   dest,
                                   std::filesystem::copy_options::overwrite_existing |
                                       std::filesystem::copy_options::skip_symlinks,
                                   ec);
        if (ec)
        {
            std::clog << "[rocprofiler-compute] [" << __FUNCTION__
                      << "] Failed to copy source file: " << src << ", error: " << ec.message() << "\n";
            continue;
        }
    }
}
