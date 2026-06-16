// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "source_snapshot.h"

#include <memory>
#include <set>
#include <system_error>

using namespace rocprofiler_compute_tool;

std::shared_ptr<source_snapshot_t> source_snapshot_t::create()
{
    return std::make_shared<source_snapshot_impl_t>();
}

std::optional<std::string> rocprofiler_compute_tool::source_snapshot_impl_t::parse_ref(
    const std::string& comment) const
{
    if (comment.empty())
    {
        return std::nullopt;
    }

    const auto pos = comment.rfind(':');
    if (pos == std::string::npos || pos == 0)
    {
        return std::nullopt;
    }

    return comment.substr(0, pos);
}

size_t rocprofiler_compute_tool::source_snapshot_impl_t::snapshot(
    const std::vector<std::string>& source_refs,
    const std::filesystem::path&    output_root,
    const std::filesystem::path&    allowed_root) const
{
    const std::filesystem::path sources_root = output_root / "code_obj_sources";

    std::set<std::string> unique_refs;
    for (const auto& ref : source_refs)
    {
        unique_refs.insert(ref);
    }

    // Canonicalize the roots so the bounds checks below resolve symlinks.
    std::error_code root_ec;
    const auto      canon_allowed_root = std::filesystem::weakly_canonical(allowed_root, root_ec);
    const auto      canon_sources_root = std::filesystem::weakly_canonical(sources_root, root_ec);

    const auto is_inside = [](const std::filesystem::path& base, const std::filesystem::path& candidate)
    {
        const auto rel = candidate.lexically_relative(base);
        return !rel.empty() && *rel.begin() != "..";
    };

    size_t copied = 0;
    for (const auto& ref : unique_refs)
    {
        std::error_code             ec;
        const std::filesystem::path src{ref};

        // Reject a symlinked source outright (lstat, no follow): copy_file follows
        // symlinks at the source, so a symlink could redirect the read outside
        // allowed_root after the containment check (TOCTOU). Refusing symlinks
        // closes that window at the cost of not snapshotting symlinked sources.
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(src, ec)) || ec)
        {
            continue;
        }

        // Resolve symlinks and require the source inside allowed_root.
        const auto canon_src = std::filesystem::weakly_canonical(src, ec);
        if (ec || canon_allowed_root.empty() || !is_inside(canon_allowed_root, canon_src))
        {
            continue;
        }
        if (!std::filesystem::is_regular_file(canon_src, ec) || ec)
        {
            continue;
        }

        std::string relative = ref;
        while (!relative.empty() && relative.front() == '/')
        {
            relative.erase(relative.begin());
        }

        const std::filesystem::path dst = sources_root / relative;

        // Reject destinations escaping code_obj_sources, canonicalizing the existing
        // prefix so a planted symlink cannot redirect the write out of the snapshot dir.
        const auto canon_dst_parent = std::filesystem::weakly_canonical(dst.parent_path(), ec);
        if (ec || canon_sources_root.empty() || !is_inside(canon_sources_root, canon_dst_parent))
        {
            continue;
        }

        std::filesystem::create_directories(dst.parent_path(), ec);
        if (ec)
        {
            continue;
        }

        // skip_existing avoids following/clobbering a pre-planted destination.
        const bool ok = std::filesystem::copy_file(canon_src,
                                                   dst,
                                                   std::filesystem::copy_options::skip_existing,
                                                   ec);
        if (ok && !ec)
        {
            ++copied;
        }
    }

    return copied;
}
