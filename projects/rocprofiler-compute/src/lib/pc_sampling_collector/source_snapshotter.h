// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "filesystem_wrapper.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>

namespace rocprofiler_compute_tool
{
class source_snapshotter_t
{
public:
    using ptr = std::shared_ptr<source_snapshotter_t>;
    static ptr create();

    virtual ~source_snapshotter_t()                                                = default;
    virtual void snapshot(const std::set<std::filesystem::path>& source_paths,
                          const std::filesystem::path&           destination_root) = 0;
};

class source_snapshotter_impl_t : public source_snapshotter_t
{
public:
    source_snapshotter_impl_t();
    explicit source_snapshotter_impl_t(filesystem_wrapper_t::ptr filesystem);

    void snapshot(const std::set<std::filesystem::path>& source_paths,
                  const std::filesystem::path&           destination_root) override;

private:
    // Each raw DWARF path from the disassembly, against the canonical path its
    // snapshot copy is filed under. Only files that were copied appear.
    using source_path_map_t = std::map<std::filesystem::path, std::filesystem::path>;

    std::optional<std::filesystem::path> get_canonical_source_path(
        const std::filesystem::path& absolute_source_path) const;

    bool is_copyable(const std::filesystem::path& source_path,
                     std::filesystem::path&       absolute_source_path);
    bool create_destination_parent_directory(const std::filesystem::path& destination_path);
    bool copy_source(const std::filesystem::path& source_path,
                     const std::filesystem::path& destination_path);
    void write_source_path_map(const source_path_map_t&     source_path_map,
                               const std::filesystem::path& destination_root);

    filesystem_wrapper_t::ptr m_filesystem;
};
}  // namespace rocprofiler_compute_tool
