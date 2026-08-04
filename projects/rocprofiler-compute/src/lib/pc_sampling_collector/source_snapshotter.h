// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "filesystem_wrapper.h"

#include <filesystem>
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
    std::optional<std::filesystem::path> get_destination_path(
        const std::filesystem::path& absolute_source_path,
        const std::filesystem::path& destination_root) const;

    bool is_copyable(const std::filesystem::path& source_path,
                     std::filesystem::path&       absolute_source_path);
    bool create_destination_parent_directory(const std::filesystem::path& destination_path);
    void copy_source(const std::filesystem::path& source_path,
                     const std::filesystem::path& destination_path);

    filesystem_wrapper_t::ptr m_filesystem;
};
}  // namespace rocprofiler_compute_tool
