// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "filesystem_wrapper.h"

#include <filesystem>
#include <set>

namespace rocprofiler_compute_tool
{
class source_snapshotter_t
{
public:
    virtual ~source_snapshotter_t() = default;
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
    filesystem_wrapper_t::ptr m_filesystem;
};
}  // namespace rocprofiler_compute_tool
