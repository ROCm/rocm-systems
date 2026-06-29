// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

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
    void snapshot(const std::set<std::filesystem::path>& source_paths,
                  const std::filesystem::path&           destination_root) override;
};
}  // namespace rocprofiler_compute_tool
