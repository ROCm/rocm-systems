// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{
class source_snapshot_t
{
public:
    static std::shared_ptr<source_snapshot_t> create();

    virtual ~source_snapshot_t() = default;

    // Returns the substring before the last ':' (the path); std::nullopt if there is none.
    virtual std::optional<std::string> parse_ref(const std::string& comment) const = 0;

    // Copies each unique ref under output_root/"code_obj_sources"/<ref path>; returns the count.
    // Refs are untrusted, so one is read only if it resolves inside allowed_root and its
    // destination stays within code_obj_sources; others are skipped without throwing.
    virtual size_t snapshot(
        const std::vector<std::string>& source_refs,
        const std::filesystem::path&    output_root,
        const std::filesystem::path&    allowed_root = std::filesystem::current_path()) const = 0;
};

class source_snapshot_impl_t : public source_snapshot_t
{
public:
    std::optional<std::string> parse_ref(const std::string& comment) const override;
    size_t snapshot(const std::vector<std::string>& source_refs,
                    const std::filesystem::path&    output_root,
                    const std::filesystem::path& allowed_root = std::filesystem::current_path()) const override;
};
}  // namespace rocprofiler_compute_tool
