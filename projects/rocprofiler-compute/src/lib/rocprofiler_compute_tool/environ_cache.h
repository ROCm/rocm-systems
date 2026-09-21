// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rocprofiler_compute_tool
{
// Snapshots all ROCPROF_* variables directly from the process environment block
// at load time. Bypasses libc getenv() so LD_PRELOAD shims, such as shell-script
// launchers, cannot intercept the lookup.
class EnvironCache
{
public:
    static std::shared_ptr<const EnvironCache> instance();

    explicit EnvironCache(char** envp);
    virtual ~EnvironCache() = default;

    EnvironCache(const EnvironCache&)            = delete;
    EnvironCache& operator=(const EnvironCache&) = delete;
    EnvironCache(EnvironCache&&)                 = delete;
    EnvironCache& operator=(EnvironCache&&)      = delete;

    virtual std::optional<std::string_view> get(std::string_view name) const;

private:
    std::unordered_map<std::string, std::string> m_values;
};
}  // namespace rocprofiler_compute_tool
