// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler_compute_tool
{

class ToolSetUp
{
public:
    virtual void set_up() = 0;
    virtual ~ToolSetUp()  = default;
};

class EnvironmentSetUp : public ToolSetUp
{
public:
    EnvironmentSetUp() = default;

    void set_up() override;
    bool is_shell_target() const;
    /// Republish every cached env var whose key starts with "ROCPROF".
    void republish_rocprof_env() const;

protected:
    virtual std::vector<std::string> get_env_entries() const;
    virtual void set_env_var(const std::string& key, const std::string& value) const;
    void         build_env_cache();

    std::unordered_map<std::string, std::string> m_env_cache;
};

}  // namespace rocprofiler_compute_tool
