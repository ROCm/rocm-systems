// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "tool_setup.h"

#include <stdlib.h>

#include <string>
#include <string_view>
#include <vector>

extern "C" char** environ;

using namespace rocprofiler_compute_tool;

std::vector<std::string> EnvironmentSetUp::get_env_entries() const
{
    std::vector<std::string> entries;
    for (char** entry_ptr = ::environ; entry_ptr != nullptr && *entry_ptr != nullptr; ++entry_ptr)
    {
        entries.emplace_back(*entry_ptr);
    }
    return entries;
}

void EnvironmentSetUp::set_env_var(const std::string& key, const std::string& value) const
{
    ::setenv(key.c_str(), value.c_str(), /*overwrite=*/1);
}

void EnvironmentSetUp::build_env_cache()
{
    m_env_cache.clear();
    for (const auto& entry : get_env_entries())
    {
        const auto eq_pos = entry.find('=');
        if (eq_pos == std::string::npos)
            continue;

        // Store the key and value in the cache
        m_env_cache.try_emplace(entry.substr(0, eq_pos), entry.substr(eq_pos + 1));
    }
}

bool EnvironmentSetUp::is_shell_target() const
{
    const auto it = m_env_cache.find("ROCPROF_SHELL_TARGET");
    return it != m_env_cache.end() && it->second == "1";
}

void EnvironmentSetUp::republish_rocprof_env() const
{
    constexpr std::string_view rocprof_prefix = "ROCPROF";

    for (const auto& [key, value] : m_env_cache)
    {
        if (std::string_view{key}.substr(0, rocprof_prefix.size()) != rocprof_prefix)
            continue;

        set_env_var(key, value);
    }
}

void EnvironmentSetUp::set_up()
{
    build_env_cache();
    if (is_shell_target())
    {
        republish_rocprof_env();
    }
}
