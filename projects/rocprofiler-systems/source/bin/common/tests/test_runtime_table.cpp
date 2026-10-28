// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/runtime_table.hpp"

#include <gtest/gtest.h>

#include <cctype>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

using rocprofsys::runtime_table::expand_legacy_parallel;
using rocprofsys::runtime_table::find;
using rocprofsys::runtime_table::find_trace_name;
using rocprofsys::runtime_table::rows;

namespace
{
[[nodiscard]] std::string_view
skip_space(std::string_view text)
{
    while(!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
    {
        text.remove_prefix(1);
    }
    return text;
}

[[nodiscard]] bool
is_name_char(char ch)
{
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

// Advance past one C++ string literal. The returned view starts after the closing quote.
[[nodiscard]] std::optional<std::string_view>
skip_string(std::string_view text)
{
    if(text.empty() || text.front() != '"')
    {
        return std::nullopt;
    }
    text.remove_prefix(1);
    while(!text.empty())
    {
        if(text.front() == '\\')
        {
            if(text.size() < 2)
            {
                return std::nullopt;
            }
            text.remove_prefix(2);
            continue;
        }
        if(text.front() == '"')
        {
            text.remove_prefix(1);
            return text;
        }
        text.remove_prefix(1);
    }
    return std::nullopt;
}

// text begins at the comma before the setting's description strings.
[[nodiscard]] std::optional<bool>
default_after_description(std::string_view text)
{
    text = skip_space(text);
    if(text.empty() || text.front() != ',')
    {
        return std::nullopt;
    }
    text.remove_prefix(1);
    while(true)
    {
        text = skip_space(text);
        if(text.empty() || text.front() != '"')
        {
            break;
        }
        const auto rest = skip_string(text);
        if(!rest.has_value())
        {
            return std::nullopt;
        }
        text = *rest;
    }
    text = skip_space(text);
    if(text.empty() || text.front() != ',')
    {
        return std::nullopt;
    }
    text.remove_prefix(1);
    text = skip_space(text);
    if(text.starts_with("true"))
    {
        return true;
    }
    if(text.starts_with("false"))
    {
        return false;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<bool>
config_setting_default(std::string_view source, std::string_view variable)
{
    constexpr std::string_view k_macro = "ROCPROFSYS_CONFIG_SETTING";
    const auto  needle = std::string{ "env_vars::" } + std::string{ variable };
    std::size_t search = 0;
    while(search < source.size())
    {
        const auto macro = source.find(k_macro, search);
        if(macro == std::string_view::npos)
        {
            return std::nullopt;
        }
        const auto end = source.find(';', macro);
        if(end == std::string_view::npos)
        {
            return std::nullopt;
        }
        const auto body = source.substr(macro, end - macro);
        const auto at   = body.find(needle);
        const auto tail = at == std::string_view::npos ? std::string_view{}
                                                       : body.substr(at + needle.size());
        if(at != std::string_view::npos && (tail.empty() || !is_name_char(tail.front())))
        {
            return default_after_description(tail);
        }
        search = end + 1;
    }
    return std::nullopt;
}

[[nodiscard]] std::string
read_config_source()
{
    std::ifstream      file{ ROCPROFSYS_CONFIG_SOURCE };
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

[[nodiscard]] std::string_view
setting_symbol(std::string_view env_key)
{
    constexpr std::string_view k_prefix = "ROCPROFSYS_";
    return env_key.substr(k_prefix.size());
}
}  // namespace

TEST(runtime_table_test, library_defaults_match_config_cpp)
{
    const auto source = read_config_source();
    ASSERT_FALSE(source.empty());
    for(const auto& spec : rows())
    {
        const auto parsed = config_setting_default(source, setting_symbol(spec.env_key));
        ASSERT_TRUE(parsed.has_value()) << spec.env_key;
        EXPECT_EQ(*parsed, spec.library_default) << spec.env_key;
    }
}

TEST(runtime_table_test, legacy_parallel_is_the_original_four_runtimes)
{
    std::set<std::string> legacy;
    for(const auto& spec : rows())
    {
        if(spec.in_legacy_parallel)
        {
            legacy.emplace(spec.name);
        }
    }
    EXPECT_EQ(legacy, (std::set<std::string>{ "kokkos", "mpi", "openmp", "rccl" }));

    const auto all = expand_legacy_parallel("all");
    EXPECT_EQ(all, expand_legacy_parallel(""));
    EXPECT_EQ(all.size(), 4U);
    EXPECT_EQ(all.at(rocprofsys::env_vars::USE_MPIP), "true");
    EXPECT_EQ(all.at(rocprofsys::env_vars::USE_OMPT), "true");
    EXPECT_EQ(all.at(rocprofsys::env_vars::USE_KOKKOSP), "true");
    EXPECT_EQ(all.at(rocprofsys::env_vars::USE_RCCLP), "true");
    EXPECT_EQ(all.count(rocprofsys::env_vars::USE_UCX), 0U);
    EXPECT_EQ(all.count(rocprofsys::env_vars::USE_OPENSHMEM), 0U);
}

TEST(runtime_table_test, legacy_parallel_ignores_case_of_all_and_unknown_names)
{
    EXPECT_TRUE(expand_legacy_parallel("ALL").empty());
    EXPECT_TRUE(expand_legacy_parallel("ucx,shmem,openshmem,rccl_api").empty());

    const auto listed = expand_legacy_parallel("mpi,openmp");
    EXPECT_EQ(listed.size(), 2U);
    EXPECT_EQ(listed.at(rocprofsys::env_vars::USE_MPIP), "true");
    EXPECT_EQ(listed.at(rocprofsys::env_vars::USE_OMPT), "true");

    // "all" is only special when it is the entire value, not one token.
    const auto mixed = expand_legacy_parallel("all,mpi");
    EXPECT_EQ(mixed.size(), 1U);
    EXPECT_EQ(mixed.at(rocprofsys::env_vars::USE_MPIP), "true");
}

TEST(runtime_table_test, aliases_select_legacy_rows_and_trace_overrides_do_not)
{
    const auto* openmp = find("ompt");
    ASSERT_NE(openmp, nullptr);
    EXPECT_EQ(openmp->name, "openmp");
    EXPECT_EQ(expand_legacy_parallel("mpip,omp,kokkosp,rcclp").size(), 4U);

    EXPECT_EQ(find("rccl_api"), nullptr);
    const auto* rccl = find_trace_name("rccl_api");
    ASSERT_NE(rccl, nullptr);
    EXPECT_EQ(rccl->name, "rccl");
    EXPECT_EQ(rccl->rocm_domain, "rccl_api");

    const auto* from_override = find_trace_name("ompt");
    ASSERT_NE(from_override, nullptr);
    EXPECT_EQ(from_override->name, "openmp");
    EXPECT_EQ(from_override->rocm_domain, "ompt");

    const auto* shmem = find("shmem");
    ASSERT_NE(shmem, nullptr);
    EXPECT_EQ(shmem->name, "openshmem");
    EXPECT_FALSE(shmem->in_legacy_parallel);
}
