// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/agent_info.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// agent_info::to_json_string — output is stored verbatim as
// rocpd_info_agent.extdata, so it must be valid JSON (no SQL-style "" escaping)
// ═══════════════════════════════════════════════════════════════════════════

namespace
{
constexpr std::size_t k_max_json_length = 128;

rocprofiler_agent_v0_t
make_gpu_agent_data()
{
    rocprofiler_agent_v0_t agent_data{};
    agent_data.type                 = ROCPROFILER_AGENT_TYPE_GPU;
    agent_data.name                 = "gfx942";
    agent_data.vendor_name          = "AMD";
    agent_data.product_name         = "Instinct MI300X";
    agent_data.model_name           = "aqua_vanjaram";
    agent_data.logical_node_type_id = 3;
    return agent_data;
}
}  // namespace

TEST(agent_info_json_test, to_json_string_is_valid_json)
{
    auto const agent_data = make_gpu_agent_data();
    auto const json_str   = rocprofsys::agent_info::to_json_string(agent_data);

    ASSERT_TRUE(nlohmann::json::accept(json_str))
        << json_str.substr(0, k_max_json_length);

    auto const parsed = nlohmann::json::parse(json_str);
    EXPECT_TRUE(parsed.is_object());
    EXPECT_EQ(parsed.at("name"), "gfx942");
    EXPECT_EQ(parsed.at("vendor_name"), "AMD");
    EXPECT_EQ(parsed.at("product_name"), "Instinct MI300X");
    EXPECT_EQ(parsed.at("model_name"), "aqua_vanjaram");
    EXPECT_EQ(parsed.at("gpu_index"), 3);
    EXPECT_TRUE(parsed.at("capability").is_object());
    EXPECT_TRUE(parsed.at("caches").is_array());
    EXPECT_TRUE(parsed.at("mem_banks").is_array());
    EXPECT_TRUE(parsed.at("io_links").is_array());
}

TEST(agent_info_json_test, to_json_string_with_arrays_is_valid_json)
{
    constexpr std::uint64_t k_l1_cache_size = 32;
    constexpr std::uint64_t k_l2_cache_size = 4096;
    constexpr std::uint64_t k_mem_bank_size = 1ULL << 34;

    std::array<rocprofiler_agent_cache_t, 2>    caches{};
    std::array<rocprofiler_agent_mem_bank_t, 1> mem_banks{};
    std::array<rocprofiler_agent_io_link_t, 1>  io_links{};
    caches[0].size             = k_l1_cache_size;
    caches[0].level            = 1;
    caches[1].size             = k_l2_cache_size;
    caches[1].level            = 2;
    mem_banks[0].size_in_bytes = k_mem_bank_size;
    io_links[0].node_to        = 1;

    auto agent_data            = make_gpu_agent_data();
    agent_data.caches          = caches.data();
    agent_data.caches_count    = static_cast<std::uint32_t>(caches.size());
    agent_data.mem_banks       = mem_banks.data();
    agent_data.mem_banks_count = static_cast<std::uint32_t>(mem_banks.size());
    agent_data.io_links        = io_links.data();
    agent_data.io_links_count  = static_cast<std::uint32_t>(io_links.size());

    auto const json_str = rocprofsys::agent_info::to_json_string(agent_data);

    ASSERT_TRUE(nlohmann::json::accept(json_str))
        << json_str.substr(0, k_max_json_length);

    auto const parsed = nlohmann::json::parse(json_str);
    ASSERT_EQ(parsed.at("caches").size(), 2U);
    EXPECT_EQ(parsed.at("caches")[1].at("size"), k_l2_cache_size);
    EXPECT_EQ(parsed.at("caches")[1].at("level"), 2);
    ASSERT_EQ(parsed.at("mem_banks").size(), 1U);
    EXPECT_EQ(parsed.at("mem_banks")[0].at("size_in_bytes"), k_mem_bank_size);
    ASSERT_EQ(parsed.at("io_links").size(), 1U);
    EXPECT_EQ(parsed.at("io_links")[0].at("node_to"), 1);
}

TEST(agent_info_json_test, to_json_string_escapes_quotes_as_json)
{
    auto agent_data         = make_gpu_agent_data();
    agent_data.product_name = "Instinct \"MI300X\"";

    auto const json_str = rocprofsys::agent_info::to_json_string(agent_data);

    ASSERT_TRUE(nlohmann::json::accept(json_str))
        << json_str.substr(0, k_max_json_length);
    EXPECT_NE(json_str.find(R"("Instinct \"MI300X\"")"), std::string::npos);
    EXPECT_EQ(nlohmann::json::parse(json_str).at("product_name"), "Instinct \"MI300X\"");
}
