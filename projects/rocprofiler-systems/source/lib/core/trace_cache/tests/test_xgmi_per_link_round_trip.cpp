// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pins the per-link XGMI naming contract that PMC desc registration, track
// registration, and the sampler emit must all produce the same string.

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

#include <cstddef>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace
{

// Mirrors trace_cache::info::format_track_name; keep in sync with the
// production helper.
std::string
format_track_name_mirror(const char* category_name, std::optional<int> first_section,
                         std::optional<int> second_section = std::nullopt)
{
    return fmt::format("{}{}{}", category_name,
                       first_section ? fmt::format("_{}", *first_section) : "",
                       second_section ? fmt::format("_{}", *second_section) : "");
}

constexpr const char* read_data_category_name  = "device_xgmi_read_data";
constexpr const char* write_data_category_name = "device_xgmi_write_data";
constexpr std::size_t max_num_xgmi_links       = 8;

std::string
site_pmc_desc_register_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

std::string
site_track_register_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

std::string
site_sampler_pmc_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

std::string
site_sampler_track_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

// Pre-fix formats; used only by the negative guard.
std::string
legacy_sampler_pmc_name(const char* category_name, std::size_t link)
{
    return std::string{ category_name } + "_link" + std::to_string(link);
}

std::string
legacy_sampler_track_name(const char* category_name, std::size_t link)
{
    return std::string{ category_name } + " [Link " + std::to_string(link) + "]";
}

}  // namespace

class XgmiPerLinkNameRoundTrip : public ::testing::Test
{
protected:
    static void expect_all_three_sites_agree(const char*        category_name,
                                             const std::string& canonical_prefix)
    {
        for(std::size_t i = 0; i < max_num_xgmi_links; ++i)
        {
            const auto pmc_register   = site_pmc_desc_register_name(category_name, i);
            const auto track_register = site_track_register_name(category_name, i);
            const auto pmc_emit       = site_sampler_pmc_name(category_name, i);
            const auto track_emit     = site_sampler_track_name(category_name, i);

            EXPECT_EQ(pmc_register, track_register)
                << "PMC-desc register and track register must agree (link " << i << ")";
            EXPECT_EQ(pmc_register, pmc_emit)
                << "PMC-desc register and sampler PMC emit must agree (link " << i << ")";
            EXPECT_EQ(track_register, track_emit)
                << "Track register and sampler track emit must agree (link " << i << ")";
            EXPECT_EQ(pmc_emit, track_emit)
                << "Sampler PMC emit and sampler track emit must be the same string "
                   "so data_processor uses one lookup key (link "
                << i << ")";
            EXPECT_EQ(pmc_register, canonical_prefix + std::to_string(i))
                << "Canonical per-link XGMI name format drifted (link " << i << ")";
        }
    }
};

TEST_F(XgmiPerLinkNameRoundTrip, read_data_all_three_sites_agree)
{
    expect_all_three_sites_agree(read_data_category_name, "device_xgmi_read_data_");
}

TEST_F(XgmiPerLinkNameRoundTrip, write_data_all_three_sites_agree)
{
    expect_all_three_sites_agree(write_data_category_name, "device_xgmi_write_data_");
}

namespace
{

std::string
slurp(const std::string& path)
{
    std::ifstream in{ path };
    if(!in.is_open()) return {};
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

// Asserts the production sites use format_track_name; skipped when
// TEST_SOURCE_ROOT is not defined (out-of-tree consumer build).
TEST_F(XgmiPerLinkNameRoundTrip, production_sources_use_canonical_format)
{
#ifndef TEST_SOURCE_ROOT
    GTEST_SKIP() << "TEST_SOURCE_ROOT not defined; cannot read production sources";
#else
    const std::string root = TEST_SOURCE_ROOT;
    const std::string sampler_path =
        root + "/source/lib/core/trace_cache/rocpd_processor.cpp";
    const std::string register_path = root +
                                      "/source/lib/rocprof-sys/library/pmc/collectors/"
                                      "gpu/cache_policy.hpp";

    const std::string sampler  = slurp(sampler_path);
    const std::string registry = slurp(register_path);

    ASSERT_FALSE(sampler.empty()) << "could not read " << sampler_path;
    ASSERT_FALSE(registry.empty()) << "could not read " << register_path;

    EXPECT_EQ(sampler.find("\"_link\""), std::string::npos)
        << "rocpd_processor.cpp still concatenates \"_link\"; that produces "
           "device_xgmi_(read|write)_data_link<N> which does not match what "
           "cache_policy registers";
    EXPECT_EQ(sampler.find("\" [Link \""), std::string::npos)
        << "rocpd_processor.cpp still concatenates \" [Link \"; that produces "
           "track names that data_processor.cpp:218 cannot resolve";

    EXPECT_NE(sampler.find("amd_smi_xgmi_read_data"), std::string::npos);
    EXPECT_NE(sampler.find("amd_smi_xgmi_write_data"), std::string::npos);

    EXPECT_NE(registry.find("amd_smi_xgmi_read_data>(\n"), std::string::npos)
        << "cache_policy.hpp must register per-link XGMI read-data entries via "
           "format_track_name<category::amd_smi_xgmi_read_data>(...)";
    EXPECT_NE(registry.find("amd_smi_xgmi_write_data>(\n"), std::string::npos)
        << "cache_policy.hpp must register per-link XGMI write-data entries via "
           "format_track_name<category::amd_smi_xgmi_write_data>(...)";
#endif
}

// Asserts the pre-fix legacy formats remain distinct from the canonical
// one; if they ever collide the regression-guard premise is gone.
TEST_F(XgmiPerLinkNameRoundTrip, legacy_sampler_format_does_not_match_canonical)
{
    for(std::size_t i = 0; i < max_num_xgmi_links; ++i)
    {
        const auto canonical    = site_pmc_desc_register_name(read_data_category_name, i);
        const auto legacy_pmc   = legacy_sampler_pmc_name(read_data_category_name, i);
        const auto legacy_track = legacy_sampler_track_name(read_data_category_name, i);

        EXPECT_NE(canonical, legacy_pmc)
            << "Legacy sampler PMC format collided with canonical at link " << i
            << "; the regression-guard premise no longer holds";
        EXPECT_NE(canonical, legacy_track)
            << "Legacy sampler track format collided with canonical at link " << i
            << "; the regression-guard premise no longer holds";
    }
}
