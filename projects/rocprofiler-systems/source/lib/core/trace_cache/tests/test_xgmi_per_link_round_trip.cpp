// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Regression test for XGMI per-link PMC + track name alignment.
//
// Background. Three sites construct per-link XGMI metric names on the rocpd
// path:
//
//   1. PMC desc registration (cache_policy::initialize_pmc_metadata, in
//      library/pmc/collectors/gpu/cache_policy.hpp)
//   2. Track registration   (cache_policy::initialize_tracks_metadata, same file)
//   3. Sampler emit         (rocpd_processor_t::handle for gpu_pmc_sample, in
//      core/trace_cache/rocpd_processor.cpp)
//
// They MUST all produce the same string, otherwise data_processor.cpp:201/221
// emits a "non-existing PMC description" / "Unexisting track" warning on every
// sample on multi-GPU XGMI hardware.
//
// Historically site 2 used info::format_track_name<C>(nullopt, link) (underscore
// suffix), but site 1 registered a single non-per-link entry and site 3 emitted
// "<base>_link<N>" (PMC name) + "<base> [Link <N>]" (track name). All three
// formats disagreed. The fix aligned all three on info::format_track_name.
//
// Test strategy. The production headers chain (cache_policy.hpp -> cache_manager
// -> categories.hpp) pulls timemory + rocprofiler-sdk, so we cannot link
// cache_policy directly into the unit-test object. Instead we replicate the
// format helper and the three sites' name-construction logic side-by-side and
// assert they all agree, AND that they match the form the registry/track
// lookup uses. If any of the three sites drift back to a different format,
// this test fails.
//

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

#include <cstddef>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace
{

// Mirror of trace_cache::info::format_track_name from
// core/trace_cache/metadata_registry.hpp:107-115. Kept tiny on purpose: if the
// production helper changes shape, both this mirror and the production sites
// must be updated together.
std::string
format_track_name_mirror(const char* category_name, std::optional<int> first_section,
                         std::optional<int> second_section = std::nullopt)
{
    return fmt::format("{}{}{}", category_name,
                       first_section ? fmt::format("_{}", *first_section) : "",
                       second_section ? fmt::format("_{}", *second_section) : "");
}

// Trait names matching the ROCPROFSYS_DEFINE_CATEGORY entries in
// core/categories.hpp:113-114.
constexpr const char* read_data_category_name  = "device_xgmi_read_data";
constexpr const char* write_data_category_name = "device_xgmi_write_data";

// MAX_NUM_XGMI_LINKS from library/pmc/collectors/gpu/types.hpp:32.
constexpr std::size_t max_num_xgmi_links = 8;

// ---------------------------------------------------------------------------
// Site 1: PMC desc registration loop (after fix), cache_policy.hpp
// ---------------------------------------------------------------------------
std::string
site_pmc_desc_register_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

// ---------------------------------------------------------------------------
// Site 2: Track registration loop (already correct), cache_policy.hpp
// ---------------------------------------------------------------------------
std::string
site_track_register_name(const char* category_name, std::size_t link)
{
    return format_track_name_mirror(category_name, std::nullopt, static_cast<int>(link));
}

// ---------------------------------------------------------------------------
// Site 3: Sampler emit (after fix), rocpd_processor.cpp:455-475
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Pre-fix sampler emit: kept ONLY for the negative-side guard below. Do not
// use this format in production.
// ---------------------------------------------------------------------------
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

// Source-content guard. The compile-time mirror above only proves the helper
// math is right, not that the production sites use the helper. This test
// reads the two production sources and asserts:
//   - the legacy "_link" / " [Link " literal patterns are gone,
//   - the canonical format_track_name<...>(std::nullopt, ...) call is present
//     for both XGMI read-data and write-data categories.
// If TEST_SOURCE_ROOT is not provided (out-of-tree consumer build), the test
// is skipped rather than failed.
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

// Negative-side guard. The legacy "_link<N>" / " [Link <N>]" formats that
// shipped before the fix must NOT match the canonical format. Documents the
// bug class explicitly so a future refactor cannot silently revert it.
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
