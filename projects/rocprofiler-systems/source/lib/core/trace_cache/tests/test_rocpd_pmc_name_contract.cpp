// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//
// Contract tests for the PMC names used by array-valued GPU metrics.
//
// The rocpd writer rejects a sample whose PMC name was never registered, so the
// registration loops in library/pmc/collectors/gpu/cache_policy.hpp and the
// insertion loops in core/trace_cache/rocpd_processor.cpp must agree character
// for character. These tests pin the shared formatters and mirror both loops, so
// a change applied to one side and not the other fails here instead of aborting
// in the middle of a profiling run.
//

#include "backends/amd_smi/gpu_types.hpp"
#include "core/categories.hpp"
#include "core/trace_cache/metadata_registry.hpp"

#include <gtest/gtest.h>
#include <spdlog/fmt/fmt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace amd_smi  = ::rocprofsys::backends::amd_smi::gpu;
namespace category = ::tim::category;
namespace info     = ::rocprofsys::trace_cache::info;

using xgmi_link_array = std::array<std::uint64_t, amd_smi::MAX_NUM_XGMI_LINKS>;

struct emitted_sample
{
    std::string pmc_name;
    std::string track_name;
};

template <typename Category>
std::string_view
base_name()
{
    return tim::trait::name<Category>::value;
}

// Mirrors the per-link PMC registration loop in
// cache_policy::initialize_pmc_metadata.
std::vector<std::string>
registered_xgmi_pmc_names()
{
    auto names = std::vector<std::string>{};
    for(size_t link = 0; link < amd_smi::MAX_NUM_XGMI_LINKS; ++link)
    {
        names.emplace_back(info::format_link_pmc_name(
            base_name<category::amd_smi_xgmi_read_data>(), link));
        names.emplace_back(info::format_link_pmc_name(
            base_name<category::amd_smi_xgmi_write_data>(), link));
    }
    return names;
}

// Mirrors insert_xgmi_link_metrics in rocpd_processor.cpp.
std::vector<emitted_sample>
inserted_xgmi_samples(std::string_view base, bool is_enabled, const xgmi_link_array& arr)
{
    auto samples = std::vector<emitted_sample>{};
    if(!is_enabled) return samples;
    for(size_t i = 0; i < arr.size(); ++i)
    {
        if(arr[i] == amd_smi::METRIC_VALUE_NOT_SUPPORTED_64) continue;
        samples.push_back({ info::format_link_pmc_name(base, i),
                            info::format_link_track_name(base, i) });
    }
    return samples;
}

xgmi_link_array
all_links_reporting()
{
    auto arr = xgmi_link_array{};
    for(size_t i = 0; i < arr.size(); ++i)
        arr[i] = static_cast<std::uint64_t>(i + 1) * 1024;
    return arr;
}

xgmi_link_array
no_links_reporting()
{
    auto arr = xgmi_link_array{};
    arr.fill(amd_smi::METRIC_VALUE_NOT_SUPPORTED_64);
    return arr;
}

std::set<std::string>
pmc_names_of(const std::vector<emitted_sample>& samples)
{
    auto names = std::set<std::string>{};
    for(const auto& itr : samples)
        names.insert(itr.pmc_name);
    return names;
}

// Mirrors the device-level registration loops in
// cache_policy::initialize_pmc_metadata.
template <typename Category>
std::set<std::string>
registered_device_level_names(size_t count)
{
    auto names = std::set<std::string>{};
    for(size_t i = 0; i < count; ++i)
        names.insert(info::format_track_name<Category>(static_cast<int>(i)));
    return names;
}

// Mirrors insert_device_level_metrics in rocpd_processor.cpp.
template <typename Category>
std::vector<std::string>
inserted_device_level_names(size_t count)
{
    const auto base  = info::format_track_name<Category>();
    auto       names = std::vector<std::string>{};
    for(size_t i = 0; i < count; ++i)
        names.push_back(fmt::format("{}_{}", base, i));
    return names;
}

// Mirrors the per-XCP registration and insertion loops, which both format the
// XCP index and the engine index into the same name.
template <typename Category>
std::set<std::string>
per_xcp_names(size_t engines)
{
    auto names = std::set<std::string>{};
    for(size_t xcp = 0; xcp < amd_smi::MAX_NUM_XCP; ++xcp)
        for(size_t engine = 0; engine < engines; ++engine)
            names.insert(info::format_track_name<Category>(static_cast<int>(xcp),
                                                           static_cast<int>(engine)));
    return names;
}
}  // namespace

TEST(rocpd_pmc_name_contract, xgmi_read_data_link_names)
{
    const auto base = base_name<category::amd_smi_xgmi_read_data>();

    EXPECT_EQ(info::format_link_pmc_name(base, 0), "device_xgmi_read_data_link0");
    EXPECT_EQ(info::format_link_pmc_name(base, 7), "device_xgmi_read_data_link7");
    EXPECT_EQ(info::format_link_track_name(base, 0), "device_xgmi_read_data [Link 0]");
    EXPECT_EQ(info::format_link_track_name(base, 7), "device_xgmi_read_data [Link 7]");
}

TEST(rocpd_pmc_name_contract, xgmi_write_data_link_names)
{
    const auto base = base_name<category::amd_smi_xgmi_write_data>();

    EXPECT_EQ(info::format_link_pmc_name(base, 0), "device_xgmi_write_data_link0");
    EXPECT_EQ(info::format_link_pmc_name(base, 7), "device_xgmi_write_data_link7");
    EXPECT_EQ(info::format_link_track_name(base, 0), "device_xgmi_write_data [Link 0]");
    EXPECT_EQ(info::format_link_track_name(base, 7), "device_xgmi_write_data [Link 7]");
}

// The PMC name is a database identifier and the track name is a display label, so
// they are intentionally spelled differently and must not be swapped.
TEST(rocpd_pmc_name_contract, link_pmc_and_track_names_are_distinct)
{
    const auto base = base_name<category::amd_smi_xgmi_read_data>();

    for(size_t link = 0; link < amd_smi::MAX_NUM_XGMI_LINKS; ++link)
    {
        EXPECT_NE(info::format_link_pmc_name(base, link),
                  info::format_link_track_name(base, link));
    }
}

TEST(rocpd_pmc_name_contract, every_link_gets_a_distinct_pmc_name)
{
    const auto names = registered_xgmi_pmc_names();
    const auto uniq  = std::set<std::string>{ names.begin(), names.end() };

    EXPECT_EQ(names.size(), 2 * amd_smi::MAX_NUM_XGMI_LINKS);
    EXPECT_EQ(uniq.size(), names.size());
}

// Regression test for the abort reported as
//   "PMC Info not registered: pmc_id: [name=device_xgmi_read_data_link0, ...]"
TEST(rocpd_pmc_name_contract, every_inserted_link_name_is_registered)
{
    const auto names      = registered_xgmi_pmc_names();
    const auto registered = std::set<std::string>{ names.begin(), names.end() };

    for(const auto base : { base_name<category::amd_smi_xgmi_read_data>(),
                            base_name<category::amd_smi_xgmi_write_data>() })
    {
        const auto samples = inserted_xgmi_samples(base, true, all_links_reporting());

        EXPECT_EQ(samples.size(), amd_smi::MAX_NUM_XGMI_LINKS);
        for(const auto& itr : samples)
        {
            EXPECT_EQ(registered.count(itr.pmc_name), 1U)
                << "unregistered PMC name: " << itr.pmc_name;
        }
    }
}

// Registering only the base name, which is what the code did before per-link
// registration was added, covers none of the names the insertion path emits.
TEST(rocpd_pmc_name_contract, base_name_alone_covers_no_inserted_link_name)
{
    const auto base    = base_name<category::amd_smi_xgmi_read_data>();
    const auto samples = inserted_xgmi_samples(base, true, all_links_reporting());

    ASSERT_FALSE(samples.empty());
    for(const auto& itr : samples)
        EXPECT_NE(itr.pmc_name, base);
}

TEST(rocpd_pmc_name_contract, unsupported_links_emit_no_sample)
{
    const auto base = base_name<category::amd_smi_xgmi_read_data>();

    auto arr = all_links_reporting();
    arr[2]   = amd_smi::METRIC_VALUE_NOT_SUPPORTED_64;
    arr[5]   = amd_smi::METRIC_VALUE_NOT_SUPPORTED_64;

    const auto names = pmc_names_of(inserted_xgmi_samples(base, true, arr));

    EXPECT_EQ(names.size(), amd_smi::MAX_NUM_XGMI_LINKS - 2);
    EXPECT_EQ(names.count(info::format_link_pmc_name(base, 2)), 0U);
    EXPECT_EQ(names.count(info::format_link_pmc_name(base, 5)), 0U);
    EXPECT_EQ(names.count(info::format_link_pmc_name(base, 3)), 1U);
}

TEST(rocpd_pmc_name_contract, all_links_unsupported_emits_nothing)
{
    const auto base = base_name<category::amd_smi_xgmi_read_data>();

    EXPECT_TRUE(inserted_xgmi_samples(base, true, no_links_reporting()).empty());
}

TEST(rocpd_pmc_name_contract, disabled_xgmi_emits_no_link_samples)
{
    const auto base = base_name<category::amd_smi_xgmi_read_data>();

    EXPECT_TRUE(inserted_xgmi_samples(base, false, all_links_reporting()).empty());
}

TEST(rocpd_pmc_name_contract, device_level_jpeg_registration_matches_insertion)
{
    const auto registered =
        registered_device_level_names<category::amd_smi_jpeg_activity>(
            amd_smi::MAX_NUM_JPEG);
    const auto inserted = inserted_device_level_names<category::amd_smi_jpeg_activity>(
        amd_smi::MAX_NUM_JPEG);

    EXPECT_EQ(inserted.front(), "device_jpeg_activity_0");
    EXPECT_EQ(inserted.size(), amd_smi::MAX_NUM_JPEG);
    for(const auto& itr : inserted)
    {
        EXPECT_EQ(registered.count(itr), 1U) << "unregistered PMC name: " << itr;
    }
}

TEST(rocpd_pmc_name_contract, device_level_vcn_registration_matches_insertion)
{
    const auto registered = registered_device_level_names<category::amd_smi_vcn_activity>(
        amd_smi::MAX_NUM_VCN);
    const auto inserted =
        inserted_device_level_names<category::amd_smi_vcn_activity>(amd_smi::MAX_NUM_VCN);

    EXPECT_EQ(inserted.front(), "device_vcn_activity_0");
    EXPECT_EQ(inserted.size(), amd_smi::MAX_NUM_VCN);
    for(const auto& itr : inserted)
    {
        EXPECT_EQ(registered.count(itr), 1U) << "unregistered PMC name: " << itr;
    }
}

// Device-level and per-XCP JPEG metrics are reported by different GPU families
// and land in the same database, so their names must stay disjoint.
TEST(rocpd_pmc_name_contract, device_level_and_per_xcp_names_do_not_collide)
{
    const auto device_level =
        registered_device_level_names<category::amd_smi_jpeg_activity>(
            amd_smi::MAX_NUM_JPEG);
    const auto per_xcp =
        per_xcp_names<category::amd_smi_jpeg_activity>(amd_smi::MAX_NUM_JPEG_V1);

    for(const auto& itr : device_level)
        EXPECT_EQ(per_xcp.count(itr), 0U) << "colliding PMC name: " << itr;
}
