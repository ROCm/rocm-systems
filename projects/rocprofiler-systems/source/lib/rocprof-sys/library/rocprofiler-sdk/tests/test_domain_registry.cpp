// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/string_utility.hpp"
#include "core/sdk/tracing-config-deps.hpp"
#include "core/sdk/tracing-config.hpp"
#include "library/rocprofiler-sdk/domain_registry.hpp"
#include "library/rocprofiler-sdk/tests/mock_domain_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace rocprofsys::domains
{
namespace
{

// registry<SdkBackend, Externals> instantiates every buffered/callback domain
// definition in library/rocprofiler-sdk/{buffered,callback}/*.hpp, so it needs a fake
// that satisfies the union of everything those headers touch on SdkBackend and
// Externals -- not just what a single domain needs. test_support::mock_sdk/externals
// already carry every kfd_* domain's record type, id, and category name, so registry<>
// needs no domain-specific additions of its own.
using test_support::externals;
using test_support::mock_sdk;

using sut_t = registry<mock_sdk, externals>;

TEST(domain_registry_test, find_descriptor_finds_buffered_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("KfD_QuEuE");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "kfd_queue");
    EXPECT_EQ(descriptor->id, mock_sdk::BUFFER_TRACING_KFD_QUEUE);
    EXPECT_EQ(descriptor->mode, collection_mode::buffered);
}

TEST(domain_registry_test, find_descriptor_finds_callback_domain_case_insensitively)
{
    const domain_descriptor* descriptor = sut_t::find_descriptor("HIP_RUNTIME_API");

    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->name, "hip_runtime_api");
    EXPECT_EQ(descriptor->id, mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API);
    EXPECT_EQ(descriptor->mode, collection_mode::callback);
}

TEST(domain_registry_test, find_descriptor_returns_nullptr_for_unknown_name)
{
    EXPECT_EQ(sut_t::find_descriptor("not_a_real_domain"), nullptr);
}

TEST(domain_registry_test, get_buffered_returns_definition_matching_domain_id)
{
    const auto& definition = sut_t::get_buffered(mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);

    EXPECT_EQ(definition.meta.name, "kfd_page_fault");
    EXPECT_EQ(definition.meta.id, mock_sdk::BUFFER_TRACING_KFD_PAGE_FAULT);
}

TEST(domain_registry_test, get_buffered_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_buffered(k_unknown_id)); }, std::runtime_error);
}

TEST(domain_registry_test, get_callback_returns_definition_matching_domain_id)
{
    const auto& definition =
        sut_t::get_callback(mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API);

    EXPECT_EQ(definition.meta.name, "hip_runtime_api");
    EXPECT_EQ(definition.meta.id, mock_sdk::CALLBACK_TRACING_HIP_RUNTIME_API);
}

TEST(domain_registry_test, get_callback_throws_runtime_error_for_unknown_domain_id)
{
    constexpr domain_id_t k_unknown_id = 9999;

    EXPECT_THROW(
        { static_cast<void>(sut_t::get_callback(k_unknown_id)); }, std::runtime_error);
}

// compile_time_version uses the formatted scheme: major * 10000 + minor * 100 + patch.
template <std::size_t FormattedVersion>
struct mock_sdk_at_version : mock_sdk
{
    // Mirrors the SDK member name that the registry reads.
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr std::size_t compile_time_version = FormattedVersion;
};

template <std::size_t FormattedVersion>
bool
is_domain_registered_at(std::string_view name)
{
    return registry<mock_sdk_at_version<FormattedVersion>, externals>::find_descriptor(
               name) != nullptr;
}

constexpr std::size_t k_version_0_5_9 = 509;
constexpr std::size_t k_version_0_6_0 = 600;
constexpr std::size_t k_version_0_6_1 = 601;

TEST(domain_registry_test, ompt_and_rocdecode_are_gated_on_sdk_0_6_0)
{
    for(const std::string_view name : { "ompt", "rocdecode_api" })
    {
        EXPECT_FALSE(is_domain_registered_at<k_version_0_5_9>(name)) << name;
        EXPECT_TRUE(is_domain_registered_at<k_version_0_6_0>(name)) << name;
        EXPECT_TRUE(is_domain_registered_at<k_version_0_6_1>(name)) << name;
    }
}

TEST(domain_registry_test, get_callback_throws_for_ompt_below_sdk_0_6_0)
{
    using sut_below_0_6_t = registry<mock_sdk_at_version<k_version_0_5_9>, externals>;

    EXPECT_THROW(
        {
            static_cast<void>(
                sut_below_0_6_t::get_callback(mock_sdk::CALLBACK_TRACING_OMPT));
        },
        std::runtime_error);
}

TEST(domain_registry_test, get_callback_returns_ompt_definition_at_sdk_0_6_0)
{
    using sut_at_0_6_t = registry<mock_sdk_at_version<k_version_0_6_0>, externals>;

    const auto& definition = sut_at_0_6_t::get_callback(mock_sdk::CALLBACK_TRACING_OMPT);

    EXPECT_EQ(definition.meta.name, "ompt");
    EXPECT_EQ(definition.meta.mode, collection_mode::callback);
}

// Every tracing kind name rocprofiler-sdk 1.5 reports (rocprofiler-sdk/fwd.h), except
// the pre-1.0 PAGE_MIGRATION that tracing_config accepts without a registry entry.
constexpr auto k_sdk_tracing_kind_names = std::to_array<std::string_view>({
    "HSA_CORE_API",
    "HSA_AMD_EXT_API",
    "HSA_IMAGE_EXT_API",
    "HSA_FINALIZE_EXT_API",
    "HIP_RUNTIME_API",
    "HIP_COMPILER_API",
    "HIP_RUNTIME_API_EXT",
    "HIP_COMPILER_API_EXT",
    "MARKER_CORE_API",
    "MARKER_CONTROL_API",
    "MARKER_NAME_API",
    "MARKER_CORE_RANGE_API",
    "CODE_OBJECT",
    "SCRATCH_MEMORY",
    "KERNEL_DISPATCH",
    "KERNEL_REPLAY",
    "MEMORY_COPY",
    "MEMORY_ALLOCATION",
    "CORRELATION_ID_RETIREMENT",
    "RCCL_API",
    "OMPT",
    "RUNTIME_INITIALIZATION",
    "ROCDECODE_API",
    "ROCDECODE_API_EXT",
    "ROCJPEG_API",
    "HIP_STREAM",
    "HIP_GRAPH",
    "HIP_EVENT",
    "ROCSHMEM_API",
    "ROCSHMEM_API_EXT",
    "HIPFILE_API",
    "HIPFILE_API_EXT",
    "KFD_EVENT_PAGE_MIGRATE",
    "KFD_EVENT_PAGE_FAULT",
    "KFD_EVENT_QUEUE",
    "KFD_EVENT_UNMAP_FROM_GPU",
    "KFD_EVENT_DROPPED_EVENTS",
    "KFD_PAGE_MIGRATE",
    "KFD_PAGE_FAULT",
    "KFD_QUEUE",
});

// Also satisfies policies::tracing_config::backend, and reports every SDK tracing kind,
// so only the compile-time version gates decide which domains tracing_config accepts.
template <std::size_t FormattedVersion>
struct mock_sdk_with_all_kinds : mock_sdk_at_version<FormattedVersion>
{
    // NOLINTBEGIN(readability-identifier-naming)
    static constexpr std::size_t BUFFER_TRACING_HSA_CORE_API         = 40;
    static constexpr std::size_t BUFFER_TRACING_HSA_AMD_EXT_API      = 41;
    static constexpr std::size_t BUFFER_TRACING_HSA_IMAGE_EXT_API    = 42;
    static constexpr std::size_t BUFFER_TRACING_HSA_FINALIZE_EXT_API = 43;
    static constexpr std::size_t BUFFER_TRACING_HIP_RUNTIME_API      = 44;
    static constexpr std::size_t BUFFER_TRACING_HIP_COMPILER_API     = 45;
    static constexpr std::size_t BUFFER_TRACING_MARKER_CORE_API      = 46;
    // NOLINTEND(readability-identifier-naming)

    // Declared only: get_domain_choices() never calls it.
    static void get_version(std::uint32_t* major, std::uint32_t* minor,
                            std::uint32_t* patch);

    static test_support::tracing_names_t get_buffer_tracing_names() { return {}; }
    static test_support::tracing_names_t get_callback_tracing_names()
    {
        auto table = test_support::tracing_names_t{};
        for(const auto name : k_sdk_tracing_kind_names)
        {
            table.entries.push_back({ .name = name, .operations = {} });
        }
        return table;
    }
};

// Temporary: tracing_config keeps its own list of supported domains because core/
// cannot include the registry; a domain it accepts but the registry lacks aborts
// tool_init. Delete this check (and mock_sdk_with_all_kinds, k_sdk_tracing_kind_names)
// once domain choices are probed from domain_service's available domains instead of
// tracing_config::k_supported_domains.
template <std::size_t FormattedVersion>
void
expect_choices_match_registry_at()
{
    using sdk_t = mock_sdk_with_all_kinds<FormattedVersion>;
    // get_domain_choices() reads no externals, so the production ones are never called.
    using tracing_config_t =
        rocprofiler_sdk::tracing_config<sdk_t, rocprofiler_sdk::default_externals>;

    SCOPED_TRACE(fmt::format("compile_time_version {}", FormattedVersion));

    const auto choices = tracing_config_t::get_domain_choices();
    for(const auto sdk_name : k_sdk_tracing_kind_names)
    {
        const auto name      = utility::string::to_lower(sdk_name);
        const bool is_choice = std::ranges::find(choices, name) != choices.end();
        const bool is_registered =
            registry<sdk_t, externals>::find_descriptor(name) != nullptr;
        EXPECT_EQ(is_choice, is_registered) << name;
    }
}

template <std::size_t... FormattedVersions>
void
expect_choices_match_registry()
{
    (expect_choices_match_registry_at<FormattedVersions>(), ...);
}

constexpr std::size_t k_version_0_6_9 = 609;
constexpr std::size_t k_version_0_7_0 = 700;
constexpr std::size_t k_version_1_2_1 = 10201;
constexpr std::size_t k_version_1_2_2 = 10202;
constexpr std::size_t k_version_1_3_3 = 10303;
constexpr std::size_t k_version_1_3_4 = 10304;
constexpr std::size_t k_version_1_3_5 = 10305;

TEST(domain_registry_test, rocm_domain_choices_match_registry_on_both_sides_of_each_gate)
{
    expect_choices_match_registry<k_version_0_5_9, k_version_0_6_0, k_version_0_6_9,
                                  k_version_0_7_0, k_version_1_2_1, k_version_1_2_2,
                                  k_version_1_3_3, k_version_1_3_4, k_version_1_3_5>();
}

}  // namespace
}  // namespace rocprofsys::domains
