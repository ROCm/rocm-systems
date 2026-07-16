// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/rocprofiler-sdk.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string_view>

namespace rocprofsys::rocprofiler_sdk::testing
{

namespace gm = ::testing;

// ─── Mock ─────────────────────────────────────────────────────────────────────
//
// Only get_version() needs to be mocked; get_callback_tracing_names() and
// get_buffer_tracing_names() are inherited from the real backend and return
// compile-time SDK name tables (no ROCm runtime required).

class gmock_sdk_core_backend
{
public:
    MOCK_METHOD(void, get_version,
                (std::uint32_t * major, std::uint32_t* minor, std::uint32_t* patch));
};

inline std::unique_ptr<gmock_sdk_core_backend> g_mock;

// ─── Backend policy ───────────────────────────────────────────────────────────
//
// Inherits from the real backend so all SDK type aliases, enum constants, and
// name-info table accessors come for free.  Only get_version() is overridden
// (hidden) by the mock version.
//
// Each distinct Tag gives sdk_core<tagged_backend<Tag>>::get_version() its own
// static _version cache, so tests that verify first-call behaviour don't
// interfere with one another.

// Each enumerator maps to a distinct Backend type, giving sdk_core<tagged_backend<N>>
// its own static caches (get_version cache, operation option maps, tracing info tables).
enum backend_tag : int
{
    version_fields    = 1,
    version_formatted = 2,
    version_caching   = 3,
    helpers           = 20,
    filter_logic      = 30,
    kind_impl         = 40,
    ops_injected      = 50,
    ops_throw         = 60,
};

template <int Tag>
struct tagged_backend : ::rocprofsys::rocprofiler_sdk::backend
{
    static void get_version(std::uint32_t* major, std::uint32_t* minor,
                            std::uint32_t* patch)
    {
        g_mock->get_version(major, minor, patch);
    }
};

// ─── Fixtures ─────────────────────────────────────────────────────────────────

class sdk_core_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<gmock_sdk_core_backend>(); }
    void TearDown() override { g_mock.reset(); }
};

// ─── get_version ─────────────────────────────────────────────────────────────

TEST_F(sdk_core_test, get_version_populates_major_minor_patch)
{
    using sut = sdk_core<tagged_backend<version_fields>>;

    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 2;
            *pat = 3;
        });

    auto& ver = sut::get_version();
    EXPECT_EQ(ver.major, 1u);
    EXPECT_EQ(ver.minor, 2u);
    EXPECT_EQ(ver.patch, 3u);
}

TEST_F(sdk_core_test, get_version_formatted_equals_major_10000_minor_100_patch)
{
    using sut = sdk_core<tagged_backend<version_formatted>>;

    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 1;
            *pat = 0;
        });

    EXPECT_EQ(sut::get_version().formatted, (1u * 10000u) + (1u * 100u) + 0u);
}

TEST_F(sdk_core_test, get_version_caches_result_calling_backend_exactly_once)
{
    using sut = sdk_core<tagged_backend<version_caching>>;

    EXPECT_CALL(*g_mock, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 2;
            *min = 0;
            *pat = 0;
        });

    auto& ver1 = sut::get_version();
    auto& ver2 = sut::get_version();
    auto& ver3 = sut::get_version();

    EXPECT_EQ(&ver1, &ver2);  // Same cached object
    EXPECT_EQ(&ver2, &ver3);
    EXPECT_EQ(ver1.formatted, 20000u);
}

// ─── to_lower ────────────────────────────────────────────────────────────────

using helper_sut = sdk_core<tagged_backend<helpers>>;

TEST(sdk_core_helpers, to_lower_converts_uppercase_string_view)
{
    EXPECT_EQ(helper_sut::to_lower(std::string_view{ "HIP_RUNTIME_API" }),
              "hip_runtime_api");
}

TEST(sdk_core_helpers, to_lower_leaves_lowercase_unchanged)
{
    EXPECT_EQ(helper_sut::to_lower(std::string_view{ "memory_copy" }), "memory_copy");
}

TEST(sdk_core_helpers, to_lower_handles_mixed_case)
{
    EXPECT_EQ(helper_sut::to_lower(std::string_view{ "HsA_CoRe_Api" }), "hsa_core_api");
}

// ─── get_setting_name ────────────────────────────────────────────────────────

TEST(sdk_core_helpers, get_setting_name_strips_rocprofsys_prefix)
{
    EXPECT_EQ(helper_sut::get_setting_name("rocprofsys_rocm_domains"), "rocm_domains");
}

TEST(sdk_core_helpers, get_setting_name_lowercases_before_stripping)
{
    EXPECT_EQ(helper_sut::get_setting_name("ROCPROFSYS_ROCM_EVENTS"), "rocm_events");
}

TEST(sdk_core_helpers, get_setting_name_returns_unchanged_without_prefix)
{
    EXPECT_EQ(helper_sut::get_setting_name("OTHER_SETTING"), "other_setting");
}

// ─── get_operations_impl (pure filtering) ───────────────────────────────────

using sut_filter = sdk_core<tagged_backend<filter_logic>>;

TEST(sdk_core_filter, empty_include_and_exclude_returns_sorted_complete)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 3, 1, 2 }, {}, {}),
                gm::ElementsAre(1, 2, 3));
}

TEST(sdk_core_filter, non_empty_include_restricts_to_include_set)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 1, 2, 3, 4, 5 }, { 2, 4 }, {}),
                gm::ElementsAre(2, 4));
}

TEST(sdk_core_filter, exclude_removes_from_complete_when_include_empty)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 1, 2, 3, 4, 5 }, {}, { 2, 4 }),
                gm::ElementsAre(1, 3, 5));
}

TEST(sdk_core_filter, exclude_applied_after_include)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 1, 2, 3, 4, 5 }, { 1, 2, 3 }, { 2 }),
                gm::ElementsAre(1, 3));
}

TEST(sdk_core_filter, empty_complete_returns_empty)
{
    EXPECT_THAT(sut_filter::get_operations_impl({}, {}, {}), gm::IsEmpty());
}

TEST(sdk_core_filter, include_not_in_complete_still_appears_in_result)
{
    // include is used as-is when non-empty; complete is not the authority.
    EXPECT_THAT(sut_filter::get_operations_impl({ 1, 2 }, { 5, 6 }, {}),
                gm::ElementsAre(5, 6));
}

TEST(sdk_core_filter, result_is_always_sorted)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 9, 3, 7, 1, 5 }, {}, {}),
                gm::ElementsAre(1, 3, 5, 7, 9));
}

TEST(sdk_core_filter, exclude_all_complete_returns_empty)
{
    EXPECT_THAT(sut_filter::get_operations_impl({ 1, 2, 3 }, {}, { 1, 2, 3 }),
                gm::IsEmpty());
}

// ─── get_operations_impl (kind-based, uses SDK name tables) ──────────────────
//
// These call Backend::get_callback_tracing_names() which returns the real
// SDK compile-time name table.  No ROCm runtime required.

using kind_sut = sdk_core<tagged_backend<kind_impl>>;

TEST(sdk_core_kind_impl, callback_kind_empty_optname_returns_nonempty_set_for_valid_kind)
{
    // HIP_RUNTIME_API has many operations; the SDK table should give us some.
    auto ops =
        kind_sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);
    EXPECT_THAT(ops, gm::Not(gm::IsEmpty()));
}

TEST(sdk_core_kind_impl, buffer_kind_empty_optname_returns_nonempty_set_for_valid_kind)
{
    auto ops = kind_sut::get_operations_impl(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH);
    EXPECT_THAT(ops, gm::Not(gm::IsEmpty()));
}

TEST(sdk_core_kind_impl, callback_kind_unknown_operation_name_none_excluded)
{
    // Operations whose SDK name is "none" must be excluded.
    auto ops = kind_sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_NONE);
    // NONE kind has no real operations — result must be empty.
    EXPECT_THAT(ops, gm::IsEmpty());
}

// ─── get_operations / get_backtrace_operations (injected options) ─────────────
//
// set_operation_options() pre-populates the static map, making get_operations()
// and get_backtrace_operations() testable without the settings infrastructure.

using ops_sut = sdk_core<tagged_backend<ops_injected>>;

TEST(sdk_core_ops, get_operations_returns_sorted_ops_for_injected_kind)
{
    const auto kind = ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;

    // No include/exclude filter — all operations for the kind are returned.
    ops_sut::set_operation_options(kind, "", "", "");

    const auto result = ops_sut::get_operations(kind);

    EXPECT_THAT(result, gm::Not(gm::IsEmpty()));
    EXPECT_TRUE(std::ranges::is_sorted(result));
}

TEST(sdk_core_ops, get_backtrace_operations_returns_empty_when_no_backtrace_setting)
{
    const auto kind = ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;

    ops_sut::set_operation_options(kind, "", "", "");

    // Empty backtrace string → no operations annotated for backtrace.
    EXPECT_THAT(ops_sut::get_backtrace_operations(kind), gm::IsEmpty());
}

TEST(sdk_core_ops, get_operations_buffer_kind_returns_sorted_ops)
{
    const auto kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

    ops_sut::set_operation_options(kind, "", "", "");

    const auto result = ops_sut::get_operations(kind);

    EXPECT_THAT(result, gm::Not(gm::IsEmpty()));
    EXPECT_TRUE(std::ranges::is_sorted(result));
}

TEST(sdk_core_ops, get_backtrace_operations_buffer_kind_returns_empty_for_no_filter)
{
    const auto kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

    ops_sut::set_operation_options(kind, "", "", "");

    EXPECT_THAT(ops_sut::get_backtrace_operations(kind), gm::IsEmpty());
}

// ─── throw on unregistered kind ───────────────────────────────────────────────
//
// tagged_backend<60> has an empty map — no kind was ever injected — so all
// get_operations / get_backtrace_operations calls must throw std::runtime_error.

using throw_sut = sdk_core<tagged_backend<ops_throw>>;

TEST(sdk_core_throw, get_operations_callback_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_operations(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_operations_buffer_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_operations(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_backtrace_callback_throws_when_kind_not_registered)
{
    EXPECT_THROW(
        throw_sut::get_backtrace_operations(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API),
        std::runtime_error);
}

TEST(sdk_core_throw, get_backtrace_buffer_throws_when_kind_not_registered)
{
    EXPECT_THROW(
        throw_sut::get_backtrace_operations(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH),
        std::runtime_error);
}

TEST(sdk_core_throw, get_operations_error_message_contains_kind_value)
{
    try
    {
        throw_sut::get_operations(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gm::HasSubstr("callback tracing kind"));
    }
}

}  // namespace rocprofsys::rocprofiler_sdk::testing
