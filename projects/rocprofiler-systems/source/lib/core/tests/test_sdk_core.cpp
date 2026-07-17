// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/rocprofiler-sdk.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

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
    version_fields         = 1,
    version_formatted      = 2,
    version_caching        = 3,
    version_reset          = 4,
    tracing_cache_reset    = 10,
    helpers                = 20,
    filter_logic           = 30,
    kind_impl              = 40,
    kind_impl_mocked       = 41,
    ops_injected_mocked    = 51,
    ops_injected           = 50,
    ops_throw              = 60,
    domains_callback       = 70,
    domains_buffer         = 71,
    domains_buffer_old_kfd = 72,
    config_domains         = 80,
    config_events          = 81,
    config_operations      = 82,
    config_duplicate       = 83,
    rocm_events            = 90,
};

// FakeCompileTimeVersion defaults to the real installed SDK version. Overriding it
// lets tests control get_buffered_domains()'s KFD runtime-support gate, which (per
// its own doc comment) compares Wrapper::compile_time_version rather than the
// dynamically-queried version — deliberately mock-friendly.
template <int Tag, std::uint32_t FakeCompileTimeVersion = ROCPROFILER_VERSION>
struct tagged_backend : ::rocprofsys::rocprofiler_sdk::backend
{
    static constexpr std::uint32_t compile_time_version = FakeCompileTimeVersion;

    static void get_version(std::uint32_t* major, std::uint32_t* minor,
                            std::uint32_t* patch)
    {
        g_mock->get_version(major, minor, patch);
    }
};

// ─── Settings mock (Externals::Settings) ──────────────────────────────────────
//
// tim::settings has no virtual methods and is dominated by member templates
// (insert<Tp,Vp,Sp,Args...>, find<Sp>, at<Sp>) — it cannot be expressed as a
// GMock interface (GMock requires concrete, non-template signatures). fake_settings
// is a minimal in-memory stand-in implementing only what sdk_core actually calls
// through Externals::Settings: insert<Tp,Tp>(...), find(key), at(key), and the
// returned handle's get_choices()/set_choices()/get<Tp>(). Only std::string and
// bool are ever used as Tp in production, so get<Tp>() only supports those two.

class fake_setting_handle
{
public:
    void set_choices(std::vector<std::string> choices) { m_choices = std::move(choices); }
    const std::vector<std::string>& get_choices() const { return m_choices; }

    template <typename Tp>
    std::pair<bool, Tp> get() const
    {
        if constexpr(std::is_same_v<Tp, std::string>)
            return { true, m_string_value };
        else if constexpr(std::is_same_v<Tp, bool>)
            return { true, m_bool_value };
        else
            static_assert(!sizeof(Tp), "fake_setting_handle::get<Tp>: unsupported Tp");
    }

    std::string              m_string_value = {};
    bool                     m_bool_value   = false;
    std::vector<std::string> m_choices      = {};
};

class fake_settings
{
public:
    using handle_t = std::shared_ptr<fake_setting_handle>;
    using map_t    = std::unordered_map<std::string, handle_t>;
    using iterator = map_t::iterator;

    iterator end() { return m_data.end(); }
    iterator find(const std::string& key) { return m_data.find(key); }
    handle_t at(const std::string& key) { return m_data.at(key); }

    template <typename Tp, typename Vp, typename Sp, typename... Args>
    std::pair<iterator, bool> insert(Sp&& env, const std::string& /*name*/,
                                     const std::string& /*desc*/, Vp init, Args&&...)
    {
        auto handle = std::make_shared<fake_setting_handle>();
        if constexpr(std::is_same_v<Tp, std::string>)
            handle->m_string_value = init;
        else if constexpr(std::is_same_v<Tp, bool>)
            handle->m_bool_value = init;
        return m_data.emplace(std::string{ std::forward<Sp>(env) }, std::move(handle));
    }

private:
    map_t m_data{};
};

// Builds a standalone settings registry (never the process-wide singleton) with
// ROCPROFSYS_ROCM_DOMAINS registered and its choices set to exactly `valid_choices`.
inline std::shared_ptr<fake_settings>
make_domains_settings(std::vector<std::string> valid_choices)
{
    auto config = std::make_shared<fake_settings>();
    config->insert<std::string, std::string>(
        std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS }, "rocm_domains",
        "test-only setting", std::string{}, std::set<std::string>{ "rocm" });
    config->find(std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS })
        ->second->set_choices(std::move(valid_choices));
    return config;
}

// ─── Externals mock ───────────────────────────────────────────────────────────
//
// get_callback_domains() / get_buffered_domains() / get_rocm_events() read through
// the Externals policy. Mocking it lets tests supply a private fake_settings
// registry and control the RCCLP/OMPT/unified-memory-profiling flags without ever
// touching rocprofsys::settings::instance().

class gmock_sdk_externals
{
public:
    MOCK_METHOD(fake_settings*, get_settings, ());
    MOCK_METHOD(bool, get_use_rcclp, ());
    MOCK_METHOD(bool, get_use_ompt, ());
    MOCK_METHOD(bool, get_use_unified_memory_profiling, ());
    MOCK_METHOD(std::string, get_rocm_domains, ());
    MOCK_METHOD(std::string, get_rocm_events_setting, ());
    MOCK_METHOD(std::optional<std::string>, get_setting_value, (std::string_view));
};

inline std::unique_ptr<gm::StrictMock<gmock_sdk_externals>> g_mock_externals;

struct mock_sdk_externals
{
    using Settings = fake_settings;

    static auto* get_settings() { return g_mock_externals->get_settings(); }
    static bool  get_use_rcclp() { return g_mock_externals->get_use_rcclp(); }
    static bool  get_use_ompt() { return g_mock_externals->get_use_ompt(); }
    static bool  get_use_unified_memory_profiling()
    {
        return g_mock_externals->get_use_unified_memory_profiling();
    }
    static std::string get_rocm_domains() { return g_mock_externals->get_rocm_domains(); }
    static std::string get_rocm_events_setting()
    {
        return g_mock_externals->get_rocm_events_setting();
    }
    static std::optional<std::string> get_setting_value(std::string_view s)
    {
        return g_mock_externals->get_setting_value(s);
    }
};

// ─── Fixtures ─────────────────────────────────────────────────────────────────

class sdk_core_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock = std::make_unique<gmock_sdk_core_backend>(); }
    void TearDown() override { g_mock.reset(); }
};

// Fixture for functions that read through the Externals policy
// (get_callback_domains, get_buffered_domains, get_rocm_events).
class sdk_core_domains_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock           = std::make_unique<gmock_sdk_core_backend>();
        g_mock_externals = std::make_unique<gm::StrictMock<gmock_sdk_externals>>();
    }
    void TearDown() override
    {
        g_mock.reset();
        g_mock_externals.reset();
    }
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

TEST_F(sdk_core_test, reset_version_cache_forces_next_call_to_requery_backend)
{
    using sut = sdk_core<tagged_backend<version_reset>>;

    EXPECT_CALL(*g_mock, get_version)
        .Times(2)
        .WillRepeatedly([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 3;
            *min = 0;
            *pat = 0;
        });

    sut::get_version();
    sut::reset_version_cache();
    sut::get_version();
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

TEST(sdk_core_filter, production_alias_get_operations_impl_returns_sorted_complete)
{
    // get_operations_impl(complete, include, exclude) never touches Externals, so
    // core_sdk = sdk_core<backend, default_sdk_externals> — the exact instantiation
    // production code links against — can be exercised directly and safely here,
    // without triggering default_sdk_externals's process-init requirement.
    EXPECT_THAT(core_sdk::get_operations_impl({ 3, 1, 2 }, {}, {}),
                gm::ElementsAre(1, 2, 3));
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

TEST(sdk_core_kind_impl, callback_kind_with_unregistered_setting_name_throws)
{
    // operations_setting names a rocprofsys setting looked up via
    // Externals::get_setting_value() — a name that was never registered must
    // throw, independent of the tracing-name cache. kind_sut uses the default
    // (real) Externals, whose get_setting_value() reads the process-wide
    // tim::settings singleton; this name is guaranteed absent there.
    EXPECT_THROW(
        kind_sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                      "ROCPROFSYS_TEST_SDK_CORE_NONEXISTENT_SETTING_XYZ"),
        std::runtime_error);
}

TEST_F(sdk_core_domains_test, callback_kind_with_registered_regex_filter_matches_by_name)
{
    // Externals::get_setting_value() is mockable, so the regex-match branch of
    // operation_ids_for_tracing_kind() can be exercised without touching the
    // real tim::settings singleton.
    using sut = sdk_core<tagged_backend<kind_impl_mocked>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_HIP_OPERATIONS_FILTER" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "malloc" }));

    // HIP_RUNTIME_API has hipMalloc/hipMallocManaged/etc. — "malloc" must match some.
    auto ops = sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                        "SOME_HIP_OPERATIONS_FILTER");

    EXPECT_THAT(ops, gm::Not(gm::IsEmpty()));
}

TEST_F(sdk_core_domains_test,
       callback_kind_with_registered_empty_filter_value_returns_empty)
{
    // The setting IS registered (get_setting_value succeeds) but its value is the
    // empty string — a distinct branch from "setting not registered at all".
    using sut = sdk_core<tagged_backend<kind_impl_mocked>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_EMPTY_OPERATIONS_FILTER" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "" }));

    auto ops = sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                        "SOME_EMPTY_OPERATIONS_FILTER");

    EXPECT_THAT(ops, gm::IsEmpty());
}

TEST(sdk_core_kind_impl, reset_tracing_names_cache_allows_subsequent_lookup_to_succeed)
{
    using cache_sut = sdk_core<tagged_backend<tracing_cache_reset>>;

    // Populate then clear the cached tracing-name tables; a subsequent lookup must
    // still succeed and return the same result (no stale/dangling optional access).
    auto first =
        cache_sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);
    cache_sut::reset_tracing_names_cache();
    auto second =
        cache_sut::get_operations_impl(ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API);

    EXPECT_EQ(first, second);
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

TEST_F(sdk_core_domains_test,
       get_backtrace_operations_returns_matching_ops_when_backtrace_setting_set)
{
    // operations_annotate_backtrace holds a *setting name*, not a regex pattern —
    // get_operations_impl(kind, bt) resolves it via Externals::get_setting_value().
    // Mocking that lets us return an actual pattern without touching the real
    // tim::settings singleton.
    using sut       = sdk_core<tagged_backend<ops_injected_mocked>, mock_sdk_externals>;
    const auto kind = ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;

    sut::set_operation_options(kind, "", "", "SOME_BACKTRACE_SETTING");

    // "." matches any single character — any kind with operations matches at
    // least one, exercising get_operations_impl(kind, bt) and the vector->set
    // conversion on the way out.
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_BACKTRACE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "." }));

    EXPECT_THAT(sut::get_backtrace_operations(kind), gm::Not(gm::IsEmpty()));
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

TEST_F(
    sdk_core_domains_test,
    get_backtrace_operations_buffer_kind_returns_matching_ops_when_backtrace_setting_set)
{
    using sut       = sdk_core<tagged_backend<ops_injected_mocked>, mock_sdk_externals>;
    const auto kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

    sut::set_operation_options(kind, "", "", "SOME_BUFFER_BACKTRACE_SETTING");

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_BUFFER_BACKTRACE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "." }));

    EXPECT_THAT(sut::get_backtrace_operations(kind), gm::Not(gm::IsEmpty()));
}

TEST_F(sdk_core_domains_test,
       get_operations_applies_registered_include_and_exclude_filters)
{
    // operations_include/operations_exclude are also setting names — this exercises
    // the non-empty branches of get_operations()'s include/exclude ternaries, which
    // every other test in this file leaves empty.
    using sut       = sdk_core<tagged_backend<ops_injected_mocked>, mock_sdk_externals>;
    const auto kind = ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;

    sut::set_operation_options(kind, "SOME_INCLUDE_SETTING", "SOME_EXCLUDE_SETTING", "");

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_INCLUDE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "malloc" }));
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_EXCLUDE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "managed" }));

    // hipMalloc/hipMalloc3D/etc. match "malloc"; hipMallocManaged is removed by the
    // "managed" exclude filter but other "malloc" matches must remain.
    EXPECT_THAT(sut::get_operations(kind), gm::Not(gm::IsEmpty()));
}

TEST_F(sdk_core_domains_test,
       get_operations_buffer_kind_applies_registered_include_and_exclude_filters)
{
    using sut       = sdk_core<tagged_backend<ops_injected_mocked>, mock_sdk_externals>;
    const auto kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;

    sut::set_operation_options(kind, "SOME_BUFFER_INCLUDE_SETTING",
                               "SOME_BUFFER_EXCLUDE_SETTING", "");

    // "." (include) matches every operation; a pattern with no real match (exclude)
    // removes nothing — guarantees a non-empty result without depending on exact
    // KERNEL_DISPATCH operation names.
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_BUFFER_INCLUDE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "." }));
    EXPECT_CALL(*g_mock_externals,
                get_setting_value(std::string_view{ "SOME_BUFFER_EXCLUDE_SETTING" }))
        .WillOnce(gm::Return(std::optional<std::string>{ "nonexistent_xyz_pattern" }));

    EXPECT_THAT(sut::get_operations(kind), gm::Not(gm::IsEmpty()));
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

// ─── get_callback_domains ─────────────────────────────────────────────────────
//
// All tests share one Wrapper tag (fresh version cache reset per test via
// reset_version_cache()) — get_callback_domains() never touches the tracing-name
// cache or the operation-option maps, so no other static state can leak between
// these tests.

using callback_domains_sut =
    sdk_core<tagged_backend<domains_callback>, mock_sdk_externals>;

TEST_F(sdk_core_domains_test,
       returns_empty_set_when_rocm_domains_setting_is_empty_and_flags_false)
{
    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "hip_api", "hsa_api", "marker_api", "roctx" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(), gm::IsEmpty());
}

TEST_F(sdk_core_domains_test, selects_kinds_bound_to_hip_api_token)
{
    using wrapper = tagged_backend<domains_callback>;

    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "hip_api" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "hip_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(),
                gm::UnorderedElementsAre(wrapper::CALLBACK_TRACING_HIP_RUNTIME_API,
                                         wrapper::CALLBACK_TRACING_HIP_COMPILER_API));
}

TEST_F(sdk_core_domains_test, selects_kinds_bound_to_hsa_api_token)
{
    using wrapper = tagged_backend<domains_callback>;

    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "hsa_api" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "hsa_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(),
                gm::UnorderedElementsAre(wrapper::CALLBACK_TRACING_HSA_CORE_API,
                                         wrapper::CALLBACK_TRACING_HSA_AMD_EXT_API,
                                         wrapper::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                                         wrapper::CALLBACK_TRACING_HSA_FINALIZE_EXT_API));
}

TEST_F(sdk_core_domains_test, selects_kind_bound_to_marker_api_or_roctx_token)
{
    using wrapper = tagged_backend<domains_callback>;

    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "roctx" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "roctx" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(),
                gm::UnorderedElementsAre(wrapper::CALLBACK_TRACING_MARKER_CORE_API));
}

TEST_F(sdk_core_domains_test, throws_runtime_error_on_domain_token_not_in_valid_choices)
{
    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "hip_api" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "not_a_real_domain" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    try
    {
        callback_domains_sut::get_callback_domains();
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gm::HasSubstr("not_a_real_domain"));
    }
}

TEST_F(sdk_core_domains_test, enables_rccl_and_ompt_when_flags_set_and_runtime_supports)
{
    using wrapper = tagged_backend<domains_callback>;

    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ "hip_api" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(true));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(true));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(),
                gm::UnorderedElementsAre(wrapper::CALLBACK_TRACING_RCCL_API,
                                         wrapper::CALLBACK_TRACING_OMPT));
}

TEST_F(sdk_core_domains_test,
       falls_back_to_name_matched_kind_for_domain_without_binding_entry)
{
    using wrapper = tagged_backend<domains_callback>;

    const auto callback_info     = wrapper::get_callback_tracing_names();
    const auto code_object_token = callback_domains_sut::to_lower(
        callback_info[static_cast<size_t>(wrapper::CALLBACK_TRACING_CODE_OBJECT)].name);

    callback_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 0;
            *pat = 0;
        });

    auto config = make_domains_settings({ code_object_token });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(code_object_token));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).WillOnce(gm::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).WillOnce(gm::Return(false));

    EXPECT_THAT(callback_domains_sut::get_callback_domains(),
                gm::UnorderedElementsAre(wrapper::CALLBACK_TRACING_CODE_OBJECT));
}

// ─── get_buffered_domains ─────────────────────────────────────────────────────
//
// kfd_supported_by_runtime is computed from Wrapper::compile_time_version alone
// (a compile-time constant), not from the dynamically-queried version — so the
// "old SDK" test uses a dedicated tag with a lowered FakeCompileTimeVersion while
// every other test uses the default (real installed SDK, >= 1.2.2 here).

using buffer_domains_sut = sdk_core<tagged_backend<domains_buffer>, mock_sdk_externals>;

TEST_F(sdk_core_domains_test, selects_kinds_bound_to_memory_copy_token)
{
    using wrapper = tagged_backend<domains_buffer>;

    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ "memory_copy" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "memory_copy" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(false));

    EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                gm::UnorderedElementsAre(wrapper::BUFFER_TRACING_MEMORY_COPY));
}

TEST_F(sdk_core_domains_test,
       selects_kinds_bound_to_hsa_hip_marker_and_memory_allocation_tokens)
{
    using wrapper = tagged_backend<domains_buffer>;
    using kind_t  = wrapper::buffer_tracing_kind;

    struct case_t
    {
        std::string         token;
        std::vector<kind_t> expected;
    };

    const std::vector<case_t> cases = {
        { "hsa_api",
          { wrapper::BUFFER_TRACING_HSA_CORE_API, wrapper::BUFFER_TRACING_HSA_AMD_EXT_API,
            wrapper::BUFFER_TRACING_HSA_IMAGE_EXT_API,
            wrapper::BUFFER_TRACING_HSA_FINALIZE_EXT_API } },
        { "hip_api",
          { wrapper::BUFFER_TRACING_HIP_COMPILER_API,
            wrapper::BUFFER_TRACING_HIP_RUNTIME_API } },
        { "roctx", { wrapper::BUFFER_TRACING_MARKER_CORE_API } },
        { "memory_allocation", { wrapper::BUFFER_TRACING_MEMORY_ALLOCATION } },
    };

    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.token);

        buffer_domains_sut::reset_version_cache();
        EXPECT_CALL(*g_mock, get_version)
            .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
                *maj = 1;
                *min = 3;
                *pat = 2;
            });

        auto config = make_domains_settings({ c.token });

        EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
        EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(c.token));
        EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
            .WillOnce(gm::Return(false));

        EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                    gm::UnorderedElementsAreArray(c.expected));

        gm::Mock::VerifyAndClearExpectations(g_mock.get());
        gm::Mock::VerifyAndClearExpectations(g_mock_externals.get());
    }
}

TEST_F(sdk_core_domains_test,
       buffer_throws_runtime_error_on_domain_token_not_in_valid_choices)
{
    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ "memory_copy" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "not_a_real_domain" }));
    // The invalid-domain throw happens inside the domain loop, before the trailing
    // unified-memory-profiling check runs — so that Externals call never happens.

    try
    {
        buffer_domains_sut::get_buffered_domains();
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gm::HasSubstr("not_a_real_domain"));
    }
}

TEST_F(sdk_core_domains_test, gates_kfd_events_and_warns_when_runtime_sdk_too_old)
{
    // 10100 (1.1.0) < the 1.2.2 KFD minimum, so kfd_supported_by_runtime is false.
    using old_kfd_sut =
        sdk_core<tagged_backend<domains_buffer_old_kfd, 10100u>, mock_sdk_externals>;

    old_kfd_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 1;
            *pat = 0;
        });

    auto config = make_domains_settings({ "kfd_events" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "kfd_events" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(false));

    EXPECT_THAT(old_kfd_sut::get_buffered_domains(), gm::IsEmpty());
}

TEST_F(sdk_core_domains_test,
       warns_when_unified_memory_profiling_requested_but_runtime_sdk_too_old)
{
    using old_kfd_sut =
        sdk_core<tagged_backend<domains_buffer_old_kfd, 10100u>, mock_sdk_externals>;

    old_kfd_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 1;
            *pat = 0;
        });

    auto config = make_domains_settings({ "memory_copy" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(true));

    // kfd_supported_by_runtime is false for this tag — the warning branch runs and
    // no KFD kinds are added.
    EXPECT_THAT(old_kfd_sut::get_buffered_domains(), gm::IsEmpty());
}

TEST_F(sdk_core_domains_test,
       selects_all_kfd_kinds_for_kfd_events_token_when_runtime_sdk_new_enough)
{
    using wrapper = tagged_backend<domains_buffer>;

    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ "kfd_events" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "kfd_events" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(false));

    EXPECT_THAT(
        buffer_domains_sut::get_buffered_domains(),
        gm::UnorderedElementsAre(wrapper::BUFFER_TRACING_KFD_PAGE_FAULT,
                                 wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                 wrapper::BUFFER_TRACING_KFD_QUEUE,
                                 wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                 wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                 wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

TEST_F(sdk_core_domains_test, selects_kfd_page_fault_when_runtime_sdk_new_enough)
{
    using wrapper = tagged_backend<domains_buffer>;

    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ "kfd_page_fault" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(std::string{ "kfd_page_fault" }));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(false));

    EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                gm::UnorderedElementsAre(wrapper::BUFFER_TRACING_KFD_PAGE_FAULT));
}

TEST_F(sdk_core_domains_test,
       selects_each_remaining_kfd_domain_token_when_runtime_sdk_new_enough)
{
    using wrapper = tagged_backend<domains_buffer>;
    using kind_t  = wrapper::buffer_tracing_kind;

    const std::vector<std::pair<std::string, kind_t>> cases = {
        { "kfd_page_migrate", wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE },
        { "kfd_queue", wrapper::BUFFER_TRACING_KFD_QUEUE },
        { "kfd_event_queue", wrapper::BUFFER_TRACING_KFD_EVENT_QUEUE },
        { "kfd_event_unmap_from_gpu", wrapper::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU },
        { "kfd_event_dropped_events", wrapper::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS },
    };

    for(const auto& [token, expected_kind] : cases)
    {
        SCOPED_TRACE(token);

        buffer_domains_sut::reset_version_cache();
        EXPECT_CALL(*g_mock, get_version)
            .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
                *maj = 1;
                *min = 3;
                *pat = 2;
            });

        auto config = make_domains_settings({ token });

        EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
        EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(token));
        EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
            .WillOnce(gm::Return(false));

        EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                    gm::UnorderedElementsAre(expected_kind));

        gm::Mock::VerifyAndClearExpectations(g_mock.get());
        gm::Mock::VerifyAndClearExpectations(g_mock_externals.get());
    }
}

TEST_F(sdk_core_domains_test,
       buffer_falls_back_to_name_matched_kind_for_domain_without_binding_entry)
{
    using wrapper = tagged_backend<domains_buffer>;

    const auto buffer_info          = wrapper::get_buffer_tracing_names();
    const auto scratch_memory_token = buffer_domains_sut::to_lower(
        buffer_info[static_cast<size_t>(wrapper::BUFFER_TRACING_SCRATCH_MEMORY)].name);

    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ scratch_memory_token });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .WillOnce(gm::Return(scratch_memory_token));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(false));

    EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                gm::UnorderedElementsAre(wrapper::BUFFER_TRACING_SCRATCH_MEMORY));
}

TEST_F(
    sdk_core_domains_test,
    enables_kfd_page_fault_and_migrate_when_unified_memory_profiling_and_runtime_new_enough)
{
    using wrapper = tagged_backend<domains_buffer>;

    buffer_domains_sut::reset_version_cache();
    EXPECT_CALL(*g_mock, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 3;
            *pat = 2;
        });

    auto config = make_domains_settings({ "memory_copy" });

    EXPECT_CALL(*g_mock_externals, get_settings).WillOnce(gm::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_rocm_domains).WillOnce(gm::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .WillOnce(gm::Return(true));

    EXPECT_THAT(buffer_domains_sut::get_buffered_domains(),
                gm::UnorderedElementsAre(wrapper::BUFFER_TRACING_KFD_PAGE_FAULT,
                                         wrapper::BUFFER_TRACING_KFD_PAGE_MIGRATE));
}

// ─── config_settings ──────────────────────────────────────────────────────────
//
// config_settings() never calls Externals::get_*() or Wrapper::get_version(), only
// Wrapper::get_{buffer,callback}_tracing_names() (the real, compile-time SDK name
// tables — no ROCm runtime required, same as the kind_impl tests). Each test uses
// its own Tag: _option_names inside config_settings() is a function-local static
// scoped per (Wrapper, Externals) instantiation, so a shared Tag across tests would
// make the 2nd/3rd test silently skip already-registered operation-filter settings.
//
// mock_sdk_externals is reused purely to supply Externals::Settings = fake_settings;
// none of its other methods are ever called by config_settings(), so g_mock_externals
// is never set up here.

TEST(sdk_core_config_settings, registers_rocm_domains_setting_with_expected_choices)
{
    using sut = sdk_core<tagged_backend<config_domains>, mock_sdk_externals>;

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS });
    ASSERT_NE(itr, config->end());

    const auto& choices = itr->second->get_choices();
    EXPECT_THAT(choices,
                gm::IsSupersetOf({ "hip_api", "hsa_api", "marker_api", "roctx" }));
    EXPECT_THAT(choices, gm::Not(gm::Contains(std::string{ "code_object" })));
    EXPECT_THAT(choices, gm::Not(gm::Contains(std::string{ "none" })));

    EXPECT_EQ(itr->second->get<std::string>().second,
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
}

TEST(sdk_core_config_settings, registers_rocm_events_setting)
{
    using sut = sdk_core<tagged_backend<config_events>, mock_sdk_externals>;

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_EVENTS });
    ASSERT_NE(itr, config->end());
    EXPECT_EQ(itr->second->get<std::string>().second, std::string{});
}

TEST(sdk_core_config_settings, registers_operation_filter_settings_for_marker_api_domain)
{
    using sut = sdk_core<tagged_backend<config_operations>, mock_sdk_externals>;

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    for(const char* name : { "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS",
                             "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE",
                             "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_ANNOTATE_BACKTRACE" })
    {
        SCOPED_TRACE(name);
        auto itr = config->find(std::string{ name });
        ASSERT_NE(itr, config->end());
        EXPECT_THAT(itr->second->get_choices(), gm::Not(gm::IsEmpty()));
    }
}

TEST(sdk_core_config_settings,
     calling_twice_on_same_config_logs_duplicate_and_keeps_value)
{
    // insert_config_setting()'s "already registered" branch only fires when the
    // *same config object* already has that env var — calling config_settings()
    // twice on one fake_settings triggers it for ROCM_DOMAINS/ROCM_EVENTS (which,
    // unlike the per-domain operation settings, aren't guarded by _option_names).
    using sut = sdk_core<tagged_backend<config_duplicate>, mock_sdk_externals>;

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS });
    ASSERT_NE(itr, config->end());
    EXPECT_EQ(itr->second->get<std::string>().second,
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
}

// ─── get_rocm_events ──────────────────────────────────────────────────────────

using rocm_events_sut = sdk_core<tagged_backend<rocm_events>, mock_sdk_externals>;

TEST_F(sdk_core_domains_test, get_rocm_events_splits_delimited_setting_string)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting)
        .WillOnce(gm::Return(std::string{ "EventA,EventB;EventC" }));

    EXPECT_THAT(rocm_events_sut::get_rocm_events(),
                gm::ElementsAre("EventA", "EventB", "EventC"));
}

TEST_F(sdk_core_domains_test, get_rocm_events_returns_empty_for_empty_setting)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting)
        .WillOnce(gm::Return(std::string{}));

    EXPECT_THAT(rocm_events_sut::get_rocm_events(), gm::IsEmpty());
}

}  // namespace rocprofsys::rocprofiler_sdk::testing
