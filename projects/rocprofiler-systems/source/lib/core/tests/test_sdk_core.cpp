// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/rocprofiler-sdk.hpp"
#include "core/tests/mock_wrapper.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace rocprofsys::rocprofiler_sdk::testing
{

namespace gtest = ::testing;

using ::rocprofsys::mock::rocprofiler_sdk::g_mock_wrapper;
using ::rocprofsys::mock::rocprofiler_sdk::gmock_wrapper;
using mock_backend = ::rocprofsys::mock::rocprofiler_sdk::wrapper;

// ─── Backend policy ───────────────────────────────────────────────────────────
//
// tagged_backend is a self-contained mock (no real SDK headers): it inherits
// mock_backend, whose get_version()/get_callback_tracing_names()/
// get_buffer_tracing_names() all forward to g_mock_wrapper.
//
// Each distinct Tag gives sdk_core<tagged_backend<Tag>>::get_version() its own
// static _version cache, so tests that verify first-call behaviour don't
// interfere with one another.

// Each enumerator maps to a distinct Backend type, giving sdk_core<tagged_backend<N>>
// its own static caches (get_version cache, operation option maps, tracing info tables).
enum backend_tag : int
{
    version_fields                  = 1,
    version_formatted               = 2,
    version_caching                 = 3,
    ops_throw                       = 60,
    config_domains                  = 80,
    config_events                   = 81,
    config_operations               = 82,
    config_duplicate                = 83,
    callback_backtrace_operations   = 84,
    buffered_backtrace_operations   = 85,
    callback_operations             = 86,
    buffered_operations             = 87,
    buffered_domains_memory_copy    = 88,
    buffered_domains_aliases        = 89,
    rocm_events                     = 90,
    buffered_domains_kfd_events     = 91,
    buffered_domains_kfd_individual = 92,
    buffered_domains_allocation     = 93,
    buffered_domains_unified_memory = 94,
    buffered_domains_generic_lookup = 95,
    buffered_domains_invalid        = 96,
    buffered_domains_page_migration = 97,
    callback_domains_aliases        = 98,
    callback_domains_generic_lookup = 99,
    callback_domains_implicit_flags = 100,
    callback_domains_invalid        = 101,
    callback_domains_version_gating = 102,
};

template <int Tag>
struct tagged_backend : mock_backend
{};

// buffered_domains_page_migration simulates a pre-1.0 SDK: below the KFD gate
// (>= 10000), so get_buffered_domains() falls back to the legacy
// BUFFER_TRACING_PAGE_MIGRATION path instead of the granular KFD_* domains.
template <>
struct tagged_backend<buffered_domains_page_migration> : mock_backend
{
    static constexpr std::uint32_t compile_time_version = 500;
};

// ─── Shared fake name tables ───────────────────────────────────────────────────
//
// Every test that calls config_settings()/get_buffered_domains()/
// get_callback_domains() needs Wrapper::get_buffer_tracing_names()/
// get_callback_tracing_names() (both GMock methods on mock_backend) to return a
// populated table, since sdk_core validates/resolves domain names against it.

// Names mirror the real SDK's tracing-kind name tables, which are UPPER_SNAKE_CASE
// (e.g. "MEMORY_COPY", not "memory_copy"): config_settings()'s per-domain
// operation-filter env var names are built directly from this raw name (not
// lowercased) — only the ROCPROFSYS_ROCM_DOMAINS choice list is lowercased.
mock_backend::buffer_name_info_t
make_buffer_name_info()
{
    auto table = mock_backend::buffer_name_info_t{};
    table.emplace(mock_backend::BUFFER_TRACING_HSA_CORE_API, "HSA_CORE_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_AMD_EXT_API, "HSA_AMD_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_IMAGE_EXT_API, "HSA_IMAGE_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HSA_FINALIZE_EXT_API,
                  "HSA_FINALIZE_EXT_API");
    table.emplace(mock_backend::BUFFER_TRACING_HIP_COMPILER_API, "HIP_COMPILER_API");
    table.emplace(mock_backend::BUFFER_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API");
    table.emplace(mock_backend::BUFFER_TRACING_MARKER_CORE_API, "MARKER_CORE_API");
    table.emplace(mock_backend::BUFFER_TRACING_KERNEL_DISPATCH, "KERNEL_DISPATCH");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, "MEMORY_COPY");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 0, "HOST_TO_DEVICE");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 1, "DEVICE_TO_HOST");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_COPY, 2, "DEVICE_TO_DEVICE");
    table.emplace(mock_backend::BUFFER_TRACING_SCRATCH_MEMORY, "SCRATCH_MEMORY");
    table.emplace(mock_backend::BUFFER_TRACING_MEMORY_ALLOCATION, "MEMORY_ALLOCATION");
    table.emplace(mock_backend::BUFFER_TRACING_PAGE_MIGRATION, "PAGE_MIGRATION");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_PAGE_FAULT, "KFD_PAGE_FAULT");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_PAGE_MIGRATE, "KFD_PAGE_MIGRATE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_QUEUE, "KFD_QUEUE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_QUEUE, "KFD_EVENT_QUEUE");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                  "KFD_EVENT_UNMAP_FROM_GPU");
    table.emplace(mock_backend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
                  "KFD_EVENT_DROPPED_EVENTS");
    return table;
}

mock_backend::callback_name_info_t
make_callback_name_info()
{
    auto table = mock_backend::callback_name_info_t{};
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_CORE_API, "HSA_CORE_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_AMD_EXT_API, "HSA_AMD_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_IMAGE_EXT_API, "HSA_IMAGE_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
                  "HSA_FINALIZE_EXT_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_RUNTIME_API, "HIP_RUNTIME_API");
    table.emplace(mock_backend::CALLBACK_TRACING_HIP_COMPILER_API, "HIP_COMPILER_API");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API, "MARKER_CORE_API");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxMarkA, "roctxMarkA");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxRangePushA, "roctxRangePushA");
    table.emplace(mock_backend::CALLBACK_TRACING_MARKER_CORE_API,
                  mock_backend::MARKER_CORE_API_ID_roctxRangePop, "roctxRangePop");
    table.emplace(mock_backend::CALLBACK_TRACING_CODE_OBJECT, "CODE_OBJECT");
    table.emplace(mock_backend::CALLBACK_TRACING_RCCL_API, "RCCL_API");
    table.emplace(mock_backend::CALLBACK_TRACING_OMPT, "OMPT");
    return table;
}

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
    MOCK_METHOD(void, set_state, (std::uint32_t));
};

inline std::unique_ptr<gtest::StrictMock<gmock_sdk_externals>> g_mock_externals;

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

    using StateType                           = std::uint32_t;
    constexpr static StateType StateFinalized = 3;
    static void set_state(StateType state) { g_mock_externals->set_state(state); }
};

// ─── Fixtures ─────────────────────────────────────────────────────────────────

class sdk_core_test : public ::testing::Test
{
protected:
    void SetUp() override { g_mock_wrapper = std::make_unique<gmock_wrapper>(); }
    void TearDown() override { g_mock_wrapper.reset(); }
};

// config_settings() only ever touches Wrapper::get_{buffer,callback}_tracing_names()
// (never Externals), so it reuses sdk_core_test's g_mock_wrapper-only setup.
class sdk_core_config_settings_test : public sdk_core_test
{};

// Fixture for functions that read through the Externals policy
// (get_callback_domains, get_buffered_domains, get_rocm_events).
class sdk_core_domains_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        g_mock_wrapper   = std::make_unique<gmock_wrapper>();
        g_mock_externals = std::make_unique<gtest::StrictMock<gmock_sdk_externals>>();
    }
    void TearDown() override
    {
        g_mock_wrapper.reset();
        g_mock_externals.reset();
    }
};

// ─── get_version ─────────────────────────────────────────────────────────────

TEST_F(sdk_core_test, get_version_populates_major_minor_patch)
{
    using sut = sdk_core<tagged_backend<version_fields>>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 2;
            *pat = 3;
            return 0;
        });

    auto& ver = sut::get_version();
    EXPECT_EQ(ver.major, 1u);
    EXPECT_EQ(ver.minor, 2u);
    EXPECT_EQ(ver.patch, 3u);
}

TEST_F(sdk_core_test, get_version_formatted_equals_major_10000_minor_100_patch)
{
    using sut = sdk_core<tagged_backend<version_formatted>>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 1;
            *min = 1;
            *pat = 0;
            return 0;
        });

    EXPECT_EQ(sut::get_version().formatted(), (1u * 10000u) + (1u * 100u) + 0u);
}

TEST_F(sdk_core_test, get_version_caches_result_calling_backend_exactly_once)
{
    using sut = sdk_core<tagged_backend<version_caching>>;

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* maj, std::uint32_t* min, std::uint32_t* pat) {
            *maj = 2;
            *min = 0;
            *pat = 0;
            return 0;
        });

    auto& ver1 = sut::get_version();
    auto& ver2 = sut::get_version();
    auto& ver3 = sut::get_version();

    EXPECT_EQ(&ver1, &ver2);  // Same cached object
    EXPECT_EQ(&ver2, &ver3);
    EXPECT_EQ(ver1.formatted(), 20000u);
}

// ─── throw on unregistered kind ───────────────────────────────────────────────
//
// tagged_backend<60> has an empty map — no kind was ever injected — so all
// get_operations / get_backtrace_operations calls must throw std::runtime_error.
// These never reach a Wrapper call (the "no options registered" check throws
// first), so no g_mock_wrapper setup is needed.

using throw_sut = sdk_core<tagged_backend<ops_throw>>;

TEST(sdk_core_throw, get_operations_callback_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_operations(
                     tagged_backend<ops_throw>::CALLBACK_TRACING_HIP_RUNTIME_API),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_operations_buffer_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_operations(
                     tagged_backend<ops_throw>::BUFFER_TRACING_KERNEL_DISPATCH),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_backtrace_callback_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_backtrace_operations(
                     tagged_backend<ops_throw>::CALLBACK_TRACING_HIP_RUNTIME_API),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_backtrace_buffer_throws_when_kind_not_registered)
{
    EXPECT_THROW(throw_sut::get_backtrace_operations(
                     tagged_backend<ops_throw>::BUFFER_TRACING_KERNEL_DISPATCH),
                 std::runtime_error);
}

TEST(sdk_core_throw, get_operations_error_message_contains_kind_value)
{
    try
    {
        throw_sut::get_operations(
            tagged_backend<ops_throw>::CALLBACK_TRACING_HIP_RUNTIME_API);
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& ex)
    {
        EXPECT_THAT(ex.what(), gtest::HasSubstr("callback tracing kind"));
    }
}

// ─── config_settings ──────────────────────────────────────────────────────────
//
// config_settings() always queries Wrapper::get_{buffer,callback}_tracing_names()
// exactly once per call (regardless of domain), so every test below sets up
// exactly matching EXPECT_CALLs. Each test uses its own Tag: _option_names inside
// config_settings() is a function-local static scoped per (Wrapper, Externals)
// instantiation, so a shared Tag across tests would make the 2nd/3rd test
// silently skip already-registered operation-filter settings.
//
// mock_sdk_externals is reused purely to supply Externals::Settings = fake_settings;
// none of its other methods are ever called by config_settings(), so g_mock_externals
// is never set up here.

TEST_F(sdk_core_config_settings_test,
       registers_rocm_domains_setting_with_expected_choices)
{
    using sut = sdk_core<tagged_backend<config_domains>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS });
    ASSERT_NE(itr, config->end());

    const auto& choices = itr->second->get_choices();
    EXPECT_THAT(choices,
                gtest::IsSupersetOf({ "hip_api", "hsa_api", "marker_api", "roctx" }));
    EXPECT_THAT(choices, gtest::Not(gtest::Contains(std::string{ "code_object" })));
    EXPECT_THAT(choices, gtest::Not(gtest::Contains(std::string{ "none" })));

    EXPECT_EQ(itr->second->get<std::string>().second,
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
}

TEST_F(sdk_core_config_settings_test, registers_rocm_events_setting)
{
    using sut = sdk_core<tagged_backend<config_events>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_EVENTS });
    ASSERT_NE(itr, config->end());
    EXPECT_EQ(itr->second->get<std::string>().second, std::string{});
}

TEST_F(sdk_core_config_settings_test,
       registers_operation_filter_settings_for_marker_api_domain)
{
    using sut = sdk_core<tagged_backend<config_operations>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    for(const char* name : { "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS",
                             "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE",
                             "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_ANNOTATE_BACKTRACE" })
    {
        SCOPED_TRACE(name);
        auto itr = config->find(std::string{ name });
        ASSERT_NE(itr, config->end());
        EXPECT_THAT(itr->second->get_choices(), gtest::Not(gtest::IsEmpty()));
    }
}

TEST_F(sdk_core_config_settings_test,
       calling_twice_on_same_config_logs_duplicate_and_keeps_value)
{
    // insert_config_setting()'s "already registered" branch only fires when the
    // *same config object* already has that env var — calling config_settings()
    // twice on one fake_settings triggers it for ROCM_DOMAINS/ROCM_EVENTS (which,
    // unlike the per-domain operation settings, aren't guarded by _option_names).
    using sut = sdk_core<tagged_backend<config_duplicate>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);
    sut::config_settings(config);

    auto itr = config->find(std::string{ ::rocprofsys::env_vars::ROCM_DOMAINS });
    ASSERT_NE(itr, config->end());
    EXPECT_EQ(itr->second->get<std::string>().second,
              "hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory");
}

// ─── get_callback_domains ─────────────────────────────────────────────────────
//
// Every test below calls config_settings() then get_callback_domains(), each of
// which independently calls Wrapper::get_callback_tracing_names() once (2 total);
// get_buffer_tracing_names() is only called by config_settings() (1 total).

TEST_F(sdk_core_domains_test, get_callback_domains_aliases_expand_to_exact_domains)
{
    using backend_t = tagged_backend<callback_domains_aliases>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "hsa_api,hip_api,marker_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    EXPECT_THAT(
        sut::get_callback_domains(),
        gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_HSA_CORE_API,
                                    backend_t::CALLBACK_TRACING_HSA_AMD_EXT_API,
                                    backend_t::CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                                    backend_t::CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
                                    backend_t::CALLBACK_TRACING_HIP_RUNTIME_API,
                                    backend_t::CALLBACK_TRACING_HIP_COMPILER_API,
                                    backend_t::CALLBACK_TRACING_MARKER_CORE_API));
}

TEST_F(sdk_core_domains_test,
       get_callback_domains_supported_callback_info_name_returns_domain)
{
    using backend_t = tagged_backend<callback_domains_generic_lookup>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "rccl_api" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    EXPECT_THAT(sut::get_callback_domains(),
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_RCCL_API));
}

TEST_F(sdk_core_domains_test,
       get_callback_domains_rccl_and_ompt_flags_return_both_domains)
{
    using backend_t = tagged_backend<callback_domains_implicit_flags>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(true));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(true));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    EXPECT_THAT(sut::get_callback_domains(),
                gtest::UnorderedElementsAre(backend_t::CALLBACK_TRACING_RCCL_API,
                                            backend_t::CALLBACK_TRACING_OMPT));
}

TEST_F(sdk_core_domains_test, get_callback_domains_invalid_domain)
{
    using backend_t = tagged_backend<callback_domains_invalid>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "invalid_domain" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(false));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    try
    {
        static_cast<void>(sut::get_callback_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(),
                     "unsupported ROCPROFSYS_ROCM_DOMAINS value: invalid_domain");
    }
}

TEST_F(sdk_core_domains_test,
       get_callback_domains_runtime_before_0_6_excludes_rccl_and_ompt)
{
    using backend_t = tagged_backend<callback_domains_version_gating>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 0;
            *minor = 5;
            *patch = 99;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "rccl_api,ompt" }));
    EXPECT_CALL(*g_mock_externals, get_use_rcclp).Times(1).WillOnce(gtest::Return(true));
    EXPECT_CALL(*g_mock_externals, get_use_ompt).Times(1).WillOnce(gtest::Return(true));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    EXPECT_THAT(sut::get_callback_domains(), gtest::IsEmpty());
}

// ─── get_buffered_domains ─────────────────────────────────────────────────────
//
// Every test below calls config_settings() then get_buffered_domains(), each of
// which independently calls Wrapper::get_buffer_tracing_names() once (2 total);
// get_callback_tracing_names() is only called by config_settings() (1 total).

TEST_F(sdk_core_domains_test, get_buffered_domains_memory_copy_returns_memory_copy)
{
    using backend_t = tagged_backend<buffered_domains_memory_copy>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_copy" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_COPY));
}

TEST_F(sdk_core_domains_test, get_buffered_domains_aliases_expand_to_exact_domains)
{
    using backend_t = tagged_backend<buffered_domains_aliases>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "hsa_api,hip_api,marker_api" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_HSA_CORE_API,
                                    backend_t::BUFFER_TRACING_HSA_AMD_EXT_API,
                                    backend_t::BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                    backend_t::BUFFER_TRACING_HSA_FINALIZE_EXT_API,
                                    backend_t::BUFFER_TRACING_HIP_COMPILER_API,
                                    backend_t::BUFFER_TRACING_HIP_RUNTIME_API,
                                    backend_t::BUFFER_TRACING_MARKER_CORE_API));
}

TEST_F(sdk_core_domains_test, get_buffered_domains_kfd_events_returns_all_kfd_domains)
{
    using backend_t = tagged_backend<buffered_domains_kfd_events>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "kfd_events" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                    backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                    backend_t::BUFFER_TRACING_KFD_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

TEST_F(sdk_core_domains_test,
       get_buffered_domains_individual_kfd_names_return_all_kfd_domains)
{
    using backend_t = tagged_backend<buffered_domains_kfd_individual>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(
            std::string{ "kfd_page_fault, kfd_page_migrate, kfd_queue, kfd_event_queue, "
                         "kfd_event_unmap_from_gpu, kfd_event_dropped_events" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(
        sut::get_buffered_domains(),
        gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                    backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE,
                                    backend_t::BUFFER_TRACING_KFD_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_QUEUE,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                                    backend_t::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS));
}

TEST_F(sdk_core_domains_test,
       get_buffered_domains_memory_allocation_returns_memory_allocation)
{
    using backend_t = tagged_backend<buffered_domains_allocation>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "memory_allocation" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_MEMORY_ALLOCATION));
}

TEST_F(sdk_core_domains_test,
       get_buffered_domains_unified_memory_enables_page_fault_and_migrate)
{
    using backend_t = tagged_backend<buffered_domains_unified_memory>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{}));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(true));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_KFD_PAGE_FAULT,
                                            backend_t::BUFFER_TRACING_KFD_PAGE_MIGRATE));
}

TEST_F(sdk_core_domains_test,
       get_buffered_domains_supported_buffer_info_name_returns_domain)
{
    using backend_t = tagged_backend<buffered_domains_generic_lookup>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "scratch_memory" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));
    EXPECT_CALL(*g_mock_externals, get_use_unified_memory_profiling)
        .Times(1)
        .WillOnce(gtest::Return(false));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(backend_t::BUFFER_TRACING_SCRATCH_MEMORY));
}

TEST_F(sdk_core_domains_test, get_buffered_domains_invalid_domain_throws)
{
    using backend_t = tagged_backend<buffered_domains_invalid>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_wrapper, get_version)
        .Times(1)
        .WillOnce([](std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
            *major = 1;
            *minor = 2;
            *patch = 2;
            return 0;
        });
    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "invalid_domain" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    try
    {
        static_cast<void>(sut::get_buffered_domains());
        FAIL() << "Expected std::runtime_error";
    } catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(),
                     "unsupported ROCPROFSYS_ROCM_DOMAINS value: invalid_domain");
    }
}

TEST_F(sdk_core_domains_test,
       get_buffered_domains_legacy_page_migration_returns_page_migration)
{
    using sut =
        sdk_core<tagged_backend<buffered_domains_page_migration>, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_externals, get_rocm_domains)
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "page_migration" }));
    EXPECT_CALL(*g_mock_externals, get_settings)
        .Times(1)
        .WillOnce(gtest::Return(config.get()));

    EXPECT_THAT(sut::get_buffered_domains(),
                gtest::UnorderedElementsAre(
                    tagged_backend<
                        buffered_domains_page_migration>::BUFFER_TRACING_PAGE_MIGRATION));
}

// ─── get_operations ───────────────────────────────────────────────────────────

TEST_F(sdk_core_domains_test,
       get_operations_callback_applies_include_and_exclude_settings)
{
    using backend_t = tagged_backend<callback_operations>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq("ROCPROFSYS_ROCM_MARKER_API_OPERATIONS")))
        .Times(1)
        .WillOnce(
            gtest::Return(std::string{ "roctxMarkA|roctxRangePushA|roctxRangePop" }));
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::Eq(
                                       "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "roctxRangePushA" }));

    EXPECT_THAT(
        sut::get_operations(backend_t::CALLBACK_TRACING_MARKER_CORE_API),
        gtest::ElementsAre(
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxMarkA),
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxRangePop)));
}

TEST_F(sdk_core_domains_test,
       get_operations_buffered_applies_include_and_exclude_settings)
{
    using backend_t = tagged_backend<buffered_operations>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    // Memory-copy operation ids: match the order populated by make_buffer_name_info()
    // ("HOST_TO_DEVICE"=0, "DEVICE_TO_HOST"=1, "DEVICE_TO_DEVICE"=2).
    constexpr std::int32_t k_memory_copy_host_to_device = 0;
    constexpr std::int32_t k_memory_copy_device_to_host = 1;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq("ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS")))
        .Times(1)
        .WillOnce(gtest::Return(
            std::string{ "HOST_TO_DEVICE|DEVICE_TO_HOST|DEVICE_TO_DEVICE" }));
    EXPECT_CALL(*g_mock_externals, get_setting_value(gtest::Eq(
                                       "ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS_EXCLUDE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "DEVICE_TO_DEVICE" }));

    EXPECT_THAT(
        sut::get_operations(backend_t::BUFFER_TRACING_MEMORY_COPY),
        gtest::ElementsAre(k_memory_copy_host_to_device, k_memory_copy_device_to_host));
}

// ─── get_backtrace_operations ─────────────────────────────────────────────────

TEST_F(sdk_core_domains_test,
       get_backtrace_operations_callback_returns_operations_matching_setting)
{
    using backend_t = tagged_backend<callback_backtrace_operations>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq(
                    "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_ANNOTATE_BACKTRACE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "roctxMarkA|roctxRangePop" }));

    EXPECT_THAT(
        sut::get_backtrace_operations(backend_t::CALLBACK_TRACING_MARKER_CORE_API),
        gtest::UnorderedElementsAre(
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxMarkA),
            static_cast<std::int32_t>(backend_t::MARKER_CORE_API_ID_roctxRangePop)));
}

TEST_F(sdk_core_domains_test,
       get_backtrace_operations_buffered_returns_operations_matching_setting)
{
    using backend_t = tagged_backend<buffered_backtrace_operations>;
    using sut       = sdk_core<backend_t, mock_sdk_externals>;

    // Memory-copy operation ids: match the order populated by make_buffer_name_info()
    // ("HOST_TO_DEVICE"=0, "DEVICE_TO_HOST"=1, "DEVICE_TO_DEVICE"=2).
    constexpr std::int32_t k_memory_copy_host_to_device = 0;
    constexpr std::int32_t k_memory_copy_device_to_host = 1;

    EXPECT_CALL(*g_mock_wrapper, get_buffer_tracing_names)
        .Times(2)
        .WillRepeatedly(gtest::Return(make_buffer_name_info()));
    EXPECT_CALL(*g_mock_wrapper, get_callback_tracing_names)
        .Times(1)
        .WillOnce(gtest::Return(make_callback_name_info()));

    auto config = std::make_shared<fake_settings>();
    sut::config_settings(config);

    EXPECT_CALL(*g_mock_externals,
                get_setting_value(gtest::Eq(
                    "ROCPROFSYS_ROCM_MEMORY_COPY_OPERATIONS_ANNOTATE_BACKTRACE")))
        .Times(1)
        .WillOnce(gtest::Return(std::string{ "HOST_TO_DEVICE|DEVICE_TO_HOST" }));

    EXPECT_THAT(sut::get_backtrace_operations(backend_t::BUFFER_TRACING_MEMORY_COPY),
                gtest::UnorderedElementsAre(k_memory_copy_host_to_device,
                                            k_memory_copy_device_to_host));
}

// ─── get_rocm_events ──────────────────────────────────────────────────────────

using rocm_events_sut = sdk_core<tagged_backend<rocm_events>, mock_sdk_externals>;

TEST_F(sdk_core_domains_test, get_rocm_events_splits_delimited_setting_string)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting)
        .WillOnce(gtest::Return(std::string{ "EventA,EventB;EventC" }));

    EXPECT_THAT(rocm_events_sut::get_rocm_events(),
                gtest::ElementsAre("EventA", "EventB", "EventC"));
}

TEST_F(sdk_core_domains_test, get_rocm_events_returns_empty_for_empty_setting)
{
    EXPECT_CALL(*g_mock_externals, get_rocm_events_setting)
        .WillOnce(gtest::Return(std::string{}));

    EXPECT_THAT(rocm_events_sut::get_rocm_events(), gtest::IsEmpty());
}

}  // namespace rocprofsys::rocprofiler_sdk::testing
