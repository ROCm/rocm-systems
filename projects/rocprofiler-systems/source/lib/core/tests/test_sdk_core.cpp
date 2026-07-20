// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/rocprofiler-sdk.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
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
    version_fields    = 1,
    version_formatted = 2,
    version_caching   = 3,
    ops_throw         = 60,
    config_domains    = 80,
    config_events     = 81,
    config_operations = 82,
    config_duplicate  = 83,
    rocm_events       = 90,
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

// ─── config_settings ──────────────────────────────────────────────────────────
//
// config_settings() never calls Externals::get_*() or Wrapper::get_version(), only
// Wrapper::get_{buffer,callback}_tracing_names() (the real, compile-time SDK name
// tables — no ROCm runtime required). Each test uses
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
