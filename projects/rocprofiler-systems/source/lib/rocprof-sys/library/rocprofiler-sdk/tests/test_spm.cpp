// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace
{
using rocprofsys::rocprofiler_sdk::spm::is_config_valid;
using rocprofsys::rocprofiler_sdk::spm::request;
using rocprofsys::rocprofiler_sdk::spm::sdk_beta_opt_in_enabled;

constexpr auto sdk_spm_beta_env = "ROCPROFILER_SPM_BETA_ENABLED";

class scoped_env
{
public:
    explicit scoped_env(const char* name)
    : m_name{ name }
    {
        if(const auto* value = std::getenv(name)) m_original = std::string{ value };
    }

    ~scoped_env()
    {
        if(m_original)
            ::setenv(m_name, m_original->c_str(), 1);
        else
            ::unsetenv(m_name);
    }

    scoped_env(const scoped_env&)            = delete;
    scoped_env& operator=(const scoped_env&) = delete;
    scoped_env(scoped_env&&)                 = delete;
    scoped_env& operator=(scoped_env&&)      = delete;

private:
    const char*                m_name = nullptr;
    std::optional<std::string> m_original;
};

class spm_settings_test : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        auto settings = rocprofsys::settings::shared_instance();
        rocprofsys::rocprofiler_sdk::config_settings(settings);
    }

    void TearDown() override
    {
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS }, std::string{});
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
            std::uint64_t{ 0 });
    }
};

request
make_valid_requested_spm_request()
{
    const auto unit =
        std::string{ rocprofsys::env_vars::SPM_SAMPLE_INTERVAL_UNIT_SCLK_CYCLES };
    return request{ { "SQ_WAVES" }, 4200, unit };
}
}  // namespace

TEST_F(spm_settings_test, from_settings_reflects_configured_spm_settings)
{
    const auto unit =
        std::string{ rocprofsys::env_vars::SPM_SAMPLE_INTERVAL_UNIT_SCLK_CYCLES };

    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
        std::string{ "SQ_WAVES,TD_TD_BUSY" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 4200 }));

    const auto spm_request = request::from_settings();

    EXPECT_TRUE(spm_request.requested());
    ASSERT_EQ(spm_request.events.size(), 2);
    EXPECT_EQ(spm_request.events.at(0), "SQ_WAVES");
    EXPECT_EQ(spm_request.events.at(1), "TD_TD_BUSY");
    EXPECT_EQ(spm_request.sample_interval, 4200);
    EXPECT_EQ(spm_request.sample_interval_unit, unit);
}

TEST(spm_request, requested_reflects_events)
{
    EXPECT_FALSE(request{}.requested());

    auto event_request   = request{};
    event_request.events = { "SQ_WAVES" };
    EXPECT_TRUE(event_request.requested());
}

TEST_F(spm_settings_test, events_request_spm_but_default_interval_is_invalid)
{
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS }, std::string{ "SQ_WAVES" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 0 }));

    const auto spm_request = request::from_settings();

    EXPECT_TRUE(spm_request.requested());
    EXPECT_EQ(spm_request.events, std::vector<std::string>{ "SQ_WAVES" });
    EXPECT_EQ(spm_request.sample_interval, 0);
    EXPECT_FALSE(is_config_valid(spm_request, {}, {}));
}

TEST(spm_config_validation, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(is_config_valid(request{}, {}, {}));
}

TEST(spm_config_validation, rejects_rocm_dispatch_counter_conflict)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_FALSE(is_config_valid(request, { "SQ_WAVES" }, {}));
}

TEST(spm_config_validation, rejects_gpu_perf_counter_conflict)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_FALSE(is_config_valid(request, {}, "SQ_WAVES"));
}

TEST(spm_config_validation, rejects_zero_sample_interval)
{
    auto request            = make_valid_requested_spm_request();
    request.sample_interval = 0;

    EXPECT_FALSE(is_config_valid(request, {}, {}));
}

TEST(spm_config_validation, rejects_unsupported_sample_interval_unit)
{
    auto request                 = make_valid_requested_spm_request();
    request.sample_interval_unit = "ns";

    EXPECT_FALSE(is_config_valid(request, {}, {}));
}

TEST(spm_config_validation, accepts_valid_requested_spm_request)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_TRUE(is_config_valid(request, {}, {}));
}

TEST(spm_beta_opt_in, accepts_when_spm_is_not_requested)
{
    scoped_env env{ sdk_spm_beta_env };
    ::unsetenv(sdk_spm_beta_env);

    EXPECT_TRUE(sdk_beta_opt_in_enabled(request{}));
}

TEST(spm_beta_opt_in, rejects_requested_spm_without_sdk_beta_env)
{
    scoped_env env{ sdk_spm_beta_env };
    ::unsetenv(sdk_spm_beta_env);

    EXPECT_FALSE(sdk_beta_opt_in_enabled(make_valid_requested_spm_request()));
}

TEST(spm_beta_opt_in, accepts_requested_spm_with_sdk_beta_env)
{
    scoped_env env{ sdk_spm_beta_env };
    ::setenv(sdk_spm_beta_env, "ON", 1);

    EXPECT_TRUE(sdk_beta_opt_in_enabled(make_valid_requested_spm_request()));
}
