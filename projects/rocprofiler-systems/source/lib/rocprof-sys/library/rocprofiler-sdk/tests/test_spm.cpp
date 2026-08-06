// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/env_vars.hpp"
#include "common/environment.hpp"
#include "core/config.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "rocprof-sys/library/rocprofiler-sdk/spm.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
using rocprofsys::rocprofiler_sdk::spm::is_config_valid;
using rocprofsys::rocprofiler_sdk::spm::is_spm_enabled_for_sdk;
using rocprofsys::rocprofiler_sdk::spm::request;

constexpr auto sdk_spm_beta_env = "ROCPROFILER_SPM_BETA_ENABLED";

struct fake_env
{
    inline static std::unordered_map<std::string, std::string> store;

    static int setenv(const char* name, const char* value, int overwrite)
    {
        if(!overwrite && store.count(name) > 0) return 0;
        store[name] = value;
        return 0;
    }

    static char* getenv(const char* name)
    {
        auto it = store.find(name);
        return it != store.end() ? it->second.data() : nullptr;
    }

    static void reset() { store.clear(); }
};

using fake_environment = rocprofsys::common::environment<fake_env>;

void
ensure_spm_settings_registered()
{
    auto settings = rocprofsys::settings::shared_instance();
    if(settings->find(std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS }) ==
           settings->end() ||
       settings->find(std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL }) ==
           settings->end())
    {
        rocprofsys::rocprofiler_sdk::config_settings(settings);
    }
}

class spm_settings_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ensure_spm_settings_registered();
        previous_events = rocprofsys::config::get_setting_value<std::string>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS });
        previous_sample_interval = rocprofsys::config::get_setting_value<std::uint64_t>(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL });
    }

    void TearDown() override
    {
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
            previous_events.value_or(std::string{}));
        rocprofsys::config::set_setting_value(
            std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
            previous_sample_interval.value_or(std::uint64_t{ 0 }));
    }

    std::optional<std::string>   previous_events          = std::nullopt;
    std::optional<std::uint64_t> previous_sample_interval = std::nullopt;
};

class beta_opt_in_test : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

request
make_valid_requested_spm_request()
{
    return request{ { "SQ_WAVES" }, 4200 };
}
}  // namespace

TEST_F(spm_settings_test, accessors_reflect_configured_spm_settings)
{
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_EVENTS },
        std::string{ "SQ_WAVES,TD_TD_BUSY" }));
    ASSERT_TRUE(rocprofsys::config::set_setting_value(
        std::string{ rocprofsys::env_vars::ROCM_SPM_SAMPLE_INTERVAL },
        std::uint64_t{ 4200 }));

    const auto events = rocprofsys::rocprofiler_sdk::spm::get_events();

    ASSERT_EQ(events.size(), 2);
    EXPECT_EQ(events.at(0), "SQ_WAVES");
    EXPECT_EQ(events.at(1), "TD_TD_BUSY");
    EXPECT_EQ(rocprofsys::rocprofiler_sdk::spm::get_sample_interval(), 4200);
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

    const auto events = rocprofsys::rocprofiler_sdk::spm::get_events();

    EXPECT_EQ(events, std::vector<std::string>{ "SQ_WAVES" });
    EXPECT_EQ(rocprofsys::rocprofiler_sdk::spm::get_sample_interval(), 0);
    EXPECT_FALSE(is_config_valid(request{ events, 0 }, {}, {}));
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

TEST(spm_config_validation, accepts_valid_requested_spm_request)
{
    const auto request = make_valid_requested_spm_request();

    EXPECT_TRUE(is_config_valid(request, {}, {}));
}

TEST_F(beta_opt_in_test, accepts_when_spm_is_not_requested)
{
    EXPECT_TRUE(is_spm_enabled_for_sdk<fake_environment>(request{}));
}

TEST_F(beta_opt_in_test, rejects_requested_spm_without_sdk_beta_env)
{
    EXPECT_FALSE(
        is_spm_enabled_for_sdk<fake_environment>(make_valid_requested_spm_request()));
}

TEST_F(beta_opt_in_test, accepts_requested_spm_with_sdk_beta_env)
{
    fake_env::setenv(sdk_spm_beta_env, "ON", 1);

    EXPECT_TRUE(
        is_spm_enabled_for_sdk<fake_environment>(make_valid_requested_spm_request()));
}
