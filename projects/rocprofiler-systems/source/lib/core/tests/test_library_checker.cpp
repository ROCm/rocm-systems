// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/version.hpp"
#include "core/library_checker.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

namespace rocprofsys::library_checker
{
namespace
{

using ::testing::HasSubstr;

struct fake_sdk_backend
{
    static constexpr std::uint32_t compile_time_version = 10305U;  // 1.3.5

    static inline version s_loaded = {};

    static int get_version(std::uint32_t* major, std::uint32_t* minor,
                           std::uint32_t* patch)
    {
        *major = s_loaded.major;
        *minor = s_loaded.minor;
        *patch = s_loaded.patch;
        return 0;
    }
};

constexpr auto k_built = version{ .major = 1, .minor = 2, .patch = 3 };

TEST(library_checker_test, check_version_accepts_equal_versions)
{
    EXPECT_NO_THROW(check_version("lib", k_built, k_built));
}

TEST(library_checker_test, check_version_throws_on_any_component_difference)
{
    EXPECT_THROW(check_version("lib", k_built, { .major = 2, .minor = 2, .patch = 3 }),
                 std::runtime_error);
    EXPECT_THROW(check_version("lib", k_built, { .major = 1, .minor = 3, .patch = 3 }),
                 std::runtime_error);
    EXPECT_THROW(check_version("lib", k_built, { .major = 1, .minor = 2, .patch = 4 }),
                 std::runtime_error);
}

TEST(library_checker_test, check_version_message_names_library_and_both_versions)
{
    try
    {
        check_version("profiler-hub", { .major = 0, .minor = 2, .patch = 0 },
                      { .major = 0, .minor = 3, .patch = 1 });
        FAIL() << "expected std::runtime_error";
    } catch(const std::runtime_error& e)
    {
        EXPECT_THAT(
            e.what(),
            HasSubstr("profiler-hub version mismatch: built against v0.2.0 but v0.3.1 "
                      "is loaded"));
        EXPECT_THAT(e.what(), HasSubstr("LD_LIBRARY_PATH"));
    }
}

TEST(library_checker_test, rocprofiler_sdk_accepts_matching_loaded_version)
{
    fake_sdk_backend::s_loaded = { .major = 1, .minor = 3, .patch = 5 };

    EXPECT_NO_THROW(check_rocprofiler_sdk_version_mismatch<fake_sdk_backend>());
}

TEST(library_checker_test, rocprofiler_sdk_throws_on_mismatched_loaded_version)
{
    fake_sdk_backend::s_loaded = { .major = 1, .minor = 4, .patch = 0 };

    try
    {
        check_rocprofiler_sdk_version_mismatch<fake_sdk_backend>();
        FAIL() << "expected std::runtime_error";
    } catch(const std::runtime_error& e)
    {
        EXPECT_THAT(e.what(), HasSubstr("rocprofiler-sdk version mismatch: built against "
                                        "v1.3.5 but v1.4.0 is loaded"));
    }
}

}  // namespace
}  // namespace rocprofsys::library_checker
