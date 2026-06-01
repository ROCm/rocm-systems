/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "configuration.h"
#include "environment.h"
#include "hipfile-warnings.h"
#include "mhip.h"
#include "msys.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>
#include <string>

using namespace hipFile;
using namespace testing;
using namespace std;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileConfiguration : public Test {
    StrictMock<MSys> msys;
    StrictMock<MHip> mhip;

    void expect_configuration_fastpath(const char *hipfile_force_compat_mode,
                                       void       *hipAmdFileRead  = reinterpret_cast<void *>(0xDEADBEEF),
                                       void       *hipAmdFileWrite = reinterpret_cast<void *>(0xCAFEBABE))
    {
        EXPECT_CALL(msys, getenv(StrEq(hipFile::Environment::FORCE_COMPAT_MODE)))
            .WillOnce(Return(const_cast<char *>(hipfile_force_compat_mode)));
        EXPECT_CALL(mhip, hipRuntimeGetVersion).Times(2);
        EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileRead"), _, _, _))
            .WillOnce(Return(hipAmdFileRead));
        EXPECT_CALL(mhip, hipGetProcAddress(StrEq("hipAmdFileWrite"), _, _, _))
            .WillOnce(Return(hipAmdFileWrite));
    }

    void expect_configuration_fallback(const char *hipfile_allow_compat_mode)
    {
        EXPECT_CALL(msys, getenv(StrEq(hipFile::Environment::ALLOW_COMPAT_MODE)))
            .WillOnce(Return(const_cast<char *>(hipfile_allow_compat_mode)));
    }

    void expect_configuration_statslevel(const char *hipfile_stats_level)
    {
        EXPECT_CALL(msys, getenv(StrEq(hipFile::Environment::STATS_LEVEL)))
            .WillOnce(Return(const_cast<char *>(hipfile_stats_level)));
    }

    void expect_configuration_unsupported_file_systems(const char *hipfile_unsupported_file_systems)
    {
        EXPECT_CALL(msys, getenv(StrEq(hipFile::Environment::UNSUPPORTED_FILE_SYSTEMS)))
            .WillOnce(Return(const_cast<char *>(hipfile_unsupported_file_systems)));
    }
};

TEST_F(HipFileConfiguration, FastpathEnabledIfForceCompatModeEnvironmentVariableIsNotSet)
{
    expect_configuration_fastpath(nullptr);
    ASSERT_TRUE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, FastpathEnabledIfForceCompatModeEnvironmentVariableIsInvalid)
{
    expect_configuration_fastpath("not-a-bool");
    ASSERT_TRUE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, FastpathEnabledIfForceCompatModeEnvironmentVariableIsFalse)
{
    expect_configuration_fastpath("false");
    ASSERT_TRUE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, OverrideEnabledFastpathBackend)
{
    Configuration config{};
    expect_configuration_fastpath("false");
    ASSERT_TRUE(config.fastpath());

    config.fastpath(false);
    ASSERT_FALSE(config.fastpath());
}

TEST_F(HipFileConfiguration, FastpathDisabledIfForceCompatModeEnvironmentVariableIsTrue)
{
    expect_configuration_fastpath("true");
    ASSERT_FALSE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, FastpathDisabledIfHipAmdFileReadIsNotFound)
{
    expect_configuration_fastpath(nullptr, nullptr);
    ASSERT_FALSE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, FastpathDisabledIfHipAmdFileWriteIsNotFound)
{
    expect_configuration_fastpath(nullptr, reinterpret_cast<void *>(0x1), nullptr);
    ASSERT_FALSE(Configuration().fastpath());
}

TEST_F(HipFileConfiguration, OverrideDisabledFastpathBackend)
{
    Configuration config{};
    expect_configuration_fastpath("true");
    ASSERT_FALSE(config.fastpath());

    config.fastpath(true);
    ASSERT_TRUE(config.fastpath());
}

TEST_F(HipFileConfiguration, CantOverrideDisabledFastpathBackendIfHipAmdFileReadIsNotAvailable)
{
    Configuration config{};
    expect_configuration_fastpath(nullptr, nullptr);
    ASSERT_FALSE(config.fastpath());

    config.fastpath(true);
    ASSERT_FALSE(config.fastpath());
}

TEST_F(HipFileConfiguration, CantOverrideDisabledFastpathBackendIfHipAmdFileWriteIsNotAvailable)
{
    Configuration config{};
    expect_configuration_fastpath(nullptr, reinterpret_cast<void *>(0x1), nullptr);
    ASSERT_FALSE(config.fastpath());

    config.fastpath(true);
    ASSERT_FALSE(config.fastpath());
}

TEST_F(HipFileConfiguration, FallbackEnabledIfAllowCompatModeEnvironmentVariableIsNotSet)
{
    expect_configuration_fallback(nullptr);
    ASSERT_TRUE(Configuration().fallback());
}

TEST_F(HipFileConfiguration, FallbackEnabledIfAllowCompatModeEnvironmentVariableIsInvalid)
{
    expect_configuration_fallback("not-a-bool");
    ASSERT_TRUE(Configuration().fallback());
}

TEST_F(HipFileConfiguration, FallbackEnabledIfAllowCompatModeEnvironmentVariableIsTrue)
{
    expect_configuration_fallback("true");
    ASSERT_TRUE(Configuration().fallback());
}

TEST_F(HipFileConfiguration, OverrideEnabledFallbackBackend)
{
    Configuration config{};
    expect_configuration_fallback(nullptr);
    ASSERT_TRUE(config.fallback());

    config.fallback(false);
    ASSERT_FALSE(config.fallback());
}

TEST_F(HipFileConfiguration, FallbackDisabledIfAllowCompatModeEnvironmentVariableIsFalse)
{
    expect_configuration_fallback("false");
    ASSERT_FALSE(Configuration().fallback());
}

TEST_F(HipFileConfiguration, OverrideDisabledFallbackBackend)
{
    Configuration config{};
    expect_configuration_fallback("false");
    ASSERT_FALSE(config.fallback());

    config.fallback(true);
    ASSERT_TRUE(config.fallback());
}

TEST_F(HipFileConfiguration, StatsLevelEnvironmentVariableIsNotSet)
{
    expect_configuration_statslevel(nullptr);
    ASSERT_EQ(1, Configuration().statsLevel());
}

TEST_F(HipFileConfiguration, StatsLevelEnvironmentVariableIsInvalid)
{
    expect_configuration_statslevel("not-a-number");
    ASSERT_EQ(1, Configuration().statsLevel());
}

TEST_F(HipFileConfiguration, StatsLevelEnvironmentVariableIsSet)
{
    expect_configuration_statslevel("2");
    ASSERT_EQ(2, Configuration().statsLevel());
}

TEST_F(HipFileConfiguration, UnsupportedFilesystemsDisabledIfEnvironmentVariableIsNotSet)
{
    expect_configuration_unsupported_file_systems(nullptr);
    ASSERT_FALSE(Configuration().unsupportedFileSystems());
}

TEST_F(HipFileConfiguration, UnsupportedFilesystemsDisabledIfEnvironmentVariableIsInvalid)
{
    expect_configuration_unsupported_file_systems("not-a-bool");
    ASSERT_FALSE(Configuration().unsupportedFileSystems());
}

TEST_F(HipFileConfiguration, UnsupportedFilesystemsEnabledIfEnvironmentVariableIsTrue)
{
    expect_configuration_unsupported_file_systems("true");
    ASSERT_TRUE(Configuration().unsupportedFileSystems());
}

TEST_F(HipFileConfiguration, UnsupportedFilesystemsDisabledIfEnvironmentVariableIsFalse)
{
    expect_configuration_unsupported_file_systems("false");
    ASSERT_FALSE(Configuration().unsupportedFileSystems());
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
