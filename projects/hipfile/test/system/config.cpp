/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace std;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// Test hipFile APIs that set/get configuration values
struct HipFileConfig : public ::testing::Test {

    void SetUp() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
    }

    // Ensure the driver is deinitialized/closed after the test
    void TearDown() override
    {
        // Ensure driver is opened so staged values are applied then cleared
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);

        // Ensure driver is closed to clear applied values
        while (hipFileUseCount()) {
            ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        }
    }
};

TEST_F(HipFileConfig, SizeTParameterCantSetIfDriverOpen)
{
    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    auto set_after_open = hipFileSetParameterSizeT(hipFileParamExecutionMaxIOThreads, 10);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileDriverAlreadyOpen));
#endif
}

TEST_F(HipFileConfig, SizeTParameterGetDefault)
{
    size_t default_;
    auto   get_default = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &default_);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, 0);
#endif
}

TEST_F(HipFileConfig, SizeTParameterSetStagesValue)
{
    size_t default_;
    auto   get_default = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &default_);

    size_t target{10};
    auto   stage_target = hipFileSetParameterSizeT(hipFileParamExecutionMaxIOThreads, target);

    size_t staged;
    auto   get_staged = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &staged);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(get_staged, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(get_staged, HIPFILE_SUCCESS);
    ASSERT_EQ(target, staged);
#endif
}

TEST_F(HipFileConfig, SizeTParameterOpenAppliesValue)
{
    size_t default_;
    auto   get_default = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &default_);

    size_t target{10};
    auto   stage_target = hipFileSetParameterSizeT(hipFileParamExecutionMaxIOThreads, target);

    auto driver_open = hipFileDriverOpen();

    size_t applied;
    auto   get_applied = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &applied);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HIPFILE_SUCCESS);
    ASSERT_EQ(target, applied);
#endif
}

TEST_F(HipFileConfig, SizeTParameterCloseClearsAppliedValue)
{
    size_t default_;
    auto   get_default = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &default_);

    size_t target{10};
    auto   stage_target = hipFileSetParameterSizeT(hipFileParamExecutionMaxIOThreads, target);

    // Open applies staged value and clears staged
    auto driver_open = hipFileDriverOpen();

    // Close clears applied value
    auto driver_close = hipFileDriverClose();

    size_t cleared;
    auto   get_cleared = hipFileGetParameterSizeT(hipFileParamExecutionMaxIOThreads, &cleared);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, cleared);
#endif
}

TEST_F(HipFileConfig, BoolParameterCantSetIfDriverOpen)
{
    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    auto set_after_open = hipFileSetParameterBool(hipFileParamUsePcip2pdma, true);
#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileDriverAlreadyOpen));
#endif
}

TEST_F(HipFileConfig, BoolParameterGetDefault)
{
    bool default_;
    auto get_default = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &default_);
#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, false);
#endif
}

TEST_F(HipFileConfig, BoolParameterSetStagesValue)
{
    bool default_;
    auto get_default = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &default_);

    constexpr bool target{true};
    auto           stage_target = hipFileSetParameterBool(hipFileParamUsePcip2pdma, target);

    bool staged;
    auto get_staged = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &staged);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(get_staged, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(get_staged, HIPFILE_SUCCESS);
    ASSERT_EQ(target, staged);
#endif
}

TEST_F(HipFileConfig, BoolParameterOpenAppliesValue)
{
    bool default_;
    auto get_default = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &default_);

    constexpr bool target{true};
    auto           stage_target = hipFileSetParameterBool(hipFileParamUsePcip2pdma, target);

    auto driver_open = hipFileDriverOpen();

    bool applied;
    auto get_applied = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &applied);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HIPFILE_SUCCESS);
    ASSERT_EQ(target, applied);
#endif
}

TEST_F(HipFileConfig, BoolParameterCloseClearsAppliedValue)
{
    bool default_;
    auto get_default = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &default_);

    constexpr bool target{true};
    auto           stage_target = hipFileSetParameterBool(hipFileParamUsePcip2pdma, target);

    // Open applies staged value and clears staged
    auto driver_open = hipFileDriverOpen();

    // Close clears applied value
    auto driver_close = hipFileDriverClose();

    bool cleared;
    auto get_cleared = hipFileGetParameterBool(hipFileParamUsePcip2pdma, &cleared);

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, cleared);
#endif
}

TEST_F(HipFileConfig, StringParameterCantSetIfDriverOpen)
{
    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    auto set_after_open = hipFileSetParameterString(hipFileParamLogDir, "/foo/bar");
#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(set_after_open, HipFileOpError(hipFileDriverAlreadyOpen));
#endif
}

TEST_F(HipFileConfig, StringParameterGetDefault)
{
    vector<char> buffer(64);

    auto get_default =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string default_{buffer.data()};

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, "");
#endif
}

TEST_F(HipFileConfig, StringParameterSetStagesValue)
{
    vector<char> buffer(64);

    auto get_default =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string default_{buffer.data()};

    const string target{"/tmp"};
    auto         stage_target = hipFileSetParameterString(hipFileParamLogDir, target.c_str());

    auto get_staged =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string staged{buffer.data()};

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(get_staged, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(get_staged, HIPFILE_SUCCESS);
    ASSERT_EQ(target, staged);
#endif
}

TEST_F(HipFileConfig, StringParameterOpenAppliesValue)
{
    vector<char> buffer(64);

    auto get_default =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string default_{buffer.data()};

    // The target value must specify a directory that exists otherwise it will not be applied when the driver
    // is opened.
    const string target{"/tmp"};
    auto         stage_target = hipFileSetParameterString(hipFileParamLogDir, target.c_str());

    auto driver_open = hipFileDriverOpen();

    auto get_applied =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string applied{buffer.data()};

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(get_applied, HIPFILE_SUCCESS);
    ASSERT_EQ(target, applied);
#endif
}

TEST_F(HipFileConfig, StringParameterCloseClearsAppliedValue)
{
    vector<char> buffer(64);

    auto get_default =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string default_{buffer.data()};

    // The target value must specify a directory that exists otherwise it will not be applied when the driver
    // is opened.
    const string target{"/tmp"};
    auto         stage_target = hipFileSetParameterString(hipFileParamLogDir, target.c_str());

    auto driver_open  = hipFileDriverOpen();
    auto driver_close = hipFileDriverClose();

    auto get_cleared =
        hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size()));
    string cleared{buffer.data()};

#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(get_default, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(stage_target, HipFileOpError(hipFileInternalError));
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(get_default, HIPFILE_SUCCESS);
    ASSERT_NE(default_, target);
    ASSERT_EQ(stage_target, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_open, HIPFILE_SUCCESS);
    ASSERT_EQ(driver_close, HIPFILE_SUCCESS);
    ASSERT_EQ(get_cleared, HIPFILE_SUCCESS);
    ASSERT_EQ(default_, cleared);
#endif
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
