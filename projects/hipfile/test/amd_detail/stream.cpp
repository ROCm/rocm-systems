/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mhip.h"
#include "msys.h"
#include "stream.h"

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <tuple>

using namespace hipFile;

using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

static auto
hipFileFlagsPowerSet()
{
    return ::testing::Combine(::testing::Values(0, HIPFILE_STREAM_FIXED_BUF_OFFSET),
                              ::testing::Values(0, HIPFILE_STREAM_FIXED_FILE_OFFSET),
                              ::testing::Values(0, HIPFILE_STREAM_FIXED_FILE_SIZE),
                              ::testing::Values(0, HIPFILE_STREAM_PAGE_ALIGNED_INPUTS));
}

struct HipFileStream : public ::testing::Test {
    HipFileStream()
    {
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
    }
    StrictMock<MHip> mhip;
    StrictMock<MSys> msys;
    hipStream_t      nonnull_stream;
    StreamMap        stream_map;
};

struct HipFileStreamValidParams
    : public HipFileStream,
      public ::testing::WithParamInterface<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {
protected:
    HipFileStreamValidParams() : HipFileStream{}
    {
        auto params = GetParam();
        flags       = std::get<0>(params) | std::get<1>(params) | std::get<2>(params) | std::get<3>(params);
    }
    unsigned flags;
};

TEST_P(HipFileStreamValidParams, register_stream_valid_flags_internal)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, flags);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_EQ(stream->getHipStream(), nonnull_stream);
}

INSTANTIATE_TEST_SUITE_P(StreamSuite, HipFileStreamValidParams, hipFileFlagsPowerSet());

TEST_F(HipFileStream, get_stream_with_unregistered_stream_works)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_EQ(nonnull_stream, stream->getHipStream());
}

TEST_F(HipFileStream, register_with_invalid_flags_throws)
{
    ASSERT_THROW(stream_map.registerStream(nonnull_stream, HIPFILE_STREAM_FLAGS_MASK + 1),
                 std::invalid_argument);
}

TEST_F(HipFileStream, register_twice_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    ASSERT_THROW(stream_map.registerStream(nonnull_stream, 0), std::invalid_argument);
}

TEST_F(HipFileStream, deregister_with_registered_stream_works)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    stream_map.deregisterStream(nonnull_stream);
}

TEST_F(HipFileStream, deregister_twice_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    stream_map.deregisterStream(nonnull_stream);
    ASSERT_THROW(stream_map.deregisterStream(nonnull_stream), std::invalid_argument);
}

TEST_F(HipFileStream, deregister_with_unregistered_stream_throws)
{
    ASSERT_THROW(stream_map.deregisterStream(nonnull_stream), std::invalid_argument);
}

TEST(HipFileStreamDestructor, destructor_with_streams_in_use_logs)
{
    StrictMock<MHip>         mhip;
    StrictMock<MSys>         msys;
    std::shared_ptr<IStream> stream;
    {
        StreamMap stream_map;
        EXPECT_CALL(mhip, hipStreamGetDevice);
        stream_map.registerStream(reinterpret_cast<hipStream_t>(1), 0);
        EXPECT_CALL(msys, syslog);
        stream = stream_map.getStream(reinterpret_cast<hipStream_t>(1));
    }
}

struct HipFileStreamExternal : public HipFileOpened {
    HipFileStreamExternal()
    {
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
    }
    StrictMock<MHip> mhip;
    StrictMock<MSys> msys;
    hipStream_t      nonnull_stream;
};

TEST_F(HipFileStreamExternal, register_and_deregister_with_valid_stream_works)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    ASSERT_EQ(hipFileStreamRegister(nonnull_stream, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(nonnull_stream), HIPFILE_SUCCESS);
}

TEST_F(HipFileStreamExternal, deregister_exception_returns_error)
{
    ASSERT_EQ(hipFileStreamDeregister(nullptr), HipFileOpError(hipFileInvalidValue));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
