/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mconfiguration.h"
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

// A managed (registered) stream allocates one signal slot under a DeviceGuard and frees it when
// released. Point the mocked allocation at real storage so the ctor's zero-init write is valid.
static void
expectSignalMemory(StrictMock<MHip> &mhip, uint64_t *storage)
{
    using ::testing::AnyNumber;
    using ::testing::Return;
    EXPECT_CALL(mhip, hipDeviceGetAttribute).Times(AnyNumber()).WillRepeatedly(Return(1));
    EXPECT_CALL(mhip, hipGetDevice).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(mhip, hipExtMallocWithFlags)
        .Times(AnyNumber())
        .WillRepeatedly(Return(static_cast<void *>(storage)));
    EXPECT_CALL(mhip, hipFree).Times(AnyNumber());
}

struct HipFileStream : public ::testing::Test {
    HipFileStream()
    {
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
        expectSignalMemory(mhip, &signal_storage);
    }
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    hipStream_t                nonnull_stream;
    uint64_t                   signal_storage{};
    StreamMap                  stream_map;
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

TEST_F(HipFileStream, register_default_stream_is_noop)
{
    stream_map.registerStream(nullptr, 0);
    stream_map.registerStream(hipStreamPerThread, 0);
    EXPECT_CALL(mhip, hipStreamGetDevice).Times(2);
    auto null_stream = stream_map.getStream(nullptr);
    auto per_thread  = stream_map.getStream(hipStreamPerThread);
    EXPECT_FALSE(null_stream->isManaged());
    EXPECT_FALSE(per_thread->isManaged());
}

TEST_F(HipFileStream, deregister_default_stream_is_noop)
{
    EXPECT_EQ(stream_map.deregisterStream(nullptr), nullptr);
    EXPECT_EQ(stream_map.deregisterStream(hipStreamPerThread), nullptr);
}

TEST_F(HipFileStream, deregister_returns_removed_stream)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto removed = stream_map.deregisterStream(nonnull_stream);
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->getHipStream(), nonnull_stream);
    EXPECT_THROW(stream_map.deregisterStream(nonnull_stream), std::invalid_argument);
}

TEST(HipFileStreamDestructor, destructor_with_streams_in_use_logs)
{
    StrictMock<MHip>         mhip;
    StrictMock<MSys>         msys;
    uint64_t                 signal_storage{};
    std::shared_ptr<IStream> stream;
    expectSignalMemory(mhip, &signal_storage);
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
        expectSignalMemory(mhip, &signal_storage);
    }
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    hipStream_t                nonnull_stream;
    uint64_t                   signal_storage{};
};

TEST_F(HipFileStreamExternal, register_and_deregister_with_valid_stream_works)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    ASSERT_EQ(hipFileStreamRegister(nonnull_stream, 0), HIPFILE_SUCCESS);
    // Deregistering a registered stream drains its in-flight IO before freeing its signal slot.
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    ASSERT_EQ(hipFileStreamDeregister(nonnull_stream), HIPFILE_SUCCESS);
}

TEST_F(HipFileStreamExternal, deregister_unregistered_stream_returns_error)
{
    ASSERT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamExternal, register_and_deregister_default_stream_succeed)
{
    ASSERT_EQ(hipFileStreamRegister(nullptr, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(nullptr), HIPFILE_SUCCESS);
}

TEST_F(HipFileStream, registered_stream_is_managed_with_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    EXPECT_TRUE(stream->isManaged());
    EXPECT_TRUE(stream->canUseStreamWaitValue());
    EXPECT_NE(stream->signalSlot(), nullptr);
    EXPECT_FALSE(stream->isRetired());
}

TEST_F(HipFileStream, unregistered_stream_is_unmanaged_without_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    auto stream = stream_map.getStream(nonnull_stream);
    EXPECT_FALSE(stream->isManaged());
    EXPECT_FALSE(stream->canUseStreamWaitValue());
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

TEST_F(HipFileStream, next_and_release_signal_target)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    EXPECT_EQ(stream->nextSignalTarget(), 1u);
    EXPECT_EQ(stream->nextSignalTarget(), 2u);
    EXPECT_EQ(stream->releaseSignalTarget(), 1u);
}

TEST_F(HipFileStream, drain_and_release_retires_syncs_and_frees)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    stream->drainAndRelease();
    EXPECT_TRUE(stream->isRetired());
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
