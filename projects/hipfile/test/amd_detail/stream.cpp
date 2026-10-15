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

static uint64_t g_stream_signal_slot_storage = 0;

static void
expectStreamResources(StrictMock<MHip> &mhip)
{
    EXPECT_CALL(mhip, hipStreamCreateWithFlags)
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Return(reinterpret_cast<hipStream_t>(0xC0C0)));
    EXPECT_CALL(mhip, hipStreamDestroy).Times(::testing::AnyNumber());
    EXPECT_CALL(mhip, hipExtMallocWithFlags)
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Return(&g_stream_signal_slot_storage));
    EXPECT_CALL(mhip, hipFree).Times(::testing::AnyNumber());
    EXPECT_CALL(mhip, hipGetDevice).Times(::testing::AnyNumber()).WillRepeatedly(::testing::Return(0));
    EXPECT_CALL(mhip, hipSetDevice).Times(::testing::AnyNumber());
    EXPECT_CALL(mhip, hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, ::testing::_))
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Return(1));
}

static void
expectStreamBuffer(StrictMock<MConfiguration> &mconfig, StrictMock<MHip> &mhip)
{
    (void)mconfig;
    expectStreamResources(mhip);
}

struct HipFileStream : public ::testing::Test {
    HipFileStream()
    {
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
        expectStreamBuffer(mconfig, mhip);
    }
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    hipStream_t                nonnull_stream;
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

TEST_F(HipFileStream, unregistered_stream_is_unmanaged)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_FALSE(stream->isManaged());
    ASSERT_EQ(stream->copyStream(), nullptr);
    ASSERT_EQ(stream->signalSlot(), nullptr);
    ASSERT_FALSE(stream->canUseStreamWaitValue());
}

TEST_F(HipFileStream, registered_stream_is_managed)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    ASSERT_TRUE(stream_map.getStream(nonnull_stream)->isManaged());
}

TEST_F(HipFileStream, canUseStreamWaitValue_reflects_device_attribute_supported)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, ::testing::_))
        .WillRepeatedly(::testing::Return(1));
    stream_map.registerStream(nonnull_stream, 0);
    ASSERT_TRUE(stream_map.getStream(nonnull_stream)->canUseStreamWaitValue());
}

TEST_F(HipFileStream, canUseStreamWaitValue_reflects_device_attribute_unsupported)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    stream_map.registerStream(nonnull_stream, 0);
    ASSERT_FALSE(stream_map.getStream(nonnull_stream)->canUseStreamWaitValue());
}

TEST_F(HipFileStream, owns_copy_stream_and_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_EQ(stream->copyStream(), reinterpret_cast<hipStream_t>(0xC0C0));
    ASSERT_EQ(stream->signalSlot(), &g_stream_signal_slot_storage);
}

TEST_F(HipFileStream, creates_resources_on_the_stream_device)
{
    EXPECT_CALL(mhip, hipStreamGetDevice).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mhip, hipGetDevice).WillRepeatedly(::testing::Return(0));

    {
        ::testing::InSequence seq;
        EXPECT_CALL(mhip, hipSetDevice(1));
        EXPECT_CALL(mhip, hipStreamCreateWithFlags)
            .WillOnce(::testing::Return(reinterpret_cast<hipStream_t>(0xC0C0)));
        EXPECT_CALL(mhip, hipExtMallocWithFlags).WillOnce(::testing::Return(&g_stream_signal_slot_storage));
        EXPECT_CALL(mhip, hipSetDevice(0));
    }

    stream_map.registerStream(nonnull_stream, 0);
    ::testing::Mock::VerifyAndClearExpectations(&mhip);

    expectStreamResources(mhip);
    EXPECT_CALL(mhip, hipStreamGetDevice).WillRepeatedly(::testing::Return(1));
    EXPECT_CALL(mhip, hipGetDevice).WillRepeatedly(::testing::Return(0));
}

TEST_F(HipFileStream, destroys_copy_stream_when_signal_alloc_throws)
{
    auto handle = reinterpret_cast<hipStream_t>(0xC0C0);
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamCreateWithFlags).WillOnce(::testing::Return(handle));
    EXPECT_CALL(mhip, hipExtMallocWithFlags)
        .WillOnce(::testing::Throw(Hip::RuntimeError(hipErrorOutOfMemory)));
    EXPECT_CALL(mhip, hipStreamDestroy(handle));
    ASSERT_THROW(stream_map.registerStream(nonnull_stream, 0), Hip::RuntimeError);
}

TEST_F(HipFileStream, construction_failure_survives_throwing_stream_destroy)
{
    auto handle = reinterpret_cast<hipStream_t>(0xC0C0);
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamCreateWithFlags).WillOnce(::testing::Return(handle));
    EXPECT_CALL(mhip, hipExtMallocWithFlags)
        .WillOnce(::testing::Throw(Hip::RuntimeError(hipErrorOutOfMemory)));
    EXPECT_CALL(mhip, hipStreamDestroy(handle))
        .WillOnce(::testing::Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(msys, syslog);
    ASSERT_THROW(stream_map.registerStream(nonnull_stream, 0), Hip::RuntimeError);
}

TEST_F(HipFileStream, no_signal_slot_when_wait_value_unsupported)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, ::testing::_))
        .WillRepeatedly(::testing::Return(0));
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_EQ(stream->signalSlot(), nullptr);
    ASSERT_NE(stream->copyStream(), nullptr);
}

TEST_F(HipFileStream, next_signal_target_increments_monotonically)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.getStream(nonnull_stream);
    ASSERT_EQ(stream->nextSignalTarget(), 1u);
    ASSERT_EQ(stream->nextSignalTarget(), 2u);
    ASSERT_EQ(stream->nextSignalTarget(), 3u);
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
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    stream_map.deregisterStream(nonnull_stream);
}

TEST_F(HipFileStream, deregister_twice_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    stream_map.deregisterStream(nonnull_stream);
    ASSERT_THROW(stream_map.deregisterStream(nonnull_stream), std::invalid_argument);
}

TEST_F(HipFileStream, deregister_with_unregistered_stream_throws)
{
    ASSERT_THROW(stream_map.deregisterStream(nonnull_stream), std::invalid_argument);
}

TEST_F(HipFileStream, deregister_drains_the_stream_before_releasing_resources)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);

    ::testing::InSequence seq;
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    EXPECT_CALL(mhip, hipFree(&g_stream_signal_slot_storage));
    EXPECT_CALL(mhip, hipStreamDestroy(reinterpret_cast<hipStream_t>(0xC0C0)));
    stream_map.deregisterStream(nonnull_stream);
}

TEST_F(HipFileStream, register_null_stream_throws)
{
    ASSERT_THROW(stream_map.registerStream(nullptr, 0), std::invalid_argument);
}

TEST_F(HipFileStream, register_per_thread_stream_throws)
{
    ASSERT_THROW(stream_map.registerStream(hipStreamPerThread, 0), std::invalid_argument);
}

TEST(HipFileStreamDestructor, resource_free_failure_logs)
{
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    (void)mconfig;
    expectStreamResources(mhip);
    EXPECT_CALL(mhip, hipFree)
        .Times(::testing::AnyNumber())
        .WillRepeatedly(::testing::Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(mhip, hipStreamDestroy(reinterpret_cast<hipStream_t>(0xC0C0)));
    EXPECT_CALL(mhip, hipStreamSynchronize(reinterpret_cast<hipStream_t>(1)));
    EXPECT_CALL(msys, syslog);
    {
        StreamMap stream_map;
        EXPECT_CALL(mhip, hipStreamGetDevice);
        stream_map.registerStream(reinterpret_cast<hipStream_t>(1), 0);
        stream_map.deregisterStream(reinterpret_cast<hipStream_t>(1));
    }
}

TEST(HipFileStreamDestructor, destructor_with_streams_in_use_logs)
{
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    std::shared_ptr<IStream>   stream;
    expectStreamBuffer(mconfig, mhip);
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
        expectStreamBuffer(mconfig, mhip);
    }
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    hipStream_t                nonnull_stream;
};

TEST_F(HipFileStreamExternal, register_and_deregister_with_valid_stream_works)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    ASSERT_EQ(hipFileStreamRegister(nonnull_stream, 0), HIPFILE_SUCCESS);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    ASSERT_EQ(hipFileStreamDeregister(nonnull_stream), HIPFILE_SUCCESS);
}

TEST_F(HipFileStreamExternal, deregister_exception_returns_error)
{
    ASSERT_EQ(hipFileStreamDeregister(nullptr), HipFileOpError(hipFileInvalidValue));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
