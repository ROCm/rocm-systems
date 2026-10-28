/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mconfiguration.h"
#include "mhip.h"
#include "mmountinfo.h"
#include "msys.h"
#include "state.h"
#include "stream.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <thread>
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

// A registered stream checks its priority, then allocates one signal slot under a DeviceGuard and frees it
// when released. Streams default to normal priority. Point the mocked allocation at real storage so the
// ctor's zero-init write is valid.
static void
expectSignalMemory(StrictMock<MHip> &mhip, uint64_t *storage)
{
    using ::testing::AnyNumber;
    using ::testing::Return;
    EXPECT_CALL(mhip, hipDeviceGetStreamPriorityRange)
        .Times(AnyNumber())
        .WillRepeatedly(Return(HipStreamPriorityRange{1, -1}));
    EXPECT_CALL(mhip, hipStreamGetPriority).Times(AnyNumber()).WillRepeatedly(Return(0));
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
    auto stream = stream_map.findStream(nonnull_stream);
    ASSERT_EQ(stream->getHipStream(), nonnull_stream);
}

INSTANTIATE_TEST_SUITE_P(StreamSuite, HipFileStreamValidParams, hipFileFlagsPowerSet());

TEST_F(HipFileStream, find_stream_returns_registered_stream)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    ASSERT_NE(stream, nullptr);
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
    EXPECT_EQ(stream_map.findStream(nullptr), nullptr);
    EXPECT_EQ(stream_map.findStream(hipStreamPerThread), nullptr);
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
        stream = stream_map.findStream(reinterpret_cast<hipStream_t>(1));
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

TEST_F(HipFileStreamExternal, register_stream_at_reserved_priority_returns_invalid_value)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillOnce(::testing::Return(-1));
    ASSERT_EQ(hipFileStreamRegister(nonnull_stream, 0), HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStream, registered_stream_has_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    EXPECT_TRUE(stream->canUseStreamWaitValue());
    EXPECT_NE(stream->signalSlot(), nullptr);
    EXPECT_FALSE(stream->isRetired());
}

TEST_F(HipFileStream, default_streams_have_no_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice).Times(2);
    EXPECT_CALL(mhip, hipStreamGetPriority).Times(0);
    EXPECT_CALL(mhip, hipDeviceGetAttribute).Times(0);
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    for (auto hip_stream : {static_cast<hipStream_t>(nullptr), hipStreamPerThread}) {
        auto stream = StreamMap::makeDefaultStream(hip_stream);
        EXPECT_EQ(stream->getHipStream(), hip_stream);
        EXPECT_FALSE(stream->canUseStreamWaitValue());
        EXPECT_EQ(stream->signalSlot(), nullptr);
    }
}

TEST_F(HipFileStream, next_and_release_signal_target)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    EXPECT_EQ(stream->nextSignalTarget(), 1u);
    EXPECT_EQ(stream->nextSignalTarget(), 2u);
    EXPECT_EQ(stream->releaseSignalTarget(), 1u);
}

TEST_F(HipFileStream, drain_and_release_retires_syncs_and_frees)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    stream->drainAndRelease();
    EXPECT_TRUE(stream->isRetired());
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

TEST_F(HipFileStream, drain_and_release_waits_for_reserved_targets)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    {
        // A reserved target that no stream wait covers, like an op whose wait failed to arm.
        auto lock = stream->getLock();
        stream->nextSignalTarget();
    }

    std::atomic<bool> drained{false};
    std::thread       drainer{[&] {
        stream->drainAndRelease();
        drained = true;
    }};
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(drained);
    std::atomic_ref<uint64_t>{signal_storage}.fetch_add(1, std::memory_order_release);
    drainer.join();
    EXPECT_TRUE(drained);
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

TEST_F(HipFileStream, registered_stream_without_wait_value_support_has_no_signal_slot)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(::testing::Return(0));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    ASSERT_NE(stream, nullptr);
    EXPECT_FALSE(stream->canUseStreamWaitValue());
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

TEST_F(HipFileStream, apply_flags_updates_and_validates_flags)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    stream->applyFlags(HIPFILE_STREAM_FIXED_FILE_SIZE);
    EXPECT_TRUE(stream->fixedIOSize());
    EXPECT_FALSE(stream->fixedFileOffset());
    EXPECT_THROW(stream->applyFlags(HIPFILE_STREAM_FLAGS_MASK + 1), std::invalid_argument);
    EXPECT_TRUE(stream->fixedIOSize());
}

TEST_F(HipFileStream, find_stream_returns_only_registered_streams)
{
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    EXPECT_NE(stream_map.findStream(nonnull_stream), nullptr);
}

TEST_F(HipFileStream, adopt_registers_implicit_stream)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    auto candidate = StreamMap::makeImplicitStream(nonnull_stream);
    EXPECT_NE(candidate->signalSlot(), nullptr);
    EXPECT_EQ(stream_map.adoptImplicitStream(nonnull_stream, candidate), candidate);
    EXPECT_EQ(stream_map.findStream(nonnull_stream), candidate);
}

TEST_F(HipFileStream, adopt_keeps_existing_registration)
{
    EXPECT_CALL(mhip, hipStreamGetDevice).Times(2);
    stream_map.registerStream(nonnull_stream, 0);
    auto registered = stream_map.findStream(nonnull_stream);
    auto candidate  = StreamMap::makeImplicitStream(nonnull_stream);
    EXPECT_EQ(stream_map.adoptImplicitStream(nonnull_stream, candidate), registered);
}

TEST_F(HipFileStream, register_after_implicit_registration_applies_flags)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    auto stream =
        stream_map.adoptImplicitStream(nonnull_stream, StreamMap::makeImplicitStream(nonnull_stream));
    EXPECT_FALSE(stream->fixedIOSize());
    stream_map.registerStream(nonnull_stream, HIPFILE_STREAM_FIXED_FILE_SIZE);
    EXPECT_TRUE(stream->fixedIOSize());
    EXPECT_EQ(stream_map.findStream(nonnull_stream), stream);
    EXPECT_THROW(stream_map.registerStream(nonnull_stream, 0), std::invalid_argument);
}

TEST_F(HipFileStream, register_after_implicit_registration_rejects_invalid_flags)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.adoptImplicitStream(nonnull_stream, StreamMap::makeImplicitStream(nonnull_stream));
    EXPECT_THROW(stream_map.registerStream(nonnull_stream, HIPFILE_STREAM_FLAGS_MASK + 1),
                 std::invalid_argument);
    // Still implicitly registered, so a valid explicit registration succeeds.
    stream_map.registerStream(nonnull_stream, 0);
}

TEST_F(HipFileStream, register_stream_at_reserved_priority_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillOnce(::testing::Return(-1));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_THROW(stream_map.registerStream(nonnull_stream, 0), ReservedStreamPriority);
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
}

TEST_F(HipFileStream, implicit_stream_at_reserved_priority_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillOnce(::testing::Return(-1));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_THROW(StreamMap::makeImplicitStream(nonnull_stream), ReservedStreamPriority);
}

TEST_F(HipFileStream, reserved_priority_is_not_enforced_with_one_priority_level)
{
    // With a single level, bounce streams can't be separated by priority, so rejecting would reject
    // everything.
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetStreamPriorityRange)
        .WillOnce(::testing::Return(HipStreamPriorityRange{0, 0}));
    stream_map.registerStream(nonnull_stream, 0);
    EXPECT_NE(stream_map.findStream(nonnull_stream), nullptr);
}

// Implicit registration through the real DriverState, as hipFileReadAsync/hipFileWriteAsync use it.
struct HipFileStreamImplicit : public HipFileOpened {
    HipFileStreamImplicit()
    {
        ON_CALL(mhip, hipDeviceGetStreamPriorityRange)
            .WillByDefault(::testing::Return(HipStreamPriorityRange{1, -1}));
        ON_CALL(mhip, hipStreamGetPriority).WillByDefault(::testing::Return(0));
        ON_CALL(mhip, hipDeviceGetAttribute).WillByDefault(::testing::Return(1));
        ON_CALL(mhip, hipExtMallocWithFlags)
            .WillByDefault(::testing::Return(static_cast<void *>(&signal_storage)));

        hipFileDescr_t descr{};
        descr.handle.fd = 1234;
        descr.type      = hipFileHandleTypeOpaqueFD;
        expect_file_registration(msys, mlibmounthelper);
        EXPECT_EQ(hipFileHandleRegister(&fh, &descr), HIPFILE_SUCCESS);
        expect_buffer_registration(mhip, hipMemoryTypeDevice);
        EXPECT_EQ(hipFileBufRegister(buf, 4096, 0), HIPFILE_SUCCESS);
    }

    ~HipFileStreamImplicit() override
    {
        // Release everything while the mocks are still installed.
        (void)hipFileStreamDeregister(nonnull_stream);
        (void)hipFileBufDeregister(buf);
        hipFileHandleDeregister(fh);
    }

    file_buffer_stream_tuple lookup(hipStream_t hip_stream)
    {
        return Context<DriverState>::get()->getFileBufferAndStream(fh, buf, hip_stream);
    }

    ::testing::NiceMock<MHip>   mhip;
    StrictMock<MSys>            msys;
    StrictMock<MLibMountHelper> mlibmounthelper;
    uint64_t                    signal_storage{};
    hipFileHandle_t             fh{};
    void                       *buf{reinterpret_cast<void *>(0x1)};
    hipStream_t                 nonnull_stream{reinterpret_cast<hipStream_t>(1)};
};

TEST_F(HipFileStreamImplicit, first_async_lookup_registers_stream_implicitly)
{
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(1);
    auto first  = lookup(nonnull_stream);
    auto second = lookup(nonnull_stream);
    ASSERT_NE(first.stream, nullptr);
    EXPECT_NE(first.stream->signalSlot(), nullptr);
    EXPECT_EQ(first.stream, second.stream);
}

TEST_F(HipFileStreamImplicit, default_streams_are_not_registered_implicitly)
{
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_EQ(lookup(nullptr).stream->signalSlot(), nullptr);
    EXPECT_EQ(lookup(hipStreamPerThread).stream->signalSlot(), nullptr);
}

TEST_F(HipFileStreamImplicit, register_after_implicit_registration_succeeds)
{
    auto implicit = lookup(nonnull_stream).stream;
    EXPECT_EQ(hipFileStreamRegister(nonnull_stream, HIPFILE_STREAM_FIXED_FILE_SIZE), HIPFILE_SUCCESS);
    EXPECT_TRUE(implicit->fixedIOSize());
    EXPECT_EQ(lookup(nonnull_stream).stream, implicit);
    EXPECT_EQ(hipFileStreamRegister(nonnull_stream, 0), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamImplicit, implicitly_registered_stream_deregisters)
{
    (void)lookup(nonnull_stream);
    EXPECT_CALL(mhip, hipStreamSynchronize(nonnull_stream));
    EXPECT_EQ(hipFileStreamDeregister(nonnull_stream), HIPFILE_SUCCESS);
    EXPECT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
