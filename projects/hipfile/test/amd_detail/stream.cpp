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

// The Stream constructor asks Configuration whether to force the blocking async path for every
// non-default, non-transient stream. Default it to off so these tests cover the wait-value path.
template <class MockConfiguration>
static void
expectNoForceBlocking(MockConfiguration &mconfig)
{
    using ::testing::AnyNumber;
    using ::testing::Return;
    EXPECT_CALL(mconfig, forceBlockingAsync()).Times(AnyNumber()).WillRepeatedly(Return(false));
}

struct HipFileStream : public ::testing::Test {
    HipFileStream()
    {
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
        expectSignalMemory(mhip, &signal_storage);
        expectNoForceBlocking(mconfig);
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
    StrictMock<MConfiguration> mconfig;
    StrictMock<MHip>           mhip;
    StrictMock<MSys>           msys;
    uint64_t                   signal_storage{};
    std::shared_ptr<IStream>   stream;
    expectSignalMemory(mhip, &signal_storage);
    expectNoForceBlocking(mconfig);
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
        expectNoForceBlocking(mconfig);
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

TEST_F(HipFileStream, registered_stream_allocates_signal_slot_on_first_use)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    EXPECT_TRUE(stream->canUseStreamWaitValue());
    // Registration no longer allocates; the slot appears only on the first gated use.
    EXPECT_EQ(stream->signalSlot(), nullptr);
    EXPECT_FALSE(stream->isRetired());
    {
        auto lock = stream->getLock();
        stream->ensureSignalSlot();
    }
    auto *slot = stream->signalSlot();
    EXPECT_NE(slot, nullptr);
    // ensureSignalSlot allocates at most once: a second call keeps the same slot.
    {
        auto lock = stream->getLock();
        stream->ensureSignalSlot();
    }
    EXPECT_EQ(stream->signalSlot(), slot);
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
        // A reserved target that no stream wait covers, like an op whose wait failed to arm. Allocate
        // the slot first (as the first gated op would) so drain has something to spin on.
        auto lock = stream->getLock();
        stream->ensureSignalSlot();
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

TEST_F(HipFileStream, registered_stream_with_forced_blocking_has_no_signal_slot)
{
    EXPECT_CALL(mconfig, forceBlockingAsync()).WillOnce(::testing::Return(true));
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipDeviceGetAttribute).Times(0);
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    stream_map.registerStream(nonnull_stream, 0);
    auto stream = stream_map.findStream(nonnull_stream);
    ASSERT_NE(stream, nullptr);
    EXPECT_FALSE(stream->canUseStreamWaitValue());
    {
        auto lock = stream->getLock();
        stream->ensureSignalSlot();
    }
    EXPECT_EQ(stream->signalSlot(), nullptr);
}

TEST_F(HipFileStream, find_stream_returns_only_registered_streams)
{
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
    EXPECT_CALL(mhip, hipStreamGetDevice);
    stream_map.registerStream(nonnull_stream, 0);
    EXPECT_NE(stream_map.findStream(nonnull_stream), nullptr);
}

TEST_F(HipFileStream, make_transient_stream_is_blocking_and_unmapped)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    auto transient = StreamMap::makeTransientStream(nonnull_stream);
    ASSERT_NE(transient, nullptr);
    // A transient stream uses the blocking path (no wait-value gating, no slot) and is never mapped,
    // so nothing persistent is keyed to the handle.
    EXPECT_FALSE(transient->canUseStreamWaitValue());
    EXPECT_EQ(transient->signalSlot(), nullptr);
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
}

TEST_F(HipFileStream, register_after_transient_use_is_independent)
{
    EXPECT_CALL(mhip, hipStreamGetDevice).Times(2);
    // Using a stream transiently leaves it unregistered, so a later explicit registration starts fresh.
    (void)StreamMap::makeTransientStream(nonnull_stream);
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
    stream_map.registerStream(nonnull_stream, 0);
    EXPECT_NE(stream_map.findStream(nonnull_stream), nullptr);
    EXPECT_THROW(stream_map.registerStream(nonnull_stream, 0), std::invalid_argument);
}

TEST_F(HipFileStream, register_stream_at_reserved_priority_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillOnce(::testing::Return(-1));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_THROW(stream_map.registerStream(nonnull_stream, 0), ReservedStreamPriority);
    EXPECT_EQ(stream_map.findStream(nonnull_stream), nullptr);
}

TEST_F(HipFileStream, transient_stream_at_reserved_priority_throws)
{
    EXPECT_CALL(mhip, hipStreamGetDevice);
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillOnce(::testing::Return(-1));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_THROW(StreamMap::makeTransientStream(nonnull_stream), ReservedStreamPriority);
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

// Unregistered-stream async lookups through the real DriverState, as hipFileReadAsync/hipFileWriteAsync
// use them.
struct HipFileStreamImplicit : public HipFileOpened {
    HipFileStreamImplicit()
    {
        ON_CALL(mhip, hipDeviceGetStreamPriorityRange)
            .WillByDefault(::testing::Return(HipStreamPriorityRange{1, -1}));
        ON_CALL(mhip, hipStreamGetPriority).WillByDefault(::testing::Return(0));
        ON_CALL(mhip, hipDeviceGetAttribute).WillByDefault(::testing::Return(1));
        ON_CALL(mhip, hipExtMallocWithFlags)
            .WillByDefault(::testing::Return(static_cast<void *>(&signal_storage)));
        ON_CALL(mconfig, host()).WillByDefault(::testing::Return(true));
        expectNoForceBlocking(mconfig);

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

    ::testing::NiceMock<MConfiguration> mconfig;
    ::testing::NiceMock<MHip>           mhip;
    StrictMock<MSys>                    msys;
    StrictMock<MLibMountHelper>         mlibmounthelper;
    uint64_t                            signal_storage{};
    hipFileHandle_t                     fh{};
    void                               *buf{reinterpret_cast<void *>(0x1)};
    hipStream_t                         nonnull_stream{reinterpret_cast<hipStream_t>(1)};
};

TEST_F(HipFileStreamImplicit, first_async_lookup_builds_transient_stream)
{
    // An unregistered stream is used transiently: blocking path, no signal slot, never mapped. Each
    // lookup builds a fresh object, and nothing is left registered.
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    auto first  = lookup(nonnull_stream);
    auto second = lookup(nonnull_stream);
    ASSERT_NE(first.stream, nullptr);
    EXPECT_FALSE(first.stream->canUseStreamWaitValue());
    EXPECT_EQ(first.stream->signalSlot(), nullptr);
    EXPECT_NE(first.stream, second.stream);
    EXPECT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamImplicit, default_streams_have_no_signal_slot)
{
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    EXPECT_EQ(lookup(nullptr).stream->signalSlot(), nullptr);
    EXPECT_EQ(lookup(hipStreamPerThread).stream->signalSlot(), nullptr);
}

TEST_F(HipFileStreamImplicit, register_after_transient_use_succeeds)
{
    // Transient use leaves the stream unregistered, so an explicit registration still succeeds.
    (void)lookup(nonnull_stream);
    EXPECT_EQ(hipFileStreamRegister(nonnull_stream, HIPFILE_STREAM_FIXED_FILE_SIZE), HIPFILE_SUCCESS);
    EXPECT_TRUE(lookup(nonnull_stream).stream->fixedIOSize());
    EXPECT_EQ(hipFileStreamRegister(nonnull_stream, 0), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamImplicit, transiently_used_stream_is_not_registered)
{
    // Using a stream for async IO without registering it leaves nothing to deregister.
    (void)lookup(nonnull_stream);
    EXPECT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamImplicit, async_io_on_reserved_priority_stream_is_not_supported)
{
    EXPECT_CALL(mhip, hipStreamGetPriority(nonnull_stream)).WillRepeatedly(::testing::Return(-1));
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);
    size_t  size{4096};
    hoff_t  file_offset{0};
    hoff_t  buf_offset{0};
    ssize_t bytes{0};
    EXPECT_EQ(hipFileReadAsync(fh, buf, &size, &file_offset, &buf_offset, &bytes, nonnull_stream),
              HipFileOpError(hipFileAsyncNotSupported));
    EXPECT_EQ(hipFileWriteAsync(fh, buf, &size, &file_offset, &buf_offset, &bytes, nonnull_stream),
              HipFileOpError(hipFileAsyncNotSupported));
    EXPECT_EQ(hipFileStreamDeregister(nonnull_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(HipFileStreamImplicit, async_io_on_host_buffer_is_not_supported)
{
    void *host_buf{reinterpret_cast<void *>(0x2)};
    expect_buffer_registration(mhip, hipMemoryTypeHost);
    ASSERT_EQ(hipFileBufRegister(host_buf, 4096, 0), HIPFILE_SUCCESS);
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);
    EXPECT_CALL(mhip, hipExtMallocWithFlags).Times(0);

    size_t  size{4096};
    hoff_t  file_offset{0};
    hoff_t  buf_offset{0};
    ssize_t bytes{0};
    EXPECT_EQ(hipFileReadAsync(fh, host_buf, &size, &file_offset, &buf_offset, &bytes, nonnull_stream),
              HipFileOpError(hipFileAsyncNotSupported));
    EXPECT_EQ(hipFileWriteAsync(fh, host_buf, &size, &file_offset, &buf_offset, &bytes, nonnull_stream),
              HipFileOpError(hipFileAsyncNotSupported));
    EXPECT_EQ(hipFileBufDeregister(host_buf), HIPFILE_SUCCESS);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
