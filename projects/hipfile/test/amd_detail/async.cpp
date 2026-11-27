/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend/fallback.h"
#include "configuration.h"
#include "context.h"
#include "cooperative-copy.h"
#include "hip.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "io.h"
#include "masyncmonitor.h"
#include "mbackend.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mstream.h"
#include "msys.h"
#include "thread-pool.h"

#include <array>
#include <cstdint>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

using namespace hipFile;
using ::testing::_;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Throw;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

namespace {

hipStream_t
nonnull_stream()
{
    return reinterpret_cast<hipStream_t>(0x1);
}

// A dedicated pool that runs each task inline, so the real AsyncMonitor can be driven without threads.
struct InlineTaskGroup : ITaskGroup {
    void run(std::function<void()> task) override
    {
        task();
    }
    void cancel() override
    {
    }
    void wait() override
    {
    }
};
struct InlinePool : IThreadPool {
    std::unique_ptr<ITaskGroup> makeTaskGroup() override
    {
        return std::make_unique<InlineTaskGroup>();
    }
};

struct AsyncTest : public ::testing::Test {
    AsyncTest()
    {
        ON_CALL(mhip, hipHostMalloc).WillByDefault(Return(storage.get()));
        ON_CALL(mhip, hipHostGetDevicePointer).WillByDefault(Return(storage.get()));
        copy = std::make_unique<CooperativeCopy>(mhip);
        ON_CALL(*stream, ensureCooperativeCopy).WillByDefault(::testing::ReturnRef(*copy));
        configure(nonnull_stream(), true);
        ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
        ON_CALL(*buffer, getType).WillByDefault(Return(hipMemoryTypeDevice));
        ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));
    }

    ~AsyncTest() override
    {
        // hipLaunchHostFunc is mocked, so a queued op whose host function never ran still owns itself.
        if (queued) {
            queued->self.reset();
        }
    }

    void configure(hipStream_t hip_stream, bool wait_value_capable)
    {
        ON_CALL(*stream, getHipStream).WillByDefault(Return(hip_stream));
        ON_CALL(*stream, getHipDevice).WillByDefault(Return(0));
        ON_CALL(*stream, isRetired).WillByDefault(Return(false));
        ON_CALL(*stream, canUseStreamWaitValue).WillByDefault(Return(wait_value_capable));
        ON_CALL(*stream, signalSlot).WillByDefault(Return(wait_value_capable ? &slot : nullptr));
        ON_CALL(*stream, nextSignalTarget).WillByDefault(Return(1));
    }

    void captureDispatch()
    {
        EXPECT_CALL(mhip, hipLaunchHostFunc(nonnull_stream(), async_dispatch, _))
            .WillOnce([this](hipStream_t, hipHostFn_t, void *userdata) {
                queued = static_cast<AsyncOp *>(userdata)->self;
            });
    }

    void enqueue()
    {
        enqueueAsync(backend, IoType::Read, file, buffer, &size, &file_offset, &buffer_offset, &bytes,
                     stream);
    }

    std::shared_ptr<AsyncOp> makeOp()
    {
        return std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                         &buffer_offset, &bytes, backend);
    }

    NiceMock<MHip>                      mhip;
    std::shared_ptr<NiceMock<MStream>>  stream{std::make_shared<NiceMock<MStream>>()};
    std::shared_ptr<NiceMock<MBuffer>>  buffer{std::make_shared<NiceMock<MBuffer>>()};
    std::shared_ptr<NiceMock<MFile>>    file{std::make_shared<NiceMock<MFile>>()};
    std::shared_ptr<NiceMock<MBackend>> backend{std::make_shared<NiceMock<MBackend>>()};
    std::unique_ptr<CopyExchange>       storage{std::make_unique<CopyExchange>()};
    std::unique_ptr<CooperativeCopy>    copy;
    uint64_t                            slot{0};
    size_t                              size{512};
    hoff_t                              file_offset{0};
    hoff_t                              buffer_offset{0};
    ssize_t                             bytes{-1};
    std::shared_ptr<AsyncOp>            queued;
};

}

TEST_F(AsyncTest, SecondReadUsesOtherSlotBeforeGpuAcknowledgesFirst)
{
    NiceMock<MSys>   sys;
    auto            &first  = storage->slots.front();
    auto            &second = storage->slots.back();
    constexpr hoff_t start  = 17;
    EXPECT_CALL(sys, pread(3, first.data.data(), 16_MiB, start)).WillOnce(Return(16_MiB));
    EXPECT_CALL(sys, pread(3, second.data.data(), 1, start + 16_MiB))
        .WillOnce([&](int, void *, size_t, off_t) {
            EXPECT_EQ(first.requested, 1U);
            EXPECT_EQ(first.count, 16_MiB);
            EXPECT_EQ(first.offset, 5U);
            return 1;
        });
    EXPECT_EQ(copy->transfer(sys, IoType::Read, 3, 16_MiB + 1, start, 5), 16_MiB + 1);
    EXPECT_EQ(second.requested, 1U);
    EXPECT_EQ(second.offset, 16_MiB + 5);
}

TEST_F(AsyncTest, InvalidReadCountNeverReachesKernel)
{
    NiceMock<MSys> sys;
    EXPECT_CALL(sys, pread).WillOnce(Return(2));
    EXPECT_THROW((void)copy->transfer(sys, IoType::Read, 3, 1, 0, 0), std::system_error);
    EXPECT_EQ(storage->slots.front().requested, 0U);
}

TEST_F(AsyncTest, RunIoRecordsBytes)
{
    EXPECT_CALL(*backend, io(IoType::Read, _, _, size, file_offset, buffer_offset)).WillOnce(Return(512));
    auto op = makeOp();
    async_run_io(op.get());
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, RunIoInvalidParamsReportsInvalidValue)
{
    ON_CALL(*buffer, getLength).WillByDefault(Return(16));
    EXPECT_CALL(*backend, io).Times(0);
    auto op = makeOp();
    async_run_io(op.get());
    EXPECT_EQ(bytes, -hipFileInvalidValue);
}

TEST_F(AsyncTest, RunIoBackendThrowReportsInternalError)
{
    EXPECT_CALL(*backend, io).WillOnce(Throw(std::runtime_error("boom")));
    auto op = makeOp();
    async_run_io(op.get());
    EXPECT_EQ(bytes, -hipFileInternalError);
}

TEST_F(AsyncTest, NullStreamRunsSynchronously)
{
    configure(nullptr, false);
    EXPECT_CALL(mhip, hipStreamSynchronize(nullptr));
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);
    EXPECT_CALL(*backend, io).WillOnce(Return(512));
    enqueue();
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, PerThreadStreamRunsSynchronously)
{
    configure(hipStreamPerThread, false);
    EXPECT_CALL(mhip, hipStreamSynchronize(hipStreamPerThread));
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);
    EXPECT_CALL(*backend, io).WillOnce(Return(512));
    enqueue();
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, GatedStreamDispatchesThenArmsWait)
{
    {
        InSequence seq;
        EXPECT_CALL(*stream, nextSignalTarget).WillOnce(Return(1));
        captureDispatch();
        EXPECT_CALL(mhip,
                    hipStreamWaitValue64(nonnull_stream(), &slot, 1u, hipStreamWaitValueGte, UINT64_MAX));
    }
    EXPECT_CALL(*backend, io).Times(0);

    enqueue();
    ASSERT_NE(queued, nullptr);
    EXPECT_FALSE(queued->block);
    EXPECT_EQ(queued->state.load(), AsyncOpState::Pending);
    EXPECT_EQ(bytes, 0);
}

TEST_F(AsyncTest, UngatedStreamDispatchesBlockingOp)
{
    configure(nonnull_stream(), false);
    captureDispatch();
    EXPECT_CALL(*stream, nextSignalTarget).Times(0);
    EXPECT_CALL(mhip, hipStreamWaitValue64).Times(0);

    enqueue();
    ASSERT_NE(queued, nullptr);
    EXPECT_TRUE(queued->block);
}

TEST_F(AsyncTest, ArmFailureBeforeStartAbandonsOp)
{
    captureDispatch();
    EXPECT_CALL(mhip, hipStreamWaitValue64).WillOnce(Throw(Hip::RuntimeError(hipErrorNotSupported)));

    EXPECT_THROW(enqueue(), Hip::RuntimeError);
    ASSERT_NE(queued, nullptr);
    EXPECT_EQ(queued->state.load(), AsyncOpState::Abandoned);

    // When the host function runs later, the pool skips the IO and only releases the stream signal.
    AsyncMonitor                   monitor{std::make_unique<InlinePool>()};
    ContextOverride<IAsyncMonitor> co{&monitor};
    EXPECT_CALL(*backend, io).Times(0);
    async_dispatch(queued.get());
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(bytes, 0);
}

TEST_F(AsyncTest, ArmFailureAfterStartWaitsForIo)
{
    AsyncMonitor                   monitor{std::make_unique<InlinePool>()};
    ContextOverride<IAsyncMonitor> co{&monitor};
    captureDispatch();
    EXPECT_CALL(*backend, io).WillOnce(Return(512));
    // The host function and pool finish the op before the arm reports its failure.
    EXPECT_CALL(mhip, hipStreamWaitValue64)
        .WillOnce([this](hipStream_t, uint64_t *, uint64_t, unsigned int, uint64_t) {
            async_dispatch(queued.get());
            throw Hip::RuntimeError(hipErrorNotSupported);
        });

    EXPECT_NO_THROW(enqueue());
    EXPECT_EQ(queued->state.load(), AsyncOpState::Done);
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, KernelLaunchFailureAbandonsPendingIo)
{
    captureDispatch();
    EXPECT_CALL(mhip, hipLaunchKernel).WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidDeviceFunction)));
    EXPECT_THROW(enqueue(), Hip::RuntimeError);
    ASSERT_NE(queued, nullptr);
    EXPECT_EQ(queued->state.load(), AsyncOpState::Abandoned);
    AsyncMonitor                   monitor{std::make_unique<InlinePool>()};
    ContextOverride<IAsyncMonitor> co{&monitor};
    async_dispatch(queued.get());
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(bytes, 0);
}

TEST_F(AsyncTest, KernelLaunchFailureCancelsDeferredFallback)
{
    AsyncMonitor                   monitor{std::make_unique<InlinePool>()};
    ContextOverride<IAsyncMonitor> co{&monitor};
    captureDispatch();
    EXPECT_CALL(mhip, hipLaunchKernel)
        .WillOnce([this](const void *, dim3, dim3, void **, size_t, hipStream_t) {
            async_dispatch(queued.get());
            EXPECT_EQ(slot, 1u);
            EXPECT_EQ(storage->ticket, 1u);
            throw Hip::RuntimeError(hipErrorInvalidDeviceFunction);
        });
    EXPECT_CALL(mhip, hipMemcpyWithStream).Times(0);
    Context<Configuration>::get()->fallback(true);
    enqueueAsync(std::make_shared<Fallback>(), IoType::Read, file, buffer, &size, &file_offset,
                 &buffer_offset, &bytes, stream);
    EXPECT_EQ(queued->state.load(), AsyncOpState::Done);
    EXPECT_EQ(bytes, -hipFileInternalError);
}

TEST_F(AsyncTest, LaunchFailureReleasesReservation)
{
    EXPECT_CALL(*stream, nextSignalTarget).WillOnce(Return(1));
    EXPECT_CALL(mhip, hipLaunchHostFunc).WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(*stream, releaseSignalTarget);
    EXPECT_CALL(mhip, hipStreamWaitValue64).Times(0);

    EXPECT_THROW(enqueue(), Hip::RuntimeError);
}

TEST_F(AsyncTest, RetiredStreamThrows)
{
    ON_CALL(*stream, isRetired).WillByDefault(Return(true));
    EXPECT_CALL(*stream, nextSignalTarget).Times(0);
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);

    EXPECT_THROW(enqueue(), std::invalid_argument);
}

TEST_F(AsyncTest, GpuMismatchThrows)
{
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(1));
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);

    EXPECT_THROW(enqueue(), std::invalid_argument);
}

TEST_F(AsyncTest, HostBufferIsNotSupported)
{
    // A host buffer's gpu_id of -1 would also fail the device check; the explicit rejection must win.
    ON_CALL(*buffer, getType).WillByDefault(Return(hipMemoryTypeHost));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(-1));
    EXPECT_CALL(*stream, nextSignalTarget).Times(0);
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);

    try {
        enqueue();
        FAIL() << "enqueueAsync accepted a host buffer";
    }
    catch (const hipFileError_t &e) {
        EXPECT_EQ(e.err, hipFileAsyncNotSupported);
        EXPECT_EQ(e.hip_drv_err, hipSuccess);
    }
}

TEST_F(AsyncTest, DispatchBlockingOpWaitsForIo)
{
    // A real one-worker pool runs the op on another thread, so returning early would leave it unfinished.
    AsyncMonitor                   monitor{std::make_unique<ThreadPool>(1)};
    ContextOverride<IAsyncMonitor> co{&monitor};
    auto                           op = makeOp();
    op->block                         = true;
    op->self                          = op;
    EXPECT_CALL(*backend, io).WillOnce(Return(512));

    async_dispatch(op.get());
    EXPECT_EQ(op->state.load(), AsyncOpState::Done);
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, DispatchNonBlockingOpSubmits)
{
    MAsyncMonitor monitor;
    auto          op = makeOp();
    op->self         = op;
    EXPECT_CALL(monitor, submitIo);

    async_dispatch(op.get());
    EXPECT_EQ(op->self, nullptr);
}

TEST_F(AsyncTest, DispatchSubmitFailureFinishesOpWithError)
{
    // A blocking op, so a failed submission must still finish the op or the host function would hang.
    MAsyncMonitor monitor;
    auto          op = makeOp();
    op->block        = true;
    op->self         = op;
    EXPECT_CALL(monitor, submitIo).WillOnce(Throw(std::runtime_error("no workers")));

    async_dispatch(op.get());
    EXPECT_EQ(bytes, -hipFileInternalError);
    EXPECT_EQ(op->state.load(), AsyncOpState::Done);
    EXPECT_EQ(slot, 1u);
}

TEST_F(AsyncTest, MonitorSubmitIoRunsOpAndSignals)
{
    AsyncMonitor monitor{std::make_unique<InlinePool>()};
    auto         op = makeOp();
    EXPECT_CALL(*backend, io).WillOnce(Return(512));

    monitor.submitIo(op);
    EXPECT_EQ(bytes, 512);
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(op->state.load(), AsyncOpState::Done);
}

TEST_F(AsyncTest, MonitorSkipsAbandonedOp)
{
    AsyncMonitor monitor{std::make_unique<InlinePool>()};
    auto         op = makeOp();
    op->state.store(AsyncOpState::Abandoned);
    EXPECT_CALL(*backend, io).Times(0);

    monitor.submitIo(op);
    EXPECT_EQ(bytes, -1);
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(op->state.load(), AsyncOpState::Abandoned);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
