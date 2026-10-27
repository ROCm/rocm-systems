/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "context.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-warnings.h"
#include "io.h"
#include "masyncmonitor.h"
#include "mbackend.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mstream.h"
#include "thread-pool.h"

#include <cstdint>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <utility>

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
        configure(nonnull_stream(), true);
        ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
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

    std::shared_ptr<AsyncOp> makeOp(std::function<void(AsyncOp *)> io_fn)
    {
        return std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                         &buffer_offset, &bytes, std::move(io_fn), backend);
    }

    NiceMock<MHip>                      mhip;
    std::shared_ptr<NiceMock<MStream>>  stream{std::make_shared<NiceMock<MStream>>()};
    std::shared_ptr<NiceMock<MBuffer>>  buffer{std::make_shared<NiceMock<MBuffer>>()};
    std::shared_ptr<NiceMock<MFile>>    file{std::make_shared<NiceMock<MFile>>()};
    std::shared_ptr<NiceMock<MBackend>> backend{std::make_shared<NiceMock<MBackend>>()};
    uint64_t                            slot{0};
    size_t                              size{512};
    hoff_t                              file_offset{0};
    hoff_t                              buffer_offset{0};
    ssize_t                             bytes{-1};
    std::shared_ptr<AsyncOp>            queued;
};

}

TEST_F(AsyncTest, RunIoRecordsBytes)
{
    EXPECT_CALL(*backend, io(IoType::Read, _, _, size, file_offset, buffer_offset)).WillOnce(Return(512));
    auto op = makeOp(async_run_io);
    async_run_io(op.get());
    EXPECT_EQ(bytes, 512);
}

TEST_F(AsyncTest, RunIoInvalidParamsReportsInvalidValue)
{
    ON_CALL(*buffer, getLength).WillByDefault(Return(16));
    EXPECT_CALL(*backend, io).Times(0);
    auto op = makeOp(async_run_io);
    async_run_io(op.get());
    EXPECT_EQ(bytes, -hipFileInvalidValue);
}

TEST_F(AsyncTest, RunIoBackendThrowReportsInternalError)
{
    EXPECT_CALL(*backend, io).WillOnce(Throw(std::runtime_error("boom")));
    auto op = makeOp(async_run_io);
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

TEST_F(AsyncTest, DispatchBlockingOpWaitsForIo)
{
    // A real one-worker pool runs the op on another thread, so returning early would leave it unfinished.
    AsyncMonitor                   monitor{std::make_unique<ThreadPool>(1)};
    ContextOverride<IAsyncMonitor> co{&monitor};
    auto                           op = makeOp(async_run_io);
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
    auto          op = makeOp(async_run_io);
    op->self         = op;
    EXPECT_CALL(monitor, submitIo);

    async_dispatch(op.get());
    EXPECT_EQ(op->self, nullptr);
}

TEST_F(AsyncTest, DispatchSubmitFailureFinishesOpWithError)
{
    // A blocking op, so a failed submission must still finish the op or the host function would hang.
    MAsyncMonitor monitor;
    auto          op = makeOp(async_run_io);
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
    bool         ran{false};
    auto         op = makeOp([&ran](AsyncOp *) { ran = true; });

    monitor.submitIo(op);
    EXPECT_TRUE(ran);
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(op->state.load(), AsyncOpState::Done);
}

TEST_F(AsyncTest, MonitorSkipsAbandonedOp)
{
    AsyncMonitor monitor{std::make_unique<InlinePool>()};
    bool         ran{false};
    auto         op = makeOp([&ran](AsyncOp *) { ran = true; });
    op->state.store(AsyncOpState::Abandoned);

    monitor.submitIo(op);
    EXPECT_FALSE(ran);
    EXPECT_EQ(slot, 1u);
    EXPECT_EQ(op->state.load(), AsyncOpState::Abandoned);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
