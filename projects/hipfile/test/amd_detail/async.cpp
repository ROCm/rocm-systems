/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "context.h"
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

void
configureStream(NiceMock<MStream> &stream, hipStream_t hip_stream, bool wait_value_capable)
{
    ON_CALL(stream, getHipStream).WillByDefault(Return(hip_stream));
    ON_CALL(stream, getHipDevice).WillByDefault(Return(0));
    ON_CALL(stream, fixedIOSize).WillByDefault(Return(false));
    ON_CALL(stream, fixedFileOffset).WillByDefault(Return(false));
    ON_CALL(stream, fixedBufferOffset).WillByDefault(Return(false));
    ON_CALL(stream, isRetired).WillByDefault(Return(false));
    ON_CALL(stream, canUseStreamWaitValue).WillByDefault(Return(wait_value_capable));
    ON_CALL(stream, signalSlot).WillByDefault(Return(nullptr));
    ON_CALL(stream, nextSignalTarget).WillByDefault(Return(1));
}

std::shared_ptr<AsyncOp>
makeOp(std::shared_ptr<IStream> stream, std::shared_ptr<IBuffer> buffer, std::shared_ptr<Backend> backend,
       size_t *size, hoff_t *file_offset, hoff_t *buffer_offset, ssize_t *bytes,
       std::function<void(AsyncOp *)> io_fn)
{
    return std::make_shared<AsyncOp>(IoType::Read, std::make_shared<NiceMock<MFile>>(), std::move(buffer),
                                     std::move(stream), size, file_offset, buffer_offset, bytes,
                                     std::move(io_fn), std::move(backend));
}

}

TEST(AsyncRunIo, SuccessRecordsBytes)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), false);
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));

    auto op = makeOp(stream, buffer, backend, &size, &file_offset, &buffer_offset, &bytes, async_run_io);
    EXPECT_CALL(*backend, io(IoType::Read, _, _, size, file_offset, buffer_offset)).WillOnce(Return(512));

    async_run_io(op.get());
    EXPECT_EQ(bytes, 512);
}

TEST(AsyncRunIo, InvalidParamsReportsInvalidValue)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{0};
    configureStream(*stream, nonnull_stream(), false);
    ON_CALL(*buffer, getLength).WillByDefault(Return(16)); // smaller than size

    auto op = makeOp(stream, buffer, backend, &size, &file_offset, &buffer_offset, &bytes, async_run_io);
    EXPECT_CALL(*backend, io).Times(0);

    async_run_io(op.get());
    EXPECT_EQ(bytes, -hipFileInvalidValue);
}

TEST(AsyncRunIo, BackendThrowReportsInternalError)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{0};
    configureStream(*stream, nonnull_stream(), false);
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));

    auto op = makeOp(stream, buffer, backend, &size, &file_offset, &buffer_offset, &bytes, async_run_io);
    EXPECT_CALL(*backend, io).WillOnce(Throw(std::runtime_error("boom")));

    async_run_io(op.get());
    EXPECT_EQ(bytes, -hipFileInternalError);
}

TEST(EnqueueAsync, NullStreamRunsSynchronously)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nullptr, false);
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));

    EXPECT_CALL(mhip, hipStreamSynchronize(nullptr));
    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);
    EXPECT_CALL(*backend, io).WillOnce(Return(512));

    enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size, &file_offset,
                 &buffer_offset, &bytes, stream);
    EXPECT_EQ(bytes, 512);
}

TEST(EnqueueAsync, RegisteredStreamArmsWaitValue)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    uint64_t       slot{0};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*stream, signalSlot).WillByDefault(Return(&slot));
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));

    // hipLaunchHostFunc is mocked, so the committed op is never run or freed by a real host function;
    // capture and free it here to stand in for that ownership transfer.
    AsyncOp *committed{nullptr};
    EXPECT_CALL(mhip, hipLaunchHostFunc(nonnull_stream(), _, _))
        .WillOnce([&committed](hipStream_t, hipHostFn_t, void *userdata) {
            committed = static_cast<AsyncOp *>(userdata);
        });
    EXPECT_CALL(mhip, hipStreamWaitValue64(nonnull_stream(), &slot, 1u, _, _));

    enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size, &file_offset,
                 &buffer_offset, &bytes, stream);
    ASSERT_NE(committed, nullptr);
    EXPECT_FALSE(committed->block); // wait value armed -> fire and forget
    delete committed;
}

TEST(EnqueueAsync, ArmFailureStillSucceeds)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    uint64_t       slot{0};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*stream, signalSlot).WillByDefault(Return(&slot));
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));

    AsyncOp *committed{nullptr};
    EXPECT_CALL(mhip, hipLaunchHostFunc(nonnull_stream(), _, _))
        .WillOnce([&committed](hipStream_t, hipHostFn_t, void *userdata) {
            committed = static_cast<AsyncOp *>(userdata);
        });
    EXPECT_CALL(mhip, hipStreamWaitValue64).WillOnce(Throw(Hip::RuntimeError(hipErrorNotSupported)));

    EXPECT_NO_THROW(enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size,
                                 &file_offset, &buffer_offset, &bytes, stream));
    ASSERT_NE(committed, nullptr);
    EXPECT_TRUE(committed->block); // arm failed -> op self-blocks
    delete committed;
}

TEST(EnqueueAsync, LaunchFailureThrows)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));

    EXPECT_CALL(mhip, hipLaunchHostFunc).WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(mhip, hipStreamWaitValue64).Times(0);

    EXPECT_THROW(enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size,
                              &file_offset, &buffer_offset, &bytes, stream),
                 Hip::RuntimeError);
}

TEST(EnqueueAsync, RetiredStreamThrows)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*stream, isRetired).WillByDefault(Return(true));
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(0));

    EXPECT_CALL(mhip, hipLaunchHostFunc).Times(0);

    EXPECT_THROW(enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size,
                              &file_offset, &buffer_offset, &bytes, stream),
                 std::invalid_argument);
}

TEST(EnqueueAsync, GpuMismatchThrows)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{512};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{-1};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*buffer, getLength).WillByDefault(Return(4096));
    ON_CALL(*buffer, getGpuId).WillByDefault(Return(1)); // stream device is 0

    EXPECT_THROW(enqueueAsync(backend, IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, &size,
                              &file_offset, &buffer_offset, &bytes, stream),
                 std::invalid_argument);
}

TEST(AsyncDispatch, BlockTrueSubmitsAndWaits)
{
    MAsyncMonitor monitor;
    auto          stream = std::make_shared<NiceMock<MStream>>();
    auto          buffer = std::make_shared<NiceMock<MBuffer>>();
    auto          backend{std::make_shared<NiceMock<MBackend>>()};
    size_t        size{512};
    hoff_t        file_offset{0};
    hoff_t        buffer_offset{0};
    ssize_t       bytes{0};
    configureStream(*stream, nonnull_stream(), false);

    auto *op  = new AsyncOp(IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, stream, &size,
                            &file_offset, &buffer_offset, &bytes, async_run_io, backend);
    op->block = true;
    EXPECT_CALL(monitor, submitIoAndWait);
    EXPECT_CALL(monitor, submitIo).Times(0);

    async_dispatch(op);
}

TEST(AsyncDispatch, BlockFalseSubmits)
{
    MAsyncMonitor monitor;
    auto          stream = std::make_shared<NiceMock<MStream>>();
    auto          buffer = std::make_shared<NiceMock<MBuffer>>();
    auto          backend{std::make_shared<NiceMock<MBackend>>()};
    size_t        size{512};
    hoff_t        file_offset{0};
    hoff_t        buffer_offset{0};
    ssize_t       bytes{0};
    configureStream(*stream, nonnull_stream(), false);

    auto *op  = new AsyncOp(IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, stream, &size,
                            &file_offset, &buffer_offset, &bytes, async_run_io, backend);
    op->block = false;
    EXPECT_CALL(monitor, submitIo);
    EXPECT_CALL(monitor, submitIoAndWait).Times(0);

    async_dispatch(op);
}

TEST(AsyncDispatch, SubmitFailureReportsError)
{
    MAsyncMonitor monitor;
    auto          stream = std::make_shared<NiceMock<MStream>>();
    auto          buffer = std::make_shared<NiceMock<MBuffer>>();
    auto          backend{std::make_shared<NiceMock<MBackend>>()};
    size_t        size{512};
    hoff_t        file_offset{0};
    hoff_t        buffer_offset{0};
    ssize_t       bytes{0};
    configureStream(*stream, nonnull_stream(), false);

    auto *op  = new AsyncOp(IoType::Read, std::make_shared<NiceMock<MFile>>(), buffer, stream, &size,
                            &file_offset, &buffer_offset, &bytes, async_run_io, backend);
    op->block = true;
    EXPECT_CALL(monitor, submitIoAndWait).WillOnce(Throw(std::runtime_error("no workers")));

    async_dispatch(op);
    EXPECT_EQ(bytes, -hipFileInternalError);
}

TEST(AsyncMonitorReal, SubmitIoRunsIoFnAndSignalsSlot)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    uint64_t       slot{0};
    size_t         size{0};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{0};
    bool           ran{false};
    configureStream(*stream, nonnull_stream(), true);
    ON_CALL(*stream, signalSlot).WillByDefault(Return(&slot));

    AsyncMonitor monitor{std::make_unique<InlinePool>()};
    auto         op = makeOp(stream, buffer, backend, &size, &file_offset, &buffer_offset, &bytes,
                             [&ran](AsyncOp *) { ran = true; });

    monitor.submitIo(op);
    EXPECT_TRUE(ran);
    EXPECT_EQ(slot, 1u);
}

TEST(AsyncMonitorReal, SubmitIoAndWaitRunsIoFn)
{
    NiceMock<MHip> mhip;
    auto           stream = std::make_shared<NiceMock<MStream>>();
    auto           buffer = std::make_shared<NiceMock<MBuffer>>();
    auto           backend{std::make_shared<NiceMock<MBackend>>()};
    size_t         size{0};
    hoff_t         file_offset{0};
    hoff_t         buffer_offset{0};
    ssize_t        bytes{0};
    bool           ran{false};
    configureStream(*stream, nonnull_stream(), false);

    AsyncMonitor monitor{std::make_unique<InlinePool>()};
    auto         op = makeOp(stream, buffer, backend, &size, &file_offset, &buffer_offset, &bytes,
                             [&ran](AsyncOp *) { ran = true; });

    monitor.submitIoAndWait(op);
    EXPECT_TRUE(ran);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
