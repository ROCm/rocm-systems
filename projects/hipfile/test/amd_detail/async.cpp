/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "hipfile-literals.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "hip.h"
#include "io.h"
#include "masyncmonitor.h"
#include "mbackend.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mstate.h"
#include "mstream.h"
#include "msys.h"
#include "mthread-pool.h"
#include "state.h"
#include "util.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/types.h>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>

using namespace std::chrono_literals;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

using namespace hipFile;
using std::shared_ptr;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::ByMove;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Test;
using ::testing::Throw;
using ::testing::Values;
using ::testing::WithParamInterface;

struct HipFileAsyncOp : public Test {
    HipFileAsyncOp()
        : buffer{std::make_shared<StrictMock<MBuffer>>()}, file{std::make_shared<StrictMock<MFile>>()},
          stream{std::make_shared<StrictMock<MStream>>()}
    {
        EXPECT_CALL(*stream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, getHipDevice).Times(AnyNumber()).WillRepeatedly(Return(0));

        EXPECT_CALL(*buffer, getBuffer)
            .Times(AnyNumber())
            .WillRepeatedly(Return(reinterpret_cast<hipStream_t>(0xFEFEFEFE)));
        EXPECT_CALL(*buffer, getLength).Times(AnyNumber()).WillRepeatedly(Return(size_t{1} << 20));
        EXPECT_CALL(*buffer, getGpuId).Times(AnyNumber()).WillRepeatedly(Return(0));
    }
    StrictMock<MHip>    mhip;
    StrictMock<MSys>    msys;
    shared_ptr<MBuffer> buffer;
    shared_ptr<MFile>   file;
    shared_ptr<MStream> stream;
};

namespace {
std::atomic<int> g_test_io_fn_calls{0};
void
testAsyncIoFn(void *)
{
    g_test_io_fn_calls.fetch_add(1, std::memory_order_relaxed);
}
}

TEST_F(HipFileAsyncOp, submitIo_runs_io_fn_on_pool_and_signals_completion)
{
    StrictMock<MThreadPool> mpool;
    auto                    tg     = std::make_unique<StrictMock<MTaskGroup>>();
    auto                    tg_raw = tg.get();
    std::function<void()>   captured;
    EXPECT_CALL(mpool, makeTaskGroup()).WillOnce(Return(ByMove(std::move(tg))));
    EXPECT_CALL(*tg_raw, wait());
    StrictMock<MAsyncMonitor> mmon;

    uint64_t slot_storage      = 0;
    size_t   size              = 100;
    hoff_t   file_offset       = 0;
    hoff_t   buffer_offset     = 0;
    ssize_t  bytes_transferred = 0;
    EXPECT_CALL(*stream, signalSlot).WillRepeatedly(Return(&slot_storage));
    auto op      = std::make_unique<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                             &buffer_offset, &bytes_transferred);
    op->io_fn    = testAsyncIoFn;
    AsyncOp *raw = op.get();
    g_test_io_fn_calls.store(0);

    EXPECT_CALL(mmon, submitIo(raw)).WillOnce([&mmon](AsyncOp *o) { mmon.AsyncMonitor::submitIo(o); });
    EXPECT_CALL(*tg_raw, run(_)).WillOnce([&captured](std::function<void()> work) {
        captured = std::move(work);
    });
    mmon.submitIo(op.release());

    captured();
    ASSERT_EQ(g_test_io_fn_calls.load(), 1);
    ASSERT_EQ(slot_storage, 1u);
}

TEST_F(HipFileAsyncOp, async_dispatch_delegates_to_submitIo)
{
    StrictMock<MAsyncMonitor> mmon;
    size_t                    size              = 100;
    hoff_t                    file_offset       = 0;
    hoff_t                    buffer_offset     = 0;
    ssize_t                   bytes_transferred = 0;
    auto op = std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                        &buffer_offset, &bytes_transferred);
    EXPECT_CALL(mmon, submitIo(op.get()));
    async_dispatch(op.get());
}

TEST_F(HipFileAsyncOp, async_dispatch_runs_inline_when_submit_throws)
{
    StrictMock<MAsyncMonitor> mmon;
    uint64_t                  slot_storage      = 0;
    size_t                    size              = 100;
    hoff_t                    file_offset       = 0;
    hoff_t                    buffer_offset     = 0;
    ssize_t                   bytes_transferred = 0;
    EXPECT_CALL(*stream, signalSlot).WillRepeatedly(Return(&slot_storage));
    auto op      = std::make_unique<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                             &buffer_offset, &bytes_transferred);
    op->io_fn    = testAsyncIoFn;
    AsyncOp *raw = op.get();
    g_test_io_fn_calls.store(0);

    EXPECT_CALL(mmon, submitIo(raw)).WillOnce(Throw(std::runtime_error("no capacity")));
    async_dispatch(op.release());
    ASSERT_EQ(g_test_io_fn_calls.load(), 1);
    ASSERT_EQ(slot_storage, 1u);
}

TEST_F(HipFileAsyncOp, enqueueAsync_supported_emits_dispatch_and_wait)
{
    StrictMock<MAsyncMonitor> mmon;
    auto                      backend       = std::make_shared<StrictMock<MBackend>>();
    uint64_t                  slot_storage  = 0;
    size_t                    size          = 100;
    hoff_t                    file_offset   = 0;
    hoff_t                    buffer_offset = 0;
    ssize_t                   bytes         = 42;
    auto                      hip_stream    = reinterpret_cast<hipStream_t>(0xABCD);

    EXPECT_CALL(*stream, getHipStream).WillRepeatedly(Return(hip_stream));
    EXPECT_CALL(*stream, canUseStreamWaitValue).WillRepeatedly(Return(true));
    EXPECT_CALL(*stream, nextSignalTarget).WillOnce(Return(5));
    EXPECT_CALL(*stream, signalSlot).WillRepeatedly(Return(&slot_storage));
    EXPECT_CALL(*stream, getLock);
    EXPECT_CALL(mhip, hipLaunchHostFunc(hip_stream, Eq(&async_dispatch), _))
        .WillOnce(
            [](hipStream_t, hipHostFn_t, void *ud) { std::unique_ptr<AsyncOp>{static_cast<AsyncOp *>(ud)}; });
    EXPECT_CALL(mhip, hipStreamWaitValue64(hip_stream, &slot_storage, 5u, hipStreamWaitValueGte, _));

    enqueueAsync(backend, IoType::Read, file, buffer, &size, &file_offset, &buffer_offset, &bytes, stream);
    ASSERT_EQ(bytes, 0);
}

TEST_F(HipFileAsyncOp, enqueueAsync_unsupported_emits_inline)
{
    StrictMock<MAsyncMonitor> mmon;
    auto                      backend       = std::make_shared<StrictMock<MBackend>>();
    size_t                    size          = 100;
    hoff_t                    file_offset   = 0;
    hoff_t                    buffer_offset = 0;
    ssize_t                   bytes         = 0;
    auto                      hip_stream    = reinterpret_cast<hipStream_t>(0xABCD);

    EXPECT_CALL(*stream, getHipStream).WillRepeatedly(Return(hip_stream));
    EXPECT_CALL(*stream, canUseStreamWaitValue).WillRepeatedly(Return(false));
    EXPECT_CALL(*stream, getLock);
    EXPECT_CALL(mhip, hipLaunchHostFunc(hip_stream, Eq(&async_run_inline), _))
        .WillOnce(
            [](hipStream_t, hipHostFn_t, void *ud) { std::unique_ptr<AsyncOp>{static_cast<AsyncOp *>(ud)}; });

    enqueueAsync(backend, IoType::Read, file, buffer, &size, &file_offset, &buffer_offset, &bytes, stream);
}

TEST_F(HipFileAsyncOp, enqueueAsync_compensates_signal_when_dispatch_fails)
{
    StrictMock<MAsyncMonitor> mmon;
    auto                      backend       = std::make_shared<StrictMock<MBackend>>();
    uint64_t                  slot_storage  = 0;
    size_t                    size          = 100;
    hoff_t                    file_offset   = 0;
    hoff_t                    buffer_offset = 0;
    ssize_t                   bytes         = 0;
    auto                      hip_stream    = reinterpret_cast<hipStream_t>(0xABCD);

    EXPECT_CALL(*stream, getHipStream).WillRepeatedly(Return(hip_stream));
    EXPECT_CALL(*stream, canUseStreamWaitValue).WillRepeatedly(Return(true));
    EXPECT_CALL(*stream, nextSignalTarget).WillOnce(Return(1));
    EXPECT_CALL(*stream, signalSlot).WillRepeatedly(Return(&slot_storage));
    EXPECT_CALL(*stream, getLock);
    EXPECT_CALL(mhip, hipLaunchHostFunc(hip_stream, Eq(&async_dispatch), _))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));

    EXPECT_THROW(enqueueAsync(backend, IoType::Read, file, buffer, &size, &file_offset, &buffer_offset,
                              &bytes, stream),
                 Hip::RuntimeError);
    ASSERT_EQ(slot_storage, 1u);
}

TEST_F(HipFileAsyncOp, enqueueAsync_throws_on_invalid_params)
{
    StrictMock<MAsyncMonitor> mmon;
    auto                      backend       = std::make_shared<StrictMock<MBackend>>();
    size_t                    size          = 100;
    hoff_t                    file_offset   = 0;
    hoff_t                    buffer_offset = -1;
    ssize_t                   bytes         = 0;

    EXPECT_THROW(enqueueAsync(backend, IoType::Read, file, buffer, &size, &file_offset, &buffer_offset,
                              &bytes, stream),
                 std::invalid_argument);
}

TEST_F(HipFileAsyncOp, async_run_io_reports_invalid_value_on_bad_params)
{
    size_t  size          = 100;
    hoff_t  file_offset   = 0;
    hoff_t  buffer_offset = size_t{2} << 20;
    ssize_t bytes         = 0;
    auto    op            = std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                                      &buffer_offset, &bytes);
    async_run_io(op.get());
    ASSERT_EQ(bytes, -hipFileInvalidValue);
}

static auto
hipfileFlagsPowerSet()
{
    return Combine(Values(0, HIPFILE_STREAM_FIXED_BUF_OFFSET), Values(0, HIPFILE_STREAM_FIXED_FILE_OFFSET),
                   Values(0, HIPFILE_STREAM_FIXED_FILE_SIZE), Values(0, HIPFILE_STREAM_PAGE_ALIGNED_INPUTS));
}

struct HipFileAsyncOpStreamParams
    : public HipFileAsyncOp,
      public WithParamInterface<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {
    HipFileAsyncOpStreamParams() : HipFileAsyncOp{}
    {
        auto params            = GetParam();
        bool fixed_buf_offset  = std::get<0>(params) > 0;
        bool fixed_file_offset = std::get<1>(params) > 0;
        bool fixed_file_size   = std::get<2>(params) > 0;
        bool page_aligned      = std::get<3>(params) > 0;
        flags  = std::get<0>(params) | std::get<1>(params) | std::get<2>(params) | std::get<3>(params);
        stream = std::make_shared<StrictMock<MStream>>();
        EXPECT_CALL(*stream, getHipStream)
            .Times(AnyNumber())
            .WillRepeatedly(Return(reinterpret_cast<hipStream_t>(0xDEADBEEF)));
        EXPECT_CALL(*stream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(fixed_buf_offset));
        EXPECT_CALL(*stream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(fixed_file_offset));
        EXPECT_CALL(*stream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(fixed_file_size));
        EXPECT_CALL(*stream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(page_aligned));
    }
    unsigned flags;
};

TEST_P(HipFileAsyncOpStreamParams, asyncOp_construction_has_correct_variants)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op = std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                           &buffer_offset, &bytes_transferred);

    // Unfixed flags will be pointers
    if (flags & HIPFILE_STREAM_FIXED_BUF_OFFSET) {
        EXPECT_NO_THROW(std::get<const hoff_t>(op->buffer_offset));
    }
    else {
        EXPECT_NO_THROW(std::get<hoff_t *>(op->buffer_offset));
    }
    if (flags & HIPFILE_STREAM_FIXED_FILE_OFFSET) {
        EXPECT_NO_THROW(std::get<const hoff_t>(op->file_offset));
    }
    else {
        EXPECT_NO_THROW(std::get<hoff_t *>(op->file_offset));
    }
    if (flags & HIPFILE_STREAM_FIXED_FILE_SIZE) {
        EXPECT_NO_THROW(std::get<size_t>(op->size));
    }
    else {
        EXPECT_NO_THROW(std::get<size_t *>(op->size));
    }
}
INSTANTIATE_TEST_SUITE_P(StreamSuite, HipFileAsyncOpStreamParams, hipfileFlagsPowerSet());

HIPFILE_WARN_NO_EXIT_DTOR_OFF
struct AsyncIoFunction {
    hipFileError_t (*function)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
};
static std::array<AsyncIoFunction, 2> asyncIOFns{
    {{hipFileReadAsync, "hipFileReadAsync"}, {hipFileWriteAsync, "hipFileWriteAsync"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

struct HipFileReadWriteAsync : public HipFileOpened, public ::testing::WithParamInterface<AsyncIoFunction> {
    void SetUp() override
    {
        nonnull_void   = reinterpret_cast<void *>(1);
        nonnull_size   = reinterpret_cast<size_t *>(1);
        nonnull_offset = reinterpret_cast<hoff_t *>(1);
        nonnull_ssize  = reinterpret_cast<ssize_t *>(1);
        nonnull_stream = reinterpret_cast<hipStream_t>(1);
        io_op          = GetParam().function;
        name           = GetParam().name;
    }
    StrictMock<MHip> mhip;
    StrictMock<MSys> msys;
    void            *nonnull_void;
    size_t          *nonnull_size;
    ssize_t         *nonnull_ssize;
    hoff_t          *nonnull_offset;
    hipStream_t      nonnull_stream;
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
};

TEST_P(HipFileReadWriteAsync, nullSizeReturnsError)
{
    ASSERT_EQ(io_op(nonnull_void, nonnull_void, nullptr, nonnull_offset, nonnull_offset, nonnull_ssize,
                    nonnull_stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipFileReadWriteAsync, nullFileOffsetReturnsError)
{
    ASSERT_EQ(io_op(nonnull_void, nonnull_void, nonnull_size, nullptr, nonnull_offset, nonnull_ssize,
                    nonnull_stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipFileReadWriteAsync, nullBufferOffsetReturnsError)
{
    ASSERT_EQ(io_op(nonnull_void, nonnull_void, nonnull_size, nonnull_offset, nullptr, nonnull_ssize,
                    nonnull_stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipFileReadWriteAsync, nullBytesTransferredReturnsError)
{
    ASSERT_EQ(io_op(nonnull_void, nonnull_void, nonnull_size, nonnull_offset, nonnull_offset, nullptr,
                    nonnull_stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipFileReadWriteAsync, unregisteredFileReturnsError)
{
    size_t  size          = 1;
    hoff_t  file_offset   = 0;
    hoff_t  buffer_offset = 0;
    ssize_t bytes_written = 0;

    ASSERT_EQ(io_op(nonnull_void, nonnull_void, &size, &file_offset, &buffer_offset, &bytes_written,
                    nonnull_stream),
              HipFileOpError(hipFileHandleNotRegistered));
}

TEST_P(HipFileReadWriteAsync, badAllocReturnsHipErrorOutOfMemory)
{
    MDriverState driver;
    EXPECT_CALL(driver, getRefCount).WillOnce(Throw(std::bad_alloc()));
    ASSERT_EQ(io_op(nonnull_void, nonnull_void, nonnull_size, nonnull_offset, nonnull_offset, nullptr,
                    nonnull_stream),
              HipFileHipError(hipErrorOutOfMemory));
}

INSTANTIATE_TEST_SUITE_P(HipFileAsyncSuite, HipFileReadWriteAsync, ::testing::ValuesIn(asyncIOFns),
                         [](const testing::TestParamInfo<HipFileReadWriteAsync::ParamType> &param_info) {
                             return param_info.param.name;
                         });

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
