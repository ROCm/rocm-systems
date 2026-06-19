/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "backend/asyncop-fallback.h"
#include "backend/fallback.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "hipfile-literals.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "hip.h"
#include "io.h"
#include "masyncmonitor.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mstate.h"
#include "mstream.h"
#include "msys.h"
#include "state.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Test;
using ::testing::Throw;
using ::testing::Values;
using ::testing::WithParamInterface;

static std::shared_ptr<void>
make_shared_void(std::size_t num_bytes)
{
    return std::static_pointer_cast<void>(std::make_shared<std::byte[]>(num_bytes));
}

struct HipFileAsyncOp : public Test {
    HipFileAsyncOp()
        : buffer{std::make_shared<StrictMock<MBuffer>>()}, file{std::make_shared<StrictMock<MFile>>()},
          stream{std::make_shared<StrictMock<MStream>>()}
    {
        EXPECT_CALL(*stream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*stream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(false));

        EXPECT_CALL(*buffer, getBuffer)
            .Times(AnyNumber())
            .WillRepeatedly(Return(reinterpret_cast<hipStream_t>(0xFEFEFEFE)));
    }
    StrictMock<MHip>    mhip;
    StrictMock<MSys>    msys;
    shared_ptr<MBuffer> buffer;
    shared_ptr<MFile>   file;
    shared_ptr<MStream> stream;
};

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

TEST_F(HipFileAsyncOp, AsyncOpFallback_new_uses_pinned_host_memory)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get())).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer.get()), _));
    EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
}

TEST_F(HipFileAsyncOp, AsyncOpFallbackLimitsMaxIoSize)
{
    size_t  size              = 4_GiB;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(1_KiB);
    EXPECT_CALL(mhip, hipHostMalloc(hipFile::getMaxRwCount(), _)).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(mhip, hipHostMalloc(sizeof(AsyncOpFallback), _)).WillOnce(Return(op_data.get()));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer.get()), _));
    EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
    ASSERT_EQ(op->submitted_size, hipFile::getMaxRwCount());
}

TEST_F(HipFileAsyncOp, AsyncOpFallback_new_failure_throws_bad_alloc)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Throw(Hip::RuntimeError(hipErrorOutOfMemory)));
    EXPECT_THROW(std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{IoType::Read, file, buffer, stream,
                                                                      &size, &file_offset, &buffer_offset,
                                                                      &bytes_transferred}),
                 std::bad_alloc);
}

TEST_F(HipFileAsyncOp, AsyncOpFallback_bounce_alloc_failure_throws)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    EXPECT_CALL(mhip, hipHostMalloc)
        .WillOnce(Return(op_data.get()))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorOutOfMemory)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    EXPECT_THROW(std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{IoType::Read, file, buffer, stream,
                                                                      &size, &file_offset, &buffer_offset,
                                                                      &bytes_transferred}),
                 Hip::RuntimeError);
}

TEST_F(HipFileAsyncOp, AsyncOpFallback_bounce_buffer_deleter_failure_calls_syslog)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get())).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer.get()), _));
    EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    EXPECT_CALL(msys, syslog);
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
}

TEST_F(HipFileAsyncOp, AsyncOpFallback_delete_failure_calls_syslog)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get())).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer.get()), _));
    EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidValue)));
    EXPECT_CALL(msys, syslog);
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
}

struct HipFileAsyncOpFallbackMethods : public HipFileAsyncOp {
    HipFileAsyncOpFallbackMethods() : bounce_buffer{make_shared_void(size)}
    {
        //  make_shared uses placement new, which will not use hipHostMalloc/hipHostFree for
        //  AsyncOpFallback
        EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(bounce_buffer.get()));
        EXPECT_CALL(mhip, hipHostGetDevicePointer).WillOnce(Return(bounce_buffer_dev_ptr));
        op = std::make_shared<AsyncOpFallback>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                               &buffer_offset, &bytes_transferred);
    }
    ~HipFileAsyncOpFallbackMethods() override
    {
        EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())));
    }
    void                            *bounce_buffer_dev_ptr = reinterpret_cast<void *>(0xDECDECDE);
    size_t                           size                  = 100;
    hoff_t                           file_offset           = 0;
    hoff_t                           buffer_offset         = 0;
    ssize_t                          bytes_transferred     = 0;
    std::shared_ptr<void>            bounce_buffer;
    std::shared_ptr<AsyncOpFallback> op;
};

TEST_F(HipFileAsyncOpFallbackMethods, bounceBufferHostPtr_returns_pointer)
{
    ASSERT_EQ(op->bounceBufferHostPtr(), bounce_buffer.get());
}

TEST_F(HipFileAsyncOpFallbackMethods, devPtr_calls_hipHostGetDevicePointer)
{
    void *addr = reinterpret_cast<void *>(0xABACADBA);
    EXPECT_CALL(mhip, hipHostGetDevicePointer).WillOnce(Return(addr));
    ASSERT_EQ(op->devPtr(), addr);
}

struct HipFileAsyncMonitor : HipFileAsyncOp {
    AsyncMonitor monitor;
};

TEST_F(HipFileAsyncMonitor, addOp_and_completeOp_with_valid_params_works)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op = std::make_shared<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                           &buffer_offset, &bytes_transferred);

    monitor.addOp(op);
    EXPECT_NO_THROW(monitor.completeOp(op.get()));
}

TEST_F(HipFileAsyncMonitor, completeOp_with_invalid_op_throws)
{
    EXPECT_THROW(monitor.completeOp(reinterpret_cast<AsyncOp *>(0xDEADBEEF)), std::invalid_argument);
}

TEST_F(HipFileAsyncMonitor, addOp_without_completeOp_prints_error_on_AsyncMonitor_destruction)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op = std::make_unique<AsyncOp>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                           &buffer_offset, &bytes_transferred);
    monitor.addOp(std::move(op));
    EXPECT_CALL(msys, syslog);
}

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

struct FallbackAsyncIO : public HipFileOpened, public ::testing::WithParamInterface<IoType> {
    FallbackAsyncIO()
        : mfile{std::make_shared<StrictMock<MFile>>()}, mbuffer{std::make_shared<StrictMock<MBuffer>>()},
          mstream{std::make_shared<StrictMock<MStream>>()}, io_type{IoType::Read}, size{1024 * 1024},
          file_offset{0}, buffer_offset{0}, bytes_written{0}
    {
    }
    void SetUp() override
    {
        io_type = GetParam();
    }
    StrictMock<MHip>                     mhip;
    StrictMock<MSys>                     msys;
    std::shared_ptr<StrictMock<MFile>>   mfile;
    std::shared_ptr<StrictMock<MBuffer>> mbuffer;
    std::shared_ptr<StrictMock<MStream>> mstream;
    IoType                               io_type;
    size_t                               size;
    hoff_t                               file_offset;
    hoff_t                               buffer_offset;
    ssize_t                              bytes_written;
};

TEST_P(FallbackAsyncIO, bufferOffsetNegativeReturnsError)
{
    buffer_offset = -1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, bufferOffsetTooLargeReturnsError)
{
    buffer_offset = 1024 * 1024;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(buffer_offset));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, sizeTooLargeReturnsError)
{
    size = 1024 * 1024 + 1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, paramsValidUsesLimitedSize)
{
    size = 4_GiB;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(2_GiB));
    // Use GPU mismatch to exit function early. These functions are only called because the size was limited
    // and now does not exceed the buffer size.
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(1));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, fileOffsetNegativeReturnsError)
{
    file_offset = -1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, mismatchingGpuIdsThrows)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(1));
    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 std::invalid_argument);
}

TEST_P(FallbackAsyncIO, attemptToQueueCleanupOnStreamSubmissionFailure)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_CALL(*mbuffer, getGpuId).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<hipStream_t>(0xBADBADBB)));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(false));
    EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
    EXPECT_CALL(*mstream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(false));

    auto op_data = malloc(sizeof(AsyncOpFallback));
    ASSERT_NE(op_data, nullptr);
    auto bounce_buffer = malloc(size);
    ASSERT_NE(bounce_buffer, nullptr);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data)).WillOnce(Return(bounce_buffer));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDEBBBBBB)));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(op_data), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDE000000)));
    EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(Return(1024));
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    EXPECT_CALL(mhip, hipLaunchHostFunc).WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 Hip::RuntimeError);
    // Call to allow destruction of the op and bounce buffer
    async_io_cleanup(op_data);
    // Sleep to allow cleanup thread to destruct op
    std::this_thread::sleep_for(500ms);
}

INSTANTIATE_TEST_SUITE_P(FallbackAsyncIOSuite, FallbackAsyncIO,
                         ::testing::Values(IoType::Read, IoType::Write));

struct AsyncIoOp : public ::testing::Test {

    AsyncIoOp()
        : mfile{std::make_shared<StrictMock<MFile>>()}, mbuffer{std::make_shared<StrictMock<MBuffer>>()},
          mstream{std::make_shared<StrictMock<MStream>>()}
    {
    }
    void SetUp() override
    {
        op_data       = make_shared_void(sizeof(AsyncOpFallback));
        bounce_buffer = make_shared_void(size);
        EXPECT_CALL(mhip, hipHostMalloc)
            .WillOnce(Return(op_data.get()))
            .WillOnce(Return(bounce_buffer.get()));
        EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(bounce_buffer.get()), _));
        EXPECT_CALL(mhip, hipHostFree(Eq(bounce_buffer.get())));
        EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
        EXPECT_CALL(*mstream, fixedBufferOffset)
            .Times(AnyNumber())
            .WillRepeatedly(Return(fixed_buffer_offset));
        EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(fixed_file_offset));
        EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(fixed_io_size));
        EXPECT_CALL(*mstream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(true));
        EXPECT_CALL(*mbuffer, getBuffer);
        op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback(
            io_type, mfile, mbuffer, mstream, &size, &file_offset, &buffer_offset, &bytes_transferred));
    }
    StrictMock<MHip>                     mhip;
    StrictMock<MSys>                     msys;
    StrictMock<MAsyncMonitor>            masync_monitor;
    std::shared_ptr<StrictMock<MFile>>   mfile;
    std::shared_ptr<StrictMock<MBuffer>> mbuffer;
    std::shared_ptr<StrictMock<MStream>> mstream;
    IoType                               io_type             = IoType::Read;
    size_t                               size                = 1_MiB;
    hoff_t                               file_offset         = 0;
    hoff_t                               buffer_offset       = 0;
    ssize_t                              bytes_transferred   = 0;
    bool                                 fixed_buffer_offset = false;
    bool                                 fixed_file_offset   = false;
    bool                                 fixed_io_size       = false;
    std::shared_ptr<void>                op_data;
    std::shared_ptr<void>                bounce_buffer;
    std::shared_ptr<AsyncOpFallback>     op;
};

TEST_F(AsyncIoOp, bindIncreasingSizeReturnsError)
{
    size += 1;
    async_io_bind_params(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

TEST_F(AsyncIoOp, bindInvalidRegionReturnsError)
{
    buffer_offset = 1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    async_io_bind_params(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

TEST_F(AsyncIoOp, bindAllParamsAreFixedAfter)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    async_io_bind_params(op.get());
    auto current_buffer_offset = std::get<const hoff_t>(op->buffer_offset);
    (void)current_buffer_offset;
    auto current_file_offset = std::get<const hoff_t>(op->file_offset);
    (void)current_file_offset;
    auto current_io_size = std::get<size_t>(op->size);
    (void)current_io_size;
}

struct AsyncIoOpCleanup : public AsyncIoOp {
    void SetUp() override
    {
        fixed_buffer_offset = true;
        fixed_file_offset   = true;
        fixed_io_size       = true;
        AsyncIoOp::SetUp();
    }
};

TEST_F(AsyncIoOpCleanup, cleanupBytesTransferredIsUpdated)
{
    op->bytes_transferred_internal = 1000;
    EXPECT_CALL(masync_monitor, completeOp);
    async_io_cleanup(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, *op->bytes_transferred);
}

TEST_F(AsyncIoOpCleanup, cleanupInvalidOpSetsError)
{
    op->bytes_transferred_internal = 1000;
    EXPECT_CALL(masync_monitor, completeOp).WillOnce(Throw(std::invalid_argument("error")));
    async_io_cleanup(op.get());
    ASSERT_EQ(*op->bytes_transferred, -hipFileInternalError);
}

struct AsyncIoOpLimitedSize : public AsyncIoOp {
    void SetUp() override
    {
        size = hipFile::getMaxRwCount() + 1;
        AsyncIoOp::SetUp();
    }
};

TEST_F(AsyncIoOpLimitedSize, bindLimitsSize)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(hipFile::getMaxRwCount()));
    async_io_bind_params(op.get());
    ASSERT_EQ(std::get<size_t>(op->size), hipFile::getMaxRwCount());
}

struct AsyncIoOpBindParams {
    IoType io_type;
    bool   fixed_buffer_offset;
    bool   fixed_file_offset;
    bool   fixed_io_size;
};

struct AsyncIoOpWithParams : public AsyncIoOp, public ::testing::WithParamInterface<AsyncIoOpBindParams> {
    void SetUp() override
    {
        auto [_io_type, _fixed_buffer_offset, _fixed_file_offset, _fixed_io_size] = GetParam();
        io_type                                                                   = _io_type;
        fixed_buffer_offset                                                       = _fixed_buffer_offset;
        fixed_file_offset                                                         = _fixed_file_offset;
        fixed_io_size                                                             = _fixed_io_size;
        AsyncIoOp::SetUp();
    }
};

TEST_P(AsyncIoOpWithParams, cpuCopyWillReturnEarlyOnPreviousError)
{
    op->bytes_transferred_internal = -1;
    // No other mocks will be called
    async_io_cpu_copy(op.get());
}

TEST_P(AsyncIoOpWithParams, cpuIOReturnsFullSize)
{
    if (io_type == IoType::Read) {
        EXPECT_CALL(msys, pread).WillOnce(Return(size));
    }
    else {
        EXPECT_CALL(msys, pwrite).WillOnce(Return(size));
    }
    EXPECT_CALL(*mfile, bufferedFd);
    async_io_cpu_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, size);
}

TEST_P(AsyncIoOpWithParams, cpuCopyReadPreadPwriteErrorReturnsError)
{
    if (io_type == IoType::Read) {
        EXPECT_CALL(msys, pread).WillOnce(Throw(std::system_error(3, std::generic_category())));
    }
    else {
        EXPECT_CALL(msys, pwrite).WillOnce(Throw(std::system_error(3, std::generic_category())));
    }
    EXPECT_CALL(*mfile, bufferedFd);
    async_io_cpu_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -1);
}

TEST_P(AsyncIoOpWithParams, cpuCopyReadPreadPwriteRetriesOnEINTR)
{
    if (io_type == IoType::Read) {
        EXPECT_CALL(msys, pread)
            .WillOnce(Throw(std::system_error(EINTR, std::generic_category())))
            .WillOnce(Return(size));
    }
    else {
        EXPECT_CALL(msys, pwrite)
            .WillOnce(Throw(std::system_error(EINTR, std::generic_category())))
            .WillOnce(Return(size));
    }
    EXPECT_CALL(*mfile, bufferedFd).Times(2);
    async_io_cpu_copy(op.get());

    ASSERT_EQ(op->bytes_transferred_internal, size);
}

INSTANTIATE_TEST_SUITE_P(AsyncIoOpWithParamsSuite, AsyncIoOpWithParams,
                         ::testing::Values(AsyncIoOpBindParams{IoType::Read, true, true, true},
                                           AsyncIoOpBindParams{IoType::Write, true, true, true}));

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
