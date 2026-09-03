/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "backend/asyncop-fallback.h"
#include "backend/fallback.h"
#include "backend/fastpath.h"
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
#include "state.h"
#include "util.h"

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
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get()));
    EXPECT_CALL(*stream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferSize).WillOnce(Return(size));
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
    EXPECT_CALL(mhip, hipHostMalloc(sizeof(AsyncOpFallback), _)).WillOnce(Return(op_data.get()));
    EXPECT_CALL(*stream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferSize).WillOnce(Return(1_KiB));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
    ASSERT_EQ(op->submitted_size, hipFile::getMaxRwCount());
}

TEST_F(HipFileAsyncOp, AsyncOpLimitsFixedIoSize)
{
    size_t  size              = 4_GiB;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(1_KiB);
    EXPECT_CALL(mhip, hipHostMalloc(sizeof(AsyncOpFallback), _)).WillOnce(Return(op_data.get()));
    EXPECT_CALL(*stream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferSize).WillOnce(Return(1_KiB));
    EXPECT_CALL(*stream, fixedIOSize).WillOnce(Return(true));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data.get())));
    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback{
        IoType::Read, file, buffer, stream, &size, &file_offset, &buffer_offset, &bytes_transferred});
    ASSERT_EQ(op->submitted_size, hipFile::getMaxRwCount());
    ASSERT_EQ(std::get<size_t>(op->size), hipFile::getMaxRwCount());
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

TEST_F(HipFileAsyncOp, AsyncOpFallback_delete_failure_calls_syslog)
{
    size_t  size              = 100;
    hoff_t  file_offset       = 0;
    hoff_t  buffer_offset     = 0;
    ssize_t bytes_transferred = 0;
    auto    op_data           = make_shared_void(sizeof(AsyncOpFallback));
    auto    bounce_buffer     = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get()));
    EXPECT_CALL(*stream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*stream, asyncBufferSize).WillOnce(Return(size));
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
        //  AsyncOpFallback. The bounce buffer is now owned by Stream.
        EXPECT_CALL(*stream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
        EXPECT_CALL(*stream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer_dev_ptr));
        EXPECT_CALL(*stream, asyncBufferSize).WillOnce(Return(size));
        op = std::make_shared<AsyncOpFallback>(IoType::Read, file, buffer, stream, &size, &file_offset,
                                               &buffer_offset, &bytes_transferred);
    }
    ~HipFileAsyncOpFallbackMethods() override              = default;
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
    ASSERT_EQ(op->bounce_buffer_host_ptr, bounce_buffer.get());
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
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc): freed via mocked hipHostFree on cleanup
    ASSERT_NE(op_data, nullptr);
    auto bounce_buffer = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data));
    EXPECT_CALL(*mstream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*mstream, asyncBufferDevPtr).WillOnce(Return(reinterpret_cast<void *>(0xDEBBBBBB)));
    // asyncBufferSize is called once by async_io to compute chunk_count, and once in the constructor
    EXPECT_CALL(*mstream, asyncBufferSize).Times(2).WillRepeatedly(Return(size));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(op_data), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDE000000)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(Return(1024));
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    EXPECT_CALL(mhip, hipLaunchHostFunc).WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    EXPECT_THROW(Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_written, mstream),
                 Hip::RuntimeError);
    // Call to allow destruction of the op; bounce buffer is owned by mstream, free it here
    async_io_cleanup(op_data);
    // Sleep to allow cleanup thread to destruct op
    std::this_thread::sleep_for(500ms);
}

TEST_P(FallbackAsyncIO, multipleChunksAreQueued)
{
    size               = 1_MiB;
    size_t chunk_size  = 256_KiB;
    int    chunk_count = 4;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<hipStream_t>(0xBADBADBB)));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    // All params fixed so no bind_params host function is enqueued.
    EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(true));

    auto op_data = malloc(sizeof(AsyncOpFallback));
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc): freed via mocked hipHostFree on cleanup
    ASSERT_NE(op_data, nullptr);
    auto bounce_buffer = make_shared_void(chunk_size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data));
    EXPECT_CALL(*mstream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*mstream, asyncBufferDevPtr).WillOnce(Return(reinterpret_cast<void *>(0xDEBBBBBB)));
    // asyncBufferSize is called once by async_io to compute chunk_count, and once in the constructor
    EXPECT_CALL(*mstream, asyncBufferSize).Times(2).WillRepeatedly(Return(chunk_size));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(op_data), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDE000000)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(Return(1024));
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    // Each chunk enqueues a cpu copy, a memcpy kernel, and an advance; cleanup runs once at the end.
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cpu_copy), _)).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchKernel).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_advance), _)).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    Fallback().async_io(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_written,
                        mstream);
    // The enqueued host functions are mocked and never run; free the op manually.
    // Bounce buffer is owned by mstream, free it here.
    async_io_cleanup(op_data);
    // Sleep to allow cleanup thread to destruct op
    std::this_thread::sleep_for(500ms);
}

TEST_P(FallbackAsyncIO, failoverEnqueuesGateAndCommitsOnSuccess)
{
    auto failover      = std::make_shared<AsyncFailoverState>();
    size               = 1_MiB;
    size_t chunk_size  = 256_KiB;
    int    chunk_count = 4;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<hipStream_t>(0xBADBADBB)));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(true));

    auto op_data = malloc(sizeof(AsyncOpFallback));
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc): freed via mocked hipHostFree on cleanup
    ASSERT_NE(op_data, nullptr);
    auto bounce_buffer = make_shared_void(chunk_size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data));
    EXPECT_CALL(*mstream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*mstream, asyncBufferDevPtr).WillOnce(Return(reinterpret_cast<void *>(0xDEBBBBBB)));
    EXPECT_CALL(*mstream, asyncBufferSize).Times(2).WillRepeatedly(Return(chunk_size));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(op_data), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDE000000)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(Return(1024));
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    // The gate is enqueued first, ahead of the per-chunk work and cleanup.
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_failover_gate), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cpu_copy), _)).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchKernel).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_advance), _)).Times(chunk_count);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    // The primary owns the output in the failover path; the fallback must not reset it.
    bytes_written = 4242;
    Fallback().enqueueAsyncIo(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_written,
                              mstream, failover);
    ASSERT_EQ(bytes_written, 4242);
    ASSERT_TRUE(failover->fallback_committed.load());
    async_io_cleanup(op_data);
    std::this_thread::sleep_for(500ms);
}

TEST_P(FallbackAsyncIO, failoverPartialEnqueueLeavesUncommitted)
{
    auto failover = std::make_shared<AsyncFailoverState>();
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024 * 1024));
    EXPECT_CALL(*mbuffer, getGpuId).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<hipStream_t>(0xBADBADBB)));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(true));
    EXPECT_CALL(*mstream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(true));

    auto op_data = malloc(sizeof(AsyncOpFallback));
    // NOLINTNEXTLINE(clang-analyzer-unix.Malloc): freed via mocked hipHostFree on cleanup
    ASSERT_NE(op_data, nullptr);
    auto bounce_buffer = make_shared_void(size);
    EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data));
    EXPECT_CALL(*mstream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
    EXPECT_CALL(*mstream, asyncBufferDevPtr).WillOnce(Return(reinterpret_cast<void *>(0xDEBBBBBB)));
    EXPECT_CALL(*mstream, asyncBufferSize).Times(2).WillRepeatedly(Return(size));
    EXPECT_CALL(mhip, hipHostGetDevicePointer(Eq(op_data), _))
        .WillOnce(Return(reinterpret_cast<void *>(0xDE000000)));
    EXPECT_CALL(mhip, hipHostFree(Eq(op_data))).WillOnce([](void *ptr) { free(ptr); });
    EXPECT_CALL(mhip, hipDeviceGetAttribute).WillOnce(Return(1024));
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    // Fail the gate (the first host function) to simulate a partially submitted fallback; cleanup is
    // still enqueued from the catch, but the sequence must not be marked committed.
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_failover_gate), _))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    EXPECT_THROW(Fallback().enqueueAsyncIo(io_type, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                           &bytes_written, mstream, failover),
                 Hip::RuntimeError);
    ASSERT_FALSE(failover->fallback_committed.load());
    async_io_cleanup(op_data);
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
        EXPECT_CALL(mhip, hipHostMalloc).WillOnce(Return(op_data.get()));
        EXPECT_CALL(*mstream, asyncBufferHostPtr).WillOnce(Return(bounce_buffer.get()));
        EXPECT_CALL(*mstream, asyncBufferDevPtr).WillOnce(Return(bounce_buffer.get()));
        EXPECT_CALL(*mstream, asyncBufferSize).WillOnce(Return(size));
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

TEST_F(AsyncIoOp, failoverGateArmsWhenNeededAndCommitted)
{
    op->failover                     = std::make_shared<AsyncFailoverState>();
    op->failover->fallback_needed    = true;
    op->failover->fallback_committed = true;
    op->bytes_transferred_internal   = -1;
    op->write_result                 = false;
    async_failover_gate(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, 0);
    ASSERT_TRUE(op->write_result);
}

TEST_F(AsyncIoOp, failoverGateSkipsWhenFallbackNotNeeded)
{
    op->failover                     = std::make_shared<AsyncFailoverState>();
    op->failover->fallback_needed    = false;
    op->failover->fallback_committed = true;
    op->bytes_transferred_internal   = -1;
    op->write_result                 = false;
    async_failover_gate(op.get());
    ASSERT_LT(op->bytes_transferred_internal, 0);
    ASSERT_FALSE(op->write_result);
}

TEST_F(AsyncIoOp, failoverGateSkipsWhenNotCommitted)
{
    op->failover                     = std::make_shared<AsyncFailoverState>();
    op->failover->fallback_needed    = true;
    op->failover->fallback_committed = false;
    op->bytes_transferred_internal   = -1;
    op->write_result                 = false;
    async_failover_gate(op.get());
    ASSERT_LT(op->bytes_transferred_internal, 0);
    ASSERT_FALSE(op->write_result);
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

TEST_F(AsyncIoOpCleanup, cleanupSkipsResultWriteWhenWriteResultFalse)
{
    op->bytes_transferred_internal = 1000;
    op->write_result               = false;
    *op->bytes_transferred         = 555;
    EXPECT_CALL(masync_monitor, completeOp);
    async_io_cleanup(op.get());
    ASSERT_EQ(*op->bytes_transferred, 555);
}

TEST_F(AsyncIoOpCleanup, cleanupSkipsErrorWriteWhenWriteResultFalse)
{
    op->bytes_transferred_internal = 1000;
    op->write_result               = false;
    *op->bytes_transferred         = 555;
    EXPECT_CALL(masync_monitor, completeOp).WillOnce(Throw(std::invalid_argument("error")));
    async_io_cleanup(op.get());
    ASSERT_EQ(*op->bytes_transferred, 555);
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
        op->chunk_bytes_copied = size;
        EXPECT_CALL(msys, pwrite).WillOnce(Return(size));
    }
    EXPECT_CALL(*mfile, bufferedFd);
    async_io_cpu_copy(op.get());
    ASSERT_EQ(op->chunk_bytes_copied, size);
}

TEST_P(AsyncIoOpWithParams, cpuCopyReadPreadPwriteErrorReturnsError)
{
    if (io_type == IoType::Read) {
        EXPECT_CALL(msys, pread).WillOnce(Throw(std::system_error(3, std::generic_category())));
    }
    else {
        op->chunk_bytes_copied = size;
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
        op->chunk_bytes_copied = size;
        EXPECT_CALL(msys, pwrite)
            .WillOnce(Throw(std::system_error(EINTR, std::generic_category())))
            .WillOnce(Return(size));
    }
    EXPECT_CALL(*mfile, bufferedFd).Times(2);
    async_io_cpu_copy(op.get());

    ASSERT_EQ(op->chunk_bytes_copied, size);
}

INSTANTIATE_TEST_SUITE_P(AsyncIoOpWithParamsSuite, AsyncIoOpWithParams,
                         ::testing::Values(AsyncIoOpBindParams{IoType::Read, true, true, true},
                                           AsyncIoOpBindParams{IoType::Write, true, true, true}));

struct AsyncIoCpuCopyChunkParams {
    IoType io_type;
    size_t io_size;         // Total size of the op
    size_t chunk_size;      // op->chunk_size (async buffer size)
    size_t bytes_completed; // op->bytes_transferred_internal at entry (offset already transferred)
    size_t expected_chunk;  // Bytes this single cpu_copy call should transfer
};

struct AsyncIoOpChunked : public AsyncIoOp, public ::testing::WithParamInterface<AsyncIoCpuCopyChunkParams> {
    void SetUp() override
    {
        auto params         = GetParam();
        io_type             = params.io_type;
        size                = params.io_size;
        fixed_buffer_offset = true;
        fixed_file_offset   = true;
        fixed_io_size       = true;
        AsyncIoOp::SetUp();
        op->bounce_buffer_size         = params.chunk_size;
        op->bytes_transferred_internal = static_cast<ssize_t>(params.bytes_completed);
        // The write path consumes the chunk length the kernel publishes before cpu_copy runs.
        if (io_type == IoType::Write) {
            op->chunk_bytes_copied = params.expected_chunk;
        }
    }
};

// cpu_copy transfers only the current chunk of a larger IO, at the file offset implied by the bytes
// already completed, and reports the chunk length back through chunk_bytes_copied.
TEST_P(AsyncIoOpChunked, cpuCopyTransfersSingleChunkOfLargerIo)
{
    auto        params          = GetParam();
    void       *bounce_host     = op->bounce_buffer_host_ptr;
    const off_t expected_offset = file_offset + static_cast<off_t>(params.bytes_completed);
    EXPECT_CALL(*mfile, bufferedFd).WillOnce(Return(7));
    if (io_type == IoType::Read) {
        EXPECT_CALL(msys, pread(Eq(7), Eq(bounce_host), Eq(params.expected_chunk), Eq(expected_offset)))
            .WillOnce(Return(params.expected_chunk));
    }
    else {
        EXPECT_CALL(msys, pwrite(Eq(7), Eq(bounce_host), Eq(params.expected_chunk), Eq(expected_offset)))
            .WillOnce(Return(params.expected_chunk));
    }
    async_io_cpu_copy(op.get());
    ASSERT_EQ(op->chunk_bytes_copied, params.expected_chunk);
}

INSTANTIATE_TEST_SUITE_P(
    AsyncIoOpChunkedSuite, AsyncIoOpChunked,
    ::testing::Values(
        // Read: first, middle, and clamped-tail chunks of a 1 MiB IO with a 256 KiB buffer
        AsyncIoCpuCopyChunkParams{IoType::Read, 1_MiB, 256_KiB, 0, 256_KiB},
        AsyncIoCpuCopyChunkParams{IoType::Read, 1_MiB, 256_KiB, 512_KiB, 256_KiB},
        AsyncIoCpuCopyChunkParams{IoType::Read, 1_MiB, 256_KiB, 896_KiB, 128_KiB},
        // Write: middle and clamped-tail chunks; the length comes from chunk_bytes_copied
        AsyncIoCpuCopyChunkParams{IoType::Write, 1_MiB, 256_KiB, 512_KiB, 256_KiB},
        AsyncIoCpuCopyChunkParams{IoType::Write, 1_MiB, 256_KiB, 896_KiB, 128_KiB}),
    [](const testing::TestParamInfo<AsyncIoOpChunked::ParamType> &param_info) {
        const auto &p    = param_info.param;
        std::string name = p.io_type == IoType::Read ? "Read" : "Write";
        name += "_chunk" + std::to_string(p.chunk_size);
        name += "_at" + std::to_string(p.bytes_completed);
        return name;
    });

struct FastpathAsyncIO : public HipFileOpened {
    FastpathAsyncIO()
        : mfile{std::make_shared<StrictMock<MFile>>()}, mbuffer{std::make_shared<StrictMock<MBuffer>>()},
          mstream{std::make_shared<StrictMock<MStream>>()}, size{1024 * 1024}, file_offset{0},
          buffer_offset{0}, bytes_transferred{0}
    {
    }
    void SetUpStreamFlags()
    {
        EXPECT_CALL(*mstream, fixedBufferOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*mstream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*mbuffer, getBuffer)
            .Times(AnyNumber())
            .WillRepeatedly(Return(reinterpret_cast<hipStream_t>(0xFEFEFEFE)));
    }
    StrictMock<MHip>                     mhip;
    StrictMock<MSys>                     msys;
    std::shared_ptr<StrictMock<MFile>>   mfile;
    std::shared_ptr<StrictMock<MBuffer>> mbuffer;
    std::shared_ptr<StrictMock<MStream>> mstream;
    size_t                               size;
    hoff_t                               file_offset;
    hoff_t                               buffer_offset;
    ssize_t                              bytes_transferred;
};

TEST_F(FastpathAsyncIO, bufferOffsetNegativeThrows)
{
    buffer_offset = -1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 std::invalid_argument);
}

TEST_F(FastpathAsyncIO, bufferOffsetTooLargeThrows)
{
    buffer_offset = static_cast<hoff_t>(size);
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 std::invalid_argument);
}

TEST_F(FastpathAsyncIO, fileOffsetNegativeThrows)
{
    file_offset = -1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 std::invalid_argument);
}

TEST_F(FastpathAsyncIO, mismatchingGpuIdsThrows)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(1));
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 std::invalid_argument);
}

TEST_F(FastpathAsyncIO, validParamsEnqueuesCopyAndCleanup)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(2);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));
    Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_transferred,
                        mstream);
}

TEST_F(FastpathAsyncIO, zeroSizeEnqueuesCopyAndCleanup)
{
    size = 0;
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(1024));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(2);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));
    Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_transferred,
                        mstream);
}

TEST_F(FastpathAsyncIO, attemptToQueueCleanupOnStreamSubmissionFailure)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 Hip::RuntimeError);
}

TEST_F(FastpathAsyncIO, syslogCalledWhenCleanupEnqueueAlsoFails)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(AnyNumber());
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _))
        .WillOnce(Throw(Hip::RuntimeError(hipErrorInvalidHandle)));
    EXPECT_CALL(msys, syslog);
    EXPECT_THROW(Fastpath().async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                     &bytes_transferred, mstream),
                 Hip::RuntimeError);
}

TEST_F(FastpathAsyncIO, orchestratorEnqueuesFallbackWhenScoreNonNegative)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    auto                      m_fallback = std::make_shared<StrictMock<MBackend>>();
    Fastpath                  fastpath;
    fastpath.register_fallback_backend(m_fallback);

    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(2);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    EXPECT_CALL(*m_fallback, score).WillOnce(Return(0));
    EXPECT_CALL(*m_fallback, enqueueAsyncIo);

    fastpath.async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_transferred,
                      mstream);
}

TEST_F(FastpathAsyncIO, orchestratorSkipsFallbackWhenScoreNegative)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    auto                      m_fallback = std::make_shared<StrictMock<MBackend>>();
    Fastpath                  fastpath;
    fastpath.register_fallback_backend(m_fallback);

    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(2);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    // Fallback refuses the IO, so no fallback jobs are enqueued.
    EXPECT_CALL(*m_fallback, score).WillOnce(Return(-1));

    fastpath.async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset, &bytes_transferred,
                      mstream);
}

TEST_F(FastpathAsyncIO, orchestratorSwallowsFallbackEnqueueFailure)
{
    SetUpStreamFlags();
    StrictMock<MAsyncMonitor> masync_monitor;
    auto                      m_fallback = std::make_shared<StrictMock<MBackend>>();
    Fastpath                  fastpath;
    fastpath.register_fallback_backend(m_fallback);

    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mbuffer, getGpuId).WillOnce(Return(0));
    EXPECT_CALL(*mstream, getHipDevice).WillOnce(Return(0));
    EXPECT_CALL(masync_monitor, addOp);
    EXPECT_CALL(*mstream, getLock);
    EXPECT_CALL(*mstream, getHipStream).Times(2);
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_fastpath_copy), _));
    EXPECT_CALL(mhip, hipLaunchHostFunc(_, Eq(&async_io_cleanup), _));

    // The primary jobs are already queued, so a failed fallback enqueue is logged
    // and swallowed rather than propagated.
    EXPECT_CALL(*m_fallback, score).WillOnce(Return(0));
    EXPECT_CALL(*m_fallback, enqueueAsyncIo).WillOnce(Throw(std::runtime_error("boom")));
    EXPECT_CALL(msys, syslog);

    ASSERT_NO_THROW(fastpath.async_io(IoType::Read, mfile, mbuffer, &size, &file_offset, &buffer_offset,
                                      &bytes_transferred, mstream));
}

struct AsyncFastpathCopyParams {
    IoType io_type;
    bool   fixed_buffer_offset;
    bool   fixed_file_offset;
    bool   fixed_io_size;
};

struct AsyncFastpathCopyOp : public ::testing::Test,
                             public ::testing::WithParamInterface<AsyncFastpathCopyParams> {
    AsyncFastpathCopyOp()
        : mfile{std::make_shared<StrictMock<MFile>>()}, mbuffer{std::make_shared<StrictMock<MBuffer>>()},
          mstream{std::make_shared<StrictMock<MStream>>()}
    {
    }
    void SetUp() override
    {
        auto [_io_type, _fixed_buffer_offset, _fixed_file_offset, _fixed_io_size] = GetParam();
        io_type                                                                   = _io_type;
        fixed_buffer_offset                                                       = _fixed_buffer_offset;
        fixed_file_offset                                                         = _fixed_file_offset;
        fixed_io_size                                                             = _fixed_io_size;

        EXPECT_CALL(*mstream, fixedBufferOffset)
            .Times(AnyNumber())
            .WillRepeatedly(Return(fixed_buffer_offset));
        EXPECT_CALL(*mstream, fixedFileOffset).Times(AnyNumber()).WillRepeatedly(Return(fixed_file_offset));
        EXPECT_CALL(*mstream, fixedIOSize).Times(AnyNumber()).WillRepeatedly(Return(fixed_io_size));
        EXPECT_CALL(*mstream, pageAligned).Times(AnyNumber()).WillRepeatedly(Return(false));
        EXPECT_CALL(*mbuffer, getBuffer)
            .Times(AnyNumber())
            .WillRepeatedly(Return(reinterpret_cast<hipStream_t>(0xFEFEFEFE)));
        EXPECT_CALL(mhip, hipInit).Times(AnyNumber());

        op = std::make_shared<AsyncOp>(io_type, mfile, mbuffer, mstream, &size, &file_offset, &buffer_offset,
                                       &bytes_transferred);
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
    std::shared_ptr<AsyncOp>             op;
};

TEST_P(AsyncFastpathCopyOp, invalidRegionSetsError)
{
    // Overwrite the buffer_offset variant directly so both fixed and non-fixed
    // cases present the invalid value when async_fastpath_copy runs.
    op->buffer_offset.emplace<const hoff_t>(static_cast<hoff_t>(size));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

TEST_P(AsyncFastpathCopyOp, ioReturnsFullSize)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Return(size));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Return(size));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, static_cast<ssize_t>(size));
}

TEST_P(AsyncFastpathCopyOp, hipExceptionSetsError)
{
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInternalError);
}

TEST_P(AsyncFastpathCopyOp, sizeClampedToMaxRwCount)
{
    // Overwrite the size variant directly so both fixed and non-fixed cases
    // present the large size to async_fastpath_copy.
    op->size = hipFile::getMaxRwCount() + 1;
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(hipFile::getMaxRwCount() + 1));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead(_, _, Eq(hipFile::getMaxRwCount()), _))
            .WillOnce(Return(hipFile::getMaxRwCount()));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite(_, _, Eq(hipFile::getMaxRwCount()), _))
            .WillOnce(Return(hipFile::getMaxRwCount()));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, static_cast<ssize_t>(hipFile::getMaxRwCount()));
}

TEST_P(AsyncFastpathCopyOp, eligibleSystemErrorSetsFallbackNeeded)
{
    op->failover = std::make_shared<AsyncFailoverState>();
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(std::system_error(ENODEV, std::generic_category())));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite)
            .WillOnce(Throw(std::system_error(EREMOTEIO, std::generic_category())));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInternalError);
    ASSERT_TRUE(op->failover->fallback_needed);
}

TEST_P(AsyncFastpathCopyOp, ineligibleSystemErrorLeavesFallbackNotNeeded)
{
    op->failover = std::make_shared<AsyncFailoverState>();
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(std::system_error(EIO, std::generic_category())));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(std::system_error(EIO, std::generic_category())));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInternalError);
    ASSERT_FALSE(op->failover->fallback_needed);
}

TEST_P(AsyncFastpathCopyOp, successLeavesFallbackNotNeeded)
{
    op->failover = std::make_shared<AsyncFailoverState>();
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(size));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(42));
    if (io_type == IoType::Read) {
        EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Return(size));
    }
    else {
        EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Return(size));
    }
    async_fastpath_copy(op.get());
    ASSERT_EQ(op->bytes_transferred_internal, static_cast<ssize_t>(size));
    ASSERT_FALSE(op->failover->fallback_needed);
}

INSTANTIATE_TEST_SUITE_P(AsyncFastpathCopyOpSuite, AsyncFastpathCopyOp,
                         ::testing::Values(AsyncFastpathCopyParams{IoType::Read, true, true, true},
                                           AsyncFastpathCopyParams{IoType::Write, true, true, true},
                                           AsyncFastpathCopyParams{IoType::Read, false, false, false},
                                           AsyncFastpathCopyParams{IoType::Write, false, false, false}));

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
