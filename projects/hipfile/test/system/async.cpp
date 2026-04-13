/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef __HIP_PLATFORM_AMD__
#include "backend/asyncop-fallback.h"
#include "backend/memcpy-kernel.h"
#include "buffer.h"
#include "context.h"
#include "hip.h"
#include "io.h"
#include "state.h"
#include "stream.h"
#endif

#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <variant>
#include <vector>
#include <unistd.h>

#ifdef __HIP_PLATFORM_NVIDIA__
#include <cuda.h>
#endif

#ifdef __HIP_PLATFORM_AMD__
using namespace hipFile;
#endif

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF
extern SystemTestOptions test_env;

HIPFILE_WARN_NO_EXIT_DTOR_OFF
struct AsyncIoFunction {
    hipFileError_t (*function)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
};
static std::array<AsyncIoFunction, 2> asyncIOFns{
    {{hipFileReadAsync, "hipFileReadAsync"}, {hipFileWriteAsync, "hipFileWriteAsync"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

static bool
isGpuMemory(void *mem)
{
    hipPointerAttribute_t attrs;
    hipError_t            err = hipPointerGetAttributes(&attrs, mem);

#ifdef __HIP_PLATFORM_NVIDIA__
    // NVIDIA doesn't support pointers allocated outside of runtime
    if (err == hipErrorInvalidValue) {
        return false;
    }
#endif
    assert(err == hipSuccess);
    return attrs.type == hipMemoryTypeDevice;
}

static std::vector<uint8_t>
copyGpuMemory(void *gpu_mem, hoff_t gpu_mem_offset, size_t region_size)
{
    std::vector<uint8_t> mem_region(region_size);
    assert(hipMemcpy(mem_region.data(),
                     reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(gpu_mem) +
                                              static_cast<size_t>(gpu_mem_offset)),
                     region_size, hipMemcpyDeviceToHost) == hipSuccess);
    assert(hipStreamSynchronize(nullptr) == hipSuccess);
    return mem_region;
}

static void
assertMemoryRegionsMatch(void *mem1, hoff_t mem1_offset, void *mem2, hoff_t mem2_offset, size_t region_size)
{
    std::vector<uint8_t> mem1_v;
    std::vector<uint8_t> mem2_v;
    if (isGpuMemory(mem1)) {
        mem1_v = copyGpuMemory(mem1, mem1_offset, region_size);
        mem1   = mem1_v.data();
    }
    if (isGpuMemory(mem2)) {
        mem2_v = copyGpuMemory(mem2, mem2_offset, region_size);
        mem2   = mem2_v.data();
    }
    assert(std::memcmp(mem1, mem2, region_size) == 0);
}

static void
assertFileAndMemoryRegionsMatch(void *mem, hoff_t mem_offset, int fd, hoff_t fd_offset, size_t region_size)
{
    assert(fd_offset >= 0);
    auto file_region = std::vector<uint8_t>(region_size);

    ssize_t rv = pread(fd, file_region.data(), region_size, fd_offset);
    assert(rv > 0 && static_cast<size_t>(rv) == region_size);

    assertMemoryRegionsMatch(file_region.data(), 0, mem, mem_offset, region_size);
}

static void
assertZeroedMemRegion(void *mem, hoff_t mem_offset, size_t region_size)
{
    std::vector<uint8_t> mem_v;
    if (isGpuMemory(mem)) {
        mem_v = copyGpuMemory(mem, mem_offset, region_size);
        mem   = mem_v.data();
    }
    for (size_t i = 0; i < region_size; ++i) {
        assert(reinterpret_cast<uint8_t *>(mem)[i] == 0);
    }
}

static void
assertZeroedFileRegion(int fd, hoff_t fd_offset, size_t region_size)
{
    assert(fd_offset >= 0);
    auto    file_region = std::vector<uint8_t>(region_size);
    ssize_t rv          = pread(fd, file_region.data(), region_size, fd_offset);
    assert(rv > 0 && static_cast<size_t>(rv) == region_size);
    for (size_t i = 0; i < region_size; ++i) {
        assert(file_region.data()[i] == 0);
    }
}

static void
randomizeMemoryRegion(void *mem, hoff_t offset, size_t region_size)
{
    ssize_t rv;
    int     rand_fd = open("/dev/urandom", O_RDONLY);
    assert(rand_fd != -1);
    if (isGpuMemory(mem)) {
        std::vector<uint8_t> mem_v(region_size);
        rv = read(rand_fd, mem_v.data(), region_size);
        assert(rv > 0 && static_cast<size_t>(rv) == region_size);
        assert(hipMemcpy(
                   reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                   mem_v.data(), region_size, hipMemcpyHostToDevice) == hipSuccess);
        assert(hipStreamSynchronize(nullptr) == hipSuccess);
    }
    else {
        rv = read(rand_fd,
                  reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                  region_size);
        assert(rv > 0 && static_cast<size_t>(rv) == region_size);
    }
    assert(close(rand_fd) == 0);
}

static void
zeroMemoryRegion(void *mem, hoff_t offset, size_t region_size)
{
    if (isGpuMemory(mem)) {
        assert(hipMemset(
                   reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                   0, region_size) == hipSuccess);
        assert(hipStreamSynchronize(nullptr) == hipSuccess);
    }
    else {
        memset(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)), 0,
               region_size);
    }
}

static void
zeroFileRegion(int fd, size_t size, hoff_t offset = 0)
{
    auto    vec = std::vector<uint8_t>(size, 0);
    ssize_t rv  = pwrite(fd, vec.data(), size, offset);
    assert(rv > 0 && static_cast<size_t>(rv) == size);
}

static void
randomizeFileRegion(int fd, size_t size, hoff_t offset = 0)
{
    auto vec = std::vector<uint8_t>(size, 0);
    randomizeMemoryRegion(vec.data(), 0, size);
    ssize_t rv = pwrite(fd, vec.data(), size, offset);
    assert(rv > 0 && static_cast<size_t>(rv) == size);
}

#ifdef __HIP_PLATFORM_AMD__
class HipAsyncMemcpyKernel : public ::testing::Test {
public:
    HipAsyncMemcpyKernel() : tf{test_env.ais_capable_dir}
    {
    }
    void SetUp() override
    {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipMalloc(&dev_ptr, buffer_size), hipSuccess);
        ASSERT_EQ(hipFileBufRegister(dev_ptr, buffer_size, 0), HIPFILE_SUCCESS);
        hipFileDescr_t d{hipFileHandleTypeOpaqueFD, {tf.fd}, nullptr};
        ASSERT_EQ(hipFileHandleRegister(&fh, &d), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamCreateWithFlags(&hip_stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(hip_stream, 0xf), HIPFILE_SUCCESS);
        auto [_file, _buffer, _stream] =
            Context<DriverState>::get()->getFileBufferAndStream(fh, dev_ptr, hip_stream);
        file              = _file;
        buffer            = _buffer;
        stream            = _stream;
        op                = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback(
            io_type, file, buffer, stream, &io_size, &file_offset, &buffer_offset, &bytes_transferred));
        bytes_transferred = static_cast<ssize_t>(io_size);
        if (io_type == IoType::Read) {
            // For a read, bytes_transferred_internal needs to be set to simulate that the read from disk
            // occurred. We fill the source (CPU bounce buffer) with random data and memset the destination
            // (GPU buffer) to zero.
            randomizeMemoryRegion(op->bounceBufferHostPtr(), 0, io_size);
            op->bytes_transferred_internal = static_cast<ssize_t>(io_size);
            ASSERT_EQ(hipMemset(op->gpu_buffer, 0, buffer_size), hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        }
        else {
            // For a write, we fill the source (GPU bounce buffer) with random data and memset the destination
            // (CPU bounce buffer) to zero.
            randomizeMemoryRegion(op->gpu_buffer, 0, buffer_size);
            memset(op->bounceBufferHostPtr(), 0, io_size);
        }
        op_dev_ptr            = op->devPtr();
        kernel_args[0]        = {&op_dev_ptr};
        max_threads_per_block = Context<Hip>::get()->hipDeviceGetAttribute(
            hipDeviceAttributeMaxThreadsPerBlock, buffer->getGpuId());
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(hip_stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(hip_stream), hipSuccess);
        ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
    }
    void                            *dev_ptr{};
    Tmpfile                          tf;
    hipFileHandle_t                  fh{};
    IoType                           io_type           = IoType::Read;
    size_t                           file_size         = 4_KiB;
    size_t                           buffer_size       = 4_KiB;
    size_t                           io_size           = 4_KiB;
    hoff_t                           file_offset       = 0;
    hoff_t                           buffer_offset     = 0;
    ssize_t                          bytes_transferred = 0;
    hipStream_t                      hip_stream{};
    std::shared_ptr<IFile>           file;
    std::shared_ptr<IBuffer>         buffer;
    std::shared_ptr<IStream>         stream;
    std::shared_ptr<AsyncOpFallback> op;
    void                            *op_dev_ptr{};
    void                            *kernel_args[1]{};
    int                              max_threads_per_block = 0;
};

TEST_F(HipAsyncMemcpyKernel, unfixedSizeReturnsInval)
{
    op->size = &io_size;
    Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                         dim3(static_cast<uint32_t>(max_threads_per_block)), kernel_args, 0,
                                         op->stream->getHipStream());
    Context<Hip>::get()->hipStreamSynchronize(op->stream->getHipStream());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

TEST_F(HipAsyncMemcpyKernel, unfixedBufferOffsetReturnsInval)
{
    op->buffer_offset = &buffer_offset;
    Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                         dim3(static_cast<uint32_t>(max_threads_per_block)), kernel_args, 0,
                                         op->stream->getHipStream());
    Context<Hip>::get()->hipStreamSynchronize(op->stream->getHipStream());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

TEST_F(HipAsyncMemcpyKernel, nullCpuBufferDevPointerReturnsInval)
{
    op->bounce_buffer_dev_ptr = nullptr;
    Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                         dim3(static_cast<uint32_t>(max_threads_per_block)), kernel_args, 0,
                                         op->stream->getHipStream());
    Context<Hip>::get()->hipStreamSynchronize(op->stream->getHipStream());
    ASSERT_EQ(op->bytes_transferred_internal, -hipFileInvalidValue);
}

struct AsyncMemcpyKernelTestParams {
    IoType  io_type;
    size_t  file_size;
    size_t  buffer_size;
    size_t  io_size;
    hoff_t  file_offset;
    hoff_t  buffer_offset;
    ssize_t bytes_transferred_start;
    ssize_t bytes_transferred_result;
};

class HipAsyncMemcpyKernelWithParams : public HipAsyncMemcpyKernel,
                                       public ::testing::WithParamInterface<AsyncMemcpyKernelTestParams> {
public:
    void SetUp() override
    {
        auto [_io_type, _file_size, _buffer_size, _io_size, _file_offset, _buffer_offset,
              _bytes_transferred_start, _bytes_transferred_result] = GetParam();
        io_type                                                    = _io_type;
        file_size                                                  = _file_size;
        buffer_size                                                = _buffer_size;
        io_size                                                    = _io_size;
        file_offset                                                = _file_offset;
        buffer_offset                                              = _buffer_offset;
        bytes_transferred_start                                    = _bytes_transferred_start;
        bytes_transferred_result                                   = _bytes_transferred_result;
        HipAsyncMemcpyKernel::SetUp();
        op->bytes_transferred_internal = bytes_transferred_start;
    }
    void TearDown() override
    {
        HipAsyncMemcpyKernel::TearDown();
    }
    ssize_t bytes_transferred_start;
    ssize_t bytes_transferred_result;
};

TEST_P(HipAsyncMemcpyKernelWithParams, verifyIoRegions)
{
    Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                         dim3(static_cast<uint32_t>(max_threads_per_block)), kernel_args, 0,
                                         op->stream->getHipStream());
    Context<Hip>::get()->hipStreamSynchronize(op->stream->getHipStream());
    ASSERT_EQ(op->bytes_transferred_internal, bytes_transferred_result);
    if (op->bytes_transferred_internal > 0) {
        if (io_type == IoType::Read) {
            assertZeroedMemRegion(op->gpu_buffer, 0, static_cast<size_t>(buffer_offset));
        }
        assertMemoryRegionsMatch(op->bounceBufferHostPtr(), 0, op->gpu_buffer, buffer_offset,
                                 static_cast<size_t>(op->bytes_transferred_internal));
        if (io_type == IoType::Read) {
            size_t end_length = buffer_size - (static_cast<size_t>(buffer_offset) +
                                               static_cast<size_t>(op->bytes_transferred_internal));
            assertZeroedMemRegion(op->gpu_buffer, buffer_offset + op->bytes_transferred_internal, end_length);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    HipAsyncMemcpyKernelSuite, HipAsyncMemcpyKernelWithParams,
    testing::Values(
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB, 0, 0, 1_MiB, 1_MiB},
        AsyncMemcpyKernelTestParams{IoType::Write, 1_MiB, 1_MiB, 1_MiB, 0, 0, 1_MiB, 1_MiB},
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB - 8_KiB, 0, 4_KiB, 1_MiB - 8_KiB,
                                    1_MiB - 8_KiB},
        AsyncMemcpyKernelTestParams{IoType::Write, 1_MiB, 1_MiB, 1_MiB - 8_KiB, 0, 4_KiB, 1_MiB - 8_KiB,
                                    1_MiB - 8_KiB},
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB - 1, 0, 0, 1_MiB - 1, 1_MiB - 1},
        AsyncMemcpyKernelTestParams{IoType::Write, 1_MiB, 1_MiB, 1_MiB - 1, 0, 0, 1_MiB - 1, 1_MiB - 1},
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB - 1, 0, 1, 1_MiB - 1, 1_MiB - 1},
        AsyncMemcpyKernelTestParams{IoType::Write, 1_MiB, 1_MiB, 1_MiB - 1, 0, 1, 1_MiB - 1, 1_MiB - 1},
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB - 2, 0, 1, 1_MiB - 2, 1_MiB - 2},
        AsyncMemcpyKernelTestParams{IoType::Write, 1_MiB, 1_MiB, 1_MiB - 2, 0, 1, 1_MiB - 2, 1_MiB - 2},
        AsyncMemcpyKernelTestParams{IoType::Read, 1_MiB, 1_MiB, 1_MiB, 0, 0, 512_KiB, 512_KiB}),
    [](const testing::TestParamInfo<HipAsyncMemcpyKernelWithParams::ParamType> &param_info) {
        auto params = param_info.param;

        std::stringstream label;
        if (params.io_type == IoType::Read) {
            label << "read";
        }
        else {
            label << "write";
        }
        label << "_" << params.file_size << "_" << params.buffer_size << "_" << params.io_size << "_"
              << params.file_offset << "_" << params.buffer_offset << "_" << params.bytes_transferred_start
              << "_" << params.bytes_transferred_result;
        return label.str();
    });
#endif

class HipAsync : public ::testing::Test {
public:
    HipAsync() : tf{test_env.ais_capable_dir}
    {
    }
    void SetUp() override
    {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipMalloc(&dev_ptr, buffer_size), hipSuccess);

        randomizeFileRegion(tf.fd, file_size);

        ASSERT_EQ(hipFileBufRegister(dev_ptr, buffer_size, 0), HIPFILE_SUCCESS);
        hipFileDescr_t d{hipFileHandleTypeOpaqueFD, {tf.fd}, nullptr};
        ASSERT_EQ(hipFileHandleRegister(&fh, &d), HIPFILE_SUCCESS);
    }

    void TearDown() override
    {
        ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
    }
    void           *dev_ptr{};
    Tmpfile         tf;
    hipFileHandle_t fh{};
    size_t          io_size           = 1_MiB;
    size_t          file_size         = 1_MiB;
    size_t          buffer_size       = 1_MiB;
    hoff_t          file_offset       = 0;
    hoff_t          buffer_offset     = 0;
    ssize_t         bytes_transferred = 0;
};

class HipAsyncStreamFixed : public HipAsync {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        zeroFileRegion(tf.fd, file_size);
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, 0xf), HIPFILE_SUCCESS);
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        HipAsync::TearDown();
    }
    hipStream_t stream;
};

TEST_F(HipAsyncStreamFixed, readRegionPastEndOfFile)
{
    file_offset += 4_KiB;
    ASSERT_EQ(
        hipFileReadAsync(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
        HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, file_size - 4_KiB);
    ASSERT_EQ(bytes_transferred, file_size - 4_KiB);
}

class HipAsyncReadWriteStreamFixedWithParams : public HipAsync,
                                               public ::testing::WithParamInterface<AsyncIoFunction> {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        auto params = GetParam();
        io_op       = params.function;
        name        = params.name;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, 0xf), HIPFILE_SUCCESS);
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        HipAsync::TearDown();
    }
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
    hipStream_t stream;
};

TEST_P(HipAsyncReadWriteStreamFixedWithParams, nullBufferBaseReturnsError)
{
    ASSERT_EQ(io_op(fh, nullptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipAsyncReadWriteStreamFixedWithParams, nullSizeReturnsError)
{
    ASSERT_EQ(io_op(fh, dev_ptr, nullptr, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipAsyncReadWriteStreamFixedWithParams, nullFileOffsetReturnsError)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, nullptr, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipAsyncReadWriteStreamFixedWithParams, nullBufferOffsetReturnsError)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, nullptr, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
}

TEST_P(HipAsyncReadWriteStreamFixedWithParams, nullBytesReadReturnsError)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, nullptr, stream),
              HipFileOpError(hipFileInvalidValue));
}

// cuFile accepts operation and returns error through bytes_transferred
TEST_P(HipAsyncReadWriteStreamFixedWithParams, ioSizeTooLargeForBuffer)
{
    io_size += 4096;
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInvalidMappingSize);
#endif
}

// cuFile accepts operation and returns error through bytes_transferred
TEST_P(HipAsyncReadWriteStreamFixedWithParams, fileOffsetNegative)
{
    file_offset = -4096;
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInvalidValue);
#endif
}

// cuFile accepts operation and returns error through bytes_transferred
TEST_P(HipAsyncReadWriteStreamFixedWithParams, bufferOffsetNegative)
{
    buffer_offset = -4096;
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInternalError);
#endif
}

// cuFile accepts operation and returns error through bytes_transferred
TEST_P(HipAsyncReadWriteStreamFixedWithParams, bufferOffsetTooLarge)
{
    buffer_offset = static_cast<hoff_t>(buffer_size) + 4096;
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInvalidMappingSize);
#endif
}

INSTANTIATE_TEST_SUITE_P(
    HipAsyncReadWriteStreamFixedWithParamsSuite, HipAsyncReadWriteStreamFixedWithParams,
    testing::ValuesIn(asyncIOFns),
    [](const testing::TestParamInfo<HipAsyncReadWriteStreamFixedWithParams::ParamType> &param_info) {
        auto params = param_info.param;
        return params.name;
    });

class HipAsyncReadWriteParamsUnchanged
    : public HipAsync,
      public ::testing::WithParamInterface<std::tuple<AsyncIoFunction, uint32_t, uint32_t, uint32_t>> {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        auto params  = GetParam();
        auto op_info = std::get<0>(params);
        io_op        = op_info.function;
        name         = op_info.name;
        if (name == "hipFileReadAsync") {
            zeroMemoryRegion(dev_ptr, 0, buffer_size);
            randomizeFileRegion(tf.fd, file_size);
        }
        else {
            zeroFileRegion(tf.fd, file_size);
            randomizeMemoryRegion(dev_ptr, 0, buffer_size);
        }
        auto flags = std::get<1>(params) | std::get<2>(params) | std::get<3>(params);
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, flags), HIPFILE_SUCCESS);
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        HipAsync::TearDown();
    }
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
    hipStream_t stream;
};

TEST_P(HipAsyncReadWriteParamsUnchanged, zeroOffsets)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
}

TEST_P(HipAsyncReadWriteParamsUnchanged, ioSizeUnaligned)
{
    io_size -= 1;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 1);
    }
    else {
        assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 1);
    }
}

TEST_P(HipAsyncReadWriteParamsUnchanged, fileOffsetAndIoSizeUnaligned)
{
    file_offset = 1;
    io_size -= 1;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 1);
    }
    else {
        assertZeroedFileRegion(tf.fd, 0, 1);
    }
}

TEST_P(HipAsyncReadWriteParamsUnchanged, bufferOffsetAndIoSizeUnaligned)
{
    buffer_offset = 1;
    io_size -= 1;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, 0, 1);
    }
    else {
        assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 1);
    }
}

TEST_P(HipAsyncReadWriteParamsUnchanged, zeroSized)
{
    io_size           = 0;
    bytes_transferred = 1;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(bytes_transferred, 0);
}

TEST_P(HipAsyncReadWriteParamsUnchanged, multipleOpsOnSameStreamAreSequential)
{

    size_t num_ios = (io_size / 8_KiB) - 1;
    io_size        = 16_KiB;
    std::vector<ssize_t> bytes_transferred_v(num_ios, 0);
    std::vector<hoff_t>  buffer_offset_v(num_ios, 0);
    std::vector<hoff_t>  file_offset_v(num_ios, 0);
    if (name == "hipFileReadAsync") {
        for (size_t i = 0; i < num_ios; ++i) {
            buffer_offset_v[i] = static_cast<hoff_t>(8_KiB) * static_cast<hoff_t>(i);
            ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset_v[i], &bytes_transferred_v[i],
                            stream),
                      HIPFILE_SUCCESS);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        for (size_t i = 0; i < num_ios; ++i) {
            ASSERT_EQ(bytes_transferred_v[i], io_size);
            size_t region_size = 8_KiB;
            if (i == num_ios - 1) {
                region_size = 16_KiB;
            }
            assertFileAndMemoryRegionsMatch(dev_ptr, static_cast<hoff_t>(i * 8_KiB), tf.fd, 0, region_size);
        }
    }
    else {
        for (size_t i = 0; i < num_ios; ++i) {
            file_offset_v[i] = static_cast<hoff_t>(8_KiB) * static_cast<hoff_t>(i);
            ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset_v[i], &buffer_offset, &bytes_transferred_v[i],
                            stream),
                      HIPFILE_SUCCESS);
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        for (size_t i = 0; i < num_ios; ++i) {
            ASSERT_EQ(bytes_transferred_v[i], io_size);
            size_t region_size = 8_KiB;
            if (i == num_ios - 1) {
                region_size = 16_KiB;
            }
            assertFileAndMemoryRegionsMatch(dev_ptr, 0, tf.fd, static_cast<hoff_t>(i * 8_KiB), region_size);
        }
    }
}

static auto
hipFileAsyncIOFlagsPowerSet()
{
    return ::testing::Combine(::testing::ValuesIn(asyncIOFns),
                              ::testing::Values(0, HIPFILE_STREAM_FIXED_BUF_OFFSET),
                              ::testing::Values(0, HIPFILE_STREAM_FIXED_FILE_OFFSET),
                              ::testing::Values(0, HIPFILE_STREAM_FIXED_FILE_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
    HipAsyncStreamUnchanged, HipAsyncReadWriteParamsUnchanged, hipFileAsyncIOFlagsPowerSet(),
    [](const testing::TestParamInfo<HipAsyncReadWriteParamsUnchanged::ParamType> &param_info) {
        auto params = param_info.param;
        auto flags  = std::get<1>(params) | std::get<2>(params) | std::get<3>(params);
        return std::get<0>(param_info.param).name + "_" + std::to_string(flags);
    });

#ifdef __HIP_PLATFORM_AMD__

class HipAsyncStreamUnfixedPaused : public HipAsync {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        int attr = 0;
        ASSERT_EQ(hipDeviceGetAttribute(&attr, hipDeviceAttributeCanUseStreamWaitValue, 0), hipSuccess);
        ASSERT_EQ(attr, 1);
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, 0), HIPFILE_SUCCESS);
        ASSERT_EQ(
            hipExtMallocWithFlags(reinterpret_cast<void **>(&flag), sizeof(uint64_t), hipMallocSignalMemory),
            hipSuccess);
        updateFlag(0);
        ASSERT_EQ(hipStreamWaitValue64(stream, flag, 1, hipStreamWaitValueEq), hipSuccess);
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        HipAsync::TearDown();
    }

    void updateFlag(uint64_t new_flag)
    {
        ASSERT_EQ(hipMemcpy(flag, &new_flag, sizeof(uint64_t), hipMemcpyHostToDevice), hipSuccess);
    }
    hipStream_t stream;
    uint64_t   *flag;
};

class HipAsyncStreamUnfixedPausedReadWrite : public HipAsyncStreamUnfixedPaused,
                                             public ::testing::WithParamInterface<AsyncIoFunction> {
public:
    void SetUp() override
    {
        HipAsyncStreamUnfixedPaused::SetUp();
        io_op = GetParam().function;
        name  = GetParam().name;
        if (name == "hipFileReadAsync") {
            zeroMemoryRegion(dev_ptr, 0, buffer_size);
            randomizeFileRegion(tf.fd, file_size);
        }
        else {
            randomizeMemoryRegion(dev_ptr, 0, buffer_size);
            zeroFileRegion(tf.fd, file_size);
        }
    }
    void TearDown() override
    {
        HipAsyncStreamUnfixedPaused::TearDown();
    }
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
};

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, smallerIOSizeIsValid)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    io_size -= 4_KiB;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 4_KiB);
    }
    else {
        assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 4_KiB);
    }
}

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, increasedIoSizeIsInvalid)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    io_size += 4096;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInvalidValue);
}

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, changedBufferOffsetCanCauseInvalidIoSize)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    buffer_offset += 4096;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInvalidValue);
}

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, changedIOSizeAndFileOffset)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    io_size -= 4096;
    file_offset += 4096;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 4_KiB);
    }
    else {
        assertZeroedFileRegion(tf.fd, 0, 4_KiB);
    }
}

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, changedIOSizeAndBufferOffset)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    io_size -= 4096;
    buffer_offset += 4096;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, 0, 4_KiB);
    }
    else {
        assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 4_KiB);
    }
}

TEST_P(HipAsyncStreamUnfixedPausedReadWrite, changedIOSizeAndBufferOffsetAndFileOffset)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    io_size -= 4096;
    buffer_offset += 4096;
    file_offset += 4096;
    updateFlag(1);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        assertZeroedMemRegion(dev_ptr, 0, 4_KiB);
    }
    else {
        assertZeroedFileRegion(tf.fd, 0, 4_KiB);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HipAsyncStreamUnfixed, HipAsyncStreamUnfixedPausedReadWrite, ::testing::ValuesIn(asyncIOFns),
    [](const testing::TestParamInfo<HipAsyncStreamUnfixedPausedReadWrite::ParamType> &param_info) {
        return param_info.param.name;
    });

#endif

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
