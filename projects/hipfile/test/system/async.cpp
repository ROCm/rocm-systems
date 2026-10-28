/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef __HIP_PLATFORM_AMD__
#include "configuration.h"
#include "context.h"
#include "hip.h"
#endif

#include "ais-capability.h"
#include "hipfile-data-ops.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <tuple>
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

class HipAsync : public ::testing::Test {
public:
    HipAsync() : tf{test_env.ais_capable_dir}
    {
    }
    void SetUp() override
    {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipMalloc(&dev_ptr, buffer_size), hipSuccess);

        HipFileDataOps::randomizeFileRegion(tf.fd, file_size);

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
        HipFileDataOps::zeroFileRegion(tf.fd, file_size);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset,
                                                    file_size - 4_KiB);
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
            HipFileDataOps::zeroMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::randomizeFileRegion(tf.fd, file_size);
        }
        else {
            HipFileDataOps::zeroFileRegion(tf.fd, file_size);
            HipFileDataOps::randomizeMemoryRegion(dev_ptr, 0, buffer_size);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
}

TEST_P(HipAsyncReadWriteParamsUnchanged, ioSizeUnaligned)
{
    io_size -= 1;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 1);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 1);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 1);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, 0, 1);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, 0, 1);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 1);
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
            HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, static_cast<hoff_t>(i * 8_KiB), tf.fd, 0,
                                                            region_size);
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
            HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, 0, tf.fd, static_cast<hoff_t>(i * 8_KiB),
                                                            region_size);
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

enum class AsyncIoTestBackend {
    Fastpath,
    Fallback,
};

struct AsyncBackendParam {
    AsyncIoTestBackend backend;
    std::string        backend_name;
    AsyncIoFunction    io;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
static std::array<AsyncBackendParam, 4> asyncBackendParams{{
    {AsyncIoTestBackend::Fastpath, "Fastpath", {hipFileReadAsync, "hipFileReadAsync"}},
    {AsyncIoTestBackend::Fastpath, "Fastpath", {hipFileWriteAsync, "hipFileWriteAsync"}},
    {AsyncIoTestBackend::Fallback, "Fallback", {hipFileReadAsync, "hipFileReadAsync"}},
    {AsyncIoTestBackend::Fallback, "Fallback", {hipFileWriteAsync, "hipFileWriteAsync"}},
}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// Exercises the read/write async golden path against each backend explicitly.
// Only one backend is enabled at a time so that if the selected backend's
// score() rejects the op, hipFileIOAsync throws instead of silently choosing
// the other backend. This guarantees the intended backend is actually run.
class HipAsyncBackendForced : public HipAsync, public ::testing::WithParamInterface<AsyncBackendParam> {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        backend = GetParam().backend;
        io_op   = GetParam().io.function;
        name    = GetParam().io.name;

        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);
        switch (backend) {
            case AsyncIoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;
            case AsyncIoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;
            default:
                FAIL() << "Unsupported AsyncIoTestBackend";
        }

        if (name == "hipFileReadAsync") {
            HipFileDataOps::zeroMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::randomizeFileRegion(tf.fd, file_size);
        }
        else {
            HipFileDataOps::zeroFileRegion(tf.fd, file_size);
            HipFileDataOps::randomizeMemoryRegion(dev_ptr, 0, buffer_size);
        }

        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, 0xf), HIPFILE_SUCCESS);
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        Context<Configuration>::get()->fastpath(true);
        Context<Configuration>::get()->fallback(true);
        HipAsync::TearDown();
    }
    AsyncIoTestBackend backend = AsyncIoTestBackend::Fastpath;
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
    hipStream_t stream;
};

TEST_P(HipAsyncBackendForced, zeroOffsets)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
}

TEST_P(HipAsyncBackendForced, bufferOffsetAligned)
{
    buffer_offset = 4_KiB;
    io_size -= 4_KiB;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, 0, 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 4_KiB);
    }
}

TEST_P(HipAsyncBackendForced, fileOffsetAligned)
{
    file_offset = 4_KiB;
    io_size -= 4_KiB;
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, 0, 4_KiB);
    }
}

INSTANTIATE_TEST_SUITE_P(HipAsyncBackendForcedSuite, HipAsyncBackendForced,
                         ::testing::ValuesIn(asyncBackendParams),
                         [](const testing::TestParamInfo<HipAsyncBackendForced::ParamType> &param_info) {
                             return param_info.param.backend_name + "_" + param_info.param.io.name;
                         });

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
            HipFileDataOps::zeroMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::randomizeFileRegion(tf.fd, file_size);
        }
        else {
            HipFileDataOps::randomizeMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::zeroFileRegion(tf.fd, file_size);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 4_KiB);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, static_cast<hoff_t>(io_size), 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, 0, 4_KiB);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, 0, 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, static_cast<hoff_t>(io_size), 4_KiB);
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
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, 0, 4_KiB);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, 0, 4_KiB);
    }
}

INSTANTIATE_TEST_SUITE_P(
    HipAsyncStreamUnfixed, HipAsyncStreamUnfixedPausedReadWrite, ::testing::ValuesIn(asyncIOFns),
    [](const testing::TestParamInfo<HipAsyncStreamUnfixedPausedReadWrite::ParamType> &param_info) {
        return param_info.param.name;
    });

// Async IO on a stream that was never registered; hipFile registers it implicitly on first use.
class HipAsyncUnregisteredStream : public HipAsync, public ::testing::WithParamInterface<AsyncIoFunction> {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        io_op = GetParam().function;
        name  = GetParam().name;
        if (name == "hipFileReadAsync") {
            HipFileDataOps::zeroMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::randomizeFileRegion(tf.fd, file_size);
        }
        else {
            HipFileDataOps::zeroFileRegion(tf.fd, file_size);
            HipFileDataOps::randomizeMemoryRegion(dev_ptr, 0, buffer_size);
        }
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    }
    void TearDown() override
    {
        // A stream used for async IO must be deregistered before it's destroyed, even if it was only
        // registered implicitly.
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        HipAsync::TearDown();
    }
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
    hipStream_t stream;
};

TEST_P(HipAsyncUnregisteredStream, roundTrip)
{
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
}

// The app holds the stream on its own wait value while it queues work, then releases it. Submitting
// must not wait for the stream, or the app would never get to release it.
TEST_P(HipAsyncUnregisteredStream, submitWhileStreamIsHeld)
{
    int attr = 0;
    ASSERT_EQ(hipDeviceGetAttribute(&attr, hipDeviceAttributeCanUseStreamWaitValue, 0), hipSuccess);
    ASSERT_EQ(attr, 1);
    uint64_t *gate{};
    uint64_t  value{0};
    ASSERT_EQ(
        hipExtMallocWithFlags(reinterpret_cast<void **>(&gate), sizeof(uint64_t), hipMallocSignalMemory),
        hipSuccess);
    ASSERT_EQ(hipMemcpy(gate, &value, sizeof(uint64_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipStreamWaitValue64(stream, gate, 1, hipStreamWaitValueEq), hipSuccess);

    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    value = 1;
    ASSERT_EQ(hipMemcpy(gate, &value, sizeof(uint64_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
    ASSERT_EQ(hipFree(gate), hipSuccess);
}

INSTANTIATE_TEST_SUITE_P(HipAsyncUnregistered, HipAsyncUnregisteredStream, ::testing::ValuesIn(asyncIOFns),
                         [](const testing::TestParamInfo<HipAsyncUnregisteredStream::ParamType> &param_info) {
                             return param_info.param.name;
                         });

// The greatest priority is reserved for the fallback path's bounce streams.
TEST_F(HipAsync, reservedPriorityStreamIsRejected)
{
    int least{};
    int greatest{};
    ASSERT_EQ(hipDeviceGetStreamPriorityRange(&least, &greatest), hipSuccess);
    if (least == greatest) {
        GTEST_SKIP() << "Device has a single stream priority";
    }
    hipStream_t stream{};
    ASSERT_EQ(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, greatest), hipSuccess);

    EXPECT_EQ(hipFileStreamRegister(stream, 0), HipFileOpError(hipFileInvalidValue));
    for (const auto &fn : asyncIOFns) {
        EXPECT_EQ(
            fn.function(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
            HipFileOpError(hipFileAsyncNotSupported))
            << fn.name;
    }
    EXPECT_EQ(hipFileStreamDeregister(stream), HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

// Overrides only the fastpath AIS entry points to throw, so a fastpath async IO that is otherwise
// selected fails at stream-execution time. Every other HIP call inherits the real implementation, so
// the fallback path runs for real.
struct FaultInjectingHip : public Hip {
    int errno_to_throw;
    explicit FaultInjectingHip(int e) : errno_to_throw{e}
    {
    }
    [[noreturn]] uint64_t hipAmdFileRead(hipAmdFileHandle_t, void *, uint64_t, int64_t) const override
    {
        throw std::system_error(errno_to_throw, std::generic_category());
    }
    [[noreturn]] uint64_t hipAmdFileWrite(hipAmdFileHandle_t, void *, uint64_t, int64_t) const override
    {
        throw std::system_error(errno_to_throw, std::generic_category());
    }
};

// Exercises async backend failover end-to-end. Both backends are enabled so the fastpath is selected
// (score 100) and the fallback is registered and eligible. A fault injected into the fastpath IO then
// determines, at stream-execution time, whether the fallback engages. The AIS gate guarantees the
// fastpath is actually the selected backend, so these results are attributable to failover.
class HipAsyncFailover : public HipAsync, public ::testing::WithParamInterface<AsyncIoFunction> {
public:
    void SetUp() override
    {
        HipAsync::SetUp();
        io_op = GetParam().function;
        name  = GetParam().name;

        Context<Configuration>::get()->fastpath(true);
        Context<Configuration>::get()->fallback(true);

        if (name == "hipFileReadAsync") {
            HipFileDataOps::zeroMemoryRegion(dev_ptr, 0, buffer_size);
            HipFileDataOps::randomizeFileRegion(tf.fd, file_size);
        }
        else {
            HipFileDataOps::zeroFileRegion(tf.fd, file_size);
            HipFileDataOps::randomizeMemoryRegion(dev_ptr, 0, buffer_size);
        }

        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_EQ(hipFileStreamRegister(stream, 0xf), HIPFILE_SUCCESS);

        // Failover is only meaningful where the fastpath is actually selected and functional. Gate last so
        // TearDown's stream cleanup remains valid when the test is skipped.
        hipFile::test::AisCapability ais_capability{test_env.allow_skip_fastpath};
        switch (ais_capability.populate(fh, dev_ptr)) {
            case hipFile::test::AisCapability::GateDecision::Run:
                break;
            case hipFile::test::AisCapability::GateDecision::Skip:
                // Keep this marker synchronized with test/CMakeLists.txt SKIP_REGULAR_EXPRESSION.
                GTEST_SKIP() << "fastpath not available in this environment\n" << ais_capability.report();
            case hipFile::test::AisCapability::GateDecision::Fail:
                FAIL() << "Fastpath Validation Failed!\n"
                       << ais_capability.report() << "\n"
                       << ais_capability.skipHint();
            default:
                break;
        }
    }
    void TearDown() override
    {
        ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
        Context<Configuration>::get()->fastpath(true);
        Context<Configuration>::get()->fallback(true);
        HipAsync::TearDown();
    }
    hipFileError_t (*io_op)(hipFileHandle_t, void *, size_t *, hoff_t *, hoff_t *, ssize_t *, hipStream_t);
    std::string name;
    hipStream_t stream;
};

// A fallback-eligible fastpath failure (ENODEV) diverts the IO to the fallback, which completes it.
TEST_P(HipAsyncFailover, eligibleFaultEngagesFallback)
{
    FaultInjectingHip    fault{ENODEV};
    ContextOverride<Hip> guard{&fault};
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, io_size);
    HipFileDataOps::assertFileAndMemoryRegionsMatch(dev_ptr, buffer_offset, tf.fd, file_offset, io_size);
}

// A non-eligible fastpath failure (EIO) is not retried; the destination is left untouched and the
// fastpath error surfaces through bytes_transferred.
TEST_P(HipAsyncFailover, ineligibleFaultSkipsFallback)
{
    FaultInjectingHip    fault{EIO};
    ContextOverride<Hip> guard{&fault};
    ASSERT_EQ(io_op(fh, dev_ptr, &io_size, &file_offset, &buffer_offset, &bytes_transferred, stream),
              HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(bytes_transferred, -hipFileInternalError);
    if (name == "hipFileReadAsync") {
        HipFileDataOps::assertZeroedMemRegion(dev_ptr, 0, io_size);
    }
    else {
        HipFileDataOps::assertZeroedFileRegion(tf.fd, 0, io_size);
    }
}

INSTANTIATE_TEST_SUITE_P(HipAsyncFailoverSuite, HipAsyncFailover, ::testing::ValuesIn(asyncIOFns),
                         [](const testing::TestParamInfo<HipAsyncFailover::ParamType> &param_info) {
                             return param_info.param.name;
                         });

#endif

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
