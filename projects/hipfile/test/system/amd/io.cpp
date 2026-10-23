/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "configuration.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"

#include "io-test.h"
#include "test-common.h"
#include "test-options.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <linux/stat.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

extern SystemTestOptions test_env;

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileIo : public testing::TestWithParam<IoTestParam> {

    Tmpfile              tmpfile;
    size_t               tmpfile_size;
    uint32_t             tmpfile_dio_offset_align;
    hipFileHandle_t      tmpfile_handle;
    void                *unregistered_device_buffer;
    size_t               unregistered_device_buffer_size;
    std::vector<uint8_t> host_buffer;

    HipFileIo()
        : tmpfile{test_env.ais_capable_dir}, tmpfile_size{1_MiB}, tmpfile_dio_offset_align{4_KiB},
          tmpfile_handle{nullptr}, unregistered_device_buffer{nullptr},
          unregistered_device_buffer_size{tmpfile_size}, host_buffer(tmpfile_size)
    {
#if defined(STATX_DIOALIGN)
        struct statx stx {};
        if (0 == statx(tmpfile.fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) &&
            stx.stx_mask & STATX_DIOALIGN) {
            if (std::popcount(stx.stx_dio_offset_align) != 1) {
                throw std::runtime_error("Invalid statx dio offset");
            }
            tmpfile_dio_offset_align = stx.stx_dio_offset_align;
        }
#endif
    }

    void SetUp() override
    {
        // Disable all backends
        Context<Configuration>::get()->fastpath(false);
        Context<Configuration>::get()->fallback(false);

        // Enable the desired backend
        switch (GetParam().backend) {
            case IoTestBackend::Fastpath:
                Context<Configuration>::get()->fastpath(true);
                break;

            case IoTestBackend::Fallback:
                Context<Configuration>::get()->fallback(true);
                break;

            default:
                FAIL() << "Unsupported IoTestBackend";
        }

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tmpfile_size)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));

        ASSERT_EQ(hipSuccess, hipMalloc(&unregistered_device_buffer, unregistered_device_buffer_size));

        if (GetParam().backend == IoTestBackend::Fastpath) {
            enforceFastpathGate(tmpfile_handle, unregistered_device_buffer);
        }
    }

    void TearDown() override
    {
        ASSERT_EQ(hipSuccess, hipFree(unregistered_device_buffer));

        hipFileHandleDeregister(tmpfile_handle);
    }
};

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffset)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size - static_cast<size_t>(io_buffer_offset)};

    ASSERT_EQ(io_size, hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

TEST_P(HipFileIo, ReadToUnregisteredBufferAtOffsetReturnsErrorIfOverflow)
{
    hoff_t io_buffer_offset{4096};
    size_t io_size{unregistered_device_buffer_size};

    ASSERT_EQ(-hipFileInvalidValue,
              hipFileRead(tmpfile_handle, unregistered_device_buffer, io_size, 0, io_buffer_offset));
}

TEST_P(HipFileIo, readAtNegativeFileOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(-1, pread(tmpfile.fd, host_buffer.data(), host_buffer.size(), -1));
    ASSERT_EQ(EINVAL, errno);

    errno = 0;
    ASSERT_EQ(
        -1, hipFileRead(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, -1, 0));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, writeAtNegativeFileOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(-1, pwrite(tmpfile.fd, host_buffer.data(), host_buffer.size(), -1));
    ASSERT_EQ(EINVAL, errno);

    errno = 0;
    ASSERT_EQ(
        -1, hipFileWrite(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, -1, 0));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, readAtNegativeBufferOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(
        -1, hipFileRead(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, 0, -1));
    ASSERT_EQ(EINVAL, errno);
}

TEST_P(HipFileIo, writeAtNegativeBufferOffsetReturnsEINVAL)
{
    errno = 0;
    ASSERT_EQ(
        -1, hipFileWrite(tmpfile_handle, unregistered_device_buffer, unregistered_device_buffer_size, 0, -1));
    ASSERT_EQ(EINVAL, errno);
}

// Zero-sized IO tests require >= ROCm 7.14
#if HIP_VERSION_MAJOR > 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 14)
TEST_P(HipFileIo, zeroSizedReadAtAlignedFileOffsetReturnsZero)
{
    for (hoff_t offset = 0; offset < static_cast<hoff_t>(tmpfile_size); offset += tmpfile_dio_offset_align) {
        ASSERT_EQ(0, pread(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileRead(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));
    }
}

TEST_P(HipFileIo, zeroSizedWriteAtAlignedFileOffsetReturnsZero)
{
    for (hoff_t offset = 0; offset < static_cast<hoff_t>(tmpfile_size); offset += tmpfile_dio_offset_align) {
        ASSERT_EQ(0, pwrite(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileWrite(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));
    }
}

TEST_P(HipFileIo, zeroSizedReadAtAlignedOffsetBeyondEndOfFileReturnsZero)
{
    size_t aligned_eof{align_up(tmpfile_size + 1, tmpfile_dio_offset_align)};
    for (size_t i = 0; i < 10; i++) {
        hoff_t offset{static_cast<hoff_t>(aligned_eof * i)};
        ASSERT_EQ(0, pread(tmpfile.fd, host_buffer.data(), 0, offset));
        ASSERT_EQ(0, hipFileRead(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));

        // ensure the file size has not increased
        struct stat st {};
        ASSERT_EQ(0, fstat(tmpfile.fd, &st));
        ASSERT_EQ(tmpfile_size, static_cast<size_t>(st.st_size));
    }
}

TEST_P(HipFileIo, zeroSizedWriteAtAlignedOffsetBeyondEndOfFileReturnsZero)
{
    size_t aligned_eof{align_up(tmpfile_size + 1, tmpfile_dio_offset_align)};
    for (size_t i = 0; i < 10; i++) {
        hoff_t offset{static_cast<hoff_t>(aligned_eof * i)};
        ASSERT_EQ(0, pwrite(tmpfile.fd, host_buffer.data(), 0, static_cast<off_t>(offset)));
        ASSERT_EQ(0, hipFileWrite(tmpfile_handle, unregistered_device_buffer, 0, offset, 0));

        // ensure the file size has not increased
        struct stat st {};
        ASSERT_EQ(0, fstat(tmpfile.fd, &st));
        ASSERT_EQ(tmpfile_size, static_cast<size_t>(st.st_size));
    }
}
#endif

INSTANTIATE_TEST_SUITE_P(, HipFileIo, testing::ValuesIn(io_test_params),
                         [](const testing::TestParamInfo<HipFileIo::ParamType> &param_info) {
                             return param_info.param.name;
                         });

struct HipFileIoHipInit : public testing::Test {

    Tmpfile         tmpfile;
    size_t          tmpfile_size;
    hipFileHandle_t tmpfile_handle;
    void           *registered_device_buffer;
    size_t          registered_device_buffer_size;

    HipFileIoHipInit()
        : tmpfile{test_env.ais_capable_dir}, tmpfile_size{1024 * 1024}, tmpfile_handle{nullptr},
          registered_device_buffer{nullptr}, registered_device_buffer_size{1024 * 1024}
    {
    }

    void SetUp() override
    {
        Context<Configuration>::get()->fastpath(true);
        Context<Configuration>::get()->fallback(false);

        ASSERT_EQ(0, ftruncate(tmpfile.fd, static_cast<off_t>(tmpfile_size)));

        hipFileDescr_t descr{};
        descr.type      = hipFileHandleTypeOpaqueFD;
        descr.handle.fd = tmpfile.fd;

        ASSERT_EQ(HIPFILE_SUCCESS, hipFileHandleRegister(&tmpfile_handle, &descr));
        ASSERT_EQ(hipSuccess, hipMalloc(&registered_device_buffer, registered_device_buffer_size));
        ASSERT_EQ(HIPFILE_SUCCESS,
                  hipFileBufRegister(registered_device_buffer, registered_device_buffer_size, 0));

        enforceFastpathGate(tmpfile_handle, registered_device_buffer);
    }

    void TearDown() override
    {
        ASSERT_EQ(HIPFILE_SUCCESS, hipFileBufDeregister(registered_device_buffer));
        ASSERT_EQ(hipSuccess, hipFree(registered_device_buffer));
        hipFileHandleDeregister(tmpfile_handle);
    }
};

TEST_F(HipFileIoHipInit, spawnedThreadReadRunsWithoutSegfault)
{
    size_t  io_size{registered_device_buffer_size};
    ssize_t res{};
    std::thread([&]() { res = hipFileRead(tmpfile_handle, registered_device_buffer, io_size, 0, 0); }).join();
    ASSERT_EQ(io_size, res);
}

TEST_F(HipFileIoHipInit, spawnedThreadWriteRunsWithoutSegfault)
{
    size_t  io_size{registered_device_buffer_size};
    ssize_t res{};
    std::thread([&]() {
        res = hipFileWrite(tmpfile_handle, registered_device_buffer, io_size, 0, 0);
    }).join();
    ASSERT_EQ(io_size, res);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
