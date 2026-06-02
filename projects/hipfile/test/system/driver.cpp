/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"
#include "test-shared-fixtures.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <thread>
#include <time.h> // Do NOT convert to <ctime> as that is not guaranteed to have `struct timespec`
#include <vector>

#ifdef __HIP_PLATFORM_NVIDIA__
#include <driver_types.h>
#endif

using namespace std;

extern SystemTestOptions test_env;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// hipFile APIs that trigger driver init/deinit.

TEST_F(DriverInit, driverOpenIncrementsUseCount)
{
    for (auto i{0}; i < 10; i++) {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileUseCount(), i + 1);
    }
}

TEST_F(DriverInit, driverCloseDecrementsUseCount)
{
    auto count{10};
    for (auto i{0}; i < count; i++) {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileUseCount(), i + 1);
    }
    for (auto i{count}; 0 < i; i--) {
        ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileUseCount(), i - 1);
    }
}

TEST_F(DriverInit, hipFileHandleRegisterNullDescr)
{
    hipFileHandle_t handle;
    ASSERT_EQ(hipFileHandleRegister(&handle, nullptr), HipFileOpError(hipFileInvalidValue));

    // NVIDIA inits the driver, even when the parameters are garbage. We do not.
    // We could mimic this, but it's probably a waste of time given that it's an
    // edge case.
#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(hipFileUseCount(), 0);
#else
    ASSERT_EQ(hipFileUseCount(), 1);
#endif
}

TEST_F(DriverInit, hipFileHandleRegisterNullHandle)
{
    Tmpfile tmpfile{test_env.ais_capable_dir};

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = tmpfile.fd;

    ASSERT_EQ(hipFileHandleRegister(nullptr, &descr), HipFileOpError(hipFileInvalidValue));

    // NVIDIA inits the driver, even when the parameters are garbage. We do not.
    // We could mimic this, but it's probably a waste of time given that it's an
    // edge case.
#if defined(__HIP_PLATFORM_AMD__)
    ASSERT_EQ(hipFileUseCount(), 0);
#else
    ASSERT_EQ(hipFileUseCount(), 1);
#endif
}

TEST_F(DriverInit, hipFileHandleRegisterInitsDriver)
{
    Tmpfile tmpfile{test_env.ais_capable_dir};

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = tmpfile.fd;

    hipFileHandle_t handle;
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
}

TEST_F(DriverInit, hipFileDriverOpenAfterHandleRegisterIncrementsUseCount)
{
    Tmpfile tmpfile{test_env.ais_capable_dir};

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = tmpfile.fd;

    hipFileHandle_t handle;
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);

    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 2);
}

TEST_F(DriverInit, hipFileHandleRegisterThreadedInitsDriverOnce)
{
    constexpr size_t             count{64};
    std::vector<Tmpfile>         tmpfiles;
    std::vector<std::thread>     threads;
    std::vector<hipFileDescr_t>  descrs(count);
    std::vector<hipFileHandle_t> handles(count);

    tmpfiles.reserve(count);

    for (size_t i{0}; i < count; i++) {
        tmpfiles.emplace_back(test_env.ais_capable_dir);
        descrs[i].type      = hipFileHandleTypeOpaqueFD;
        descrs[i].handle.fd = tmpfiles[i].fd;
    }

    for (size_t i{0}; i < count; i++) {
        threads.emplace_back([i, &handles, &descrs] {
            EXPECT_EQ(hipFileHandleRegister(&handles[i], &descrs[i]), HIPFILE_SUCCESS);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    ASSERT_EQ(hipFileUseCount(), 1);

    for (size_t i{0}; i < count; i++) {
        hipFileHandleDeregister(handles[i]);
    }
}

TEST_F(DriverInit, hipFileDriverCloseDeregisteresHandle)
{
    Tmpfile         tmpfile{test_env.ais_capable_dir};
    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};

    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = tmpfile.fd;

    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HipFileOpError(hipFileHandleAlreadyRegistered));
    ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
}

TEST_F(DriverInit, hipFileReadAsync)
{
    ASSERT_EQ(hipFileReadAsync(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileUseCount(), 1);
}

TEST_F(DriverInit, hipFileWriteAsync)
{
    ASSERT_EQ(hipFileWriteAsync(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileUseCount(), 1);
}

// hipFile APIs that do not trigger driver init/deinit

TEST_F(DriverNoInit, hipFileDriverCloseReturnsNotInitIfDriverNotOpened)
{
    ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));
}

TEST_F(DriverNoInit, hipFileBatchIOSetUp)
{
    hipFileBatchHandle_t handle;
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(hipFileBatchIOSetUp(&handle, 1), HipFileOpError(hipFileSuccess));
#else
    // CU_FILE_INTERNAL_ERROR returned if this API is called prior to the driver being opened.
    ASSERT_EQ(hipFileBatchIOSetUp(&handle, 1), HipFileOpError(hipFileInternalError));
#endif
}

TEST_F(DriverNoInit, hipFileBatchIOSubmitNullArgs)
{
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(hipFileBatchIOSubmit(nullptr, 0, nullptr, 0), HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(hipFileBatchIOSubmit(nullptr, 0, nullptr, 0), HipFileOpError(hipFileInternalError));
#endif
}

TEST_F(DriverNoInit, hipFileBatchIOSubmit)
{
    hipFileBatchHandle_t handle{};
    hipFileIOParams_t    param{};

#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(hipFileBatchIOSubmit(handle, 1, &param, 0), HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(hipFileBatchIOSubmit(handle, 1, &param, 0), HipFileOpError(hipFileInternalError));
#endif
}

TEST_F(DriverNoInit, hipFileBatchIOGetStatusNullArgs)
{
    ASSERT_EQ(hipFileBatchIOGetStatus(nullptr, 0, nullptr, nullptr, nullptr),
              HipFileOpError(hipFileInternalError));
}

TEST_F(DriverNoInit, hipFileBatchIOGetStatus)
{
    hipFileBatchHandle_t handle{};
    hipFileIOEvents_t    event{};
    unsigned             nr{1};
    struct timespec      ts {
        0, 0
    };

    ASSERT_EQ(hipFileBatchIOGetStatus(handle, 0, &nr, &event, &ts), HipFileOpError(hipFileInternalError));
}

TEST_F(DriverNoInit, hipFileBatchIOCancelNullArgs)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileBatchIOCancel(nullptr), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileBatchIOCancel(nullptr), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileBatchIOCancel)
{
    hipFileBatchHandle_t handle{};

#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileBatchIOCancel(handle), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileBatchIOCancel(handle), HIPFILE_SUCCESS); // Weird
#endif
}

TEST_F(DriverNoInit, hipFileBatchIODestroy)
{
    hipFileBatchHandle_t handle{};

    hipFileBatchIODestroy(handle);
}

TEST_F(DriverNoInit, hipFileGetVersion)
{
    unsigned major = UINT_MAX;
    unsigned minor = UINT_MAX;
    unsigned patch = UINT_MAX;
    ASSERT_EQ(hipFileGetVersion(&major, &minor, &patch), HIPFILE_SUCCESS);
}

TEST_F(DriverNoInit, hipFileGetParameterSizeT)
{
    size_t value;
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileGetParameterSizeT(hipFileParamProfileStats, &value),
              HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileGetParameterSizeT(hipFileParamProfileStats, &value), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileGetParameterBool)
{
    bool value;
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileGetParameterBool(hipFileParamUsePcip2pdma, &value),
              HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileGetParameterBool(hipFileParamUsePcip2pdma, &value), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileGetParameterString)
{
    vector<char> buffer(64);
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size())),
              HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileGetParameterString(hipFileParamLogDir, buffer.data(), static_cast<int>(buffer.size())),
              HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileSetParameterSizeT)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileSetParameterSizeT(hipFileParamProfileStats, 1), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileSetParameterSizeT(hipFileParamProfileStats, 1), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileSetParameterBool)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileSetParameterBool(hipFileParamUsePcip2pdma, false), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileSetParameterBool(hipFileParamUsePcip2pdma, false), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileSetParameterString)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented on AMD
    ASSERT_EQ(hipFileSetParameterString(hipFileParamLogDir, "/tmp"), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileSetParameterString(hipFileParamLogDir, "/tmp"), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNoInit, hipFileBufDeregister)
{
    EXPECT_EQ(hipFileBufDeregister(reinterpret_cast<void *>(0x1)), HipFileOpError(hipFileDriverClosing));
}

TEST_F(DriverNoInit, hipFileHandleDeregister)
{
    hipFileHandleDeregister(reinterpret_cast<hipFileHandle_t>(0x1));
}

TEST_F(DriverNoInit, hipFileRead)
{
    vector<uint8_t> buffer(64);
    hipFileHandle_t handle{reinterpret_cast<void *>(0xCAFEBABE)};

    errno = 0;
    ASSERT_EQ(hipFileRead(handle, buffer.data(), buffer.size(), 0, 0), -1);
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(DriverNoInit, hipFileWrite)
{
    vector<uint8_t> buffer(64);
    hipFileHandle_t handle{reinterpret_cast<void *>(0xCAFEBABE)};

    errno = 0;
    ASSERT_EQ(hipFileWrite(handle, buffer.data(), buffer.size(), 0, 0), -1);
    ASSERT_EQ(errno, EINVAL);
}

// DriverNotReallyNoInit tests API functions that are not supposed to
// initialize the driver but they have some side effect (in cuFile) that makes
// hipFileSetParameterSizeT/hipFileSetParameterBool/hipFileSetParameterString
// return hipFileDriverAlreadyOpen. They are split out from DriverNoInit tests
// because the driver needs to be cycled to clear out driver state. Cycling the
// driver is time consuming.
struct DriverNotReallyNoInit : public DriverNoInit {

    void SetUp() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
        ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));
    }

    void TearDown() override
    {
        ASSERT_EQ(hipFileUseCount(), 0);
        ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));

#ifdef __HIP_PLATFORM_NVIDIA__
        // Workaround: Open and close the driver to clear the state that these API functions set
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
#endif

        ASSERT_EQ(hipFileUseCount(), 0);
        ASSERT_EQ(hipFileDriverClose(), HipFileOpError(hipFileDriverNotInitialized));
    }
};

TEST_F(DriverNotReallyNoInit, hipFileDriverGetPropertiesNullArgs)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverGetProperties(nullptr), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverGetProperties(nullptr), HipFileOpError(hipFileInvalidValue));
#endif
}

TEST_F(DriverNotReallyNoInit, hipFileDriverGetProperties)
{
    hipFileDriverProps_t props;

#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverGetProperties(&props), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverGetProperties(&props), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNotReallyNoInit, hipFileDriverSetPollMode)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverSetPollMode(true, 4096), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverSetPollMode(true, 4096), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNotReallyNoInit, hipFileDriverSetMaxDirectIOSize)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverSetMaxDirectIOSize(16 * 1024), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverSetMaxDirectIOSize(16 * 1024), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNotReallyNoInit, hipFileDriverSetMaxCacheSize)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverSetMaxCacheSize(16 * 1024), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverSetMaxCacheSize(16 * 1024), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverNotReallyNoInit, hipFileDriverSetMaxPinnedMemSize)
{
#ifdef __HIP_PLATFORM_AMD__ // Not implemented
    ASSERT_EQ(hipFileDriverSetMaxPinnedMemSize(32 * 1024), HipFileOpError(hipFileInternalError));
#else
    ASSERT_EQ(hipFileDriverSetMaxPinnedMemSize(32 * 1024), HIPFILE_SUCCESS);
#endif
}

// hipFile APIs that trigger driver init/deinit.
// These call into the HIP driver.
TEST_F(DriverInit, hipFileBufRegisterNullptrInitsDriver)
{
    ASSERT_EQ(hipFileBufRegister(nullptr, 4096, 0), HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileUseCount(), 1);
}

TEST_F(DriverInit, hipFileBufRegisterInitsDriver)
{
    size_t bufsz{4096};
    void  *buf{nullptr};

    ASSERT_EQ(hipMalloc(&buf, bufsz), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(buf, bufsz, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
    ASSERT_EQ(hipFileBufDeregister(buf), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFree(buf), hipSuccess);
}

TEST_F(DriverInit, hipFileDriverOpenAfterBufRegisterIncrementsUseCount)
{
    size_t bufsz{4096};
    void  *buf{nullptr};

    ASSERT_EQ(hipMalloc(&buf, bufsz), hipSuccess);
    EXPECT_EQ(hipFileBufRegister(buf, bufsz, 0), HIPFILE_SUCCESS);
    EXPECT_EQ(hipFileUseCount(), 1);

    ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
    EXPECT_EQ(hipFileUseCount(), 2);

    ASSERT_EQ(hipFileBufDeregister(buf), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFree(buf), hipSuccess);
}

TEST_F(DriverInit, hipFileBufRegisterThreadedInitsDriverOnce)
{
    constexpr size_t         count{64};
    constexpr size_t         bufsz{4096};
    std::vector<std::thread> threads;
    std::vector<void *>      buffers(count);

    for (size_t i{0}; i < count; i++) {
        threads.emplace_back([i, &buffers] {
            // Need to allocate in the same thread that we register in
            // otherwise hipFileBufRegister returns CUDA_ERROR_INVALID_CONTEXT
            // on NVIDIA
            ASSERT_EQ(hipMalloc(&buffers[i], bufsz), hipSuccess);
            EXPECT_EQ(hipFileBufRegister(buffers[i], bufsz, 0), HIPFILE_SUCCESS);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    ASSERT_EQ(hipFileUseCount(), 1);

    for (size_t i{0}; i < count; i++) {
        EXPECT_EQ(hipFileBufDeregister(buffers[i]), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFree(buffers[i]), hipSuccess);
    }
}

TEST_F(DriverInit, hipFileDriverCloseDeregistersBuffer)
{
    size_t bufsz{4096};
    void  *buf{nullptr};

    ASSERT_EQ(hipMalloc(&buf, bufsz), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(buf, bufsz, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufRegister(buf, bufsz, 0), HipFileOpError(hipFileMemoryAlreadyRegistered));
    ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileBufRegister(buf, bufsz, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFree(buf), hipSuccess);
}

TEST_F(DriverInit, hipFileStream_register_correct_flags_returns_success)
{
    hipStream_t hip_stream;
    ASSERT_EQ(hipStreamCreate(&hip_stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK), HIPFILE_SUCCESS);
}

TEST_F(DriverInit, hipFileStream_register_then_deregister_returns_success)
{
    hipStream_t hip_stream;
    ASSERT_EQ(hipStreamCreate(&hip_stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(hip_stream), HIPFILE_SUCCESS);
}

TEST_F(DriverInit, hipFileStream_register_incorrect_flags_returns_error)
{
    hipStream_t hip_stream;

    ASSERT_EQ(hipStreamCreate(&hip_stream), hipSuccess);
#ifdef __HIP_PLATFORM_AMD__
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK + 1),
              HipFileOpError(hipFileInvalidValue));
#else
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK + 1), HIPFILE_SUCCESS);
#endif
}

TEST_F(DriverInit, hipFileStream_register_twice_returns_error)
{
    hipStream_t hip_stream;
    ASSERT_EQ(hipStreamCreate(&hip_stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK),
              HipFileOpError(hipFileInvalidValue));
}

TEST_F(DriverInit, hipFileStream_deregister_twice_returns_error)
{
    hipStream_t hip_stream;
    ASSERT_EQ(hipStreamCreate(&hip_stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(hip_stream, HIPFILE_STREAM_FLAGS_MASK), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(hip_stream), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(hip_stream), HipFileOpError(hipFileInvalidValue));
}

TEST_F(DriverInit, hipFileStream_deregister_invalid_stream_returns_error)
{
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(1);
    ASSERT_EQ(hipFileStreamDeregister(hip_stream), HipFileOpError(hipFileInvalidValue));
}

// hipFile APIs that do not trigger driver init/deinit
// These call into the HIP driver.
TEST_F(DriverNoInit, hipFileStreamRegister)
{
    hipStream_t stream;

    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(stream, 0), HIPFILE_SUCCESS);

    ASSERT_EQ(hipFileUseCount(), 0);

    // Cleanup
    ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(DriverNoInit, hipFileStreamRegister_register_and_deregister_default_stream_works)
{
    ASSERT_EQ(hipFileStreamRegister(nullptr, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileStreamDeregister(nullptr), HIPFILE_SUCCESS);
}

TEST_F(DriverNoInit, hipFileStreamDeregisterStream)
{
    hipStream_t stream;

    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_EQ(hipFileStreamRegister(stream, 0), HIPFILE_SUCCESS);

    ASSERT_EQ(hipFileUseCount(), 0);

    ASSERT_EQ(hipFileStreamDeregister(stream), HIPFILE_SUCCESS);

    ASSERT_EQ(hipFileUseCount(), 0);

    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
