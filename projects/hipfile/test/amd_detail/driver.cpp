/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "mhip.h"
#include "mmountinfo.h"
#include "msys.h"

#include <cerrno>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <system_error>

using namespace hipFile;
using namespace testing;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileDriverAdmin : public HipFileUnopened {};

// Ensure that hipFileOpen() and hipFileClose() increment and
// decrement the reference count
TEST_F(HipFileDriverAdmin, OpenClose)
{
    const int64_t N = 10;

    ASSERT_EQ(hipFileUseCount(), 0);

    for (int64_t i = 1; i <= N; i++) {
        ASSERT_EQ(hipFileDriverOpen(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileUseCount(), i);
    }

    for (int64_t i = N; i >= 1; i--) {
        ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
        ASSERT_EQ(hipFileUseCount(), i - 1);
    }

    ASSERT_EQ(hipFileUseCount(), 0);
}

// Ensure hipFileHandleRegister() initializes the driver
// and DOES bump the reference count
TEST_F(HipFileDriverAdmin, HandleRegisterInitsDriver)
{
    StrictMock<MSys>            msys{};
    StrictMock<MLibMountHelper> mlibmounthelper{};

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};

    descr.handle.fd = 1234;
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.fs_ops    = nullptr;

    ASSERT_EQ(hipFileUseCount(), 0);
    expect_file_registration(msys, mlibmounthelper);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure hipFileHandleRegister() handles a file descriptor
// of zero (technically a legal POSIX value) and DOES bump
// the reference count
TEST_F(HipFileDriverAdmin, HandleRegisterGoodFD)
{
    StrictMock<MSys>            msys{};
    StrictMock<MLibMountHelper> mlibmounthelper{};

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};

    descr.handle.fd = 0;
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.fs_ops    = nullptr;

    expect_file_registration(msys, mlibmounthelper);

    ASSERT_EQ(hipFileUseCount(), 0);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
    hipFileHandleDeregister(handle);
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure hipFileHandleRegister() fails when passed a negative file
// descriptor and does NOT bump the reference count
TEST_F(HipFileDriverAdmin, HandleRegisterBadFD)
{
    StrictMock<MSys>            msys{};
    StrictMock<MLibMountHelper> mlibmounthelper{};

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};

    descr.handle.fd = -1;
    descr.type      = hipFileHandleTypeOpaqueFD;

    EXPECT_CALL(msys, statx).WillOnce(Throw(std::system_error(EBADF, std::generic_category())));

    ASSERT_EQ(hipFileUseCount(), 0);
    ASSERT_NE(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 0);
}

// Ensure hipFileHandleDeregister() does NOT bump the driver reference
// count when passed a NULL or unregistered pointer
TEST_F(HipFileDriverAdmin, HandleDeregisterDoesNotInitDriver)
{
    StrictMock<MSys>            msys{};
    StrictMock<MLibMountHelper> mlibmounthelper{};

    // Check NULL
    ASSERT_EQ(hipFileUseCount(), 0);
    hipFileHandleDeregister(nullptr);
    ASSERT_EQ(hipFileUseCount(), 0);

    // Check unregistered handle
    hipFileHandle_t handle{};

    ASSERT_EQ(hipFileUseCount(), 0);
    hipFileHandleDeregister(handle);
    ASSERT_EQ(hipFileUseCount(), 0);
}

// Ensure that closing the driver will also close any open handles. This
// is checked by trying to re-register a handle, which should fail if
// the handle were still open.
TEST_F(HipFileDriverAdmin, CloseDeregistersFile)
{
    StrictMock<MSys>            msys{};
    StrictMock<MLibMountHelper> mlibmounthelper{};

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};

    descr.handle.fd = 1234;
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.fs_ops    = nullptr;

    ASSERT_EQ(hipFileUseCount(), 0);
    expect_file_registration(msys, mlibmounthelper);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
    ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 0);
    expect_file_registration(msys, mlibmounthelper);
    ASSERT_EQ(hipFileHandleRegister(&handle, &descr), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure that registering a buffer increments the driver reference
// count
TEST_F(HipFileDriverAdmin, BufRegisterInitsDriver)
{
    StrictMock<MHip> mhip;

    expect_buffer_registration(mhip, hipMemoryTypeDevice);

    ASSERT_EQ(hipFileUseCount(), 0);
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0x1), 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure that buffer deregistration does not increment the driver
// reference count (the odd error mimics cuFile)
TEST_F(HipFileDriverAdmin, BufDeregisterDoesNotInitDriver)
{
    ASSERT_EQ(hipFileUseCount(), 0);
    ASSERT_EQ(hipFileBufDeregister(nullptr), HipFileOpError(hipFileDriverClosing));
    ASSERT_EQ(hipFileUseCount(), 0);
}

// Ensure that closing the driver also closes open buffers. This is
// checked by attempting to re-register a buffer, which should fail
// if it's already registered.
TEST_F(HipFileDriverAdmin, CloseDeregistersBuffer)
{
    StrictMock<MHip> mhip;

    ASSERT_EQ(hipFileUseCount(), 0);
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0x1), 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);

    ASSERT_EQ(hipFileDriverClose(), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 0);

    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>(0x1), 0, 0), HIPFILE_SUCCESS);
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure hipFileReadAsync():
// * Returns hipFileInvalidValue when called w/o a driver init
// * Initializes the driver and returns a reference count of 1 (like cuFile)
TEST_F(HipFileDriverAdmin, ReadAsyncInitsDriver)
{
    ASSERT_EQ(hipFileReadAsync(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure hipFileWriteAsync():
// * Returns hipFileInvalidValue when called w/o a driver init
// * Initializes the driver and returns a reference count of 1 (like cuFile)
TEST_F(HipFileDriverAdmin, WriteAsyncInitsDriver)
{
    // Error value matches cuFile
    ASSERT_EQ(hipFileWriteAsync(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
              HipFileOpError(hipFileInvalidValue));
    ASSERT_EQ(hipFileUseCount(), 1);
}

// Ensure hipFileRead():
// * Returns -1 (like cuFile) when called w/o a driver init
// * Does NOT initialize the driver and returns a reference count of 0
TEST_F(HipFileDriverAdmin, ReadDoesNotInitsDriver)
{
    ASSERT_EQ(hipFileRead(nullptr, nullptr, 0, 0, 0), -1);
    ASSERT_EQ(hipFileUseCount(), 0);
}

// Ensure hipFileWrite():
// * Returns -1 (like cuFile) when called w/o a driver init
// * Does NOT initialize the driver and returns a reference count of 0
TEST_F(HipFileDriverAdmin, WriteDoesNotInitDriver)
{
    ASSERT_EQ(hipFileWrite(nullptr, nullptr, 0, 0, 0), -1);
    ASSERT_EQ(hipFileUseCount(), 0);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
