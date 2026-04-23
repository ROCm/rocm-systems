/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "backend/fallback.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "io.h"
#include "mbuffer.h"
#include "mconfiguration.h"
#include "mfile.h"
#include "mhip.h"
#include "mmountinfo.h"
#include "mstats.h"
#include "msys.h"
#include "state.h"

#include <array>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace hipFile;
using namespace testing;
using namespace std;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// Fills vector with random data
static void
rand_fill(std::vector<uint8_t> &v)
{
    // *Quickly* fill the vector with data. Reading from /dev/urandom is
    // faster than C++'s prngs and C's rand.
    auto fd{open("/dev/urandom", O_RDONLY)};
    if (fd == -1) {
        throw std::runtime_error("Can't open /dev/urandom");
    }
    size_t total_bytes_read{0};
    while (total_bytes_read < v.size()) {
        auto bytes_read{read(fd, v.data() + total_bytes_read, v.size() - total_bytes_read)};
        if (bytes_read == -1) {
            throw std::runtime_error("Can't read from /dev/urandom");
        }
        total_bytes_read += static_cast<size_t>(bytes_read);
    }
    if (close(fd) == -1) {
        throw std::runtime_error("Can't close /dev/urandom");
    }
}

// Checks that count bytes in buffer starting at buffer_offset, match count
// bytes in expected starting at expected offset. Also ensure that all other
// bytes in buffer are zero.
static bool
contains_expected_data(std::vector<uint8_t> &buffer, hoff_t buffer_offset, std::vector<uint8_t> &expected,
                       hoff_t expected_offset, size_t count)
{
    if (buffer_offset < 0 || buffer.size() < static_cast<size_t>(buffer_offset) + count) {
        throw std::invalid_argument("out of bounds: buffer");
    }

    if (expected_offset < 0 || expected.size() < static_cast<size_t>(expected_offset) + count) {
        throw std::invalid_argument("out of bounds: expected");
    }

    for (hoff_t i{0}; i < buffer_offset; i++) {
        if (buffer.data()[i] != 0) {
            return false;
        }
    }

    if (memcmp(buffer.data() + buffer_offset, expected.data() + expected_offset, count)) {
        return false;
    }

    for (size_t i{static_cast<size_t>(buffer_offset) + count}; i < buffer.size(); i++) {
        if (buffer.data()[i] != 0) {
            return false;
        }
    }

    return true;
}

struct FallbackIo : public HipFileOpened {

    shared_ptr<IBuffer>  buffer{};
    std::vector<uint8_t> buffer_data{};
    shared_ptr<IFile>    file{};
    std::vector<uint8_t> file_data{};
    void                *nonnull_ptr{reinterpret_cast<void *>(0x1)};

    StrictMock<MHip>             mhip;
    StrictMock<MSys>             msys;
    StrictMock<MLibMountHelper>  mlibmounthelper;
    StrictMock<MConfiguration>   mcfg{};
    StrictMock<MStatsCollection> mstats{};

    FallbackIo() : buffer_data(1024 * 1024)
    {

        expect_buffer_registration(mhip, hipMemoryTypeDevice);
        Context<DriverState>::get()->registerBuffer(buffer_data.data(), buffer_data.size(), 0);
        buffer = Context<DriverState>::get()->getRegisteredBuffer(buffer_data.data());

        expect_file_registration(msys, mlibmounthelper);
        file = Context<DriverState>::get()->getFile(Context<DriverState>::get()->registerFile(0xBADF00D));
    }

    virtual ~FallbackIo() override
    {
        buffer.reset();
        file.reset();
    }

    hipError_t fake_hipMemcpy(void *dst, const void *src, size_t count, hipMemcpyKind kind)
    {
        (void)kind;
        memcpy(dst, src, count);
        return hipSuccess;
    }
};

struct FallbackScoring : public testing::Test {
    const size_t                    io_size{2048};
    const hoff_t                    file_offset{4096};
    const hoff_t                    buffer_offset{1024};
    shared_ptr<StrictMock<MFile>>   mfile{make_shared<StrictMock<MFile>>()};
    shared_ptr<StrictMock<MBuffer>> mbuffer{make_shared<StrictMock<MBuffer>>()};

    StrictMock<MConfiguration> mcfg;
};

TEST_F(FallbackScoring, ScoreAcceptsIoTargetingDeviceMemory)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    EXPECT_CALL(*mbuffer, getType).WillOnce(Return(hipMemoryTypeDevice));

    ASSERT_EQ(Fallback().score(mfile, mbuffer, io_size, file_offset, buffer_offset), 0);
}

TEST_F(FallbackScoring, ScoreRejectsIoTargetingUnsupportedMemoryType)
{
    EXPECT_CALL(mcfg, fallback()).WillRepeatedly(Return(true));
    for (const auto memoryType : UnsupportedHipMemoryTypes) {
        EXPECT_CALL(*mbuffer, getType).WillOnce(Return(memoryType));
        ASSERT_EQ(Fallback().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
    }
}

TEST_F(FallbackScoring, ScoreRejectsIoIfFallbackDisabled)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(false));
    ASSERT_EQ(Fallback().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
}

struct FallbackParam : ::testing::TestWithParam<IoType> {

    shared_ptr<IBuffer> buffer{};
    shared_ptr<IFile>   file{};

    StrictMock<MHip>             mhip{};
    StrictMock<MSys>             msys{};
    StrictMock<MLibMountHelper>  mlibmounthelper{};
    StrictMock<MConfiguration>   mcfg{};
    StrictMock<MStatsCollection> mstats{};

    FallbackParam()
    {
        assert(hipFileDriverOpen() == HIPFILE_SUCCESS);

        expect_buffer_registration(mhip, hipMemoryTypeDevice);
        void *buf{reinterpret_cast<void *>(0xFEFEFEFE)};
        Context<DriverState>::get()->registerBuffer(buf, 4096, 0);
        buffer = Context<DriverState>::get()->getRegisteredBuffer(buf);

        expect_file_registration(msys, mlibmounthelper);
        file = Context<DriverState>::get()->getFile(Context<DriverState>::get()->registerFile(0xBADF00D));
    }

    ~FallbackParam() override
    {
        // Drop the references to the file & buffer so that they can be
        // deregistered in hipFileDriverClose()
        file.reset();
        buffer.reset();

        while (hipFileUseCount()) {
            assert(hipFileDriverClose() == HIPFILE_SUCCESS);
        }
    }

protected:
    void SetUp() override
    {
        io_type = GetParam();
    }

    IoType io_type{IoType::Read};
};

TEST_P(FallbackParam, FallbackIoRejectedIfBackendIsDisabled)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(false));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, 0, 0, -1, 4096), BackendDisabled);
}

TEST_P(FallbackParam, FallbackIoThrowsOnNegativeBufferOffset)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, 0, 0, -1, 4096), std::invalid_argument);
}

TEST_P(FallbackParam, FallbackIoThrowsIfBufferOffsetIsOutOfBounds)
{
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength())};

    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, 0, 0, buffer_offset, 4096), std::invalid_argument);
}

TEST_P(FallbackParam, FallbackIoThrowsIfOpCouldOverrunBuffer)
{
    size_t size{10};
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength()) - 9};

    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, size, 0, buffer_offset, 4096), std::invalid_argument);
}

TEST_P(FallbackParam, FallbackIoThrowsOnNegativeFileOffset)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, 0, -1, 0, 4096), std::invalid_argument);
}

TEST_P(FallbackParam, FallbackIoTruncatesSizeToMAX_RW_COUNT)
{
    expect_buffer_registration(mhip, hipMemoryTypeDevice);
    auto buf{reinterpret_cast<void *>(0xABABABAB)};
    Context<DriverState>::get()->registerBuffer(buf, hipFile::getMaxRwCount() + 1, 0);
    auto big_buffer{Context<DriverState>::get()->getRegisteredBuffer(buf)};

    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Return(reinterpret_cast<void *>(0xFEFEFEFE)));
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (io_type) {
        case IoType::Read:
            EXPECT_CALL(msys, pread)
                .WillRepeatedly(testing::Invoke([](int, void *, size_t count, hoff_t) -> ssize_t {
                    return static_cast<ssize_t>(count);
                }));
            EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Return());
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Return());
            EXPECT_CALL(mhip, hipStreamSynchronize).WillRepeatedly(testing::Return());
            EXPECT_CALL(msys, pwrite)
                .WillRepeatedly(testing::Invoke([](int, void *, size_t count, hoff_t) -> ssize_t {
                    return static_cast<ssize_t>(count);
                }));
            EXPECT_CALL(msys, fdatasync).Times(AnyNumber());
            break;
        default:
            FAIL();
    }

    EXPECT_CALL(msys, munmap);
    ASSERT_EQ(hipFile::getMaxRwCount(),
              Fallback().io(io_type, file, std::move(big_buffer), SIZE_MAX, 0, 0, 16 * 1024 * 1024));
}

TEST_P(FallbackParam, FallbackIoThrowsOnBounceBufferAllocationFailure)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Throw(std::system_error(ENOMEM, std::generic_category())));
    ASSERT_THROW(Fallback().io(io_type, file, buffer, 4096, 0, 0, 4096), std::system_error);
}

TEST_P(FallbackParam, FallbackIoAllocatesChunkSizedHostBounceBuffer)
{
    size_t chunk_size{1024 * 1024};
    auto   ptr{reinterpret_cast<void *>(0xFEFEFEFE)};

    EXPECT_CALL(mcfg, fallback()).WillOnce(Return(true));
    EXPECT_CALL(msys, mmap(testing::_, chunk_size, testing::_, testing::_, testing::_, testing::_))
        .WillOnce(testing::Return(ptr));
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (io_type) {
        case IoType::Read:
            EXPECT_CALL(msys, pread).WillOnce(testing::Return(0));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipMemcpy);
            EXPECT_CALL(mhip, hipStreamSynchronize);
            EXPECT_CALL(msys, pwrite).WillOnce(testing::Return(0));
            EXPECT_CALL(msys, fdatasync);
            break;
        default:
            FAIL();
    }
    EXPECT_CALL(msys, munmap(ptr, chunk_size));
    ASSERT_EQ(0, Fallback().io(io_type, file, buffer, 4096, 0, 0, chunk_size));
}

INSTANTIATE_TEST_SUITE_P(Fallback, FallbackParam, ::testing::Values(IoType::Read, IoType::Write));

struct FallbackWrite : public FallbackIo {

    ssize_t fake_pwrite(int fd, void *buf, size_t count, hoff_t offset)
    {
        (void)fd;

        if (offset < 0) {
            return -1;
        }

        if (count >= static_cast<size_t>(SSIZE_MAX) + 1) {
            return -1;
        }

        auto uoffset{static_cast<size_t>(offset)};
        if (file_data.size() < uoffset + count) {
            file_data.resize(uoffset + count);
        }

        memcpy(file_data.data() + uoffset, buf, count);

        return static_cast<ssize_t>(count);
    }

    void expect_fallback_write()
    {
        EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
        EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
        EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Invoke(this, &FallbackWrite::fake_hipMemcpy));
        EXPECT_CALL(mhip, hipStreamSynchronize).WillRepeatedly(testing::Return());
        EXPECT_CALL(msys, pwrite).WillRepeatedly(testing::Invoke(this, &FallbackWrite::fake_pwrite));
        EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
        EXPECT_CALL(msys, fdatasync).Times(AnyNumber());
        EXPECT_CALL(mstats, addIo).Times(1);
    }

    // Test if the file contains the correct data
    bool file_contains_expected_data(hoff_t file_offset, hoff_t buffer_offset, size_t count)
    {
        return contains_expected_data(file_data, file_offset, buffer_data, buffer_offset, count);
    }

    void randomize_device_buffer()
    {
        rand_fill(buffer_data);
    }
};

TEST_F(FallbackWrite, FallbackWriteHandlesZeroSizedWrite)
{
    expect_fallback_write();
    ASSERT_EQ(0, Fallback().io(IoType::Write, file, buffer, 0, 0, 0));
}

TEST_F(FallbackWrite, FallbackWriteThrowsOnPwriteException)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(mhip, hipMemcpy);
    EXPECT_CALL(mhip, hipStreamSynchronize);
    EXPECT_CALL(msys, pwrite).WillOnce(testing::Throw(std::system_error(EIO, std::generic_category())));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));

    ASSERT_THROW(Fallback().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0), std::system_error);
}

TEST_F(FallbackWrite, FallbackWriteThrowsOnHipmemcpyFailure)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(mhip, hipMemcpy).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));

    ASSERT_THROW(Fallback().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0), Hip::RuntimeError);
}

TEST_F(FallbackWrite, FallbackWriteThrowsOnHipStreamSynchronizeError)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(mhip, hipMemcpy);
    EXPECT_CALL(mhip, hipStreamSynchronize).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));

    ASSERT_THROW(Fallback().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0), Hip::RuntimeError);
}

TEST_F(FallbackWrite, FallbackWriteToEmptyFile)
{
    size_t size{64 * 1024};
    size_t chunk_size{4096};

    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(size, Fallback().io(IoType::Write, file, buffer, size, 0, 0, chunk_size));
    ASSERT_TRUE(file_contains_expected_data(0, 0, size));
}

TEST_F(FallbackWrite, FallbackWriteToEmptyFileAtFileOffset)
{
    size_t size{64 * 1024};
    size_t chunk_size{4096};
    hoff_t file_offset{1024};

    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(size, Fallback().io(IoType::Write, file, buffer, size, file_offset, 0, chunk_size));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, size));
}

TEST_F(FallbackWrite, FallbackWriteToEmptyFileAtBufferOffset)
{
    size_t size{64 * 1024};
    size_t chunk_size{4096};
    hoff_t buffer_offset{1024};

    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(size, Fallback().io(IoType::Write, file, buffer, size, 0, buffer_offset, chunk_size));
    ASSERT_TRUE(file_contains_expected_data(0, buffer_offset, size));
}

TEST_F(FallbackWrite, FallbackWriteToEmptyFileAtBufferOffsetFileOffset)
{
    size_t size{64 * 1024};
    size_t chunk_size{4096};
    hoff_t buffer_offset{1024};
    hoff_t file_offset{512};

    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(size, Fallback().io(IoType::Write, file, buffer, size, file_offset, buffer_offset, chunk_size));
    ASSERT_TRUE(file_contains_expected_data(file_offset, buffer_offset, size));
}

TEST_F(FallbackWrite, FallbackWriteOverwriteEntireFile)
{
    file_data.resize(buffer->getLength());
    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(buffer->getLength(), Fallback().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(file_contains_expected_data(0, 0, file_data.size()));
}

TEST_F(FallbackWrite, FallbackWriteToFileSubregion)
{
    size_t file_length{buffer->getLength() * 2};
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength() / 2)};
    file_data.resize(file_length);
    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(buffer->getLength(),
              Fallback().io(IoType::Write, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(FallbackWrite, FallbackWriteAppendNonEmptySmallFile)
{
    file_data.resize(64);
    size_t size{64 * 1024};
    hoff_t file_offset{64};

    randomize_device_buffer();
    expect_fallback_write();

    ASSERT_EQ(size, Fallback().io(IoType::Write, file, buffer, size, file_offset, 0));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, size));
}

struct FallbackRead : public FallbackIo {

    void init_file(size_t length)
    {
        auto file_data_size_before = file_data.size();
        file_data.resize(length);

        if (file_data.size() <= file_data_size_before) {
            return;
        }

        rand_fill(file_data);
    }

    ssize_t fake_pread(int fd, void *buf, size_t count, hoff_t offset)
    {
        (void)fd;

        if (offset < 0) {
            return -1;
        }

        auto uoffset{static_cast<size_t>(offset)};

        if (count >= static_cast<size_t>(SSIZE_MAX) + 1) {
            return -1;
        }

        if (file_data.size() < uoffset) {
            return 0;
        }

        if (file_data.size() < uoffset + count) {
            count = file_data.size() - uoffset;
        }

        memcpy(buf, file_data.data() + uoffset, count);

        return static_cast<ssize_t>(count);
    }

    bool device_buffer_contains_expected_data(hoff_t file_offset, hoff_t buffer_offset, size_t count)
    {
        return contains_expected_data(buffer_data, buffer_offset, file_data, file_offset, count);
    }

    void expect_fallback_read()
    {
        EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
        EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
        EXPECT_CALL(msys, pread).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_pread));
        EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_hipMemcpy));
        EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
        EXPECT_CALL(mstats, addIo).Times(1);
    }
};

TEST_F(FallbackRead, FallbackReadHandlesZeroSizedRead)
{
    expect_fallback_read();
    ASSERT_EQ(0, Fallback().io(IoType::Read, file, buffer, 0, 0, 0));
}

/// @brief Test reading from a region within the file
///
/// [SOF.....[....REGION....]....EOF]
TEST_F(FallbackRead, ReadFromRegionWithinFile)
{
    size_t file_length{buffer->getLength() * 3};
    init_file(file_length);

    size_t size{buffer->getLength() / 2};
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength() / 4)};
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_fallback_read();
    ASSERT_EQ(Fallback().io(IoType::Read, file, buffer, size, file_offset, buffer_offset), size);
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, buffer_offset, size));
}

TEST_F(FallbackRead, FallbackReadThrowsOnPreadException)
{
    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(msys, pread).WillOnce(testing::Throw(std::system_error(EIO, std::generic_category())));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
    ASSERT_THROW(Fallback().io(IoType::Read, file, buffer, 4096, 0, 0), std::system_error);
}

TEST_F(FallbackRead, FallbackReadThrowsOnHipmemcpyFailure)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(msys, pread).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_pread));
    EXPECT_CALL(mhip, hipMemcpy).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
    ASSERT_THROW(Fallback().io(IoType::Read, file, buffer, file_length, 0, 0), Hip::RuntimeError);
}

TEST_F(FallbackRead, FallbackReadHandlesEmptyFile)
{
    const size_t file_length{0};
    init_file(file_length);

    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(msys, pread).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_pread));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
    EXPECT_CALL(mstats, addIo).Times(1);
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesShortPreads)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(msys, pread)
        .WillOnce(testing::Invoke([this](int fd, void *buf, size_t count, hoff_t offset) -> ssize_t {
            return this->fake_pread(fd, buf, count / 2, offset);
        }))
        .WillRepeatedly(testing::Invoke([this](int fd, void *buf, size_t count, hoff_t offset) -> ssize_t {
            return this->fake_pread(fd, buf, count, offset);
        }));
    EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_hipMemcpy));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
    EXPECT_CALL(mstats, addIo).Times(1);
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesInterruptedPread)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    EXPECT_CALL(mcfg, fallback()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, mmap).WillOnce(testing::Invoke(::mmap));
    EXPECT_CALL(msys, pread)
        .WillOnce(testing::Throw(std::system_error(EINTR, std::generic_category())))
        .WillRepeatedly(testing::Invoke([this](int fd, void *buf, size_t count, hoff_t offset) -> ssize_t {
            return this->fake_pread(fd, buf, count, offset);
        }));
    EXPECT_CALL(mhip, hipMemcpy).WillRepeatedly(testing::Invoke(this, &FallbackRead::fake_hipMemcpy));
    EXPECT_CALL(msys, munmap).WillOnce(testing::Invoke(::munmap));
    EXPECT_CALL(mstats, addIo).Times(1);
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesFileSmallerThanBuffer)
{
    size_t file_length{buffer->getLength() / 2};
    init_file(file_length);

    expect_fallback_read();
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesFileSameSizeAsBuffer)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    expect_fallback_read();
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesFileLargerThanBuffer)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);

    expect_fallback_read();
    ASSERT_EQ(buffer->getLength(), Fallback().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, buffer->getLength()));
}

TEST_F(FallbackRead, FallbackReadHandlesFileWithSizeMultipleOfChunkSize)
{
    size_t chunk_size{4096};
    size_t file_length{chunk_size};
    init_file(file_length);

    expect_fallback_read();
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, file_length, 0, 0, chunk_size));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadHandlesFilesWithSizeNotMultipleOfChunkSize)
{
    size_t chunk_size{4096};
    size_t file_length{chunk_size + 1};
    init_file(file_length);

    expect_fallback_read();
    ASSERT_EQ(file_length, Fallback().io(IoType::Read, file, buffer, file_length, 0, 0, chunk_size));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(FallbackRead, FallbackReadWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 3};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_fallback_read();
    ASSERT_EQ(buffer->getLength(),
              Fallback().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(FallbackRead, FallbackReadToEofWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_fallback_read();
    ASSERT_EQ(buffer->getLength(),
              Fallback().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(FallbackRead, FallbackReadPastEofWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength()) + 1};

    expect_fallback_read();
    ASSERT_EQ(buffer->getLength() - 1,
              Fallback().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, 0, buffer->getLength() - 1));
}

TEST_F(FallbackRead, FallbackReadCanReadSingleByteAtEndOfFile)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(file_length) - 1};

    expect_fallback_read();
    ASSERT_EQ(1, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, 0, 1));
}

TEST_F(FallbackRead, FallbackReadEmptyFileWithNonZeroFileOffset)
{
    size_t file_length{0};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_fallback_read();
    ASSERT_EQ(0, Fallback().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, 0, 0));
}

TEST_F(FallbackRead, FallbackReadWithNonZeroBufferOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t buffer_offset{static_cast<hoff_t>(1)};

    expect_fallback_read();
    ASSERT_EQ(buffer->getLength() - 1,
              Fallback().io(IoType::Read, file, buffer, buffer->getLength() - 1, 0, buffer_offset));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, buffer_offset, buffer->getLength() - 1));
}

TEST_F(FallbackRead, FallbackReadCanReadIntoLastByteOfBuffer)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength() - 1)};

    expect_fallback_read();
    ASSERT_EQ(1, Fallback().io(IoType::Read, file, buffer, 1, 0, buffer_offset));
    ASSERT_TRUE(device_buffer_contains_expected_data(0, buffer_offset, 1));
}

TEST_F(FallbackRead, FallbackReadWithNonZeroBufferOffsetAndFileOffset)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t file_offset{74};
    hoff_t buffer_offset{97};
    size_t read_size{buffer->getLength() / 2};

    expect_fallback_read();
    ASSERT_EQ(read_size, Fallback().io(IoType::Read, file, buffer, read_size, file_offset, buffer_offset));
    ASSERT_TRUE(device_buffer_contains_expected_data(file_offset, buffer_offset, read_size));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
