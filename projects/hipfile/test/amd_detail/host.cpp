/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "backend/host.h"
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
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace hipFile;
using namespace testing;
using namespace std;

// Put tests inside the macros to suppress the global constructor warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// Fills vector with random data
static void
rand_fill(std::vector<uint8_t> &v)
{
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

// Checks that count bytes in buffer starting at buffer_offset match count bytes in expected
// starting at expected_offset. Also verifies all other bytes in buffer are zero.
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

    if (count > 0 && memcmp(buffer.data() + buffer_offset, expected.data() + expected_offset, count)) {
        return false;
    }

    for (size_t i{static_cast<size_t>(buffer_offset) + count}; i < buffer.size(); i++) {
        if (buffer.data()[i] != 0) {
            return false;
        }
    }

    return true;
}

// ***********************************************************************
//  SCORING TESTS
// ***********************************************************************

struct HostScoring : public testing::Test {
    const size_t                    io_size{2048};
    const hoff_t                    file_offset{4096};
    const hoff_t                    buffer_offset{1024};
    shared_ptr<StrictMock<MFile>>   mfile{make_shared<StrictMock<MFile>>()};
    shared_ptr<StrictMock<MBuffer>> mbuffer{make_shared<StrictMock<MBuffer>>()};

    StrictMock<MConfiguration> mcfg;
};

TEST_F(HostScoring, ScoreAcceptsIoTargetingHostMemory)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    EXPECT_CALL(*mbuffer, getType).WillOnce(Return(hipMemoryTypeHost));

    ASSERT_EQ(Host().score(mfile, mbuffer, io_size, file_offset, buffer_offset), 1);
}

TEST_F(HostScoring, ScoreRejectsIoTargetingDeviceMemory)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    EXPECT_CALL(*mbuffer, getType).WillOnce(Return(hipMemoryTypeDevice));

    ASSERT_EQ(Host().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
}

TEST_F(HostScoring, ScoreRejectsIoTargetingUnsupportedMemoryTypes)
{
    EXPECT_CALL(mcfg, host()).WillRepeatedly(Return(true));
    for (const auto memoryType : UnsupportedHipMemoryTypes) {
        EXPECT_CALL(*mbuffer, getType).WillOnce(Return(memoryType));
        ASSERT_EQ(Host().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
    }
}

TEST_F(HostScoring, ScoreRejectsIoIfHostBackendDisabled)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(false));
    ASSERT_EQ(Host().score(mfile, mbuffer, io_size, file_offset, buffer_offset), -1);
}

// ***********************************************************************
//  BASE IO FIXTURE
// ***********************************************************************

struct HostIo : public HipFileOpened {

    shared_ptr<IBuffer>  buffer{};
    std::vector<uint8_t> buffer_data{};
    shared_ptr<IFile>    file{};
    std::vector<uint8_t> file_data{};

    StrictMock<MHip>             mhip;
    StrictMock<MSys>             msys;
    StrictMock<MLibMountHelper>  mlibmounthelper;
    StrictMock<MConfiguration>   mcfg{};
    StrictMock<MStatsCollection> mstats{};

    HostIo() : buffer_data(1024 * 1024)
    {
        EXPECT_CALL(mhip, hipGetDevice).Times(AnyNumber()).WillRepeatedly(Return(0));

        expect_buffer_registration(mhip, hipMemoryTypeHost);
        Context<DriverState>::get()->registerBuffer(buffer_data.data(), buffer_data.size(), 0);
        buffer = Context<DriverState>::get()->getRegisteredBuffer(buffer_data.data());

        expect_file_registration(msys, mlibmounthelper);
        file = Context<DriverState>::get()->getFile(Context<DriverState>::get()->registerFile(0xBADF00D));
    }

    virtual ~HostIo() override
    {
        buffer.reset();
        file.reset();
    }
};

// ***********************************************************************
//  PARAMETRIZED PARAM VALIDATION TESTS
// ***********************************************************************

struct HostParam : ::testing::TestWithParam<IoType> {

    shared_ptr<IBuffer> buffer{};
    shared_ptr<IFile>   file{};

    StrictMock<MHip>             mhip{};
    StrictMock<MSys>             msys{};
    StrictMock<MLibMountHelper>  mlibmounthelper{};
    StrictMock<MConfiguration>   mcfg{};
    StrictMock<MStatsCollection> mstats{};
    StrictMock<MStatsServer>     mserver{};

    HostParam()
    {
        assert(hipFileDriverOpen() == HIPFILE_SUCCESS);

        EXPECT_CALL(mhip, hipGetDevice).Times(AnyNumber()).WillRepeatedly(Return(0));

        expect_buffer_registration(mhip, hipMemoryTypeHost);
        void *buf{reinterpret_cast<void *>(0xFEFEFEFE)};
        Context<DriverState>::get()->registerBuffer(buf, 4096, 0);
        buffer = Context<DriverState>::get()->getRegisteredBuffer(buf);

        expect_file_registration(msys, mlibmounthelper);
        file = Context<DriverState>::get()->getFile(Context<DriverState>::get()->registerFile(0xBADF00D));
    }

    ~HostParam() override
    {
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

TEST_P(HostParam, HostIoRejectedIfBackendIsDisabled)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(false));
    ASSERT_THROW(Host().io(io_type, file, buffer, 0, 0, 0), BackendDisabled);
}

TEST_P(HostParam, HostIoThrowsOnNegativeBufferOffset)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    ASSERT_THROW(Host().io(io_type, file, buffer, 0, 0, -1), std::invalid_argument);
}

TEST_P(HostParam, HostIoThrowsIfBufferOffsetIsOutOfBounds)
{
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength())};

    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    ASSERT_THROW(Host().io(io_type, file, buffer, 0, 0, buffer_offset), std::invalid_argument);
}

TEST_P(HostParam, HostIoThrowsIfOpCouldOverrunBuffer)
{
    size_t size{10};
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength()) - 9};

    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    ASSERT_THROW(Host().io(io_type, file, buffer, size, 0, buffer_offset), std::invalid_argument);
}

TEST_P(HostParam, HostIoThrowsOnNegativeFileOffset)
{
    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    ASSERT_THROW(Host().io(io_type, file, buffer, 0, -1, 0), std::invalid_argument);
}

TEST_P(HostParam, HostIoTruncatesSizeToMAX_RW_COUNT)
{
    expect_buffer_registration(mhip, hipMemoryTypeHost);
    auto buf{reinterpret_cast<void *>(0xABABABAB)};
    Context<DriverState>::get()->registerBuffer(buf, hipFile::getMaxRwCount() + 1, 0);
    auto big_buffer{Context<DriverState>::get()->getRegisteredBuffer(buf)};

    EXPECT_CALL(mcfg, host()).WillOnce(Return(true));
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (io_type) {
        case IoType::Read:
            EXPECT_CALL(msys, pread)
                .WillRepeatedly(testing::Invoke([](int, void *, size_t count, hoff_t) -> ssize_t {
                    return static_cast<ssize_t>(count);
                }));
            break;
        case IoType::Write:
            EXPECT_CALL(msys, pwrite)
                .WillRepeatedly(testing::Invoke([](int, void *, size_t count, hoff_t) -> ssize_t {
                    return static_cast<ssize_t>(count);
                }));
            EXPECT_CALL(msys, fdatasync).Times(AnyNumber());
            break;
        default:
            FAIL();
    }

    ASSERT_EQ(hipFile::getMaxRwCount(),
              Host().io(io_type, file, std::move(big_buffer), SIZE_MAX, 0, 0));
}

INSTANTIATE_TEST_SUITE_P(Host, HostParam, ::testing::Values(IoType::Read, IoType::Write));

// ***********************************************************************
//  WRITE TESTS
// ***********************************************************************

struct HostWrite : public HostIo {

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

        if (count > 0)
            memcpy(file_data.data() + uoffset, buf, count);

        return static_cast<ssize_t>(count);
    }

    void expect_host_write()
    {
        EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
        EXPECT_CALL(msys, pwrite).WillRepeatedly(testing::Invoke(this, &HostWrite::fake_pwrite));
        EXPECT_CALL(msys, fdatasync).Times(AnyNumber());
        EXPECT_CALL(mstats, addIo).Times(1);
    }

    bool file_contains_expected_data(hoff_t file_offset, hoff_t buffer_offset, size_t count)
    {
        return contains_expected_data(file_data, file_offset, buffer_data, buffer_offset, count);
    }

    void randomize_host_buffer()
    {
        rand_fill(buffer_data);
    }
};

TEST_F(HostWrite, HostWriteHandlesZeroSizedWrite)
{
    expect_host_write();
    ASSERT_EQ(0, Host().io(IoType::Write, file, buffer, 0, 0, 0));
}

TEST_F(HostWrite, HostWriteThrowsOnPwriteException)
{
    EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, pwrite).WillOnce(testing::Throw(std::system_error(EIO, std::generic_category())));
    EXPECT_CALL(mstats, error).Times(1);

    ASSERT_THROW(Host().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0), std::system_error);
}

TEST_F(HostWrite, HostWriteHandlesInterruptedPwrite)
{
    size_t size{buffer->getLength()};
    randomize_host_buffer();

    EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, pwrite)
        .WillOnce(testing::Throw(std::system_error(EINTR, std::generic_category())))
        .WillRepeatedly(testing::Invoke(this, &HostWrite::fake_pwrite));
    EXPECT_CALL(msys, fdatasync).Times(AnyNumber());
    EXPECT_CALL(mstats, addIo).Times(1);

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, 0, 0));
    ASSERT_TRUE(file_contains_expected_data(0, 0, size));
}

TEST_F(HostWrite, HostWriteToEmptyFile)
{
    size_t size{64 * 1024};

    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, 0, 0));
    ASSERT_TRUE(file_contains_expected_data(0, 0, size));
}

TEST_F(HostWrite, HostWriteToEmptyFileAtFileOffset)
{
    size_t size{64 * 1024};
    hoff_t file_offset{1024};

    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, file_offset, 0));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, size));
}

TEST_F(HostWrite, HostWriteToEmptyFileAtBufferOffset)
{
    size_t size{64 * 1024};
    hoff_t buffer_offset{1024};

    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, 0, buffer_offset));
    ASSERT_TRUE(file_contains_expected_data(0, buffer_offset, size));
}

TEST_F(HostWrite, HostWriteToEmptyFileAtBufferOffsetAndFileOffset)
{
    size_t size{64 * 1024};
    hoff_t buffer_offset{1024};
    hoff_t file_offset{512};

    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, file_offset, buffer_offset));
    ASSERT_TRUE(file_contains_expected_data(file_offset, buffer_offset, size));
}

TEST_F(HostWrite, HostWriteOverwriteEntireFile)
{
    file_data.resize(buffer->getLength());
    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(buffer->getLength(), Host().io(IoType::Write, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(file_contains_expected_data(0, 0, file_data.size()));
}

TEST_F(HostWrite, HostWriteToFileSubregion)
{
    size_t file_length{buffer->getLength() * 2};
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength() / 2)};
    file_data.resize(file_length);
    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(buffer->getLength(), Host().io(IoType::Write, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(HostWrite, HostWriteAppendNonEmptySmallFile)
{
    file_data.resize(64);
    size_t size{64 * 1024};
    hoff_t file_offset{64};

    randomize_host_buffer();
    expect_host_write();

    ASSERT_EQ(size, Host().io(IoType::Write, file, buffer, size, file_offset, 0));
    ASSERT_TRUE(file_contains_expected_data(file_offset, 0, size));
}

// ***********************************************************************
//  READ TESTS
// ***********************************************************************

struct HostRead : public HostIo {

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

        if (count > 0)
            memcpy(buf, file_data.data() + uoffset, count);

        return static_cast<ssize_t>(count);
    }

    bool host_buffer_contains_expected_data(hoff_t file_offset, hoff_t buffer_offset, size_t count)
    {
        return contains_expected_data(buffer_data, buffer_offset, file_data, file_offset, count);
    }

    void expect_host_read()
    {
        EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
        EXPECT_CALL(msys, pread).WillRepeatedly(testing::Invoke(this, &HostRead::fake_pread));
        EXPECT_CALL(mstats, addIo).Times(1);
    }
};

TEST_F(HostRead, HostReadHandlesZeroSizedRead)
{
    expect_host_read();
    ASSERT_EQ(0, Host().io(IoType::Read, file, buffer, 0, 0, 0));
}

TEST_F(HostRead, HostReadThrowsOnPreadException)
{
    EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, pread).WillOnce(testing::Throw(std::system_error(EIO, std::generic_category())));
    EXPECT_CALL(mstats, error).Times(1);

    ASSERT_THROW(Host().io(IoType::Read, file, buffer, 4096, 0, 0), std::system_error);
}

TEST_F(HostRead, HostReadHandlesInterruptedPread)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, pread)
        .WillOnce(testing::Throw(std::system_error(EINTR, std::generic_category())))
        .WillRepeatedly(testing::Invoke(this, &HostRead::fake_pread));
    EXPECT_CALL(mstats, addIo).Times(1);

    ASSERT_EQ(file_length, Host().io(IoType::Read, file, buffer, file_length, 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(HostRead, HostReadHandlesShortPreads)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    EXPECT_CALL(mcfg, host()).WillOnce(testing::Return(true));
    EXPECT_CALL(msys, pread)
        .WillOnce(testing::Invoke([this](int fd, void *buf, size_t count, hoff_t offset) -> ssize_t {
            return this->fake_pread(fd, buf, count / 2, offset);
        }))
        .WillRepeatedly(testing::Invoke(this, &HostRead::fake_pread));
    EXPECT_CALL(mstats, addIo).Times(1);

    ASSERT_EQ(file_length, Host().io(IoType::Read, file, buffer, file_length, 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(HostRead, HostReadHandlesEmptyFile)
{
    const size_t file_length{0};
    init_file(file_length);

    expect_host_read();
    ASSERT_EQ(file_length, Host().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(HostRead, HostReadHandlesFileSmallerThanBuffer)
{
    size_t file_length{buffer->getLength() / 2};
    init_file(file_length);

    expect_host_read();
    ASSERT_EQ(file_length, Host().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(HostRead, HostReadHandlesFileSameSizeAsBuffer)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);

    expect_host_read();
    ASSERT_EQ(file_length, Host().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, file_length));
}

TEST_F(HostRead, HostReadHandlesFileLargerThanBuffer)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);

    expect_host_read();
    ASSERT_EQ(buffer->getLength(), Host().io(IoType::Read, file, buffer, buffer->getLength(), 0, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, buffer->getLength()));
}

TEST_F(HostRead, HostReadWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 3};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_host_read();
    ASSERT_EQ(buffer->getLength(), Host().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(HostRead, HostReadToEofWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_host_read();
    ASSERT_EQ(buffer->getLength(), Host().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, 0, buffer->getLength()));
}

TEST_F(HostRead, HostReadPastEofWithNonZeroFileOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength()) + 1};

    expect_host_read();
    ASSERT_EQ(buffer->getLength() - 1,
              Host().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, 0, buffer->getLength() - 1));
}

TEST_F(HostRead, HostReadCanReadSingleByteAtEndOfFile)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(file_length) - 1};

    expect_host_read();
    ASSERT_EQ(1, Host().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, 0, 1));
}

TEST_F(HostRead, HostReadEmptyFileWithNonZeroFileOffset)
{
    size_t file_length{0};
    init_file(file_length);
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_host_read();
    ASSERT_EQ(0, Host().io(IoType::Read, file, buffer, buffer->getLength(), file_offset, 0));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, 0, 0));
}

TEST_F(HostRead, HostReadWithNonZeroBufferOffset)
{
    size_t file_length{buffer->getLength() * 2};
    init_file(file_length);
    hoff_t buffer_offset{static_cast<hoff_t>(1)};

    expect_host_read();
    ASSERT_EQ(buffer->getLength() - 1,
              Host().io(IoType::Read, file, buffer, buffer->getLength() - 1, 0, buffer_offset));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, buffer_offset, buffer->getLength() - 1));
}

TEST_F(HostRead, HostReadCanReadIntoLastByteOfBuffer)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength() - 1)};

    expect_host_read();
    ASSERT_EQ(1, Host().io(IoType::Read, file, buffer, 1, 0, buffer_offset));
    ASSERT_TRUE(host_buffer_contains_expected_data(0, buffer_offset, 1));
}

TEST_F(HostRead, HostReadWithNonZeroBufferOffsetAndFileOffset)
{
    size_t file_length{buffer->getLength()};
    init_file(file_length);
    hoff_t file_offset{74};
    hoff_t buffer_offset{97};
    size_t read_size{buffer->getLength() / 2};

    expect_host_read();
    ASSERT_EQ(read_size, Host().io(IoType::Read, file, buffer, read_size, file_offset, buffer_offset));
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, buffer_offset, read_size));
}

/// @brief Test reading from a region within the file
///
/// [SOF.....[....REGION....]....EOF]
TEST_F(HostRead, ReadFromRegionWithinFile)
{
    size_t file_length{buffer->getLength() * 3};
    init_file(file_length);

    size_t size{buffer->getLength() / 2};
    hoff_t buffer_offset{static_cast<hoff_t>(buffer->getLength() / 4)};
    hoff_t file_offset{static_cast<hoff_t>(buffer->getLength())};

    expect_host_read();
    ASSERT_EQ(Host().io(IoType::Read, file, buffer, size, file_offset, buffer_offset), size);
    ASSERT_TRUE(host_buffer_contains_expected_data(file_offset, buffer_offset, size));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
