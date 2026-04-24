/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/fallback.h"
#include "backend/fastpath.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "io.h"
#include "mbuffer.h"
#include "mconfiguration.h"
#include "mbackend.h"
#include "mfile.h"
#include "mhip.h"
#include "mstats.h"
#include "msys.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <exception>
#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <linux/stat.h>
#include <memory>
#include <stdexcept>
#include <sys/types.h>
#include <system_error>
#include <tuple>
#include <utility>

using namespace hipFile;
using namespace testing;
using namespace std;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

static int       SCORE_ACCEPT{100};
static const int SCORE_REJECT{-1};

#if defined(STATX_DIOALIGN)
static const uint32_t DEFAULT_MEM_ALIGN{4};
static const uint32_t DEFAULT_OFFSET_ALIGN{512};
#else
static const uint32_t DEFAULT_MEM_ALIGN{4096};
static const uint32_t DEFAULT_OFFSET_ALIGN{4096};
#endif

namespace hipFile {
inline bool
operator==(const hipAmdFileHandle_t &lhs, const hipAmdFileHandle_t &rhs)
{
    return lhs.fd == rhs.fd;
}
}

// Provide default values for variables used in fastpath tests
class FastpathTestBase {
public:
    static constexpr bool          DEFAULT_ENABLE{true};
    static constexpr size_t        DEFAULT_IO_SIZE{1024 * 1024};
    static constexpr uintptr_t     DEFAULT_BUFFER_ADDR{0xABAD'CAFE'0000'0000};
    static constexpr off_t         DEFAULT_BUFFER_OFFSET{DEFAULT_MEM_ALIGN};
    static constexpr size_t        DEFAULT_BUFFER_LENGTH{DEFAULT_IO_SIZE +
                                                  static_cast<size_t>(DEFAULT_BUFFER_OFFSET)};
    static constexpr hipMemoryType DEFAULT_BUFFER_TYPE{hipMemoryTypeDevice};
    static constexpr optional<int> DEFAULT_UNBUFFERED_FD{7};
    static constexpr off_t         DEFAULT_FILE_OFFSET{8192};
    static constexpr bool          DEFAULT_IS_REGULAR_FILE{true};
    static constexpr bool          DEFAULT_IS_BLOCK_DEVICE{false};
    static constexpr bool          DEFAULT_ON_EXT4_ORDERED{true};
    static constexpr bool          DEFAULT_ON_XFS{false};
    static constexpr bool          DEFAULT_UNSUPPORTED_FILE_SYSTEMS{false};

    // Buffer and file mocks used to setup expectations
    shared_ptr<StrictMock<MFile>>   mfile{make_shared<StrictMock<MFile>>()};
    shared_ptr<StrictMock<MBuffer>> mbuffer{make_shared<StrictMock<MBuffer>>()};

    StrictMock<MConfiguration> mcfg{};
};

class FastpathScoreExpectations;

class FastpathScoreExpectationsBuilder {
public:
    StrictMock<MConfiguration>     &m_mcfg;
    shared_ptr<StrictMock<MFile>>   m_mfile;
    shared_ptr<StrictMock<MBuffer>> m_mbuffer;

    optional<bool>          m_fastpath_enabled;
    optional<void *>        m_buffer_addr;
    optional<hipMemoryType> m_buffer_type;
    optional<optional<int>> m_unbuffered_fd;
    optional<bool>          m_is_regular_file;
    optional<bool>          m_is_block_device;
    optional<bool>          m_on_ext4_ordered;
    optional<bool>          m_on_xfs;
    optional<bool>          m_unsupported_file_systems;

    FastpathScoreExpectationsBuilder(StrictMock<MConfiguration> &mcfg, shared_ptr<StrictMock<MFile>> mfile,
                                     shared_ptr<StrictMock<MBuffer>> mbuffer)
        : m_mcfg(mcfg), m_mfile(std::move(mfile)), m_mbuffer(std::move(mbuffer))
    {
    }

    FastpathScoreExpectationsBuilder &fastpathEnabled(bool enabled)
    {
        m_fastpath_enabled = enabled;
        return *this;
    }

    FastpathScoreExpectationsBuilder &bufferAddr(void *addr)
    {
        m_buffer_addr = addr;
        return *this;
    }

    FastpathScoreExpectationsBuilder &bufferType(hipMemoryType type)
    {
        m_buffer_type = type;
        return *this;
    }

    FastpathScoreExpectationsBuilder &unbufferedFd(optional<int> fd)
    {
        m_unbuffered_fd = fd;
        return *this;
    }

    FastpathScoreExpectationsBuilder &isRegularFile(bool is_regular_file)
    {
        m_is_regular_file = is_regular_file;
        return *this;
    }

    FastpathScoreExpectationsBuilder &isBlockDevice(bool is_block_device)
    {
        m_is_block_device = is_block_device;
        return *this;
    }

    FastpathScoreExpectationsBuilder &onExt4Ordered(bool on_ext4_ordered)
    {
        m_on_ext4_ordered = on_ext4_ordered;
        return *this;
    }

    FastpathScoreExpectationsBuilder &onXfs(bool on_xfs)
    {
        m_on_xfs = on_xfs;
        return *this;
    }

    FastpathScoreExpectationsBuilder &unsupportedFileSystems(bool unsupported_file_systems)
    {
        m_unsupported_file_systems = unsupported_file_systems;
        return *this;
    }

    FastpathScoreExpectations build();
};

class FastpathScoreExpectations {
public:
    FastpathScoreExpectations(const FastpathScoreExpectationsBuilder &builder)
    {
        EXPECT_CALL(builder.m_mcfg, fastpath())
            .WillOnce(Return(builder.m_fastpath_enabled.value_or(FastpathTestBase::DEFAULT_ENABLE)));
        EXPECT_CALL(*builder.m_mfile, unbufferedFd)
            .WillOnce(Return(builder.m_unbuffered_fd.value_or(FastpathTestBase::DEFAULT_UNBUFFERED_FD)));
        EXPECT_CALL(*builder.m_mbuffer, getType)
            .WillOnce(Return(builder.m_buffer_type.value_or(FastpathTestBase::DEFAULT_BUFFER_TYPE)));
        EXPECT_CALL(*builder.m_mfile, isRegularFile)
            .WillOnce(Return(builder.m_is_regular_file.value_or(FastpathTestBase::DEFAULT_IS_REGULAR_FILE)));
        EXPECT_CALL(*builder.m_mfile, onExt4Ordered)
            .WillOnce(Return(builder.m_on_ext4_ordered.value_or(FastpathTestBase::DEFAULT_ON_EXT4_ORDERED)));
        EXPECT_CALL(*builder.m_mfile, onXfs)
            .WillOnce(Return(builder.m_on_xfs.value_or(FastpathTestBase::DEFAULT_ON_XFS)));
        EXPECT_CALL(builder.m_mcfg, unsupportedFileSystems)
            .WillOnce(Return(builder.m_unsupported_file_systems.value_or(
                FastpathTestBase::DEFAULT_UNSUPPORTED_FILE_SYSTEMS)));
        EXPECT_CALL(*builder.m_mfile, isBlockDevice)
            .WillOnce(Return(builder.m_is_block_device.value_or(FastpathTestBase::DEFAULT_IS_BLOCK_DEVICE)));
        EXPECT_CALL(*builder.m_mfile, dioOffsetAlign).WillOnce(Return(DEFAULT_OFFSET_ALIGN));
        EXPECT_CALL(*builder.m_mbuffer, getBuffer)
            .WillOnce(Return(builder.m_buffer_addr.value_or(
                reinterpret_cast<void *>(FastpathTestBase::DEFAULT_BUFFER_ADDR))));
        EXPECT_CALL(*builder.m_mfile, dioMemAlign).WillOnce(Return(DEFAULT_MEM_ALIGN));
    }
};

FastpathScoreExpectations
FastpathScoreExpectationsBuilder::build()
{
    return FastpathScoreExpectations(*this);
}

struct FastpathTest : public FastpathTestBase, public Test {};

TEST_F(FastpathTest, TestDefaults)
{
    ASSERT_TRUE(DEFAULT_ENABLE);
    ASSERT_FALSE((DEFAULT_MEM_ALIGN & (DEFAULT_MEM_ALIGN - 1)));
    ASSERT_TRUE(DEFAULT_MEM_ALIGN > 1);
    ASSERT_FALSE((DEFAULT_OFFSET_ALIGN & (DEFAULT_OFFSET_ALIGN - 1)));
    ASSERT_TRUE(DEFAULT_OFFSET_ALIGN > 1);
    ASSERT_FALSE((DEFAULT_BUFFER_ADDR & (DEFAULT_MEM_ALIGN - 1)));
    ASSERT_FALSE((DEFAULT_BUFFER_OFFSET & (DEFAULT_MEM_ALIGN - 1)));
    ASSERT_FALSE((DEFAULT_IO_SIZE & (DEFAULT_OFFSET_ALIGN - 1)));
    ASSERT_FALSE((DEFAULT_FILE_OFFSET & (DEFAULT_OFFSET_ALIGN - 1)));
    ASSERT_TRUE(DEFAULT_IS_REGULAR_FILE);
    ASSERT_FALSE(DEFAULT_IS_BLOCK_DEVICE);
    ASSERT_TRUE(DEFAULT_ON_EXT4_ORDERED);
    ASSERT_FALSE(DEFAULT_ON_XFS);
}

TEST_F(FastpathTest, ScoreAcceptsIoWithDefaults)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

TEST_F(FastpathTest, ScoreRejectsIoIfFastpathIsDisabled)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).fastpathEnabled(false).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreRejectsIoIfUnbufferedFdNotAvailable)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).unbufferedFd(nullopt).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreRejectsIoWithNegativeAlignedFileOffset)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN),
                               DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreRejectsIoWithNegativeAlignedBufferOffset)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                               -static_cast<hoff_t>(DEFAULT_MEM_ALIGN)),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreRejectsIoIfBufferAddressPlusBufferOffsetIsUnaligned)
{
    // The DEFAULT_BUFFER_ADDR is DEFAULT_MEM_ALIGN aligned. Ensure that this
    // test's buffer is not DEFAULT_MEM_ALIGN aligned.
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer)
        .bufferAddr(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR + (DEFAULT_MEM_ALIGN >> 1)))
        .build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                               static_cast<hoff_t>(DEFAULT_MEM_ALIGN)),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreAcceptsIoIfFileIsRegularAndOnExt4Ordered)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).isRegularFile(true).onExt4Ordered(true).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

TEST_F(FastpathTest, ScoreAcceptsIoIfFileIsRegularAndOnXfs)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).isRegularFile(true).onXfs(true).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

TEST_F(FastpathTest, ScoreRejectsIoIfFileIsRegularAndNotOnExt4OrderedNorXfs)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer)
        .isRegularFile(true)
        .onExt4Ordered(false)
        .onXfs(false)
        .build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

TEST_F(FastpathTest, ScoreAcceptsIoIfFileIsRegularNotOnExt4OrderedNorXfsIfUnsupportedFileSystemsIsTrue)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer)
        .isRegularFile(true)
        .onExt4Ordered(false)
        .onXfs(false)
        .unsupportedFileSystems(true)
        .build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

TEST_F(FastpathTest, ScoreAcceptsIoIfFileIsBlockDevice)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).isRegularFile(false).isBlockDevice(true).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

TEST_F(FastpathTest, ScoreRejectsIoIfFileIsNotRegularFileOrBlockDevice)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).isRegularFile(false).isBlockDevice(false).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

struct FastpathSupportedHipMemoryParam : public FastpathTestBase, public TestWithParam<hipMemoryType> {};

TEST_P(FastpathSupportedHipMemoryParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).bufferType(GetParam()).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathSupportedHipMemoryParam, ValuesIn(SupportedHipMemoryTypes));

struct FastpathUnsupportedHipMemoryParam : public FastpathTestBase, public TestWithParam<hipMemoryType> {};

TEST_P(FastpathUnsupportedHipMemoryParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).bufferType(GetParam()).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathUnsupportedHipMemoryParam,
                         ValuesIn(UnsupportedHipMemoryTypes));

struct FastpathAlignedIoSizesParam : public FastpathTestBase, public TestWithParam<size_t> {};

TEST_P(FastpathAlignedIoSizesParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, GetParam(), DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_ACCEPT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathAlignedIoSizesParam,
                         Values(0, DEFAULT_OFFSET_ALIGN, DEFAULT_OFFSET_ALIGN << 1,
                                DEFAULT_OFFSET_ALIGN << 2));

struct FastpathUnalignedIoSizesParam : public FastpathTestBase, public TestWithParam<size_t> {};

TEST_P(FastpathUnalignedIoSizesParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, GetParam(), DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathUnalignedIoSizesParam,
                         Values(1, DEFAULT_OFFSET_ALIGN >> 1, DEFAULT_OFFSET_ALIGN - 1,
                                DEFAULT_OFFSET_ALIGN + 1));

struct FastpathAlignedFileOffsetsParam : public FastpathTestBase, public TestWithParam<hoff_t> {};

TEST_P(FastpathAlignedFileOffsetsParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, GetParam(), DEFAULT_BUFFER_OFFSET),
              GetParam() >= 0 ? SCORE_ACCEPT : SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathAlignedFileOffsetsParam,
                         Values(-static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN << 2),
                                -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN << 1),
                                -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN), 0, DEFAULT_OFFSET_ALIGN,
                                DEFAULT_OFFSET_ALIGN << 1, DEFAULT_OFFSET_ALIGN << 2));

struct FastpathUnalignedFileOffsetsParam : public FastpathTestBase, public TestWithParam<hoff_t> {};

TEST_P(FastpathUnalignedFileOffsetsParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, GetParam(), DEFAULT_BUFFER_OFFSET),
              SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathUnalignedFileOffsetsParam,
                         Values(-static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN << 2) - 1,
                                -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN << 1) + 1,
                                -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN) - 1,
                                -static_cast<hoff_t>(DEFAULT_OFFSET_ALIGN >> 1), 1, DEFAULT_OFFSET_ALIGN >> 1,
                                DEFAULT_OFFSET_ALIGN - 1, (DEFAULT_OFFSET_ALIGN << 1) + 1,
                                (DEFAULT_OFFSET_ALIGN << 2) - 1));

struct FastpathAlignedBufferOffsetsParam : public FastpathTestBase, public TestWithParam<hoff_t> {};

TEST_P(FastpathAlignedBufferOffsetsParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, GetParam()),
              GetParam() >= 0 ? SCORE_ACCEPT : SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathAlignedBufferOffsetsParam,
                         Values(-static_cast<hoff_t>(DEFAULT_MEM_ALIGN << 2),
                                -static_cast<hoff_t>(DEFAULT_MEM_ALIGN << 1),
                                -static_cast<hoff_t>(DEFAULT_MEM_ALIGN), 0, DEFAULT_MEM_ALIGN,
                                DEFAULT_MEM_ALIGN << 1, DEFAULT_MEM_ALIGN << 2));

/// @brief Tests negative and unaligned buffer offsets
struct FastpathUnalignedBufferOffsetsParam : public FastpathTestBase, public TestWithParam<hoff_t> {};

TEST_P(FastpathUnalignedBufferOffsetsParam, Score)
{
    FastpathScoreExpectationsBuilder(mcfg, mfile, mbuffer).build();

    ASSERT_EQ(Fastpath().score(mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, GetParam()),
              SCORE_REJECT);
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathUnalignedBufferOffsetsParam,
                         Values(-static_cast<hoff_t>(DEFAULT_MEM_ALIGN << 2) - 1,
                                -static_cast<hoff_t>(DEFAULT_MEM_ALIGN << 1) + 1,
                                -static_cast<hoff_t>(DEFAULT_MEM_ALIGN) - 1,
                                -static_cast<hoff_t>(DEFAULT_MEM_ALIGN >> 1), 1, DEFAULT_MEM_ALIGN >> 1,
                                DEFAULT_MEM_ALIGN - 1, (DEFAULT_MEM_ALIGN << 1) + 1,
                                (DEFAULT_MEM_ALIGN << 2) - 1));

struct FastpathIoParam : public FastpathTestBase, public TestWithParam<IoType> {

    StrictMock<MHip>             mhip;
    StrictMock<MStatsCollection> mstats;

    // Setup expectations on the mocks called to validate IO arguments
    void expect_validate()
    {
        EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(DEFAULT_ENABLE));
        EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
        EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
        EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));
    }

    // Setup expectations on the mocks called to validate IO arguments and
    // the mocks called up to, but not including, hipAmdFileRead/hipAmdFileWrite
    void expect_io()
    {
        expect_validate();
        EXPECT_CALL(mhip, hipInit);
    }

    // Setup expectations on the mocks called to validate IO arguments
    void expect_validate(optional<int> fd, void *bufptr, size_t buflen)
    {
        EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(DEFAULT_ENABLE));
        EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(bufptr));
        EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(buflen));
        EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(fd));
    }

    // Setup expectations on the mocks called to validate IO arguments and
    // the mocks called up to, but not including, hipAmdFileRead/hipAmdFileWrite
    void expect_io(optional<int> fd, void *bufptr, size_t buflen)
    {
        expect_validate(fd, bufptr, buflen);
        EXPECT_CALL(mhip, hipInit);
    }
};

TEST_P(FastpathIoParam, IoRejectedIfFastpathDisabled)
{
    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(false));
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, 0, -1, 0), BackendDisabled);
}

TEST_P(FastpathIoParam, IoRejectsNegativeFileOffset)
{
    expect_validate();
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, 0, -1, 0), std::invalid_argument);
}

TEST_P(FastpathIoParam, IoRejectsNegativeBufferOffset)
{
    expect_validate();
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, 0, -1), std::invalid_argument);
}

TEST_P(FastpathIoParam, IoRejectsBufferOffsetLargerThanBufferLength)
{
    expect_validate();
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, 0,
                               static_cast<hoff_t>(DEFAULT_BUFFER_LENGTH) + 1),
                 std::invalid_argument);
}

TEST_P(FastpathIoParam, IoRejectsIoSizeLargerThanBufferLength)
{
    expect_validate();
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_BUFFER_LENGTH + 1, 0, 0),
                 std::invalid_argument);
}

TEST_P(FastpathIoParam, IoRejectsIoThatCouldOverflowBuffer)
{
    expect_validate();
    ASSERT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_BUFFER_LENGTH, DEFAULT_FILE_OFFSET, 1),
                 std::invalid_argument);
}

TEST_P(FastpathIoParam, IoConfiguresHandle)
{
    hipAmdFileHandle_t handle{};
    handle.fd = DEFAULT_UNBUFFERED_FD.value();

    expect_io();
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(Eq(handle), _, _, _));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(Eq(handle), _, _, _));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET);
}

TEST_P(FastpathIoParam, IoCalculatesCorrectDevicePointer)
{
    off_t buffer_offset{0x1000};
    void *buffer_addr{reinterpret_cast<void *>(0x20000)};
    void *expected_device_ptr{reinterpret_cast<void *>(0x21000)};

    expect_io(DEFAULT_UNBUFFERED_FD, buffer_addr, static_cast<size_t>(buffer_offset) + DEFAULT_IO_SIZE);
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(_, expected_device_ptr, _, _));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(_, expected_device_ptr, _, _));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, buffer_offset);
}

TEST_P(FastpathIoParam, IoPassesThroughSizeAndFileOffset)
{
    expect_io();
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(_, _, Eq(DEFAULT_IO_SIZE), Eq(DEFAULT_FILE_OFFSET)));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(_, _, Eq(DEFAULT_IO_SIZE), Eq(DEFAULT_FILE_OFFSET)));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET);
}

TEST_P(FastpathIoParam, IoReturnsBytesTransferred)
{
    expect_io();
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(_, _, _, _)).WillOnce(Return(DEFAULT_IO_SIZE));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(_, _, _, _)).WillOnce(Return(DEFAULT_IO_SIZE));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    ASSERT_EQ(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                            DEFAULT_BUFFER_OFFSET),
              DEFAULT_IO_SIZE);
}

TEST_P(FastpathIoParam, IoReturnsBytesTransferredShort)
{
    size_t nbytes{4096};

    ASSERT_LT(nbytes, DEFAULT_IO_SIZE);

    expect_io();
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(_, _, _, _)).WillOnce(Return(nbytes));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(_, _, _, _)).WillOnce(Return(nbytes));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    ASSERT_EQ(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                            DEFAULT_BUFFER_OFFSET),
              nbytes);
}

// Ensure hip errors thrown by Hip::hipAmdFileRead/Hip::hipAmdFileWrite are not masked
TEST_P(FastpathIoParam, IoDoesNotMaskHipRuntimeError)
{
    expect_io();
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    EXPECT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                               DEFAULT_BUFFER_OFFSET),
                 Hip::RuntimeError);
}

// Ensure hip errors thrown by Hip::hipAmdFileRead/Hip::hipAmdFileWrite are not masked
TEST_P(FastpathIoParam, IoDoesNotMaskSystemError)
{
    expect_io();
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(system_error(ENODEV, generic_category())));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(system_error(EINVAL, generic_category())));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    EXPECT_THROW(Fastpath().io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET,
                               DEFAULT_BUFFER_OFFSET),
                 std::system_error);
}

// Ensure IO size is truncated to MAX_RW_COUNT
TEST_P(FastpathIoParam, IoSizeIsTruncatedToMaxRWCount)
{
    const size_t buffer_size{SIZE_MAX};
    const size_t io_size{SIZE_MAX};

    expect_io(DEFAULT_UNBUFFERED_FD, reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR), buffer_size);
    EXPECT_CALL(mstats, addIo).Times(1);
    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead(_, _, getMaxRwCount(), _)).WillOnce(Return(getMaxRwCount()));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite(_, _, getMaxRwCount(), _)).WillOnce(Return(getMaxRwCount()));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    ASSERT_EQ(Fastpath().io(GetParam(), mfile, mbuffer, io_size, 0, 0), getMaxRwCount());
}

// Note: Tests for fallback eligible exceptions are further down this file
//       in a separate test suite.
TEST_P(FastpathIoParam, IoWithFallbackThrowsAFallbackIneligibleException)
{
    auto backend    = std::make_shared<Fastpath>();
    auto m_fallback = std::make_shared<StrictMock<MBackend>>();
    backend->register_fallback_backend(m_fallback);

    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(true));
    EXPECT_CALL(mhip, hipInit).WillRepeatedly(Return());
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));

    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(std::system_error(EBADFD, generic_category())));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(std::system_error(EBADFD, generic_category())));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    ASSERT_THROW(
        backend->io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
        std::system_error);
}

// To cover the branch of non-std::system_error exceptions
TEST_P(FastpathIoParam, IoWithFallbackThrowsHipRuntimeException)
{
    auto backend    = std::make_shared<Fastpath>();
    auto m_fallback = std::make_shared<StrictMock<MBackend>>();
    backend->register_fallback_backend(m_fallback);

    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(true));
    EXPECT_CALL(mhip, hipInit).WillOnce(Return());
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));

    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    ASSERT_THROW(
        backend->io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET),
        Hip::RuntimeError);
}

TEST_P(FastpathIoParam, IoThrowsAFallbackEligibleENODEV)
{
    auto backend    = std::make_shared<Fastpath>();
    auto m_fallback = std::make_shared<StrictMock<MBackend>>();
    backend->register_fallback_backend(m_fallback);

    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(true));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
    EXPECT_CALL(mhip, hipInit).WillOnce(Return());
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));

    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(std::system_error(ENODEV, generic_category())));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(std::system_error(ENODEV, generic_category())));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    EXPECT_CALL(*m_fallback, io).WillOnce(Return(DEFAULT_IO_SIZE));
    EXPECT_CALL(*m_fallback, score).WillOnce(Return(SCORE_ACCEPT));

    ssize_t nbytes =
        backend->io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET);
    ASSERT_EQ(nbytes, DEFAULT_IO_SIZE);
}

TEST_P(FastpathIoParam, IoThrowsAFallbackEligibleEREMOTEIO)
{
    auto backend    = std::make_shared<Fastpath>();
    auto m_fallback = std::make_shared<StrictMock<MBackend>>();
    backend->register_fallback_backend(m_fallback);

    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(true));
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
    EXPECT_CALL(mhip, hipInit).WillOnce(Return());
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));

    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead)
                .WillOnce(Throw(std::system_error(EREMOTEIO, generic_category())));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite)
                .WillOnce(Throw(std::system_error(EREMOTEIO, generic_category())));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    EXPECT_CALL(*m_fallback, io).WillOnce(Return(DEFAULT_IO_SIZE));
    EXPECT_CALL(*m_fallback, score).WillOnce(Return(SCORE_ACCEPT));

    ssize_t nbytes =
        backend->io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET);
    ASSERT_EQ(nbytes, DEFAULT_IO_SIZE);
}

// If an IO is marked as fallback eligible, but the fallback backend
// still rejects the IO request, the original exception should still
// be raised.
// This test in particular also checks that the errno returned is the same.
TEST_P(FastpathIoParam, FallbackRejectsIoRequest)
{
    auto backend    = std::make_shared<Fastpath>();
    auto m_fallback = std::make_shared<StrictMock<MBackend>>();
    backend->register_fallback_backend(m_fallback);

    EXPECT_CALL(mcfg, fastpath()).WillOnce(Return(true));
    EXPECT_CALL(mhip, hipInit).WillRepeatedly(Return());
    EXPECT_CALL(*mbuffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(DEFAULT_BUFFER_ADDR)));
    EXPECT_CALL(*mbuffer, getLength).WillOnce(Return(DEFAULT_BUFFER_LENGTH));
    EXPECT_CALL(*mfile, unbufferedFd).WillOnce(Return(DEFAULT_UNBUFFERED_FD));
    EXPECT_CALL(*m_fallback, score).WillOnce(Return(SCORE_REJECT));

    switch (GetParam()) {
        case IoType::Read:
            EXPECT_CALL(mhip, hipAmdFileRead).WillOnce(Throw(std::system_error(ENODEV, generic_category())));
            break;
        case IoType::Write:
            EXPECT_CALL(mhip, hipAmdFileWrite).WillOnce(Throw(std::system_error(ENODEV, generic_category())));
            break;
        default:
            FAIL() << "Invalid IoType";
    }

    try {
        backend->io(GetParam(), mfile, mbuffer, DEFAULT_IO_SIZE, DEFAULT_FILE_OFFSET, DEFAULT_BUFFER_OFFSET);
        FAIL() << "io() was expected to throw, but it returned normally";
    }
    catch (const std::system_error &actual_exc) {
        ASSERT_EQ(typeid(std::system_error), typeid(actual_exc));
        ASSERT_EQ(ENODEV, actual_exc.code().value());
    }
    catch (...) {
        FAIL() << "io() threw something other than a std::system_error";
    }
}

INSTANTIATE_TEST_SUITE_P(FastpathTest, FastpathIoParam, Values(IoType::Read, IoType::Write));

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
