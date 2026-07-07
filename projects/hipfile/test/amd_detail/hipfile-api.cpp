/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * These tests are integration-level tests against the hipFile API.
 * They should test that the various hipFile modules are compatible with
 * one another. Running IO is not intended to be guaranteed to run here.
 *
 * All tests in this file are expected to pass on a CPU-only node.
 */

#include "batch/batch.h"
#include "context.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "hipfile-private.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "io.h"
#include "mbackend.h"
#include "mbatch.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mhip.h"
#include "mmountinfo.h"
#include "mstate.h"
#include "mstats.h"
#include "msys.h"
#include "state.h"
#include "stats.h"

#include <array>
#include <cerrno>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <sys/types.h>
#include <system_error>
#include <vector>

namespace hipFile {
struct Backend;
}

using namespace hipFile;
using namespace std;
using namespace testing;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileUnit : public HipFileUnopened {
    StrictMock<MDriverState> mock_state;
    void                     SetUp() override
    {
    }
};

TEST_F(HipFileUnit, TestHipFileBatchIOSetupSuccess)
{
    hipFileBatchHandle_t b_handle          = nullptr;
    hipFileBatchHandle_t expected_b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);

    EXPECT_CALL(mock_state, createBatchContext).WillOnce(Return(expected_b_handle));

    auto result = hipFileBatchIOSetUp(&b_handle, 1);
    EXPECT_EQ(result, HIPFILE_SUCCESS);
    EXPECT_EQ(b_handle, expected_b_handle);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSetupBadArgument)
{
    hipFileBatchHandle_t b_handle = nullptr;

    EXPECT_CALL(mock_state, createBatchContext).WillOnce(Throw(std::invalid_argument("")));

    auto result = hipFileBatchIOSetUp(&b_handle, 0);
    EXPECT_EQ(result, HIPFILE_INVALID_VALUE);
    EXPECT_EQ(b_handle, nullptr);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSetupNullptrHandle)
{
    auto result = hipFileBatchIOSetUp(nullptr, 1);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitSuccess)
{
    hipFileBatchHandle_t           b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t              io_param;
    std::shared_ptr<MBatchContext> mock_b_context = std::make_shared<MBatchContext>();

    EXPECT_CALL(mock_state, getBatchContext).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, submit_operations);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_SUCCESS);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBadHandle)
{
    hipFileBatchHandle_t           b_handle = nullptr;
    hipFileIOParams_t              io_param;
    std::shared_ptr<MBatchContext> mock_b_context = std::make_shared<MBatchContext>();

    EXPECT_CALL(mock_state, getBatchContext).WillOnce(Throw(InvalidBatchHandle()));
    EXPECT_CALL(*mock_b_context, submit_operations).Times(0);

    auto           result          = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    hipFileError_t expected_result = {hipFileInvalidValue, hipSuccess};
    ASSERT_EQ(result, expected_result);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBadArgument)
{
    hipFileBatchHandle_t           b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t              io_param;
    std::shared_ptr<MBatchContext> mock_b_context = std::make_shared<MBatchContext>();

    EXPECT_CALL(mock_state, getBatchContext).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, submit_operations).WillOnce(Throw(std::invalid_argument("")));

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitNullptrParams)
{
    hipFileBatchHandle_t           b_handle       = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<MBatchContext> mock_b_context = std::make_shared<MBatchContext>();

    // With nr > 0 and a nullptr iocbp, the API must reject the call before
    // ever reaching the batch context (avoids dereferencing the nullptr).
    EXPECT_CALL(mock_state, getBatchContext).Times(0);
    EXPECT_CALL(*mock_b_context, submit_operations).Times(0);

    auto           result          = hipFileBatchIOSubmit(b_handle, 1, nullptr, 0);
    hipFileError_t expected_result = {hipFileInvalidValue, hipSuccess};
    ASSERT_EQ(result, expected_result);
}

/// @brief Test hipFileIO function
struct HipFileIoParam : public TestWithParam<IoType> {
    hipFileHandle_t                  file_handle{};
    void                            *bufptr{reinterpret_cast<void *>(0xDEC0DE)};
    size_t                           buflen{4096};
    const void                      *unreg_bufptr{reinterpret_cast<void *>(0xFACEFEED)};
    shared_ptr<StrictMock<MBackend>> mbackend{make_shared<StrictMock<MBackend>>()};
    vector<shared_ptr<Backend>>      mbackends{mbackend};

    HipFileIoParam()
    {
        {
            StrictMock<MSys>            msys;
            StrictMock<MLibMountHelper> mlmh;
            expect_file_registration(msys, mlmh);
            file_handle = Context<DriverState>::get()->registerFile(0x0BADCAFE);
        }

        {
            StrictMock<MHip> mhip;
            expect_buffer_registration(mhip, hipMemoryTypeDevice);
            Context<DriverState>::get()->registerBuffer(bufptr, buflen, 0);
        }
    }
};

TEST_P(HipFileIoParam, HipFileIoHandlesHipPointerGetAttributesError)
{
    StrictMock<MHip> mhip;
    EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Throw(Hip::RuntimeError(hipErrorUnknown)));
    ASSERT_EQ(hipFileIo(GetParam(), file_handle, unreg_bufptr, 0, 0, 0, mbackends), -hipErrorUnknown);
}

TEST_P(HipFileIoParam, HipFileIoHandlesUnsupportedHipMemoryType)
{
    for (const auto memoryType : UnsupportedHipMemoryTypes) {
        StrictMock<MHip>      mhip;
        hipPointerAttribute_t attrs{};
        attrs.type = memoryType;
        EXPECT_CALL(mhip, hipPointerGetAttributes).WillOnce(testing::Return(attrs));
        ASSERT_EQ(hipFileIo(GetParam(), file_handle, unreg_bufptr, 0, 0, 0, mbackends),
                  -static_cast<ssize_t>(hipFileHipMemoryTypeInvalid));
    }
}

TEST_P(HipFileIoParam, HipFileIoHandlesInvalidFileHandle)
{
    auto invalid_handle{reinterpret_cast<hipFileHandle_t>(0xdeadbeef)};
    ASSERT_EQ(hipFileIo(GetParam(), invalid_handle, bufptr, 0, 0, 0, mbackends), -hipFileHandleNotRegistered);
}

TEST_P(HipFileIoParam, HipFileIoHandlesSysRuntimeError)
{
    EXPECT_CALL(*mbackend, score).WillOnce(Return(1));
    EXPECT_CALL(*mbackend, io).WillOnce(Throw(std::system_error(EBADFD, std::generic_category())));
    errno = 0;
    ASSERT_EQ(hipFileIo(GetParam(), file_handle, bufptr, buflen, 0, 0, mbackends), -1);
    ASSERT_EQ(errno, EBADFD);
}

TEST_P(HipFileIoParam, HipFileIoHandlesHipRuntimeError)
{
    EXPECT_CALL(*mbackend, score).WillOnce(Return(1));
    EXPECT_CALL(*mbackend, io).WillOnce(Throw(Hip::RuntimeError(hipErrorUnknown)));
    ASSERT_EQ(hipFileIo(GetParam(), file_handle, bufptr, buflen, 0, 0, mbackends), -hipErrorUnknown);
}

TEST_P(HipFileIoParam, HipFileIoHandlesInvalidArgumentError)
{
    EXPECT_CALL(*mbackend, score).WillOnce(Return(1));
    EXPECT_CALL(*mbackend, io).WillOnce(Throw(std::invalid_argument("")));
    ASSERT_EQ(hipFileIo(GetParam(), file_handle, bufptr, buflen, 0, 0, mbackends), -hipFileInvalidValue);
}

TEST_P(HipFileIoParam, HipFileIoHandlesBackendDisabled)
{
    EXPECT_CALL(*mbackend, score).WillOnce(Return(1));
    EXPECT_CALL(*mbackend, io).WillOnce(Throw(BackendDisabled()));
    ASSERT_EQ(hipFileIo(GetParam(), file_handle, bufptr, buflen, 0, 0, mbackends), -hipFileInternalError);
}

INSTANTIATE_TEST_SUITE_P(HipFileIo, HipFileIoParam, Values(IoType::Read, IoType::Write));

struct HipFileIoBackendSelectionParam : public ::testing::TestWithParam<IoType> {

    IoType                                io_type;
    hipFileHandle_t                       handle;
    void                                 *buffer;
    size_t                                io_size;
    hoff_t                                file_offset;
    hoff_t                                buffer_offset;
    int                                   flags;
    std::shared_ptr<StrictMock<MFile>>    mfile;
    std::shared_ptr<StrictMock<MBuffer>>  mbuffer;
    std::shared_ptr<StrictMock<MBackend>> mbe1;
    std::shared_ptr<StrictMock<MBackend>> mbe2;
    std::shared_ptr<StrictMock<MBackend>> mbe3;
    StrictMock<MDriverState>              mds;

    HipFileIoBackendSelectionParam()
        : io_type{GetParam()}, handle{reinterpret_cast<hipFileHandle_t>(0xBADF00D)},
          buffer{reinterpret_cast<void *>(0xDEADBEEF)}, io_size{1024}, file_offset{32}, buffer_offset{64},
          flags{0}, mfile{std::make_shared<StrictMock<MFile>>()},
          mbuffer{std::make_shared<StrictMock<MBuffer>>()}, mbe1{std::make_shared<StrictMock<MBackend>>()},
          mbe2{std::make_shared<StrictMock<MBackend>>()}, mbe3{std::make_shared<StrictMock<MBackend>>()}
    {
    }
};

TEST_P(HipFileIoBackendSelectionParam, HipFileIoThrowsIfThereAreNoBackends)
{
    auto backends{std::vector<std::shared_ptr<Backend>>()};

    EXPECT_CALL(mds, getFileAndBuffer(handle, buffer)).WillOnce(Return(file_buffer_pair{mfile, mbuffer}));
    EXPECT_CALL(mds, getBackends).WillOnce(Return(std::move(backends)));

    switch (io_type) {
        case IoType::Read:
            ASSERT_EQ(hipFileRead(handle, buffer, io_size, file_offset, buffer_offset),
                      -hipFileInternalError);
            break;
        case IoType::Write:
            ASSERT_EQ(hipFileWrite(handle, buffer, io_size, file_offset, buffer_offset),
                      -hipFileInternalError);
            break;
        default:
            FAIL() << "Unhandled IO Type";
    }
}

TEST_P(HipFileIoBackendSelectionParam, HipFileIoThrowsIfAllBackendsRejectTheIO)
{
    std::vector<std::shared_ptr<Backend>> backends{mbe1, mbe2, mbe3};

    EXPECT_CALL(mds, getFileAndBuffer(handle, buffer)).WillOnce(Return(file_buffer_pair{mfile, mbuffer}));
    EXPECT_CALL(mds, getBackends).WillOnce(Return(std::move(backends)));
    EXPECT_CALL(*mbe1, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(-1));
    EXPECT_CALL(*mbe2, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(-1));
    EXPECT_CALL(*mbe3, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(-1));

    switch (io_type) {
        case IoType::Read:
            ASSERT_EQ(hipFileRead(handle, buffer, io_size, file_offset, buffer_offset),
                      -hipFileInternalError);
            break;
        case IoType::Write:
            ASSERT_EQ(hipFileWrite(handle, buffer, io_size, file_offset, buffer_offset),
                      -hipFileInternalError);
            break;
        default:
            FAIL() << "Unhandled IO Type";
    }
}

TEST_P(HipFileIoBackendSelectionParam, HipFileIoIssuesIoToHighestScoringBackend)
{
    std::vector<std::shared_ptr<Backend>> backends{mbe1, mbe2, mbe3};

    EXPECT_CALL(mds, getFileAndBuffer(handle, buffer)).WillOnce(Return(file_buffer_pair{mfile, mbuffer}));
    EXPECT_CALL(mds, getBackends).WillOnce(Return(std::move(backends)));
    EXPECT_CALL(*mbe1, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(0));
    EXPECT_CALL(*mbe2, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(2));
    EXPECT_CALL(*mbe3, score(Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
        .WillOnce(Return(1));

    switch (io_type) {
        case IoType::Read:
            EXPECT_CALL(*mbe2,
                        io(Eq(IoType::Read), Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
                .WillOnce(Return(io_size));
            ASSERT_EQ(hipFileRead(handle, buffer, io_size, file_offset, buffer_offset), io_size);
            break;
        case IoType::Write:
            EXPECT_CALL(*mbe2,
                        io(Eq(IoType::Write), Eq(mfile), Eq(mbuffer), io_size, file_offset, buffer_offset))
                .WillOnce(Return(io_size));
            ASSERT_EQ(hipFileWrite(handle, buffer, io_size, file_offset, buffer_offset), io_size);
            break;
        default:
            FAIL() << "Unhandled IO Type";
    }
}

INSTANTIATE_TEST_SUITE_P(HipFileIoBackendSelection, HipFileIoBackendSelectionParam,
                         ::testing::Values(IoType::Read, IoType::Write));

// ***********************************************************************
//  hipFileGetStatsL1 tests
// ***********************************************************************

struct HipFileGetStatsL1 : public HipFileUnopened {
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};

    void SetUp() override
    {
        EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
        stats.setLevel(StatsLevel::Basic);
    }
};

TEST_F(HipFileGetStatsL1, NullptrReturnsInvalidValue)
{
    EXPECT_EQ(hipFileGetStatsL1(nullptr), HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileGetStatsL1, NullStatsServerReturnsInternalError)
{
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(nullptr));
    hipFileStatsLevel1_t out{};
    EXPECT_EQ(hipFileGetStatsL1(&out), HipFileOpError(hipFileInternalError));
}

TEST_F(HipFileGetStatsL1, ZeroWhenNoActivity)
{
    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_bytes, 0u);
    EXPECT_EQ(out.write_bytes, 0u);
    EXPECT_EQ(out.read_ops.ok, 0u);
    EXPECT_EQ(out.write_ops.ok, 0u);
}

TEST_F(HipFileGetStatsL1, AggregatesReadBytesAcrossGpusAndBackends)
{
    // GPU 0 fastpath: 100 bytes, 1 op, 1000us
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 100;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        1;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        1000;
    // GPU 1 fallback: 200 bytes, 2 ops, 2000us
    stats.getPerGpuStats(1, StatsBackend::Fallback)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 200;
    stats.getPerGpuStats(1, StatsBackend::Fallback)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        2;
    stats.getPerGpuStats(1, StatsBackend::Fallback)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        2000;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_bytes, 300u);
    EXPECT_EQ(out.read_ops.ok, 3u);
    EXPECT_EQ(out.read_lat_sum_us, 3000u);
}

TEST_F(HipFileGetStatsL1, AggregatesWriteBytesAcrossGpusAndBackends)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 512;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Write)].buckets[0] =
        4;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Write)].buckets[0] =
        4000;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.write_bytes, 512u);
    EXPECT_EQ(out.write_ops.ok, 4u);
    EXPECT_EQ(out.write_lat_sum_us, 4000u);
}

TEST_F(HipFileGetStatsL1, AggregatesErrorCounts)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 3;
    stats.getPerGpuStats(1, StatsBackend::Fallback)
        ->errorCount[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 5;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_ops.err, 3u);
    EXPECT_EQ(out.write_ops.err, 5u);
}

TEST_F(HipFileGetStatsL1, PopulatesRegistrationCounters)
{
    stats.getFileRegistrations()   = 7;
    stats.getBufferRegistrations() = 11;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.hdl_register_ops.ok, 7u);
    EXPECT_EQ(out.buf_register_ops.ok, 11u);
}

TEST_F(HipFileGetStatsL1, DerivesAverageLatency)
{
    // 2 reads totalling 200us -> avg 100us
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        2;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        200;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_lat_avg_us, 100u);
    EXPECT_EQ(out.write_lat_avg_us, 0u);
}

TEST_F(HipFileGetStatsL1, DerivesBandwidth)
{
    // 1e6 bytes over 1e6 us -> 1e6 bytes/sec
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1000000;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        1;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        1000000;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_bw_bytes_per_sec, 1000000u);
}

TEST_F(HipFileGetStatsL1, ZeroLatencySumLeavesAverageAndBandwidthZero)
{
    // bytes recorded but no time -> no divide-by-zero
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1000;

    hipFileStatsLevel1_t out{};
    ASSERT_EQ(hipFileGetStatsL1(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.read_lat_avg_us, 0u);
    EXPECT_EQ(out.read_bw_bytes_per_sec, 0u);
}

// ***********************************************************************
//  hipFileGetStatsL2 tests
// ***********************************************************************

struct HipFileGetStatsL2 : public HipFileUnopened {
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};

    void SetUp() override
    {
        EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
        stats.setLevel(StatsLevel::Basic);
    }
};

TEST_F(HipFileGetStatsL2, NullptrReturnsInvalidValue)
{
    EXPECT_EQ(hipFileGetStatsL2(nullptr), HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileGetStatsL2, PopulatesL1BasicFields)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 256;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        1;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        500;

    hipFileStatsLevel2_t out{};
    ASSERT_EQ(hipFileGetStatsL2(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.basic.read_bytes, 256u);
    EXPECT_EQ(out.basic.read_ops.ok, 1u);
    EXPECT_EQ(out.basic.read_lat_sum_us, 500u);
}

TEST_F(HipFileGetStatsL2, SizeHistogramsAreZeroFilled)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1024;

    hipFileStatsLevel2_t out{};
    ASSERT_EQ(hipFileGetStatsL2(&out), HIPFILE_SUCCESS);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(out.read_size_kb_hist[i], 0u) << "read_size_kb_hist[" << i << "] should be zero";
        EXPECT_EQ(out.write_size_kb_hist[i], 0u) << "write_size_kb_hist[" << i << "] should be zero";
    }
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
