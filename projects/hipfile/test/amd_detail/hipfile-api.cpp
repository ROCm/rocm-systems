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
#include "invalid-enum.h"
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
#include <chrono>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <optional>
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
    StrictMock<MDriverState>             mock_state;
    std::shared_ptr<StrictMock<MBuffer>> batch_buffer;
    std::shared_ptr<StrictMock<MFile>>   batch_file;
    int                                  batch_cookie{};

    void SetUp() override
    {
        setUpBatchMocks();
    }

    /// @brief Create the registered buffer and file that batch parameters refer to.
    ///
    /// The buffer is one byte long, so any non-zero offset or size greater
    /// than one is out of range.
    void setUpBatchMocks()
    {
        batch_buffer = std::make_shared<StrictMock<MBuffer>>();
        EXPECT_CALL(*batch_buffer, getBuffer).WillRepeatedly(Return(reinterpret_cast<void *>(0x123)));
        EXPECT_CALL(*batch_buffer, getLength).WillRepeatedly(Return(1));

        batch_file = std::make_shared<StrictMock<MFile>>();
        EXPECT_CALL(*batch_file, handle).WillRepeatedly(Return(batch_file.get()));
    }

    /// @brief Build a valid batch parameter referring to the fixture mocks.
    hipFileIOParams_t makeBatchParam()
    {
        hipFileIOParams_t param{};
        param.u.batch.devPtr_base = batch_buffer->getBuffer();
        param.u.batch.size        = 1;
        param.fh                  = batch_file->handle();
        param.mode                = hipFileBatch;
        param.opcode              = hipFileBatchRead;
        param.cookie              = &batch_cookie;
        return param;
    }

    void expectBatchLookup(const hipFileIOParams_t &param)
    {
        EXPECT_CALL(mock_state, getFileAndBuffer(param.fh, param.u.batch.devPtr_base))
            .WillOnce(Return(file_buffer_pair{batch_file, batch_buffer}));
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
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).WillOnce([this](BatchOperations ops) {
        ASSERT_EQ(ops.size(), 1);
        EXPECT_EQ(ops[0]->event().cookie, &batch_cookie);
        EXPECT_EQ(ops[0]->event().status, hipFileWaiting);
    });

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_SUCCESS);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBadHandle)
{
    hipFileBatchHandle_t                       b_handle = nullptr;
    hipFileIOParams_t                          io_param;
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Throw(InvalidBatchHandle()));
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto           result          = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    hipFileError_t expected_result = {hipFileInvalidValue, hipSuccess};
    ASSERT_EQ(result, expected_result);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBadArgument)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).WillOnce(Throw(std::invalid_argument("")));

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBatchFull)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).WillOnce(Throw(BatchFull()));

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HipFileOpError(hipFileBatchFull));
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitInvalidOperation)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    io_param.u.batch.file_offset                              = -1;

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitNegativeBufferOffset)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    io_param.u.batch.devPtr_offset                            = -1;

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitBufferOffsetTooLarge)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    // The mocked buffer is one byte long, so any offset is past its end.
    io_param.u.batch.devPtr_offset = 1;

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitIOSizeTooLarge)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    // The mocked buffer is one byte long.
    io_param.u.batch.size = 2;

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitInvalidOpcode)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    io_param.opcode = invalidEnum<hipFileOpcode_t>(maxEnum<hipFileOpcode_t>());

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitInvalidMode)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    io_param.mode = invalidEnum<hipFileBatchMode_t>(maxEnum<hipFileBatchMode_t>());

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    expectBatchLookup(io_param);
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitInvalidSecondOperationDoesNotSubmit)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::array<hipFileIOParams_t, 2>           io_params{io_param, io_param};
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();
    io_params[1].u.batch.file_offset                          = -1;

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(mock_state, getFileAndBuffer(io_param.fh, io_param.u.batch.devPtr_base))
        .Times(2)
        .WillRepeatedly(Return(file_buffer_pair{batch_file, batch_buffer}));
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, io_params.size(), io_params.data(), 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitFileNotRegistered)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(mock_state, getFileAndBuffer(io_param.fh, io_param.u.batch.devPtr_base))
        .WillOnce(Throw(FileNotRegistered()));
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HipFileOpError(hipFileHandleNotRegistered));
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitDriverNotInitialized)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(mock_state, getFileAndBuffer(io_param.fh, io_param.u.batch.devPtr_base))
        .WillOnce(Throw(DriverNotInitialized()));
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HipFileOpError(hipFileDriverNotInitialized));
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitInvalidMemoryType)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t                          io_param = makeBatchParam();
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(mock_state, getFileAndBuffer(io_param.fh, io_param.u.batch.devPtr_base))
        .WillOnce(Throw(InvalidMemoryType()));
    EXPECT_CALL(*mock_b_context, submitOperations(_)).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 1, &io_param, 0);
    ASSERT_EQ(result, HipFileOpError(hipFileHipMemoryTypeInvalid));
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitNullptrParams)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    // With a nullptr iocbp, the API must reject the call before
    // ever reaching the batch context (avoids dereferencing the nullptr).
    EXPECT_CALL(mock_state, getBatchContext).Times(0);
    EXPECT_CALL(*mock_b_context, submitOperations).Times(0);

    auto           result          = hipFileBatchIOSubmit(b_handle, 1, nullptr, 0);
    hipFileError_t expected_result = {hipFileInvalidValue, hipSuccess};
    ASSERT_EQ(result, expected_result);
}

TEST_F(HipFileUnit, TestHipFileBatchIOSubmitZeroRequests)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOParams_t    io_param;

    EXPECT_CALL(mock_state, getBatchContext).Times(0);

    auto result = hipFileBatchIOSubmit(b_handle, 0, &io_param, 0);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusUnknownHandle)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                                   nr       = 1;
    hipFileIOEvents_t                          event{};
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Throw(InvalidBatchHandle()));
    EXPECT_CALL(*mock_b_context, getStatus).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusNullNumEvents)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    hipFileIOEvents_t    event{};

    EXPECT_CALL(mock_state, getBatchContext).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 0, nullptr, &event, nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusNullEventsWithCapacity)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned             nr       = 1;

    EXPECT_CALL(mock_state, getBatchContext).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 0, &nr, nullptr, nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusZeroCapacityWithMinimum)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned             nr       = 0;
    hipFileIOEvents_t    event{};

    EXPECT_CALL(mock_state, getBatchContext).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusMinimumExceedsCapacity)
{
    hipFileBatchHandle_t             b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                         nr       = 1;
    std::array<hipFileIOEvents_t, 1> events{};

    EXPECT_CALL(mock_state, getBatchContext).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 2, &nr, events.data(), nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusSuccess)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                                   nr       = 2;
    std::array<hipFileIOEvents_t, 2>           events{};
    struct timespec                            timeout {};
    BatchDeadline                              deadline;
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, getStatus(1, &nr, events.data(), _))
        .WillOnce(DoAll(SaveArg<3>(&deadline), Invoke([](unsigned, unsigned *num_events,
                                                         hipFileIOEvents_t *io_events, BatchDeadline) {
                            io_events[0].status = hipFileComplete;
                            io_events[0].ret    = 7;
                            *num_events         = 1;
                        })));

    const auto before = std::chrono::steady_clock::now();
    auto       result = hipFileBatchIOGetStatus(b_handle, 1, &nr, events.data(), &timeout);
    const auto after  = std::chrono::steady_clock::now();
    ASSERT_EQ(result, HIPFILE_SUCCESS);
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(events[0].status, hipFileComplete);
    ASSERT_EQ(events[0].ret, 7);

    // A zero timeout becomes a deadline that has already elapsed by the time it is waited on.
    ASSERT_TRUE(deadline.has_value());
    ASSERT_GE(*deadline, before);
    ASSERT_LE(*deadline, after);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusNullTimeoutHasNoDeadline)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                                   nr       = 1;
    hipFileIOEvents_t                          event{};
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, getStatus(1, &nr, &event, Eq(BatchDeadline{}))).Times(1);

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, nullptr);
    ASSERT_EQ(result, HIPFILE_SUCCESS);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusInvalidTimeout)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned             nr       = 1;
    hipFileIOEvents_t    event{};
    struct timespec      timeout {
        0, -1
    };
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, getStatus).Times(0);

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, &timeout);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusBadArgument)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                                   nr       = 1;
    hipFileIOEvents_t                          event{};
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, getStatus).WillOnce(Throw(std::invalid_argument("")));

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, nullptr);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOGetStatusUnexpectedException)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    unsigned                                   nr       = 1;
    hipFileIOEvents_t                          event{};
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, getStatus).WillOnce(Throw(std::runtime_error("test error")));

    auto result = hipFileBatchIOGetStatus(b_handle, 1, &nr, &event, nullptr);
    ASSERT_EQ(result, HipFileOpError(hipFileInternalError));
}

TEST_F(HipFileUnit, TestHipFileBatchIOCancelUnknownHandle)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Throw(InvalidBatchHandle()));
    EXPECT_CALL(*mock_b_context, cancelOperations).Times(0);

    auto result = hipFileBatchIOCancel(b_handle);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOCancelSuccess)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, cancelOperations);

    auto result = hipFileBatchIOCancel(b_handle);
    ASSERT_EQ(result, HIPFILE_SUCCESS);
}

TEST_F(HipFileUnit, TestHipFileBatchIOCancelBadArgument)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, cancelOperations).WillOnce(Throw(std::invalid_argument("")));

    auto result = hipFileBatchIOCancel(b_handle);
    ASSERT_EQ(result, HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileUnit, TestHipFileBatchIOCancelUnexpectedException)
{
    hipFileBatchHandle_t                       b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);
    std::shared_ptr<StrictMock<MBatchContext>> mock_b_context = std::make_shared<StrictMock<MBatchContext>>();

    EXPECT_CALL(mock_state, getBatchContext(b_handle)).WillOnce(Return(mock_b_context));
    EXPECT_CALL(*mock_b_context, cancelOperations).WillOnce(Throw(std::runtime_error("test error")));

    auto result = hipFileBatchIOCancel(b_handle);
    ASSERT_EQ(result, HipFileOpError(hipFileInternalError));
}

TEST_F(HipFileUnit, TestHipFileBatchIODestroySuccess)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);

    EXPECT_CALL(mock_state, destroyBatchContext(b_handle));

    ASSERT_NO_THROW(hipFileBatchIODestroy(b_handle));
}

TEST_F(HipFileUnit, TestHipFileBatchIODestroyUnknownHandle)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);

    EXPECT_CALL(mock_state, destroyBatchContext(b_handle)).WillOnce(Throw(InvalidBatchHandle()));

    ASSERT_NO_THROW(hipFileBatchIODestroy(b_handle));
}

TEST_F(HipFileUnit, TestHipFileBatchIODestroyBadArgument)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);

    EXPECT_CALL(mock_state, destroyBatchContext(b_handle)).WillOnce(Throw(std::invalid_argument("")));

    ASSERT_NO_THROW(hipFileBatchIODestroy(b_handle));
}

TEST_F(HipFileUnit, TestHipFileBatchIODestroyUnexpectedException)
{
    hipFileBatchHandle_t b_handle = reinterpret_cast<hipFileBatchHandle_t>(0x12345678);

    EXPECT_CALL(mock_state, destroyBatchContext(b_handle)).WillOnce(Throw(std::runtime_error("test error")));

    ASSERT_NO_THROW(hipFileBatchIODestroy(b_handle));
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

// ***********************************************************************
//  hipFileGetStatsL3 tests
// ***********************************************************************

struct HipFileGetStatsL3 : public HipFileUnopened {
    Stats                    stats{};
    StrictMock<MStatsServer> mstats{};

    void SetUp() override
    {
        EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(&stats));
        stats.setLevel(StatsLevel::Basic);
    }
};

TEST_F(HipFileGetStatsL3, NullptrReturnsInvalidValue)
{
    EXPECT_EQ(hipFileGetStatsL3(nullptr), HIPFILE_INVALID_VALUE);
}

TEST_F(HipFileGetStatsL3, NullStatsServerReturnsInternalError)
{
    EXPECT_CALL(mstats, getStats).WillRepeatedly(testing::Return(nullptr));
    hipFileStatsLevel3_t out{};
    EXPECT_EQ(hipFileGetStatsL3(&out), HipFileOpError(hipFileInternalError));
}

TEST_F(HipFileGetStatsL3, ZeroWhenNoActivity)
{
    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.num_gpus, 0u);
    EXPECT_EQ(out.detailed.basic.read_bytes, 0u);
}

TEST_F(HipFileGetStatsL3, PopulatesDetailedL2Fields)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Write)]
        .buckets[0] = 128;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Write)].buckets[0] =
        1;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.detailed.basic.write_bytes, 128u);
}

TEST_F(HipFileGetStatsL3, CountsActiveGpus)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;
    stats.getPerGpuStats(2, StatsBackend::Fallback)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.num_gpus, 2u);
}

TEST_F(HipFileGetStatsL3, PerGpuReadBytesAggregatesBothBackends)
{
    // GPU 0: fastpath 100 bytes, fallback 50 bytes
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0]                                        = 100;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 50;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].read_bytes, 150u);
}

TEST_F(HipFileGetStatsL3, FastpathMapsToNvfsAndFallbackMapsToPosix)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        3;
    stats.getPerGpuStats(0, StatsBackend::Fallback)->ioCount[static_cast<size_t>(IoType::Read)].buckets[0] =
        7;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].n_nvfs_reads, 3u);
    EXPECT_EQ(out.per_gpu_stats[0].n_posix_reads, 7u);
    EXPECT_EQ(out.per_gpu_stats[0].n_total_reads, 10u);
}

TEST_F(HipFileGetStatsL3, PerGpuUnalignedCountsPopulated)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->unalignedCount[static_cast<size_t>(IoType::Read)]  = 4;
    stats.getPerGpuStats(0, StatsBackend::Fallback)->unalignedCount[static_cast<size_t>(IoType::Write)] = 6;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse                                              = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].n_unaligned_reads, 4u);
    EXPECT_EQ(out.per_gpu_stats[0].n_unaligned_writes, 6u);
}

TEST_F(HipFileGetStatsL3, PerGpuErrorCountsPopulated)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->errorCount[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 2;
    stats.getPerGpuStats(0, StatsBackend::Fallback)
        ->errorCount[static_cast<size_t>(IoType::Write)]
        .buckets[0]                                        = 9;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].n_reads_err, 2u);
    EXPECT_EQ(out.per_gpu_stats[0].n_writes_err, 9u);
}

TEST_F(HipFileGetStatsL3, PerGpuBandwidthDerived)
{
    // 1e6 bytes over 1e6 us -> 1e6 bytes/sec
    stats.getPerGpuStats(0, StatsBackend::Fastpath)
        ->ioSizeBytes[static_cast<size_t>(IoType::Read)]
        .buckets[0] = 1000000;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->ioTimeUs[static_cast<size_t>(IoType::Read)].buckets[0] =
        1000000;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].read_bw_bytes_per_sec, 1000000u);
}

TEST_F(HipFileGetStatsL3, InactiveGpuSlotIsZero)
{
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    // GPU 1 was never used — its slot must be zeroed
    EXPECT_EQ(out.per_gpu_stats[1].read_bytes, 0u);
    EXPECT_EQ(out.per_gpu_stats[1].n_total_reads, 0u);
}

TEST_F(HipFileGetStatsL3, BufferRegistrationsReportedOnActiveGpu)
{
    stats.getBufferRegistrations()                         = 5;
    stats.getPerGpuStats(0, StatsBackend::Fastpath)->inUse = 1;

    hipFileStatsLevel3_t out{};
    ASSERT_EQ(hipFileGetStatsL3(&out), HIPFILE_SUCCESS);
    EXPECT_EQ(out.per_gpu_stats[0].n_mmap, 5u);
    EXPECT_EQ(out.per_gpu_stats[0].n_mmap_ok, 5u);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
