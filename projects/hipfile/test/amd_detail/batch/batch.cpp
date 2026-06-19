/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "batch/batch.h"
#include "buffer.h"
#include "file.h"
#include "hipfile.h"
#include "hipfile-test.h"
#include "hipfile-warnings.h"
#include "invalid-enum.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mstate.h"
#include "state.h"

#include <cstdio>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

using ::testing::_;
using ::testing::DoDefault;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileBatch : public HipFileUnopened {
    BatchContextMap                      batch_map = BatchContextMap{};
    std::unique_ptr<hipFileIOParams_t>   io_params;
    std::shared_ptr<StrictMock<MBuffer>> default_mock_buffer;
    std::shared_ptr<StrictMock<MFile>>   default_mock_file;
    const hipFileHandle_t                file_handle{reinterpret_cast<void *>(0xDEADBEEF)};
    void *const                          buffer_pointer{reinterpret_cast<void *>(0x0BADF00D)};

    void SetUp() override
    {
        default_mock_buffer = std::make_shared<StrictMock<MBuffer>>();
        EXPECT_CALL(*default_mock_buffer, getBuffer).WillRepeatedly(Return(buffer_pointer));
        EXPECT_CALL(*default_mock_buffer, getLength).WillRepeatedly(Return(1));

        default_mock_file = std::make_shared<StrictMock<MFile>>();
        EXPECT_CALL(*default_mock_file, handle).WillRepeatedly(Return(file_handle));

        io_params                      = std::make_unique<hipFileIOParams_t>();
        io_params->u.batch.devPtr_base = const_cast<void *>(buffer_pointer);
        io_params->u.batch.size        = 1;
        io_params->fh                  = file_handle;
        io_params->mode                = hipFileBatch;
        io_params->opcode              = hipFileBatchRead;
    }

    HipFileBatch()
    {
        batch_map.clear();
    }
};

TEST_F(HipFileBatch, CreateOperationRead)
{
    io_params->opcode = hipFileBatchRead;

    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};
}

TEST_F(HipFileBatch, CreateOperationWrite)
{
    io_params->opcode = hipFileBatchWrite;

    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};
}

TEST_F(HipFileBatch, CreateOperationBadBuffer)
{
    EXPECT_CALL(*default_mock_buffer, getBuffer).WillOnce(Return(reinterpret_cast<void *>(0xFACEFEED)));
    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadFileHandle)
{
    EXPECT_CALL(*default_mock_file, handle).WillOnce(Return(reinterpret_cast<hipFileHandle_t>(0xFACEFEED)));
    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadbufferOffsetIsNegative)
{
    io_params->u.batch.devPtr_offset = -1;

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadBufferOffsetExceedsBuffer)
{
    io_params->u.batch.devPtr_offset = 1;

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadOperationLargerThanBuffer)
{
    io_params->u.batch.size = 2;

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadOperationLargerThanBufferWithOffset)
{
    EXPECT_CALL(*default_mock_buffer, getLength).WillRepeatedly(Return(10));
    io_params->u.batch.devPtr_offset = 6;
    io_params->u.batch.size          = 5;

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadFileOffsetIsNegative)
{
    io_params->u.batch.file_offset = -1;

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadOpcode)
{
    io_params->opcode = invalidEnum<hipFileOpcode_t>(maxEnum<hipFileOpcode_t>());

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadMode)
{
    io_params->mode = invalidEnum<hipFileBatchMode_t>(maxEnum<hipFileBatchMode_t>());

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateContext)
{
    hipFileBatchHandle_t handle = batch_map.createContext(32);

    ASSERT_NE(nullptr, handle);
}

TEST_F(HipFileBatch, CreateTwoContexts)
{
    hipFileBatchHandle_t handle1 = batch_map.createContext(1);
    hipFileBatchHandle_t handle2 = batch_map.createContext(1);

    ASSERT_NE(handle1, handle2);
}

TEST_F(HipFileBatch, CreateContextZeroCapacity)
{
    ASSERT_THROW(batch_map.createContext(0), std::invalid_argument);
}

TEST_F(HipFileBatch, CreateContextMaxCapacity)
{
    hipFileBatchHandle_t handle = batch_map.createContext(BatchContext::MAX_SIZE);

    ASSERT_NE(nullptr, handle);
}

TEST_F(HipFileBatch, CreateContextOverCapacity)
{
    ASSERT_THROW(batch_map.createContext(BatchContext::MAX_SIZE + 1), std::invalid_argument);
}

TEST_F(HipFileBatch, DestroyContext)
{
    hipFileBatchHandle_t handle = batch_map.createContext(1);

    batch_map.destroyContext(handle);
}

TEST_F(HipFileBatch, DestroyMissingContext)
{
    ASSERT_THROW(batch_map.destroyContext(reinterpret_cast<hipFileBatchHandle_t>(1)), InvalidBatchHandle);
}

TEST_F(HipFileBatch, DestroyNullptrContext)
{
    ASSERT_THROW(batch_map.destroyContext(nullptr), InvalidBatchHandle);
}

TEST_F(HipFileBatch, GetContext)
{
    hipFileBatchHandle_t           handle  = batch_map.createContext(1);
    std::shared_ptr<IBatchContext> context = batch_map.get(handle);

    ASSERT_EQ(handle, context.get());
}

TEST_F(HipFileBatch, GetNullptrContext)
{
    ASSERT_THROW(batch_map.get(nullptr), InvalidBatchHandle);
}

TEST_F(HipFileBatch, GetInvalidContext)
{
    ASSERT_THROW(batch_map.get(reinterpret_cast<hipFileBatchHandle_t>(0xBAC00001)), InvalidBatchHandle);
}

TEST_F(HipFileBatch, GetDestroyedContext)
{
    hipFileBatchHandle_t handle = batch_map.createContext(1);
    batch_map.destroyContext(handle);
    ASSERT_THROW(batch_map.get(handle), InvalidBatchHandle);
}

struct HipFileBatchContext : public HipFileUnopened {
    BatchContextMap                           batch_map = BatchContextMap{};
    std::shared_ptr<IBatchContext>            _context;
    unsigned                                  _context_capacity = 2;
    std::unique_ptr<StrictMock<MDriverState>> mock_driver_state;

    hipFileIOParams_t                    io_params{};
    std::shared_ptr<StrictMock<MBuffer>> default_mock_buffer;
    int                                  default_mock_buffer_length = 1;
    std::shared_ptr<StrictMock<MFile>>   default_mock_file;

    void SetUp() override
    {
        default_mock_buffer = std::make_shared<StrictMock<MBuffer>>();
        EXPECT_CALL(*default_mock_buffer, getBuffer).WillRepeatedly(Return(reinterpret_cast<void *>(0x123)));
        EXPECT_CALL(*default_mock_buffer, getLength).WillRepeatedly(Return(default_mock_buffer_length));

        default_mock_file = std::make_shared<StrictMock<MFile>>();
        EXPECT_CALL(*default_mock_file, handle).WillRepeatedly(Return(default_mock_file.get()));

        file_buffer_pair default_fb_pair = {default_mock_file, default_mock_buffer};

        io_params.u.batch.devPtr_base = default_mock_buffer->getBuffer();
        io_params.u.batch.size        = 1;
        io_params.fh                  = default_mock_file->handle();
        io_params.mode                = hipFileBatch;
        io_params.opcode              = hipFileBatchRead;

        mock_driver_state = std::make_unique<StrictMock<MDriverState>>();
        _context          = batch_map.get(batch_map.createContext(_context_capacity));
        // May be overridden with EXPECT_CALL in the test.
        EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillRepeatedly(Return(std::move(default_fb_pair)));
    }

    void TearDown() override
    {
        batch_map.destroyContext(_context.get());
        mock_driver_state.reset();
    }
};

TEST_F(HipFileBatchContext, SubmitSingleGoodOp)
{
    _context->submit_operations(&io_params, 1);
}

TEST_F(HipFileBatchContext, SubmitZeroOperations)
{
    _context->submit_operations(nullptr, 0);
}

TEST_F(HipFileBatchContext, SubmitOverCapacity)
{
    // We should fail before we ever try touching the nullptr.
    ASSERT_THROW(_context->submit_operations(nullptr, _context_capacity + 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitOverCapacityOverMultipleSubmissions)
{
    // Submit one at a time up to the capacity.
    // In the future we might care that we are submitting the same operation.
    for (unsigned i = 0; i < _context_capacity; i++) {
        printf("i: %u\n", i);
        _context->submit_operations(&io_params, 1);
    }

    ASSERT_THROW(_context->submit_operations(nullptr, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadBuffer)
{
    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillOnce(Throw(BufferNotRegistered()));
    ASSERT_THROW(_context->submit_operations(&io_params, 1), BufferNotRegistered);
}

TEST_F(HipFileBatchContext, SubmitSingleBadFileHandle)
{
    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillOnce(Throw(FileNotRegistered()));
    ASSERT_THROW(_context->submit_operations(&io_params, 1), FileNotRegistered);
}

// BatchOperation is not mocked.
TEST_F(HipFileBatchContext, SubmitSingleBadParamBufferOffsetNegative)
{
    hipFileIOParams_t bad_io_params     = io_params;
    bad_io_params.u.batch.devPtr_offset = -1;
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamBufferOffsetTooLarge)
{
    hipFileIOParams_t bad_io_params     = io_params;
    bad_io_params.u.batch.devPtr_offset = default_mock_buffer_length;
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamIOSizeTooLarge)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.u.batch.size      = static_cast<size_t>(default_mock_buffer_length + 1);
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamFileOffsetNegative)
{
    hipFileIOParams_t bad_io_params   = io_params;
    bad_io_params.u.batch.file_offset = -1;
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamOpcodeInvalid)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.opcode            = invalidEnum<hipFileOpcode_t>(maxEnum<hipFileOpcode_t>());
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamModeInvalid)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.mode              = invalidEnum<hipFileBatchMode_t>(maxEnum<hipFileBatchMode_t>());
    ASSERT_THROW(_context->submit_operations(&bad_io_params, 1), std::invalid_argument);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
