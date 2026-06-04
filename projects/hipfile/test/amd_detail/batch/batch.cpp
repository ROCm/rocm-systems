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
#include "mbatch.h"
#include "mbuffer.h"
#include "mfile.h"
#include "mstate.h"
#include "mthread-pool.h"
#include "state.h"

#include "gmock/gmock.h"
#include <array>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using ::testing::_;
using ::testing::ByMove;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

struct HipFileBatch : public HipFileUnopened {
    BatchContextMap                           batch_map = BatchContextMap{};
    std::unique_ptr<hipFileIOParams_t>        io_params;
    std::unique_ptr<StrictMock<MDriverState>> mock_driver_state;
    std::shared_ptr<StrictMock<MBuffer>>      default_mock_buffer;
    std::shared_ptr<StrictMock<MFile>>        default_mock_file;
    const hipFileHandle_t                     file_handle{reinterpret_cast<void *>(0xDEADBEEF)};
    void *const                               buffer_pointer{reinterpret_cast<void *>(0x0BADF00D)};
    int                                       cookie{};

    void SetUp() override
    {
        mock_driver_state   = std::make_unique<StrictMock<MDriverState>>();
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
        io_params->cookie              = &cookie;
    }
};

TEST_F(HipFileBatch, CreateOperationRead)
{
    io_params->opcode = hipFileBatchRead;

    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    ASSERT_EQ(op.event().status, hipFileWaiting);
}

TEST_F(HipFileBatch, CreateOperationWrite)
{
    io_params->opcode = hipFileBatchWrite;

    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    ASSERT_EQ(op.event().status, hipFileWaiting);
}

TEST_F(HipFileBatch, MarkOperationPending)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();

    ASSERT_EQ(op.event().status, hipFilePending);
}

TEST_F(HipFileBatch, TryCancelWaitingOperationIsNoOp)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.tryCancel();

    ASSERT_EQ(op.event().status, hipFileWaiting);
}

TEST_F(HipFileBatch, MarkPendingPendingOperationThrowsInvalidStateTransition)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();

    ASSERT_THROW(op.markPending(), InvalidStateTransition);
    ASSERT_EQ(op.event().status, hipFilePending);
}

TEST_F(HipFileBatch, CancelPendingOperation)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();
    op.tryCancel();

    ASSERT_EQ(op.event().status, hipFileCanceled);
}

TEST_F(HipFileBatch, CancelPendingOperationIsIdempotent)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();
    op.tryCancel();
    op.tryCancel();

    ASSERT_EQ(op.event().status, hipFileCanceled);
}

TEST_F(HipFileBatch, MarkPendingCanceledOperationThrowsInvalidStateTransition)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();
    op.tryCancel();

    ASSERT_THROW(op.markPending(), InvalidStateTransition);

    ASSERT_EQ(op.event().status, hipFileCanceled);
}

TEST_F(HipFileBatch, RunCanceledOperationReturnsImmediately)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    op.markPending();
    op.tryCancel();
    EXPECT_CALL(*mock_driver_state, getBackends).Times(0);
    op.run();

    hipFileIOEvents_t event = op.event();
    ASSERT_EQ(event.status, hipFileCanceled);
}

TEST_F(HipFileBatch, RunInternalStateErrorRecordsFailedEvent)
{
    BatchOperation op = BatchOperation{std::move(io_params), default_mock_buffer, default_mock_file};

    // Running waiting op results in invalid transition
    op.run();

    hipFileIOEvents_t event = op.event();
    ASSERT_EQ(event.status, hipFileFailed);
    ASSERT_EQ(event.ret, static_cast<size_t>(-hipFileInternalError));
    ASSERT_EQ(event.cookie, &cookie);
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
    io_params->opcode = invalidEnum<hipFileOpcode_t>(-1);

    EXPECT_THROW(BatchOperation(std::move(io_params), default_mock_buffer, default_mock_file),
                 std::invalid_argument);
}

TEST_F(HipFileBatch, CreateOperationBadMode)
{
    io_params->mode = invalidEnum<hipFileBatchMode_t>(-1);

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
    std::unique_ptr<StrictMock<MThreadPool>>  mock_thread_pool;
    StrictMock<MTaskGroup>                   *mock_task_group = nullptr;
    StrictMock<MBatchOperationFactory>        mock_operation_factory;

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
        mock_thread_pool  = std::make_unique<StrictMock<MThreadPool>>();
        mock_task_group   = expectTaskGroupCreated();
        _context          = batch_map.get(batch_map.createContext(_context_capacity));
        // May be overridden with EXPECT_CALL in the test.
        EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillRepeatedly(Return(std::move(default_fb_pair)));
    }

    StrictMock<MTaskGroup> *expectTaskGroupCreated()
    {
        auto task_group = std::make_unique<StrictMock<MTaskGroup>>();
        auto raw        = task_group.get();
        EXPECT_CALL(*mock_thread_pool, makeTaskGroup()).WillOnce(Return(ByMove(std::move(task_group))));
        return raw;
    }

    std::shared_ptr<StrictMock<MBatchOperation>> makeOperation()
    {
        auto op = std::make_shared<StrictMock<MBatchOperation>>();
        EXPECT_CALL(*op, markPending()).Times(1);
        return op;
    }

    void expectOperationFactoryCreates(const std::vector<std::shared_ptr<StrictMock<MBatchOperation>>> &ops)
    {
        auto index = std::make_shared<size_t>(0);
        EXPECT_CALL(mock_operation_factory, create(_, _, _))
            .Times(static_cast<int>(ops.size()))
            .WillRepeatedly([ops, index](std::unique_ptr<const hipFileIOParams_t>, std::shared_ptr<IBuffer>,
                                         std::shared_ptr<IFile>) -> std::shared_ptr<IBatchOperation> {
                return ops[(*index)++];
            });
    }

    void submitMockOperations(const std::vector<std::shared_ptr<StrictMock<MBatchOperation>>> &ops)
    {
        std::vector<hipFileIOParams_t> params(ops.size(), io_params);

        expectOperationFactoryCreates(ops);
        EXPECT_CALL(*mock_task_group, run(_)).Times(static_cast<int>(ops.size()));

        _context->submitOperations(params.data(), static_cast<unsigned>(params.size()),
                                   &mock_operation_factory);
    }
};

TEST_F(HipFileBatchContext, SubmitSingleGoodOp)
{
    EXPECT_CALL(*mock_task_group, run(_)).Times(1);

    ASSERT_NO_THROW(_context->submitOperations(&io_params, 1));
}

TEST_F(HipFileBatchContext, SubmitSingleGoodWriteOp)
{
    io_params.opcode = hipFileBatchWrite;

    EXPECT_CALL(*mock_task_group, run(_)).Times(1);

    ASSERT_NO_THROW(_context->submitOperations(&io_params, 1));
}

TEST_F(HipFileBatchContext, SubmitMultipleGoodOps)
{
    std::array<hipFileIOParams_t, 2> params{io_params, io_params};

    EXPECT_CALL(*mock_task_group, run(_)).Times(params.size());

    ASSERT_NO_THROW(_context->submitOperations(params.data(), params.size()));
}

TEST_F(HipFileBatchContext, nullptrParamsThrows)
{
    ASSERT_THROW(_context->submitOperations(nullptr, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, zeroNumParamsThrows)
{
    ASSERT_THROW(_context->submitOperations(&io_params, 0), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitUsesProvidedOperationFactory)
{
    auto op = makeOperation();

    EXPECT_CALL(mock_operation_factory, create(_, _, _))
        .WillOnce([this, op](std::unique_ptr<const hipFileIOParams_t> params, std::shared_ptr<IBuffer> buffer,
                             std::shared_ptr<IFile> file) -> std::shared_ptr<IBatchOperation> {
            EXPECT_EQ(params->fh, io_params.fh);
            EXPECT_EQ(params->u.batch.devPtr_base, io_params.u.batch.devPtr_base);
            EXPECT_EQ(buffer.get(), default_mock_buffer.get());
            EXPECT_EQ(file.get(), default_mock_file.get());
            return op;
        });
    EXPECT_CALL(*mock_task_group, run(_)).Times(1);

    _context->submitOperations(&io_params, 1, &mock_operation_factory);
}

TEST_F(HipFileBatchContext, SubmittedFactoryOperationRunsFromQueuedWork)
{
    std::function<void()> enqueued_work;
    auto                  op = makeOperation();

    expectOperationFactoryCreates({op});
    EXPECT_CALL(*mock_task_group, run(_)).WillOnce([&enqueued_work](std::function<void()> work) {
        enqueued_work = std::move(work);
    });

    _context->submitOperations(&io_params, 1, &mock_operation_factory);
    ASSERT_TRUE(enqueued_work);

    EXPECT_CALL(*op, run()).Times(1);
    enqueued_work();
}

TEST_F(HipFileBatchContext, SubmitFactoryFailureFailsBatch)
{
    std::array<hipFileIOParams_t, 2> params{io_params, io_params};
    auto                             op = std::make_shared<StrictMock<MBatchOperation>>();

    EXPECT_CALL(mock_operation_factory, create(_, _, _))
        .WillOnce([op](std::unique_ptr<const hipFileIOParams_t>, std::shared_ptr<IBuffer>,
                       std::shared_ptr<IFile>) -> std::shared_ptr<IBatchOperation> { return op; })
        .WillOnce([](std::unique_ptr<const hipFileIOParams_t>, std::shared_ptr<IBuffer>,
                     std::shared_ptr<IFile>) -> std::shared_ptr<IBatchOperation> {
            throw std::invalid_argument("factory error");
        });
    EXPECT_CALL(*mock_task_group, run(_)).Times(0);

    ASSERT_THROW(_context->submitOperations(params.data(), static_cast<unsigned>(params.size()),
                                            &mock_operation_factory),
                 std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitOverCapacity)
{
    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).Times(0);
    ASSERT_THROW(_context->submitOperations(&io_params, _context_capacity + 1), BatchFull);
}

TEST_F(HipFileBatchContext, SubmitOverCapacityOverMultipleSubmissions)
{
    // Submit one at a time up to the capacity.
    // In the future we might care that we are submitting the same operation.
    EXPECT_CALL(*mock_task_group, run(_)).Times(static_cast<int>(_context_capacity));

    for (unsigned i = 0; i < _context_capacity; i++) {
        _context->submitOperations(&io_params, 1);
    }

    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).Times(0);
    ASSERT_THROW(_context->submitOperations(&io_params, 1), BatchFull);
}

TEST_F(HipFileBatchContext, SubmitBatchWithBadOpThrows)
{
    std::array<hipFileIOParams_t, 2> params{io_params, io_params};
    params[1].opcode = invalidEnum<hipFileOpcode_t>(-1);

    EXPECT_CALL(*mock_task_group, run(_)).Times(0);

    ASSERT_THROW(_context->submitOperations(params.data(), params.size()), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadBuffer)
{
    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillOnce(Throw(BufferNotRegistered()));
    ASSERT_THROW(_context->submitOperations(&io_params, 1), BufferNotRegistered);
}

TEST_F(HipFileBatchContext, SubmitSingleBadFileHandle)
{
    EXPECT_CALL(*mock_driver_state, getFileAndBuffer).WillOnce(Throw(FileNotRegistered()));
    ASSERT_THROW(_context->submitOperations(&io_params, 1), FileNotRegistered);
}

// BatchOperation is not mocked.
TEST_F(HipFileBatchContext, SubmitSingleBadParamBufferOffsetNegative)
{
    hipFileIOParams_t bad_io_params     = io_params;
    bad_io_params.u.batch.devPtr_offset = -1;
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamBufferOffsetTooLarge)
{
    hipFileIOParams_t bad_io_params     = io_params;
    bad_io_params.u.batch.devPtr_offset = default_mock_buffer_length;
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamIOSizeTooLarge)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.u.batch.size      = static_cast<size_t>(default_mock_buffer_length + 1);
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamFileOffsetNegative)
{
    hipFileIOParams_t bad_io_params   = io_params;
    bad_io_params.u.batch.file_offset = -1;
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamOpcodeInvalid)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.opcode            = invalidEnum<hipFileOpcode_t>(-1);
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmitSingleBadParamModeInvalid)
{
    hipFileIOParams_t bad_io_params = io_params;
    bad_io_params.mode              = invalidEnum<hipFileBatchMode_t>(-1);
    ASSERT_THROW(_context->submitOperations(&bad_io_params, 1), std::invalid_argument);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
