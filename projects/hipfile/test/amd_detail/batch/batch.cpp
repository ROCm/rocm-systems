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

#include <array>
#include <chrono>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using ::testing::_;
using ::testing::AllOf;
using ::testing::ByMove;
using ::testing::Field;
using ::testing::Return;
using ::testing::StrictMock;
using ::testing::Throw;
using ::testing::UnorderedElementsAre;

using namespace hipFile;

static_assert(std::ratio_equal_v<std::chrono::steady_clock::period, std::nano>,
              "Tests require a nanosecond steady_clock period");

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

TEST_F(HipFileBatch, OnlyFinalStatesAreTerminal)
{
    using namespace batchOperationState;

    static_assert(!Waiting::isTerminal());
    static_assert(!Pending::isTerminal());
    static_assert(!Running::isTerminal());

    static_assert(Complete::isTerminal());
    static_assert(Canceled::isTerminal());
    static_assert(Invalid::isTerminal());
    static_assert(Timeout::isTerminal());
    static_assert(Failed::isTerminal());
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

TEST_F(HipFileBatch, DestroyContextRemovesHandle)
{
    hipFileBatchHandle_t handle = batch_map.createContext(1);

    batch_map.destroyContext(handle);

    ASSERT_THROW(batch_map.get(handle), InvalidBatchHandle);
}

TEST_F(HipFileBatch, DestroyContextPreservesOtherContexts)
{
    hipFileBatchHandle_t handle1 = batch_map.createContext(1);
    hipFileBatchHandle_t handle2 = batch_map.createContext(1);

    batch_map.destroyContext(handle1);

    ASSERT_THROW(batch_map.get(handle1), InvalidBatchHandle);
    ASSERT_NE(batch_map.get(handle2), nullptr);
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

TEST_F(HipFileBatch, ClearEmptyMapSucceeds)
{
    ASSERT_NO_THROW(batch_map.clear());
}

TEST_F(HipFileBatch, ClearRemovesAllContexts)
{
    hipFileBatchHandle_t handle1 = batch_map.createContext(1);
    hipFileBatchHandle_t handle2 = batch_map.createContext(1);

    batch_map.clear();

    ASSERT_THROW(batch_map.get(handle1), InvalidBatchHandle);
    ASSERT_THROW(batch_map.get(handle2), InvalidBatchHandle);
}

TEST_F(HipFileBatch, ClearIsIdempotent)
{
    batch_map.createContext(1);

    batch_map.clear();

    ASSERT_NO_THROW(batch_map.clear());
}

struct HipFileBatchContext : public HipFileUnopened {
    BatchContextMap                          batch_map = BatchContextMap{};
    std::shared_ptr<IBatchContext>           _context;
    unsigned                                 _context_capacity = 2;
    std::unique_ptr<StrictMock<MThreadPool>> mock_thread_pool;
    StrictMock<MTaskGroup>                  *mock_task_group = nullptr;
    std::vector<std::function<void()>>       enqueued_tasks;

    void SetUp() override
    {
        mock_thread_pool = std::make_unique<StrictMock<MThreadPool>>();
        mock_task_group  = expectTaskGroupCreated();
        _context         = batch_map.get(batch_map.createContext(_context_capacity));
    }

    StrictMock<MTaskGroup> *expectTaskGroupCreated()
    {
        auto task_group = std::make_unique<StrictMock<MTaskGroup>>();
        auto raw        = task_group.get();
        EXPECT_CALL(*mock_thread_pool, makeTaskGroup()).WillOnce(Return(ByMove(std::move(task_group))));
        return raw;
    }

    static void expectTaskQueueFlushed(StrictMock<MTaskGroup> *tg)
    {
        EXPECT_CALL(*tg, cancel()).Times(1);
        EXPECT_CALL(*tg, wait()).Times(1);
    }

    std::shared_ptr<StrictMock<MBatchOperation>> makeOperation()
    {
        auto op = std::make_shared<StrictMock<MBatchOperation>>();
        EXPECT_CALL(*op, markPending()).Times(1);
        EXPECT_CALL(*op, tryCancel()).Times(testing::AnyNumber()).WillRepeatedly(Return(false));
        return op;
    }

    void submitMockOperations(const std::vector<std::shared_ptr<StrictMock<MBatchOperation>>> &ops)
    {
        EXPECT_CALL(*mock_task_group, run(_))
            .Times(static_cast<int>(ops.size()))
            .WillRepeatedly(
                [this](std::function<void()> work) { enqueued_tasks.push_back(std::move(work)); });
        _context->submitOperations(BatchOperations{ops.begin(), ops.end()});
    }

    /// Run the work queued for the operation submitted at @p index, which moves
    /// the operation from submitted to completed.
    void completeOperation(const std::shared_ptr<StrictMock<MBatchOperation>> &op, size_t index)
    {
        EXPECT_CALL(*op, run()).Times(1);
        enqueued_tasks.at(index)();
    }

    /// Complete every operation of a single submission, in submission order.
    void completeOperations(const std::vector<std::shared_ptr<StrictMock<MBatchOperation>>> &ops)
    {
        for (size_t i = 0; i < ops.size(); i++) {
            completeOperation(ops[i], i);
        }
    }
};

TEST_F(HipFileBatchContext, SubmitSingleGoodOp)
{
    auto op = makeOperation();
    EXPECT_CALL(*mock_task_group, run(_)).Times(1);

    ASSERT_NO_THROW(_context->submitOperations(BatchOperations{op}));
}

TEST_F(HipFileBatchContext, SubmitMultipleGoodOps)
{
    auto op1 = makeOperation();
    auto op2 = makeOperation();

    EXPECT_CALL(*mock_task_group, run(_)).Times(2);

    ASSERT_NO_THROW(_context->submitOperations(BatchOperations{op1, op2}));
}

TEST_F(HipFileBatchContext, EmptyOperationsThrows)
{
    ASSERT_THROW(_context->submitOperations({}), std::invalid_argument);
}

TEST_F(HipFileBatchContext, SubmittedOperationRunsFromQueuedWork)
{
    std::function<void()> enqueued_work;
    auto                  op = makeOperation();

    EXPECT_CALL(*mock_task_group, run(_)).WillOnce([&enqueued_work](std::function<void()> work) {
        enqueued_work = std::move(work);
    });

    _context->submitOperations(BatchOperations{op});
    ASSERT_TRUE(enqueued_work);

    EXPECT_CALL(*op, run()).Times(1);
    enqueued_work();
}

TEST_F(HipFileBatchContext, SubmitOverCapacity)
{
    BatchOperations ops;
    for (unsigned i = 0; i <= _context_capacity; i++) {
        ops.push_back(std::make_shared<StrictMock<MBatchOperation>>());
    }

    ASSERT_THROW(_context->submitOperations(std::move(ops)), BatchFull);
}

TEST_F(HipFileBatchContext, SubmitOverCapacityOverMultipleSubmissions)
{
    EXPECT_CALL(*mock_task_group, run(_)).Times(static_cast<int>(_context_capacity));

    for (unsigned i = 0; i < _context_capacity; i++) {
        _context->submitOperations(BatchOperations{makeOperation()});
    }

    auto op = std::make_shared<StrictMock<MBatchOperation>>();
    ASSERT_THROW(_context->submitOperations(BatchOperations{op}), BatchFull);
}

TEST_F(HipFileBatchContext, TaskGroupSubmissionFailureCancelsEntireSubmission)
{
    int  queued_cookie{};
    int  unsubmitted_cookie{};
    auto queued_op      = std::make_shared<StrictMock<MBatchOperation>>();
    auto unsubmitted_op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*queued_op, markPending()).Times(1);
    EXPECT_CALL(*unsubmitted_op, markPending()).Times(1);
    EXPECT_CALL(*queued_op, tryCancel()).WillOnce(Return(true));
    EXPECT_CALL(*unsubmitted_op, tryCancel()).WillOnce(Return(true));
    EXPECT_CALL(*mock_task_group, run(_))
        .WillOnce([this](std::function<void()> work) { enqueued_tasks.push_back(std::move(work)); })
        .WillOnce(Throw(std::runtime_error{"Failed to submit task"}));

    ASSERT_THROW(_context->submitOperations(BatchOperations{queued_op, unsubmitted_op}), std::runtime_error);
    ASSERT_EQ(enqueued_tasks.size(), 1);

    hipFileIOEvents_t unsubmitted_event{&unsubmitted_cookie, hipFileCanceled, 0};
    EXPECT_CALL(*unsubmitted_op, event()).WillOnce(Return(unsubmitted_event));
    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::nullopt));
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.cookie, &unsubmitted_cookie);
    ASSERT_EQ(event.status, hipFileCanceled);

    EXPECT_CALL(*queued_op, run()).Times(1);
    enqueued_tasks.front()();

    hipFileIOEvents_t queued_event{&queued_cookie, hipFileCanceled, 0};
    EXPECT_CALL(*queued_op, event()).WillOnce(Return(queued_event));
    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::nullopt));
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.cookie, &queued_cookie);
    ASSERT_EQ(event.status, hipFileCanceled);
}

TEST_F(HipFileBatchContext, SubmitWithContextRetainedBeforeDestroyIsRejected)
{
    hipFileBatchHandle_t handle           = _context.get();
    auto                 retained_context = batch_map.get(handle);
    auto                 op               = std::make_shared<StrictMock<MBatchOperation>>();

    expectTaskQueueFlushed(mock_task_group);
    batch_map.destroyContext(handle);

    ASSERT_THROW(retained_context->submitOperations(BatchOperations{op}), InvalidBatchHandle);
}

TEST_F(HipFileBatchContext, SubmittedWorkKeepsContextAliveUntilReleased)
{
    std::function<void()>        enqueued_work;
    std::weak_ptr<IBatchContext> weak_context = _context;
    auto                         op           = makeOperation();

    // enqueued_work will capture the function that has been passed to the thread pool
    EXPECT_CALL(*mock_task_group, run(_)).WillOnce([&enqueued_work](std::function<void()> work) {
        enqueued_work = std::move(work);
    });

    ASSERT_NO_THROW(_context->submitOperations(BatchOperations{op}));
    // enqueued_work has been assigned
    ASSERT_TRUE(enqueued_work);

    expectTaskQueueFlushed(mock_task_group);
    // Destroy the context and our shared_ptr to it
    batch_map.destroyContext(_context.get());
    _context.reset();

    // BatchContext is still valid because it was captured by lambda
    ASSERT_FALSE(weak_context.expired());

    // Assigning empty function will call destructor for the lambda that was assigned
    enqueued_work = {};

    // The shared_ptr has been destroyed
    ASSERT_TRUE(weak_context.expired());
}

TEST_F(HipFileBatchContext, GetStatusNoOutstandingReturnsNothing)
{
    unsigned          nr = 1;
    hipFileIOEvents_t event{};

    ASSERT_NO_THROW(
        _context->getStatus(1, &nr, &event, std::chrono::steady_clock::now() + std::chrono::seconds{1}));
    ASSERT_EQ(nr, 0);
}

TEST_F(HipFileBatchContext, GetStatusNullNumEventsThrowsInvalidArgument)
{
    hipFileIOEvents_t event{};

    ASSERT_THROW(_context->getStatus(0, nullptr, &event, std::nullopt), std::invalid_argument);
}

TEST_F(HipFileBatchContext, GetStatusNullEventsThrowsInvalidArgument)
{
    unsigned nr = 0;

    ASSERT_THROW(_context->getStatus(0, &nr, nullptr, std::nullopt), std::invalid_argument);
}

TEST_F(HipFileBatchContext, GetStatusMinimumExceedsCapacityThrowsInvalidArgument)
{
    unsigned          nr = 1;
    hipFileIOEvents_t event{};

    ASSERT_THROW(_context->getStatus(2, &nr, &event, std::nullopt), std::invalid_argument);
}

TEST_F(HipFileBatchContext, GetStatusMinimumExceedsBatchCapacityThrowsInvalidArgument)
{
    const unsigned                 min_nr = _context_capacity + 1;
    unsigned                       nr     = min_nr;
    std::vector<hipFileIOEvents_t> events(min_nr);

    // The event buffer holds min_nr events, so only the batch capacity limit can reject this.
    ASSERT_THROW(_context->getStatus(min_nr, &nr, events.data(), std::nullopt), std::invalid_argument);
}

TEST_F(HipFileBatchContext, GetStatusReturnsCompletedOperationAndConsumesIt)
{
    int               cookie{};
    auto              op = makeOperation();
    hipFileIOEvents_t completed_event{&cookie, hipFileComplete, 9};
    EXPECT_CALL(*op, event()).WillOnce(Return(completed_event));
    submitMockOperations({op});
    completeOperation(op, 0);

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    ASSERT_NO_THROW(_context->getStatus(1, &nr, &event, std::nullopt));

    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.cookie, &cookie);
    ASSERT_EQ(event.status, hipFileComplete);
    ASSERT_EQ(event.ret, 9);

    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::nullopt));
    ASSERT_EQ(nr, 0);
}

TEST_F(HipFileBatchContext, GetStatusReturnsFailedAndCanceledOperations)
{
    int               failed_cookie{};
    int               canceled_cookie{};
    auto              failed_op      = makeOperation();
    auto              canceled_op    = makeOperation();
    hipFileIOEvents_t failed_event   = {&failed_cookie, hipFileFailed,
                                        static_cast<size_t>(-hipFileInternalError)};
    hipFileIOEvents_t canceled_event = {&canceled_cookie, hipFileCanceled, 0};
    EXPECT_CALL(*failed_op, event()).WillRepeatedly(Return(failed_event));
    EXPECT_CALL(*canceled_op, event()).WillRepeatedly(Return(canceled_event));
    submitMockOperations({failed_op, canceled_op});
    completeOperations({failed_op, canceled_op});

    unsigned                         nr = 2;
    std::array<hipFileIOEvents_t, 2> events{};
    ASSERT_NO_THROW(_context->getStatus(2, &nr, events.data(), std::nullopt));

    ASSERT_EQ(nr, 2);
    ASSERT_THAT(events, UnorderedElementsAre(
                            AllOf(Field(&hipFileIOEvents_t::cookie, static_cast<void *>(&failed_cookie)),
                                  Field(&hipFileIOEvents_t::status, hipFileFailed),
                                  Field(&hipFileIOEvents_t::ret, static_cast<size_t>(-hipFileInternalError))),
                            AllOf(Field(&hipFileIOEvents_t::cookie, static_cast<void *>(&canceled_cookie)),
                                  Field(&hipFileIOEvents_t::status, hipFileCanceled))));
}

TEST_F(HipFileBatchContext, GetStatusDoesNotReturnOperationUntilItCompletes)
{
    auto              op = makeOperation();
    hipFileIOEvents_t completed_event{nullptr, hipFileComplete, 1};
    EXPECT_CALL(*op, event()).WillOnce(Return(completed_event));
    submitMockOperations({op});

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::chrono::steady_clock::now()));

    ASSERT_EQ(nr, 0);

    completeOperation(op, 0);

    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::nullopt));
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.status, hipFileComplete);
}

TEST_F(HipFileBatchContext, GetStatusWaitsForOperationToComplete)
{
    int               cookie{};
    auto              op = makeOperation();
    hipFileIOEvents_t completed_event{&cookie, hipFileComplete, 3};
    EXPECT_CALL(*op, event()).WillOnce(Return(completed_event));
    submitMockOperations({op});

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    std::thread       waiter{[this, &nr, &event]() { _context->getStatus(1, &nr, &event, std::nullopt); }};

    completeOperation(op, 0);
    waiter.join();

    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.cookie, &cookie);
    ASSERT_EQ(event.status, hipFileComplete);
    ASSERT_EQ(event.ret, 3);
}

TEST_F(HipFileBatchContext, CompletedOperationsOccupyCapacityUntilRetrieved)
{
    auto op1 = makeOperation();
    auto op2 = makeOperation();
    EXPECT_CALL(*op1, event()).WillOnce(Return(hipFileIOEvents_t{nullptr, hipFileComplete, 1}));
    EXPECT_CALL(*op2, event()).WillOnce(Return(hipFileIOEvents_t{nullptr, hipFileComplete, 2}));
    submitMockOperations({op1, op2});
    completeOperations({op1, op2});

    auto full_op = std::make_shared<StrictMock<MBatchOperation>>();
    ASSERT_THROW(_context->submitOperations(BatchOperations{full_op}), BatchFull);

    unsigned                         nr = 2;
    std::array<hipFileIOEvents_t, 2> events{};
    ASSERT_NO_THROW(_context->getStatus(0, &nr, events.data(), std::nullopt));
    ASSERT_EQ(nr, 2);

    ASSERT_NO_THROW(submitMockOperations({makeOperation()}));
}

TEST_F(HipFileBatchContext, GetStatusReturnsAtMostCallerCapacity)
{
    auto op1 = makeOperation();
    auto op2 = makeOperation();
    EXPECT_CALL(*op1, event()).WillRepeatedly(Return(hipFileIOEvents_t{nullptr, hipFileComplete, 1}));
    EXPECT_CALL(*op2, event()).WillRepeatedly(Return(hipFileIOEvents_t{nullptr, hipFileComplete, 2}));
    submitMockOperations({op1, op2});
    completeOperations({op1, op2});

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, std::nullopt));

    ASSERT_EQ(nr, 1);

    nr = 2;
    std::array<hipFileIOEvents_t, 2> events{};
    ASSERT_NO_THROW(_context->getStatus(0, &nr, events.data(), std::nullopt));
    ASSERT_EQ(nr, 1);
}

TEST_F(HipFileBatchContext, GetStatusElapsedDeadlineWillReturnLessThanMin)
{
    int               cookie{};
    auto              complete_op = makeOperation();
    auto              pending_op  = makeOperation();
    hipFileIOEvents_t completed_event{&cookie, hipFileComplete, 4};
    EXPECT_CALL(*complete_op, event()).WillOnce(Return(completed_event));
    submitMockOperations({complete_op, pending_op});
    completeOperation(complete_op, 0);

    unsigned          nr = 2;
    hipFileIOEvents_t event[2];
    ASSERT_NO_THROW(_context->getStatus(2, &nr, &event[0], std::chrono::steady_clock::now()));

    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event[0].cookie, &cookie);
    ASSERT_EQ(event[0].status, hipFileComplete);
    ASSERT_EQ(event[0].ret, 4);
}

TEST(HipFileBatchDeadline, NullTimeoutHasNoDeadline)
{
    ASSERT_EQ(makeBatchDeadline(nullptr), std::nullopt);
}

TEST(HipFileBatchDeadline, ZeroTimeoutIsAnImmediateDeadline)
{
    const auto      now = std::chrono::steady_clock::time_point{std::chrono::seconds{123}};
    struct timespec timeout {
        0, 0
    };

    const auto deadline = makeBatchDeadline(&timeout, now);

    ASSERT_TRUE(deadline.has_value());
    ASSERT_EQ(*deadline, now);
}

TEST(HipFileBatchDeadline, PositiveTimeoutIsOffsetFromNow)
{
    const auto      now = std::chrono::steady_clock::time_point{std::chrono::seconds{123}};
    struct timespec timeout {
        1, 500000000L
    };

    const auto deadline = makeBatchDeadline(&timeout, now);

    ASSERT_TRUE(deadline.has_value());
    ASSERT_EQ(*deadline, now + std::chrono::milliseconds{1500});
}

TEST(HipFileBatchDeadline, MaximumTimeoutFits)
{
    const auto now =
        std::chrono::steady_clock::time_point::max() - std::chrono::seconds{2} - std::chrono::nanoseconds{42};
    struct timespec timeout {
        2, 42
    };

    const auto deadline = makeBatchDeadline(&timeout, now);

    ASSERT_TRUE(deadline.has_value());
    ASSERT_EQ(*deadline, std::chrono::steady_clock::time_point::max());
}

TEST(HipFileBatchDeadline, NanosecondsPastMaximumTimeoutSaturates)
{
    const auto now =
        std::chrono::steady_clock::time_point::max() - std::chrono::seconds{2} - std::chrono::nanoseconds{42};
    struct timespec timeout {
        2, 43
    };

    const auto deadline = makeBatchDeadline(&timeout, now);

    ASSERT_TRUE(deadline.has_value());
    ASSERT_EQ(*deadline, std::chrono::steady_clock::time_point::max());
}

TEST(HipFileBatchDeadline, SecondsPastMaximumTimeoutSaturates)
{
    const auto now =
        std::chrono::steady_clock::time_point::max() - std::chrono::seconds{2} - std::chrono::nanoseconds{42};
    struct timespec timeout {
        3, 0
    };

    const auto deadline = makeBatchDeadline(&timeout, now);

    ASSERT_TRUE(deadline.has_value());
    ASSERT_EQ(*deadline, std::chrono::steady_clock::time_point::max());
}

TEST(HipFileBatchDeadline, NegativeSecondsThrowsInvalidArgument)
{
    struct timespec timeout {
        - 1, 0
    };

    ASSERT_THROW(makeBatchDeadline(&timeout), std::invalid_argument);
}

TEST(HipFileBatchDeadline, NegativeNanosecondsThrowsInvalidArgument)
{
    struct timespec timeout {
        0, -1
    };

    ASSERT_THROW(makeBatchDeadline(&timeout), std::invalid_argument);
}

TEST(HipFileBatchDeadline, UnnormalizedNanosecondsThrowsInvalidArgument)
{
    struct timespec timeout {
        0, 1000000000L
    };

    ASSERT_THROW(makeBatchDeadline(&timeout), std::invalid_argument);
}

TEST_F(HipFileBatchContext, CancelOperationsEmptySucceeds)
{
    EXPECT_CALL(*mock_task_group, cancel()).Times(1);

    ASSERT_NO_THROW(_context->cancelOperations());
}

TEST_F(HipFileBatchContext, CancelOperationsCancelsTaskGroup)
{
    auto op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});

    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*op, tryCancel()).WillOnce(Return(true));

    ASSERT_NO_THROW(_context->cancelOperations());
}

TEST_F(HipFileBatchContext, CancelOperationsAndWaitCancelsAndWaitsForTaskGroup)
{
    auto op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});

    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*mock_task_group, wait()).Times(1);
    EXPECT_CALL(*op, tryCancel()).WillOnce(Return(true));

    ASSERT_NO_THROW(_context->cancelOperationsAndWait());
}

TEST_F(HipFileBatchContext, CancelOperationsCancelsPendingOperations)
{
    auto op1 = std::make_shared<StrictMock<MBatchOperation>>();
    auto op2 = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op1, markPending()).Times(1);
    EXPECT_CALL(*op2, markPending()).Times(1);
    submitMockOperations({op1, op2});

    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*op1, tryCancel()).WillOnce(Return(true));
    EXPECT_CALL(*op2, tryCancel()).WillOnce(Return(true));
    ASSERT_NO_THROW(_context->cancelOperations());
}

TEST_F(HipFileBatchContext, CanceledOperationIsReportedWithoutRunningQueuedWork)
{
    int               cookie{};
    hipFileIOEvents_t canceled_event{&cookie, hipFileCanceled, 0};
    auto              op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});

    // A canceled operation's queued work is dropped by the task group, so cancel
    // is what makes the operation retrievable.
    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*op, tryCancel()).WillOnce(Return(true));
    ASSERT_NO_THROW(_context->cancelOperations());

    EXPECT_CALL(*op, event()).WillOnce(Return(canceled_event));

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    const auto        deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    ASSERT_NO_THROW(_context->getStatus(1, &nr, &event, deadline));
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.cookie, &cookie);
    ASSERT_EQ(event.status, hipFileCanceled);

    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, deadline));
    ASSERT_EQ(nr, 0);
}

TEST_F(HipFileBatchContext, OperationThatCannotBeCanceledIsCompletedByItsQueuedWork)
{
    hipFileIOEvents_t complete_event{nullptr, hipFileComplete, 5};
    auto              op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});

    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*op, tryCancel()).WillOnce(Return(false));
    ASSERT_NO_THROW(_context->cancelOperations());

    unsigned          nr = 1;
    hipFileIOEvents_t event{};
    const auto        deadline = std::chrono::steady_clock::now();
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, deadline));
    ASSERT_EQ(nr, 0);

    EXPECT_CALL(*op, event()).WillOnce(Return(complete_event));
    completeOperation(op, 0);

    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, deadline));
    ASSERT_EQ(nr, 1);
    ASSERT_EQ(event.status, hipFileComplete);
    ASSERT_EQ(event.ret, 5);

    nr = 1;
    ASSERT_NO_THROW(_context->getStatus(0, &nr, &event, deadline));
    ASSERT_EQ(nr, 0);
}

TEST_F(HipFileBatchContext, CancelOperationsLeavesCompletedOperationsUnchanged)
{
    hipFileIOEvents_t complete_event{nullptr, hipFileComplete, 7};
    hipFileIOEvents_t failed_event{nullptr, hipFileFailed, static_cast<size_t>(-hipFileInternalError)};
    auto              complete_op = std::make_shared<StrictMock<MBatchOperation>>();
    auto              failed_op   = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*complete_op, markPending()).Times(1);
    EXPECT_CALL(*failed_op, markPending()).Times(1);
    submitMockOperations({complete_op, failed_op});
    completeOperations({complete_op, failed_op});

    // Both operations already completed, so cancel has nothing left to cancel.
    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    ASSERT_NO_THROW(_context->cancelOperations());

    EXPECT_CALL(*complete_op, event()).WillRepeatedly(Return(complete_event));
    EXPECT_CALL(*failed_op, event()).WillRepeatedly(Return(failed_event));

    unsigned                         nr = 2;
    std::array<hipFileIOEvents_t, 2> events{};
    ASSERT_NO_THROW(_context->getStatus(2, &nr, events.data(), std::nullopt));
    ASSERT_EQ(nr, 2);
    ASSERT_THAT(events, UnorderedElementsAre(AllOf(Field(&hipFileIOEvents_t::status, hipFileComplete),
                                                   Field(&hipFileIOEvents_t::ret, static_cast<size_t>(7))),
                                             AllOf(Field(&hipFileIOEvents_t::status, hipFileFailed),
                                                   Field(&hipFileIOEvents_t::ret,
                                                         static_cast<size_t>(-hipFileInternalError)))));
}

TEST_F(HipFileBatchContext, CancelOperationsCancelableAndTerminalOp)
{
    int               pending_cookie{};
    int               failed_cookie{};
    hipFileIOEvents_t pending_event{&pending_cookie, hipFileCanceled, 0};
    hipFileIOEvents_t failed_event{&failed_cookie, hipFileFailed, static_cast<size_t>(-hipFileInternalError)};
    auto              pending_op = std::make_shared<StrictMock<MBatchOperation>>();
    auto              failed_op  = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*pending_op, markPending()).Times(1);
    EXPECT_CALL(*failed_op, markPending()).Times(1);
    submitMockOperations({pending_op, failed_op});
    completeOperation(failed_op, 1);

    EXPECT_CALL(*mock_task_group, cancel()).Times(1);
    EXPECT_CALL(*pending_op, tryCancel()).WillOnce(Return(true));
    ASSERT_NO_THROW(_context->cancelOperations());

    EXPECT_CALL(*pending_op, event()).WillRepeatedly(Return(pending_event));
    EXPECT_CALL(*failed_op, event()).WillRepeatedly(Return(failed_event));

    unsigned                         nr = 2;
    std::array<hipFileIOEvents_t, 2> events{};
    ASSERT_NO_THROW(_context->getStatus(2, &nr, events.data(), std::nullopt));

    ASSERT_EQ(nr, 2);
    std::vector<hipFileIOEvents_t> returned_events{events.begin(), events.begin() + nr};
    ASSERT_THAT(returned_events,
                UnorderedElementsAre(
                    AllOf(Field(&hipFileIOEvents_t::cookie, static_cast<void *>(&pending_cookie)),
                          Field(&hipFileIOEvents_t::status, hipFileCanceled)),
                    AllOf(Field(&hipFileIOEvents_t::cookie, static_cast<void *>(&failed_cookie)),
                          Field(&hipFileIOEvents_t::status, hipFileFailed),
                          Field(&hipFileIOEvents_t::ret, static_cast<size_t>(-hipFileInternalError)))));
}

TEST_F(HipFileBatchContext, DestroyContextCancelsPendingOperations)
{
    auto op1 = std::make_shared<StrictMock<MBatchOperation>>();
    auto op2 = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op1, markPending()).Times(1);
    EXPECT_CALL(*op2, markPending()).Times(1);
    submitMockOperations({op1, op2});

    EXPECT_CALL(*op1, tryCancel()).WillOnce(Return(true));
    EXPECT_CALL(*op2, tryCancel()).WillOnce(Return(true));
    expectTaskQueueFlushed(mock_task_group);
    batch_map.destroyContext(_context.get());
}

TEST_F(HipFileBatchContext, DestroyContextRemovesHandle)
{
    auto op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});
    hipFileBatchHandle_t handle = _context.get();

    EXPECT_CALL(*op, tryCancel()).WillOnce(Return(true));
    expectTaskQueueFlushed(mock_task_group);
    batch_map.destroyContext(handle);

    ASSERT_THROW(batch_map.get(handle), InvalidBatchHandle);
}

TEST_F(HipFileBatchContext, ClearCancelsOutstandingOperationsAndRemovesHandle)
{
    auto op = std::make_shared<StrictMock<MBatchOperation>>();
    EXPECT_CALL(*op, markPending()).Times(1);
    submitMockOperations({op});
    hipFileBatchHandle_t handle = _context.get();

    EXPECT_CALL(*op, tryCancel()).Times(1);
    expectTaskQueueFlushed(mock_task_group);
    batch_map.clear();

    ASSERT_THROW(batch_map.get(handle), InvalidBatchHandle);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
