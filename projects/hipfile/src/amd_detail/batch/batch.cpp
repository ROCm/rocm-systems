/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "batch.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "hipfile-private.h"
#include "state.h"
#include "thread-pool.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipFile {

using batchOperationState::Canceled;
using batchOperationState::Complete;
using batchOperationState::Failed;
using batchOperationState::Invalid;
using batchOperationState::OperationState;
using batchOperationState::Pending;
using batchOperationState::Running;
using batchOperationState::Timeout;
using batchOperationState::Waiting;

BatchDeadline
makeBatchDeadline(const struct timespec *timeout, std::chrono::time_point<std::chrono::steady_clock> now)
{
    if (timeout == nullptr) {
        return std::nullopt;
    }
    if (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000L) {
        throw std::invalid_argument("Invalid batch status timeout");
    }

    using clock_duration = std::chrono::steady_clock::duration;
    static_assert(std::is_integral_v<clock_duration::rep>,
                  "steady_clock must use an integral representation");
    static_assert(std::is_convertible_v<std::chrono::seconds, clock_duration>,
                  "whole seconds must be representable");
    auto maxWait    = std::chrono::steady_clock::time_point::max() - now;
    auto maxSeconds = std::chrono::duration_cast<std::chrono::seconds>(maxWait);
    auto maxNanos   = std::chrono::duration_cast<std::chrono::nanoseconds>(maxWait - maxSeconds);
    if (std::cmp_greater(timeout->tv_sec, maxSeconds.count()) ||
        (std::cmp_equal(timeout->tv_sec, maxSeconds.count()) &&
         std::cmp_greater(timeout->tv_nsec, maxNanos.count()))) {
        return std::chrono::steady_clock::time_point::max();
    }

    return now + std::chrono::duration_cast<clock_duration>(std::chrono::seconds(timeout->tv_sec)) +
           std::chrono::ceil<clock_duration>(std::chrono::nanoseconds(timeout->tv_nsec));
}

InvalidStateTransition::InvalidStateTransition(const char *from, const char *to)
    : std::logic_error{std::string{"Invalid batch operation state transition: "} + from + " -> " + to}
{
}

namespace batchOperationState {

    Pending Waiting::transitionTo(const Pending &next) const
    {
        return next;
    }

    Invalid Waiting::transitionTo(const Invalid &next) const
    {
        return next;
    }

    Failed Waiting::transitionTo(const Failed &next) const
    {
        return next;
    }

    Running Pending::transitionTo(const Running &next) const
    {
        return next;
    }

    Canceled Pending::transitionTo(const Canceled &next) const
    {
        return next;
    }

    Failed Pending::transitionTo(const Failed &next) const
    {
        return next;
    }

    Complete Running::transitionTo(const Complete &next) const
    {
        return next;
    }

    Failed Running::transitionTo(const Failed &next) const
    {
        return next;
    }

    Timeout Running::transitionTo(const Timeout &next) const
    {
        return next;
    }

    Failed Complete::transitionTo(const Failed &next) const
    {
        return next;
    }

    Canceled Canceled::transitionTo(const Canceled &) const
    {
        return *this;
    }

    Failed Canceled::transitionTo(const Failed &next) const
    {
        return next;
    }

    Failed Invalid::transitionTo(const Failed &next) const
    {
        return next;
    }

    Failed Timeout::transitionTo(const Failed &next) const
    {
        return next;
    }

    Failed Failed::transitionTo(const Failed &next) const
    {
        return next;
    }

}

BatchOperation::BatchOperation(std::unique_ptr<const hipFileIOParams_t> params,
                               std::shared_ptr<IBuffer> _buffer, std::shared_ptr<IFile> _file)
    : io_params{std::move(params)}, buffer{std::move(_buffer)}, file{std::move(_file)}
{
    // Cookie allows the user to track which operation caused the error.
    // It would be ideal if this could be passed as a member within the exception.

    // Check Buffer parameters
    if (io_params->u.batch.devPtr_base != buffer->getBuffer()) {
        throw std::invalid_argument("Buffer does not match buffer specified in io_params.");
    }
    if (io_params->u.batch.devPtr_offset < 0) {
        std::stringstream msg;
        msg << "Negative buffer offset specified. Value: " << io_params->u.batch.devPtr_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() <= static_cast<size_t>(io_params->u.batch.devPtr_offset)) {
        std::stringstream msg;
        msg << "Buffer offset exceeds the size of the buffer. Size: " << buffer->getLength();
        msg << ". Value: " << io_params->u.batch.devPtr_offset << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
    if (buffer->getLength() - static_cast<size_t>(io_params->u.batch.devPtr_offset) <
        io_params->u.batch.size) {
        std::stringstream msg;
        msg << "IO Size exceeds the size of the buffer & offset. Buffer size: " << buffer->getLength();
        msg << ". Buffer offset: " << io_params->u.batch.devPtr_offset
            << ". IO size: " << io_params->u.batch.size;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check File parameters
    if (io_params->fh != file->handle()) {
        throw std::invalid_argument("File does not match handle specified in io_params.");
    }
    if (io_params->u.batch.file_offset < 0) {
        std::stringstream msg;
        msg << "Negative file offset specified. Value: " << io_params->u.batch.file_offset;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check OpCode. A C caller may pass a value outside the enum's valid range,
    // and an lvalue-to-rvalue load of such a value as the enum type is undefined
    // behavior. Reinterpret the bits as the underlying integer type instead.
    auto opcode = std::bit_cast<std::underlying_type_t<hipFileOpcode_t>>(io_params->opcode);
    if (opcode != hipFileBatchRead && opcode != hipFileBatchWrite) {
        std::stringstream msg;
        msg << "Bad opcode specified. Value: " << opcode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check Batch Mode (reinterpret the bits as the underlying integer, see above).
    auto mode = std::bit_cast<std::underlying_type_t<hipFileBatchMode_t>>(io_params->mode);
    if (mode != hipFileBatch) {
        std::stringstream msg;
        msg << "Invalid Batch mode specified. Value: " << mode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }
}

template <class Next>
void
BatchOperation::transitionTo(Next next)
{
    state = std::visit([&next](const auto &current) -> OperationState { return current.transitionTo(next); },
                       state);
    // Drop references when they're no longer needed
    if constexpr (Next::isTerminal()) {
        buffer.reset();
        file.reset();
    }
}

void
BatchOperation::markPending()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    transitionTo(Pending{});
}

bool
BatchOperation::tryCancel()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    try {
        transitionTo(Canceled{});
        return true;
    }
    catch (const InvalidStateTransition &) {
        return false;
    }
}

hipFileIOEvents_t
BatchOperation::event() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return std::visit(
        [this](const auto &current) -> hipFileIOEvents_t {
            return {io_params->cookie, current.toPublic(), static_cast<size_t>(current.ret())};
        },
        state);
}

void
BatchOperation::run() noexcept
try {
    {
        std::lock_guard<std::mutex> lock{state_mutex};
        if (std::holds_alternative<Canceled>(state)) {
            return;
        }
        transitionTo(Running{});
    }

    ssize_t result   = 0;
    auto    backends = Context<DriverState>::get()->getBackends();
    try {
        if (io_params->opcode == hipFileBatchRead) {
            result = hipFileIo(IoType::Read, file, buffer, io_params->u.batch.size,
                               io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset, backends);
        }
        else {
            result = hipFileIo(IoType::Write, file, buffer, io_params->u.batch.size,
                               io_params->u.batch.file_offset, io_params->u.batch.devPtr_offset, backends);
        }
        if (result == -1) {
            result = -errno;
        }
    }
    catch (const Hip::RuntimeError &) {
        // Return hipFileHipDriverError for HIP runtime errors to avoid
        // collision with POSIX errors.
        result = -hipFileHipDriverError;
    }

    std::lock_guard<std::mutex> lock{state_mutex};
    if (result >= 0) {
        transitionTo(Complete{result});
    }
    else {
        transitionTo(Failed{result});
    }
}
catch (...) {
    recordInternalError();
}

void
BatchOperation::recordInternalError()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    transitionTo(Failed{-hipFileInternalError});
}

BatchContext::BatchContext(unsigned _capacity) : capacity{_capacity}
{
    if (_capacity == 0) {
        throw std::invalid_argument("Batch capacity cannot be zero");
    }
    if (_capacity > MAX_SIZE) {
        throw std::invalid_argument("Batch capacity is limited to " + std::to_string(MAX_SIZE));
    }

    task_group = Context<IThreadPool>::get()->makeTaskGroup();
}

BatchContext::~BatchContext() = default;

unsigned
BatchContext::getCapacity() const noexcept
{
    return capacity;
}

void
BatchContext::submitOperations(BatchOperations pending_ops)
{
    if (pending_ops.empty()) {
        throw std::invalid_argument("ops must not be empty");
    }
    {
        std::lock_guard<std::mutex> context_lock{context_mutex};

        if (finished) {
            throw InvalidBatchHandle();
        }

        if (pending_ops.size() > capacity - (submitted_ops.size() + completed_ops.size())) {
            throw BatchFull();
        }

        for (const auto &op : pending_ops) {
            op->markPending();
        }
        submitted_ops.insert(pending_ops.begin(), pending_ops.end());

        // Destruction takes context_mutex before canceling the task group. Keep
        // all enqueue operations in this critical section so destroy cannot
        // finish before an accepted operation is handed to the task group.
        auto   self     = shared_from_this();
        size_t op_index = 0;
        try {
            for (; op_index < pending_ops.size(); op_index++) {
                const auto &op = pending_ops[op_index];
                task_group->run([self, op]() {
                    op->run();
                    {
                        std::lock_guard<std::mutex> _lock{self->context_mutex};

                        // A cancel may have moved this operation to completed_ops
                        // already. Cancellation only stops queued work that has not
                        // started, so this work may still run after the operation was
                        // canceled. See completeCanceledOperations().
                        if (auto node = self->submitted_ops.extract(op); !node.empty()) {
                            self->completed_ops.push_back(std::move(node.value()));
                        }
                    }
                    self->status_cv.notify_all();
                });
            }
        }
        catch (...) {
            // Cancel work already accepted by the task group. Its queued work
            // will move it to completed_ops after observing the canceled state.
            for (size_t i = 0; i < op_index; i++) {
                pending_ops[i]->tryCancel();
            }

            // No queued work exists to complete the failed operation or any
            // operation after it, so complete those operations here.
            for (; op_index < pending_ops.size(); op_index++) {
                const auto &op = pending_ops[op_index];
                if (!op->tryCancel()) {
                    continue;
                }

                if (auto node = submitted_ops.extract(op); !node.empty()) {
                    completed_ops.push_back(std::move(node.value()));
                }
            }
            status_cv.notify_all();
            throw;
        }
    }
}

void
BatchContext::getStatus(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, BatchDeadline deadline)
{
    if (nr == nullptr) {
        throw std::invalid_argument("Number of events cannot be null");
    }
    if (*nr == 0) {
        throw std::invalid_argument("Number of events cannot be zero");
    }
    if (iocbp == nullptr) {
        throw std::invalid_argument("Event buffer cannot be null");
    }
    if (min_nr > *nr) {
        throw std::invalid_argument("Minimum event count exceeds event buffer capacity");
    }
    if (min_nr > capacity) {
        throw std::invalid_argument("Minimum event count exceeds batch capacity");
    }

    const unsigned event_capacity = *nr;
    *nr                           = 0;

    std::unique_lock<std::mutex> ulock{context_mutex};

    auto collect_completed_events = [this, event_capacity, nr, iocbp]() {
        const auto copied = std::min(static_cast<size_t>(event_capacity), completed_ops.size());
        for (size_t i = 0; i < copied; i++) {
            iocbp[i] = completed_ops[i]->event();
        }
        completed_ops.erase(completed_ops.begin(),
                            completed_ops.begin() + static_cast<std::ptrdiff_t>(copied));
        *nr = static_cast<unsigned>(copied);
    };

    auto ready = [min_nr, this]() { return completed_ops.size() >= min_nr || submitted_ops.empty(); };

    if (deadline.has_value()) {
        status_cv.wait_until(ulock, *deadline, ready);
    }
    else {
        status_cv.wait(ulock, ready);
    }

    collect_completed_events();
}

void
BatchContext::cancelOperations()
{
    std::lock_guard<std::mutex> lock{context_mutex};

    task_group->cancel();
    completeCanceledOperations();
    status_cv.notify_all();
}

void
BatchContext::cancelOperationsAndWait()
{
    {
        std::lock_guard<std::mutex> lock{context_mutex};

        finished = true;
        task_group->cancel();
        completeCanceledOperations();
    }
    task_group->wait();
    {
        std::lock_guard<std::mutex> lock{context_mutex};
        status_cv.notify_all();
    }
}

void
BatchContext::completeCanceledOperations()
{
    // A canceled operation never runs, so its queued work will not move it to
    // completed_ops. Move it here instead. Operations that could not be
    // canceled are either running or already complete, and are moved by their
    // own queued work.
    for (auto op_iter = submitted_ops.begin(); op_iter != submitted_ops.end();) {
        if (!(*op_iter)->tryCancel()) {
            ++op_iter;
            continue;
        }

        auto node = submitted_ops.extract(op_iter++);
        completed_ops.push_back(std::move(node.value()));
    }
}

void
BatchContextMap::clear()
{
    std::vector<std::shared_ptr<IBatchContext>> contexts;

    {
        std::unique_lock<std::shared_mutex> ulock{batch_mutex};
        contexts.reserve(active_contexts.size());
        for (auto &context : active_contexts) {
            contexts.push_back(std::move(context.second));
        }
        active_contexts.clear();
    }

    for (const auto &context : contexts) {
        context->cancelOperationsAndWait();
    }
}

hipFileBatchHandle_t
BatchContextMap::createContext(unsigned capacity)
{
    auto                 context = std::shared_ptr<IBatchContext>{new BatchContext{capacity}};
    hipFileBatchHandle_t handle  = context.get();

    // Should not need to worry about duplicate keys unless the application
    // somehow deallocates this handle...

    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts[handle] = std::move(context);
    return handle;
}

void
BatchContextMap::destroyContext(hipFileBatchHandle_t handle)
{
    std::shared_ptr<IBatchContext> context;

    {
        std::unique_lock<std::shared_mutex> ulock{batch_mutex};

        auto iter = active_contexts.find(handle);
        if (iter == active_contexts.end()) {
            throw InvalidBatchHandle();
        }

        context = std::move(iter->second);
        active_contexts.erase(iter);
    }

    context->cancelOperationsAndWait();
}

std::shared_ptr<IBatchContext>
BatchContextMap::get(hipFileBatchHandle_t handle)
{
    // NOTE: This mutex only protects the map, so we'll
    //       also need to protect the data
    std::shared_lock<std::shared_mutex> slock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    return context->second;
}

}
