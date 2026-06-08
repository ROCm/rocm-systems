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
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipFile {

namespace {

    using batchOperationState::Canceled;
    using batchOperationState::Complete;
    using batchOperationState::Failed;
    using batchOperationState::Invalid;
    using batchOperationState::OperationState;
    using batchOperationState::Pending;
    using batchOperationState::Running;
    using batchOperationState::Timeout;
    using batchOperationState::Waiting;

    bool is_zero_timeout(const struct timespec *timeout) noexcept
    {
        return timeout != nullptr && timeout->tv_sec == 0 && timeout->tv_nsec == 0;
    }

    void validate_timeout(const struct timespec *timeout)
    {
        if (timeout == nullptr) {
            return;
        }
        if (timeout->tv_sec < 0 || timeout->tv_nsec < 0 || timeout->tv_nsec >= 1000000000L) {
            throw std::invalid_argument("Invalid batch status timeout");
        }
    }

    std::chrono::steady_clock::time_point timeout_deadline(const struct timespec *timeout)
    {
        return std::chrono::steady_clock::now() + std::chrono::seconds{timeout->tv_sec} +
               std::chrono::nanoseconds{timeout->tv_nsec};
    }

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
    : io_params{std::move(params)}, buffer{_buffer}, file{_file}
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

    // Check OpCode
    if (io_params->opcode != hipFileBatchRead && io_params->opcode != hipFileBatchWrite) {
        std::stringstream msg;
        msg << "Bad opcode specified. Value: " << io_params->opcode;
        msg << ". Cookie: " << io_params->cookie;
        throw std::invalid_argument(msg.str());
    }

    // Check Batch Mode
    if (io_params->mode != hipFileBatch) {
        std::stringstream msg;
        msg << "Invalid Batch mode specified. Value: " << io_params->mode;
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
}

void
BatchOperation::markPending()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    transitionTo(Pending{});
}

void
BatchOperation::tryCancel()
{
    std::lock_guard<std::mutex> lock{state_mutex};

    try {
        transitionTo(Canceled{});
    }
    catch (const InvalidStateTransition &) {
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

bool
BatchOperation::isTerminal() const
{
    std::lock_guard<std::mutex> lock{state_mutex};
    return std::visit([](const auto &current) { return current.isTerminal(); }, state);
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

std::shared_ptr<IBatchOperation>
BatchOperationFactory::create(std::unique_ptr<const hipFileIOParams_t> params,
                              std::shared_ptr<IBuffer> buffer, std::shared_ptr<IFile> file)
{
    return std::make_shared<BatchOperation>(std::move(params), std::move(buffer), std::move(file));
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
BatchContext::submitOperations(const hipFileIOParams_t *params, unsigned num_params,
                               IBatchOperationFactory *operation_factory)
{
    if (!params) {
        throw std::invalid_argument("params cannot be null");
    }
    if (num_params == 0) {
        throw std::invalid_argument("num_params must be greater than 0");
    }
    std::unique_lock<std::shared_mutex> _ulock{context_mutex};

    // Check num_params first before doing anything else
    if (num_params > capacity - outstanding_ops.size()) {
        throw BatchFull();
    }

    std::vector<std::shared_ptr<IBatchOperation>> pending_ops{};
    BatchOperationFactory                         default_factory{};
    IBatchOperationFactory &factory = operation_factory == nullptr ? default_factory : *operation_factory;

    // It would be more performant to be able to perform multiple lookups
    // rather than waiting to lock the DriverState lock for each lookup.
    for (unsigned i = 0; i < num_params; i++) {
        // Make a copy of the params so another thread cannot modify the operation.
        auto param_copy = std::make_unique<const hipFileIOParams_t>(params[i]);
        // flags currently unused. Ambiguous if flags in hipFileBatchIOSubmit is for buffer or
        // file flags.
        auto [_file, _buffer] =
            Context<DriverState>::get()->getFileAndBuffer(param_copy->fh, param_copy->u.batch.devPtr_base);
        auto op = factory.create(std::move(param_copy), std::move(_buffer), std::move(_file));

        pending_ops.push_back(std::move(op));
    }

    // All submitted operations look valid at this point. Accept them.
    for (const auto &op : pending_ops) {
        op->markPending();
    }
    outstanding_ops.insert(pending_ops.begin(), pending_ops.end());

    auto self = shared_from_this();
    for (const auto &op : pending_ops) {
        task_group->run([self, op]() {
            op->run();
            {
                std::shared_lock<std::shared_mutex> _slock{self->context_mutex};
                self->status_cv.notify_all();
            }
        });
    }
}

void
BatchContext::getStatus(unsigned min_nr, unsigned *nr, hipFileIOEvents_t *iocbp, struct timespec *timeout)
{
    if (iocbp == nullptr) {
        throw std::invalid_argument("Event buffer cannot be null");
    }
    if (nr == nullptr) {
        throw std::invalid_argument("Number of events cannot be null");
    }
    if (*nr == 0) {
        throw std::invalid_argument("Number of events cannot be zero");
    }
    if (min_nr > *nr) {
        throw std::invalid_argument("Minimum event count exceeds event buffer capacity");
    }
    validate_timeout(timeout);

    const unsigned event_capacity = *nr;
    *nr                           = 0;

    std::unique_lock<std::shared_mutex> ulock{context_mutex};

    auto terminal_count = [this]() {
        unsigned count = 0;
        for (const auto &op : outstanding_ops) {
            if (op->isTerminal()) {
                count++;
            }
        }
        return count;
    };

    auto collect_terminal_events = [this, event_capacity, nr, iocbp]() {
        unsigned copied = 0;
        for (auto op_iter = outstanding_ops.begin();
             op_iter != outstanding_ops.end() && copied < event_capacity;) {
            if (!(*op_iter)->isTerminal()) {
                ++op_iter;
                continue;
            }

            iocbp[copied++] = (*op_iter)->event();
            op_iter         = outstanding_ops.erase(op_iter);
        }
        *nr = copied;
        return copied;
    };

    // Cap to what's actually outstanding so an over-large min_nr does not block forever.
    min_nr                = std::min(min_nr, static_cast<unsigned>(outstanding_ops.size()));
    unsigned num_terminal = terminal_count();

    if (num_terminal >= min_nr || num_terminal == outstanding_ops.size() || is_zero_timeout(timeout)) {
        collect_terminal_events();
        return;
    }

    auto ready = [&terminal_count, min_nr, this]() {
        unsigned num_terminal_now = terminal_count();
        return num_terminal_now >= min_nr || num_terminal_now == outstanding_ops.size();
    };

    if (timeout == nullptr) {
        status_cv.wait(ulock, ready);
    }
    else {
        status_cv.wait_until(ulock, timeout_deadline(timeout), ready);
    }

    collect_terminal_events();
}

void
BatchContext::cancelOperations()
{
    auto task_group_to_cancel = Context<IThreadPool>::get()->makeTaskGroup();
    {
        std::unique_lock<std::shared_mutex> lock{context_mutex};

        task_group.swap(task_group_to_cancel);
        task_group_to_cancel->cancel();
        for (const auto &op : outstanding_ops) {
            op->tryCancel();
        }
        status_cv.notify_all();
    }

    try {
        task_group_to_cancel->wait();
    }
    catch (...) {
        // This shouldn't throw, but catch for safety
    }

    {
        std::unique_lock<std::shared_mutex> lock{context_mutex};
        status_cv.notify_all();
    }
}

void
BatchContextMap::clear()
{
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};
    active_contexts.clear();
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

    context->cancelOperations();
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
