/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "batch.h"
#include "buffer.h"
#include "context.h"
#include "file.h"
#include "hipfile.h"
#include "state.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipFile {

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

BatchContext::BatchContext(unsigned _capacity) : capacity{_capacity}
{
    if (_capacity == 0) {
        throw std::invalid_argument("Batch capacity cannot be zero");
    }
    if (_capacity > MAX_SIZE) {
        throw std::invalid_argument("Batch capacity is limited to " + std::to_string(MAX_SIZE));
    }
}

unsigned
BatchContext::get_capacity() const noexcept
{
    return capacity;
}

void
BatchContext::submit_operations(const hipFileIOParams_t *params, unsigned num_params)
{
    std::unique_lock<std::shared_mutex> _ulock{context_mutex};

    // Check num_params first before doing anything else
    if (num_params > capacity - outstanding_ops.size()) {
        std::stringstream msg;
        msg << "Submission exceeds the capacity of this context. Number of ops submitted: ";
        msg << num_params << ". Context capacity: " << capacity << ". Current outstanding ops: ";
        msg << outstanding_ops.size();
        throw std::invalid_argument(msg.str());
    }

    std::vector<std::shared_ptr<BatchOperation>> pending_ops{};

    // It would be more performant to be able to perform multiple lookups
    // rather than waiting to lock the DriverState lock for each lookup.
    for (unsigned i = 0; i < num_params; i++) {
        // Make a copy of the params so another thread cannot modify the operation.
        auto param_copy = std::make_unique<const hipFileIOParams_t>(params[i]);
        // flags currently unused. Ambiguous if flags in hipFileBatchIOSubmit is for buffer or
        // file flags.
        auto [_file, _buffer] =
            Context<DriverState>::get()->getFileAndBuffer(param_copy->fh, param_copy->u.batch.devPtr_base);
        auto op = std::make_shared<BatchOperation>(std::move(param_copy), _buffer, _file);

        pending_ops.push_back(std::move(op));
    }

    // All submitted operations look valid at this point. Accept them.
    outstanding_ops.insert(pending_ops.begin(), pending_ops.end());
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
    std::unique_lock<std::shared_mutex> ulock{batch_mutex};

    auto context = active_contexts.find(handle);
    if (context == active_contexts.end()) {
        throw InvalidBatchHandle();
    }
    // TODO: Check for outstanding operations.
    // TODO: Attempt to cancel any outstanding operations.
    // TODO: Determine if we return unconditionally or require
    //       outstanding ops to terminate first.
    active_contexts.erase(handle);
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
