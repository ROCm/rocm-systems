/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "context.h"
#include "device-guard.h"
#include "hip.h"
#include "hipfile.h"
#include "io.h"
#include "stream.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <utility>

namespace hipFile {

static void
signalOffloadComplete(AsyncOp *op)
{
    op->file.reset();
    op->buffer.reset();
    if (uint64_t *slot = op->stream->signalSlot()) {
        std::atomic_ref<uint64_t>{*slot}.fetch_add(1, std::memory_order_release);
    }
    op->stream->finishOutstanding();
}

// Finishes an op exactly once. An abandoned op only releases its stream signal and touches no caller memory.
static void
runOp(AsyncOp *op, bool run_io)
{
    auto expected = AsyncOpState::Pending;
    if (!op->state.compare_exchange_strong(expected, AsyncOpState::Started, std::memory_order_acq_rel)) {
        signalOffloadComplete(op);
        return;
    }

    if (run_io) {
        op->io_fn(op);
    }
    else {
        *op->bytes_transferred = -hipFileInternalError;
    }
    signalOffloadComplete(op);
    op->state.store(AsyncOpState::Done, std::memory_order_release);
    op->state.notify_all();
}

// Only for ops that can't be abandoned: those the pool has already started, or that never arm a wait.
static void
waitUntilDone(AsyncOp *op)
{
    for (auto state = op->state.load(std::memory_order_acquire); state != AsyncOpState::Done;
         state      = op->state.load(std::memory_order_acquire)) {
        op->state.wait(state, std::memory_order_acquire);
    }
}

AsyncOp::AsyncOp(IoType _io_type, std::shared_ptr<IFile> _file, std::shared_ptr<IBuffer> _buffer,
                 std::shared_ptr<IStream> _stream, size_t *_size, hoff_t *_file_offset,
                 hoff_t *_buffer_offset, ssize_t *_bytes_transferred, std::function<void(AsyncOp *)> _io_fn,
                 std::shared_ptr<Backend> _backend)
    : io_type{_io_type}, file{std::move(_file)}, buffer{std::move(_buffer)}, stream{std::move(_stream)},
      size{stream->fixedIOSize() ? std::variant<size_t, size_t *>{std::min(*_size, hipFile::getMaxRwCount())}
                                 : std::variant<size_t, size_t *>{_size}},
      file_offset{stream->fixedFileOffset() ? std::variant<const hoff_t, hoff_t *>{*_file_offset}
                                            : std::variant<const hoff_t, hoff_t *>{_file_offset}},
      buffer_offset{stream->fixedBufferOffset() ? std::variant<const hoff_t, hoff_t *>{*_buffer_offset}
                                                : std::variant<const hoff_t, hoff_t *>{_buffer_offset}},
      bytes_transferred{_bytes_transferred}, io_fn{std::move(_io_fn)}, backend{std::move(_backend)}
{
}

AsyncMonitor::AsyncMonitor(std::unique_ptr<IThreadPool> _pool)
    : pool{std::move(_pool)}, task_group{pool->makeTaskGroup()}
{
}

AsyncMonitor::~AsyncMonitor()
{
    task_group->wait();
}

void
AsyncMonitor::submitIo(std::shared_ptr<AsyncOp> op)
{
    task_group->run([task_op = std::move(op)]() { runOp(task_op.get(), true); });
}

void
async_run_io(AsyncOp *op)
{
    const size_t size          = std::min(*get_variant_ptr(op->size), hipFile::getMaxRwCount());
    const hoff_t file_offset   = *get_variant_ptr(op->file_offset);
    const hoff_t buffer_offset = *get_variant_ptr(op->buffer_offset);

    if (!paramsValid(op->buffer, size, file_offset, buffer_offset)) {
        *op->bytes_transferred = -hipFileInvalidValue;
        return;
    }

    try {
        DeviceGuard guard{op->stream->getHipDevice()};
        *op->bytes_transferred =
            op->backend->io(op->io_type, op->file, op->buffer, size, file_offset, buffer_offset);
    }
    catch (...) {
        *op->bytes_transferred = -hipFileInternalError;
    }
}

void
enqueueAsync(std::shared_ptr<Backend> backend, IoType io_type, std::shared_ptr<IFile> file,
             std::shared_ptr<IBuffer> buffer, size_t *size, hoff_t *file_offset, hoff_t *buffer_offset,
             ssize_t *bytes_transferred, std::shared_ptr<IStream> stream)
{
    size_t limited_size = std::min(*size, hipFile::getMaxRwCount());
    if (!paramsValid(buffer, limited_size, *file_offset, *buffer_offset)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }
    if (buffer->getGpuId() != stream->getHipDevice()) {
        throw std::invalid_argument("Buffer GPU ID does not match Stream GPU ID");
    }

    *bytes_transferred = 0;

    if (isDefaultStream(stream->getHipStream())) {
        AsyncOp op(io_type, std::move(file), std::move(buffer), stream, size, file_offset, buffer_offset,
                   bytes_transferred, async_run_io, std::move(backend));
        Context<Hip>::get()->hipStreamSynchronize(stream->getHipStream());
        op.io_fn(&op);
        return;
    }

    const bool               gated = stream->canUseStreamWaitValue();
    std::shared_ptr<AsyncOp> op;
    std::exception_ptr       arm_error;
    {
        // Held across the HIP calls below so each op's dispatch and wait reach the stream in target order.
        // Host functions never take it.
        auto stream_lock = stream->getLock();
        if (stream->isRetired()) {
            throw std::invalid_argument("Stream is being deregistered");
        }

        op = std::make_shared<AsyncOp>(io_type, std::move(file), std::move(buffer), stream, size, file_offset,
                                       buffer_offset, bytes_transferred, async_run_io, std::move(backend));
        op->block       = !gated;
        uint64_t target = gated ? stream->nextSignalTarget() : 0;
        stream->addOutstanding();

        op->self = op;
        try {
            Context<Hip>::get()->hipLaunchHostFunc(stream->getHipStream(), async_dispatch, op.get());
        }
        catch (...) {
            op->self.reset();
            if (gated) {
                stream->releaseSignalTarget();
            }
            stream->finishOutstanding();
            throw;
        }

        if (gated) {
            try {
                Context<Hip>::get()->hipStreamWaitValue64(stream->getHipStream(), stream->signalSlot(),
                                                          target, hipStreamWaitValueGte, UINT64_MAX);
            }
            catch (...) {
                arm_error = std::current_exception();
            }
        }
    }

    if (!arm_error) {
        return;
    }

    // The stream won't wait for this op. If the pool hasn't started it, abandon it and report the failure.
    // Otherwise the stream has already passed the dispatch, so the IO can't depend on stream progress and
    // it's safe to wait for it here.
    auto expected = AsyncOpState::Pending;
    if (op->state.compare_exchange_strong(expected, AsyncOpState::Abandoned, std::memory_order_acq_rel)) {
        std::rethrow_exception(arm_error);
    }
    waitUntilDone(op.get());
}

}

extern "C" {
void
async_dispatch(void *userargs)
{
    using namespace hipFile;
    std::shared_ptr<AsyncOp> op = std::move(static_cast<AsyncOp *>(userargs)->self);
    try {
        Context<IAsyncMonitor>::get()->submitIo(op);
    }
    catch (...) {
        runOp(op.get(), false);
    }
    if (op->block) {
        waitUntilDone(op.get());
    }
}
}
