/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "context.h"
#include "cooperative-copy.h"
#include "device-guard.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "io.h"
#include "stats.h"
#include "stream.h"
#include "sys.h"
#include "thread-pool.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace hipFile {

static void
signalOffloadComplete(AsyncOp *op)
{
    op->file.reset();
    op->buffer.reset();
    if (uint64_t *slot = op->signal_slot) {
        std::atomic_ref<uint64_t>{*slot}.fetch_add(1, std::memory_order_release);
    }
}

static void
markDone(AsyncOp *op)
{
    op->state.store(AsyncOpState::Done, std::memory_order_release);
    op->state.notify_all();
}

static bool
startOp(AsyncOp *op)
{
    auto expected = AsyncOpState::Pending;
    if (op->state.compare_exchange_strong(expected, AsyncOpState::Started, std::memory_order_acq_rel)) {
        return true;
    }
    signalOffloadComplete(op);
    return false;
}

static void
dispatchFailed(AsyncOp *op)
{
    if (startOp(op)) {
        *op->bytes_transferred = -hipFileInternalError;
        signalOffloadComplete(op);
        markDone(op);
    }
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
                 hoff_t *_buffer_offset, ssize_t *_bytes_transferred, std::shared_ptr<Backend> _backend)
    : io_type{_io_type}, file{std::move(_file)}, buffer{std::move(_buffer)}, stream{std::move(_stream)},
      size{stream->fixedIOSize() ? std::variant<size_t, size_t *>{std::min(*_size, hipFile::getMaxRwCount())}
                                 : std::variant<size_t, size_t *>{_size}},
      file_offset{stream->fixedFileOffset() ? std::variant<const hoff_t, hoff_t *>{*_file_offset}
                                            : std::variant<const hoff_t, hoff_t *>{_file_offset}},
      buffer_offset{stream->fixedBufferOffset() ? std::variant<const hoff_t, hoff_t *>{*_buffer_offset}
                                                : std::variant<const hoff_t, hoff_t *>{_buffer_offset}},
      bytes_transferred{_bytes_transferred}, backend{std::move(_backend)}, signal_slot{stream->signalSlot()}
{
}

AsyncMonitor::AsyncMonitor(std::unique_ptr<IThreadPool> _pool)
    : pool{std::move(_pool)}, task_group{pool->makeTaskGroup()}
{
}

AsyncMonitor::~AsyncMonitor()
{
    task_group->wait();
    if (!progress.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        stopping = true;
    }
    changed.notify_one();
    progress.join();
    task_group->wait();
}

void
AsyncMonitor::submitIo(std::shared_ptr<AsyncOp> op)
{
    task_group->run([this, task = std::move(op)] { runDirect(task); });
}

// Snapshot execution-time inputs once. Both phases of a fallback use these same validated offsets.
static bool
runIoOrDeferFallback(AsyncOp *op)
{
    const size_t size          = std::min(*get_variant_ptr(op->size), getMaxRwCount());
    const hoff_t file_offset   = *get_variant_ptr(op->file_offset);
    const hoff_t buffer_offset = *get_variant_ptr(op->buffer_offset);
    if (!paramsValid(op->buffer, size, file_offset, buffer_offset) ||
        (op->copy && std::cmp_greater(size, std::numeric_limits<hoff_t>::max() - file_offset))) {
        *op->bytes_transferred = -hipFileInvalidValue;
        return false;
    }
    try {
        DeviceGuard guard{op->stream->getHipDevice()};
        if (!op->copy) {
            *op->bytes_transferred =
                op->backend->io(op->io_type, op->file, op->buffer, size, file_offset, buffer_offset);
            return false;
        }
        auto result =
            op->backend->tryDirectIo(op->io_type, op->file, op->buffer, size, file_offset, buffer_offset);
        if (result) {
            *op->bytes_transferred = *result;
            return false;
        }
        op->copy_size          = size;
        op->copy_file_offset   = file_offset;
        op->copy_buffer_offset = static_cast<size_t>(buffer_offset);
        return true;
    }
    catch (...) {
        *op->bytes_transferred = -hipFileInternalError;
        return false;
    }
}

void
AsyncMonitor::runDirect(const std::shared_ptr<AsyncOp> &op)
{
    if (!startOp(op.get())) {
        return;
    }
    if (runIoOrDeferFallback(op.get())) {
        try {
            {
                std::lock_guard<std::mutex> lock{mutex};
                if (!progress.joinable()) {
                    progress = std::thread{[this] { progressCopies(); }};
                }
                pending.push_back(op);
                op->copy->prepare(op->ticket);
                std::atomic_ref<uint64_t>{*op->signal_slot}.fetch_add(1, std::memory_order_release);
            }
            changed.notify_one();
            return;
        }
        catch (...) {
            *op->bytes_transferred = -hipFileInternalError;
        }
    }
    signalOffloadComplete(op.get());
    markDone(op.get());
}

void
async_run_io(AsyncOp *op)
{
    (void)runIoOrDeferFallback(op);
}

static ssize_t
transferChunks(AsyncOp *op)
{
    StatsIoTracker tracker{op->io_type,
                           StatsBackend::Fallback,
                           op->file,
                           op->buffer,
                           op->copy_size,
                           op->copy_file_offset,
                           static_cast<hoff_t>(op->copy_buffer_offset)};
    ssize_t        result = -hipFileInternalError;
    try {
        result = op->copy->transfer(*Context<Sys>::get(), op->io_type, op->file->bufferedFd(), op->copy_size,
                                    op->copy_file_offset, op->copy_buffer_offset);
    }
    catch (...) {
        Context<StatsCollection>::get()->error(op->io_type, StatsBackend::Fallback, op->copy_size,
                                               op->buffer->getGpuId());
    }
    op->copy->stop(op->io_type);
    if (result >= 0) {
        tracker.complete(static_cast<size_t>(result));
    }
    return result;
}

static void
finishCopy(AsyncOp *op, ssize_t result, bool resident)
{
    *op->bytes_transferred = result;
    op->file.reset();
    op->buffer.reset();
    if (resident) {
        op->copy->finish();
    }
    markDone(op);
}

void
AsyncMonitor::progressCopies()
{
    std::unique_lock<std::mutex> lock{mutex};
    for (;;) {
        changed.wait(lock, [this] { return stopping || !pending.empty(); });
        if (pending.empty()) {
            return;
        }
        auto ready = std::find_if(pending.begin(), pending.end(), [](const auto &op) {
            return op->cancel_copy.load(std::memory_order_acquire) || op->copy->resident(op->ticket);
        });
        if (ready == pending.end()) {
            changed.wait_for(lock, std::chrono::microseconds{50});
            continue;
        }
        auto op = std::move(*ready);
        pending.erase(ready);
        lock.unlock();
        if (op->cancel_copy.load(std::memory_order_acquire)) {
            finishCopy(op.get(), -hipFileInternalError, false);
        }
        else {
            try {
                task_group->run([op] { finishCopy(op.get(), transferChunks(op.get()), true); });
            }
            catch (...) {
                op->copy->stop(op->io_type);
                finishCopy(op.get(), -hipFileInternalError, true);
            }
        }
        op.reset();
        lock.lock();
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
    // Host buffers haven't been tested on the async path, so reject them up front rather than letting
    // their gpu_id of -1 fail the device check below as a less specific error.
    if (buffer->getType() == hipMemoryTypeHost) {
        throw hipFileError_t{hipFileAsyncNotSupported, hipSuccess};
    }
    if (buffer->getGpuId() != stream->getHipDevice()) {
        throw std::invalid_argument("Buffer GPU ID does not match Stream GPU ID");
    }

    *bytes_transferred = 0;

    if (isDefaultStream(stream->getHipStream())) {
        AsyncOp op(io_type, std::move(file), std::move(buffer), stream, size, file_offset, buffer_offset,
                   bytes_transferred, std::move(backend));
        Context<Hip>::get()->hipStreamSynchronize(stream->getHipStream());
        async_run_io(&op);
        return;
    }

    const bool               gated = stream->canUseStreamWaitValue();
    std::shared_ptr<AsyncOp> op;

    {
        // Serialize hipFile submissions so each dispatch, wait, and kernel reach the stream in ticket order.
        // Host functions never take it.
        auto stream_lock = stream->getLock();
        if (stream->isRetired()) {
            throw std::invalid_argument("Stream is being deregistered");
        }

        // Allocate the signal slot before building the op, so the AsyncOp ctor caches a valid pointer.
        // Held under the lock and sequenced after the retire check, so a retiring drain can't race it.
        if (gated) {
            stream->ensureSignalSlot();
        }

        void *device_buffer = gated ? buffer->getBuffer() : nullptr;
        op = std::make_shared<AsyncOp>(io_type, std::move(file), std::move(buffer), stream, size, file_offset,
                                       buffer_offset, bytes_transferred, std::move(backend));
        op->block = !gated;
        if (gated) {
            op->copy   = &stream->ensureCooperativeCopy();
            op->ticket = stream->nextSignalTarget();
        }
        op->self = op;
        try {
            Context<Hip>::get()->hipLaunchHostFunc(stream->getHipStream(), async_dispatch, op.get());
        }
        catch (...) {
            // Nothing will bump the slot for this target, so give it back.
            op->self.reset();
            if (gated) {
                stream->releaseSignalTarget();
            }
            throw;
        }

        if (gated) {
            try {
                Context<Hip>::get()->hipStreamWaitValue64(stream->getHipStream(), op->signal_slot, op->ticket,
                                                          hipStreamWaitValueGte, UINT64_MAX);
                DeviceGuard guard{stream->getHipDevice()};
                op->copy->launch(stream->getHipStream(), device_buffer, io_type == IoType::Read, op->ticket);
            }
            catch (...) {
                // No kernel was launched. If the callback already started, its direct phase can finish
                // independently and the monitor can cancel a deferred copy without waiting for the GPU.
                // Serialize this recovery against the next op initializing the same workspace.
                op->cancel_copy.store(true, std::memory_order_release);
                auto expected = AsyncOpState::Pending;
                if (op->state.compare_exchange_strong(expected, AsyncOpState::Abandoned)) {
                    throw;
                }
                waitUntilDone(op.get());
            }
        }
    }
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
        dispatchFailed(op.get());
    }
    if (op->block) {
        waitUntilDone(op.get());
    }
}
}
