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
#include <future>
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
    task_group->run([task_op = std::move(op)]() {
        task_op->io_fn(task_op.get());
        signalOffloadComplete(task_op.get());
    });
}

void
AsyncMonitor::submitIoAndWait(std::shared_ptr<AsyncOp> op)
{
    std::promise<void> done;
    std::future<void>  ready{done.get_future()};
    task_group->run([&op, &done]() {
        op->io_fn(op.get());
        signalOffloadComplete(op.get());
        done.set_value();
    });
    ready.wait();
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

    auto op = std::make_unique<AsyncOp>(io_type, std::move(file), buffer, stream, size, file_offset,
                                        buffer_offset, bytes_transferred, async_run_io, std::move(backend));

    if (stream->getHipStream() == nullptr) {
        Context<Hip>::get()->hipStreamSynchronize(nullptr);
        op->io_fn(op.get());
        signalOffloadComplete(op.get());
        return;
    }

    AsyncOp *raw         = op.get();
    auto     stream_lock = stream->getLock();

    if (stream->isRetired()) {
        throw std::invalid_argument("Stream is being deregistered");
    }

    Context<Hip>::get()->hipLaunchHostFunc(stream->getHipStream(), async_dispatch, raw);
    op.release();

    if (stream->canUseStreamWaitValue()) {
        uint64_t target = stream->nextSignalTarget();
        try {
            Context<Hip>::get()->hipStreamWaitValue64(stream->getHipStream(), stream->signalSlot(), target,
                                                      hipStreamWaitValueGte, UINT64_MAX);
            raw->block = false;
        }
        catch (...) {
        }
    }
}

}

extern "C" {
void
async_dispatch(void *userargs)
{
    using namespace hipFile;
    std::shared_ptr<AsyncOp> op{static_cast<AsyncOp *>(userargs)};
    try {
        bool block;
        {
            auto lock = op->stream->getLock();
            block     = op->block;
        }
        if (block) {
            Context<IAsyncMonitor>::get()->submitIoAndWait(op);
        }
        else {
            Context<IAsyncMonitor>::get()->submitIo(op);
        }
    }
    catch (...) {
        *op->bytes_transferred = -hipFileInternalError;
        signalOffloadComplete(op.get());
    }
}
}
