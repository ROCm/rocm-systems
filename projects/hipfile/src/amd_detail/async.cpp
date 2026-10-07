/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "context.h"
#include "hip.h"
#include "hipfile.h"
#include "io.h"
#include "stream.h"
#include "thread-pool.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <utility>

namespace hipFile {
class IBuffer;
}
namespace hipFile {
class IFile;
}
namespace hipFile {
enum class IoType;
}

namespace hipFile {

AsyncMonitor::AsyncMonitor() : task_group{Context<IThreadPool>::get()->makeTaskGroup()}
{
}

AsyncMonitor::~AsyncMonitor()
{
    task_group->wait();
}

static void
signalOffloadComplete(AsyncOp *op)
{
    op->file.reset();
    op->buffer.reset();
    if (uint64_t *slot = op->stream->signalSlot()) {
        std::atomic_ref<uint64_t>{*slot}.fetch_add(1, std::memory_order_release);
    }
}

void
AsyncMonitor::submitIo(AsyncOp *op)
{
    task_group->run([op]() {
        std::unique_ptr<AsyncOp> owned{op};
        owned->io_fn(owned.get());
        signalOffloadComplete(owned.get());
    });
}

AsyncOp::AsyncOp(IoType _io_type, std::shared_ptr<IFile> _file, std::shared_ptr<IBuffer> _buffer,
                 std::shared_ptr<IStream> _stream, size_t *_size, hoff_t *_file_offset,
                 hoff_t *_buffer_offset, ssize_t *_bytes_transferred)
    : io_type{_io_type}, file{std::move(_file)}, buffer{std::move(_buffer)}, stream{std::move(_stream)},

      size{stream->fixedIOSize() ? std::variant<size_t, size_t *>{std::min(*_size, hipFile::getMaxRwCount())}
                                 : std::variant<size_t, size_t *>{_size}},
      file_offset{stream->fixedFileOffset() ? std::variant<const hoff_t, hoff_t *>{*_file_offset}
                                            : std::variant<const hoff_t, hoff_t *>{_file_offset}},
      buffer_offset{stream->fixedBufferOffset() ? std::variant<const hoff_t, hoff_t *>{*_buffer_offset}
                                                : std::variant<const hoff_t, hoff_t *>{_buffer_offset}},
      bytes_transferred{_bytes_transferred}
{
}

AsyncOp::~AsyncOp()
{
}

void
async_run_io(void *userargs)
{
    auto         op   = static_cast<AsyncOp *>(userargs);
    const size_t size = std::min(*get_variant_ptr(op->size), getMaxRwCount());
    const hoff_t fo   = *get_variant_ptr(op->file_offset);
    const hoff_t bo   = *get_variant_ptr(op->buffer_offset);
    if (!paramsValid(op->buffer, size, fo, bo)) {
        *op->bytes_transferred = -hipFileInvalidValue;
    }
    else {
        try {
            *op->bytes_transferred =
                op->backend->io(op->io_type, op->file, op->buffer, size, fo, bo, op->stream->copyStream());
        }
        catch (...) {
            *op->bytes_transferred = -hipFileInternalError;
        }
    }
}

void
enqueueAsync(std::shared_ptr<Backend> backend, IoType type, std::shared_ptr<IFile> file,
             std::shared_ptr<IBuffer> buffer, size_t *size_p, hoff_t *file_offset_p, hoff_t *buffer_offset_p,
             ssize_t *bytes_transferred_p, std::shared_ptr<IStream> stream)
{
    if (!paramsValid(buffer, std::min(*size_p, getMaxRwCount()), *file_offset_p, *buffer_offset_p)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    if (buffer->getGpuId() != stream->getHipDevice()) {
        throw std::invalid_argument("Buffer GPU ID does not match Stream GPU ID");
    }

    *bytes_transferred_p = 0;

    auto op     = std::make_unique<AsyncOp>(type, std::move(file), std::move(buffer), stream, size_p,
                                            file_offset_p, buffer_offset_p, bytes_transferred_p);
    op->backend = std::move(backend);
    op->io_fn   = async_run_io;

    auto        stream_lock = stream->getLock();
    hipStream_t hip_stream  = stream->getHipStream();
    bool        wait_value  = stream->canUseStreamWaitValue();
    bool        targeted    = false;
    bool        launched    = false;

    AsyncOp *raw = op.release();
    try {
        if (wait_value) {
            uint64_t target = stream->nextSignalTarget();
            targeted        = true;
            Context<Hip>::get()->hipLaunchHostFunc(hip_stream, async_dispatch, raw);
            launched = true;
            Context<Hip>::get()->hipStreamWaitValue64(hip_stream, stream->signalSlot(), target,
                                                      hipStreamWaitValueGte, ~uint64_t{0});
        }
        else {
            Context<Hip>::get()->hipLaunchHostFunc(hip_stream, async_run_inline, raw);
            launched = true;
        }
    }
    catch (...) {
        if (!launched) {
            std::unique_ptr<AsyncOp> owned{raw};
            if (targeted) {
                std::atomic_ref<uint64_t>{*stream->signalSlot()}.fetch_add(1, std::memory_order_release);
            }
        }
        throw;
    }
}

}

extern "C" {
void
async_dispatch(void *userargs)
{
    using namespace hipFile;
    auto op = static_cast<AsyncOp *>(userargs);
    try {
        Context<AsyncMonitor>::get()->submitIo(op);
    }
    catch (...) {
        std::unique_ptr<AsyncOp> owned{op};
        owned->io_fn(owned.get());
        signalOffloadComplete(owned.get());
    }
}

void
async_run_inline(void *userargs)
{
    using namespace hipFile;
    std::unique_ptr<AsyncOp> owned{static_cast<AsyncOp *>(userargs)};
    async_run_io(owned.get());
    signalOffloadComplete(owned.get());
}
}
