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
#include "sys.h"
#include "thread-pool.h"
#include "util.h"

#include <algorithm>
#include <atomic>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <syslog.h>
#include <thread>
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

AsyncMonitor::AsyncMonitor() : task_group{Context<IThreadPool>::get()->makeTaskGroup()}, is_finished{false}
{
    thread = std::thread(&AsyncMonitor::completion_thread, this);
}

AsyncMonitor::~AsyncMonitor()
{
    task_group->wait();
    {
        std::lock_guard<std::mutex> lock{mutex};
        is_finished = true;
    }
    cv.notify_one();
    thread.join();
    if (submitted_ops.size() > 0) {
        Context<Sys>::get()->syslog(LOG_CRIT,
                                    "Async state is being destructed while operations are outstanding.");
    }
}

static void
signalOffloadComplete(AsyncOp *op)
{
    if (uint64_t *slot = op->stream->signalSlot()) {
        std::atomic_ref<uint64_t>{*slot}.fetch_add(1, std::memory_order_release);
    }
    try {
        Context<AsyncMonitor>::get()->completeOp(op);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Unable to complete async op. This will leak memory.");
    }
}

static void
drainStream(const std::shared_ptr<IStream> &stream)
{
    while (auto op = stream->popPending()) {
        if (uint64_t *dispatch = stream->dispatchSlot()) {
            std::atomic_ref<uint64_t> ref{*dispatch};
            while (ref.load(std::memory_order_acquire) < op->wait_target) {
                std::this_thread::yield();
            }
        }
        op->io_fn(op.get());
        signalOffloadComplete(op.get());
    }
}

void
AsyncMonitor::spawnDrain(std::shared_ptr<IStream> stream)
{
    task_group->run([drain_stream = std::move(stream)]() { drainStream(drain_stream); });
}

void
AsyncMonitor::addOp(std::shared_ptr<AsyncOp> op)
{
    std::lock_guard<std::mutex> lock{mutex};
    submitted_ops.insert({op.get(), std::move(op)});
}

void
AsyncMonitor::completeOp(AsyncOp *op)
{
    {
        std::lock_guard<std::mutex> lock{mutex};
        if (auto found = submitted_ops.find(op); found == submitted_ops.end()) {
            throw std::invalid_argument("Op does not appear in submitted_ops");
        }
        op->file.reset();
        op->buffer.reset();
        completed_ops.push_back(op);
    }
    cv.notify_one();
}

void
AsyncMonitor::completion_thread()
{
    while (true) {
        std::unique_lock<std::mutex> lock{mutex};
        if (!completed_ops.empty()) {
            AsyncOp *op{completed_ops.back()};
            completed_ops.pop_back();
            auto nh = submitted_ops.extract(op);
            // Lock needs to be released before destructor runs, as hipHostFree calls hipDeviceSynchronize.
            // If another host function is running completeOp and waiting for the lock, this would cause
            // deadlock.
            lock.unlock();
        }
        else if (!is_finished) {
            cv.wait(lock, [this] { return is_finished || !completed_ops.empty(); });
        }
        else {
            break;
        }
    }
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

    auto op     = std::make_shared<AsyncOp>(type, std::move(file), std::move(buffer), stream, size_p,
                                            file_offset_p, buffer_offset_p, bytes_transferred_p);
    op->backend = std::move(backend);
    op->io_fn   = async_run_io;
    Context<AsyncMonitor>::get()->addOp(op);

    auto        stream_lock = stream->getLock();
    hipStream_t hip_stream  = stream->getHipStream();
    bool        wait_value  = stream->canUseStreamWaitValue();
    bool        targeted    = false;
    bool        pushed      = false;
    bool        spawn       = false;

    try {
        if (wait_value) {
            op->wait_target = stream->nextSignalTarget();
            targeted        = true;
            Context<Hip>::get()->hipStreamWriteValue64(hip_stream, stream->dispatchSlot(), op->wait_target,
                                                       0);
            Context<Hip>::get()->hipStreamWaitValue64(hip_stream, stream->signalSlot(), op->wait_target,
                                                      hipStreamWaitValueGte, ~uint64_t{0});
            spawn  = stream->pushPending(op);
            pushed = true;
            if (spawn) {
                Context<AsyncMonitor>::get()->spawnDrain(stream);
            }
        }
        else {
            Context<Hip>::get()->hipLaunchHostFunc(hip_stream, async_run_inline, op.get());
        }
    }
    catch (...) {
        if (pushed && spawn) {
            stream->unpushPending();
            pushed = false;
        }
        if (!pushed && targeted) {
            std::atomic_ref<uint64_t>{*stream->signalSlot()}.fetch_add(1, std::memory_order_release);
            try {
                Context<AsyncMonitor>::get()->completeOp(op.get());
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Unable to complete async op. This will leak memory.");
            }
        }
        throw;
    }
}

}

extern "C" {
void
async_run_inline(void *userargs)
{
    using namespace hipFile;
    auto op = static_cast<AsyncOp *>(userargs);
    async_run_io(op);
    signalOffloadComplete(op);
}
}
