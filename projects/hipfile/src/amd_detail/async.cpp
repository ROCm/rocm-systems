/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "context.h"
#include "hip.h"
#include "hipfile.h"
#include "stream.h"
#include "sys.h"
#include "thread-pool.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <syslog.h>
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
    std::atomic_ref<uint64_t>{*op->signal_slot}.fetch_add(1, std::memory_order_release);
}

void
AsyncMonitor::submitIo(AsyncOp *op)
{
    task_group->run([op]() {
        op->io_fn(op);
        signalOffloadComplete(op);
    });
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
      bytes_transferred{_bytes_transferred}, bytes_transferred_internal{0}
{
}

uint64_t *
allocateSignalSlot()
{
    auto slot = static_cast<uint64_t *>(
        Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory));
    std::atomic_ref<uint64_t>{*slot}.store(0, std::memory_order_release);
    return slot;
}

AsyncOp::~AsyncOp()
{
    if (signal_slot) {
        try {
            Context<Hip>::get()->hipFree(signal_slot);
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT, "Unable to free async signal memory.");
        }
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
        op->io_fn(op);
        signalOffloadComplete(op);
    }
}

void
async_io_cleanup(void *userargs)
{
    using namespace hipFile;
    auto     op                         = static_cast<AsyncOp *>(userargs);
    ssize_t *bytes_transferred          = op->bytes_transferred;
    ssize_t  bytes_transferred_internal = op->bytes_transferred_internal;
    bool     publish                    = op->write_result && op->committed;
    try {
        Context<AsyncMonitor>::get()->completeOp(op);
    }
    catch (const std::invalid_argument &) {
        if (publish) {
            *bytes_transferred = -hipFileInternalError;
        }
        return;
    }
    if (publish) {
        *bytes_transferred = bytes_transferred_internal;
    }
}
}
