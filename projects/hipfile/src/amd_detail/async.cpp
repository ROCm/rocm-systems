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
            bool idle{false};
            {
                auto nh = submitted_ops.extract(op);
                idle    = submitted_ops.empty();
                // Lock released before the op is destroyed. Returning the op to the pool is cheap, but
                // holding the lock while another host function runs completeOp and waits for it would
                // deadlock the moment a synchronizing free runs.
                lock.unlock();
            }
            // With nothing outstanding, release the pooled pinned allocations. hipHostFree/hipFree
            // synchronize, so this runs only at idle rather than once per completed op.
            if (idle) {
                Context<AsyncResourcePool>::get()->drain();
            }
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
    return Context<AsyncResourcePool>::get()->acquireSignal();
}

AsyncOp::~AsyncOp()
{
    if (signal_slot) {
        Context<AsyncResourcePool>::get()->releaseSignal(signal_slot);
    }
}

AsyncResourcePool::AsyncResourcePool()  = default;
AsyncResourcePool::~AsyncResourcePool() = default;

void *
AsyncResourcePool::acquireOp(size_t size)
{
    int device = Context<Hip>::get()->hipGetDevice();
    {
        std::lock_guard<std::mutex> lock{mutex};
        auto                        found = free_lists.find(device);
        if (found != free_lists.end() && !found->second.ops.empty()) {
            void *ptr = found->second.ops.back();
            found->second.ops.pop_back();
            return ptr;
        }
    }
    void *ptr = Context<Hip>::get()->hipHostMalloc(size, 0);
    {
        std::lock_guard<std::mutex> lock{mutex};
        op_devices.emplace(ptr, device);
    }
    return ptr;
}

void
AsyncResourcePool::releaseOp(void *ptr) noexcept
{
    std::lock_guard<std::mutex> lock{mutex};
    auto                        found = op_devices.find(ptr);
    if (found == op_devices.end()) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Releasing an untracked AsyncOpFallback allocation.");
        return;
    }
    free_lists[found->second].ops.push_back(ptr);
}

uint64_t *
AsyncResourcePool::acquireSignal()
{
    int       device = Context<Hip>::get()->hipGetDevice();
    uint64_t *slot   = nullptr;
    {
        std::lock_guard<std::mutex> lock{mutex};
        auto                        found = free_lists.find(device);
        if (found != free_lists.end() && !found->second.signals.empty()) {
            slot = found->second.signals.back();
            found->second.signals.pop_back();
        }
    }
    if (slot == nullptr) {
        slot = static_cast<uint64_t *>(
            Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory));
        std::lock_guard<std::mutex> lock{mutex};
        signal_devices.emplace(slot, device);
    }
    std::atomic_ref<uint64_t>{*slot}.store(0, std::memory_order_release);
    return slot;
}

void
AsyncResourcePool::releaseSignal(uint64_t *slot) noexcept
{
    std::lock_guard<std::mutex> lock{mutex};
    auto                        found = signal_devices.find(slot);
    if (found == signal_devices.end()) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Releasing an untracked async signal slot.");
        return;
    }
    free_lists[found->second].signals.push_back(slot);
}

void
AsyncResourcePool::drain()
{
    std::vector<void *>     ops;
    std::vector<uint64_t *> signals;
    {
        std::lock_guard<std::mutex> lock{mutex};
        for (auto &[device, pool] : free_lists) {
            for (void *ptr : pool.ops) {
                ops.push_back(ptr);
                op_devices.erase(ptr);
            }
            for (uint64_t *slot : pool.signals) {
                signals.push_back(slot);
                signal_devices.erase(slot);
            }
            pool.ops.clear();
            pool.signals.clear();
        }
    }
    // Freed outside the lock: hipHostFree/hipFree synchronize the device, and holding the pool lock
    // across them would stall op enqueues that need to acquire.
    for (void *ptr : ops) {
        try {
            Context<Hip>::get()->hipHostFree(ptr);
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT, "Freeing pooled AsyncOpFallback failed.");
        }
    }
    for (uint64_t *slot : signals) {
        try {
            Context<Hip>::get()->hipFree(slot);
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT, "Freeing pooled async signal memory failed.");
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
