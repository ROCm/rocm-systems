/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "context.h"
#include "cooperative-copy.h"
#include "hip.h"
#include "io.h"
#include "sys.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <span>
#include <syslog.h>
#include <system_error>
#include <thread>
#include <utility>

namespace hipFile {

static_assert(std::atomic_ref<uint32_t>::is_always_lock_free);
static_assert(std::atomic_ref<uint64_t>::is_always_lock_free);

static void
waitFor(uint32_t &word, uint32_t value)
{
    while (std::atomic_ref<uint32_t>{word}.load(std::memory_order_acquire) != value) {
        std::this_thread::yield();
    }
}

CooperativeCopy::CooperativeCopy(Hip &hip)
    : host{std::construct_at(static_cast<CopyExchange *>(
               hip.hipHostMalloc(sizeof(CopyExchange), hipHostMallocMapped | hipHostMallocCoherent))),
           Deleter{}},
      device{static_cast<CopyExchange *>(hip.hipHostGetDevicePointer(host.get(), 0))}
{
}

void
CooperativeCopy::Deleter::operator()(CopyExchange *ptr) const noexcept
{
    try {
        Context<Hip>::get()->hipHostFree(ptr);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Unable to free cooperative copy buffers");
    }
}

void
CooperativeCopy::launch(hipStream_t stream, void *buffer, bool read, uint64_t ticket)
{
    std::array<void *, 4> args{&device, &buffer, &read, &ticket};
    Context<Hip>::get()->hipLaunchKernel(cooperativeCopyKernel(), dim3(1), dim3(256), args.data(), 0, stream);
}

void
CooperativeCopy::prepare(uint64_t ticket)
{
    host->stop     = 0;
    host->stopped  = 0;
    host->finished = 0;
    for (auto &slot : host->slots) {
        slot.requested = 0;
    }
    std::atomic_ref<uint64_t>{host->ticket}.store(ticket, std::memory_order_release);
}

bool
CooperativeCopy::resident(uint64_t ticket) const
{
    return std::atomic_ref<uint64_t>{host->resident}.load(std::memory_order_acquire) == ticket;
}

template <typename F>
static auto
retryInterrupted(F syscall)
{
    for (;;) {
        try {
            return syscall();
        }
        catch (const std::system_error &e) {
            if (e.code().value() != EINTR) {
                throw;
            }
        }
    }
}

// A count published to the GPU must fit both the requested region and its pinned slot.
static size_t
checkedCount(ssize_t bytes, size_t requested)
{
    if (bytes < 0 || std::cmp_greater(bytes, requested)) {
        throw std::system_error(EIO, std::generic_category());
    }
    return static_cast<size_t>(bytes);
}

static size_t
readChunk(const Sys &sys, int fd, std::span<unsigned char> data, int64_t offset)
{
    const auto bytes = retryInterrupted([&] { return sys.pread(fd, data.data(), data.size(), offset); });
    return checkedCount(bytes, data.size());
}

static size_t
writeChunk(const Sys &sys, int fd, std::span<unsigned char> data, int64_t offset)
{
    const auto bytes = retryInterrupted([&] { return sys.pwrite(fd, data.data(), data.size(), offset); });
    const auto count = checkedCount(bytes, data.size());
    retryInterrupted([&] { sys.fdatasync(fd); });
    return count;
}

static void
requestCopy(CopySlot &slot, size_t offset, size_t count)
{
    slot.offset = offset;
    slot.count  = count;
    std::atomic_ref<uint32_t>{slot.requested}.store(1, std::memory_order_release);
}

ssize_t
CooperativeCopy::transfer(Sys &sys, IoType type, int fd, size_t size, int64_t file_offset,
                          size_t buffer_offset)
{
    size_t completed = 0;
    if (size == 0) {
        auto empty = std::span{host->slots.front().data}.first(0);
        return static_cast<ssize_t>(type == IoType::Read ? readChunk(sys, fd, empty, file_offset)
                                                         : writeChunk(sys, fd, empty, file_offset));
    }
    if (type == IoType::Read) {
        for (size_t index = 0; completed < size; index ^= 1U) {
            auto &slot = host->slots[index];
            waitFor(slot.requested, 0);
            const size_t count = std::min(CopySlot::Capacity, size - completed);
            const auto   bytes = readChunk(sys, fd, std::span{slot.data}.first(count),
                                           file_offset + static_cast<int64_t>(completed));
            if (bytes == 0) {
                break;
            }
            requestCopy(slot, buffer_offset + completed, bytes);
            completed += bytes;
        }
    }
    else {
        size_t submitted = 0;
        auto   submit    = [&](CopySlot &slot) {
            const size_t count = std::min(CopySlot::Capacity, size - submitted);
            requestCopy(slot, buffer_offset + submitted, count);
            submitted += count;
        };
        for (auto &slot : host->slots) {
            if (submitted < size) {
                submit(slot);
            }
        }
        for (size_t index = 0; completed < size; index ^= 1U) {
            auto &slot = host->slots[index];
            waitFor(slot.requested, 0);
            auto remaining = std::span{slot.data}.first(slot.count);
            while (!remaining.empty()) {
                const auto bytes =
                    writeChunk(sys, fd, remaining, file_offset + static_cast<int64_t>(completed));
                if (bytes == 0) {
                    return static_cast<ssize_t>(completed);
                }
                completed += bytes;
                remaining = remaining.subspan(bytes);
            }
            if (submitted < size) {
                submit(slot);
            }
        }
    }
    return static_cast<ssize_t>(completed);
}

void
CooperativeCopy::stop(IoType type)
{
    // Preserve already-read bytes even when a subsequent pread failed. A failed write may
    // discard speculative device-to-host copies once the workgroup has stopped using the slots.
    if (type == IoType::Read) {
        for (auto &slot : host->slots) {
            waitFor(slot.requested, 0);
        }
    }
    std::atomic_ref<uint32_t>{host->stop}.store(1, std::memory_order_release);
    waitFor(host->stopped, 1);
}

void
CooperativeCopy::finish()
{
    std::atomic_ref<uint32_t>{host->finished}.store(1, std::memory_order_release);
}

}
