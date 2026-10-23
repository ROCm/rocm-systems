/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "buffer.h"
#include "configuration.h"
#include "context.h"
#include "device-guard.h"
#include "fallback.h"
#include "file.h"
#include "hip.h"
#include "hipfile.h"
#include "io.h"
#include "stats.h"
#include "sys.h"
#include "stream.h"
#include "util.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <syslog.h>
#include <system_error>
#include <variant>
#include <utility>

using namespace hipFile;

using std::min;
using std::shared_ptr;
using std::unique_ptr;

BounceStream::BounceStream(int _device_id, hipStream_t _stream) : device_id{_device_id}, stream{_stream}
{
}

BounceStream::~BounceStream()
{
    try {
        DeviceGuard device_guard{device_id};
        Context<Hip>::get()->hipStreamDestroy(stream);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Unable to destroy fallback bounce stream");
    }
}

BounceStreamPool::~BounceStreamPool()
{
    clear();
}

void
BounceStreamPool::clear()
{
    std::lock_guard<std::mutex> lock(mtx);
    stream_rings.clear();
}

std::shared_ptr<BounceStream>
BounceStreamPool::getStream(int device_id)
{
    std::lock_guard<std::mutex> lock(mtx);

    DeviceGuard device_guard{device_id};

    auto [it, inserted] = stream_rings.try_emplace(device_id, StreamRing{});

    auto &ring = it->second;

    if (inserted || ring.streams.empty()) {
        ring.streams.assign(Context<Configuration>::get()->deviceStreamPoolSize(), nullptr);
    }
    auto &slot   = ring.streams[ring.current];
    ring.current = (ring.current + 1) % ring.streams.size();
    if (!slot) {
        // HIP pools hardware queues per stream priority. A priority callers rarely use keeps bounce copies
        // off queues where caller streams can be parked on stream waits or host functions.
        auto priority = Context<Hip>::get()->hipDeviceGetStreamPriorityRange().greatest;
        slot          = std::make_shared<BounceStream>(
            device_id, Context<Hip>::get()->hipStreamCreateWithPriority(hipStreamNonBlocking, priority));
    }
    return slot;
}

int
Fallback::score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
                hoff_t file_offset, hoff_t buffer_offset) const
{
    (void)buffer_offset;
    (void)file;
    (void)file_offset;
    (void)size;

    return Context<Configuration>::get()->fallback() && buffer->getType() == hipMemoryTypeDevice ? 0 : -1;
}

ssize_t
Fallback::io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
             hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size)
{
    return _io_impl(type, std::move(file), std::move(buffer), size, file_offset, buffer_offset, chunk_size);
}

ssize_t
Fallback::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                   hoff_t file_offset, hoff_t buffer_offset)
{
    return _io_impl(type, std::move(file), std::move(buffer), size, file_offset, buffer_offset,
                    DefaultChunkSize);
}

ssize_t
Fallback::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                   hoff_t file_offset, hoff_t buffer_offset, size_t chunk_size)
{
    if (!Context<Configuration>::get()->fallback()) {
        throw BackendDisabled();
    }

    StatsIoTracker ioTracker{type, StatsBackend::Fallback, file, buffer, size, file_offset, buffer_offset};

    size = min(size, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, size, file_offset, buffer_offset)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    DeviceGuard device_guard{buffer->getGpuId()};

    auto ptr     = Context<Sys>::get()->mmap(nullptr, chunk_size, PROT_READ | PROT_WRITE,
                                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto deleter = [&](void *addr) { Context<Sys>::get()->munmap(addr, chunk_size); };
    unique_ptr<void, decltype(deleter)> bounce_buffer{ptr, deleter};
    auto                                stream_holder = bounce_stream_pool.getStream(buffer->getGpuId());
    hipStream_t                         stream        = stream_holder->stream;

    ssize_t total_io_bytes = 0;
    do {
        auto    count                  = min(chunk_size, size - static_cast<size_t>(total_io_bytes));
        auto    offset                 = file_offset + total_io_bytes;
        ssize_t io_bytes               = 0;
        auto    device_buffer_position = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(buffer->getBuffer()) + static_cast<size_t>(buffer_offset) +
            static_cast<size_t>(total_io_bytes));
        try {
            switch (type) {
                case IoType::Read:
                    io_bytes =
                        Context<Sys>::get()->pread(file->bufferedFd(), bounce_buffer.get(), count, offset);
                    if (io_bytes > 0) {
                        Context<Hip>::get()->hipMemcpyWithStream(device_buffer_position, bounce_buffer.get(),
                                                                 static_cast<size_t>(io_bytes),
                                                                 hipMemcpyHostToDevice, stream);
                        Context<Hip>::get()->hipStreamSynchronize(stream);
                    }
                    break;
                case IoType::Write:
                    Context<Hip>::get()->hipMemcpyWithStream(bounce_buffer.get(), device_buffer_position,
                                                             count, hipMemcpyDeviceToHost, stream);
                    Context<Hip>::get()->hipStreamSynchronize(stream);
                    io_bytes =
                        Context<Sys>::get()->pwrite(file->bufferedFd(), bounce_buffer.get(), count, offset);
                    Context<Sys>::get()->fdatasync(file->bufferedFd());
                    break;
                default:
                    throw std::runtime_error("Invalid IO type");
            }
        }
        catch (const std::system_error &e) {
            if (e.code().value() == EINTR) {
                continue;
            }
            Context<StatsCollection>::get()->error(type, StatsBackend::Fallback, size, buffer->getGpuId());
            throw;
        }
        catch (...) {
            Context<StatsCollection>::get()->error(type, StatsBackend::Fallback, size, buffer->getGpuId());
            throw;
        }

        total_io_bytes += io_bytes;
        if (io_bytes == 0) {
            break;
        }
    } while (static_cast<size_t>(total_io_bytes) < size);

    ioTracker.complete(static_cast<size_t>(total_io_bytes));

    return total_io_bytes;
}

void
Fallback::driver_state_teardown()
{
    bounce_stream_pool.clear();
}
