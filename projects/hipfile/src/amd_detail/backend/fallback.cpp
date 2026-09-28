/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend.h"
#include "buffer.h"
#include "configuration.h"
#include "context.h"
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

static const size_t DefaultChunkSize = 16 * 1024 * 1024;

namespace {

// Makes the buffer's GPU current for the lifetime of the guard (hipMemcpy operates on the current
// device's context) and restores the caller's device on scope exit, including during exceptions.
class DeviceGuard {
public:
    explicit DeviceGuard(int buffer_device) : prev_device_{Context<Hip>::get()->hipGetDevice()}
    {
        if (buffer_device != prev_device_) {
            Context<Hip>::get()->hipSetDevice(buffer_device);
            switched_ = true;
        }
    }
    ~DeviceGuard()
    {
        if (switched_) {
            try {
                Context<Hip>::get()->hipSetDevice(prev_device_);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Unable to restore the caller's HIP device.");
            }
        }
    }
    DeviceGuard(const DeviceGuard &)            = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    DeviceGuard(DeviceGuard &&)                 = delete;
    DeviceGuard &operator=(DeviceGuard &&)      = delete;

private:
    int  prev_device_;
    bool switched_{false};
};

} // namespace

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
    return _io_impl(type, std::move(file), std::move(buffer), size, file_offset, buffer_offset, nullptr,
                    chunk_size);
}

ssize_t
Fallback::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                   hoff_t file_offset, hoff_t buffer_offset, hipStream_t copy_stream)
{
    return _io_impl(type, std::move(file), std::move(buffer), size, file_offset, buffer_offset, copy_stream,
                    DefaultChunkSize);
}

ssize_t
Fallback::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
                   hoff_t file_offset, hoff_t buffer_offset, hipStream_t copy_stream, size_t chunk_size)
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
                        if (copy_stream) {
                            Context<Hip>::get()->hipMemcpyWithStream(
                                device_buffer_position, bounce_buffer.get(), static_cast<size_t>(io_bytes),
                                hipMemcpyHostToDevice, copy_stream);
                        }
                        else {
                            Context<Hip>::get()->hipMemcpy(device_buffer_position, bounce_buffer.get(),
                                                           static_cast<size_t>(io_bytes),
                                                           hipMemcpyHostToDevice);
                        }
                    }
                    break;
                case IoType::Write:
                    if (copy_stream) {
                        Context<Hip>::get()->hipMemcpyWithStream(bounce_buffer.get(), device_buffer_position,
                                                                 count, hipMemcpyDeviceToHost, copy_stream);
                    }
                    else {
                        Context<Hip>::get()->hipMemcpy(bounce_buffer.get(), device_buffer_position, count,
                                                       hipMemcpyDeviceToHost);
                        Context<Hip>::get()->hipStreamSynchronize(nullptr);
                    }
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
            Context<StatsCollection>::get()->error(type, StatsBackend::Fallback, size);
            throw;
        }
        catch (...) {
            Context<StatsCollection>::get()->error(type, StatsBackend::Fallback, size);
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
