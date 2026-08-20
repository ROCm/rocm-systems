/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/host.h"
#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "configuration.h"
#include "context.h"
#include "file.h"
#include "hip.h"
#include "io.h"
#include "stats.h"
#include "stream.h"
#include "sys.h"
#include "util.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <linux/io_uring.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/types.h>
#include <syslog.h>
#include <system_error>
#include <unistd.h>

using namespace hipFile;

namespace {

template <typename T> struct HipHostAllocator {
    using value_type = T;

    HipHostAllocator()                         = default;
    HipHostAllocator(const HipHostAllocator &) = default;

    // Rebound constructor
    template <typename U> explicit HipHostAllocator(const HipHostAllocator<U> &)
    {
    }

    T *allocate(std::size_t n)
    try {
        void *p = Context<Hip>::get()->hipHostMalloc(n * sizeof(T), 0);
        return static_cast<T *>(p);
    }
    catch (...) {
        throw std::bad_alloc{};
    }

    void deallocate(T *p, std::size_t n)
    {
        if (p && n != 0) {
            Context<Hip>::get()->hipHostFree(p);
        }
    }
};

}

int
Host::score(const std::shared_ptr<IFile> &file, const std::shared_ptr<IBuffer> &buffer, size_t size,
            hoff_t file_offset, hoff_t buffer_offset) const
{
    (void)file;
    (void)size;
    (void)file_offset;
    (void)buffer_offset;

    if (!Context<Configuration>::get()->host()) {
        return -1;
    }
    if (buffer->getType() != hipMemoryTypeHost) {
        return -1;
    }
    return 1;
}

void *
AsyncOpHost::operator new(size_t size)
{
    HipHostAllocator<AsyncOpHost> alloc{};
    return static_cast<void *>(alloc.allocate(size));
}

void
AsyncOpHost::operator delete(void *ptr) noexcept
{
    HipHostAllocator<AsyncOpHost> alloc{};
    alloc.deallocate(static_cast<AsyncOpHost *>(ptr), sizeof(AsyncOpHost));
}

AsyncOpHost::AsyncOpHost(IoType _io_type, std::shared_ptr<IFile> _file, std::shared_ptr<IBuffer> _buffer,
                         std::shared_ptr<IStream> _stream, size_t *_size, hoff_t *_file_offset,
                         hoff_t *_buffer_offset, ssize_t *_bytes_transferred)
    : AsyncOp{_io_type, std::move(_file), std::move(_buffer), std::move(_stream),
              _size,    _file_offset,     _buffer_offset,     _bytes_transferred},
      submitted_size{std::min(*_size, hipFile::getMaxRwCount())}
{
}

AsyncOpHost::~AsyncOpHost() = default;

namespace {

bool
paramsValid(const AsyncOpHost &op)
{
    if (std::get<size_t>(op.size) > op.submitted_size) {
        return false;
    }
    return paramsValid(op.buffer, std::get<size_t>(op.size), std::get<const hoff_t>(op.file_offset),
                       std::get<const hoff_t>(op.buffer_offset));
}

size_t
do_copy(IoType type, IFile &file, IBuffer &buffer, size_t size, hoff_t file_offset, hoff_t buffer_offset)
{
    int   fd       = file.bufferedFd();
    auto *host_buf = static_cast<uint8_t *>(buffer.getBuffer()) + buffer_offset;

    size_t  total_io_bytes = 0;
    ssize_t io_bytes       = 0;
    do {
        auto count  = size - total_io_bytes;
        auto offset = file_offset + total_io_bytes;
        try {
            switch (type) {
                case IoType::Read:
                    io_bytes = Context<Sys>::get()->pread(fd, host_buf, count, offset);
                    break;
                case IoType::Write:
                    io_bytes = Context<Sys>::get()->pwrite(fd, host_buf, count, offset);
                    break;
                default:
                    throw std::runtime_error("Invalid IO type");
            }
        }
        catch (const std::system_error &e) {
            if (e.code().value() == EINTR) {
                continue;
            }
            Context<StatsCollection>::get()->error(type, StatsBackend::Host, size);
            throw;
        }
        catch (...) {
            Context<StatsCollection>::get()->error(type, StatsBackend::Host, size);
            throw;
        }

        total_io_bytes += io_bytes;
    } while (io_bytes > 0 && total_io_bytes < size);

    if (type == IoType::Write) {
        Context<Sys>::get()->fdatasync(file.bufferedFd());
    }

    return total_io_bytes;
}

void
async_io_host_do(void *userargs)
{
    auto &op = *static_cast<AsyncOpHost *>(userargs);

    size_t size;
    hoff_t buffer_offset, file_offset;

    // Bind params. Will maintain same value if already bound.
    buffer_offset = op.buffer_offset.emplace<const hoff_t>(*get_variant_ptr(op.buffer_offset));
    file_offset   = op.file_offset.emplace<const hoff_t>(*get_variant_ptr(op.file_offset));
    size          = op.size.emplace<size_t>(std::min(*get_variant_ptr(op.size), hipFile::getMaxRwCount()));

    if (!paramsValid(op)) {
        op.bytes_transferred_internal = -hipFileInvalidValue;
        return;
    }

    size_t bytes_transferred = do_copy(op.io_type, *op.file, *op.buffer, size, file_offset, buffer_offset);

    op.bytes_transferred_internal = static_cast<ssize_t>(bytes_transferred);
}

} // namespace

ssize_t
Host::_io_impl(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size,
               hoff_t file_offset, hoff_t buffer_offset)
{
    if (!Context<Configuration>::get()->host()) {
        throw BackendDisabled();
    }

    StatsIoTracker ioTracker{type, StatsBackend::Host, file, buffer, size, file_offset, buffer_offset};

    size = std::min(size, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, size, file_offset, buffer_offset)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    size_t total_io_bytes = do_copy(type, *file, *buffer, size, file_offset, buffer_offset);

    ioTracker.complete(total_io_bytes);

    return total_io_bytes;
}

void
Host::async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t *size_p,
               hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
               std::shared_ptr<IStream> stream)
{
    size_t limited_size = std::min(*size_p, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, limited_size, *file_offset_p, *buffer_offset_p)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    *bytes_transferred_p = 0;

    if (*size_p == 0) {
        return;
    }

    auto op = std::allocate_shared<AsyncOpHost>(HipHostAllocator<AsyncOpHost>{}, type, std::move(file),
                                                buffer, stream, size_p, file_offset_p, buffer_offset_p,
                                                bytes_transferred_p);

    Context<AsyncMonitor>::get()->addOp(op);

    try {
        auto stream_lock = stream->getLock();

        Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_host_do, op.get());
        Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_cleanup, op.get());
    }
    catch (...) {
        try {
            Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_cleanup, op.get());
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT,
                                        "Unable to enqueue async cleanup function. This will leak memory.");
        }
        throw;
    }
}
