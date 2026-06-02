/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "backend.h"
#include "buffer.h"
#include "backend/asyncop-fallback.h"
#include "backend/memcpy-kernel.h"
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

int
Fallback::score(std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t size, hoff_t file_offset,
                hoff_t buffer_offset) const
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
    StatsIoTracker ioTracker{type, StatsBackend::Fallback};
    if (!Context<Configuration>::get()->fallback()) {
        throw BackendDisabled();
    }

    size = min(size, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, size, file_offset, buffer_offset)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

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
                        Context<Hip>::get()->hipMemcpy(device_buffer_position, bounce_buffer.get(),
                                                       static_cast<size_t>(io_bytes), hipMemcpyHostToDevice);
                    }
                    break;
                case IoType::Write:
                    Context<Hip>::get()->hipMemcpy(bounce_buffer.get(), device_buffer_position, count,
                                                   hipMemcpyDeviceToHost);
                    Context<Hip>::get()->hipStreamSynchronize(nullptr);
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

void
Fallback::async_io(IoType type, std::shared_ptr<IFile> file, std::shared_ptr<IBuffer> buffer, size_t *size_p,
                   hoff_t *file_offset_p, hoff_t *buffer_offset_p, ssize_t *bytes_transferred_p,
                   std::shared_ptr<IStream> stream)
{
    size_t limited_size = min(*size_p, hipFile::getMaxRwCount());

    if (!paramsValid(buffer, limited_size, *file_offset_p, *buffer_offset_p)) {
        throw std::invalid_argument("The selected file or buffer region is invalid");
    }

    if (buffer->getGpuId() != stream->getHipDevice()) {
        throw std::invalid_argument("Buffer GPU ID does not match Stream GPU ID");
    }

    *bytes_transferred_p = 0;

    if (*size_p == 0) {
        return;
    }

    auto op = std::shared_ptr<AsyncOpFallback>(new AsyncOpFallback(
        type, std::move(file), buffer, stream, size_p, file_offset_p, buffer_offset_p, bytes_transferred_p));
    Context<AsyncMonitor>::get()->addOp(op);
    auto  op_dev_ptr     = op->devPtr();
    void *kernel_args[1] = {&op_dev_ptr};

    try {
        int max_threads_per_block = Context<Hip>::get()->hipDeviceGetAttribute(
            hipDeviceAttributeMaxThreadsPerBlock, buffer->getGpuId());
        auto stream_lock = stream->getLock();

        // Launch a host function to bind parameters if anything not fixed
        if (!op->stream->fixedBufferOffset() || !op->stream->fixedFileOffset() ||
            !op->stream->fixedIOSize()) {
            Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_bind_params,
                                                   op.get());
        }

        switch (op->io_type) {
            case IoType::Read:
                Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_cpu_copy,
                                                       op.get());
                Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                                     dim3(static_cast<uint32_t>(max_threads_per_block)),
                                                     kernel_args, 0, op->stream->getHipStream());
                break;
            case IoType::Write:
                Context<Hip>::get()->hipLaunchKernel(reinterpret_cast<void *>(hipFileMemcpyKernel), dim3(1),
                                                     dim3(static_cast<uint32_t>(max_threads_per_block)),
                                                     kernel_args, 0, op->stream->getHipStream());
                Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_cpu_copy,
                                                       op.get());
                break;
            default:
                throw std::runtime_error("Invalid IO type");
        }

        Context<Hip>::get()->hipLaunchHostFunc(op->stream->getHipStream(), async_io_cleanup, op.get());
    }
    catch (...) {
        // If something threw, still try to enqueue cleanup function
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

extern "C" {
void
async_io_bind_params(void *userargs)
{
    auto op = static_cast<AsyncOpFallback *>(userargs);
    // Bind params. Will maintain same value if already bound.
    const hoff_t *buffer_offset = get_variant_ptr(op->buffer_offset);
    op->buffer_offset.emplace<const hoff_t>(*buffer_offset);
    const hoff_t *file_offset = get_variant_ptr(op->file_offset);
    op->file_offset.emplace<const hoff_t>(*file_offset);
    const size_t *size = get_variant_ptr(op->size);
    op->size           = std::min(*size, hipFile::getMaxRwCount());

    if (std::get<size_t>(op->size) > op->submitted_size) {
        op->bytes_transferred_internal = -hipFileInvalidValue;
        return;
    }
    if (!paramsValid(op->buffer, std::get<size_t>(op->size), std::get<const hoff_t>(op->file_offset),
                     std::get<const hoff_t>(op->buffer_offset))) {
        op->bytes_transferred_internal = -hipFileInvalidValue;
    }
}

void
async_io_cleanup(void *userargs)
{
    auto     op                         = static_cast<AsyncOpFallback *>(userargs);
    ssize_t *bytes_transferred          = op->bytes_transferred;
    ssize_t  bytes_transferred_internal = op->bytes_transferred_internal;
    try {
        Context<AsyncMonitor>::get()->completeOp(op);
    }
    catch (const std::invalid_argument &) {
        *bytes_transferred = -hipFileInternalError;
        return;
    }
    *bytes_transferred = bytes_transferred_internal;
}

void
async_io_cpu_copy(void *userargs)
{
    auto         op                = static_cast<AsyncOpFallback *>(userargs);
    size_t       bytes_transferred = 0;
    const size_t size              = std::get<size_t>(op->size);
    const hoff_t file_offset       = std::get<const hoff_t>(op->file_offset);
    ssize_t      ret               = 0;

    if (op->bytes_transferred_internal < 0) {
        return;
    }

    while (bytes_transferred < size) {
        void *cur_buf_position = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(op->bounceBufferHostPtr()) + bytes_transferred);
        hoff_t cur_file_offset = file_offset + static_cast<hoff_t>(bytes_transferred);
        size_t remaining_bytes = size - bytes_transferred;
        try {
            switch (op->io_type) {
                case IoType::Read:
                    ret = Context<Sys>::get()->pread(op->file->bufferedFd(), cur_buf_position,
                                                     remaining_bytes, cur_file_offset);
                    break;
                case IoType::Write:
                    ret = Context<Sys>::get()->pwrite(op->file->bufferedFd(), cur_buf_position,
                                                      remaining_bytes, cur_file_offset);
                    break;
                default:
                    throw std::runtime_error("Invalid IO type");
            }
        }
        catch (const std::system_error &e) {
            if (e.code().value() == EINTR) {
                continue;
            }
            op->bytes_transferred_internal = -1;
            return;
        }
        if (ret == 0) {
            break;
        }
        bytes_transferred += static_cast<size_t>(ret);
    }
    op->bytes_transferred_internal = static_cast<ssize_t>(bytes_transferred);
}
}
