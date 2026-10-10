/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "backend/host.h"

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <sys/types.h>
#include <syslog.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <variant>

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

using namespace hipFile;

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

namespace {

template <typename CopyFn> struct CopyOp : CopyFn {
    CopyOp(IFile &file, IBuffer &buffer, size_t size, hoff_t file_offset, hoff_t buffer_offset);
    size_t run();

private:
    using host_buff_t = CopyFn::host_buff_t;

    int         fd;
    host_buff_t host_buf;
    size_t      size;
    off_t       offset;
};

struct ReadFileFn {
    static constexpr IoType op_type = IoType::Read;

    using host_buff_t = uint8_t *;
    Sys *system       = Context<Sys>::get();

    ssize_t run(int fd, host_buff_t host_buf, size_t size, off_t offset)
    {
        return system->pread(fd, host_buf, size, offset);
    }

    void sync(int /*fd: ignored*/)
    {
    }
};

struct WriteFileFn {
    static constexpr IoType op_type = IoType::Write;

    using host_buff_t = const uint8_t *;
    Sys *system       = Context<Sys>::get();

    ssize_t run(int fd, host_buff_t host_buf, size_t size, off_t offset)
    {
        return system->pwrite(fd, const_cast<uint8_t *>(host_buf), size, offset);
    }
    void sync(int fd)
    {
        system->fdatasync(fd);
    }
};

template <typename CopyFn>
CopyOp<CopyFn>::CopyOp(IFile &file, IBuffer &buffer, size_t op_size, hoff_t file_offset, hoff_t buffer_offset)
    : CopyFn{}, fd{file.bufferedFd()}, host_buf{static_cast<host_buff_t>(buffer.getBuffer()) + buffer_offset},
      size{op_size}, offset{file_offset}
{
}

template <typename CopyFn>
size_t
CopyOp<CopyFn>::run()
{
    size_t  total_io_bytes = 0;
    ssize_t io_bytes       = 0;
    do {
        try {
            io_bytes = CopyFn::run(fd, host_buf + total_io_bytes, size - total_io_bytes,
                                   offset + static_cast<off_t>(total_io_bytes));
            total_io_bytes += static_cast<size_t>(io_bytes);
        }
        catch (const std::system_error &e) {
            if (e.code().value() == EINTR) {
                io_bytes = -1; // while condition re-evaluated after continue
                continue;
            }
            Context<StatsCollection>::get()->error(CopyFn::op_type, StatsBackend::Host, size);
            throw;
        }
        catch (...) {
            Context<StatsCollection>::get()->error(CopyFn::op_type, StatsBackend::Host, size);
            throw;
        }
    } while (io_bytes != 0 && total_io_bytes < size);

    CopyFn::sync(fd);

    return total_io_bytes;
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

    size_t total_io_bytes = type == IoType::Read
                                ? CopyOp<ReadFileFn>(*file, *buffer, size, file_offset, buffer_offset).run()
                                : CopyOp<WriteFileFn>(*file, *buffer, size, file_offset, buffer_offset).run();

    ioTracker.complete(total_io_bytes);

    return static_cast<ssize_t>(total_io_bytes);
}

#pragma GCC diagnostic push
// Ignore missing [[noreturn]] attribute due to unimplemented exception throw.
// This function will return as soon as the async logic is implemented.
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
void
Host::async_io(IoType, std::shared_ptr<IFile>, std::shared_ptr<IBuffer>, size_t *, hoff_t *, hoff_t *,
               ssize_t *, std::shared_ptr<IStream>)
{
    throw std::runtime_error("Host::async_io is not implemented");
}

void
Host::enqueueAsyncIo(IoType, std::shared_ptr<IFile>, std::shared_ptr<IBuffer>, size_t *, hoff_t *, hoff_t *,
                     ssize_t *, std::shared_ptr<IStream>, std::shared_ptr<AsyncFailoverState>)
{
    throw std::runtime_error("Host::enqueueAsyncIo is not implemented");
}
#pragma GCC diagnostic pop
