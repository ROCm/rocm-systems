/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "async.h"
#include "asyncop-fallback.h"
#include "backend.h"
#include "buffer.h"
#include "context.h"
#include "hip.h"
#include "sys.h"

#include <memory>
#include <new>
#include <syslog.h>

namespace hipFile {
class IFile;
}
namespace hipFile {
class IStream;
}
namespace hipFile {
enum class IoType;
}

using namespace hipFile;

static void
hipHostDeleter(void *buffer)
{
    try {
        Context<Hip>::get()->hipHostFree(buffer);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Error freeing pinned host memory.");
    }
}

AsyncOpFallback::AsyncOpFallback(IoType _io_type, std::shared_ptr<IFile> _file,
                                 std::shared_ptr<IBuffer> _buffer, std::shared_ptr<IStream> _stream,
                                 size_t *_size, hoff_t *_file_offset, hoff_t *_buffer_offset,
                                 ssize_t *_bytes_transferred)
    : AsyncOp{_io_type, std::move(_file), std::move(_buffer), std::move(_stream),
              _size,    _file_offset,     _buffer_offset,     _bytes_transferred},
      submitted_size{std::min(*_size, hipFile::getMaxRwCount())}, bytes_transferred_internal{0},
      gpu_buffer{buffer->getBuffer()}, bounce_buffer_dev_ptr{nullptr},
      bounce_buffer{nullptr, [](void *addr) { (void)addr; }}
{
    void *host_ptr = Context<Hip>::get()->hipHostMalloc(submitted_size, 0);
    std::unique_ptr<void, decltype(&hipHostDeleter)> _bounce_buffer{host_ptr, hipHostDeleter};
    std::swap(bounce_buffer, _bounce_buffer);
    void *dev_ptr         = Context<Hip>::get()->hipHostGetDevicePointer(bounce_buffer.get(), 0);
    bounce_buffer_dev_ptr = dev_ptr;
}

void *
AsyncOpFallback::bounceBufferHostPtr()
{
    return bounce_buffer.get();
}

void *
AsyncOpFallback::devPtr()
{
    return Context<Hip>::get()->hipHostGetDevicePointer(this, 0);
}

AsyncOpFallback::~AsyncOpFallback()
{
}

void *
AsyncOpFallback::operator new(size_t size_)
{
    try {
        return Context<Hip>::get()->hipHostMalloc(size_, 0);
    }
    catch (...) {
        throw std::bad_alloc{};
    }
}

void
AsyncOpFallback::operator delete(void *ptr) noexcept
{
    try {
        Context<Hip>::get()->hipHostFree(ptr);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Freeing AsyncOpFallback failed.");
    }
}
