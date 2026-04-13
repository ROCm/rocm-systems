/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "buffer.h"
#include "context.h"
#include "hip.h"
#include "passkey.h"
#include "sys.h"

#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <syslog.h>
#include <utility>

using namespace std;

namespace hipFile {

static bool
isValidBufferRegion(void *ptr, size_t length)
{
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);

    try {
        HipMemAddressRange buffer_range = Context<Hip>::get()->hipMemGetAddressRange(ptr);
        uintptr_t          base         = reinterpret_cast<uintptr_t>(buffer_range.base);

        // Overflow check
        // Do this before the addition, so the sanitizers don't complain
        // about integer overflow
        if (length > std::numeric_limits<uintptr_t>::max() - uptr) {
            return false;
        }

        // Don't need to check base + buffer_range.size since
        // HIP shouldn't give us invalid values

        if (uptr + length > base + buffer_range.size) {
            return false;
        }
    }
    catch (Hip::RuntimeError &e) {
        if (e.error == hipErrorNotFound) {
            throw std::invalid_argument("Buffer pointer not found.");
        }
        throw;
    }

    return true;
}

Buffer::Buffer(const void *_buffer, size_t _length, int _flags, const PassKey<BufferMap> &)
    : buffer{const_cast<void *>(_buffer)}, length{_length}, flags{_flags}
{
    if (!buffer) {
        throw std::invalid_argument("Buffer pointer cannot be null.");
    }

    hipPointerAttribute_t _attrs = Context<Hip>::get()->hipPointerGetAttributes(buffer);
    if (_attrs.type != hipMemoryTypeDevice) {
        throw InvalidMemoryType();
    }
    type   = _attrs.type;
    gpu_id = _attrs.device;

    if (!isValidBufferRegion(buffer, length)) {
        throw InvalidPointerRange();
    }
}

Buffer::Buffer(const void *_buffer, const PassKey<BufferMap> &)
    : buffer{const_cast<void *>(_buffer)}, length{}, flags{}
{
    if (!buffer) {
        throw std::invalid_argument("Buffer pointer cannot be null.");
    }

    hipPointerAttribute_t _attrs = Context<Hip>::get()->hipPointerGetAttributes(buffer);
    if (_attrs.type != hipMemoryTypeDevice) {
        throw InvalidMemoryType();
    }
    type   = _attrs.type;
    gpu_id = _attrs.device;

    HipMemAddressRange range{Context<Hip>::get()->hipMemGetAddressRange(buffer)};
    length = range.size - (reinterpret_cast<uintptr_t>(buffer) - reinterpret_cast<uintptr_t>(range.base));
}

void *
Buffer::getBuffer() const
{
    return buffer;
}

size_t
Buffer::getLength() const
{
    return length;
}

int
Buffer::getFlags() const
{
    return flags;
}

hipMemoryType
Buffer::getType() const
{
    return type;
}

int
Buffer::getGpuId() const
{
    return gpu_id;
}

void
BufferMap::registerBuffer(const void *buf, size_t length, int flags)
{
    if (from_ptr.end() != from_ptr.find(buf)) {
        throw BufferAlreadyRegistered();
    }

    auto buffer   = std::shared_ptr<IBuffer>(new Buffer(buf, length, flags, PassKey<BufferMap>{}));
    from_ptr[buf] = std::move(buffer);
}

void
BufferMap::deregisterBuffer(const void *buf)
{
    auto itr = from_ptr.find(buf);
    if (from_ptr.end() == itr) {
        throw BufferNotRegistered();
    }

    if (1 < itr->second.use_count()) {
        throw BufferOperationsOutstanding();
    }

    from_ptr.erase(buf);
}

shared_ptr<IBuffer>
BufferMap::getRegisteredBuffer(const void *buf)
{
    auto itr = from_ptr.find(buf);
    if (from_ptr.end() == itr) {
        throw BufferNotRegistered();
    }

    return itr->second;
}

shared_ptr<IBuffer>
BufferMap::getBuffer(const void *buf)
{
    auto itr = from_ptr.find(buf);

    if (from_ptr.end() == itr) {
        // Create a temporary buffer
        return std::shared_ptr<IBuffer>(new Buffer(buf, PassKey<BufferMap>{}));
    }
    else {
        return itr->second;
    }
}

void
BufferMap::clear()
{
    from_ptr.clear();
}

BufferMap::~BufferMap()
{
    // Complain in the logs if we're shutting down a BufferMap
    // with buffers that are still in use.
    int count = 0;
    for (const auto &p : from_ptr) {
        if (p.second.use_count() > 1) {
            count++;
        }
    }

    if (count > 0) {
        Context<Sys>::get()->syslog(LOG_CRIT, "BufferMap state is being destructed with in-use buffers");
    }
}

}
