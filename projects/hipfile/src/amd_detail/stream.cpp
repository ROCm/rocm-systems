/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "configuration.h"
#include "context.h"
#include "device-guard.h"
#include "hip.h"
#include "hipfile.h"
#include "passkey.h"
#include "stream.h"
#include "sys.h"

#include <atomic>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <syslog.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hipFile {
Stream::Stream(const hipStream_t _hip_stream, uint32_t flags, bool _managed, const PassKey<StreamMap> &)
    : hip_stream{_hip_stream}, device_id{0}, fixed_buf_offset{(flags & HIPFILE_STREAM_FIXED_BUF_OFFSET) != 0},
      fixed_file_offset{(flags & HIPFILE_STREAM_FIXED_FILE_OFFSET) != 0},
      fixed_io_size{(flags & HIPFILE_STREAM_FIXED_FILE_SIZE) != 0},
      page_aligned{(flags & HIPFILE_STREAM_PAGE_ALIGNED_INPUTS) != 0}, can_use_stream_wait_value{false},
      managed{_managed}, copy_stream{nullptr}, signal_slot{nullptr}, signal_counter{0}

{
    if ((flags & HIPFILE_STREAM_FLAGS_MASK) != flags) {
        throw std::invalid_argument("Invalid flags for stream");
    }

    device_id = Context<Hip>::get()->hipStreamGetDevice(hip_stream);

    if (!managed) {
        return;
    }

    can_use_stream_wait_value =
        Context<Hip>::get()->hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, device_id) != 0;

    DeviceGuard device_guard{device_id};

    auto destroy_stream = [](hipStream_t s) noexcept {
        if (s) {
            try {
                Context<Hip>::get()->hipStreamDestroy(s);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
            }
        }
    };
    std::unique_ptr<std::remove_pointer_t<hipStream_t>, decltype(destroy_stream)> owned_stream{
        Context<Hip>::get()->hipStreamCreateWithFlags(hipStreamNonBlocking), destroy_stream};

    auto free_slot = [](uint64_t *p) noexcept {
        if (p) {
            try {
                Context<Hip>::get()->hipFree(p);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
            }
        }
    };
    std::unique_ptr<uint64_t, decltype(free_slot)> owned_slot{nullptr, free_slot};
    if (can_use_stream_wait_value) {
        owned_slot.reset(static_cast<uint64_t *>(
            Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory)));
        std::atomic_ref<uint64_t>{*owned_slot}.store(0, std::memory_order_release);
    }

    copy_stream = owned_stream.release();
    signal_slot = owned_slot.release();
}

hipStream_t
Stream::getHipStream() const
{
    return hip_stream;
}
hipDevice_t
Stream::getHipDevice() const
{
    return device_id;
}
bool
Stream::fixedBufferOffset() const
{
    return fixed_buf_offset;
}
bool
Stream::fixedFileOffset() const
{
    return fixed_file_offset;
}
bool
Stream::fixedIOSize() const
{
    return fixed_io_size;
}
bool
Stream::pageAligned() const
{
    return page_aligned;
}
std::unique_lock<std::mutex>
Stream::getLock()
{
    return std::unique_lock<std::mutex>{mutex};
}

void
StreamMap::registerStream(hipStream_t hip_stream, uint32_t flags)
{
    if (hip_stream == nullptr || hip_stream == hipStreamPerThread) {
        throw std::invalid_argument("The default stream cannot be registered for async operations");
    }

    if (StreamMap::from_ptr.end() != StreamMap::from_ptr.find(hip_stream)) {
        throw std::invalid_argument("Stream is already registered");
    }

    auto stream = std::shared_ptr<Stream>(new Stream(hip_stream, flags, true, PassKey<StreamMap>{}));
    StreamMap::from_ptr[hip_stream] = stream;
}

void
StreamMap::deregisterStream(hipStream_t hip_stream)
{
    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        throw std::invalid_argument("Stream is not registered");
    }

    itr->second->drainAndRelease();
    StreamMap::from_ptr.erase(itr);
}

std::shared_ptr<IStream>
StreamMap::getStream(hipStream_t hip_stream)
{
    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        return std::shared_ptr<Stream>(new Stream(hip_stream, 0, false, PassKey<StreamMap>{}));
    }

    return itr->second;
}

bool
Stream::canUseStreamWaitValue() const
{
    return can_use_stream_wait_value;
}

hipStream_t
Stream::copyStream() const
{
    return copy_stream;
}

uint64_t *
Stream::signalSlot() const
{
    return signal_slot;
}

uint64_t
Stream::nextSignalTarget()
{
    return ++signal_counter;
}

void
Stream::releaseSignalTarget()
{
    --signal_counter;
}

bool
Stream::isManaged() const
{
    return managed;
}

void
Stream::drainAndRelease()
{
    if (!managed) {
        return;
    }

    Context<Hip>::get()->hipStreamSynchronize(hip_stream);

    DeviceGuard device_guard{device_id};
    if (signal_slot) {
        uint64_t *slot = signal_slot;
        signal_slot    = nullptr;
        try {
            Context<Hip>::get()->hipFree(slot);
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
        }
    }
    if (copy_stream) {
        hipStream_t stream = copy_stream;
        copy_stream        = nullptr;
        try {
            Context<Hip>::get()->hipStreamDestroy(stream);
        }
        catch (...) {
            Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
        }
    }
}

Stream::~Stream()
{
    if (copy_stream == nullptr && signal_slot == nullptr) {
        return;
    }
    try {
        DeviceGuard device_guard{device_id};
        if (signal_slot) {
            try {
                Context<Hip>::get()->hipFree(signal_slot);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
            }
        }
        if (copy_stream) {
            try {
                Context<Hip>::get()->hipStreamDestroy(copy_stream);
            }
            catch (...) {
                Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
            }
        }
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Error releasing stream async resources.");
    }
}

StreamMap::~StreamMap()
{
    // Complain in the logs if we're shutting down a Stream
    // with outstanding operations.
    int count = 0;
    for (const auto &p : from_ptr) {
        if (p.second.use_count() > 1) {
            count++;
        }
    }

    if (count > 0) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Stream state is being destructed with outstanding operations");
    }
}

}
