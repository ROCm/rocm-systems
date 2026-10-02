/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "configuration.h"
#include "context.h"
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
#include <utility>

namespace hipFile {
Stream::Stream(const hipStream_t _hip_stream, uint32_t flags, const PassKey<StreamMap> &)
    : hip_stream{_hip_stream}, device_id{0}, fixed_buf_offset{(flags & HIPFILE_STREAM_FIXED_BUF_OFFSET) != 0},
      fixed_file_offset{(flags & HIPFILE_STREAM_FIXED_FILE_OFFSET) != 0},
      fixed_io_size{(flags & HIPFILE_STREAM_FIXED_FILE_SIZE) != 0},
      page_aligned{(flags & HIPFILE_STREAM_PAGE_ALIGNED_INPUTS) != 0}, can_use_stream_wait_value{false},
      copy_stream{nullptr}, signal_slot{nullptr}, signal_counter{0}

{
    if ((flags & HIPFILE_STREAM_FLAGS_MASK) != flags) {
        throw std::invalid_argument("Invalid flags for stream");
    }

    device_id = Context<Hip>::get()->hipStreamGetDevice(hip_stream);

    can_use_stream_wait_value =
        Context<Hip>::get()->hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, device_id) != 0;

    copy_stream = Context<Hip>::get()->hipStreamCreateWithFlags(hipStreamNonBlocking);

    if (can_use_stream_wait_value) {
        signal_slot = static_cast<uint64_t *>(
            Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory));
        std::atomic_ref<uint64_t>{*signal_slot}.store(0, std::memory_order_release);
    }
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
    if (StreamMap::from_ptr.end() != StreamMap::from_ptr.find(hip_stream)) {
        throw std::invalid_argument("Stream is already registered");
    }

    auto stream = std::shared_ptr<Stream>(new Stream(hip_stream, flags, PassKey<StreamMap>{}));
    StreamMap::from_ptr[hip_stream] = stream;
}

void
StreamMap::deregisterStream(hipStream_t hip_stream)
{
    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        throw std::invalid_argument("Stream is not registered");
    }

    StreamMap::from_ptr.erase(hip_stream);
}

std::shared_ptr<IStream>
StreamMap::getStream(hipStream_t hip_stream)
{
    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        return std::shared_ptr<Stream>(new Stream(hip_stream, 0, PassKey<StreamMap>{}));
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

hipEvent_t
Stream::acquireEvent()
{
    {
        std::lock_guard<std::mutex> lock{event_mutex};
        if (!event_pool.empty()) {
            hipEvent_t event = event_pool.back();
            event_pool.pop_back();
            return event;
        }
    }
    return Context<Hip>::get()->hipEventCreateWithFlags(hipEventDisableTiming);
}

void
Stream::releaseEvent(hipEvent_t event)
{
    std::lock_guard<std::mutex> lock{event_mutex};
    event_pool.push_back(event);
}

Stream::~Stream()
{
    try {
        for (hipEvent_t event : event_pool) {
            Context<Hip>::get()->hipEventDestroy(event);
        }
        if (signal_slot) {
            Context<Hip>::get()->hipFree(signal_slot);
        }
        if (copy_stream) {
            Context<Hip>::get()->hipStreamDestroy(copy_stream);
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
