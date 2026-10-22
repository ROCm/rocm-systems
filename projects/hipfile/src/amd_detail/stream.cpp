/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#include "context.h"
#include "device-guard.h"
#include "hip.h"
#include "hipfile.h"
#include "passkey.h"
#include "stream.h"
#include "sys.h"

#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <syslog.h>
#include <stdexcept>
#include <utility>

namespace hipFile {

void
SignalSlotDeleter::operator()(uint64_t *slot) const noexcept
{
    try {
        DeviceGuard guard{device_id};
        Context<Hip>::get()->hipFree(slot);
    }
    catch (...) {
        Context<Sys>::get()->syslog(LOG_CRIT, "Error freeing stream signal memory.");
    }
}

Stream::Stream(const hipStream_t _hip_stream, uint32_t flags, bool _managed, const PassKey<StreamMap> &)
    : hip_stream{_hip_stream}, device_id{0}, fixed_buf_offset{(flags & HIPFILE_STREAM_FIXED_BUF_OFFSET) != 0},
      fixed_file_offset{(flags & HIPFILE_STREAM_FIXED_FILE_OFFSET) != 0},
      fixed_io_size{(flags & HIPFILE_STREAM_FIXED_FILE_SIZE) != 0},
      page_aligned{(flags & HIPFILE_STREAM_PAGE_ALIGNED_INPUTS) != 0}, managed{_managed},
      can_use_stream_wait_value{false}, signal_slot{nullptr, SignalSlotDeleter{0}}
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

    DeviceGuard guard{device_id};
    SignalSlot  slot{static_cast<uint64_t *>(
                        Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory)),
                    SignalSlotDeleter{device_id}};
    *slot       = 0;
    signal_slot = std::move(slot);
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
bool
Stream::isManaged() const
{
    return managed;
}
bool
Stream::isRetired() const
{
    return retired;
}
bool
Stream::canUseStreamWaitValue() const
{
    return can_use_stream_wait_value;
}
uint64_t *
Stream::signalSlot() const
{
    return signal_slot.get();
}
uint64_t
Stream::nextSignalTarget()
{
    return ++signal_counter;
}
uint64_t
Stream::releaseSignalTarget()
{
    return --signal_counter;
}
void
Stream::drainAndRelease()
{
    if (!managed) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock{mutex};
        retired = true;
    }
    Context<Hip>::get()->hipStreamSynchronize(hip_stream);
    signal_slot.reset();
}

void
StreamMap::registerStream(hipStream_t hip_stream, uint32_t flags)
{
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

    StreamMap::from_ptr.erase(hip_stream);
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
