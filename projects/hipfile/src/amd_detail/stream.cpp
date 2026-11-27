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
#include <chrono>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <syslog.h>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hipFile {

bool
isDefaultStream(hipStream_t hip_stream)
{
    return hip_stream == nullptr || hip_stream == hipStreamPerThread;
}

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

Stream::Stream(const hipStream_t _hip_stream, uint32_t flags, bool blocking_only, const PassKey<StreamMap> &)
    : hip_stream{_hip_stream}, device_id{0}, can_use_stream_wait_value{false},
      signal_slot{nullptr, SignalSlotDeleter{0}}
{
    setFlags(flags);

    device_id = Context<Hip>::get()->hipStreamGetDevice(hip_stream);

    if (isDefaultStream(hip_stream)) {
        return;
    }

    // Fallback bounce copies run at the greatest priority; parking a stream there could queue them behind it.
    {
        DeviceGuard guard{device_id};
        auto        priorities = Context<Hip>::get()->hipDeviceGetStreamPriorityRange();
        if (priorities.least != priorities.greatest &&
            Context<Hip>::get()->hipStreamGetPriority(hip_stream) == priorities.greatest) {
            throw ReservedStreamPriority();
        }
    }

    // A transient (unregistered) stream always uses the blocking offload path, so it needs no signal
    // slot and keeps no persistent per-stream state. So does every stream while forceBlockingAsync() is
    // on: a stream parked on a wait value can deadlock fallback IO when HIP ignores stream priorities.
    can_use_stream_wait_value =
        !blocking_only && !Context<Configuration>::get()->forceBlockingAsync() &&
        Context<Hip>::get()->hipDeviceGetAttribute(hipDeviceAttributeCanUseStreamWaitValue, device_id) != 0;

    // The signal slot is allocated lazily by ensureSignalSlot() on first gated use, not here.
}

void
Stream::ensureSignalSlot()
{
    // The caller holds the lock. Allocates once; never for a default or non-wait-value stream.
    if (signal_slot || !can_use_stream_wait_value) {
        return;
    }
    DeviceGuard guard{device_id};
    SignalSlot  slot{static_cast<uint64_t *>(
                        Context<Hip>::get()->hipExtMallocWithFlags(sizeof(uint64_t), hipMallocSignalMemory)),
                    SignalSlotDeleter{device_id}};
    *slot       = 0;
    signal_slot = std::move(slot);
}

void
Stream::setFlags(uint32_t flags)
{
    if ((flags & HIPFILE_STREAM_FLAGS_MASK) != flags) {
        throw std::invalid_argument("Invalid flags for stream");
    }
    fixed_buf_offset  = (flags & HIPFILE_STREAM_FIXED_BUF_OFFSET) != 0;
    fixed_file_offset = (flags & HIPFILE_STREAM_FIXED_FILE_OFFSET) != 0;
    fixed_io_size     = (flags & HIPFILE_STREAM_FIXED_FILE_SIZE) != 0;
    page_aligned      = (flags & HIPFILE_STREAM_PAGE_ALIGNED_INPUTS) != 0;
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
std::unique_lock<std::mutex>
Stream::getLock()
{
    return std::unique_lock<std::mutex>{mutex};
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
bool
Stream::isRetired() const
{
    return retired;
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
    uint64_t reserved;
    {
        std::lock_guard<std::mutex> lock{mutex};
        retired  = true;
        reserved = signal_counter;
    }
    Context<Hip>::get()->hipStreamSynchronize(hip_stream);
    if (signal_slot) {
        // The synchronize covers every op whose wait armed. An op whose arm failed still bumps the slot once,
        // but the stream never waited for it, so it may still be running. This only loops in that case.
        std::atomic_ref<uint64_t> slot{*signal_slot};
        while (slot.load(std::memory_order_acquire) < reserved) {
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
    }
    signal_slot.reset();
}

void
StreamMap::registerStream(hipStream_t hip_stream, uint32_t flags)
{
    // The default stream is per-device and always available; registering it is a no-op, matching cuFile.
    if (isDefaultStream(hip_stream)) {
        return;
    }

    if (StreamMap::from_ptr.find(hip_stream) != StreamMap::from_ptr.end()) {
        throw std::invalid_argument("Stream is already registered");
    }

    StreamMap::from_ptr[hip_stream] =
        std::shared_ptr<Stream>(new Stream(hip_stream, flags, /*blocking_only=*/false, PassKey<StreamMap>{}));
}

std::shared_ptr<IStream>
StreamMap::deregisterStream(hipStream_t hip_stream)
{
    if (isDefaultStream(hip_stream)) {
        return nullptr;
    }

    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        throw std::invalid_argument("Stream is not registered");
    }

    auto stream = std::move(itr->second);
    StreamMap::from_ptr.erase(itr);
    return stream;
}

std::shared_ptr<IStream>
StreamMap::makeDefaultStream(hipStream_t hip_stream)
{
    return std::shared_ptr<Stream>(new Stream(hip_stream, 0, /*blocking_only=*/false, PassKey<StreamMap>{}));
}

std::shared_ptr<IStream>
StreamMap::findStream(hipStream_t hip_stream) const
{
    auto itr = StreamMap::from_ptr.find(hip_stream);
    if (StreamMap::from_ptr.end() == itr) {
        return nullptr;
    }

    return itr->second;
}

std::shared_ptr<IStream>
StreamMap::makeTransientStream(hipStream_t hip_stream)
{
    return std::shared_ptr<Stream>(new Stream(hip_stream, 0, /*blocking_only=*/true, PassKey<StreamMap>{}));
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
