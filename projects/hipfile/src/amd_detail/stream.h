/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

template <typename T> class PassKey;

namespace hipFile {

/// @brief True for the null stream and hipStreamPerThread, which are never registered
bool isDefaultStream(hipStream_t hip_stream);

/// @brief The stream is at the device's greatest priority, which hipFile reserves for its fallback IO streams
struct ReservedStreamPriority : public std::invalid_argument {
    ReservedStreamPriority() : std::invalid_argument("Stream priority is reserved for fallback IO")
    {
    }
};

class IStream {
public:
    virtual ~IStream() = default;

    virtual hipStream_t                  getHipStream() const = 0;
    virtual hipDevice_t                  getHipDevice() const = 0;
    virtual std::unique_lock<std::mutex> getLock()            = 0;

    virtual bool      canUseStreamWaitValue() const = 0;
    virtual uint64_t *signalSlot() const            = 0;

    // The caller must hold getLock() for these, except on a default stream, whose flags never change.
    virtual bool fixedBufferOffset() const = 0;
    virtual bool fixedFileOffset() const   = 0;
    virtual bool fixedIOSize() const       = 0;
    virtual bool pageAligned() const       = 0;

    // The caller must hold getLock() for these.
    virtual bool     isRetired() const     = 0;
    virtual uint64_t nextSignalTarget()    = 0;
    virtual uint64_t releaseSignalTarget() = 0;
    virtual void     ensureSignalSlot()    = 0; // Allocates the signal slot once, on first gated use.

    virtual void drainAndRelease() = 0; // Takes the lock; must not be called with it held.
};

struct SignalSlotDeleter {
    int  device_id{0};
    void operator()(uint64_t *slot) const noexcept;
};
using SignalSlot = std::unique_ptr<uint64_t, SignalSlotDeleter>;

class StreamMap;

class Stream : public IStream {
public:
    virtual ~Stream() override = default;

    hipStream_t                  getHipStream() const override;
    hipDevice_t                  getHipDevice() const override;
    std::unique_lock<std::mutex> getLock() override;

    bool      canUseStreamWaitValue() const override;
    uint64_t *signalSlot() const override;

    bool fixedBufferOffset() const override;
    bool fixedFileOffset() const override;
    bool fixedIOSize() const override;
    bool pageAligned() const override;

    bool     isRetired() const override;
    uint64_t nextSignalTarget() override;
    uint64_t releaseSignalTarget() override;
    void     ensureSignalSlot() override;

    void drainAndRelease() override;

    // blocking_only forces the blocking offload path (no wait-value gating, no signal slot), used for
    // transient streams that are never registered.
    Stream(const hipStream_t hip_stream, uint32_t flags, bool blocking_only, const PassKey<StreamMap> &k);

private:
    Stream(const Stream &)             = delete;
    Stream &operator=(const Stream &)  = delete;
    Stream(const Stream &&)            = delete;
    Stream &operator=(const Stream &&) = delete;

    void setFlags(uint32_t flags);

    hipStream_t hip_stream;
    hipDevice_t device_id;
    bool        can_use_stream_wait_value;
    std::mutex  mutex; // Guards the flags after construction, retired, and signal_counter.
    bool        fixed_buf_offset{false};
    bool        fixed_file_offset{false};
    bool        fixed_io_size{false};
    bool        page_aligned{false};
    bool        retired{false};
    uint64_t    signal_counter{0}; // Targets reserved; each is bumped into signal_slot exactly once.
    SignalSlot  signal_slot;       // Device signal memory; freed by drainAndRelease or the dtor.
};

class StreamMap {
public:
    ~StreamMap();

    /// @brief Registers a stream to use for async operations
    /// @param hip_stream A HIP stream
    /// @param flags Stream flags
    void registerStream(const hipStream_t hip_stream, uint32_t flags);

    /// @brief Deregisters a stream from use
    /// @param hip_stream A HIP stream
    /// @return The removed stream, for the caller to drain after releasing its lock; null for the
    ///         default stream, which is never registered
    std::shared_ptr<IStream> deregisterStream(const hipStream_t hip_stream);

    /// @brief Build the stream object for a default stream, which is never registered
    /// @param hip_stream The null stream or hipStreamPerThread
    static std::shared_ptr<IStream> makeDefaultStream(hipStream_t hip_stream);

    /// @brief Lookup a registered stream
    /// @param hip_stream A HIP stream
    /// @return The registered stream, or null if it isn't registered
    std::shared_ptr<IStream> findStream(hipStream_t hip_stream) const;

    /// @brief Build a transient, unregistered stream for a single async submission on an unregistered
    ///        stream, without touching any map
    /// @param hip_stream A non-default HIP stream
    /// @note The stream uses the blocking offload path, so nothing persistent is keyed to the handle;
    ///       a later stream that reuses the handle value cannot pick up stale state.
    static std::shared_ptr<IStream> makeTransientStream(hipStream_t hip_stream);

private:
    std::unordered_map<hipStream_t, std::shared_ptr<IStream>> from_ptr;
};

}
