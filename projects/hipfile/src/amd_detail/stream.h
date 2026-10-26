/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <unordered_map>

template <typename T> class PassKey;

namespace hipFile {

/// @brief True for the null stream and hipStreamPerThread, which are never registered
bool isDefaultStream(hipStream_t hip_stream);

class IStream {
public:
    virtual ~IStream() = default;

    virtual hipStream_t                  getHipStream() const = 0;
    virtual hipDevice_t                  getHipDevice() const = 0;
    virtual std::unique_lock<std::mutex> getLock()            = 0;

    virtual bool      isManaged() const             = 0;
    virtual bool      canUseStreamWaitValue() const = 0;
    virtual uint64_t *signalSlot() const            = 0;

    // A managed stream's caller must hold getLock() for these; a default stream's never change.
    virtual bool fixedBufferOffset() const = 0;
    virtual bool fixedFileOffset() const   = 0;
    virtual bool fixedIOSize() const       = 0;
    virtual bool pageAligned() const       = 0;

    // The caller must hold getLock() for these.
    virtual bool     isRetired() const     = 0;
    virtual uint64_t nextSignalTarget()    = 0;
    virtual uint64_t releaseSignalTarget() = 0;
    virtual void     addOutstanding()      = 0;

    virtual void finishOutstanding()        = 0; // Lock-free; safe from pool and host-function threads.
    virtual void applyFlags(uint32_t flags) = 0; // Takes the lock.
    virtual void drainAndRelease()          = 0; // Takes the lock; must not be called with it held.
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

    bool      isManaged() const override;
    bool      canUseStreamWaitValue() const override;
    uint64_t *signalSlot() const override;

    bool fixedBufferOffset() const override;
    bool fixedFileOffset() const override;
    bool fixedIOSize() const override;
    bool pageAligned() const override;

    bool     isRetired() const override;
    uint64_t nextSignalTarget() override;
    uint64_t releaseSignalTarget() override;
    void     addOutstanding() override;

    void finishOutstanding() override;
    void applyFlags(uint32_t flags) override;
    void drainAndRelease() override;

    Stream(const hipStream_t hip_stream, uint32_t flags, bool managed, const PassKey<StreamMap> &k);

private:
    Stream(const Stream &)             = delete;
    Stream &operator=(const Stream &)  = delete;
    Stream(const Stream &&)            = delete;
    Stream &operator=(const Stream &&) = delete;

    void setFlags(uint32_t flags);

    hipStream_t           hip_stream;
    hipDevice_t           device_id;
    bool                  managed;
    bool                  can_use_stream_wait_value;
    std::mutex            mutex; // Guards the flags after construction, retired, and signal_counter.
    bool                  fixed_buf_offset{false};
    bool                  fixed_file_offset{false};
    bool                  fixed_io_size{false};
    bool                  page_aligned{false};
    bool                  retired{false};
    uint64_t              signal_counter{0};
    std::atomic<uint64_t> outstanding{0}; // Ops that may still touch signal_slot; drained before it's freed.
    SignalSlot            signal_slot;    // Device signal memory; freed by drainAndRelease or the dtor.
};

class StreamMap {
public:
    ~StreamMap();

    /// @brief Registers a stream to use for async operations
    /// @param hip_stream A HIP stream
    /// @param flags Stream flags
    /// @note Registering an implicitly registered stream applies the flags to later submissions.
    void registerStream(const hipStream_t hip_stream, uint32_t flags);

    /// @brief Deregisters a stream from use
    /// @param hip_stream A HIP stream
    /// @return The removed stream, for the caller to drain after releasing its lock; null for the
    ///         default stream, which is never registered
    std::shared_ptr<IStream> deregisterStream(const hipStream_t hip_stream);

    /// @brief Lookup a stream using the hipStream
    /// @param hip_stream A HIP stream
    /// @return The registered stream, or a new unmanaged stream if it isn't registered
    std::shared_ptr<IStream> getStream(hipStream_t hip_stream);

    /// @brief Lookup a registered stream
    /// @param hip_stream A HIP stream
    /// @return The registered stream, or null if it isn't registered
    std::shared_ptr<IStream> findStream(hipStream_t hip_stream) const;

    /// @brief Build a managed stream for implicit registration without touching any map
    /// @param hip_stream A non-default HIP stream
    static std::shared_ptr<IStream> makeImplicitStream(hipStream_t hip_stream);

    /// @brief Implicitly register a stream built by makeImplicitStream unless one is already registered
    /// @param hip_stream A non-default HIP stream
    /// @param candidate The stream to register
    /// @return The registered stream: the existing one if another thread won, otherwise candidate
    std::shared_ptr<IStream> adoptImplicitStream(hipStream_t                     hip_stream,
                                                 const std::shared_ptr<IStream> &candidate);

private:
    struct Entry {
        std::shared_ptr<IStream> stream;
        bool                     implicit;
    };
    std::unordered_map<hipStream_t, Entry> from_ptr;
};

}
