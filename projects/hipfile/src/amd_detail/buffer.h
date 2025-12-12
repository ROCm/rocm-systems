/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <unordered_map>

template <typename T> class PassKey;

namespace hipFile {

/// @brief Buffer is currently registered
struct BufferAlreadyRegistered : public std::runtime_error {
    BufferAlreadyRegistered() : std::runtime_error("Buffer already registered")
    {
    }
};

/// @brief Buffer is not currently registered
struct BufferNotRegistered : public std::runtime_error {
    BufferNotRegistered() : std::runtime_error("Buffer not registered")
    {
    }
};

/// @brief Memory type is not supported
struct InvalidMemoryType : public std::runtime_error {
    InvalidMemoryType() : std::runtime_error("Invalid memory type")
    {
    }
};

/// @brief Range of memory pointer is invalid
struct InvalidPointerRange : public std::runtime_error {
    InvalidPointerRange() : std::runtime_error("Invalid pointer range")
    {
    }
};

/// @brief Buffer has outstanding operations
struct BufferOperationsOutstanding : public std::runtime_error {
    BufferOperationsOutstanding() : std::runtime_error("Buffer operations outstanding")
    {
    }
};

class IBuffer {
public:
    virtual ~IBuffer() = default;

    virtual void         *getBuffer() const = 0;
    virtual size_t        getLength() const = 0;
    virtual int           getFlags() const  = 0;
    virtual hipMemoryType getType() const   = 0;
    virtual int           getGpuId() const  = 0;
};

class BufferMap;

class Buffer : public IBuffer {

public:
    virtual ~Buffer() override = default;

    // Don't allow copying
    Buffer(const Buffer &)            = delete;
    Buffer &operator=(const Buffer &) = delete;

    // Don't allow moving
    Buffer(Buffer &&)            = delete;
    Buffer &operator=(Buffer &&) = delete;

    virtual void         *getBuffer() const override;
    virtual size_t        getLength() const override;
    virtual int           getFlags() const override;
    virtual hipMemoryType getType() const override;
    virtual int           getGpuId() const override;

    /// @brief Creates a buffer.
    /// @param buf Buffer pointer
    /// @param length Buffer length
    /// @param flags Buffer flags (unused)
    /// @param k  Key class instance (see passkey.h)
    Buffer(const void *buf, size_t length, int flags, const PassKey<BufferMap> &k);

    /// @brief Creates a buffer.
    ///
    /// The length of the buffer is determined by querying the hip runtime
    ///
    /// @param buf Buffer pointer
    /// @param k  Key class instance (see passkey.h)
    Buffer(const void *buf, const PassKey<BufferMap> &k);

private:
    /// @brief Pointer to a hip allocated buffer
    void *buffer;

    /// @brief Buffer length
    size_t length;

    /// @brief Buffer flags (unused?)
    int flags;

    /// @brief Memory type
    hipMemoryType type;

    /// @brief Gpu ID
    int gpu_id;
};

class BufferMap {
public:
    virtual ~BufferMap();

    /// @brief Creates and registers a buffer
    /// @attention A unique_lock on HipFileMutex must be held
    /// @param buf Buffer pointer
    /// @param length Buffer length
    /// @param flags Buffer flags (unused)
    virtual void registerBuffer(const void *buf, size_t length, int flags);

    /// @brief Deregisters and destroys a buffer
    /// @attention A unique_lock on HipFileMutex must be held
    /// @param buf Buffer pointer
    virtual void deregisterBuffer(const void *buf);

    /// @brief Look up a registered buffer using the buffer pointer
    /// @attention A shared_lock on HipFileMutex must be held
    /// @param buf Buffer pointer
    /// @return A registered buffer
    virtual std::shared_ptr<IBuffer> getRegisteredBuffer(const void *buf);

    /// @brief Look up a registered buffer. Returns a temporary unregistered
    ///        buffer if no registered buffer is found.
    /// @attention A shared_lock on HipFileMutex must be held
    /// @param buf Buffer pointer
    /// @return A registered or temporary unregistered buffer
    virtual std::shared_ptr<IBuffer> getBuffer(const void *buf);

    virtual void clear();

private:
    /// Buffer lookup table. Protected by hipfile.cpp's StateMutex
    std::unordered_map<const void *, std::shared_ptr<IBuffer>> from_ptr;
};

}
