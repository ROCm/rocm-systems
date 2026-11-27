/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <optional>

namespace hipFile {

class Configuration {

    // Atomic since the fastpath can be disabled from one thread while other
    // threads read this flag. This does not prevent stale reads, but it does
    // prevent a torn read while the flag is being written.
    std::atomic<std::optional<bool>> m_fastpath_override;
    std::optional<bool>              m_fallback_override;
    std::optional<bool>              m_host_override;
    std::optional<bool>              m_force_blocking_async_override;

public:
    virtual ~Configuration() = default;

    /// @brief Checks if the fastpath backend is enabled
    /// @return true if the fastpath backend is enabled, false otherwise
    virtual bool fastpath() const noexcept;

    /// @brief Override fastpath backend enablement.
    ///
    /// If hipAmdFileRead/hipAmdFileWrite are not available, or ROCr is running
    /// on WSL2 (/dev/dxg), fastpath() will return false even if fastpath(true)
    /// is called.
    virtual void fastpath(bool enabled) noexcept;

    /// @brief Checks if the fallback backend is enabled
    /// @return true if the fallback backend is enabled, false otherwise
    virtual bool fallback() const noexcept;

    /// @brief Override fallback backend enablement
    virtual void fallback(bool enabled) noexcept;

    /// @brief Shows the level of detail for stats collection
    /// @return 0 if stats collection disabled, higher levels of detail as value increases
    virtual unsigned int statsLevel() const noexcept;

    /// @brief Checks if unsupported file systems are allowed in the fastpath backend
    /// @return true if unsupported file systems are allowed, false if only supported file systems are
    /// permitted (default)
    virtual bool unsupportedFileSystems() const noexcept;

    /// @brief Checks if the host backend is enabled
    /// @return true if the host backend is enabled, false otherwise
    virtual bool host() const noexcept;

    /// @brief Override host backend enablement
    virtual void host(bool enabled) noexcept;

    /// @brief Gets the size of the device stream pool
    /// @return the size of the device stream pool in bytes
    virtual size_t deviceStreamPoolSize() const noexcept;

    /// @brief Checks if asynchronous I/O is forced onto the blocking host-function path
    /// @return true if every stream uses the blocking path (default), false if registered streams may
    /// gate on a stream wait value
    virtual bool forceBlockingAsync() const noexcept;

    /// @brief Override whether asynchronous I/O is forced onto the blocking host-function path
    ///
    /// Only affects streams constructed after the call.
    virtual void forceBlockingAsync(bool enabled) noexcept;
};

}
