/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mutex>
#include <optional>

namespace hipFile {

class Configuration {

    mutable bool           m_fastpath_env{};
    mutable bool           m_fastpath_read_exists{};
    mutable bool           m_fastpath_write_exists{};
    mutable bool           m_fallback_env{};
    mutable unsigned int   m_stats_level{};
    mutable bool           m_unsupported_file_systems{};
    mutable std::once_flag m_fastpath_once;
    mutable std::once_flag m_fallback_once;
    mutable std::once_flag m_stats_once;
    mutable std::once_flag m_fs_once;
    std::optional<bool>    m_fastpath_override;
    std::optional<bool>    m_fallback_override;

public:
    virtual ~Configuration() = default;

    /// @brief Checks if the fastpath backend is enabled
    /// @return true if the fastpath backend is enabled, false otherwise
    virtual bool fastpath() const noexcept;

    /// @brief Override fastpath backend enablement.
    ///
    /// If hipAmdFileRead/hipAmdFileWrite are not available fastpath() will
    /// return false even if fastpath(true) is called.
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
};

}
