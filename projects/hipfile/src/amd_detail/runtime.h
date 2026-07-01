/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "backend.h"
#include "state.h"

#include <memory>
#include <mutex>
#include <vector>

namespace hipFile {

/// @brief The single process-global root object.
///
/// `Runtime` owns the one `DriverState` and the one backend collection for the
/// process. It is the *only* global touched by the C-API boundary shims in
/// `hipfile.cpp`; everything below the shim receives its dependencies
/// (`DriverState&`, backends) as explicit parameters. This keeps the global at
/// the boundary and lets tests construct a fresh `DriverState` locally.
class Runtime {
public:
    /// @brief Access the process-global runtime
    /// @return A reference to the single Runtime instance
    static Runtime &instance();

    /// @brief The process-global driver state
    /// @return A reference to the owned DriverState
    DriverState &state()
    {
        return state_;
    }

    /// @brief The backends that can service IO requests
    ///
    /// Builds the Fallback/Fastpath pair exactly once (via std::call_once) and
    /// returns the cached collection. This is the single home for backend
    /// construction.
    ///
    /// @return A collection of backends that can service IO requests
    const std::vector<std::shared_ptr<Backend>> &backends() const;

    // Don't allow copying
    Runtime(const Runtime &)            = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Don't allow moving
    Runtime(Runtime &&)            = delete;
    Runtime &operator=(Runtime &&) = delete;

private:
    Runtime();
    ~Runtime();

    DriverState state_;

    mutable std::vector<std::shared_ptr<Backend>> backends_;
    mutable std::once_flag                        backends_once_;
};

}
