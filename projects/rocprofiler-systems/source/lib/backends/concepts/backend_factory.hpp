// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <memory>

namespace rocprofsys::backends::concepts
{

/**
 * @brief Concept that any BackendFactory type must satisfy.
 *
 * Generic contract shared by all backend providers (amd_smi, procfs,
 * rocprofiler_sdk). Each provider may impose additional constraints on
 * the session type it produces.
 *
 * Checked expressions:
 *  - @c F::backend_t        — the session type produced by this factory
 *  - @c F::create_backend() — returns @c shared_ptr<backend_t>
 */
template <typename F>
concept backend_factory = requires { typename F::backend_t; } && requires {
    { F::create_backend() } -> std::same_as<std::shared_ptr<typename F::backend_t>>;
};

}  // namespace rocprofsys::backends::concepts
