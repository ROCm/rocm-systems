// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file aql_queue_binding_factory.h
/// @brief Queue binding factory for the shared AQL command processor.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Create a reusable binding factory for AQL queues owned by @p command_processor.
/// @details The factory and bindings borrow the command processor. Its owner must
/// outlive the factory and every queue binding created from it.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_aql_queue_binding_factory(CommandProcessor &command_processor);

} // namespace rocjitsu::amdgpu
