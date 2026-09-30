// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_queue_binding_factory.h
/// @brief Queue binding factory for the compute command processor.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Create a reusable binding factory for compute queues owned by @p command_processor.
/// @details The factory and bindings borrow the command processor. Its owner must
/// outlive the factory and every queue binding created from it.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_compute_queue_binding_factory(CommandProcessor &command_processor,
                                   Pm4PacketCallbacks callbacks = {});

} // namespace rocjitsu::amdgpu
