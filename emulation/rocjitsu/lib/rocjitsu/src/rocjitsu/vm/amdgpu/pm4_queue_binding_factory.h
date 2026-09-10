// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4_queue_binding_factory.h
/// @brief Registry binding for synchronous PM4 compute queues.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pm4_packet_processor.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Create a reusable binding factory for PM4 queues assigned to one CP.
/// @details The factory and bindings borrow the command processor. Its owner must
/// outlive the factory and every queue binding created from it. Packet callbacks
/// are copied into each CP-owned PM4 queue and must obey their own captured
/// lifetime contracts.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_pm4_queue_binding_factory(CommandProcessor &command_processor,
                               Pm4PacketCallbacks packet_callbacks);

} // namespace rocjitsu::amdgpu
