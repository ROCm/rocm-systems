// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_queue_binding_factory.h
/// @brief Queue binding factory for the compute command processor.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_types.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Create a reusable binding factory for compute queues owned by @p command_processor.
/// @details The factory and bindings borrow the command processor. Its owner must
/// outlive the factory and every queue binding created from it. Packet callbacks
/// are copied into each CP-owned PM4 queue and must obey their own captured
/// lifetime contracts. The factory is a lifetime/notification adapter; CP owns
/// ring, packet, retry, and cursor-publication state so semantics do not migrate
/// into MES or a PCI/VFIO transport adapter.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_compute_queue_binding_factory(CommandProcessor &command_processor,
                                   Pm4PacketCallbacks callbacks = {});

} // namespace rocjitsu::amdgpu
