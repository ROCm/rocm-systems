// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4/pm4_queue_binding_factory.h
/// @brief Separate registry bindings for MES transport and native KFD PM4 rings.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4_packet_processor.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Create a reusable binding factory for explicit MES PM4 transport queues.
/// @details The factory and bindings borrow the command processor. Its owner must
/// outlive the factory and every queue binding created from it. Packet callbacks
/// are copied into each CP-owned PM4 queue and must obey their own captured
/// lifetime contracts.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_pm4_queue_binding_factory(CommandProcessor &command_processor,
                               Pm4PacketCallbacks packet_callbacks);

/// @brief Bind host-polled native KFD PM4 rings to the full compute executor.
/// @details The CP must outlive all bindings. Fresh rings start at zero and
/// publish a 32-bit wrapped dword cursor. Live ring replacement and terminal
/// fault recovery require destroy/create. Explicit producer submission is not
/// supported: the caller publishes through its mapped host doorbell. The host
/// mapping may be attached after registration through set_doorbell_base, as in
/// tinygrad's create-queue-then-mmap sequence.
[[nodiscard]] std::shared_ptr<QueueBindingFactory>
make_kfd_pm4_compute_queue_binding_factory(CommandProcessor &command_processor);

} // namespace rocjitsu::amdgpu
