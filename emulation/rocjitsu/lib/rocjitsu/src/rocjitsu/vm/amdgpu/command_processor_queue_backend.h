// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file command_processor_queue_backend.h
/// @brief QueueService adapter for the shared command processor.

#pragma once

#include "rocjitsu/vm/amdgpu/queue_service.h"

#include <memory>

namespace rocjitsu::amdgpu {

class CommandProcessor;

/// @brief Adapt an existing command processor to the frontend-neutral queue service.
[[nodiscard]] std::shared_ptr<QueueBackend>
make_command_processor_queue_backend(CommandProcessor &command_processor);

} // namespace rocjitsu::amdgpu
