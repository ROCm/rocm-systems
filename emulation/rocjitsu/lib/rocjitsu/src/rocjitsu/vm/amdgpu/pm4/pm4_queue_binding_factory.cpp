// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"

#include <utility>

namespace rocjitsu::amdgpu {
std::shared_ptr<QueueBindingFactory>
make_pm4_queue_binding_factory(CommandProcessor &command_processor, Pm4PacketCallbacks callbacks) {
  return make_compute_queue_binding_factory(command_processor, std::move(callbacks));
}
} // namespace rocjitsu::amdgpu
