// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/virtual_machine.h"

#include <memory>

namespace rocjitsu {

VirtualMachine::VirtualMachine(const Config &config)
    : simdojo::CompositeComponent("vm"), config_(config) {
  set_weight(0); // Structural container, not a work-producing component.
  auto soc = std::make_unique<SoC>("gpu_soc", config.soc);
  soc_ = soc.get();
  add_child(std::move(soc));
}

} // namespace rocjitsu
