// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/gfx1250/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/gfx1250/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::gfx1250 {

IsaTargetRegistryError register_target(IsaTargetRegistry &registry) {
  return add_isa_target<Isa>(registry, full_target_description, &execution_backend());
}

} // namespace rocjitsu::gfx1250
