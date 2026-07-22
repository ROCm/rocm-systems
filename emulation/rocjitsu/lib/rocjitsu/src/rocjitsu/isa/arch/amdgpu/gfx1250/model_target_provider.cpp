// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/gfx1250/model_target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::gfx1250 {

IsaTargetRegistryError register_model_target(IsaTargetRegistry &registry) {
  return add_isa_target<Isa>(registry, model_target_description);
}

} // namespace rocjitsu::gfx1250
