// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::cdna1 {

IsaTargetRegistryError register_target(IsaTargetRegistry &registry) {
  return add_isa_target<Isa>(registry, target_description);
}

} // namespace rocjitsu::cdna1
