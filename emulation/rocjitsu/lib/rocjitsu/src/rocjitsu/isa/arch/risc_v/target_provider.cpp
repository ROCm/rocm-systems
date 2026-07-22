// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/target_provider.h"

#include "rocjitsu/isa/arch/risc_v/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::risc_v {

IsaTargetRegistryError register_target(IsaTargetRegistry &registry) {
  return add_isa_target<Isa>(registry, target_description);
}

} // namespace rocjitsu::risc_v
