// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "downstream_isa_target_provider.h"

#include "downstream_isa_fixture.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::test {

IsaTargetRegistryError register_downstream_target(IsaTargetRegistry &registry) {
  return add_isa_target<DownstreamIsa>(registry, downstream_target_description);
}

} // namespace rocjitsu::test
