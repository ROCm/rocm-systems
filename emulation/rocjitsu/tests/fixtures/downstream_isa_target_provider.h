// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_
#define ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::test {

inline constexpr IsaTargetDescription downstream_target_description{
    .id = "vendor-downstream-test",
};

IsaTargetRegistryError register_downstream_target(IsaTargetRegistry &registry);

} // namespace rocjitsu::test

#endif // ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_REGISTRATION
ROCJITSU_GET_ISA_TARGET_REGISTRATION(rocjitsu::test::register_downstream_target)
#endif
