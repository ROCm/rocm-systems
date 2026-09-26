// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_
#define ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::test {

std::unique_ptr<rocjitsu::Decoder> create_downstream_target_decoder();

inline constexpr IsaTargetDescriptor kDownstreamTargetDescriptor{
    .id = "vendor-downstream-test",
    .decoder_factory = &create_downstream_target_decoder,
};

} // namespace rocjitsu::test

#endif // ROCJITSU_TESTS_FIXTURES_DOWNSTREAM_ISA_TARGET_PROVIDER_H_

#ifdef ROCJITSU_GET_ISA_TARGET_DESCRIPTOR
ROCJITSU_GET_ISA_TARGET_DESCRIPTOR(rocjitsu::test::kDownstreamTargetDescriptor)
#endif
