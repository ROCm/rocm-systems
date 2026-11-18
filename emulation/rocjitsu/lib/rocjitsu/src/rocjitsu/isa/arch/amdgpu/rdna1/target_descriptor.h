// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/target_registry.h"

namespace rocjitsu::rdna1 {

constexpr IsaTargetDescriptor
make_target_descriptor(bool supports_execution,
                       IsaTargetDescriptor::DecoderFactory decoder_factory) {
  return {
      .id = "rdna1",
      .architecture_id = ROCJITSU_CODE_ARCH_RDNA1,
      .decoder_factory = decoder_factory,
      .supports_execution = supports_execution,
  };
}

} // namespace rocjitsu::rdna1
