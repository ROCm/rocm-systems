// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/model_target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::cdna5 {

std::unique_ptr<rocjitsu::Decoder> create_model_target_decoder() {
  return create_model_target_decoder(kModelGpuTargets.front());
}

std::unique_ptr<rocjitsu::Decoder>
create_model_target_decoder(const IsaGpuTargetDescription &gpu_target) {
  if (gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1250 &&
      gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1251)
    return nullptr;
  return make_isa_decoder<Isa>(nullptr, gpu_target.capabilities.instruction_features);
}

} // namespace rocjitsu::cdna5
