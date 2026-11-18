// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::cdna5 {

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() {
  return create_target_decoder(kGfx1250Target);
}

std::unique_ptr<rocjitsu::Decoder>
create_target_decoder(const IsaGpuTargetDescription &gpu_target) {
  if (gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1250 &&
      gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1251)
    return nullptr;
  const IsaExecutionBackend *backend =
      gpu_target.capabilities.execution_implemented ? &execution_backend() : nullptr;
  return make_isa_decoder<Isa>(backend, gpu_target.capabilities.instruction_features);
}

} // namespace rocjitsu::cdna5
