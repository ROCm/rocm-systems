// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"

#if defined(__linux__) && defined(__x86_64__)
// Only the shared simulator defines this hook. Object-composed tools and tests
// keep the generated baseline backend without pulling in a v4 provider.
extern "C" const rocjitsu::IsaExecutionBackend *rj_cdna5_wmma_shared_backend()
    __attribute__((weak, visibility("hidden")));
#endif

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
#if defined(__linux__) && defined(__x86_64__)
  if (backend && rj_cdna5_wmma_shared_backend)
    backend = rj_cdna5_wmma_shared_backend();
#endif
  return make_isa_decoder<Isa>(backend, gpu_target.capabilities.instruction_features);
}

} // namespace rocjitsu::cdna5
