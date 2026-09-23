// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/target_provider.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/target_provider.h"
#include "util/simd.h"

namespace rocjitsu::cdna5 {

#if defined(RJ_HAS_AVX512_EXECUTION)
extern "C" const IsaExecutionBackend *rj_cdna5_avx512_backend();
extern "C" bool rj_cdna5_force_scalar() { return util::force_scalar(); }
#endif

const IsaExecutionBackend &host_execution_backend() {
#if defined(RJ_HAS_AVX512_EXECUTION)
  // This translation unit is compiled for the baseline CPU.  GCC's runtime
  // feature check includes OS-enabled vector state; none of the AVX-512 code
  // is entered unless the machine running this process supports every flag
  // used by the separately compiled execution variant.
  if (!util::force_scalar() && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
      __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
    return *rj_cdna5_avx512_backend();
#endif
  return execution_backend();
}

std::unique_ptr<rocjitsu::Decoder> create_target_decoder() {
  return create_target_decoder(kGfx1250Target);
}

std::unique_ptr<rocjitsu::Decoder>
create_target_decoder(const IsaGpuTargetDescription &gpu_target) {
  if (gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1250 &&
      gpu_target.public_id != ROCJITSU_CODE_TARGET_GFX1251)
    return nullptr;
  const IsaExecutionBackend *backend =
      gpu_target.capabilities.execution_implemented ? &host_execution_backend() : nullptr;
  return make_isa_decoder<Isa>(backend, gpu_target.capabilities.instruction_features);
}

} // namespace rocjitsu::cdna5
