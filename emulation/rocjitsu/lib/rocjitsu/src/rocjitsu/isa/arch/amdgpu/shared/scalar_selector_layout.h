// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_SELECTOR_LAYOUT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_SELECTOR_LAYOUT_H_

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

[[nodiscard]] constexpr bool selector_in_pair(int selector, int pair_base) {
  return pair_base >= 0 && selector >= pair_base && selector <= pair_base + 1;
}

[[nodiscard]] constexpr bool is_ordinary_sgpr_selector(rj_code_arch_t arch, int selector) {
  const auto properties = isa_properties(arch);
  if (selector < 0 || selector > properties.scalar_sgpr_max_selector)
    return false;
  return !selector_in_pair(selector, properties.scalar_flat_scratch_base_selector) &&
         !selector_in_pair(selector, properties.scalar_xnack_mask_base_selector);
}

[[nodiscard]] constexpr bool is_ordinary_sgpr_selector_range(rj_code_arch_t arch, int selector,
                                                             uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    if (!is_ordinary_sgpr_selector(arch, selector + static_cast<int>(i)))
      return false;
  return true;
}

[[nodiscard]] constexpr bool is_null_scalar_selector(rj_code_arch_t arch, int selector) {
  const int null_selector = isa_properties(arch).scalar_null_selector;
  return null_selector >= 0 && selector == null_selector;
}

[[nodiscard]] constexpr bool is_xnack_scalar_selector(rj_code_arch_t arch, int selector) {
  return selector_in_pair(selector, isa_properties(arch).scalar_xnack_mask_base_selector);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SCALAR_SELECTOR_LAYOUT_H_
