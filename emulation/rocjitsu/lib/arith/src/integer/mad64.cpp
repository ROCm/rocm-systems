// SPDX-License-Identifier: MIT

// 32-bit products added to split-word 64-bit accumulators, with carry output.
// Owns public entry points and their validation/dispatch policy.

#include "integer/mad64.h"
#include "common/internal.h"
#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>

namespace {

// Add 32-bit products to 64-bit accumulators for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Signed>
int run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d, uint32_t *carry,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (int error = rocjitsu::arith::validate(flags, mode & ~RJ_ARITH_ALU_CLAMP, true)) {
    return error;
  }
  if (!exec_mask) {
    *carry = 0;
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    *carry = rocjitsu::arith::mad64_x86_64_v4<Signed>(exec_mask, mode, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    *carry = rocjitsu::arith::mad64_x86_64_v3<Signed>(exec_mask, mode, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint64_t result[32];
  uint32_t output_carry = 0;
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t av = a[0][lane], bv = b[0][lane];
    uint64_t cv = c[0][lane] | (uint64_t(c[1][lane]) << 32), product;
    if constexpr (Signed) {
      int64_t sa = int64_t(av) - (int64_t(av >> 31) << 32),
              sb = int64_t(bv) - (int64_t(bv >> 31) << 32);
      product = uint64_t(sa * sb);
    } else {
      product = uint64_t(av) * bv;
    }
    uint64_t sum = product + cv;
    bool overflow, co;
    if constexpr (Signed) {
      overflow = (((product ^ sum) & (cv ^ sum)) >> 63) != 0;

      // The signed carry result is the extended sum's sign, before any saturation.
      co = bool(sum >> 63) ^ overflow;
      if ((mode & RJ_ARITH_ALU_CLAMP) && overflow) {
        sum = (product >> 63) ? 1ULL << 63 : UINT64_MAX >> 1;
      }
    } else {
      overflow = sum < product;
      co = overflow;
      if ((mode & RJ_ARITH_ALU_CLAMP) && overflow) {
        sum = UINT64_MAX;
      }
    }
    result[lane] = sum;
    output_carry |= uint32_t(co) << lane;
  }
  // Delay both destination-register writes until all source registers have been read.
  for (int reg = 0; reg < 2; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane) & 1) {
        d[reg][lane] = uint32_t(result[lane] >> (32 * reg));
      }
    }
  }
  *carry = output_carry & exec_mask;
  return RJ_ARITH_SUCCESS;
}

// Add 32-bit products to 64-bit accumulators for Wave64. Reject invalid controls before
// changing outputs.
template <bool Signed>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint32_t mode, uint32_t *const *d,
               uint64_t *carry, const uint32_t *const *a, const uint32_t *const *b,
               const uint32_t *const *c) {
  uint32_t output[2] = {};

  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Signed>(flags, 0, mode, nullptr, output, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (exec_mask) {
    for (int half = 0; half < 2; ++half) {
      uint32_t *dest[] = {d[0] + half * 32, d[1] + half * 32};
      const uint32_t *source_a = a[0] + half * 32, *source_b = b[0] + half * 32;
      const uint32_t *source_c[] = {c[0] + half * 32, c[1] + half * 32};
      if (int error = run<Signed>(flags, uint32_t(exec_mask >> (half * 32)), mode, dest,
                                  output + half, &source_a, &source_b, source_c)) {
        return error;
      }
    }
  }
  // Scalar carry storage may overlap a source VGPR, so commit it last.
  uint64_t result = output[0] | (uint64_t(output[1]) << 32);

  // The mask may start at an unaligned byte within aliased VGPR storage.
  // Erase the pointer alignment before the byte copy (also for UBSan).
  std::memcpy(static_cast<void *>(carry), &result, sizeof result);
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_co_u64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<false>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_co_i64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run<true>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_mad_co_u64_u32_w32(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_mad_co_i64_i32_w32(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_co_u64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_co_i64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (instruction_flags >> 32) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_u64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_mad_co_u64_u32_w64(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}

// Multiply-add integer operands for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mad_i64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return rj_arith_v_mad_co_i64_i32_w64(flags, exec_mask, instruction_flags, d, carry, a, b, c);
}
