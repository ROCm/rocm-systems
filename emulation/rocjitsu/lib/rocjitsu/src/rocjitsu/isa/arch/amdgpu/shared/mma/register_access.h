// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_REGISTER_ACCESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_REGISTER_ACCESS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/layout.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/log.h"
#include "util/meta_programming.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <optional>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {

/// Resolve the accumulator source (src2) for MFMA instructions.
///
/// src2 uses OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST which can be a VGPR (ev 256-511),
/// an ACCVGPR (ev 768-1023), or an inline constant (ev 0-255, e.g. literal 0
/// for zero-initializing the accumulator).
///
/// For VGPR/ACCVGPR operands, sets const_acc to ACC_FROM_VGPR and returns the
/// physical VGPR base. For constants, sets const_acc to the constant value and
/// returns dst (unused by exec functions in the constant case).
///
/// @tparam Mode AccMode for the current ISA generation.
/// @param const_acc Output: the constant value, or ACC_FROM_VGPR if src2 is a register.
/// @param get_const Lazy callback returning the 32-bit constant value; only
///        called when src2 is not a VGPR. Typically: [&]{ return
///        amdgpu::RegisterAccess(wf).read_scalar(src2); }
template <AccMode Mode = AccMode::Unified, typename F>
uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc, F &&get_const) {
  if constexpr (Mode == AccMode::Unified || Mode == AccMode::Separate) {
    if (src2_ev >= 768 && src2_ev <= 1023) {
      const_acc = ACC_FROM_VGPR;
      return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(src2_ev - 768);
    }
    if (src2_ev >= 256 && src2_ev <= 511) {
      const_acc = ACC_FROM_VGPR;
      // When MFMA writes to AccVGPR bank (acc_cd=1, dst >= vb+256),
      // the accumulator source at encoding 256-511 also refers to
      // AccVGPRs to maintain consistency. This matches gfx942 behavior
      // where v_accvgpr_write initializes the AccVGPR bank.
      if (dst >= vb + ACC_VGPR_OFFSET) {
        util::Logger::vm([&](auto &os) {
          os << std::format("MFMA resolve_acc: acc_cd path, dst={} vb={} src2_ev={} → acc_base={}",
                            dst, vb, src2_ev, vb + ACC_VGPR_OFFSET + (src2_ev - 256));
        });
        return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(src2_ev - 256);
      }
      return vb + static_cast<uint32_t>(src2_ev - 256);
    }
    const_acc = get_const();
    return dst;
  } else if constexpr (Mode == AccMode::VgprOnly) {
    if (src2_ev >= 256 && src2_ev <= 511) {
      const_acc = ACC_FROM_VGPR;
      return vb + static_cast<uint32_t>(src2_ev - 256);
    }
    const_acc = get_const();
    return dst;
  } else {
    static_assert(util::always_false_v<F>, "unhandled AccMode");
  }
}

/// Apply GPR_IDX only when an MMA operand resolves to the architectural VGPR
/// bank. AccVGPRs occupy the unified storage range above ACC_VGPR_OFFSET but
/// are not selected by MODE.GPR_IDX_EN.
inline uint32_t apply_gpr_idx_to_mma_base(const Wavefront &wf, uint32_t vb, uint32_t base,
                                          VgprMsbRole role) {
  if (base >= vb + ACC_VGPR_OFFSET)
    return base;
  return vb + apply_gpr_idx(wf, base - vb, role);
}

inline void require_wmma_wave32(const auto &cu) {
  if (cu.wf_size() != WMMA_WAVE32)
    throw util::ConfigError("gfx1250 WMMA requires wave32");
}

inline uint64_t read_swmmac_index_set(auto &cu, uint32_t index_base, uint32_t lane,
                                      uint32_t index_entries, uint32_t index_key) {
  // gfx1250 FMT_WMMA_INDEX_SET and FMT_WMMA_INDEX_SET2 contain 16 and 32
  // packed 2-bit entries respectively. LLVM names the matching SWMMAC index
  // selector by entry count, not by total bit count.
  switch (index_entries) {
  case 8:
    return (RegisterAccess(cu).read_vgpr(index_base, lane) >> (8 * (index_key & 0x3u))) & 0xFFu;
  case 16:
    if (index_key & 0x1u)
      return (RegisterAccess(cu).read_vgpr(index_base, lane) >> 16) & 0xFFFFu;
    return RegisterAccess(cu).read_vgpr(index_base, lane);
  case 32: {
    if (index_key & 0x1u)
      return RegisterAccess(cu).read_vgpr(index_base + 1, lane);
    uint64_t lo = RegisterAccess(cu).read_vgpr(index_base, lane);
    uint64_t hi = RegisterAccess(cu).read_vgpr(index_base + 1, lane);
    return lo | (hi << 32);
  }
  default:
    throw util::UnimplementedInst("unsupported SWMMAC sparse index width");
  }
}

inline uint32_t matrix_vgpr_word(const RegisterAccess::VgprReadRegion &region,
                                 uint32_t physical_reg, uint32_t lane) {
  assert(physical_reg >= region.base() && physical_reg - region.base() < region.reg_count() &&
         "matrix read exceeds acquired VGPR region");
  return region.lane(physical_reg - region.base(), lane);
}

inline uint32_t matrix_vgpr_word(RegisterAccess::VgprReadRegion &region, uint32_t physical_reg,
                                 uint32_t lane) {
  return matrix_vgpr_word(std::as_const(region), physical_reg, lane);
}

inline uint32_t matrix_vgpr_word(auto &cu, uint32_t physical_reg, uint32_t lane) {
  return RegisterAccess(cu).read_vgpr(physical_reg, lane);
}

inline uint32_t read_packed(auto &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane) >> loc.bit_offset;
  if (loc.bit_offset + loc.data_bits > 32) {
    uint32_t next = matrix_vgpr_word(cu, base + loc.vgpr_offset + 1, loc.lane);
    raw |= next << (32 - loc.bit_offset);
  }
  return raw & packed_mask(loc.data_bits);
}

// Matrix fast-path register access helpers.
//
// MFMA/WMMA specializations below read dense source regions through
// RegisterAccess views. Acquiring a read view observes the whole physical VGPR
// region. Logical traversal and copying remain independent of the backing-store
// layout. Keep the matrix-specific size calculations here so individual
// specializations do not open-code plugin notification or raw region sizing.
//
// This is still only the physical-register-region layer. The broader cleanup is
// to move operand-based SIMD paths onto the same facade and then make direct raw
// VGPR storage unavailable to instruction emulators.

inline void observe_contiguous_vgpr_reads(auto &cu, uint32_t base, uint32_t reg_count,
                                          uint32_t wf_size, uint8_t byte_mask = 0xF) {
  if (reg_count == 0)
    return;
  (void)RegisterAccess(cu).read_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size),
                                            byte_mask);
}

inline void observe_mfma_input_reads(auto &cu, uint32_t base, uint32_t dim, uint32_t K, uint32_t B,
                                     uint32_t data_bits, uint32_t wf_size) {
  uint64_t element_count = static_cast<uint64_t>(dim) * K * B;
  observe_contiguous_vgpr_reads(cu, base, mfma_dense_reg_count(element_count, data_bits, wf_size),
                                wf_size);
}

inline void observe_mfma_acc32_reads(auto &cu, uint32_t base, uint32_t M, uint32_t N, uint32_t B,
                                     uint32_t wf_size) {
  uint64_t element_count = static_cast<uint64_t>(M) * N * B;
  observe_contiguous_vgpr_reads(cu, base, mfma_dense_reg_count(element_count, 32, wf_size),
                                wf_size);
}

inline RegisterAccess::VgprWriteRegion write_mfma_acc32_region(auto &cu, uint32_t base, uint32_t M,
                                                               uint32_t N, uint32_t B,
                                                               uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t element_count = static_cast<uint64_t>(M) * N * B;
  uint32_t reg_count = mfma_dense_reg_count(element_count, 32, wf_size);
  return regs.write_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size));
}

template <typename Destination, typename Convert>
void convert_packed_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                  Destination *destination, size_t element_count,
                                  uint32_t elements_per_word, Convert convert) {
  const size_t elements_per_reg = static_cast<size_t>(source.wf_size()) * elements_per_word;
  size_t converted = 0;
  source.for_each([&](std::span<const uint32_t> lanes) {
    if (converted == element_count)
      return;
    const size_t count = std::min(elements_per_reg, element_count - converted);
    convert(lanes.data(), destination + converted, count);
    converted += count;
  });
  assert(converted == element_count && "packed conversion exceeds matrix read region");
}

template <size_t Capacity>
void copy_matrix_region_words(const RegisterAccess::VgprReadRegion &source,
                              uint32_t (&destination)[Capacity]) {
  const size_t word_count = static_cast<size_t>(source.reg_count()) * source.wf_size();
  assert(word_count <= Capacity && "matrix read region exceeds snapshot capacity");
  if (word_count > Capacity)
    std::abort();
  source.copy_to({destination, word_count});
}

inline void convert_f16_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                      float *destination, size_t element_count) {
  convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/2,
                               [](const uint32_t *words, float *out, size_t count) {
                                 util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(words),
                                                        out, count);
                               });
}

inline void convert_bf16_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                       float *destination, size_t element_count) {
  convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/2,
                               [](const uint32_t *words, float *out, size_t count) {
                                 util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(words),
                                                         out, count);
                               });
}

inline void convert_i8_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                     int32_t *destination, size_t element_count) {
  convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/4,
                               [](const uint32_t *words, int32_t *out, size_t count) {
                                 util::i8_to_i32_block(reinterpret_cast<const int8_t *>(words), out,
                                                       count);
                               });
}

inline void convert_u8_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                     int32_t *destination, size_t element_count) {
  convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/4,
                               [](const uint32_t *words, int32_t *out, size_t count) {
                                 util::u8_to_i32_block(reinterpret_cast<const uint8_t *>(words),
                                                       out, count);
                               });
}

class MatrixReadRegions {
public:
  explicit MatrixReadRegions(RegisterAccess::VgprReadRegion a, RegisterAccess::VgprReadRegion b,
                             std::optional<RegisterAccess::VgprReadRegion> acc)
      : a(std::move(a)), b(std::move(b)), acc(std::move(acc)) {}

  RegisterAccess::VgprReadRegion a;
  RegisterAccess::VgprReadRegion b;
  std::optional<RegisterAccess::VgprReadRegion> acc;
};

inline MatrixReadRegions read_mixed_matrix_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1,
                                                             uint32_t s2, uint64_t a_elements,
                                                             uint32_t a_bits, uint64_t b_elements,
                                                             uint32_t b_bits, uint64_t acc_elements,
                                                             uint32_t acc_bits, uint32_t const_acc,
                                                             uint32_t wf_size) {
  RegisterAccess regs(cu);
  const uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  const uint32_t a_regs = mfma_dense_reg_count(a_elements, a_bits, wf_size);
  const uint32_t b_regs = mfma_dense_reg_count(b_elements, b_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (const_acc == ACC_FROM_VGPR) {
    const uint32_t acc_regs = mfma_dense_reg_count(acc_elements, acc_bits, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

inline MatrixReadRegions read_mfma_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1,
                                                     uint32_t s2, uint32_t M, uint32_t N,
                                                     uint32_t K, uint32_t B, uint32_t data_bits,
                                                     uint32_t const_acc, uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  uint32_t a_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * K * B, data_bits, wf_size);
  uint32_t b_regs = mfma_dense_reg_count(static_cast<uint64_t>(N) * K * B, data_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (const_acc == ACC_FROM_VGPR) {
    uint32_t acc_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * N * B, 32, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

inline void observe_mfma_fast_path_reads(auto &cu, uint32_t s0, uint32_t s1, uint32_t s2,
                                         uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                                         uint32_t data_bits, uint32_t const_acc, uint32_t wf_size) {
  (void)read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, data_bits, const_acc, wf_size);
}

inline MatrixReadRegions read_wmma_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1,
                                                     uint32_t s2, uint32_t M, uint32_t N,
                                                     uint32_t K, uint32_t data_bits,
                                                     uint32_t acc_bits, uint32_t const_acc,
                                                     uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  uint32_t a_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * K, data_bits, wf_size);
  uint32_t b_regs = mfma_dense_reg_count(static_cast<uint64_t>(N) * K, data_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (const_acc == ACC_FROM_VGPR) {
    uint32_t acc_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * N, acc_bits, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

inline void observe_wmma_fast_path_reads(auto &cu, uint32_t s0, uint32_t s1, uint32_t s2,
                                         uint32_t M, uint32_t N, uint32_t K, uint32_t data_bits,
                                         uint32_t acc_bits, uint32_t const_acc, uint32_t wf_size) {
  (void)read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, data_bits, acc_bits, const_acc,
                                    wf_size);
}

inline RegisterAccess::VgprWriteRegion write_wmma_output_region(auto &cu, uint32_t base, uint32_t M,
                                                                uint32_t N, uint32_t output_bits,
                                                                uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint32_t reg_count = mfma_dense_reg_count(static_cast<uint64_t>(M) * N, output_bits, wf_size);
  return regs.write_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size));
}

inline RegisterAccess::VgprReadWriteRegion readwrite_wmma_output_region(auto &cu, uint32_t base,
                                                                        uint32_t M, uint32_t N,
                                                                        uint32_t output_bits,
                                                                        uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint32_t reg_count = mfma_dense_reg_count(static_cast<uint64_t>(M) * N, output_bits, wf_size);
  return regs.readwrite_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size));
}

/// Compile-time fp8 (e4m3) vs bf8 (e5m2), OCP vs FNUZ bulk-convert selector for
/// the f8 spec kernels: converts `n` packed bytes starting at `words` to f32
/// through 256-entry LUTs matching the selected extractor.
template <bool FP8, bool FNUZ = false>
void f8_to_f32_block(const uint32_t *words, float *dst, size_t n) {
  if constexpr (FP8 && FNUZ)
    util::fp8_e4m3_fnuz_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else if constexpr (FP8)
    util::fp8_e4m3_ocp_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else if constexpr (FNUZ)
    util::bf8_e5m2_fnuz_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else
    util::bf8_e5m2_ocp_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
}

template <bool FP8, bool FNUZ = false>
void convert_f8_matrix_region(const RegisterAccess::VgprReadRegion &source, float *destination,
                              size_t element_count) {
  convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/4,
                               [](const uint32_t *words, float *out, size_t count) {
                                 f8_to_f32_block<FP8, FNUZ>(words, out, count);
                               });
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_REGISTER_ACCESS_H_
