// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instruction_encoding.h
/// @brief AMDGPU encoding and register-access planning helpers.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_

#include "rocjitsu/base/api.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu {
class Operand;
namespace amdgpu {

/// Add an M0-relative VGPR offset after resolving the encoded register bank.
inline std::optional<uint32_t> relative_vgpr_index(std::optional<uint32_t> base, uint32_t offset,
                                                   uint32_t width, uint32_t allocation_size,
                                                   rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5) {
  const uint32_t max_offset = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 1023 : 255;
  if (!base || offset > max_offset || uint64_t{*base} + offset + width > allocation_size)
    return std::nullopt;
  return *base + offset;
}

/// GFX11.5 VALU can broadcast at most two different scalar values. Inline
/// constants and VGPRs use no scalar port. A repeated literal of another
/// width is a different broadcast; SGPR tuples at the same base share one.
constexpr bool gfx115_scalar_sources_fit(uint32_t src0_selector, unsigned src0_bits,
                                         uint32_t src1_selector, unsigned src1_bits,
                                         uint32_t src2_selector, unsigned src2_bits,
                                         bool implicit_vcc = false) {
  auto bus_key = [implicit_vcc](uint32_t selector, unsigned bits) -> int {
    // FMAS already broadcasts the full VCC pair. VCC_LO alone is a distinct value.
    if (implicit_vcc && selector == 106 && bits == 64)
      return -1;
    if (selector == 255)
      return static_cast<int>(selector + bits);
    if ((selector < 128 && selector != 124) || selector == 253)
      return static_cast<int>(selector);
    return -1;
  };
  const int src0_key = bus_key(src0_selector, src0_bits),
            src1_key = bus_key(src1_selector, src1_bits),
            src2_key = bus_key(src2_selector, src2_bits);
  if (implicit_vcc)
    return (src0_key < 0 || src1_key < 0 || src0_key == src1_key) &&
           (src0_key < 0 || src2_key < 0 || src0_key == src2_key) &&
           (src1_key < 0 || src2_key < 0 || src1_key == src2_key);
  return src0_key < 0 || src1_key < 0 || src2_key < 0 || src0_key == src1_key ||
         src0_key == src2_key || src1_key == src2_key;
}

/// M0 carries the barrier ID in its low five bits; other operands carry the ID itself.
constexpr int32_t barrier_operand_id(uint32_t source, bool source_is_m0) {
  return static_cast<int32_t>(source_is_m0 ? source & 0x1fu : source);
}

/// Effective lane mask for a wave32 operation with OPF_EXEC_ALL_1 semantics.
constexpr uint64_t wave32_exec_all_if_nonzero(uint64_t exec) {
  return exec ? uint64_t{0xffffffff} : 0;
}

constexpr uint64_t whole_active_quads(uint64_t exec) {
  exec |= exec >> 1;
  exec |= exec >> 2;
  return (exec & 0x1111111111111111ull) * 15;
}

constexpr bool valid_lds_direct_operand(uint32_t m0) {
  const unsigned type = (m0 >> 16) & 7;
  return !(m0 & 3) && type != 3 && type < 6;
}

enum class DsPermutation : uint8_t { None, Permute, Bpermute, BpermuteFi, Swizzle };

constexpr unsigned ds_permute_group_width(rj_code_arch_t arch, unsigned wave_size) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
      arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return 32;
  return wave_size;
}

constexpr unsigned ds_permute_lane(unsigned lane, uint32_t address, unsigned offset,
                                   unsigned group_width) {
  return (lane / group_width) * group_width + ((address + offset) / 4) % group_width;
}

constexpr unsigned ds_swizzle_lane(unsigned lane, unsigned offset) {
  const unsigned group = lane & ~31u;
  const unsigned index = lane & 31u;
  const unsigned mask = offset & 31u;
  if (offset >= 0xe000) {
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < 5; ++bit)
      reversed |= ((index >> bit) & 1u) << (4u - bit);
    return group | (reversed >> std::popcount(mask)) | (index & mask);
  }
  if (offset >= 0xc000) {
    const unsigned distance = (offset >> 5) & 31u;
    const unsigned rotated = offset & 0x400u ? index - distance : index + distance;
    return group | (index & mask) | (rotated & ~mask & 31u);
  }
  if (offset & 0x8000)
    return (lane & ~3u) | ((offset >> (2 * (lane & 3))) & 3);
  return (lane & ~31u) +
         ((((lane & 31u) & (offset & 31u)) | ((offset >> 5) & 31u)) ^ ((offset >> 10) & 31u));
}

enum class ValuPermutation : uint8_t {
  None,
  Perm16,
  Perm16Var,
  X16,
  X16Var,
  Perm64,
  Swap16,
  Swap32,
  Bcast,
  Down,
  Up,
  Xor
};

constexpr unsigned valu_permutation_source(ValuPermutation kind, unsigned lane, uint32_t selector,
                                           unsigned group_width, unsigned wave_size) {
  if (kind == ValuPermutation::Perm16 || kind == ValuPermutation::Perm16Var)
    return (lane & ~15u) | (selector & 15u);
  if (kind == ValuPermutation::X16 || kind == ValuPermutation::X16Var)
    return ((lane & ~15u) ^ 16u) | (selector & 15u);
  if (kind == ValuPermutation::Perm64)
    return lane ^ 32u;
  if (!group_width || (group_width & (group_width - 1)))
    group_width = wave_size;
  group_width = std::min(group_width, wave_size);
  const unsigned base = (lane / group_width) * group_width;
  const unsigned offset = lane - base;
  if (kind == ValuPermutation::Bcast)
    return base + selector % group_width;
  if (kind == ValuPermutation::Up)
    return selector <= offset ? lane - selector : lane;
  if (kind == ValuPermutation::Down)
    return selector < group_width - offset ? lane + selector : lane;
  // PERMLANE_XOR can read an earlier group. Only its upper group edge is a bound.
  const unsigned source = lane ^ selector;
  return selector < wave_size && source < base + group_width ? source : lane;
}

struct PermutationLanes {
  uint64_t read = 0;
  uint64_t write = 0;
};

// Compute observed source lanes and writable destinations for a lane permutation.
template <typename Select>
PermutationLanes valu_permutation_lanes(ValuPermutation kind, uint64_t exec, unsigned wave_size,
                                        unsigned group_width, bool row16, bool fi, bool bound_ctrl,
                                        Select select) {
  PermutationLanes result;
  for (unsigned lane = 0; lane < wave_size; ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    const unsigned source =
        valu_permutation_source(kind, lane, select(lane), group_width, wave_size);
    if (source >= wave_size)
      continue;
    if (row16 && !fi && !(exec & (uint64_t{1} << source))) {
      if (bound_ctrl)
        result.write |= uint64_t{1} << lane;
      continue;
    }
    result.read |= uint64_t{1} << source;
    result.write |= uint64_t{1} << lane;
  }
  return result;
}

constexpr uint64_t valu_swap_write_lanes(uint64_t exec, unsigned wave_size, unsigned stride,
                                         bool upper) {
  if (stride >= wave_size)
    return 0;
  const uint64_t lower = stride == 16 ? 0x0000ffff0000ffffull : 0x00000000ffffffffull;
  const uint64_t wave = wave_size == 64 ? ~uint64_t{0} : 0xffffffffull;
  return exec & wave & (upper ? lower << stride : lower);
}

constexpr bool has_gfx11_image_operands(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

/// VMEM and EXP substitute v0 independently for each missing source DWORD.
constexpr uint16_t gfx115_source_vgpr(uint32_t selector, uint32_t allocation) {
  return selector < allocation ? static_cast<uint16_t>(selector) : 0;
}

/// @brief Decoded register-access modifiers.
/// @details This view contains no wave state or values.
struct RegisterModifiers {
  DsPermutation ds_permutation = DsPermutation::None;
  ValuPermutation valu_permutation = ValuPermutation::None;
  uint16_t ds_offset = 0;
  uint8_t memory_result_bytes = 0xf;
  uint8_t memory_result_last_bytes = 0xf; // Last data word, excluding optional image status.
  /// A nonzero count bounds the payload masks. An appended status word owns all four bytes.
  uint8_t memory_result_payload_words = 0;
  void set_packed_image_result(unsigned components) {
    if (components & 1u) {
      memory_result_last_bytes = 0x3;
      memory_result_payload_words = (components + 1) / 2;
    }
  }
  const Operand *packed_store_data = nullptr; // Only the low half of the final word is stored.
  bool exec_all_if_nonzero = false;
  bool exec_whole_quads = false;

  /// VADDR operand enabling coordinate tracking.
  const Operand *image_address = nullptr;
  const Operand *image_resource = nullptr;
  bool image_resource_info = false;
  std::optional<uint8_t> image_transfer_dim;
  bool image_reads_address = true;
  /// Sampling controls select active lanes and implicit derivative inputs from the 2x2 quad.
  std::optional<ImageSampleSources> image_sample_sources;
  const Operand *image_store_data = nullptr;
  /// Coordinate VGPR selectors, including NSA selectors.
  ImageAddressRegisters image_coordinates{};
  /// Number of tracked coordinate components.
  uint8_t image_coordinate_count = 0;
  /// Coordinates occupy packed 16-bit halves.
  bool image_a16 = false;
  /// VGPR words before coordinates, including offset, bias, compare, and explicit gradients.
  uint8_t image_prefix_words = 0;
  /// Prefix words in which only the low half carries an input.
  uint16_t image_prefix_low16_words = 0;
  /// GFX9 derives its actual tuple from the descriptor. The sentinel
  /// leaves RDNA's instruction-derived coordinate metadata in effect.
  struct Gfx9ImageTuple {
    uint32_t resource_selector = ~0u;
    /// Coordinates include an array layer when the descriptor is layered.
    bool image_array = false;
    /// A view-relative level or sample LOD follows the coordinates.
    bool mip = false;
    /// Explicit gradient words precede the coordinates.
    bool derivatives = false;
  } gfx9_image_tuple;
  uint8_t scalar_buffer_words = 0; // Zero selects a complete vector descriptor.
  bool src2_is_wave_mask = false;
  const Operand *wordwise_source0 = nullptr;
  const Operand *wordwise_source1 = nullptr;
  /// Block load destination or store data, selected per DWORD by issue-time M0.
  const Operand *block_data = nullptr;
  const Operand *buffer_resource = nullptr;
  const Operand *buffer_address = nullptr;
  const Operand *buffer_offset = nullptr;
  const Operand *flat_address = nullptr;
  const Operand *flat_scalar_address = nullptr;
  int32_t flat_offset = 0;
  uint16_t flat_scale = 1;
  uint16_t flat_access_bytes = 0;
  bool dpp = false;
  bool dpp8 = false;
  bool sdwa = false;
  bool inactive_uses_bound_ctrl = false;
  uint32_t control = 0;
  uint8_t row_mask = 0xf;
  uint8_t bank_mask = 0xf;
  uint8_t bound_ctrl = 0;
  uint8_t fi = 1;
  const Operand *src0 = nullptr;
  const Operand *dpp_masked_compare_rhs = nullptr;
  const Operand *src1 = nullptr;
  uint8_t src0_selection = 6;
  uint8_t src1_selection = 6;
  uint8_t dst_selection = 6;
  uint8_t dst_unused = 0;
  uint8_t encoded_buffer_format = 0;   // Explicit FORMAT on typed buffer instructions.
  bool buffer_tfe = false;             // VBUFFER reserves an additional status VGPR.
  bool constant_float_compare = false; // Only reads inputs when checking for FP exceptions.
};

/// @brief VOP1/VOP2 src0 encoding values that indicate DPP or SDWA modifiers.
constexpr uint32_t SRC_SDWA = 249;
constexpr uint32_t SRC_DPP = 250;
constexpr uint32_t SRC_DPP8_FI_0 = 233;
constexpr uint32_t SRC_DPP8_FI_1 = 234;
constexpr uint32_t SRC_DPP8_LO = SRC_DPP8_FI_0;
constexpr uint32_t SRC_DPP8_HI = SRC_DPP8_FI_1;

/// @brief Bytes selected from a register by a true16 source or destination selector.
constexpr uint8_t true16_source_byte_mask(uint32_t opsel, uint32_t source_index) {
  return (opsel & (1u << source_index)) ? 0xc : 0x3;
}

namespace dpp {

/// @brief Floating source modifiers decoded from a DPP16 extension.
struct SourceModifiers {
  uint32_t absolute = 0;
  uint32_t negate = 0;

  template <typename Encoding> static SourceModifiers decode(const Encoding &encoding) {
    return {.absolute = uint32_t{encoding.src0_abs} | (uint32_t{encoding.src1_abs} << 1),
            .negate = uint32_t{encoding.src0_neg} | (uint32_t{encoding.src1_neg} << 1)};
  }
};

/// @brief DPP control value ranges encoded in VOP instruction modifiers.
enum DppCtrl : uint32_t {
  QUAD_PERM_MAX = 0xFF,
  ROW_SHL1 = 0x101,
  ROW_SHL_MAX = 0x10F,
  ROW_SHR1 = 0x111,
  ROW_SHR_MAX = 0x11F,
  ROW_ROR1 = 0x121,
  ROW_ROR_MAX = 0x12F,
  WF_SHL1 = 0x130,
  WF_ROL1 = 0x134,
  WF_SRL1 = 0x138,
  WF_ROR1 = 0x13C,
  ROW_MIRROR = 0x140,
  ROW_HALF_MIRROR = 0x141,
  ROW_BCAST15 = 0x142,
  ROW_BCAST31 = 0x143,
  ROW_SELECT_BASE = 0x150,
  ROW_SELECT_MAX = 0x15F,
  // MI400 names this range DPP_ROW_SHARE. Keep aliases for ISA-specific code.
  ROW_SHARE_BASE = ROW_SELECT_BASE,
  ROW_SHARE_MAX = ROW_SELECT_MAX,
  ROW_XMASK_BASE = 0x160,
  ROW_XMASK_MAX = 0x16F,
};

/// @brief ISA-specific names and validity rules for DPP_CTRL values.
enum class DppCtrlDialect {
  Gfx9,
  Gfx10Plus,
};

/// @brief Return true when a DPP control can read past a row or wave edge.
///
/// These controls leave some destination lanes unwritten when BOUND_CTRL is
/// zero. Rotates, mirrors, broadcasts, quad permutations, row-select, and
/// row-xmask always map to valid lanes.
inline bool dpp_ctrl_produces_oob(uint32_t dpp_ctrl) {
  return (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) ||
         (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) || dpp_ctrl == WF_SHL1 ||
         dpp_ctrl == WF_SRL1;
}

/// @brief Return true for a documented DPP16 control encoding.
inline bool dpp_ctrl_is_valid(uint32_t dpp_ctrl, bool allow_wave_ops, bool allow_row_bcast,
                              bool allow_row_xmask) {
  return dpp_ctrl <= QUAD_PERM_MAX || (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) ||
         (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) ||
         (dpp_ctrl >= ROW_ROR1 && dpp_ctrl <= ROW_ROR_MAX) ||
         (allow_wave_ops && (dpp_ctrl == WF_SHL1 || dpp_ctrl == WF_ROL1 || dpp_ctrl == WF_SRL1 ||
                             dpp_ctrl == WF_ROR1)) ||
         dpp_ctrl == ROW_MIRROR || dpp_ctrl == ROW_HALF_MIRROR ||
         (allow_row_bcast && (dpp_ctrl == ROW_BCAST15 || dpp_ctrl == ROW_BCAST31)) ||
         (dpp_ctrl >= ROW_SELECT_BASE && dpp_ctrl <= ROW_SELECT_MAX) ||
         (allow_row_xmask && dpp_ctrl >= ROW_XMASK_BASE && dpp_ctrl <= ROW_XMASK_MAX);
}

inline bool is_src_dpp8(uint32_t src0) { return src0 == SRC_DPP8_FI_0 || src0 == SRC_DPP8_FI_1; }

inline uint32_t src_dpp8_fi(uint32_t src0) { return src0 == SRC_DPP8_FI_1 ? 1u : 0u; }

/// @brief Add the lane permutation and write-mask attributes for DPP16.
inline void append_dpp16_disassembly(std::string &out, uint32_t dpp_ctrl, uint32_t row_mask,
                                     uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                                     bool has_fi, DppCtrlDialect dialect) {
  if (dpp_ctrl <= QUAD_PERM_MAX) {
    out += " quad_perm:[";
    for (uint32_t lane = 0; lane < 4; ++lane) {
      if (lane != 0)
        out += ',';
      out += std::to_string((dpp_ctrl >> (lane * 2)) & 0x3);
    }
    out += ']';
  } else if (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) {
    out += " row_shl:" + std::to_string(dpp_ctrl & 0xF);
  } else if (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) {
    out += " row_shr:" + std::to_string(dpp_ctrl & 0xF);
  } else if (dpp_ctrl >= ROW_ROR1 && dpp_ctrl <= ROW_ROR_MAX) {
    out += " row_ror:" + std::to_string(dpp_ctrl & 0xF);
  } else if (dpp_ctrl == WF_SHL1) {
    out += " wave_shl:1";
  } else if (dpp_ctrl == WF_ROL1) {
    out += " wave_rol:1";
  } else if (dpp_ctrl == WF_SRL1) {
    out += " wave_shr:1";
  } else if (dpp_ctrl == WF_ROR1) {
    out += " wave_ror:1";
  } else if (dpp_ctrl == ROW_MIRROR) {
    out += " row_mirror";
  } else if (dpp_ctrl == ROW_HALF_MIRROR) {
    out += " row_half_mirror";
  } else if (dialect == DppCtrlDialect::Gfx9 && dpp_ctrl == ROW_BCAST15) {
    out += " row_bcast:15";
  } else if (dialect == DppCtrlDialect::Gfx9 && dpp_ctrl == ROW_BCAST31) {
    out += " row_bcast:31";
  } else if (dpp_ctrl >= ROW_SHARE_BASE && dpp_ctrl <= ROW_SHARE_MAX) {
    out += dialect == DppCtrlDialect::Gfx9 ? " row_newbcast:" : " row_share:";
    out += std::to_string(dpp_ctrl & 0xF);
  } else if (dialect == DppCtrlDialect::Gfx10Plus && dpp_ctrl >= ROW_XMASK_BASE &&
             dpp_ctrl <= ROW_XMASK_MAX) {
    out += " row_xmask:" + std::to_string(dpp_ctrl & 0xF);
  } else {
    // Keep unexpected values compact for diagnostic-only callers. Executable
    // decode rejects controls outside the selected ISA profile first.
    constexpr char kHex[] = "0123456789abcdef";
    out += " dpp_ctrl:0x";
    out += kHex[(dpp_ctrl >> 8) & 0xF];
    out += kHex[(dpp_ctrl >> 4) & 0xF];
    out += kHex[dpp_ctrl & 0xF];
  }

  constexpr char kHex[] = "0123456789abcdef";
  out += " row_mask:0x";
  out += kHex[row_mask & 0xF];
  out += " bank_mask:0x";
  out += kHex[bank_mask & 0xF];
  if (bound_ctrl != 0)
    out += " bound_ctrl:1";
  if (has_fi && fi != 0)
    out += " fi:1";
}

/// @brief Add the lane selectors for a DPP8 instruction.
inline void append_dpp8_disassembly(std::string &out, uint32_t lane_sel, uint32_t fi) {
  out += " dpp8:[";
  for (uint32_t lane = 0; lane < 8; ++lane) {
    if (lane != 0)
      out += ',';
    out += std::to_string((lane_sel >> (lane * 3)) & 0x7);
  }
  out += ']';
  if (fi != 0)
    out += " fi:1";
}

} // namespace dpp

namespace vop {

inline void append_vop3_operand(std::string &out, std::string_view name, bool absolute, bool negate,
                                bool half_width, bool high_half) {
  if (negate)
    out += '-';
  if (absolute)
    out += '|';
  out += name;
  if (half_width)
    out += high_half ? ".h" : ".l";
  if (absolute)
    out += '|';
}

inline void append_bit_array(std::string &out, std::string_view name, uint32_t bits,
                             uint32_t count) {
  out += ' ';
  out += name;
  out += ":[";
  for (uint32_t index = 0; index < count; ++index) {
    if (index != 0)
      out += ',';
    out += ((bits >> index) & 1) != 0 ? '1' : '0';
  }
  out += ']';
}

/// @brief Add packed VOP3 source-selection and arithmetic attributes.
inline void append_vop3p_disassembly(std::string &out, uint32_t op_sel, uint32_t op_sel_hi,
                                     uint32_t neg_lo, uint32_t neg_hi, uint32_t clamp,
                                     uint32_t source_count, bool packed_defaults) {
  const uint32_t mask = (uint32_t{1} << source_count) - 1;
  if ((op_sel & mask) != 0)
    append_bit_array(out, "op_sel", op_sel, source_count);
  const uint32_t default_op_sel_hi = packed_defaults ? mask : 0;
  if ((op_sel_hi & mask) != default_op_sel_hi)
    append_bit_array(out, "op_sel_hi", op_sel_hi, source_count);
  if ((neg_lo & mask) != 0)
    append_bit_array(out, "neg_lo", neg_lo, source_count);
  if ((neg_hi & mask) != 0)
    append_bit_array(out, "neg_hi", neg_hi, source_count);
  if (clamp != 0)
    out += " clamp";
}

/// @brief Add standard VOP3 true16 selection and output modifiers.
inline void append_vop3_disassembly(std::string &out, uint32_t op_sel, uint32_t clamp,
                                    uint32_t omod, uint32_t source_count,
                                    bool display_true16_op_sel) {
  if (display_true16_op_sel) {
    const uint32_t source_mask = (uint32_t{1} << source_count) - 1;
    const uint32_t displayed_op_sel =
        (op_sel & source_mask) | (((op_sel >> 3) & 1) << source_count);
    if (displayed_op_sel != 0)
      append_bit_array(out, "op_sel", displayed_op_sel, source_count + 1);
  }
  if (clamp != 0)
    out += " clamp";
  if (omod == 1)
    out += " mul:2";
  else if (omod == 2)
    out += " mul:4";
  else if (omod == 3)
    out += " div:2";
}

} // namespace vop

namespace sdwa {

/// @brief SDWA sub-dword selection values stored by VOP encoding models.
enum SdwaSel : uint32_t {
  BYTE_0 = 0,
  BYTE_1 = 1,
  BYTE_2 = 2,
  BYTE_3 = 3,
  WORD_0 = 4,
  WORD_1 = 5,
  DWORD = 6,
};

/// @brief SDWA handling for destination bits outside the selected sub-dword.
enum SdwaUnused : uint32_t {
  UNUSED_PAD = 0,
  UNUSED_SEXT = 1,
  UNUSED_PRESERVE = 2,
};

/// @brief Semantic arithmetic result format for SDWA output modifiers.
enum class ResultFormat {
  NONE,
  F16,
  PK_F16,
  F32,
};

/// @brief Floating-point representation used by SDWA source modifiers.
///
/// SDWA selection and sign extension apply to every source. Absolute-value and
/// negate fields apply only to floating-point sources, and their sign bit
/// depends on the semantic source type rather than the selector width.
enum class SourceModifierFormat {
  NONE,
  F16,
  BF16,
  F32,
};

} // namespace sdwa

/// Select each result half independently, then flip its source sign bit if requested.
template <typename V>
inline V select_packed_halves(V value, uint32_t low, uint32_t high, unsigned operand,
                              uint32_t negations = 0) {
  const V selected = ((value >> (((low >> operand) & 1u) * 16)) & V(0xffffu)) |
                     ((value >> (((high >> operand) & 1u) * 16)) << 16);
  const uint32_t sign_bits = ((negations >> operand) & 0x00010001u) << 15;
  return selected ^ V(sign_bits);
}

/// @brief Return the VOP3 output-select field across MRISA spelling variants.
template <typename MachineInst> inline uint32_t vop3_opsel(const MachineInst &inst) {
  if constexpr (requires { inst.opsel; })
    return inst.opsel;
  else if constexpr (requires { inst.op_sel; })
    return inst.op_sel;
  else
    return 0;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_INSTRUCTION_ENCODING_H_
