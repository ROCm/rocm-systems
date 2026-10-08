// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXTRACT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXTRACT_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/register_access.h"
#include "util/data_types.h"
#include "util/meta_programming.h"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

struct ExtractF32 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return std::bit_cast<float>(matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane));
  }
};
inline constexpr ExtractF32 extract_f32{};

struct ExtractF16 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::f16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
  }
};
inline constexpr ExtractF16 extract_f16{};

struct ExtractBf16 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
  }
};
inline constexpr ExtractBf16 extract_bf16{};

struct ExtractI8 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return static_cast<int32_t>(static_cast<int8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractI8 extract_i8{};

struct ExtractU8 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return static_cast<int32_t>((raw >> (loc.sub_element * 8)) & 0xFF);
  }
};
inline constexpr ExtractU8 extract_u8{};

inline int32_t sign_extend_packed(uint32_t value, uint32_t bits) {
  uint32_t sign = 1u << (bits - 1);
  return static_cast<int32_t>((value ^ sign) - sign);
}

struct ExtractI4 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return sign_extend_packed(read_packed(cu, base, loc), 4);
  }
};
inline constexpr ExtractI4 extract_i4{};

struct ExtractU4 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return static_cast<int32_t>(read_packed(cu, base, loc));
  }
};
inline constexpr ExtractU4 extract_u4{};

struct ExtractFp8Ocp {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::fp8_e4m3_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractFp8Ocp extract_fp8_ocp{};

struct ExtractBf8Ocp {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf8_e5m2_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractBf8Ocp extract_bf8_ocp{};

struct ExtractFp8Fnuz {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::fp8_e4m3_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractFp8Fnuz extract_fp8_fnuz{};

struct ExtractBf8Fnuz {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf8_e5m2_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractBf8Fnuz extract_bf8_fnuz{};

struct ExtractFp8 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return extract_fp8_ocp(cu, base, loc);
  }
};
inline constexpr ExtractFp8 extract_fp8{};

struct ExtractBf8 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return extract_bf8_ocp(cu, base, loc);
  }
};
inline constexpr ExtractBf8 extract_bf8{};

struct ExtractFp4 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::fp4_e2m1_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractFp4 extract_fp4{};

struct ExtractFp6 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::fp6_e2m3_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractFp6 extract_fp6{};

struct ExtractBf6 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::bf6_e3m2_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractBf6 extract_bf6{};

struct ExtractF64 {
  double operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t lo = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    uint32_t hi = matrix_vgpr_word(cu, base + loc.vgpr_offset + 1, loc.lane);
    return std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
  }
};
inline constexpr ExtractF64 extract_f64{};

inline float decode_e8m0_scale(uint8_t raw) { return util::e8m0_to_f32(raw); }

inline float decode_wmma_scale_byte(uint8_t raw, uint32_t fmt) {
  switch (fmt) {
  case 0:
    return decode_e8m0_scale(raw);
  case 1:
    return util::fp8_e5m3_to_f32(raw);
  case 2:
    return util::fp8_e4m3_to_f32(raw);
  default:
    throw util::UnimplementedInst("unsupported WMMA scale format");
  }
}

template <typename Run> bool dispatch_matrix_fmt_pair(uint32_t a_fmt, uint32_t b_fmt, Run run) {
  switch (a_fmt) {
  case 0:
    switch (b_fmt) {
    case 0:
      run(8, 8, extract_fp8, extract_fp8);
      return true;
    case 1:
      run(8, 8, extract_fp8, extract_bf8);
      return true;
    case 2:
      run(8, 6, extract_fp8, extract_fp6);
      return true;
    case 3:
      run(8, 6, extract_fp8, extract_bf6);
      return true;
    case 4:
      run(8, 4, extract_fp8, extract_fp4);
      return true;
    }
    return false;
  case 1:
    switch (b_fmt) {
    case 0:
      run(8, 8, extract_bf8, extract_fp8);
      return true;
    case 1:
      run(8, 8, extract_bf8, extract_bf8);
      return true;
    case 2:
      run(8, 6, extract_bf8, extract_fp6);
      return true;
    case 3:
      run(8, 6, extract_bf8, extract_bf6);
      return true;
    case 4:
      run(8, 4, extract_bf8, extract_fp4);
      return true;
    }
    return false;
  case 2:
    switch (b_fmt) {
    case 0:
      run(6, 8, extract_fp6, extract_fp8);
      return true;
    case 1:
      run(6, 8, extract_fp6, extract_bf8);
      return true;
    case 2:
      run(6, 6, extract_fp6, extract_fp6);
      return true;
    case 3:
      run(6, 6, extract_fp6, extract_bf6);
      return true;
    case 4:
      run(6, 4, extract_fp6, extract_fp4);
      return true;
    }
    return false;
  case 3:
    switch (b_fmt) {
    case 0:
      run(6, 8, extract_bf6, extract_fp8);
      return true;
    case 1:
      run(6, 8, extract_bf6, extract_bf8);
      return true;
    case 2:
      run(6, 6, extract_bf6, extract_fp6);
      return true;
    case 3:
      run(6, 6, extract_bf6, extract_bf6);
      return true;
    case 4:
      run(6, 4, extract_bf6, extract_fp4);
      return true;
    }
    return false;
  case 4:
    switch (b_fmt) {
    case 0:
      run(4, 8, extract_fp4, extract_fp8);
      return true;
    case 1:
      run(4, 8, extract_fp4, extract_bf8);
      return true;
    case 2:
      run(4, 6, extract_fp4, extract_fp6);
      return true;
    case 3:
      run(4, 6, extract_fp4, extract_bf6);
      return true;
    case 4:
      run(4, 4, extract_fp4, extract_fp4);
      return true;
    }
    return false;
  default:
    return false;
  }
}

template <bool FP8, bool FNUZ> constexpr auto f8_extract_fn() {
  if constexpr (FP8) {
    if constexpr (FNUZ)
      return extract_fp8_fnuz;
    else
      return extract_fp8_ocp;
  } else {
    if constexpr (FNUZ)
      return extract_bf8_fnuz;
    else
      return extract_bf8_ocp;
  }
}

// ---------------------------------------------------------------------------
// SMFMAC (Sparse Matrix FMA) helpers and execution functions.
//
// Structured 2:4 sparsity: A is half-density (2 of every 4 K positions are
// nonzero). A per-lane index register selects which 2-of-4 positions are live.
// Each 4-bit nibble in the index encodes two 2-bit position selectors (p0, p1).
// ---------------------------------------------------------------------------

struct SmfmacReadFp8Ocp {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + byte_idx / 4, lane);
    return util::fp8_e4m3_ocp_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
  }
};
inline constexpr SmfmacReadFp8Ocp smfmac_read_fp8_ocp{};

struct SmfmacReadBf8Ocp {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + byte_idx / 4, lane);
    return util::bf8_e5m2_ocp_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
  }
};
inline constexpr SmfmacReadBf8Ocp smfmac_read_bf8_ocp{};

struct SmfmacReadFp8Fnuz {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + byte_idx / 4, lane);
    return util::fp8_e4m3_fnuz_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
  }
};
inline constexpr SmfmacReadFp8Fnuz smfmac_read_fp8_fnuz{};

struct SmfmacReadBf8Fnuz {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + byte_idx / 4, lane);
    return util::bf8_e5m2_fnuz_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
  }
};
inline constexpr SmfmacReadBf8Fnuz smfmac_read_bf8_fnuz{};

struct SmfmacReadFp8 {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    return smfmac_read_fp8_ocp(cu, base, byte_idx, lane);
  }
};
inline constexpr SmfmacReadFp8 smfmac_read_fp8{};

struct SmfmacReadBf8 {
  float operator()(auto &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) const {
    return smfmac_read_bf8_ocp(cu, base, byte_idx, lane);
  }
};
inline constexpr SmfmacReadBf8 smfmac_read_bf8{};

struct SmfmacReadF16 {
  float operator()(auto &cu, uint32_t base, uint32_t elem, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + elem / 2, lane);
    return util::f16_to_f32(static_cast<uint16_t>((raw >> ((elem % 2) * 16)) & 0xFFFF));
  }
};
inline constexpr SmfmacReadF16 smfmac_read_f16{};

struct SmfmacReadBf16 {
  float operator()(auto &cu, uint32_t base, uint32_t elem, uint32_t lane) const {
    uint32_t raw = RegisterAccess(cu).read_vgpr(base + elem / 2, lane);
    return util::bf16_to_f32(static_cast<uint16_t>((raw >> ((elem % 2) * 16)) & 0xFFFF));
  }
};
inline constexpr SmfmacReadBf16 smfmac_read_bf16{};

template <typename Extract>
inline constexpr uint32_t smfmac_input_bits =
    std::is_same_v<std::remove_cvref_t<Extract>, SmfmacReadF16> ||
            std::is_same_v<std::remove_cvref_t<Extract>, SmfmacReadBf16>
        ? 16u
        : 8u;

template <typename Extract> inline float smfmac_decode_word(uint32_t word, uint32_t element) {
  using Reader = std::remove_cvref_t<Extract>;
  if constexpr (std::is_same_v<Reader, SmfmacReadF16>) {
    return util::f16_to_f32(static_cast<uint16_t>(word >> (16 * (element % 2))));
  } else if constexpr (std::is_same_v<Reader, SmfmacReadBf16>) {
    return util::bf16_to_f32(static_cast<uint16_t>(word >> (16 * (element % 2))));
  } else {
    const auto byte = static_cast<uint8_t>(word >> (8 * (element % 4)));
    if constexpr (std::is_same_v<Reader, SmfmacReadFp8Ocp> || std::is_same_v<Reader, SmfmacReadFp8>)
      return util::fp8_e4m3_ocp_to_f32(byte);
    else if constexpr (std::is_same_v<Reader, SmfmacReadBf8Ocp> ||
                       std::is_same_v<Reader, SmfmacReadBf8>)
      return util::bf8_e5m2_ocp_to_f32(byte);
    else if constexpr (std::is_same_v<Reader, SmfmacReadFp8Fnuz>)
      return util::fp8_e4m3_fnuz_to_f32(byte);
    else if constexpr (std::is_same_v<Reader, SmfmacReadBf8Fnuz>)
      return util::bf8_e5m2_fnuz_to_f32(byte);
    else
      static_assert(util::always_false_v<Reader>, "unsupported SMFMAC input format");
  }
}

template <typename Extract, size_t Words>
inline float smfmac_decode_snapshot(const uint32_t (&words)[Words], uint32_t element,
                                    uint32_t lane) {
  constexpr uint32_t elements_per_word = 32 / smfmac_input_bits<Extract>;
  return smfmac_decode_word<Extract>(words[(element / elements_per_word) * 64 + lane], element);
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXTRACT_H_
