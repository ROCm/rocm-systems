// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx12_cache_flags.h
/// @brief GFX12 (RDNA4) memory coherency flag → Mtype mapping.
///
/// RDNA4 uses a packed SCOPE field (2 bits) and a TH field (2–3 bits depending
/// on instruction format).  For load/store instructions TH is a temporal hint;
/// for atomics the same field encodes atomic-return mode.  There are no
/// GLC/DLC/SLC fields.
///
/// SCOPE encoding:
///   0 = CU, 1 = SE (Shader Engine), 2 = DEVICE, 3 = SYSTEM
///
/// TH encoding (simplified for simulator):
///   0 = regular, 1 = non-temporal (NT), 2+ = format-specific hints
///
/// | SCOPE | TH  | Mtype | Behavior                             |
/// |-------|-----|-------|--------------------------------------|
/// |   0   |  0  | RW    | L1+L2 cached, CU scope               |
/// |   1   |  0  | RW    | L1+L2 cached, SE scope               |
/// |   2   |  0  | CC    | Coherent at device scope              |
/// |   3   |  0  | UC    | System scope, bypass caches           |
/// |   x   |  1  | NT    | Non-temporal hint                     |

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h"

#include <cstdint>
#include <string>

namespace rocjitsu {
namespace amdgpu {

/// @brief Non-temporal TH value for GFX12.
inline constexpr uint8_t GFX12_TH_NT = 1;

/// @brief Atomic-return TH value for GFX12 atomic memory instructions.
inline constexpr uint8_t GFX12_TH_ATOMIC_RETURN = 1;

enum class Gfx12TemporalHintKind : uint8_t { Load, Store, Atomic };

inline void append_gfx12_hex_hint(std::string &out, uint8_t value) {
  static constexpr char digits[] = "0123456789abcdef";
  out += "0x";
  out += digits[value & 0xf];
}

/// @brief Append the assembly spelling of GFX12 TH and SCOPE fields.
inline void append_gfx12_cache_policy(std::string &out, uint8_t th, uint8_t scope,
                                      Gfx12TemporalHintKind kind) {
  if (th != 0) {
    out += " th:";
    if (kind == Gfx12TemporalHintKind::Atomic) {
      out += "TH_ATOMIC_";
      if ((th & 4) != 0) {
        if (scope >= 2)
          out += (th & 2) != 0 ? "CASCADE_NT" : "CASCADE_RT";
        else
          append_gfx12_hex_hint(out, th);
      } else if ((th & 2) != 0) {
        out += (th & 1) != 0 ? "NT_RETURN" : "NT";
      } else if ((th & 1) != 0) {
        out += "RETURN";
      } else {
        append_gfx12_hex_hint(out, th);
      }
    } else if (kind == Gfx12TemporalHintKind::Load && th == 7) {
      append_gfx12_hex_hint(out, th);
    } else {
      out += kind == Gfx12TemporalHintKind::Store ? "TH_STORE_" : "TH_LOAD_";
      switch (th) {
      case 1:
        out += "NT";
        break;
      case 2:
        out += "HT";
        break;
      case 3:
        out += scope == 3 ? "BYPASS" : (kind == Gfx12TemporalHintKind::Store ? "WB" : "LU");
        break;
      case 4:
        out += "NT_RT";
        break;
      case 5:
        out += "RT_NT";
        break;
      case 6:
        out += "NT_HT";
        break;
      case 7:
        out += "NT_WB";
        break;
      default:
        append_gfx12_hex_hint(out, th);
        break;
      }
    }
  }

  switch (scope) {
  case 1:
    out += " scope:SCOPE_SE";
    break;
  case 2:
    out += " scope:SCOPE_DEV";
    break;
  case 3:
    out += " scope:SCOPE_SYS";
    break;
  default:
    break;
  }
}

/// @brief Derive Mtype from GFX12 (RDNA4) coherency flags.
/// @param scope_val 2-bit SCOPE field value.
/// @param th        Temporal hint field value (1 = non-temporal).
[[nodiscard]] inline constexpr Mtype mtype_from_flags_gfx12(uint8_t scope_val, uint8_t th) {
  return mtype_from_scope_nt(scope_val, th == GFX12_TH_NT);
}

/// @brief Return true when a GFX12 atomic instruction writes back the old value.
[[nodiscard]] inline constexpr bool gfx12_atomic_returns(uint8_t th) {
  return th == GFX12_TH_ATOMIC_RETURN;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_GFX12_CACHE_FLAGS_H_
