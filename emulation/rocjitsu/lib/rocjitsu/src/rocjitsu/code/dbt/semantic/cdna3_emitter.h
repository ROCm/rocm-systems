// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna3_emitter.h
/// @brief Target-side CDNA3 instruction encoders shared by semantic DBT lowerings.

#pragma once

#include <cstdint>
#include <utility>

namespace rocjitsu {

/// @brief Raw CDNA3 instruction encoder used by handwritten semantic lowerings.
///
/// @details Translation rules decide which target operation preserves the guest
/// semantics. This class owns only the target encoding details, allowing the
/// same CDNA3 emission support to be reused by other guest-to-CDNA3 translators.
/// Keeping raw bitfield construction here also prevents virtual-LDS and scratch
/// policies from spreading target machine-layout knowledge across rule bodies.
class Cdna3Emitter {
public:
  using WordPair = std::pair<uint32_t, uint32_t>;

  /// @brief Source-independent fields preserved when rebuilding a CDNA3 MUBUF.
  ///
  /// @details Guest decoding belongs to the source/target-pair lowering. The
  /// emitter accepts only fields required by the target MUBUF encoding so no
  /// source architecture types leak into this reusable target component.
  struct MubufOperands {
    uint16_t offset = 0;
    bool offen = false;
    bool idxen = false;
    bool sc0 = false;
    bool sc1 = false;
    bool nt = false;
    uint8_t vaddr = 0;
    uint8_t srsrc = 0;
    uint8_t soffset = 0;
  };

  /// @brief Common addressing fields for CDNA3 FLAT GLOBAL-segment operations.
  ///
  /// @details GLOBAL mode consumes a 64-bit VGPR address pair at @c addr, an
  /// optional 64-bit scalar base at @c saddr, and a signed immediate offset.
  /// Virtual LDS may use either a scalar backing base plus VGPR LDS offset or a
  /// complete 64-bit address materialized in VGPRs with scalar-null as the base.
  struct FlatGlobalOperands {
    uint16_t signed_offset13 = 0;
    bool sc0 = false;
    bool sc1 = false;
    bool nt = false;
    uint8_t addr = 0;
    uint8_t saddr = 0;
    bool acc = false;
  };

  /// @brief Encode a CDNA3 DS instruction.
  [[nodiscard]] static WordPair ds(uint16_t op, uint8_t vdst, uint8_t addr, uint8_t data0 = 0,
                                   uint8_t data1 = 0, uint8_t offset0 = 0, uint8_t offset1 = 0);

  /// @brief Encode a CDNA3 non-LDS MUBUF instruction.
  [[nodiscard]] static WordPair mubuf(const MubufOperands &src, uint16_t op, uint8_t vdata);

  /// @brief Encode a CDNA3 flat/global load with a VGPR address and optional SGPR base.
  [[nodiscard]] static WordPair flat_global_load(const FlatGlobalOperands &src, uint16_t op,
                                                 uint8_t vdst);

  /// @brief Encode a CDNA3 flat/global store with a VGPR address and optional SGPR base.
  [[nodiscard]] static WordPair flat_global_store(const FlatGlobalOperands &src, uint16_t op,
                                                  uint8_t data);

  /// @brief Encode a CDNA3 flat-scratch dword load or store for semantic spills.
  ///
  /// @details This form has no VGPR address or SGPR offset operand. The per-lane
  /// spill slot is carried by the signed 13-bit immediate field.
  [[nodiscard]] static WordPair flat_scratch_dword(uint16_t op, uint8_t vgpr, uint32_t byte_offset,
                                                   bool is_load);

  /// @brief Encode `s_load_dwordx2` with an immediate byte offset.
  ///
  /// @details CDNA3 encodes the scalar base as an SGPR-pair index. Entry
  /// prologues use this for original kernarg pointers and 64-bit runtime state.
  [[nodiscard]] static WordPair smem_load_dwordx2(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                  uint32_t byte_offset);

  /// @brief Encode `s_load_dword` with an immediate byte offset.
  ///
  /// @details Virtual-LDS entry prologues use this for compact per-dispatch
  /// values such as per-workgroup byte strides.
  [[nodiscard]] static WordPair smem_load_dword(uint8_t dst_sgpr, uint8_t sbase_sgpr,
                                                uint32_t byte_offset);

  /// @brief Encode a CDNA3 VOP2 instruction with a 32-bit literal source.
  ///
  /// @details The source operand uses the hardware literal selector (`0xff`),
  /// with the literal in the second dword. Virtual-LDS address materialization
  /// uses this so large DS offsets do not consume an extra scalar register.
  [[nodiscard]] static WordPair vop2_literal(uint16_t op, uint8_t vdst, uint8_t vsrc1,
                                             uint32_t literal);
};

} // namespace rocjitsu
