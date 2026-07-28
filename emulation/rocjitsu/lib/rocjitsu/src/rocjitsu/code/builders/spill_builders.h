// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file spill_builders.h
/// @brief ISA-dispatched DBI register-spilling instruction builders.
///
/// @details These are the vector/memory ops the scalar helpers in
/// instruction_builder.h do not cover (CDNA3, CDNA4, and RDNA4): the SGPR<->VGPR
/// lane bridge, off-mode scratch store/load, and the store- and load-completion
/// waits (RDNA4 splits store and load counters, so both are needed). All use
/// off-mode addressing (lds/sve left 0), so the caller passes only a within-lane
/// byte offset from SpillManager.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "util/except.h"

namespace rocjitsu {

/// @brief Encode v_writelane_b32: bridge SGPR @p sgpr_src into lane of @p vgpr_dst.
[[nodiscard]] inline std::array<uint32_t, 2>
build_v_writelane_b32(uint16_t vgpr_dst, uint16_t sgpr_src, uint16_t lane, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop3(
        cdna3::kVWritelaneB32Vop3,
        {.vdst = static_cast<uint8_t>(vgpr_dst), .src0 = sgpr_src, .src1 = vop3_inline_uint(lane)});
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop3(
        cdna4::kVWritelaneB32Vop3,
        {.vdst = static_cast<uint8_t>(vgpr_dst), .src0 = sgpr_src, .src1 = vop3_inline_uint(lane)});
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop3(
        rdna4::kVWritelaneB32Vop3,
        {.vdst = static_cast<uint8_t>(vgpr_dst), .src0 = sgpr_src, .src1 = vop3_inline_uint(lane)});
  default:
    throw util::UnimplementedInst("v_writelane_b32 for target architecture");
  }
}

/// @brief Encode v_readlane_b32: read lane of @p vgpr_src back into @p sgpr_dst.
[[nodiscard]] inline std::array<uint32_t, 2>
build_v_readlane_b32(uint16_t sgpr_dst, uint16_t vgpr_src, uint16_t lane, rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop3(cdna3::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sgpr_dst),
                                                        .src0 = vop3_vgpr_src(vgpr_src),
                                                        .src1 = vop3_inline_uint(lane)});
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop3(cdna4::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sgpr_dst),
                                                        .src0 = vop3_vgpr_src(vgpr_src),
                                                        .src1 = vop3_inline_uint(lane)});
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop3(rdna4::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sgpr_dst),
                                                        .src0 = vop3_vgpr_src(vgpr_src),
                                                        .src1 = vop3_inline_uint(lane)});
  default:
    throw util::UnimplementedInst("v_readlane_b32 for target architecture");
  }
}

/// @brief Encode an off-mode scratch store of @p vdata at @p byte_offset.
/// @param acc When true, @p vdata names an AccVGPR (CDNA `acc` bit) rather than an
///   ordinary VGPR, so the store reads its data from the accumulator file. AGPRs
///   exist only on CDNA; requesting @p acc on an arch without them is rejected.
/// @note CDNA offset is 12-bit (0..4095); RDNA is 24-bit. 2 words on CDNA, 3 on RDNA.
[[nodiscard]] inline std::vector<uint32_t>
build_scratch_store_dword(uint16_t vdata, uint32_t byte_offset, rj_code_arch_t arch,
                          bool acc = false) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto w = cdna3::build_flat(cdna3::kFlatStoreDwordFlat,
                                     {.offset = static_cast<uint16_t>(byte_offset & 0xFFFu),
                                      .seg = 1,
                                      .data = static_cast<uint8_t>(vdata),
                                      .saddr = 0x7F,
                                      .acc = static_cast<uint8_t>(acc ? 1 : 0)});
    return {w.begin(), w.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto w = cdna4::build_flat(cdna4::kFlatStoreDwordFlat,
                                     {.offset = static_cast<uint16_t>(byte_offset & 0xFFFu),
                                      .seg = 1,
                                      .data = static_cast<uint8_t>(vdata),
                                      .saddr = 0x7F,
                                      .acc = static_cast<uint8_t>(acc ? 1 : 0)});
    return {w.begin(), w.end()};
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    if (acc)
      throw util::UnimplementedInst("scratch_store of an AccVGPR on an arch without them");
    const auto w = rdna4::build_vscratch(
        rdna4::kScratchStoreB32Vscratch,
        {.saddr = 0x7C, .vsrc = static_cast<uint8_t>(vdata), .ioffset = byte_offset & 0xFFFFFFu});
    return {w.begin(), w.end()};
  }
  default:
    throw util::UnimplementedInst("scratch_store for target architecture");
  }
}

/// @brief Encode an off-mode scratch load into @p vdst at @p byte_offset.
/// @param acc When true, @p vdst names an AccVGPR (CDNA `acc` bit) rather than an
///   ordinary VGPR, so the load writes its result into the accumulator file. AGPRs
///   exist only on CDNA; requesting @p acc on an arch without them is rejected.
/// @note A build_wait_loads_complete() must precede any use of @p vdst.
[[nodiscard]] inline std::vector<uint32_t>
build_scratch_load_dword(uint16_t vdst, uint32_t byte_offset, rj_code_arch_t arch,
                         bool acc = false) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto w = cdna3::build_flat(cdna3::kFlatLoadDwordFlat,
                                     {.offset = static_cast<uint16_t>(byte_offset & 0xFFFu),
                                      .seg = 1,
                                      .saddr = 0x7F,
                                      .acc = static_cast<uint8_t>(acc ? 1 : 0),
                                      .vdst = static_cast<uint8_t>(vdst)});
    return {w.begin(), w.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto w = cdna4::build_flat(cdna4::kFlatLoadDwordFlat,
                                     {.offset = static_cast<uint16_t>(byte_offset & 0xFFFu),
                                      .seg = 1,
                                      .saddr = 0x7F,
                                      .acc = static_cast<uint8_t>(acc ? 1 : 0),
                                      .vdst = static_cast<uint8_t>(vdst)});
    return {w.begin(), w.end()};
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    if (acc)
      throw util::UnimplementedInst("scratch_load of an AccVGPR on an arch without them");
    const auto w = rdna4::build_vscratch(
        rdna4::kScratchLoadB32Vscratch,
        {.saddr = 0x7C, .vdst = static_cast<uint8_t>(vdst), .ioffset = byte_offset & 0xFFFFFFu});
    return {w.begin(), w.end()};
  }
  default:
    throw util::UnimplementedInst("scratch_load for target architecture");
  }
}

/// @brief Encode a wait for outstanding stores, used to order a scratch store
///        before a same-address reload.
///
/// CDNA uses the unified vmcnt (s_waitcnt covers stores). RDNA4/GFX12 split the
/// counters, so stores need s_wait_storecnt — s_wait_loadcnt does NOT order them.
[[nodiscard]] inline uint32_t build_wait_stores_complete(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sopp_encoding(arch, cdna4::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sopp_encoding(arch, rdna4::kSWaitStorecntSopp, 0);
  default:
    throw util::UnimplementedInst("store-completion wait for target architecture");
  }
}

/// @brief Encode a wait for outstanding loads: s_waitcnt 0 (CDNA), s_wait_loadcnt 0 (RDNA).
[[nodiscard]] inline uint32_t build_wait_loads_complete(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sopp_encoding(arch, cdna4::kSWaitcntSopp, 0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sopp_encoding(arch, rdna4::kSWaitLoadcntSopp, 0);
  default:
    throw util::UnimplementedInst("load-completion wait for target architecture");
  }
}

/// @brief Drain every memory-load counter that could still be writing a register
///        about to be spilled: VMEM/LDS loads (VGPR targets) and scalar loads
///        (SGPR targets).
///
/// A register live at the anchor may be the destination of a load issued before
/// the anchor whose consumer s_waitcnt sits *after* it in the original code.
/// Storing that register in the spill prologue without first waiting would spill
/// a stale value (and the epilogue would then restore the stale value over the
/// load's later result). Emitted once, before any store.
///
/// CDNA's monolithic s_waitcnt 0 drains vmcnt+lgkmcnt (VMEM, LDS, and scalar) in
/// one word. RDNA4/GFX12 split the counters, so it takes s_wait_loadcnt_dscnt 0
/// (VMEM + LDS -> VGPRs) and s_wait_kmcnt 0 (scalar -> SGPRs).
[[nodiscard]] inline std::vector<uint32_t> build_wait_all_loads_complete(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return {build_sopp_encoding(arch, cdna3::kSWaitcntSopp, 0)};
  case ROCJITSU_CODE_ARCH_CDNA4:
    return {build_sopp_encoding(arch, cdna4::kSWaitcntSopp, 0)};
  case ROCJITSU_CODE_ARCH_RDNA4:
    return {build_sopp_encoding(arch, rdna4::kSWaitLoadcntDscntSopp, 0),
            build_sopp_encoding(arch, rdna4::kSWaitKmcntSopp, 0)};
  default:
    throw util::UnimplementedInst("all-loads wait for target architecture");
  }
}

} // namespace rocjitsu
