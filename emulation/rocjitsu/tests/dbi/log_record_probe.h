// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file log_record_probe.h
/// @brief A hand-built DBI probe that writes one RjLogRecord per call into the
///        log buffer the framework hands it, for CDNA3, CDNA4 and RDNA4.
///
/// The probe takes five argument dwords, laid out the same on both wave sizes so
/// one body serves both:
///
///   v0, v1  LogBufferPtrLo / LogBufferPtrHi
///   v2      AnchorExecLo
///   v3      AnchorExecHi on Wave64; the immediate 0 on Wave32, which refuses
///           AnchorExecHi
///   v4      the site, the anchor's original .text byte offset as an immediate
///
/// It must be called with force_full_exec: its loads and stores are EXEC-masked,
/// so under a zero guest mask nothing would be written. It never writes EXEC
/// itself, so every lane stores identical bytes to the same addresses.
///
/// What it writes: valid, abi_version, record_size, record_type, exec_mask,
/// site, active_lane_count (the guest mask's popcount) and writer_lane (the
/// guest mask's first set lane, or 0xffffffff when it is zero). wave_id,
/// workgroup_x/y/z and payload are written as zero.
///
/// Limits, all deliberate for a single-wave test read back after the dispatch:
/// - The slot claim is a plain load and store of write_ptr, not an atomic, and
///   there is no full-ring check: slot_count is fixed when the body is built and
///   the caller keeps the ring from overflowing (at most slot_count records).
/// - Stores are in program order only, with no wait before the probe returns.
/// - It does not check the buffer header's abi_version or record_size: the tests
///   build the buffer from the same compiled ABI this body is built from.
///
/// Hazards the simulator does not model are handled where LLVM's hazard passes
/// handle the same sequence. On GFX9 a VMEM read of an SGPR a VALU wrote needs
/// five wait states, which the instructions between the base pair's
/// v_readfirstlane and the first load already provide, so neither LLVM nor the
/// body adds an s_nop. On RDNA4, in a called function such as a probe, a SALU
/// write to an SGPR must be followed by an s_wait_alu before anything reads that
/// SGPR; the body has the four waits LLVM's amdgpu-wait-sgpr-hazards pass places.
///
/// Registers: s4..s12 and v0..v29, which the kernel must allocate. Any of them
/// live at the anchor has to be spilled around the call, which needs a kernel
/// with fixed scratch; the tests keep them all dead instead. Every multi-dword
/// VGPR operand starts on an even register, which CDNA requires from gfx90a on
/// (RDNA4 does not). Each store has its own data registers, so nothing
/// overwrites a dwordx3/x4 store's data in the one or two wait states after it
/// issues, while CDNA may still be reading it (RDNA4 has no such hazard).

#pragma once

#include "global_memory_builders.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/patch/log_abi.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/rj_code.h"

#include "util/except.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu {
namespace test {

/// @brief Encode `v_readfirstlane_b32 s<sdst>, v<vsrc>`.
[[nodiscard]] inline uint32_t build_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc,
                                                        rj_code_arch_t arch) {
  // VOP1 src0 uses the same 9-bit operand code as VOP3, and vdst names the SGPR.
  const uint16_t src0 = vop3_vgpr_src(vsrc);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop1(cdna3::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop1(cdna4::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop1(rdna4::kVReadfirstlaneB32Vop1, {.src0 = src0, .vdst = sdst}).front();
  default:
    throw util::UnimplementedInst("v_readfirstlane_b32 for target architecture");
  }
}

/// @brief Encode `s_bcnt1_i32_b64 s<sdst>, s[ssrc0:ssrc0+1]`.
[[nodiscard]] inline uint32_t build_s_bcnt1_i32_b64(uint16_t sdst, uint16_t ssrc0,
                                                    rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop1_encoding(arch, cdna3::kSBcnt1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop1_encoding(arch, cdna4::kSBcnt1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop1_encoding(arch, rdna4::kSBcnt1I32B64Sop1, sdst, ssrc0);
  default:
    throw util::UnimplementedInst("s_bcnt1_i32_b64 for target architecture");
  }
}

/// @brief Encode the index of the lowest set bit of s[ssrc0:ssrc0+1] into
///        s<sdst>, or -1 when none is set.
/// @details `s_ff1_i32_b64` on CDNA; RDNA4 renames it `s_ctz_i32_b64`. Both store
/// -1 for a zero input.
[[nodiscard]] inline uint32_t build_s_first_set_bit_b64(uint16_t sdst, uint16_t ssrc0,
                                                        rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop1_encoding(arch, cdna3::kSFf1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop1_encoding(arch, cdna4::kSFf1I32B64Sop1, sdst, ssrc0);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop1_encoding(arch, rdna4::kSCtzI32B64Sop1, sdst, ssrc0);
  default:
    throw util::UnimplementedInst("s_ff1_i32_b64 for target architecture");
  }
}

/// @brief Encode `s_and_b32 s<sdst>, ssrc0, ssrc1`.
[[nodiscard]] inline uint32_t build_s_and_b32(uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1,
                                              rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return build_sop2_encoding(arch, cdna3::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return build_sop2_encoding(arch, cdna4::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return build_sop2_encoding(arch, rdna4::kSAndB32Sop2, sdst, ssrc0, ssrc1);
  default:
    throw util::UnimplementedInst("s_and_b32 for target architecture");
  }
}

/// @brief The scalar source operand code that selects a trailing 32-bit literal.
inline constexpr uint16_t kScalarSrcLiteral = rdna4::OPR_SSRC_SRC_LITERAL;
static_assert(cdna3::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral &&
              cdna4::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral);

/// @brief Argument dwords the probe takes.
inline constexpr uint8_t kLogRecordProbeArgDwords = 5;
/// @brief Highest SGPR and VGPR the body writes. It also writes s4 upward and
///        reads v0..v4 as arguments.
inline constexpr uint16_t kLogRecordProbeLastSgpr = 12;
inline constexpr uint16_t kLogRecordProbeLastVgpr = 29;

/// @brief The probe's argument list for the anchor at @p anchor_offset.
[[nodiscard]] inline std::vector<ProbeArgValue> log_record_probe_args(uint64_t anchor_offset,
                                                                      bool wave32) {
  if (anchor_offset > std::numeric_limits<uint32_t>::max())
    throw std::out_of_range("RjLogRecord::site is 32 bits");
  return {{ProbeArgSource::LogBufferPtrLo, 0},
          {ProbeArgSource::LogBufferPtrHi, 0},
          {ProbeArgSource::AnchorExecLo, 0},
          wave32 ? probe_arg_imm(0) : ProbeArgValue{ProbeArgSource::AnchorExecHi, 0},
          probe_arg_imm(static_cast<uint32_t>(anchor_offset))};
}

namespace detail {

// The body stores the record as four contiguous groups plus valid; these pin
// the groups to the ABI.
static_assert(offsetof(RjLogRecord, valid) == 0);
static_assert(offsetof(RjLogRecord, abi_version) == 4 && offsetof(RjLogRecord, reserved0) == 6 &&
              offsetof(RjLogRecord, record_size) == 8 && offsetof(RjLogRecord, record_type) == 12);
static_assert(offsetof(RjLogRecord, exec_mask) == 16 && offsetof(RjLogRecord, site) == 24 &&
              offsetof(RjLogRecord, wave_id) == 28);
static_assert(offsetof(RjLogRecord, workgroup_x) == 32 &&
              offsetof(RjLogRecord, workgroup_y) == 36 &&
              offsetof(RjLogRecord, workgroup_z) == 40 &&
              offsetof(RjLogRecord, active_lane_count) == 44);
static_assert(offsetof(RjLogRecord, writer_lane) == 48 && offsetof(RjLogRecord, reserved1) == 52 &&
              offsetof(RjLogRecord, payload) == 56);
// The slot address is a shift, not a multiply.
inline constexpr uint32_t kRecordShift = 6;
static_assert((1u << kRecordShift) == sizeof(RjLogRecord));

inline void append(std::vector<uint32_t> &words, std::span<const uint32_t> more) {
  words.insert(words.end(), more.begin(), more.end());
}

// RDNA4 only: once a VALU may have read an SGPR, a later SALU write to it is not
// guaranteed visible to a reader until s_wait_alu clears sa_sdst. LLVM assumes
// every SGPR may have been read on entry to a called function, which a probe is.
// 0xfffe is the value LLVM's amdgpu-wait-sgpr-hazards pass emits for it.
inline void append_salu_sgpr_write_wait(std::vector<uint32_t> &words, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    words.push_back(build_sopp_encoding(arch, rdna4::kSWaitAluSopp, 0xfffe));
}

} // namespace detail

/// @brief Build the probe body for @p arch.
/// @param record_type Written into every record.
/// @param slot_count The buffer's slot count, a power of two no larger than 64.
[[nodiscard]] inline std::vector<uint32_t>
build_log_record_probe_body(rj_code_arch_t arch, uint32_t record_type, uint32_t slot_count) {
  if (slot_count == 0 || (slot_count & (slot_count - 1)) != 0 || slot_count > 64)
    throw std::invalid_argument("slot_count must be a power of two no larger than 64");

  constexpr uint8_t kBase = 4;      // s[4:5]
  constexpr uint8_t kExec = 6;      // s[6:7]
  constexpr uint8_t kCount = 8;     // s8
  constexpr uint8_t kWriter = 9;    // s9
  constexpr uint8_t kWritePtr = 10; // s[10:11]
  constexpr uint8_t kSlot = 12;     // s12
  constexpr uint8_t kZero = 5;      // v5, a zero per-lane offset
  constexpr uint8_t kLoaded = 6;    // v[6:7]
  constexpr uint8_t kSlotV = 8;     // v8, the slot's byte offset
  constexpr uint8_t kNextPtr = 28;  // v[28:29]
  constexpr uint32_t kWritePtrOffset = offsetof(RjLogBufferHeader, write_ptr);
  const uint16_t zero = vop3_inline_uint(0);

  std::vector<uint32_t> words;
  words.push_back(build_v_readfirstlane_b32(kBase, 0, arch));
  words.push_back(build_v_readfirstlane_b32(kBase + 1, 1, arch));
  words.push_back(build_v_readfirstlane_b32(kExec, 2, arch));
  words.push_back(build_v_readfirstlane_b32(kExec + 1, 3, arch));
  words.push_back(build_s_bcnt1_i32_b64(kCount, kExec, arch));
  words.push_back(build_s_first_set_bit_b64(kWriter, kExec, arch));
  words.push_back(build_v_mov_b32_src(kZero, zero, arch));
  // GFX9: a VMEM read of an SGPR a VALU wrote needs five wait states. The five
  // instructions after the base pair's last v_readfirstlane provide them before
  // the load below.

  // Claim the slot at write_ptr.
  detail::append(words, build_global_load_dwordx2(kLoaded, kZero, kBase, kWritePtrOffset, arch));
  words.push_back(build_wait_loads_complete(arch));
  words.push_back(build_v_readfirstlane_b32(kWritePtr, kLoaded, arch));
  words.push_back(build_v_readfirstlane_b32(kWritePtr + 1, kLoaded + 1, arch));
  // On RDNA4 each SALU write below is waited on before its SGPR is next read,
  // at the point LLVM places the wait.
  words.push_back(
      build_s_and_b32(kSlot, kWritePtr, scalar_positive_inline_u32(slot_count - 1), arch));
  detail::append_salu_sgpr_write_wait(words, arch);
  words.push_back(
      build_s_lshl_b32(kSlot, kSlot, scalar_positive_inline_u32(detail::kRecordShift), arch));
  detail::append_salu_sgpr_write_wait(words, arch);
  words.push_back(build_s_add_u32(kSlot, kSlot, kScalarSrcLiteral, arch));
  words.push_back(static_cast<uint32_t>(sizeof(RjLogBufferHeader)));
  detail::append_salu_sgpr_write_wait(words, arch);
  words.push_back(build_v_mov_b32_src(kSlotV, kSlot, arch));

  // abi_version (reserved0 zero), record_size, record_type.
  detail::append(words, build_v_mov_b32_imm(10, kRjLogAbiVersion, arch));
  detail::append(words, build_v_mov_b32_imm(11, kRjLogRecordSize, arch));
  detail::append(words, build_v_mov_b32_imm(12, record_type, arch));
  detail::append(words, build_global_store_dwordx3(10, kSlotV, kBase,
                                                   offsetof(RjLogRecord, abi_version), arch));

  // exec_mask, site, wave_id.
  words.push_back(build_v_mov_b32_src(14, vop3_vgpr_src(2), arch));
  words.push_back(build_v_mov_b32_src(15, vop3_vgpr_src(3), arch));
  words.push_back(build_v_mov_b32_src(16, vop3_vgpr_src(4), arch));
  words.push_back(build_v_mov_b32_src(17, zero, arch));
  detail::append(
      words, build_global_store_dwordx4(14, kSlotV, kBase, offsetof(RjLogRecord, exec_mask), arch));

  // workgroup_x/y/z, active_lane_count.
  words.push_back(build_v_mov_b32_src(18, zero, arch));
  words.push_back(build_v_mov_b32_src(19, zero, arch));
  words.push_back(build_v_mov_b32_src(20, zero, arch));
  words.push_back(build_v_mov_b32_src(21, kCount, arch));
  detail::append(words, build_global_store_dwordx4(18, kSlotV, kBase,
                                                   offsetof(RjLogRecord, workgroup_x), arch));

  // writer_lane, reserved1, payload.
  words.push_back(build_v_mov_b32_src(22, kWriter, arch));
  words.push_back(build_v_mov_b32_src(23, zero, arch));
  words.push_back(build_v_mov_b32_src(24, zero, arch));
  words.push_back(build_v_mov_b32_src(25, zero, arch));
  detail::append(words, build_global_store_dwordx4(22, kSlotV, kBase,
                                                   offsetof(RjLogRecord, writer_lane), arch));

  // valid, after the rest of the record.
  words.push_back(build_v_mov_b32_src(26, vop3_inline_uint(1), arch));
  detail::append(words,
                 build_global_store_dword(26, kSlotV, kBase, offsetof(RjLogRecord, valid), arch));

  // write_ptr + 1, after valid.
  words.push_back(build_s_add_u32(kWritePtr, kWritePtr, scalar_positive_inline_u32(1), arch));
  words.push_back(
      build_s_addc_u32(kWritePtr + 1, kWritePtr + 1, scalar_positive_inline_u32(0), arch));
  detail::append_salu_sgpr_write_wait(words, arch);
  words.push_back(build_v_mov_b32_src(kNextPtr, kWritePtr, arch));
  words.push_back(build_v_mov_b32_src(kNextPtr + 1, kWritePtr + 1, arch));
  detail::append(words, build_global_store_dwordx2(kNextPtr, kZero, kBase, kWritePtrOffset, arch));

  words.push_back(build_s_setpc_b64(30, arch));
  return words;
}

} // namespace test
} // namespace rocjitsu
