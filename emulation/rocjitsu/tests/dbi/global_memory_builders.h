// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file global_memory_builders.h
/// @brief Global-memory load and store encoders for hand-built DBI simulator
///        test code, on the targets DbiSim runs (CDNA3, CDNA4, RDNA4).
///
/// Every encoder uses the SGPR-base form: the address is the 64-bit SGPR pair at
/// @p saddr plus a 32-bit per-lane offset in VGPR @p voffset plus an immediate
/// byte offset. 2 words on CDNA, 3 on RDNA4.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

#include "util/except.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rocjitsu {
namespace test {

namespace detail {

/// @brief One global memory operation's opcode on each target.
struct GlobalOpcodes {
  uint16_t cdna3 = 0;
  uint16_t cdna4 = 0;
  uint16_t rdna4 = 0;
};

/// @brief The immediate byte offset each target encodes, in this form.
/// @details CDNA's FLAT offset is a 13-bit signed field whose low 12 bits the
/// generated builder carries; only non-negative offsets below 4096 are used here.
/// RDNA4's ioffset is 24-bit signed.
[[nodiscard]] inline uint32_t checked_global_offset(uint32_t byte_offset, rj_code_arch_t arch) {
  const uint32_t limit = arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0x7FFFFFu : 0xFFFu;
  if (byte_offset > limit)
    throw std::out_of_range("global memory offset does not fit the immediate field");
  return byte_offset;
}

/// @brief CDNA3 and CDNA4 (gfx90a onward) require a multi-dword VGPR operand to
///        start on an even register; RDNA4 does not.
/// @details The simulator does not enforce this, so a misaligned tuple would
/// pass there and be illegal on hardware.
inline void check_tuple_alignment(uint8_t base, uint32_t dwords, rj_code_arch_t arch) {
  if (dwords > 1 && base % 2 != 0 && arch != ROCJITSU_CODE_ARCH_RDNA4)
    throw std::invalid_argument("multi-dword VGPR operand must start on an even register");
}

/// @brief The SGPR base is a pair, which every target requires to start on an
///        even register.
/// @details The decoder accepts an odd base, so the simulator would run it.
inline void check_scalar_base(uint8_t saddr) {
  if (saddr % 2 != 0)
    throw std::invalid_argument("SGPR base pair must start on an even register");
}

[[nodiscard]] inline std::vector<uint32_t> build_global_load(GlobalOpcodes ops, uint32_t dwords,
                                                             uint8_t vdst, uint8_t voffset,
                                                             uint8_t saddr, uint32_t byte_offset,
                                                             rj_code_arch_t arch) {
  const uint32_t offset = checked_global_offset(byte_offset, arch);
  check_tuple_alignment(vdst, dwords, arch);
  check_scalar_base(saddr);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto encoded = cdna3::build_flat(ops.cdna3, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .saddr = saddr,
                                                       .vdst = vdst});
    return {encoded.begin(), encoded.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto encoded = cdna4::build_flat(ops.cdna4, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .saddr = saddr,
                                                       .vdst = vdst});
    return {encoded.begin(), encoded.end()};
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto encoded = rdna4::build_vglobal(
        ops.rdna4, {.saddr = saddr, .vdst = vdst, .vaddr = voffset, .ioffset = offset});
    return {encoded.begin(), encoded.end()};
  }
  default:
    throw util::UnimplementedInst("global load for target architecture");
  }
}

[[nodiscard]] inline std::vector<uint32_t> build_global_store(GlobalOpcodes ops, uint32_t dwords,
                                                              uint8_t vdata, uint8_t voffset,
                                                              uint8_t saddr, uint32_t byte_offset,
                                                              rj_code_arch_t arch) {
  const uint32_t offset = checked_global_offset(byte_offset, arch);
  check_tuple_alignment(vdata, dwords, arch);
  check_scalar_base(saddr);
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto encoded = cdna3::build_flat(ops.cdna3, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .data = vdata,
                                                       .saddr = saddr});
    return {encoded.begin(), encoded.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto encoded = cdna4::build_flat(ops.cdna4, {.offset = static_cast<uint16_t>(offset),
                                                       .seg = 2,
                                                       .addr = voffset,
                                                       .data = vdata,
                                                       .saddr = saddr});
    return {encoded.begin(), encoded.end()};
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto encoded = rdna4::build_vglobal(
        ops.rdna4, {.saddr = saddr, .vsrc = vdata, .vaddr = voffset, .ioffset = offset});
    return {encoded.begin(), encoded.end()};
  }
  default:
    throw util::UnimplementedInst("global store for target architecture");
  }
}

} // namespace detail

/// @brief Encode `global_load_dword vdst, voffset, s[saddr:saddr+1] offset:byte_offset`.
/// @note A build_wait_loads_complete() must precede any use of @p vdst.
[[nodiscard]] inline std::vector<uint32_t> build_global_load_dword(uint8_t vdst, uint8_t voffset,
                                                                   uint8_t saddr,
                                                                   uint32_t byte_offset,
                                                                   rj_code_arch_t arch) {
  return detail::build_global_load(
      {cdna3::kFlatLoadDwordFlat, cdna4::kFlatLoadDwordFlat, rdna4::kGlobalLoadB32Vglobal}, 1, vdst,
      voffset, saddr, byte_offset, arch);
}

/// @brief Encode `global_load_dwordx2 v[vdst:vdst+1], voffset, s[saddr:saddr+1]
/// offset:byte_offset`.
/// @note A build_wait_loads_complete() must precede any use of the destination.
[[nodiscard]] inline std::vector<uint32_t> build_global_load_dwordx2(uint8_t vdst, uint8_t voffset,
                                                                     uint8_t saddr,
                                                                     uint32_t byte_offset,
                                                                     rj_code_arch_t arch) {
  return detail::build_global_load(
      {cdna3::kFlatLoadDwordx2Flat, cdna4::kFlatLoadDwordx2Flat, rdna4::kGlobalLoadB64Vglobal}, 2,
      vdst, voffset, saddr, byte_offset, arch);
}

/// @brief Encode `global_store_dword voffset, vdata, s[saddr:saddr+1] offset:byte_offset`.
[[nodiscard]] inline std::vector<uint32_t> build_global_store_dword(uint8_t vdata, uint8_t voffset,
                                                                    uint8_t saddr,
                                                                    uint32_t byte_offset,
                                                                    rj_code_arch_t arch) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordFlat, cdna4::kFlatStoreDwordFlat, rdna4::kGlobalStoreB32Vglobal}, 1,
      vdata, voffset, saddr, byte_offset, arch);
}

/// @brief Encode `global_store_dwordx2` of v[vdata:vdata+1].
[[nodiscard]] inline std::vector<uint32_t>
build_global_store_dwordx2(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx2Flat, cdna4::kFlatStoreDwordx2Flat, rdna4::kGlobalStoreB64Vglobal},
      2, vdata, voffset, saddr, byte_offset, arch);
}

/// @brief Encode `global_store_dwordx3` of v[vdata:vdata+2].
[[nodiscard]] inline std::vector<uint32_t>
build_global_store_dwordx3(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx3Flat, cdna4::kFlatStoreDwordx3Flat, rdna4::kGlobalStoreB96Vglobal},
      3, vdata, voffset, saddr, byte_offset, arch);
}

/// @brief Encode `global_store_dwordx4` of v[vdata:vdata+3].
[[nodiscard]] inline std::vector<uint32_t>
build_global_store_dwordx4(uint8_t vdata, uint8_t voffset, uint8_t saddr, uint32_t byte_offset,
                           rj_code_arch_t arch) {
  return detail::build_global_store(
      {cdna3::kFlatStoreDwordx4Flat, cdna4::kFlatStoreDwordx4Flat, rdna4::kGlobalStoreB128Vglobal},
      4, vdata, voffset, saddr, byte_offset, arch);
}

} // namespace test
} // namespace rocjitsu
