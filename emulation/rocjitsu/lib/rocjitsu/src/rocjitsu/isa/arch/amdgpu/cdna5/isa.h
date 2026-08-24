// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"
#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {
class Wavefront;
} // namespace rocjitsu::amdgpu

namespace rocjitsu::cdna5 {

/// @brief gfx1250 STATUS register layout.
///
/// @details Matches the GFX12/RDNA4 layout: one 32-bit scalar register per
/// wavefront, without TTRACE_EN, EXPORT_RDY, or SKIP_EXPORT fields.
class StatusReg : public ::util::Bitfield<32> {
public:
  using Bitfield::Bitfield;
  using Bitfield::operator=;
  StatusReg(const StatusReg &) = default;
  StatusReg &operator=(const StatusReg &) = default;

  auto SCC() { return member<0, 0>(); }
  auto SCC() const { return member<0, 0>(); }
  auto SPI_PRIO() { return member<1, 2>(); }
  auto SPI_PRIO() const { return member<1, 2>(); }
  auto WAVE_PRIO() { return member<3, 4>(); }
  auto WAVE_PRIO() const { return member<3, 4>(); }
  auto PRIV() { return member<5, 5>(); }
  auto PRIV() const { return member<5, 5>(); }
  auto TRAP_EN() { return member<6, 6>(); }
  auto TRAP_EN() const { return member<6, 6>(); }
  auto EXECZ() { return member<9, 9>(); }
  auto EXECZ() const { return member<9, 9>(); }
  auto VCCZ() { return member<10, 10>(); }
  auto VCCZ() const { return member<10, 10>(); }
  auto IN_TG() { return member<11, 11>(); }
  auto IN_TG() const { return member<11, 11>(); }
  auto IN_BARRIER() { return member<12, 12>(); }
  auto IN_BARRIER() const { return member<12, 12>(); }
  auto HALT() { return member<13, 13>(); }
  auto HALT() const { return member<13, 13>(); }
  auto TRAP() { return member<14, 14>(); }
  auto TRAP() const { return member<14, 14>(); }
  auto VALID() { return member<16, 16>(); }
  auto VALID() const { return member<16, 16>(); }
  auto ECC_ERR() { return member<17, 17>(); }
  auto ECC_ERR() const { return member<17, 17>(); }
  auto ALLOW_REPLAY() { return member<22, 22>(); }
  auto ALLOW_REPLAY() const { return member<22, 22>(); }
};

/// @brief gfx1250 ISA traits.
struct Isa : amdgpu::RdnaIsaBase {
  using Decoder = rocjitsu::cdna5::Decoder;
  using MachineInst = rocjitsu::cdna5::MachineInst;
  using OperandType = rocjitsu::cdna5::OperandType;
  using StatusReg = rocjitsu::cdna5::StatusReg;

  static constexpr uint32_t WF_SIZE_MAX = 32; ///< gfx1250 is Wave32-only.
  static constexpr uint32_t MAX_WF_SLOTS = 64;
  static constexpr bool SRAM_ECC = true; ///< gfx1250 enables SRAM ECC.
  // CDNA5 MODE[27] is DISABLE_PERF. High-bank VGPR selection is controlled
  // independently by the documented VGPR-MSB fields.
  static constexpr bool MODE_HAS_GPR_IDX_EN = false;
  static constexpr uint32_t MAX_ADDRESSABLE_VGPRS_PER_WF =
      isa_properties(ROCJITSU_CODE_ARCH_CDNA5).max_addressable_vgprs_per_wf;

  static std::optional<uint32_t> resolved_vgpr_offset(OperandType opr_type, int ev);
  static std::optional<uint32_t> resolved_vgpr_offset(const amdgpu::Wavefront &wf,
                                                      OperandType opr_type, int ev,
                                                      amdgpu::VgprMsbRole role);
  static bool simd_capable_value(OperandType opr_type, int ev);
  static uint32_t simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type, int ev);
};

} // namespace rocjitsu::cdna5

namespace rocjitsu {

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA5> {
  using type = cdna5::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ISA_H_
