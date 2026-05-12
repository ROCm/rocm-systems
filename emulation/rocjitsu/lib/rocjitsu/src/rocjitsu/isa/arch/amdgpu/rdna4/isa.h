// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include <cstdint>

namespace rocjitsu {
namespace rdna4 {

/// @brief RDNA4 STATUS register layout (GFX12, one 32-bit scalar register per wavefront).
///
/// @details Same layout as RDNA3 (GFX11); no TTRACE_EN, EXPORT_RDY, or
/// SKIP_EXPORT. Kept per-ISA for consistency with the other RDNA ISAs.
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

/// @brief RDNA4 ISA traits (GFX12, Wave32 default, Wave64 capable, split S_WAIT_*).
///
/// @details Inherits all defaults from `amdgpu::RdnaIsaBase` including
/// `WAITCNT_LGKMCNT_MASK = 0` — RDNA4 uses split S_WAIT_LOADCNT /
/// S_WAIT_DSCNT / S_WAIT_KMCNT instructions; there is no monolithic S_WAITCNT.
struct Isa : amdgpu::RdnaIsaBase {
  using Decoder = rdna4::Decoder;
  using MachineInst = rdna4::MachineInst;
  using OperandType = rdna4::OperandType;
  using StatusReg = rdna4::StatusReg;
};

} // namespace rdna4

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_RDNA4> {
  using type = rdna4::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ISA_H_
