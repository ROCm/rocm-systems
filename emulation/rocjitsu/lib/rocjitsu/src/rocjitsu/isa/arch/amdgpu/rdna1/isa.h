// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/rdna1/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include <cstdint>

namespace rocjitsu {
namespace rdna1 {

/// @brief RDNA1 STATUS register layout (GFX10.1, one 32-bit scalar register per wavefront).
///
/// @details RDNA1/2 expose TTRACE_EN, EXPORT_RDY, and SKIP_EXPORT bits that
/// are absent from RDNA3/4; kept per-ISA because of this layout difference.
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
  auto TTRACE_EN() { return member<7, 7>(); }
  auto TTRACE_EN() const { return member<7, 7>(); }
  auto EXPORT_RDY() { return member<8, 8>(); }
  auto EXPORT_RDY() const { return member<8, 8>(); }
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
  auto SKIP_EXPORT() { return member<15, 15>(); }
  auto SKIP_EXPORT() const { return member<15, 15>(); }
  auto VALID() { return member<16, 16>(); }
  auto VALID() const { return member<16, 16>(); }
  auto ECC_ERR() { return member<17, 17>(); }
  auto ECC_ERR() const { return member<17, 17>(); }
  auto PERF_EN() { return member<19, 19>(); }
  auto PERF_EN() const { return member<19, 19>(); }
  auto ALLOW_REPLAY() { return member<22, 22>(); }
  auto ALLOW_REPLAY() const { return member<22, 22>(); }
};

/// @brief RDNA1 ISA traits (GFX10.1, Wave32 default, Wave64 capable, monolithic S_WAITCNT).
///
/// @details Overrides `WAITCNT_LGKMCNT_MASK = 0x3F` (RDNA1 uses a 6-bit
/// lgkmcnt field at bits [13:8] in S_WAITCNT).
/// All other constants inherit from `amdgpu::RdnaIsaBase`.
struct Isa : amdgpu::RdnaIsaBase {
  static constexpr uint8_t WAITCNT_LGKMCNT_MASK = 0x3F; ///< lgkmcnt mask in S_WAITCNT [13:8].

  using Decoder = rdna1::Decoder;
  using MachineInst = rdna1::MachineInst;
  using OperandType = rdna1::OperandType;
  using StatusReg = rdna1::StatusReg;
};

} // namespace rdna1

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_RDNA1> {
  using type = rdna1::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA1_ISA_H_
