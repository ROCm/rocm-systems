// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
}
namespace rdna3 {

/// @brief RDNA3 STATUS register layout (GFX11, one 32-bit scalar register per wavefront).
///
/// @details RDNA3/4 do not expose TTRACE_EN, EXPORT_RDY, or SKIP_EXPORT
/// (those bits were RDNA1/2 only); kept per-ISA for consistency with RDNA1/2.
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

/// @brief RDNA3 ISA traits (GFX11, Wave32 default, Wave64 capable, monolithic + split waits).
///
/// @details RDNA3 retains S_WAITCNT (with a new SIMM16 bit layout: expcnt[2:0],
/// lgkmcnt[9:4], vmcnt[15:10]) AND adds named per-counter S_WAITCNT_VMCNT etc.
/// Overrides `WAITCNT_LGKMCNT_MASK = 0x3F` (6-bit lgkmcnt at bits [9:4]).
struct Isa : amdgpu::RdnaIsaBase {
  static constexpr uint8_t WAITCNT_LGKMCNT_MASK = 0x3F; ///< lgkmcnt mask in S_WAITCNT [9:4].

  using Decoder = rdna3::Decoder;
  using MachineInst = rdna3::MachineInst;
  using OperandType = rdna3::OperandType;
  using StatusReg = rdna3::StatusReg;

  // SIMD fast-path traits — consumed by AmdgpuIsaOperand<Isa> in
  // rocjitsu/isa/isa_operand_simd_inl.h. Definitions live in this arch's
  // operand.cpp; bodies forward to the anonymous-namespace helpers.
  static std::optional<uint32_t> resolved_vgpr_offset(OperandType opr_type, int ev);
  static bool simd_capable_value(OperandType opr_type, int ev);
  static uint32_t simd_broadcast_value(const amdgpu::Wavefront &wf, OperandType opr_type, int ev);
};

} // namespace rdna3

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_RDNA3> {
  using type = rdna3::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_ISA_H_
