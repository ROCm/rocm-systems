// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/operand_types.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu {
namespace rdna3_5 {

/// @brief RDNA3.5 STATUS register layout (GFX11.5, one 32-bit scalar register per wavefront).
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

struct Isa {
  static constexpr uint32_t WF_SIZE = 32;             ///< Default wave size (Wave32).
  static constexpr uint32_t WF_SIZE_MAX = 64;         ///< Max wave size (Wave64 supported).
  static constexpr uint32_t MAX_SGPRS_PER_WF = 106;   ///< Max scalar GPRs per wavefront.
  static constexpr uint32_t MAX_VGPRS_PER_WF = 256;   ///< Max vector GPRs per wavefront.
  static constexpr uint32_t MAX_ACC_VGPRS_PER_WF = 0; ///< No AccVGPRs in RDNA.
  static constexpr uint8_t WAITCNT_LGKMCNT_MASK =
      0x3F; ///< lgkmcnt field mask in S_WAITCNT (6-bit at [9:4] in GFX11 layout).

  using Context = amdgpu::Wavefront;
  using Decoder = rdna3_5::Decoder;
  using MachineInst = rdna3_5::MachineInst;
  using OperandType = rdna3_5::OperandType;
  using StatusReg = rdna3_5::StatusReg;
};

} // namespace rdna3_5

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_RDNA3_5> {
  using type = rdna3_5::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ISA_H_
