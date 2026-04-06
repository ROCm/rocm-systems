// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_

#include "rocjitsu/isa/arch/amdgpu/cdna3/decoder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/operand_types.h"
#include "rocjitsu/isa/isa_traits.h"
#include "util/bitfield.h"

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu {
namespace cdna3 {

/// @brief CDNA3 MODE register layout (one 32-bit scalar register per wavefront).
/// @details Bit positions follow the "Shader Processor Input (SPI)" section of
/// the CDNA3 ISA specification. Each accessor returns a lightweight proxy that
/// supports read (implicit integral conversion) and write (operator=).
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
  auto PERF_EN() { return member<19, 19>(); }
  auto PERF_EN() const { return member<19, 19>(); }
  auto COND_DBG_USER() { return member<20, 20>(); }
  auto COND_DBG_USER() const { return member<20, 20>(); }
  auto COND_DBG_SYS() { return member<21, 21>(); }
  auto COND_DBG_SYS() const { return member<21, 21>(); }
  auto ALLOW_REPLAY() { return member<22, 22>(); }
  auto ALLOW_REPLAY() const { return member<22, 22>(); }
};

struct Isa {
  static constexpr uint32_t WF_SIZE = 64;           ///< Lanes per wavefront (SIMD64).
  static constexpr uint32_t MAX_SGPRS_PER_WF = 104; ///< Max scalar GPRs per wavefront.
  static constexpr uint32_t MAX_VGPRS_PER_WF = 256; ///< Max vector GPRs per wavefront.

  using Context = amdgpu::Wavefront;
  using Decoder = cdna3::Decoder;
  using MachineInst = cdna3::MachineInst;
  using OperandType = cdna3::OperandType;
  using StatusReg = cdna3::StatusReg;
};

} // namespace cdna3

template <> struct IsaTrait<ROCJITSU_CODE_ARCH_CDNA3> {
  using type = cdna3::Isa;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA3_ISA_H_
