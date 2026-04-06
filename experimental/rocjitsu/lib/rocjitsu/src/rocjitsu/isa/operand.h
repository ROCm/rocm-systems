// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file operand.h
/// @brief Instruction operand base class with register read/write interface.

#ifndef ROCJITSU_ISA_OPERAND_H_
#define ROCJITSU_ISA_OPERAND_H_

#include <cstdint>
#include <string>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
} // namespace amdgpu

/// @brief Base class for an instruction operand with value resolution.
///
/// Subclasses implement read/write methods using ISA-specific encoding value
/// ranges and the wavefront's register state.
class Operand {
public:
  Operand() = default;

  /// @brief Construct an operand with the given size and encoding value.
  /// @param size_bits Operand width in bits.
  /// @param encoding_value ISA-specific encoding value identifying the register or literal.
  Operand(int size_bits, int encoding_value)
      : size_bits_(size_bits), encoding_value_(encoding_value) {}
  virtual ~Operand() = default;

  /// @brief Human-readable name for this operand (e.g. "v0", "s4", or a literal).
  /// @returns Operand name string.
  virtual std::string name() const { return std::to_string(encoding_value_); }

  /// @brief Read this operand as a scalar 32-bit value.
  /// @param wf Wavefront providing register state.
  /// @returns The 32-bit scalar value.
  virtual uint32_t read_scalar(const amdgpu::Wavefront &wf) const;

  /// @brief Read this operand's value for a specific SIMD lane.
  ///
  /// @details For scalar operands, broadcasts the scalar value to all lanes.
  /// For vector operands, reads the lane from the vector register.
  /// @param wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @returns The 32-bit lane value.
  virtual uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const;

  /// @brief Write a scalar 32-bit value to this operand's destination.
  /// @param[in,out] wf Wavefront providing register state.
  /// @param val Value to write.
  virtual void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const;

  /// @brief Write a 32-bit value to a specific SIMD lane of this operand.
  /// @param[in,out] wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @param val Value to write.
  virtual void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const;

  /// @brief Read a 64-bit value from a SIMD lane (VGPR pair).
  /// @param wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @returns The 64-bit lane value.
  virtual uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const;

  /// @brief Write a 64-bit value to a SIMD lane (VGPR pair).
  /// @param[in,out] wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @param val Value to write.
  virtual void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const;

  /// @brief Read this operand as a 64-bit scalar (e.g., SGPR pair, VCC, EXEC).
  /// @param wf Wavefront providing register state.
  /// @returns The 64-bit scalar value.
  virtual uint64_t read_scalar64(const amdgpu::Wavefront &wf) const;

  /// @brief Write a 64-bit scalar value (e.g., SGPR pair, VCC, EXEC).
  /// @param[in,out] wf Wavefront providing register state.
  /// @param val Value to write.
  virtual void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const;

  int size_bits_ = 0;
  int encoding_value_ = 0;
};

/// @brief ISA-parameterized operand that adds an ISA-specific operand type tag.
/// @tparam Isa ISA traits type providing an OperandType enum or type alias.
template <typename Isa> class IsaOperand : public Operand {
public:
  IsaOperand() = default;

  /// @brief Construct an ISA operand with size, type, and encoding value.
  /// @param size_bits Operand width in bits.
  /// @param opr_type ISA-specific operand type (e.g. SGPR, VGPR, literal).
  /// @param encoding_value ISA-specific encoding value identifying the register or literal.
  IsaOperand(int size_bits, typename Isa::OperandType opr_type, int encoding_value = 0)
      : Operand(size_bits, encoding_value), opr_type_(opr_type) {}

  /// @brief ISA-specific operand type tag.
  typename Isa::OperandType opr_type_{};
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_OPERAND_H_
