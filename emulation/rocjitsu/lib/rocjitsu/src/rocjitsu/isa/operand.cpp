// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/operand.h"
#include "simdojo/components/vector_reg.h"

#include <stdexcept>

namespace rocjitsu {

Result Operand::emit_encoding_error(const util::DiagnosticEmitter &emit_error) const {
  switch (encoding_error_) {
  case EncodingError::InvalidSelector:
    return emit_error.emit() << "invalid operand selector";
  case EncodingError::InvalidScalarRegisterSelector:
    return emit_error.emit() << "invalid scalar register selector";
  case EncodingError::InvalidLaneSelector:
    return emit_error.emit() << "invalid lane selector";
  case EncodingError::InvalidExecSelector:
    return emit_error.emit() << "invalid EXEC selector";
  case EncodingError::InvalidVgprSourceSelector:
    return emit_error.emit() << "invalid VGPR source selector";
  case EncodingError::InvalidScalarSourceSelector:
    return emit_error.emit() << "invalid scalar source selector";
  case EncodingError::None:
    break;
  }
  return emit_error.emit() << "unknown operand encoding failure";
}

std::optional<RegisterRef> Operand::to_register_ref() const { return std::nullopt; }

std::optional<RegClass> Operand::to_special_reg_class() const { return std::nullopt; }

uint32_t Operand::read_scalar(const amdgpu::Wavefront & /*wf*/) const {
  throw std::logic_error("read_scalar not implemented for this operand type");
}

uint32_t Operand::read_lane(const amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/) const {
  throw std::logic_error("read_lane not implemented for this operand type");
}

void Operand::write_scalar(amdgpu::Wavefront & /*wf*/, uint32_t /*val*/) const {
  throw std::logic_error("write_scalar not implemented for this operand type");
}

void Operand::write_lane(amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/, uint32_t /*val*/) const {
  throw std::logic_error("write_lane not implemented for this operand type");
}

uint64_t Operand::read_lane64(const amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/) const {
  throw std::logic_error("read_lane64 not implemented for this operand type");
}

void Operand::write_lane64(amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/, uint64_t /*val*/) const {
  throw std::logic_error("write_lane64 not implemented for this operand type");
}

uint64_t Operand::read_scalar64(const amdgpu::Wavefront & /*wf*/) const {
  throw std::logic_error("read_scalar64 not implemented for this operand type");
}

void Operand::write_scalar64(amdgpu::Wavefront & /*wf*/, uint64_t /*val*/) const {
  throw std::logic_error("write_scalar64 not implemented for this operand type");
}

} // namespace rocjitsu
