// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/operand.h"

#include <stdexcept>

namespace rocjitsu {

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
