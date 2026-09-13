// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/operand.h"
#include "simdojo/components/vector_reg.h"

#include <stdexcept>

namespace rocjitsu {

constinit const Operand::ExecutionBackend Operand::default_execution_backend_{
    .simd_capable = [](const Operand &op) -> bool { return op.simd_capable_fallback(); },
    .read_lane_chunk = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane_base,
                          uint32_t count, uint32_t *out) -> void {
      return op.read_lane_chunk_fallback(wf, lane_base, count, out);
    },
    .write_lane_chunk = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane_base,
                           uint32_t count, const uint32_t *vals, uint64_t mask) -> void {
      return op.write_lane_chunk_fallback(wf, lane_base, count, vals, mask);
    },
    .read_scalar = [](const Operand &op, const amdgpu::Wavefront &wf) -> uint32_t {
      return op.read_scalar_fallback(wf);
    },
    .read_lane = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane) -> uint32_t {
      return op.read_lane_fallback(wf, lane);
    },
    .write_scalar = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t val) -> void {
      return op.write_scalar_fallback(wf, val);
    },
    .write_lane = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane,
                     uint32_t val) -> void { return op.write_lane_fallback(wf, lane, val); },
    .read_lane64 = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane) -> uint64_t {
      return op.read_lane64_fallback(wf, lane);
    },
    .write_lane64 = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane,
                       uint64_t val) -> void { return op.write_lane64_fallback(wf, lane, val); },
    .read_scalar64 = [](const Operand &op, const amdgpu::Wavefront &wf) -> uint64_t {
      return op.read_scalar64_fallback(wf);
    },
    .write_scalar64 = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t val) -> void {
      return op.write_scalar64_fallback(wf, val);
    },
    .simd_vgpr_base = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> std::optional<uint32_t> { return op.simd_vgpr_base_impl(wf); },
    .simd_vgpr_base_mut = [](const Operand &op, amdgpu::Wavefront &wf) -> std::optional<uint32_t> {
      return op.simd_vgpr_base_mut_impl(wf);
    },
    .simd_vgpr_storage = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> amdgpu::ConstVgprStorage { return op.simd_vgpr_storage_impl(wf); },
    .simd_vgpr_storage_mut = [](const Operand &op, amdgpu::Wavefront &wf) -> amdgpu::VgprStorage {
      return op.simd_vgpr_storage_mut_impl(wf);
    },
    .simd_vgpr_storage64 = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> amdgpu::ConstVgprStoragePair64 { return op.simd_vgpr_storage64_impl(wf); },
    .simd_vgpr_storage64_mut = [](const Operand &op, amdgpu::Wavefront &wf)
        -> amdgpu::VgprStoragePair64 { return op.simd_vgpr_storage64_mut_impl(wf); },
    .simd_notify_read = [](const Operand &op, const amdgpu::Wavefront &wf, uint64_t lane_mask,
                           uint8_t byte_mask) -> void {
      return op.simd_notify_read_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                               uint8_t byte_mask) -> void {
      return op.simd_notify_read_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read64 = [](const Operand &op, const amdgpu::Wavefront &wf, uint64_t lane_mask,
                             uint8_t byte_mask) -> void {
      return op.simd_notify_read64_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read64_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                 uint8_t byte_mask) -> void {
      return op.simd_notify_read64_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_write_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                uint8_t byte_mask) -> void {
      return op.simd_notify_write_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_write64_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                  uint8_t byte_mask) -> void {
      return op.simd_notify_write64_mut_impl(wf, lane_mask, byte_mask);
    },
};

constinit const Operand::ExecutionBackend Operand::model_execution_backend_{
    .simd_capable = [](const Operand &op) -> bool {
      (void)op;
      return false;
    },
    .read_lane_chunk = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane_base,
                          uint32_t count, uint32_t *out) -> void {
      (void)op;
      (void)wf;
      (void)lane_base;
      (void)count;
      (void)out;
      throw std::logic_error("operand execution backend is not linked");
    },
    .write_lane_chunk = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane_base,
                           uint32_t count, const uint32_t *vals, uint64_t mask) -> void {
      (void)op;
      (void)wf;
      (void)lane_base;
      (void)count;
      (void)vals;
      (void)mask;
      throw std::logic_error("operand execution backend is not linked");
    },
    .read_scalar = [](const Operand &op, const amdgpu::Wavefront &wf) -> uint32_t {
      (void)op;
      (void)wf;
      throw std::logic_error("operand execution backend is not linked");
    },
    .read_lane = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane) -> uint32_t {
      (void)op;
      (void)wf;
      (void)lane;
      throw std::logic_error("operand execution backend is not linked");
    },
    .write_scalar = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t val) -> void {
      (void)op;
      (void)wf;
      (void)val;
      throw std::logic_error("operand execution backend is not linked");
    },
    .write_lane = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane,
                     uint32_t val) -> void {
      (void)op;
      (void)wf;
      (void)lane;
      (void)val;
      throw std::logic_error("operand execution backend is not linked");
    },
    .read_lane64 = [](const Operand &op, const amdgpu::Wavefront &wf, uint32_t lane) -> uint64_t {
      (void)op;
      (void)wf;
      (void)lane;
      throw std::logic_error("operand execution backend is not linked");
    },
    .write_lane64 = [](const Operand &op, amdgpu::Wavefront &wf, uint32_t lane,
                       uint64_t val) -> void {
      (void)op;
      (void)wf;
      (void)lane;
      (void)val;
      throw std::logic_error("operand execution backend is not linked");
    },
    .read_scalar64 = [](const Operand &op, const amdgpu::Wavefront &wf) -> uint64_t {
      (void)op;
      (void)wf;
      throw std::logic_error("operand execution backend is not linked");
    },
    .write_scalar64 = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t val) -> void {
      (void)op;
      (void)wf;
      (void)val;
      throw std::logic_error("operand execution backend is not linked");
    },
    .simd_vgpr_base = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> std::optional<uint32_t> { return op.simd_vgpr_base_impl(wf); },
    .simd_vgpr_base_mut = [](const Operand &op, amdgpu::Wavefront &wf) -> std::optional<uint32_t> {
      return op.simd_vgpr_base_mut_impl(wf);
    },
    .simd_vgpr_storage = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> amdgpu::ConstVgprStorage { return op.simd_vgpr_storage_impl(wf); },
    .simd_vgpr_storage_mut = [](const Operand &op, amdgpu::Wavefront &wf) -> amdgpu::VgprStorage {
      return op.simd_vgpr_storage_mut_impl(wf);
    },
    .simd_vgpr_storage64 = [](const Operand &op, const amdgpu::Wavefront &wf)
        -> amdgpu::ConstVgprStoragePair64 { return op.simd_vgpr_storage64_impl(wf); },
    .simd_vgpr_storage64_mut = [](const Operand &op, amdgpu::Wavefront &wf)
        -> amdgpu::VgprStoragePair64 { return op.simd_vgpr_storage64_mut_impl(wf); },
    .simd_notify_read = [](const Operand &op, const amdgpu::Wavefront &wf, uint64_t lane_mask,
                           uint8_t byte_mask) -> void {
      return op.simd_notify_read_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                               uint8_t byte_mask) -> void {
      return op.simd_notify_read_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read64 = [](const Operand &op, const amdgpu::Wavefront &wf, uint64_t lane_mask,
                             uint8_t byte_mask) -> void {
      return op.simd_notify_read64_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_read64_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                 uint8_t byte_mask) -> void {
      return op.simd_notify_read64_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_write_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                uint8_t byte_mask) -> void {
      return op.simd_notify_write_mut_impl(wf, lane_mask, byte_mask);
    },
    .simd_notify_write64_mut = [](const Operand &op, amdgpu::Wavefront &wf, uint64_t lane_mask,
                                  uint8_t byte_mask) -> void {
      return op.simd_notify_write64_mut_impl(wf, lane_mask, byte_mask);
    },
};

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

uint32_t Operand::read_scalar_fallback(const amdgpu::Wavefront & /*wf*/) const {
  throw std::logic_error("read_scalar not implemented for this operand type");
}

uint32_t Operand::read_lane_fallback(const amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/) const {
  throw std::logic_error("read_lane not implemented for this operand type");
}

void Operand::write_scalar_fallback(amdgpu::Wavefront & /*wf*/, uint32_t /*val*/) const {
  throw std::logic_error("write_scalar not implemented for this operand type");
}

void Operand::write_lane_fallback(amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/,
                                  uint32_t /*val*/) const {
  throw std::logic_error("write_lane not implemented for this operand type");
}

uint64_t Operand::read_lane64_fallback(const amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/) const {
  throw std::logic_error("read_lane64 not implemented for this operand type");
}

void Operand::write_lane64_fallback(amdgpu::Wavefront & /*wf*/, uint32_t /*lane*/,
                                    uint64_t /*val*/) const {
  throw std::logic_error("write_lane64 not implemented for this operand type");
}

uint64_t Operand::read_scalar64_fallback(const amdgpu::Wavefront & /*wf*/) const {
  throw std::logic_error("read_scalar64 not implemented for this operand type");
}

void Operand::write_scalar64_fallback(amdgpu::Wavefront & /*wf*/, uint64_t /*val*/) const {
  throw std::logic_error("write_scalar64 not implemented for this operand type");
}

} // namespace rocjitsu
