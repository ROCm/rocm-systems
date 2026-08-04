// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/operand.h"
#include "simdojo/components/vector_reg.h"

#include <memory>
#include <stdexcept>

namespace rocjitsu {

std::optional<RegisterRef> Operand::to_register_ref() const { return std::nullopt; }

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

struct StagedOperand::Storage {
  amdgpu::VgprStorage lo{};
  amdgpu::VgprStorage hi{};
};

StagedOperand::StagedOperand() : storage_(std::make_unique<Storage>()) {}

StagedOperand::~StagedOperand() = default;

StagedOperand::StagedOperand(const Operand &base, const uint32_t *data, int lane_count)
    : Operand(base.size_bits_, base.encoding_value_), storage_(std::make_unique<Storage>()),
      lane_count_(lane_count) {
  for (int i = 0; i < lane_count && i < MAX_LANES; ++i)
    storage_->lo[i] = data[i];
}

StagedOperand::StagedOperand(const Operand &base, const uint64_t *data, int lane_count)
    : Operand(base.size_bits_, base.encoding_value_), storage_(std::make_unique<Storage>()),
      lane_count_(lane_count) {
  for (int i = 0; i < lane_count && i < MAX_LANES; ++i) {
    storage_->lo[i] = static_cast<uint32_t>(data[i]);
    storage_->hi[i] = static_cast<uint32_t>(data[i] >> 32);
  }
}

uint32_t StagedOperand::read_lane(const amdgpu::Wavefront & /*wf*/, uint32_t lane) const {
  return lane < static_cast<uint32_t>(lane_count_) ? storage_->lo[lane] : 0;
}

uint64_t StagedOperand::read_lane64(const amdgpu::Wavefront & /*wf*/, uint32_t lane) const {
  if (lane >= static_cast<uint32_t>(lane_count_))
    return 0;
  return uint64_t{storage_->lo[lane]} | (uint64_t{storage_->hi[lane]} << 32);
}

uint32_t StagedOperand::read_scalar(const amdgpu::Wavefront & /*wf*/) const {
  return storage_->lo[0];
}

uint64_t StagedOperand::read_scalar64(const amdgpu::Wavefront & /*wf*/) const {
  return uint64_t{storage_->lo[0]} | (uint64_t{storage_->hi[0]} << 32);
}

void StagedOperand::read_lane_chunk(const amdgpu::Wavefront & /*wf*/, uint32_t lane_base,
                                    uint32_t count, uint32_t *out) const {
  uint32_t lanes = static_cast<uint32_t>(lane_count_);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t lane = lane_base + i;
    out[i] = lane < lanes ? storage_->lo[lane] : 0u;
  }
}

const amdgpu::VgprStorage *
StagedOperand::simd_vgpr_storage_impl(const amdgpu::Wavefront & /*wf*/) const {
  return &storage_->lo;
}

amdgpu::ConstVgprStoragePair64
StagedOperand::simd_vgpr_storage64_impl(const amdgpu::Wavefront & /*wf*/) const {
  return {&storage_->lo, &storage_->hi};
}

} // namespace rocjitsu
