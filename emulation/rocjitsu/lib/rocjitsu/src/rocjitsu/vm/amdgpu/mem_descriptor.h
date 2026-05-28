// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_MEM_DESCRIPTOR_H_
#define ROCJITSU_VM_AMDGPU_MEM_DESCRIPTOR_H_

/// @file Clean accessor for a memory instruction's access shape.
///
/// Hides the inst.data() -> ScalarMemState/VectorMemState tag dispatch +
/// downcast behind one function, exposing exactly the fields a timing model
/// consumes (per-lane addresses, active-lane mask, element bytes, Mtype,
/// non-temporal hint). Lives in the amdgpu layer because it depends on the
/// amdgpu-specific mem-state types; Instruction itself stays ISA-generic.

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu {
namespace amdgpu {

/// rocjitsu-free view of one memory instruction's access shape. The
/// per_lane_addr span points INTO the live DynamicInstState owned by
/// inst.data(); it is valid only while that instruction (and its attached
/// state) outlives the descriptor. Never points at the descriptor itself.
struct MemDescriptor {
  std::span<const uint64_t> per_lane_addr;  // 1 entry (scalar) .. 64 (vector)
  uint64_t lane_mask = 0;                   // active-lane bitmap
  uint32_t elem_bytes = 0;                  // bytes touched per active lane
  uint8_t mtype = static_cast<uint8_t>(Mtype::RW);
  bool non_temporal = false;
};

/// Decode a memory instruction's access shape from its attached pipeline
/// state. Returns nullopt for a null / non-memory / unknown-tag state.
inline std::optional<MemDescriptor> mem_descriptor(const DynamicInstState *d) {
  if (!d) return std::nullopt;
  switch (d->tag()) {
    case SCALAR_MEM: {
      const auto *s = static_cast<const ScalarMemState *>(d);
      MemDescriptor m;
      m.per_lane_addr = std::span<const uint64_t>(&s->addr, 1);
      m.lane_mask = 1;
      m.elem_bytes = s->num_dwords * 4;
      m.mtype = static_cast<uint8_t>(s->mtype);
      m.non_temporal = false;
      return m;
    }
    case GLOBAL_MEM:
    case LOCAL_MEM: {
      const auto *v = static_cast<const VectorMemState *>(d);
      MemDescriptor m;
      m.per_lane_addr =
          std::span<const uint64_t>(v->per_lane_addr.data(), v->per_lane_addr.size());
      m.lane_mask = v->lane_mask;
      m.elem_bytes = v->elem_size * v->num_elems;
      m.mtype = static_cast<uint8_t>(v->mtype);
      m.non_temporal = v->non_temporal;
      return m;
    }
    default:
      return std::nullopt;
  }
}

/// Convenience overload reading the instruction's attached state.
inline std::optional<MemDescriptor> mem_descriptor(const Instruction &inst) {
  return mem_descriptor(inst.data());
}

}  // namespace amdgpu
}  // namespace rocjitsu

#endif  // ROCJITSU_VM_AMDGPU_MEM_DESCRIPTOR_H_
