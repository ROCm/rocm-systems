// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Simulator-neutral event and finding types used by the data hazard engine.
// Simulator adapters translate native callback payloads into these types before
// entering the core hazard logic.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hash_utils.h"
#include "types.h"

namespace hazard_core {

enum class ResourceKind {
  Unknown,
  ScalarRegister,
  VectorRegister,
  LocalMemory,
  ScratchMemory,
  GlobalMemory,
  MemoryRegister,
};

enum class RegisterKind {
  None,
  Scalar,
  Vector,
  Memory,
};

struct WaitCounterClear {
  WaitCntType kind = WaitCntType::NONE;
  uint32_t keep_count = 0;
};

struct WaitAction {
  std::vector<WaitCounterClear> counters;
  bool waits_for_idle = false;
  bool is_workgroup_barrier = false;
  bool is_address_translation = false;
  bool is_wait_instruction = false;
};

inline bool wait_action_has_effect(const WaitAction &action) {
  return action.is_wait_instruction || action.waits_for_idle || action.is_workgroup_barrier ||
         action.is_address_translation || !action.counters.empty();
}

struct InstructionDescriptor {
  EntityId instruction_id = 0;
  ExecutionKey execution;
  uint64_t pc = 0;
  std::array<uint32_t, 4> raw_isa{};
};

struct InstructionHazardSemantics {
  // Wait counter that guards async writes to VGPR destinations, e.g. VMEM
  // loads or DS reads writing VGPRs. NONE means vector writes from this
  // instruction are not tracked as async destinations. Adapters must set
  // this from frontend-native instruction semantics before register write
  // events arrive; the engine does not infer it from memory resource access
  // ordering.
  WaitCntType vector_write_wait = WaitCntType::NONE;

  // Wait counter that guards explicit async VGPR reads whose source register
  // must remain live until the operation completes. Ordinary VMEM store data
  // sources are captured at issue on AMD GFX and should leave this as NONE.
  WaitCntType vector_read_wait = WaitCntType::NONE;

  // Flat instructions can require both VMEM and LDS waits before a VGPR
  // destination is safe. The primary wait is vector_write_wait; this flag
  // adds the DScnt side.
  bool vector_write_also_waits_lds = false;

  // Wait counter that guards async writes to SGPR destinations.
  WaitCntType scalar_write_wait = WaitCntType::NONE;

  // Wait counter that guards async writes to local memory. LDS covers DS/flat
  // LDS writes; TENSOR covers tensor-load-to-LDS style operations.
  WaitCntType local_write_wait = WaitCntType::NONE;
};

struct ResourceHazardSemantics {
  // Wait counter that guards an async write represented by this resource
  // access. For vector/scalar registers this tracks async destination
  // registers; for local memory this tracks async LDS/tensor writes.
  WaitCntType write_wait = WaitCntType::NONE;

  // Wait counter that guards an async read/source live range represented by
  // this resource access. This is currently meaningful for vector register
  // reads that must remain live until a later wait drains the operation.
  WaitCntType read_wait = WaitCntType::NONE;

  // Flat vector destinations can be guarded by both the primary write_wait
  // and an LDS/DScnt side. Frontends that know this per access can set it
  // here instead of encoding the fact in instruction-wide state.
  bool write_also_waits_lds = false;
};

struct InstructionEvent {
  InstructionDescriptor instruction;
  WaitAction wait_action;
  InstructionHazardSemantics hazards;
};

struct ResourceAccessEvent {
  InstructionDescriptor instruction;
  ResourceKind resource_kind = ResourceKind::Unknown;
  RegisterKind register_kind = RegisterKind::None;
  uint64_t address = 0;
  uint32_t resource_index = 0;
  uint32_t size_bytes = 0;
  bool is_atomic = false;
  bool is_read = false;
  bool is_write = false;
  uint64_t exec_mask = 0;
  ResourceHazardSemantics hazards;
};

enum class BarrierKind {
  Workgroup,
  Cluster,
  LocalMemoryAtomic,
  LocalMemoryAtomicAsync,
  Unknown,
};

struct BarrierEvent {
  ExecutionKey wave;
  EntityId barrier_id = 0;
  BarrierKind kind = BarrierKind::Unknown;
};

enum class HazardKind {
  RAW,
  WAR,
  WAW,
  LocalMemoryRace,
  GlobalMemoryRace,
};

enum class HazardAccessKind {
  Read,
  Write,
  Synchronization,
};

struct HazardFinding {
  HazardKind kind = HazardKind::RAW;
  ResourceKind resource_kind = ResourceKind::Unknown;
  HazardAccessKind access_kind = HazardAccessKind::Read;
  InstructionDescriptor instruction;
  InstructionDescriptor source_instruction;
  bool has_source_instruction = false;
  uint64_t address = 0;
  uint32_t resource_index = 0;
  uint32_t size_bytes = 0;
  WaitCntType required_wait = WaitCntType::NONE;
  std::string message_template;
  std::string suggestion_template;
};

} // namespace hazard_core
