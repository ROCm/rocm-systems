// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Frontend-agnostic resource access helpers for hazard detection.
//
// These helpers operate only on hazard_core::WaveState and simple scalar
// values. Frontend adapters can use them directly, and higher-level engines can
// use them as the shared implementation behind generic event handling.

#pragma once

#include "types.h"
#include "wait_handling.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>

namespace hazard_core {

constexpr EntityId NO_INSTRUCTION_FILTER = std::numeric_limits<EntityId>::max();

struct PendingRegisterHazard {
  uint32_t register_index = 0;
  const PendingAsyncOp *pending = nullptr;
};

inline uint32_t bytes_for_dwords(uint32_t size_dwords) { return size_dwords * BYTES_PER_DWORD; }

inline PendingRegisterHazard
find_pending_register_hazard(const std::unordered_map<uint32_t, PendingAsyncOp> &pending,
                             uint32_t first_reg, uint32_t size_dwords,
                             EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = first_reg + i;
    auto it = pending.find(reg);
    if (it == pending.end())
      continue;
    if (it->second.instruction_id == current_instruction_id)
      continue;
    return {reg, &it->second};
  }
  return {};
}

// A contiguous run of registers inside one access that are all in flight from
// the same producer instruction.
struct PendingRegisterSpan {
  uint32_t first_register = 0;
  uint32_t last_register = 0;
  const PendingAsyncOp *pending = nullptr;
};

// Visit every distinct producer conflicting with [first_reg, first_reg +
// size_dwords), together with the register run it covers.
//
// Callers that hand over a whole wide operand must use this rather than
// find_pending_register_hazard, which stops at the first conflicting register
// and so cannot see a second producer further into the range.
template <typename Fn>
inline void
for_each_pending_register_span(const std::unordered_map<uint32_t, PendingAsyncOp> &pending,
                               uint32_t first_reg, uint32_t size_dwords, uint32_t max_register,
                               EntityId current_instruction_id, Fn visit) {
  PendingRegisterSpan run;

  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = first_reg + i;
    if (reg > max_register)
      break;

    const PendingAsyncOp *op = nullptr;
    if (auto it = pending.find(reg);
        it != pending.end() && it->second.instruction_id != current_instruction_id)
      op = &it->second;

    if (op && run.pending && run.pending->instruction_id == op->instruction_id) {
      run.last_register = reg;
      continue;
    }

    if (run.pending)
      visit(run);
    run = op ? PendingRegisterSpan{reg, reg, op} : PendingRegisterSpan{};
  }

  if (run.pending)
    visit(run);
}

// Last register of the run that starts at first_register and is still in flight
// from producer, bounded by the access and the register file.
//
// Handlers that must stay on a per-register walk report once per producer and
// name the run they stand for, so the registers folded into that one report are
// still visible to the reader.
inline uint32_t pending_span_end(const std::unordered_map<uint32_t, PendingAsyncOp> &pending,
                                 uint32_t first_register, uint32_t last_access_register,
                                 uint32_t max_register, EntityId producer) {
  uint32_t last = first_register;
  for (uint32_t reg = first_register + 1; reg <= last_access_register && reg <= max_register;
       ++reg) {
    auto it = pending.find(reg);
    if (it == pending.end() || it->second.instruction_id != producer)
      break;
    last = reg;
  }
  return last;
}

inline const PendingAsyncOp *
find_pending_register_op(const std::unordered_map<uint32_t, PendingAsyncOp> &pending,
                         uint32_t first_reg, uint32_t size_dwords,
                         EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return find_pending_register_hazard(pending, first_reg, size_dwords, current_instruction_id)
      .pending;
}

// vN and accN are separate architectural namespaces (see isa/register_set.h),
// so every lookup and update names the file it applies to and an operation
// pending on one file is never visible through the other.
enum class VectorRegisterFile {
  Vector,
  AccumVector,
};

// The pending-op maps a vector register file keeps, one per kind of in-flight
// operation a later access can conflict with.
enum class PendingRegisterSet {
  Writes,   // async load destinations, drained by the load's own counter
  DsWrites, // DScnt side of a flat load destination
  Reads,    // async sources that must stay live until the operation drains
};

inline uint32_t max_register_index(VectorRegisterFile file) {
  return file == VectorRegisterFile::AccumVector ? MAX_ACC_VGPR_INDEX : MAX_VGPR_INDEX;
}

// Templated on the wave so a const wave yields a const map: checking only
// reads the state while tracking has to write it.
template <typename Wave>
inline auto &pending_registers(Wave &wave, VectorRegisterFile file, PendingRegisterSet set) {
  const bool accum = file == VectorRegisterFile::AccumVector;
  switch (set) {
  case PendingRegisterSet::DsWrites:
    return accum ? wave.pending_acc_vgpr_writes_ds : wave.pending_vgpr_writes_ds;
  case PendingRegisterSet::Reads:
    return accum ? wave.pending_acc_vgpr_reads : wave.pending_vgpr_reads;
  case PendingRegisterSet::Writes:
    break;
  }
  return accum ? wave.pending_acc_vgpr_writes : wave.pending_vgpr_writes;
}

// The FIFO that drains the matching pending map. Tracking appends to both and
// a wait clears both, so the two always move together.
inline std::deque<std::pair<uint32_t, PendingAsyncOp>> &
pending_register_fifo(WaveState &wave, VectorRegisterFile file, PendingRegisterSet set) {
  const bool accum = file == VectorRegisterFile::AccumVector;
  switch (set) {
  case PendingRegisterSet::DsWrites:
    return accum ? wave.flat_acc_vgpr_ds_fifo : wave.flat_vgpr_ds_fifo;
  case PendingRegisterSet::Reads:
    return accum ? wave.acc_vmem_store_fifo : wave.vmem_store_fifo;
  case PendingRegisterSet::Writes:
    break;
  }
  return accum ? wave.acc_vmem_load_fifo : wave.vmem_load_fifo;
}

/// Search one pending set of one vector register file for a conflict with
/// [index, index + size_dwords).
inline PendingRegisterHazard
check_vector_hazard_info(const WaveState *wave, VectorRegisterFile file, PendingRegisterSet set,
                         uint32_t index, uint32_t size_dwords,
                         EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};
  return find_pending_register_hazard(pending_registers(*wave, file, set), index, size_dwords,
                                      current_instruction_id);
}

/// A destination stays unsafe while either counter side of a flat load is
/// outstanding, so a WAW check spans both write sets.
inline PendingRegisterHazard
check_vector_waw_hazard_info(const WaveState *wave, VectorRegisterFile file, uint32_t index,
                             uint32_t size_dwords,
                             EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  const auto hazard = check_vector_hazard_info(wave, file, PendingRegisterSet::Writes, index,
                                               size_dwords, current_instruction_id);
  if (hazard.pending)
    return hazard;
  return check_vector_hazard_info(wave, file, PendingRegisterSet::DsWrites, index, size_dwords,
                                  current_instruction_id);
}

inline PendingRegisterHazard
check_sgpr_read_hazard_info(const WaveState *wave, uint32_t sgpr_index, uint32_t size_dwords = 1,
                            EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};
  return find_pending_register_hazard(wave->pending_sgpr_writes, sgpr_index, size_dwords,
                                      current_instruction_id);
}

inline const PendingAsyncOp *
check_sgpr_read_hazard(const WaveState *wave, uint32_t sgpr_index, uint32_t size_dwords = 1,
                       EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_sgpr_read_hazard_info(wave, sgpr_index, size_dwords, current_instruction_id).pending;
}

inline PendingRegisterHazard
check_sgpr_waw_hazard_info(const WaveState *wave, uint32_t sgpr_index, uint32_t size_dwords = 1,
                           EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_sgpr_read_hazard_info(wave, sgpr_index, size_dwords, current_instruction_id);
}

inline const PendingAsyncOp *
check_sgpr_waw_hazard(const WaveState *wave, uint32_t sgpr_index, uint32_t size_dwords = 1,
                      EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_sgpr_waw_hazard_info(wave, sgpr_index, size_dwords, current_instruction_id).pending;
}

inline const PendingAsyncOp *check_lds_read_hazard(const WaveState *wave, uint32_t address,
                                                   uint32_t size_bytes) {
  return find_pending_lds_write(wave, address, size_bytes);
}

inline const PendingAsyncOp *check_tensor_lds_hazard(const WaveState *wave, uint32_t address,
                                                     uint32_t size_bytes) {
  return find_pending_tensor_lds_write(wave, address, size_bytes);
}

inline const PendingAsyncOp *check_lds_write_hazard(const WaveState *wave, uint32_t address,
                                                    uint32_t size_bytes) {
  return find_pending_lds_read(wave, address, size_bytes);
}

template <typename Fn> inline void for_each_pending_op(const WaveState *wave, Fn visit) {
  if (!wave)
    return;

  for (const auto &[_, op] : wave->pending_vgpr_writes)
    visit(op);
  for (const auto &[_, op] : wave->pending_acc_vgpr_writes)
    visit(op);
  for (const auto &[_, op] : wave->pending_sgpr_writes)
    visit(op);
  for (const auto &[_, op] : wave->pending_vgpr_writes_ds)
    visit(op);
  for (const auto &[_, op] : wave->pending_acc_vgpr_writes_ds)
    visit(op);
  for (const auto &[_, op] : wave->pending_vgpr_reads)
    visit(op);
  for (const auto &[_, op] : wave->pending_acc_vgpr_reads)
    visit(op);

  for (const auto &entry : wave->vmem_load_fifo)
    visit(entry.second);
  for (const auto &entry : wave->acc_vmem_load_fifo)
    visit(entry.second);
  for (const auto &entry : wave->smem_load_fifo)
    visit(entry.second);
  for (const auto &entry : wave->lds_fifo)
    visit(entry.second);
  for (const auto &entry : wave->vmem_store_fifo)
    visit(entry.second);
  for (const auto &entry : wave->acc_vmem_store_fifo)
    visit(entry.second);
  for (const auto &entry : wave->lds_read_fifo)
    visit(entry.second);
  for (const auto &entry : wave->flat_vgpr_ds_fifo)
    visit(entry.second);
  for (const auto &entry : wave->flat_acc_vgpr_ds_fifo)
    visit(entry.second);
  for (const auto &entry : wave->tensor_lds_fifo)
    visit(entry.second);
}

template <typename Fn>
inline void for_each_pending_instruction_id(const WaveState *wave, Fn visit) {
  for_each_pending_op(wave, [&](const PendingAsyncOp &op) { visit(op.instruction_id); });
}

inline PendingAsyncOp make_pending_op(EntityId instruction_id, uint64_t pc, WaitCntType wait_type,
                                      uint32_t size_bytes, bool is_flat_ds = false) {
  PendingAsyncOp op;
  op.instruction_id = instruction_id;
  op.pc = pc;
  op.wait_type = wait_type;
  op.size = size_bytes;
  op.is_flat_ds = is_flat_ds;
  return op;
}

inline void track_pending_registers(WaveState *wave, VectorRegisterFile file,
                                    PendingRegisterSet set, uint32_t first_reg,
                                    uint32_t size_dwords, const PendingAsyncOp &op) {
  if (!wave || size_dwords == 0)
    return;

  auto &pending = pending_registers(*wave, file, set);
  auto &fifo = pending_register_fifo(*wave, file, set);
  const uint32_t max_index = max_register_index(file);
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = first_reg + i;
    if (reg > max_index)
      break;
    pending[reg] = op;
    fifo.push_back({reg, op});
  }
}

inline void track_vector_write(WaveState *wave, VectorRegisterFile file, EntityId instruction_id,
                               uint64_t pc, uint32_t first_reg, uint32_t size_dwords,
                               WaitCntType wait_type) {
  if (wait_type == WaitCntType::NONE)
    return;

  // An LDS-guarded destination is outstanding on the DScnt side only.
  const PendingRegisterSet set =
      wait_type == WaitCntType::LDS ? PendingRegisterSet::DsWrites : PendingRegisterSet::Writes;
  track_pending_registers(
      wave, file, set, first_reg, size_dwords,
      make_pending_op(instruction_id, pc, wait_type, bytes_for_dwords(size_dwords)));
}

inline void track_vector_flat_ds(WaveState *wave, VectorRegisterFile file, EntityId instruction_id,
                                 uint64_t pc, uint32_t first_reg, uint32_t size_dwords,
                                 bool is_flat_ds = true) {
  track_pending_registers(wave, file, PendingRegisterSet::DsWrites, first_reg, size_dwords,
                          make_pending_op(instruction_id, pc, WaitCntType::LDS,
                                          bytes_for_dwords(size_dwords), is_flat_ds));
}

inline void track_vector_read(WaveState *wave, VectorRegisterFile file, EntityId instruction_id,
                              uint64_t pc, uint32_t first_reg, uint32_t size_dwords,
                              WaitCntType wait_type = WaitCntType::STORE) {
  if (wait_type == WaitCntType::NONE)
    return;
  track_pending_registers(
      wave, file, PendingRegisterSet::Reads, first_reg, size_dwords,
      make_pending_op(instruction_id, pc, wait_type, bytes_for_dwords(size_dwords)));
}

inline void track_sgpr_write(WaveState *wave, EntityId instruction_id, uint64_t pc,
                             uint32_t sgpr_index, WaitCntType wait_type, uint32_t size_dwords = 1) {
  if (!wave || size_dwords == 0 || wait_type == WaitCntType::NONE)
    return;

  const PendingAsyncOp op =
      make_pending_op(instruction_id, pc, wait_type, bytes_for_dwords(size_dwords));
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = sgpr_index + i;
    if (reg > MAX_SGPR_INDEX)
      break;
    wave->pending_sgpr_writes[reg] = op;
    wave->smem_load_fifo.push_back({reg, op});
  }
}

inline void track_lds_write(WaveState *wave, EntityId instruction_id, uint64_t pc, uint32_t address,
                            uint32_t size, WaitCntType wait_type = WaitCntType::LDS) {
  if (!wave || size == 0 || wait_type == WaitCntType::NONE)
    return;
  wave->lds_fifo.push_back({address, make_pending_op(instruction_id, pc, wait_type, size)});
}

inline void track_lds_read(WaveState *wave, EntityId instruction_id, uint64_t pc, uint32_t address,
                           uint32_t size) {
  if (!wave || size == 0)
    return;
  wave->lds_read_fifo.push_back(
      {address, make_pending_op(instruction_id, pc, WaitCntType::LDS, size)});
}

inline void track_tensor_lds(WaveState *wave, EntityId instruction_id, uint64_t pc,
                             uint32_t address, uint32_t size) {
  if (!wave || size == 0)
    return;
  wave->tensor_lds_fifo.push_back(
      {address, make_pending_op(instruction_id, pc, WaitCntType::TENSOR, size)});
}

inline void process_wait_instruction(WaveState *wave, WaitCntType wait_type, uint32_t keep_count) {
  clear_pending_ops(wave, wait_type, keep_count);
}

} // namespace hazard_core
