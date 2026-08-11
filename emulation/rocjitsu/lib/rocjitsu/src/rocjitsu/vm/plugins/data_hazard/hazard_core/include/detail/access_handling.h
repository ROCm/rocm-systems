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
#include <limits>
#include <unordered_map>

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

inline PendingRegisterHazard
check_vgpr_read_hazard_info(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                            EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};
  return find_pending_register_hazard(wave->pending_vgpr_writes, vgpr_index, size_dwords,
                                      current_instruction_id);
}

inline const PendingAsyncOp *
check_vgpr_read_hazard(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                       EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_vgpr_read_hazard_info(wave, vgpr_index, size_dwords, current_instruction_id).pending;
}

inline PendingRegisterHazard
check_vgpr_ds_read_hazard_info(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                               EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};
  return find_pending_register_hazard(wave->pending_vgpr_writes_ds, vgpr_index, size_dwords,
                                      current_instruction_id);
}

inline const PendingAsyncOp *
check_vgpr_ds_read_hazard(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                          EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_vgpr_ds_read_hazard_info(wave, vgpr_index, size_dwords, current_instruction_id)
      .pending;
}

inline PendingRegisterHazard
check_vgpr_waw_hazard_info(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                           EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};

  auto hazard = find_pending_register_hazard(wave->pending_vgpr_writes, vgpr_index, size_dwords,
                                             current_instruction_id);
  if (hazard.pending)
    return hazard;
  return find_pending_register_hazard(wave->pending_vgpr_writes_ds, vgpr_index, size_dwords,
                                      current_instruction_id);
}

inline const PendingAsyncOp *
check_vgpr_waw_hazard(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                      EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_vgpr_waw_hazard_info(wave, vgpr_index, size_dwords, current_instruction_id).pending;
}

inline PendingRegisterHazard
check_vgpr_war_hazard_info(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                           EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  if (!wave)
    return {};
  return find_pending_register_hazard(wave->pending_vgpr_reads, vgpr_index, size_dwords,
                                      current_instruction_id);
}

inline const PendingAsyncOp *
check_vgpr_war_hazard(const WaveState *wave, uint32_t vgpr_index, uint32_t size_dwords,
                      EntityId current_instruction_id = NO_INSTRUCTION_FILTER) {
  return check_vgpr_war_hazard_info(wave, vgpr_index, size_dwords, current_instruction_id).pending;
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
  for (const auto &[_, op] : wave->pending_sgpr_writes)
    visit(op);
  for (const auto &[_, op] : wave->pending_vgpr_writes_ds)
    visit(op);
  for (const auto &[_, op] : wave->pending_vgpr_reads)
    visit(op);

  for (const auto &entry : wave->vmem_load_fifo)
    visit(entry.second);
  for (const auto &entry : wave->smem_load_fifo)
    visit(entry.second);
  for (const auto &entry : wave->lds_fifo)
    visit(entry.second);
  for (const auto &entry : wave->vmem_store_fifo)
    visit(entry.second);
  for (const auto &entry : wave->lds_read_fifo)
    visit(entry.second);
  for (const auto &entry : wave->flat_vgpr_ds_fifo)
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

inline void track_vgpr_write(WaveState *wave, EntityId instruction_id, uint64_t pc,
                             uint32_t vgpr_index, uint32_t size_dwords, WaitCntType wait_type) {
  if (!wave || size_dwords == 0 || wait_type == WaitCntType::NONE)
    return;

  const PendingAsyncOp op =
      make_pending_op(instruction_id, pc, wait_type, bytes_for_dwords(size_dwords));
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = vgpr_index + i;
    if (reg > MAX_VGPR_INDEX)
      break;

    if (wait_type == WaitCntType::LDS) {
      wave->pending_vgpr_writes_ds[reg] = op;
      wave->flat_vgpr_ds_fifo.push_back({reg, op});
    } else {
      wave->pending_vgpr_writes[reg] = op;
      wave->vmem_load_fifo.push_back({reg, op});
    }
  }
}

inline void track_flat_vgpr_ds(WaveState *wave, EntityId instruction_id, uint64_t pc,
                               uint32_t vgpr_index, uint32_t size_dwords, bool is_flat_ds = true) {
  if (!wave || size_dwords == 0)
    return;

  const PendingAsyncOp op = make_pending_op(instruction_id, pc, WaitCntType::LDS,
                                            bytes_for_dwords(size_dwords), is_flat_ds);
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = vgpr_index + i;
    if (reg > MAX_VGPR_INDEX)
      break;
    wave->pending_vgpr_writes_ds[reg] = op;
    wave->flat_vgpr_ds_fifo.push_back({reg, op});
  }
}

inline void track_vgpr_read(WaveState *wave, EntityId instruction_id, uint64_t pc,
                            uint32_t vgpr_index, uint32_t size_dwords,
                            WaitCntType wait_type = WaitCntType::STORE) {
  if (!wave || size_dwords == 0 || wait_type == WaitCntType::NONE)
    return;

  const PendingAsyncOp op =
      make_pending_op(instruction_id, pc, wait_type, bytes_for_dwords(size_dwords));
  for (uint32_t i = 0; i < size_dwords; ++i) {
    const uint32_t reg = vgpr_index + i;
    if (reg > MAX_VGPR_INDEX)
      break;
    wave->pending_vgpr_reads[reg] = op;
    wave->vmem_store_fifo.push_back({reg, op});
  }
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
