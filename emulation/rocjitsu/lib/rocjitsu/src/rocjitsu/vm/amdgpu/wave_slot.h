// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wave_slot.h
/// @brief Per-slot sequencer state for a cycle-level driver of a compute unit.

#pragma once

#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/instruction_timing.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {

class ComputeUnitCore;
class Wavefront;

/// @brief One wave occupancy: a CU slot and the dispatch that filled it.
/// @details Work a sequencer has in flight names its wave by key, so it can
/// tell, when the work returns, whether the wave is still the one that sent it.
struct WaveKey {
  uint32_t cu = 0; ///< The driver's index for the CU.
  uint32_t slot = 0;
  uint64_t generation = 0; ///< Wavefront::dispatch_generation() of the occupant.
};

/// @brief The sequencer state of one CU wave slot (the timing half of Wavefront).
///
/// @details Lives as long as its WaveSlots. A wave launches into it when the
/// CU dispatches into the slot, runs until it issues s_endpgm or otherwise
/// halts, and leaves the slot idle for the next launch. The CU's Wavefront
/// stays the authority on architectural state; the slot holds what only a
/// cycle-level sequencer needs: the instruction buffer, the register
/// scoreboard, and the wave's view of its memory counters.
///
/// @code
///   CU dispatch -- launch() ----> +----------------+
///   fetch -------- fetched -----> |  WaveSlot      | -- next instruction --> dependence check
///   arbiter ------ reserve() ---> | buffer         | <- issued() / squash() -- issue
///   execution ---- writeback ---> | pending writes | -- has_hazard() ------> dependence check
///   memory ------- response ----> | counters       | -- waiting_on_counters() -->
///   issue -------- halt() ------> +----------------+
/// @endcode
///
/// A register write is reserved when the sequencer picks its instruction to
/// issue, and becomes visible at writeback: when the execution pipeline that
/// carries the instruction reports it.
class WaveSlot {
public:
  /// Memory-counter increments one wave may have reserved or in flight, the
  /// range of KMCNT, the narrowest counter a compute memory instruction
  /// increments. A wave whose next instruction would exceed it waits, as
  /// hardware stalls issue on a full counter. One budget covers every counter
  /// type, a simplification that can stall a wave before its own counter is full.
  static constexpr uint32_t kMaxMemoryTokens = WaitCounters::KMCNT_MAX;

  /// @brief Start a newly launched wave in this slot.
  /// @details Fetch starts from the wave's PC once its frontend is empty.
  /// @param wave The CU's wavefront for this slot.
  /// @param arrival Launch order, for oldest-first arbitration.
  void launch(Wavefront &wave, uint64_t arrival);
  /// @brief Return the slot to idle once its wave has halted.
  void halt();

  /// @brief Whether a launched wave occupies the slot and has not halted.
  bool active() const;

  /// @brief Discard the wave's reserved, unissued instructions.
  /// @details Bumps the epoch, so work for them still in flight to issue is
  /// dropped by its consumer, and frees their register reservations.
  /// @param flush Also discard the instruction buffer, for a redirect.
  void squash(bool flush);

  /// @brief Whether @p inst must wait for a register write still in flight.
  /// @details Also forgets writes that became visible by @p cycle.
  bool has_hazard(const InstructionTiming &inst, uint64_t cycle);

  /// @brief Reserve @p inst's register writes and counter capacity for @p seq.
  void reserve(const InstructionTiming &inst, uint64_t seq);
  /// @brief Undo reserve() for instruction @p seq, which did not issue.
  /// @param tokens The counter capacity reserve() took for it.
  void cancel(uint64_t seq, uint8_t tokens);
  /// @brief Record that @p seq issued; it can no longer be squashed.
  void issued(uint64_t seq);
  /// @brief Make @p seq's register writes visible from @p cycle.
  void written_back(uint64_t seq, uint64_t cycle);

  /// @brief Count @p inst's memory request as in flight.
  void issued_memory(const InstructionTiming &inst);
  /// @brief Apply @p seq's memory response: release @p counters and write back the result.
  void memory_returned(uint64_t seq, const MemoryPipeline::WaitCounterTokens &counters,
                       uint64_t cycle);

  /// @brief Whether a counter wait from an issued wait instruction still holds.
  /// @details Clears the wait once the counters satisfy it.
  bool waiting_on_counters();

  /// @brief The buffered instruction the dependence check examines next, or null.
  const InstructionTiming *next_to_reserve() const {
    return reserved < buffer.size() ? &buffer[reserved] : nullptr;
  }
  /// @brief Whether nothing of the wave is buffered, reserved, or being fetched.
  bool frontend_empty() const { return buffer.empty() && reserved == 0 && !fetch_pending; }

  Wavefront *wf = nullptr; ///< The CU's wavefront for this slot, once one has launched.
  uint64_t generation = 0; ///< Dispatch generation of the launched wave.
  uint64_t age = 0;        ///< Launch order, for oldest-first arbitration.
  uint64_t epoch = 0;      ///< Bumped by a squash; older unissued work is discarded.
  /// Fetched instructions, oldest first. A vector: it holds a few entries, and
  /// keeps its capacity across launches, so retiring does not allocate.
  std::vector<InstructionTiming> buffer;
  uint32_t reserved = 0; ///< Entries picked to issue, not yet issued.
  uint64_t fetch_pc = 0; ///< Next byte address to fetch.
  bool fetch_pending = false;
  bool fetch_stopped = false;        ///< The last fetched instruction redirects fetch.
  bool in_ready_list = false;        ///< The next instruction is waiting to be picked.
  std::optional<WaitTarget> waiting; ///< Counter wait from an issued wait instruction.
  WaitCounters outstanding;          ///< Memory requests the sequencer has in flight.
  uint32_t memory_tokens = 0;        ///< Counter increments reserved and not yet returned.

private:
  /// A register write an instruction has not yet made visible.
  struct PendingWrite {
    RegisterSet registers;
    uint64_t ready_cycle = UINT64_MAX; ///< First cycle a reader may issue.
    uint64_t seq = 0;                  ///< Instruction that makes the write.
    bool issued = false;               ///< False while the instruction may still be squashed.
  };
  std::vector<PendingWrite> pending_writes_;
};

/// @brief The sequencer state of a CU's wave slots, one WaveSlot per slot.
/// @details Takes the CU's wave launch handler, so it hears of every wave the
/// CU dispatches, and releases it on destruction. Construct it before the CU
/// dispatches, and only one per CU.
class WaveSlots {
public:
  /// @param cu The CU whose slots to track; not owned, must outlive this.
  /// @param cu_index The driver's index for @p cu, carried in each WaveKey.
  WaveSlots(ComputeUnitCore &cu, uint32_t cu_index);
  ~WaveSlots();
  WaveSlots(const WaveSlots &) = delete;
  WaveSlots &operator=(const WaveSlots &) = delete;

  WaveSlot &operator[](uint32_t slot) { return slots_[slot]; }
  /// @returns The slot's wave if @p key still names it, or null once it has left.
  WaveSlot *live(const WaveKey &key);
  /// @returns The wave if @p key still names it and it has not been squashed
  /// since @p epoch, or null: the staleness test for work that precedes issue.
  WaveSlot *current(const WaveKey &key, uint64_t epoch) {
    WaveSlot *slot = live(key);
    return slot && slot->epoch == epoch ? slot : nullptr;
  }
  /// @brief Return the key of the wave in @p slot.
  WaveKey key(uint32_t slot) const { return {cu_index_, slot, slots_[slot].generation}; }
  uint32_t size() const { return static_cast<uint32_t>(slots_.size()); }

private:
  /// Start @p wave in the slot it launched into.
  void launch(Wavefront &wave);

  ComputeUnitCore &cu_;
  uint32_t cu_index_;
  std::vector<WaveSlot> slots_;
  uint64_t next_arrival_ = 0;
};

} // namespace rocjitsu::amdgpu
