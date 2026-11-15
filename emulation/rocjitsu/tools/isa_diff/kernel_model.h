// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernel_model.h
/// @brief One decoded kernel: CFG, linear instruction order, LocSet liveness, PC taint.
///
/// @details The kernel is decoded exactly once, by RocJITsu's BasicBlock::build_reachable
/// over a single-section code object that wraps the artifact's .text. The permitted range
/// is the kernel symbol, so edges that leave it are surfaced as successor issues rather
/// than followed. Liveness runs over the precise P0 location universe because the
/// in-tree LivenessAnalysis models ordinary registers only. It is conservative in every
/// direction that matters to the contract:
///   - an unknown successor, call edge, or unknown-effect instruction makes everything live;
///   - instructions outside P0 never kill, and are assumed to read EXEC, VCC and M0;
///   - a block with no successors (s_endpgm) has nothing live after it.
#pragma once

#include "artifact.h"
#include "effects.h"
#include "location.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
class BasicBlock;
class CodeObject;
class Decoder;
class Instruction;
} // namespace rocjitsu

namespace rocjitsu::isa_diff {

struct InstRef {
  const Instruction *inst = nullptr;
  uint64_t offset = 0; ///< Section-relative byte offset.
  uint64_t pc = 0;     ///< Executed address (text_vaddr + offset).
  std::vector<uint32_t> words;
  std::string mnemonic;
  std::string text; ///< RocJITsu disassembly.
  Effects fx;
  uint32_t block = 0;
  bool leader = false;        ///< First instruction of its block.
  bool branch_target = false; ///< Target of some direct branch in the kernel.
  /// For a direct branch: index of the target instruction, or -1.
  int64_t branch_target_index = -1;
};

/// Memory operations that may still be outstanding at a program point (gfx9 counters).
struct PendingState {
  enum Counter : unsigned { kVm = 0, kLgkm = 1, kExp = 2, kNumCounters = 3 };
  std::array<uint8_t, kNumCounters> count{}; ///< Upper bound of outstanding ops (saturating).
  std::array<LocSet, kNumCounters> dests;    ///< Tracked locations outstanding ops will write.
  std::array<bool, kNumCounters> unknown{};  ///< An outstanding op has unknown effects.
  uint64_t example_pc = 0;                   ///< One outstanding producer (for messages).

  [[nodiscard]] bool any() const { return count[0] || count[1] || count[2]; }
  [[nodiscard]] LocSet all_dests() const { return dests[0] | dests[1] | dests[2]; }
  [[nodiscard]] bool any_unknown() const {
    return (count[0] && unknown[0]) || (count[1] && unknown[1]) || (count[2] && unknown[2]);
  }
  bool operator==(const PendingState &) const = default;
};

struct BlockInfo {
  size_t first = 0; ///< Index into insts().
  size_t end = 0;   ///< Exclusive.
  std::vector<uint32_t> successors;
  std::vector<uint32_t> predecessors;
  bool unknown_successor = false; ///< Indirect/missing edge or call: everything live-out.
};

class KernelModel {
public:
  /// @brief Decode and analyse one kernel. Returns nullptr and sets @p error on failure.
  static std::unique_ptr<KernelModel> build(const Artifact &artifact, std::string &error);
  ~KernelModel();
  KernelModel(const KernelModel &) = delete;
  KernelModel &operator=(const KernelModel &) = delete;

  [[nodiscard]] const Artifact &artifact() const { return artifact_; }
  [[nodiscard]] const std::vector<InstRef> &insts() const { return insts_; }
  [[nodiscard]] const std::vector<BlockInfo> &blocks() const { return blocks_; }

  /// Locations live immediately before / after instruction @p i.
  [[nodiscard]] LocSet live_before(size_t i) const;
  [[nodiscard]] const LocSet &live_after(size_t i) const { return live_after_[i]; }
  /// Locations that may hold a PC-derived value immediately before instruction @p i.
  [[nodiscard]] const LocSet &pc_tainted_before(size_t i) const { return taint_before_[i]; }
  /// Memory operations possibly outstanding immediately before instruction @p i: nothing
  /// at the kernel entry; joins take the maximum count and the union of destinations; only
  /// an s_waitcnt field of 0 clears that counter's destinations.
  [[nodiscard]] const PendingState &pending_before(size_t i) const { return pending_before_[i]; }

  /// True when no block has an unknown successor (liveness is then exact w.r.t. the CFG).
  [[nodiscard]] bool cfg_complete() const { return cfg_complete_; }
  /// Kernel reads the PC anywhere (s_getpc/s_swappc/...).
  [[nodiscard]] bool reads_pc() const { return reads_pc_; }
  /// Kernel bytes not covered by any reachable instruction, concatenated in address order.
  [[nodiscard]] const std::vector<uint8_t> &uncovered_bytes() const { return uncovered_; }
  /// True if block @p from can reach block @p to (reflexive).
  [[nodiscard]] bool reaches(uint32_t from, uint32_t to) const;

  /// Transfer function used by the liveness pass, exposed for tests.
  [[nodiscard]] static LocSet live_transfer(const Effects &fx, const LocSet &live_after);

private:
  KernelModel() = default;
  void compute_liveness();
  void compute_taint();
  void compute_pending();
  void compute_reachability();

  Artifact artifact_;
  std::unique_ptr<CodeObject> code_object_;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> owned_blocks_;
  std::vector<InstRef> insts_;
  std::vector<BlockInfo> blocks_;
  std::vector<LocSet> live_after_;
  std::vector<LocSet> taint_before_;
  std::vector<PendingState> pending_before_;
  std::vector<std::vector<bool>> reach_;
  std::vector<uint8_t> uncovered_;
  bool cfg_complete_ = true;
  bool reads_pc_ = false;
};

} // namespace rocjitsu::isa_diff
