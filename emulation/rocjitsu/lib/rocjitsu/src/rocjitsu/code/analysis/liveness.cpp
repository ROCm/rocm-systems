// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/liveness.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/free_registers.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace {

// find_free_run() tests a candidate run as one RegisterRef, whose width is a
// uint8_t. These queries take the run width as a uint16_t, so a wider request is
// expressible; it is answered with nullopt rather than truncated, because a
// truncated width would test only the first lane of the run and hand back a base
// whose remaining lanes were never checked.
[[nodiscard]] bool run_width_is_nameable(uint16_t count) {
  return count <= std::numeric_limits<uint8_t>::max();
}

void dfs_reverse_post_order(const BasicBlock &start,
                            const std::unordered_set<const BasicBlock *> &allowed,
                            std::unordered_set<const BasicBlock *> &visited,
                            std::vector<const BasicBlock *> &postorder) {
  if (!allowed.contains(&start) || !visited.insert(&start).second)
    return;

  std::vector<std::pair<const BasicBlock *, size_t>> stack;
  stack.emplace_back(&start, 0);

  while (!stack.empty()) {
    auto &[block, next_successor] = stack.back();
    const auto &successors = block->successors();
    if (next_successor < successors.size()) {
      const BasicBlock *succ = successors[next_successor++];
      if (succ != nullptr && allowed.contains(succ) && visited.insert(succ).second)
        stack.emplace_back(succ, 0);
      continue;
    }

    postorder.push_back(block);
    stack.pop_back();
  }
}

[[nodiscard]] RegisterSet kill_defs(const InstDefUse &du, ExecState exec_before) {
  RegisterSet kills = du.defs;
  // Predicated defs preserve old values on at least one control-flow path, so
  // they can never be unconditional kills.
  if (du.has_predicated_def)
    return {};
  // EXEC-masked vector defs preserve inactive lanes' old values, so they are
  // kills only where EXEC is provably full (every lane overwritten). Where the
  // EXEC state is unknown we stay conservative and do not kill.
  if (du.has_exec_masked_vector_def && exec_before != ExecState::Full) {
    kills.clear_class(RegClass::VGPR);
    kills.clear_class(RegClass::ACC_VGPR);
  }
  return kills;
}

[[nodiscard]] std::optional<uint32_t> text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  if (text.empty() || offset + sizeof(uint32_t) > text.size())
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool may_access_vgprs_indirectly(const Instruction &inst,
                                               std::span<const uint8_t> text, rj_code_arch_t arch) {
  const std::string_view mnemonic = inst.mnemonic();
  // TODO: Move indirect-VGPR access properties into decoded instruction
  // metadata so future ISA variants cannot bypass this completeness gate.
  if (mnemonic.starts_with("v_movrel") || mnemonic.starts_with("v_swaprel")) {
    return true;
  }
  if (!isa_properties(arch).mode_has_gpr_idx_en)
    return false;
  if (mnemonic == "s_set_gpr_idx_on")
    return true;

  // GPR indexing can also be enabled by writing MODE.GPR_IDX_EN through a
  // generic S_SETREG form. Dynamic writes covering bit 27 fail closed; immediate
  // writes can stay eligible when their decoded literal proves the bit is clear.
  const bool is_dynamic_setreg = mnemonic == "s_setreg_b32";
  const bool is_immediate_setreg = mnemonic == "s_setreg_imm32_b32";
  if (!is_dynamic_setreg && !is_immediate_setreg) {
    return false;
  }
  const Operand *hwreg_operand = inst.dst_operand(0);
  if (hwreg_operand == nullptr)
    return true;
  const uint16_t hwreg = static_cast<uint16_t>(hwreg_operand->encoding_value());
  const uint16_t id = hwreg & 0x3fu;
  const uint16_t begin = (hwreg >> 6) & 0x1fu;
  const uint16_t width = static_cast<uint16_t>(((hwreg >> 11) & 0x1fu) + 1u);
  constexpr uint16_t kModeHwreg = 1;
  constexpr uint16_t kGprIdxEnableBit = 27;
  if (id != kModeHwreg || begin > kGprIdxEnableBit ||
      static_cast<uint32_t>(begin) + width <= kGprIdxEnableBit) {
    return false;
  }

  if (is_dynamic_setreg)
    return true;

  const auto literal = text_word_at(text, inst.src_loc() + sizeof(uint32_t));
  if (!literal || inst.size() < 2 * static_cast<int>(sizeof(uint32_t)))
    return true;
  return ((*literal >> (kGprIdxEnableBit - begin)) & 1u) != 0;
}

} // namespace

LivenessAnalysis::LivenessAnalysis(UnavailableTag) : available_(false) {}

LivenessAnalysis LivenessAnalysis::unavailable() { return LivenessAnalysis(UnavailableTag{}); }

std::vector<const BasicBlock *> reverse_post_order(KernelBlockScope blocks) {
  std::vector<const BasicBlock *> postorder;
  std::unordered_set<const BasicBlock *> allowed;
  std::unordered_set<const BasicBlock *> visited;

  allowed.reserve(blocks.size());
  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      allowed.insert(block);
  }

  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      dfs_reverse_post_order(*block, allowed, visited, postorder);
  }

  std::ranges::reverse(postorder);
  return postorder;
}

LivenessAnalysis::LivenessAnalysis(KernelBlockScope blocks, std::unique_ptr<ExecMaskAnalysis> exec,
                                   LivenessAnalysisOptions options,
                                   std::span<const ScopedCfgEdge> extra_edges) {
  min_free_vgpr_ = options.min_free_vgpr;
  max_free_vgpr_ =
      static_cast<uint16_t>(std::min<size_t>(options.max_free_vgpr, REGISTER_SET_MAX_VGPRS));
  // The flag is inert when an ExecMaskAnalysis is supplied, which would silently
  // give a caller that set it the conservative answer instead.
  assert(!(exec && options.exec_masked_defs_kill) &&
         "exec_masked_defs_kill has no effect alongside an ExecMaskAnalysis");
  exec_masked_defs_kill_ = options.exec_masked_defs_kill;
  // Own the EXEC-state analysis; the backward dataflow is deferred to the first
  // query (ensure_analyzed), which consults it for kills. May be null: kills then
  // treat every EXEC-masked vector def as `Unknown` (conservative, never a kill)
  // unless exec_masked_defs_kill declared the scope runs under one mask.
  exec_ = std::move(exec);
  deferred_blocks_.assign(blocks.begin(), blocks.end());
  scoped_blocks_.reserve(blocks.size());
  for (const BasicBlock *block : blocks) {
    if (block != nullptr)
      scoped_blocks_.insert(block);
  }
  deferred_extra_edges_.assign(extra_edges.begin(), extra_edges.end());
  deferred_live_before_instructions_.assign(options.live_before_instructions.begin(),
                                            options.live_before_instructions.end());
  deferred_restrict_live_before_to_instructions_ = options.restrict_live_before_to_instructions;

  const KernelBlockScope deferred_scope(deferred_blocks_);
  if (options.arch == ROCJITSU_CODE_ARCH_CDNA5 && options.entry_block != nullptr) {
    gfx1250_vgpr_msb_ = std::make_unique<Gfx1250VgprMsbAnalysis>(
        deferred_scope, options.entry_block, deferred_extra_edges_, options.text,
        options.additional_entry_blocks);
  }
  collect_global_register_usage(deferred_scope, options.text, options.arch);
}

LivenessAnalysis::~LivenessAnalysis() = default;
LivenessAnalysis::LivenessAnalysis(LivenessAnalysis &&) noexcept = default;
LivenessAnalysis &LivenessAnalysis::operator=(LivenessAnalysis &&) noexcept = default;

void LivenessAnalysis::require_available() const {
  if (!available_)
    throw std::logic_error("liveness query from a rule marked liveness-free");
}

void LivenessAnalysis::ensure_analyzed() const {
  require_available();
  if (analyzed_)
    return;

  analyze(KernelBlockScope(deferred_blocks_), deferred_restrict_live_before_to_instructions_,
          deferred_live_before_instructions_, deferred_extra_edges_);
  analyzed_ = true;
}

void LivenessAnalysis::collect_global_register_usage(KernelBlockScope blocks,
                                                     std::span<const uint8_t> text,
                                                     rj_code_arch_t arch) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (may_access_vgprs_indirectly(inst, text, arch))
        global_vgpr_usage_is_complete_ = false;

      const InstDefUse accesses(inst, gfx1250_vgpr_msb_.get(), UnknownVgprDefPolicy::ExpandAll);
      globally_used_registers_ |= accesses.defs;
      globally_used_registers_ |= accesses.uses;
    }
  }
}

void LivenessAnalysis::analyze(KernelBlockScope blocks, bool restrict_live_before_to_instructions,
                               std::span<const Instruction *const> live_before_instructions,
                               std::span<const ScopedCfgEdge> extra_edges) const {
  liveness_.resize(blocks.size());
  block_index_.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      block_index_.emplace(blocks[i], i);
  }

  // Without an EXEC-state analysis, treat every program point as `Unknown` so
  // kill_defs never promotes an EXEC-masked vector def to a kill, unless the
  // caller declared the whole scope runs under one mask (exec_masked_defs_kill).
  const auto exec_before = [this](const Instruction &inst) {
    if (exec_)
      return exec_->before(inst);
    return exec_masked_defs_kill_ ? ExecState::Full : ExecState::Unknown;
  };

  const bool filter_live_before = restrict_live_before_to_instructions;
  std::unordered_set<const Instruction *> requested_live_before;
  if (filter_live_before) {
    requested_live_before.reserve(live_before_instructions.size());
    for (const Instruction *inst : live_before_instructions) {
      if (inst != nullptr)
        requested_live_before.insert(inst);
    }
    live_before_.reserve(requested_live_before.size());
  }
  std::vector<size_t> requested_live_before_by_block;
  if (filter_live_before)
    requested_live_before_by_block.resize(blocks.size());

  std::vector<std::vector<size_t>> successors(blocks.size());
  std::vector<std::vector<size_t>> predecessors(blocks.size());
  auto add_edge = [&](const BasicBlock *from, const BasicBlock *to) {
    auto from_it = block_index_.find(from);
    auto to_it = block_index_.find(to);
    if (from_it == block_index_.end() || to_it == block_index_.end())
      return;
    auto &succs = successors[from_it->second];
    if (std::ranges::find(succs, to_it->second) != succs.end())
      return;
    succs.push_back(to_it->second);
    predecessors[to_it->second].push_back(from_it->second);
  };

  for (const BasicBlock *block : blocks) {
    if (block == nullptr)
      continue;
    for (const BasicBlock *succ : block->successors())
      add_edge(block, succ);
  }
  for (const ScopedCfgEdge &edge : extra_edges)
    add_edge(edge.from, edge.to);

  // Compute each block's local transfer function before iterating across CFG
  // edges. `gen` keeps only uses that occur before a local definition, because
  // later uses are satisfied inside the block. `kill` is every local def.
  size_t instruction_count = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto *block = blocks[i];
    if (block == nullptr)
      continue;
    auto &state = liveness_[i];
    for (const auto &inst : block->instructions()) {
      ++instruction_count;
      if (filter_live_before && requested_live_before.contains(&inst))
        ++requested_live_before_by_block[i];
      InstDefUse du(inst, gfx1250_vgpr_msb_.get());
      RegisterSet kills = kill_defs(du, exec_before(inst));
      RegisterSet upward_uses = du.uses;
      upward_uses -= state.kill;
      state.gen |= upward_uses;
      state.kill |= kills;
    }
  }
  if (!filter_live_before)
    live_before_.reserve(instruction_count);

  std::deque<size_t> worklist;
  std::vector<bool> in_worklist(blocks.size(), false);
  auto enqueue = [&](size_t idx) {
    if (idx >= in_worklist.size() || in_worklist[idx])
      return;
    in_worklist[idx] = true;
    worklist.push_back(idx);
  };

  for (size_t idx = 0; idx < blocks.size(); ++idx)
    enqueue(idx);

  while (!worklist.empty()) {
    const size_t idx = worklist.front();
    worklist.pop_front();
    in_worklist[idx] = false;

    const BasicBlock *block = blocks[idx];
    if (block == nullptr)
      continue;

    RegisterSet live_out;
    for (size_t succ_idx : successors[idx])
      live_out |= liveness_[succ_idx].live_in;

    RegisterSet live_in = live_out;
    live_in -= liveness_[idx].kill;
    live_in |= liveness_[idx].gen;

    auto &state = liveness_[idx];
    const bool live_in_changed = state.live_in != live_in;
    if (state.live_out != live_out || live_in_changed) {
      state.live_out = live_out;
      state.live_in = live_in;

      if (live_in_changed) {
        for (size_t pred_idx : predecessors[idx])
          enqueue(pred_idx);
      }
    }
  }

  // Materialize live-before for instruction-level queries. The transfer
  // function is intentionally applied per instruction, so read-modify-write
  // instructions keep their source register live before the instruction even
  // when the same register is also defined by the instruction.
  for (size_t i = 0; i < blocks.size(); ++i) {
    auto *block = blocks[i];
    if (block == nullptr)
      continue;
    if (filter_live_before && requested_live_before_by_block[i] == 0)
      continue;
    RegisterSet live = liveness_[i].live_out;
    size_t remaining_requested = filter_live_before ? requested_live_before_by_block[i] : 0;
    auto &insts = block->instructions();
    for (auto it = insts.end(); it != insts.begin();) {
      --it;
      const Instruction *inst = &*it;
      InstDefUse du(*inst, gfx1250_vgpr_msb_.get());
      RegisterSet kills = kill_defs(du, exec_before(*inst));
      live -= kills;
      live |= du.uses;
      if (!filter_live_before || requested_live_before.contains(inst)) {
        live_before_.emplace(inst, live);
        if (filter_live_before && --remaining_requested == 0)
          break;
      }
    }
  }
}

const BlockLiveness &LivenessAnalysis::block_liveness(const BasicBlock &block) const {
  ensure_analyzed();
  auto it = block_index_.find(&block);
  if (it == block_index_.end())
    throw std::out_of_range("block_liveness: block was not part of this analysis");
  return liveness_.at(it->second);
}

bool LivenessAnalysis::has_live_before(const Instruction &inst) const {
  ensure_analyzed();
  return live_before_.contains(&inst);
}

const RegisterSet &LivenessAnalysis::live_before(const Instruction &inst) const {
  ensure_analyzed();
  auto it = live_before_.find(&inst);
  return it != live_before_.end() ? it->second : empty_;
}

bool LivenessAnalysis::is_live_before(const Instruction &inst, RegisterRef ref) const {
  return live_before(inst).contains(ref);
}

std::optional<uint8_t> LivenessAnalysis::vgpr_msb_bank_before(const Instruction &inst,
                                                              amdgpu::VgprMsbRole role) const {
  require_available();
  if (gfx1250_vgpr_msb_ == nullptr)
    return std::nullopt;
  return gfx1250_vgpr_msb_->bank_before(inst, role);
}

bool LivenessAnalysis::global_vgpr_usage_is_complete() const {
  require_available();
  return global_vgpr_usage_is_complete_;
}

std::optional<uint16_t>
LivenessAnalysis::find_globally_unused_vgpr_run(const Instruction *inst, uint16_t count,
                                                uint16_t search_start, uint16_t base_alignment,
                                                uint16_t available_count) const {
  require_available();
  if (inst == nullptr || !global_vgpr_usage_is_complete_ ||
      !scoped_blocks_.contains(inst->parent()) || !run_width_is_nameable(count)) {
    return std::nullopt;
  }
  // min_free_vgpr_ is an allocation floor, not a dataflow fact; max_free_vgpr_
  // is the destination-encoding ceiling. The caller's available_count narrows
  // the latter to what the current descriptor allocation actually owns.
  const auto first_candidate =
      static_cast<uint16_t>(std::max<size_t>(search_start, min_free_vgpr_));
  const auto limit = static_cast<uint32_t>(std::min<size_t>(available_count, max_free_vgpr_));
  return rocjitsu::find_free_run(globally_used_registers_, RegClass::VGPR,
                                 static_cast<uint8_t>(count), first_candidate, base_alignment,
                                 limit);
}

std::optional<uint16_t> LivenessAnalysis::find_free_run(const Instruction *inst, uint16_t count,
                                                        uint16_t search_start,
                                                        uint16_t base_alignment) const {
  ensure_analyzed();
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end() || !run_width_is_nameable(count))
    return std::nullopt;

  const auto first_candidate =
      static_cast<uint16_t>(std::max<size_t>(search_start, min_free_vgpr_));
  return rocjitsu::find_free_run(live_it->second, RegClass::VGPR, static_cast<uint8_t>(count),
                                 first_candidate, base_alignment, max_free_vgpr_);
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr_pair(const Instruction *inst,
                                                              uint16_t search_start) const {
  ensure_analyzed();
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end())
    return std::nullopt;

  return rocjitsu::find_free_sgpr_pair(live_it->second, REGISTER_SET_ALLOCATABLE_SGPRS,
                                       search_start);
}

std::optional<uint16_t> LivenessAnalysis::find_free_sgpr(const Instruction *inst,
                                                         uint16_t search_start) const {
  ensure_analyzed();
  auto live_it = live_before_.find(inst);
  if (live_it == live_before_.end())
    return std::nullopt;

  // Only normal SGPRs that are valid across supported families are candidates
  // for temporary allocation, which is what REGISTER_SET_ALLOCATABLE_SGPRS
  // bounds here and in find_free_sgpr_pair().
  return rocjitsu::find_free_sgpr(live_it->second, REGISTER_SET_ALLOCATABLE_SGPRS, search_start);
}

} // namespace rocjitsu
