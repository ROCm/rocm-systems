// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_plan.h
/// @brief Straight-line MMA issue plans with helper chains, compiled once per block.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/helper_error_sink.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu::mma_plan {

// Long enough for full Tensile/HipKittens straight-line K-step tiles (71-128 insns).
inline constexpr unsigned kMaxSteps = 128;
/// Helper jobs a block may hand off. Each is one submission; wait masks are 64 bits.
inline constexpr unsigned kMaxJobs = 64;
using JobMask = uint64_t;
/// Jobs submitted this close to the block end may still run when the next
/// block starts, so all of their factors are copied at submission: the next
/// block's loads can then overwrite the registers without waiting.
inline constexpr unsigned kTailSteps = 16;
/// Plan steps reached from a word with no vector register operand: scalar ALU,
/// scalar memory and program control. Such an instruction can run while helper
/// jobs are outstanding without a register check. On RDNA4, MODE-modifying
/// scalar instructions (s_setreg*, s_round_mode, s_denorm_mode) are excluded
/// because WMMA reads MODE.FP16_OVFL.
inline bool scalar_only_encoding(rocjitsu::IsaFamily arch, uint32_t word) {
  const uint32_t top9 = word >> 23;
  if (top9 == 0x17d || top9 == 0x17e) // SOP1, SOPC
    return true;
  if (top9 == 0x17f) { // SOPP
    if (arch == rocjitsu::IsaFamily::Rdna4) {
      const uint32_t op = (word >> 16) & 0x7f;
      if (op == 17 || op == 18) // s_round_mode, s_denorm_mode
        return false;
    }
    return true;
  }
  if ((word >> 28) == 0xb) { // SOPK
    if (arch == rocjitsu::IsaFamily::Rdna4) {
      const uint32_t op = (word >> 23) & 0x1f;
      if (op == 18 || op == 19) // s_setreg_b32, s_setreg_imm32_b32
        return false;
    }
    return true;
  }
  if ((word >> 30) == 0x2) // SOP2
    return true;
  if (arch == rocjitsu::IsaFamily::Rdna4)
    return (word >> 26) == 0x3d; // RDNA4 SMEM
  return (word >> 26) == 0x30;   // GFX9 SMEM
}
inline bool scalar_only_encoding(uint32_t word) {
  return scalar_only_encoding(rocjitsu::IsaFamily::Cdna4, word);
}
inline constexpr unsigned kMaxChain = matrix_coexecution::Helper::kMaxChain;

enum class StepKind : uint8_t {
  Offload,   ///< Head of a helper chain; submits every member.
  Chained,   ///< Executed by an earlier Offload step's helper.
  IssuerMma, ///< Matrix instruction kept on the issuer.
  Alu,       ///< Retained pure arithmetic executed on the issuer.
  Ordinary,  ///< Ordinary issue: memory, waits and other stateful instructions.
};

/// @brief Factor registers copied at submission because a later issuer step overwrites them.
struct SnapshotSpec {
  uint16_t a_base = 0, b_base = 0; ///< Wave-relative; AccVGPRs start at 256.
  uint16_t sa_base = 0, sb_base = 0; ///< Block scales of a scaled MFMA, one register each.
  uint8_t a_regs = 0, b_regs = 0, sa_regs = 0, sb_regs = 0;
};

struct Step {
  StepKind kind = StepKind::Ordinary;
  uint8_t size = 0;
  uint16_t offset = 0; ///< Bytes from the block entry.
  uint8_t job = 0;   ///< Helper job of an Offload or Chained step.
  JobMask wait = 0;  ///< Helper jobs joined before the step.
  uint8_t members = 0;
  Instruction *inst = nullptr;
  std::array<Instruction *, kMaxChain> chain{};
  bool snapshots = false;
  std::array<SnapshotSpec, kMaxChain> snapshot{};
  std::array<uint8_t, kMaxChain> member_step{}; ///< Program position of each member.
};

/// @brief Matrix instructions a helper may execute: register-only operands.
/// @details Mirrors AsyncInstructionWindow::submit_mma(). Inline accumulator
/// and scale constants are immutable; SGPRs are outside the VGPR footprint.
inline bool offloadable(const Instruction &inst, bool has_accvgprs) {
  if (!async_mma_policy::candidate(inst))
    return false;
  const int sources = inst.num_src_operands();
  if (inst.num_dst_operands() != 1 || (sources != 3 && sources != 5))
    return false;
  for (int i = 0; i <= sources; ++i) {
    const auto *operand = i == sources ? inst.dst_operand(0) : inst.src_operand(i);
    const auto ref = operand->to_register_ref();
    if (ref && (ref->cls == RegClass::VGPR || (has_accvgprs && ref->cls == RegClass::ACC_VGPR)))
      continue;
    if (i < 2 || i == sources || !operand->const_value())
      return false;
  }
  return true;
}

/// @brief Stateless issuer work that needs no memory pipeline or wait counter.
inline bool pure_alu(const Instruction &inst) {
  const Operation operation = inst.amdgpu_operation();
  if (operation == Operation::S_WAIT_ALU || operation == Operation::S_CLAUSE)
    return true;
  if (inst.is_memory_op() || inst.is_waitcnt() || inst.data())
    return false;
  if (inst.amdgpu_secondary_operation() != Operation::Unknown)
    return async_mma_policy::is_safe_vopd(inst);
  return operation == Operation::S_SETPRIO || operation == Operation::S_MOVK_I32 ||
         operation == Operation::S_ADDK_I32 || operation == Operation::S_NOP ||
         operation == Operation::V_NOP || operation == Operation::S_DELAY_ALU ||
         async_mma_policy::inline_operation_kind(operation) ==
             async_mma_policy::InlineOperationKind::Arithmetic;
}

/// @brief Issuer steps that can neither stall nor trap, so a helper may run a
/// later member before them: LDS accesses (no virtual memory behind them) and
/// waits (functional memory completes at issue). Register effects are still
/// checked by footprint overlap. Vector memory stays a wall: a retry or fault
/// would re-enter the block and re-execute an accumulate a helper already ran.
inline bool transparent(const Instruction &inst) {
  if (inst.is_waitcnt())
    return true;
  return async_mma_policy::lds_access(inst.amdgpu_operation()) &&
         !async_mma_policy::global_data_share(inst);
}

/// Prototype knobs: RJ_MMA_PLAN_CHAIN=0 disables same-accumulator chaining,
/// RJ_MMA_PLAN_ALU=0 sends arithmetic through ordinary issue.
inline bool knob(const char *name) {
  const char *value = std::getenv(name);
  return !value || *value != '0';
}
inline bool chaining_enabled() {
  static const bool enabled = knob("RJ_MMA_PLAN_CHAIN");
  return enabled;
}
inline bool alu_enabled() {
  static const bool enabled = knob("RJ_MMA_PLAN_ALU");
  return enabled;
}
/// RJ_MMA_PLAN_SNAPSHOT=0 keeps write-after-read hazards as joins.
inline bool snapshot_enabled() {
  static const bool enabled = knob("RJ_MMA_PLAN_SNAPSHOT");
  return enabled;
}
/// RJ_MMA_CROSS_BLOCK=1 lets helper jobs outlive their block: the next block
/// joins each one at its first dependent step and tail jobs copy all their
/// factors at submission. Off by default: without per-accumulator helper
/// queues the first MFMA of each accumulator still waits for the previous
/// block's job on it, and the copies cost more than the drain they replace.
inline bool cross_block_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_CROSS_BLOCK");
    return value && *value && *value != '0';
  }();
  return enabled;
}
/// RJ_MMA_TAIL_SNAPSHOT=1 copies all factors of jobs in the last kTailSteps of
/// a block so a single-wave successor need not wait on overwrites. Off by
/// default: on multi-wave CUs sibling waves reap finished jobs before the
/// wave returns, making unconditional tail copies wasted bandwidth.
inline bool tail_snapshot_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_TAIL_SNAPSHOT");
    return value && *value && *value != '0';
  }();
  return enabled;
}
/// RJ_MMA_BOUNDED_WAIT=1 waits on the wave's oldest running job instead of
/// falling back inline when no helper is free or ready. Off by default.
inline bool bounded_wait_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_BOUNDED_WAIT");
    return value && *value && *value != '0';
  }();
  return enabled;
}
/// RJ_MMA_PLAN_INLINE=1 replays plans entirely on the issuer, isolating the
/// saved fetch, decode and scoreboard work from helper parallelism.
inline bool inline_only() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_PLAN_INLINE");
    return value && *value && *value != '0';
  }();
  return enabled;
}
/// Approximate matrix instruction weight in tenths of a microsecond on host SIMD.
inline unsigned mma_weight(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
  case Operation::V_MFMA_SCALE_F32_16X16X128_F8F6F4:
  case Operation::V_MFMA_SCALE_F32_32X32X64_F8F6F4:
  case Operation::V_MFMA_F32_16X16X128_F8F6F4:
  case Operation::V_MFMA_F32_32X32X64_F8F6F4:
    return 37;
  case Operation::V_MFMA_F32_16X16X32_FP8_FP8:
  case Operation::V_MFMA_F32_16X16X32_FP8_BF8:
  case Operation::V_MFMA_F32_16X16X32_BF8_FP8:
  case Operation::V_MFMA_F32_16X16X32_BF8_BF8:
  case Operation::V_MFMA_F32_32X32X16_FP8_FP8:
  case Operation::V_MFMA_F32_32X32X16_FP8_BF8:
  case Operation::V_MFMA_F32_32X32X16_BF8_FP8:
  case Operation::V_MFMA_F32_32X32X16_BF8_BF8:
    return 18;
  case Operation::V_MFMA_F32_16X16X32_F16:
  case Operation::V_MFMA_F32_32X32X16_F16:
  case Operation::V_MFMA_F32_16X16X32_BF16:
  case Operation::V_MFMA_F32_32X32X16_BF16:
    return 9;
  case Operation::V_MFMA_F32_16X16X16_F16:
  case Operation::V_MFMA_F32_32X32X8_F16:
    return 5;
  case Operation::V_WMMA_F32_16X16X16_FP8_FP8:
  case Operation::V_WMMA_F32_16X16X16_FP8_BF8:
  case Operation::V_WMMA_F32_16X16X16_BF8_FP8:
  case Operation::V_WMMA_F32_16X16X16_BF8_BF8:
    return 2;
  case Operation::V_WMMA_F32_16X16X16_F16:
  case Operation::V_WMMA_F32_16X16X16_BF16:
  case Operation::V_WMMA_F16_16X16X16_F16:
  case Operation::V_WMMA_BF16_16X16X16_BF16:
  case Operation::V_WMMA_I32_16X16X16_IU8:
  case Operation::V_WMMA_I32_16X16X16_IU4:
  case Operation::V_WMMA_I32_16X16X32_IU4:
    return 3;
  case Operation::V_SWMMAC_F32_16X16X32_F16:
  case Operation::V_SWMMAC_F32_16X16X32_BF16:
  case Operation::V_SWMMAC_F16_16X16X32_F16:
  case Operation::V_SWMMAC_BF16_16X16X32_BF16:
  case Operation::V_SWMMAC_I32_16X16X32_IU8:
  case Operation::V_SWMMAC_I32_16X16X32_IU4:
  case Operation::V_SWMMAC_I32_16X16X64_IU4:
  case Operation::V_SWMMAC_F32_16X16X32_FP8_FP8:
  case Operation::V_SWMMAC_F32_16X16X32_FP8_BF8:
  case Operation::V_SWMMAC_F32_16X16X32_BF8_FP8:
  case Operation::V_SWMMAC_F32_16X16X32_BF8_BF8:
    return 4;
  default:
    return 10;
  }
}
/// Maximum MMAs packed into one helper job when batching independent chains.
/// RJ_MMA_BATCH_PACK=1 disables independent-MMA batching.
inline bool batch_pack_explicit() {
  static const bool explicit_set = [] {
    const char *value = std::getenv("RJ_MMA_BATCH_PACK");
    return value && *value;
  }();
  return explicit_set;
}
inline bool batch_min_chains_explicit() {
  static const bool explicit_set = [] {
    const char *value = std::getenv("RJ_MMA_BATCH_MIN_CHAINS");
    return value && *value;
  }();
  return explicit_set;
}
inline unsigned batch_pack_limit() {
  static const unsigned limit = [] {
    const char *value = std::getenv("RJ_MMA_BATCH_PACK");
    if (!value || !*value)
      return 4u;
    const unsigned parsed = static_cast<unsigned>(std::strtoul(value, nullptr, 0));
    return std::clamp(parsed, 1u, kMaxChain);
  }();
  return limit;
}
/// Minimum logical chains in a block before independent-MFMA batching activates.
/// Keeps <= 8-chain kernels (e.g. acc8) at full 1:1 helper fan-out.
inline unsigned batch_min_chains() {
  static const unsigned min_chains = [] {
    const char *value = std::getenv("RJ_MMA_BATCH_MIN_CHAINS");
    if (!value || !*value)
      return 9u;
    return static_cast<unsigned>(std::strtoul(value, nullptr, 0));
  }();
  return min_chains;
}
/// Minimum per-job MMA weight (in tenths of a microsecond) required to accept a
/// block when RJ_MMA_MIN_JOB_WEIGHT is set. Zero disables the weight gate.
inline unsigned min_job_weight() {
  static const unsigned weight = [] {
    const char *value = std::getenv("RJ_MMA_MIN_JOB_WEIGHT");
    if (!value || !*value)
      return 0u;
    return static_cast<unsigned>(std::strtoul(value, nullptr, 0));
  }();
  return weight;
}

using Access = async_execution::Access;
inline bool overlaps(const Access &a, const Access &b) { return a.conflicts(b) || b.conflicts(a); }
/// Visit every set bit of a register set.
template <typename F> inline void for_each_register(const decltype(Access::reads) &bits, F &&f) {
#if defined(__GLIBCXX__)
  for (size_t r = bits._Find_first(); r < bits.size(); r = bits._Find_next(r))
    f(unsigned(r));
#else
  for (size_t r = 0; r != bits.size(); ++r)
    if (bits.test(r))
      f(unsigned(r));
#endif
}



/// @brief Register operands of one chain member, split into factors and the rest.
struct MemberOperands {
  unsigned step = 0;
  uint16_t a_base = 0, b_base = 0;
  uint16_t sa_base = 0, sb_base = 0; ///< Scale registers of a scaled MFMA (sa_regs/sb_regs 0 or 1).
  uint8_t a_regs = 0, b_regs = 0, sa_regs = 0, sb_regs = 0;
  bool a_snap = false, b_snap = false, sa_snap = false, sb_snap = false;
  Access other; ///< Accumulator and result registers.

  /// Registers the member reads that a snapshot can stand in for: A, B and scales.
  decltype(Access::reads) factors(bool snapped_only) const {
    decltype(Access::reads) bits;
    for (unsigned r = 0; r != a_regs; ++r)
      if (!snapped_only || a_snap)
        bits.set(a_base + r);
    for (unsigned r = 0; r != b_regs; ++r)
      if (!snapped_only || b_snap)
        bits.set(b_base + r);
    if (sa_regs && (!snapped_only || sa_snap))
      bits.set(sa_base);
    if (sb_regs && (!snapped_only || sb_snap))
      bits.set(sb_base);
    return bits;
  }
  void snapshot_all() {
    a_snap = a_regs != 0;
    b_snap = b_regs != 0;
    sa_snap = sa_regs != 0;
    sb_snap = sb_regs != 0;
  }
};

inline std::optional<MemberOperands> member_operands(const Instruction &inst, unsigned step,
                                                     uint32_t num_vgprs, bool has_accvgprs,
                                                     uint32_t wave_size = 32) {
  MemberOperands member;
  member.step = step;
  const bool tied_dst = async_mma_policy::is_rdna4_swmmac(inst.amdgpu_operation());
  const auto range = [&](const Operand *operand, uint16_t &base,
                         uint8_t &regs) -> std::optional<bool> {
    const auto ref = operand ? operand->to_register_ref() : std::nullopt;
    if (!ref || (ref->cls != RegClass::VGPR && ref->cls != RegClass::ACC_VGPR))
      return false;
    const bool acc = ref->cls == RegClass::ACC_VGPR;
    const uint32_t limit = acc ? (has_accvgprs ? Access::kRegistersPerBank : 0)
                               : std::min(Access::kRegistersPerBank, num_vgprs);
    const uint32_t width = async_mma_policy::matrix_operand_width(inst, ref->width, wave_size);
    if (ref->index >= limit || width > limit - ref->index)
      return std::nullopt;
    base = uint16_t(ref->index + (acc ? Access::kRegistersPerBank : 0));
    regs = uint8_t(width);
    return true;
  };
  for (int i = 0; i != inst.num_src_operands(); ++i) {
    uint16_t base = 0;
    uint8_t regs = 0;
    const auto found = range(inst.src_operand(i), base, regs);
    if (!found)
      return std::nullopt;
    if (!*found)
      continue;
    if (i == 0) {
      member.a_base = base;
      member.a_regs = regs;
    } else if (i == 1) {
      member.b_base = base;
      member.b_regs = regs;
    } else if (i == 3 && regs == 1) {
      member.sa_base = base;
      member.sa_regs = 1;
    } else if (i == 4 && regs == 1) {
      member.sb_base = base;
      member.sb_regs = 1;
    } else {
      for (unsigned r = 0; r != regs; ++r)
        member.other.reads.set(base + r);
    }
  }
  uint16_t base = 0;
  uint8_t regs = 0;
  const auto found = range(inst.dst_operand(0), base, regs);
  if (!found || !*found)
    return std::nullopt;
  for (unsigned r = 0; r != regs; ++r) {
    member.other.writes.set(base + r);
    if (tied_dst)
      member.other.reads.set(base + r);
  }
  return member;
}

struct Plan {
  using Words = std::array<uint32_t, 4>;
  struct Snapshot {
    uint64_t offset;
    Words words;
  };
  bool accept = false;
  bool local_only = false; ///< Lightweight MMAs stay inside the issuer's LLC domain.
  uint64_t epoch = 0;
  uint64_t length = 0; ///< Bytes covered by the steps.
  Words first{};
  std::vector<Snapshot> code;
  std::vector<Step> steps;
  std::vector<std::unique_ptr<Instruction>> owned;
  unsigned chains = 0, chained = 0, issuer_mma = 0, alu = 0, ordinary = 0;
  /// Union of every step's registers, and per register the first step that
  /// touches or writes it (kNoStep if none): a block entered while earlier
  /// jobs still run joins each job at the first step that depends on it.
  static constexpr uint8_t kNoStep = 0xff;
  Access footprint;
  std::array<uint8_t, 2 * Access::kRegistersPerBank> first_access{}, first_write{};
  /// Per-job register accesses (snapshotted factors excluded), indexed by Step::job.
  std::vector<Access> job_accesses;

  Plan() = default;
  Plan(Plan &&) = default;
  Plan &operator=(Plan &&other) {
    release();
    accept = other.accept;
    local_only = other.local_only;
    footprint = other.footprint;
    first_access = other.first_access;
    first_write = other.first_write;
    job_accesses = std::move(other.job_accesses);
    epoch = other.epoch;
    length = other.length;
    first = other.first;
    code = std::move(other.code);
    steps = std::move(other.steps);
    owned = std::move(other.owned);
    chains = other.chains;
    chained = other.chained;
    issuer_mma = other.issuer_mma;
    alu = other.alu;
    ordinary = other.ordinary;
    return *this;
  }
  ~Plan() { release(); }
  void release() {
    // Retained decodes may outlive the issuing worker; they use heap storage.
    Instruction::ScopedHeapAllocation heap;
    owned.clear();
  }
};

/// @brief Compile the straight-line block starting at a candidate MMA.
/// @details safe_inline() rejects every branch, barrier, terminator, trap and
/// EXEC write, so the recorded order is the only order the block can execute.
/// Footprints are computed once here; replay needs no register bitsets.
///
/// Scheduling first forms logical chains, so a block never joins a running
/// chain early just to free a helper. A post-pass then groups them into jobs.
template <typename Fetch>
void compile(Plan &plan, Decoder &decoder, Fetch &&fetch_words, uint64_t pc, uint32_t num_vgprs,
             bool has_accvgprs, uint64_t &decodes,
             bool tail_snapshot = cross_block_enabled() && tail_snapshot_enabled(),
             unsigned pack_limit = batch_pack_limit(),
             unsigned min_chains = batch_min_chains(),
             unsigned min_weight = min_job_weight(), uint32_t wave_size = 32) {
  Instruction::ScopedHeapAllocation heap;
  // A logical chain: one helper job if offloaded. `joined` is the first step
  // that must wait for it; the block end if none does.
  struct Chain {
    unsigned head = 0, joined = ~0u, count = 0;
    bool open = true;
    Access access;
    std::array<MemberOperands, kMaxChain> members{};
    void refresh() {
      access = {};
      for (unsigned i = 0; i != count; ++i) {
        access.reads |= members[i].other.reads | (members[i].factors(false) &
                                                  ~members[i].factors(true));
        access.writes |= members[i].other.writes;
      }
    }
  };
  std::vector<Chain> chains;
  std::vector<unsigned> live;             // logical chains not yet joined
  std::vector<Access> accesses;
  std::vector<int> step_chain;            // logical chain executing each MMA step, -1 otherwise
  std::vector<std::vector<unsigned>> joins; // logical chains each step waits for
  std::vector<bool> wall;                 // issuer step that could stall or fault
  uint64_t offset = 0;
  unsigned current = 0;
  const auto join = [&](const std::vector<unsigned> &which) {
    for (unsigned c : which) {
      chains[c].joined = current;
      std::erase(live, c);
    }
  };
  // A later step that only overwrites factor registers of a running chain need
  // not wait: those factors are copied when the chain is submitted. Any read
  // of its results, write of its results, or overwrite of an accumulator or
  // scale register still joins it.
  const auto break_war = [&](unsigned c, const Access &next) {
    auto &chain = chains[c];
    if ((chain.access.writes & (next.reads | next.writes)).any())
      return false;
    const auto war = chain.access.reads & next.writes;
    decltype(Access::reads) other, factors;
    for (unsigned i = 0; i != chain.count; ++i) {
      other |= chain.members[i].other.reads;
      factors |= chain.members[i].factors(false);
    }
    if ((war & other).any() || (war & ~factors).any())
      return false;
    for (unsigned i = 0; i != chain.count; ++i) {
      auto &member = chain.members[i];
      for (unsigned r = 0; r != member.a_regs; ++r)
        member.a_snap |= war.test(member.a_base + r);
      for (unsigned r = 0; r != member.b_regs; ++r)
        member.b_snap |= war.test(member.b_base + r);
      member.sa_snap |= member.sa_regs && war.test(member.sa_base);
      member.sb_snap |= member.sb_regs && war.test(member.sb_base);
    }
    chain.refresh();
    return true;
  };
  const auto conflicts_of = [&](const Access &next) {
    std::vector<unsigned> result;
    for (unsigned c : live)
      if (overlaps(chains[c].access, next) && !(snapshot_enabled() && break_war(c, next)))
        result.push_back(c);
    return result;
  };
  // Can `access` (an MMA at step `at`) run early, at step `from` on the helper
  // of `chains`? Its inputs must already exist there: no step in between may
  // touch its registers or be able to stall or fault (an early exit would
  // re-execute an accumulate the helper already ran). It must also not race a
  // chain that runs concurrently with that helper job.
  const auto hoistable = [&](const Access &access, unsigned from, unsigned at,
                             const std::vector<int> &own) {
    for (unsigned j = from + 1; j < at; ++j) {
      if (std::find(own.begin(), own.end(), step_chain[j]) != own.end())
        continue;
      if (wall[j] || overlaps(accesses[j], access))
        return false;
    }
    for (unsigned c = 0; c != chains.size(); ++c) {
      if (std::find(own.begin(), own.end(), int(c)) != own.end())
        continue;
      if (chains[c].head < at && chains[c].joined > from && overlaps(chains[c].access, access))
        return false;
    }
    return true;
  };

  for (unsigned k = 0; k != kMaxSteps; ++k) {
    current = k;
    Plan::Words words;
    if (!fetch_words(pc + offset, words))
      break;
    ++decodes;
    auto decoded = decoder.decode_window(words);
    if (!decoded.succeeded())
      break;
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    if (k == 0)
      plan.first = words;
    if (!async_mma_policy::safe_inline(*inst))
      break;
    const auto access = async_execution::footprint(*inst, num_vgprs, has_accvgprs, wave_size);
    if (!access)
      break;
    plan.code.push_back({offset, words});
    Step step;
    step.size = static_cast<uint8_t>(inst->size());
    step.offset = static_cast<uint16_t>(offset);
    std::optional<MemberOperands> operands;
    if (offloadable(*inst, has_accvgprs))
      operands = member_operands(*inst, k, num_vgprs, has_accvgprs, wave_size);
    // Placeholders keep step-indexed tables aligned while this step is decided.
    accesses.push_back(*access);
    step_chain.push_back(-1);
    joins.emplace_back();
    wall.push_back(false);
    const auto conflicts = conflicts_of(*access);
    if (operands) {
      // A same-accumulator successor joins its producer's chain: one handoff.
      int bundle = -1;
      if (chaining_enabled() && conflicts.size() == 1) {
        const unsigned c = conflicts.front();
        if (chains[c].open && chains[c].count < kMaxChain &&
            hoistable(*access, chains[c].head, k, std::vector<int>{int(c)}))
          bundle = int(c);
      }
      if (bundle >= 0) {
        auto &chain = chains[bundle];
        chain.members[chain.count++] = *operands;
        chain.refresh();
        step.kind = StepKind::Chained;
      } else {
        join(conflicts);
        joins.back() = conflicts;
        Chain chain;
        chain.head = k;
        chain.members[chain.count++] = *operands;
        chain.refresh();
        bundle = int(chains.size());
        chains.push_back(chain);
        live.push_back(unsigned(bundle));
        step.kind = StepKind::Offload;
      }
      step_chain.back() = bundle;
      step.inst = inst.get();
    } else {
      join(conflicts);
      joins.back() = conflicts;
      // Retain only stateless arithmetic. Memory instructions keep per-issue
      // state and go through ordinary issue and its decoded-instruction cache.
      if (alu_enabled() && pure_alu(*inst)) {
        step.kind = StepKind::Alu;
        step.inst = inst.get();
      }
      // An issuer step that could stall or fault closes every running chain
      // to later members (see hoistable()).
      if (step.kind == StepKind::Ordinary && !transparent(*inst)) {
        wall.back() = true;
        for (unsigned c : live)
          chains[c].open = false;
      }
    }
    if (step.inst)
      plan.owned.push_back(std::move(inst));
    plan.steps.push_back(step);
    offset += step.size;
  }
  plan.length = offset;
  const unsigned steps = unsigned(plan.steps.size());
  for (auto &chain : chains)
    chain.joined = std::min(chain.joined, steps);
  for (unsigned k = 0; k != steps; ++k) {
    plan.footprint.reads |= accesses[k].reads;
    plan.footprint.writes |= accesses[k].writes;
  }
  // Profitability: offloading pays only while another chain runs at the same
  // time. A serial chain (even one split at kMaxChain) never qualifies.
  unsigned max_concurrent = 0;
  for (unsigned t = 0; t != steps; ++t) {
    unsigned alive = 0;
    for (const auto &chain : chains)
      alive += chain.head <= t && t < chain.joined;
    max_concurrent = std::max(max_concurrent, alive);
  }
  // Placement: pack independent chains into helper jobs of up to `pack_limit`
  // members once a block has enough chains to keep helpers busy, grouping
  // chains that share A/B factor registers and identical join points so helper
  // handoffs drop without delaying any consumer or relaxing bit-exactness.
  // Small blocks (fewer than `min_chains` chains) keep 1:1 fan-out.
  constexpr int kIssuer = -1;
  std::vector<int> job_of(chains.size(), kIssuer);
  std::vector<unsigned> job_host;             // chain whose head submits each job
  std::vector<std::vector<int>> job_chains;   // chains of each job, host first
  std::vector<unsigned> job_size;             // members of each job
  std::vector<unsigned> job_joined;           // earliest join step among a job's chains
  std::vector<decltype(Access::reads)> job_factors; // union of A/B/scale reads per job
  unsigned total_weight = 0;
  unsigned total_members = 0;
  for (const auto &chain : chains) {
    total_members += chain.count;
    for (unsigned i = 0; i != chain.count; ++i)
      total_weight += mma_weight(*plan.steps[chain.members[i].step].inst);
  }
  const bool profitable = max_concurrent >= 2 && (min_weight == 0 || total_weight >= 2 * min_weight);
  if (profitable) {
    // Lightweight MMAs (e.g. RDNA4 16x16x16 WMMA at 200-300 ns/op) need fuller
    // jobs to amortize helper handoff and must never cross a CCD boundary; pack
    // up to kMaxChain once >= 4 chains exist unless explicitly overridden.
    const bool lightweight = total_members != 0 && total_weight <= 4u * total_members;
    plan.local_only = lightweight;
    const unsigned default_pack =
        lightweight && !batch_pack_explicit() && pack_limit == batch_pack_limit() ? kMaxChain
                                                                                  : pack_limit;
    const unsigned default_min_chains =
        lightweight && !batch_min_chains_explicit() && min_chains == batch_min_chains()
            ? 4u
            : min_chains;
    const unsigned effective_pack = std::clamp(default_pack, 1u, kMaxChain);
    const bool pack_independent = effective_pack > 1 && chains.size() >= default_min_chains;
    const auto can_merge = [&](unsigned j, unsigned c) {
      const unsigned h = job_host[j];
      if (chains[h].joined <= chains[c].head)
        return false;
      std::vector<int> own = job_chains[j];
      own.push_back(int(c));
      for (unsigned i = 0; i != chains[c].count; ++i) {
        const unsigned at = chains[c].members[i].step;
        if (!hoistable(accesses[at], chains[h].head, at, own))
          return false;
      }
      return true;
    };
    for (unsigned c = 0; c != chains.size(); ++c) {
      decltype(Access::reads) c_factors;
      for (unsigned i = 0; i != chains[c].count; ++i)
        c_factors |= chains[c].members[i].factors(false);
      int chosen = -1;
      if (pack_independent && chains[c].count < effective_pack) {
        size_t best_overlap = 0;
        unsigned best_size = kMaxChain + 1;
        for (unsigned j = 0; j != job_host.size(); ++j) {
          if (job_joined[j] != chains[c].joined ||
              job_size[j] + chains[c].count > effective_pack)
            continue;
          const size_t overlap = (job_factors[j] & c_factors).count();
          // Without shared factors, avoid diluting a nearly full A/B-affine job.
          if (overlap == 0 && job_size[j] > effective_pack / 2)
            continue;
          const bool better =
              chosen < 0 || overlap > best_overlap ||
              (overlap == best_overlap &&
               (job_size[j] < best_size || (job_size[j] == best_size && overlap == 0)));
          if (!better || !can_merge(j, c))
            continue;
          chosen = int(j);
          best_overlap = overlap;
          best_size = job_size[j];
        }
      }
      if (chosen < 0 && job_host.size() == kMaxJobs) {
        for (unsigned j = 0; j != job_host.size() && chosen < 0; ++j) {
          if (job_size[j] + chains[c].count > kMaxChain)
            continue;
          if (can_merge(j, c))
            chosen = int(j);
        }
      }
      if (chosen < 0) {
        if (job_host.size() == kMaxJobs)
          continue;
        chosen = int(job_host.size());
        job_host.push_back(c);
        job_chains.emplace_back();
        job_size.push_back(0);
        job_joined.push_back(chains[c].joined);
        job_factors.emplace_back();
      }
      job_of[c] = chosen;
      job_chains[chosen].push_back(int(c));
      job_size[chosen] += chains[c].count;
      job_joined[chosen] = std::min(job_joined[chosen], chains[c].joined);
      job_factors[chosen] |= c_factors;
    }
  }

  // Materialize steps: helper jobs, issuer MMAs and join masks.
  plan.accept = !job_host.empty();
  if (!plan.accept)
    return;
  for (unsigned k = 0; k != steps; ++k) {
    auto &step = plan.steps[k];
    const int c = step_chain[k];
    if (c >= 0) {
      if (job_of[c] == kIssuer) {
        step.kind = StepKind::IssuerMma;
      } else {
        step.job = uint8_t(job_of[c]);
        if (job_host[job_of[c]] == unsigned(c) && k == chains[c].head) {
          step.kind = StepKind::Offload;
          step.members = 0;
        } else {
          step.kind = StepKind::Chained;
        }
      }
    }
    for (unsigned j : joins[k])
      if (job_of[j] != kIssuer)
        step.wait |= JobMask{1} << job_of[j];
  }
  // Jobs submitted near the block end may outlive it; optionally copy all of
  // their factors so the next block's loads never wait for them.
  for (unsigned j = 0; tail_snapshot && snapshot_enabled() && j != job_host.size(); ++j) {
    if (chains[job_host[j]].head + kTailSteps < steps)
      continue;
    for (int c : job_chains[j]) {
      for (unsigned i = 0; i != chains[c].count; ++i)
        chains[c].members[i].snapshot_all();
      chains[c].refresh();
    }
  }
  // Fill each job's member list in program order of its host chain, then of
  // the chains appended to it.
  plan.job_accesses.assign(job_host.size(), {});
  for (unsigned j = 0; j != job_host.size(); ++j) {
    auto &head = plan.steps[chains[job_host[j]].head];
    for (int c : job_chains[j]) {
      plan.job_accesses[j].reads |= chains[c].access.reads;
      plan.job_accesses[j].writes |= chains[c].access.writes;
      for (unsigned i = 0; i != chains[c].count; ++i) {
        const auto &member = chains[c].members[i];
        const unsigned at = member.step;
        auto &spec = head.snapshot[head.members];
        if (member.a_snap)
          spec.a_base = member.a_base, spec.a_regs = member.a_regs;
        if (member.b_snap)
          spec.b_base = member.b_base, spec.b_regs = member.b_regs;
        if (member.sa_snap)
          spec.sa_base = member.sa_base, spec.sa_regs = 1;
        if (member.sb_snap)
          spec.sb_base = member.sb_base, spec.sb_regs = 1;
        head.snapshots |= member.a_snap || member.b_snap || member.sa_snap || member.sb_snap;
        head.member_step[head.members] = uint8_t(at);
        head.chain[head.members++] = plan.steps[at].inst;
      }
    }
  }
  // First-touch index for blocks entered while earlier jobs still run. A
  // member of a helper job executes when the job is submitted, at its head
  // step, so that is where a wait for it must land.
  std::vector<unsigned> effective(steps);
  for (unsigned k = 0; k != steps; ++k)
    effective[k] = k;
  for (unsigned j = 0; j != job_host.size(); ++j) {
    const unsigned head = chains[job_host[j]].head;
    for (int c : job_chains[j])
      for (unsigned i = 0; i != chains[c].count; ++i)
        effective[chains[c].members[i].step] = head;
  }
  plan.first_access.fill(Plan::kNoStep);
  plan.first_write.fill(Plan::kNoStep);
  for (unsigned k = 0; k != steps; ++k) {
    const uint8_t at = uint8_t(effective[k]);
    for_each_register(accesses[k].reads | accesses[k].writes, [&](unsigned r) {
      if (plan.first_access[r] == Plan::kNoStep || at < plan.first_access[r])
        plan.first_access[r] = at;
    });
    for_each_register(accesses[k].writes, [&](unsigned r) {
      if (plan.first_write[r] == Plan::kNoStep || at < plan.first_write[r])
        plan.first_write[r] = at;
    });
  }
  for (const auto &step : plan.steps) {
    plan.chains += step.kind == StepKind::Offload;
    plan.chained += step.kind == StepKind::Chained;
    plan.issuer_mma += step.kind == StepKind::IssuerMma;
    plan.alu += step.kind == StepKind::Alu;
    plan.ordinary += step.kind == StepKind::Ordinary;
  }
}

/// @brief Helper jobs of one wave, outliving the blocks that submitted them.
/// @details Jobs write only the wave's own registers, so a block may end with
/// jobs running: the next block joins each one at its first dependent step,
/// an instruction issued outside any plan joins everything unless its
/// encoding has no vector operand, and faults, traps, wave end and mode
/// changes drain. Slots are physical; a plan's job ids map onto them per block.
class PendingJobs {
public:
  explicit PendingJobs(matrix_coexecution::SharedPool *pool = nullptr) : pool_(pool) {}
  ~PendingJobs() { (void)drain(); }
  PendingJobs(const PendingJobs &) = delete;
  PendingJobs &operator=(const PendingJobs &) = delete;
  PendingJobs(PendingJobs &&other) noexcept { *this = std::move(other); }
  PendingJobs &operator=(PendingJobs &&other) noexcept {
    (void)drain();
    pool_ = other.pool_;
    active_ = std::exchange(other.active_, 0);
    inherited_ = std::exchange(other.inherited_, 0);
    track_access_ = other.track_access_;
    access_dirty_ = other.access_dirty_;
    next_seq_ = other.next_seq_;
    tickets_ = other.tickets_;
    heads_ = other.heads_;
    plans_ = other.plans_;
    entry_pc_ = other.entry_pc_;
    seq_ = other.seq_;
    member_offset_ = other.member_offset_;
    access_ = other.access_;
    failure_ = std::move(other.failure_);
    return *this;
  }
  void bind(matrix_coexecution::SharedPool &pool) { pool_ = &pool; }

  /// Begin a block: map its job ids to free slots as they are submitted, and
  /// compute which inherited jobs each step must join (split into result RAW/WAW
  /// waits and factor WAR waits). Returns true only when entry waits were
  /// populated, so callers can skip reading the wait arrays in the common case.
  bool begin_block(const Plan &plan, uint64_t entry_pc, unsigned cap,
                   std::array<JobMask, kMaxSteps> &result_wait,
                   std::array<JobMask, kMaxSteps> &factor_wait, bool &filtered,
                   bool track_access = true) {
    plan_ = &plan;
    block_pc_ = entry_pc;
    cap_ = cap;
    track_access_ = track_access;
    slot_of_job_.fill(kNoSlot);
    inherited_ = active_;
    filtered = false;
    if (!active_)
      return false;
    // Fast filter: nothing in this block touches a running job's registers.
    const Access &fp = plan.footprint;
    const Access &live_access = access();
    if (!(fp.writes & (live_access.reads | live_access.writes)).any() &&
        !(fp.reads & live_access.writes).any())
      return false;
    filtered = true;
    result_wait.fill(0);
    factor_wait.fill(0);
    for (JobMask m = active_; m; m &= m - 1) {
      const unsigned slot = std::countr_zero(m);
      const Access &job = job_access(slot);
      // A result the block reads or overwrites: join at the first touch. A
      // factor the block overwrites: join at the first write (snapshotted
      // factors are not in `reads`).
      for_each_register(job.writes, [&](unsigned r) {
        if (plan.first_access[r] != Plan::kNoStep)
          result_wait[plan.first_access[r]] |= JobMask{1} << slot;
      });
      for_each_register(job.reads & ~job.writes, [&](unsigned r) {
        if (plan.first_write[r] != Plan::kNoStep)
          factor_wait[plan.first_write[r]] |= JobMask{1} << slot;
      });
    }
    return true;
  }

  /// Physical slots the plan's job mask refers to in this block.
  JobMask slots_of(JobMask plan_jobs) const {
    JobMask slots = 0;
    for (JobMask m = plan_jobs; m; m &= m - 1) {
      const uint8_t slot = slot_of_job_[std::countr_zero(m)];
      if (slot != kNoSlot)
        slots |= JobMask{1} << slot;
    }
    return slots;
  }

  /// Join all jobs of this wave whose helper has finished. Returns true if any
  /// slot was freed. Sets `ok` to false if a helper was unavailable after fork.
  bool join_ready(bool *ok = nullptr) {
    JobMask ready = 0;
    for (JobMask m = active_; m; m &= m - 1) {
      const unsigned slot = std::countr_zero(m);
      if (matrix_coexecution::SharedPool::ready(tickets_[slot]))
        ready |= JobMask{1} << slot;
    }
    if (!ready)
      return false;
    const bool joined = join(ready);
    if (ok)
      *ok = joined;
    return true;
  }

  /// Claims a helper first; `fill` copies snapshot factors into that helper's
  /// own buffer only after the claim succeeded. On pool exhaustion, reclaims
  /// finished jobs of this wave, then sibling waves via `reap`, and optionally
  /// waits on this wave's oldest job. Returns false when the job must run inline.
  template <typename Fill, typename Reap>
  bool submit(unsigned plan_job, const Step &step, void *wave, Fill &&fill, Reap &&reap,
              bool bounded_wait = bounded_wait_enabled(), unsigned start_member = 0) {
    if (cap_ == 0 || start_member >= step.members)
      return false;
    if (active_ == ~JobMask{0} && !join_one_finished())
      return false;
    const bool local_only = plan_ && plan_->local_only;
    matrix_coexecution::SharedPool::Ticket ticket = pool_->claim(cap_, local_only);
    if (!ticket && active_) {
      bool ok = true;
      if (join_ready(&ok)) {
        if (!ok)
          return false;
        ticket = pool_->claim(cap_, local_only);
      }
    }
    if (!ticket && reap())
      ticket = pool_->claim(cap_, local_only);
    if (!ticket && bounded_wait && active_) {
      if (!join(JobMask{1} << oldest_slot()))
        return false;
      ticket = pool_->claim(cap_, local_only);
    }
    if (!ticket)
      return false;
    const unsigned slot = std::countr_zero(~active_);
    tickets_[slot] = ticket;
    const unsigned count = step.members - start_member;
    const MatrixInputSnapshot *const *snapshots =
        step.snapshots ? fill(matrix_coexecution::SharedPool::snapshot_buffer(ticket)) : nullptr;
    matrix_coexecution::SharedPool::start(ticket, step.chain.data() + start_member, count, wave,
                                          snapshots);
    heads_[slot] = &step;
    plans_[slot] = plan_;
    entry_pc_[slot] = block_pc_;
    seq_[slot] = next_seq_++;
    member_offset_[slot] = uint8_t(start_member);
    slot_of_job_[plan_job] = uint8_t(slot);
    active_ |= JobMask{1} << slot;
    if (track_access_ && !access_dirty_) {
      const Access &job = job_access(slot);
      access_.reads |= job.reads;
      access_.writes |= job.writes;
    }
    return true;
  }
  template <typename Fill> bool submit(unsigned plan_job, const Step &step, void *wave, Fill &&fill) {
    return submit(plan_job, step, wave, std::forward<Fill>(fill), [] { return false; });
  }
  /// Join physical slots. Returns false if a helper was unavailable (after fork).
  bool join(JobMask mask) {
    bool ok = true;
    mask &= active_;
    if (!mask)
      return true;
    while (mask) {
      const unsigned slot = std::countr_zero(mask);
      HelperErrorSink error;
      unsigned member = 0;
      ok &= pool_->finish(tickets_[slot], &error, &member).succeeded();
      if (error.error && !failure_) {
        const unsigned idx = member + member_offset_[slot];
        const Instruction &inst = *heads_[slot]->chain[idx];
        const uint64_t pc =
            entry_pc_[slot] + plans_[slot]->steps[heads_[slot]->member_step[idx]].offset;
        failure_ = Failure{std::string(inst.mnemonic()), pc, error};
      }
      active_ &= ~(JobMask{1} << slot);
      inherited_ &= ~(JobMask{1} << slot);
      mask &= mask - 1;
    }
    if (track_access_) {
      if (!active_) {
        access_ = {};
        access_dirty_ = false;
      } else {
        access_dirty_ = true;
      }
    }
    return ok;
  }
  /// The first instruction a helper failed, kept until the issuer reports it.
  struct Failure {
    std::string mnemonic;
    uint64_t pc = 0;
    HelperErrorSink error;
  };
  const std::optional<Failure> &failure() const { return failure_; }
  void clear_failure() { failure_.reset(); }
  bool drain() { return active_ ? join(active_) : true; }
  bool pending() const { return active_ != 0; }
  /// True when jobs are still running OR a reaped job left an unreported error.
  bool has_work() const { return active_ != 0 || failure_.has_value(); }
  JobMask active() const { return active_; }
  /// Physical slots inherited from earlier blocks that are still live.
  JobMask inherited() const { return inherited_; }
  const Access &access() const {
    if (access_dirty_) {
      access_ = {};
      for (JobMask m = active_; m; m &= m - 1) {
        const Access &job = job_access(std::countr_zero(m));
        access_.reads |= job.reads;
        access_.writes |= job.writes;
      }
      access_dirty_ = false;
    }
    return access_;
  }
  /// Slots whose job conflicts with an instruction about to execute on the issuer.
  JobMask conflicting(const Access &next) const {
    JobMask mask = 0;
    for (JobMask m = active_; m; m &= m - 1) {
      const unsigned slot = std::countr_zero(m);
      if (overlaps(job_access(slot), next))
        mask |= JobMask{1} << slot;
    }
    return mask;
  }
  /// Join any job whose helper has finished; otherwise join the oldest job.
  bool join_one_finished() {
    for (JobMask m = active_; m; m &= m - 1) {
      const unsigned slot = std::countr_zero(m);
      if (matrix_coexecution::SharedPool::ready(tickets_[slot]))
        return join(JobMask{1} << slot);
    }
    return join(JobMask{1} << oldest_slot());
  }

private:
  const Access &job_access(unsigned slot) const {
    return plans_[slot]->job_accesses[heads_[slot]->job];
  }
  unsigned oldest_slot() const {
    JobMask m = active_;
    unsigned oldest = std::countr_zero(m);
    for (m &= m - 1; m; m &= m - 1) {
      const unsigned slot = std::countr_zero(m);
      if (seq_[slot] < seq_[oldest])
        oldest = slot;
    }
    return oldest;
  }
  static constexpr uint8_t kNoSlot = 0xff;
  matrix_coexecution::SharedPool *pool_ = nullptr;
  const Plan *plan_ = nullptr;
  uint64_t block_pc_ = 0;
  unsigned cap_ = ~0u;
  bool track_access_ = false;
  mutable bool access_dirty_ = false;
  uint64_t next_seq_ = 0;
  std::array<uint8_t, kMaxJobs> slot_of_job_{};
  std::array<matrix_coexecution::SharedPool::Ticket, kMaxJobs> tickets_{};
  std::array<const Step *, kMaxJobs> heads_{};
  std::array<const Plan *, kMaxJobs> plans_{};
  std::array<uint64_t, kMaxJobs> entry_pc_{};
  std::array<uint64_t, kMaxJobs> seq_{};
  std::array<uint8_t, kMaxJobs> member_offset_{};
  mutable Access access_;
  std::optional<Failure> failure_;
  JobMask active_ = 0;
  JobMask inherited_ = 0;
};

/// @brief Per-CU plans keyed like MmaAdmissionCache, retaining decoded instructions.
class PlanCache {
public:
  using Words = Plan::Words;
  struct Counters {
    uint64_t decodes = 0, builds = 0, accept = 0, reject = 0, hits = 0, validations = 0;
    uint64_t replays = 0, offloads = 0, chained = 0, issuer_mma = 0, alu = 0, ordinary = 0;
    uint64_t pool_full = 0, early_exit = 0, entry_rejects = 0;
    uint64_t snapshot_chains = 0, snapshot_bytes = 0;
    /// Cross-block: blocks entered with jobs outstanding, of which the fast
    /// filter found a conflict; joins forced by a running job's result or
    /// factor; drains by cause.
    uint64_t pending_entries = 0, filtered_entries = 0, result_waits = 0, factor_waits = 0;
    uint64_t drains_gap = 0, drains_mode = 0, drains_fault = 0, drains_wave_end = 0,
             drains_evict = 0, drains_reject = 0, drains_quantum = 0;
    /// Instructions issued outside a plan while jobs ran: footprint checked
    /// and found independent, or joined the jobs they touched.
    uint64_t gap_passes = 0, gap_joins = 0;
  };

  explicit PlanCache(size_t capacity = 4096) : capacity_(capacity) {}
  ~PlanCache() {
    drain_all(stats.drains_evict);
    flush();
  }

  struct Key {
    uint64_t pc;
    uint32_t vmid, vgprs;
    bool acc;
    VmCacheNamespace cache_namespace;
    uint32_t wave_size = 32;
    bool operator==(const Key &) const = default;
  };

  /// @brief Find or build the plan for a block entry; check accept before replaying.
  template <typename Fetch>
  const Plan *lookup(Decoder &decoder, const InstructionCache &icache, Fetch &&fetch_words,
                     const Key &key, const Words &first) {
    auto it = plans_.find(key);
    bool valid = it != plans_.end() && it->second.first == first;
    if (valid && it->second.epoch != icache.epoch()) {
      ++stats.validations;
      for (const auto &snapshot : it->second.code) {
        Words words;
        if (!fetch_words(key.pc + snapshot.offset, words) || words != snapshot.words) {
          valid = false;
          break;
        }
      }
      it->second.epoch = icache.epoch();
    }
    if (!valid) {
      // Outstanding jobs point into plans; finish them before any plan dies.
      if (it != plans_.end() || plans_.size() == capacity_)
        drain_all(stats.drains_evict);
      if (it == plans_.end() && plans_.size() == capacity_) {
        plans_.clear();
        it = plans_.end();
      }
      if (it == plans_.end())
        it = plans_.try_emplace(key).first;
      Plan &plan = it->second;
      plan = Plan{};
      plan.epoch = icache.epoch();
      // Preserve entry words even if compile() rejects at step 0 so repeated
      // hits do not look like code modification and drain all waves' helpers.
      plan.first = first;
      ++stats.builds;
      compile(plan, decoder, fetch_words, key.pc, key.vgprs, key.acc, stats.decodes,
              cross_block_enabled() && tail_snapshot_enabled(), batch_pack_limit(),
              batch_min_chains(), min_job_weight(), key.wave_size);
      ++(plan.accept ? stats.accept : stats.reject);
    } else {
      ++stats.hits;
    }
    return &it->second;
  }

  /// Wave whose lazy VGPR chunks were last allocated in a slot.
  struct Materialized {
    const void *wave = nullptr;
    uint64_t generation = ~uint64_t{0};
    uint32_t base = 0, count = 0;
    bool operator==(const Materialized &) const = default;
  };
  Materialized &materialized(uint32_t wave_slot) {
    if (wave_slot >= materialized_.size())
      materialized_.resize(wave_slot + 1);
    return materialized_[wave_slot];
  }

  /// Jobs of each wave slot that may still be running.
  PendingJobs &pending(uint32_t wave_slot, matrix_coexecution::SharedPool &pool) {
    if (wave_slot >= pending_.size())
      pending_.resize(wave_slot + 1);
    pending_[wave_slot].bind(pool);
    return pending_[wave_slot];
  }
  PendingJobs *pending_if_any(uint32_t wave_slot) {
    return wave_slot < pending_.size() && pending_[wave_slot].has_work() ? &pending_[wave_slot]
                                                                         : nullptr;
  }
  /// Free helpers held by other waves of this CU whose jobs already finished.
  bool reap_finished_except(uint32_t wave_slot) {
    bool freed = false;
    for (uint32_t w = 0; w != pending_.size(); ++w)
      if (w != wave_slot && pending_[w].pending())
        freed |= pending_[w].join_ready();
    return freed;
  }
  void drain_all(uint64_t &counter) {
    for (auto &jobs : pending_)
      if (jobs.pending()) {
        ++counter;
        (void)jobs.drain();
      }
  }

  /// @brief Destroy every retained decode, e.g. when the decoder is replaced.
  void clear() {
    drain_all(stats.drains_evict);
    plans_.clear();
  }

  void flush() {
    if (!(stats.builds || stats.replays || stats.entry_rejects || stats.pool_full) ||
        !async_execution::report_enabled())
      return;
    std::fprintf(stderr, "%s\n",
                 std::format("RJ_PLAN builds={} accept={} reject={} hits={} validations={} "
                             "decodes={} replays={} offloads={} chained={} issuer_mma={} alu={} "
                             "ordinary={} pool_full={} early_exit={} entry_rejects={} "
                             "snapshot_chains={} snapshot_bytes={} pending_entries={} "
                             "filtered={} result_waits={} factor_waits={} drains(gap={} mode={} "
                             "fault={} wave_end={} evict={} reject={} quantum={}) gap_passes={} "
                             "gap_joins={}",
                             stats.builds, stats.accept, stats.reject, stats.hits,
                             stats.validations, stats.decodes, stats.replays, stats.offloads,
                             stats.chained, stats.issuer_mma, stats.alu, stats.ordinary,
                             stats.pool_full, stats.early_exit, stats.entry_rejects,
                             stats.snapshot_chains, stats.snapshot_bytes, stats.pending_entries,
                             stats.filtered_entries, stats.result_waits, stats.factor_waits,
                             stats.drains_gap, stats.drains_mode, stats.drains_fault,
                             stats.drains_wave_end, stats.drains_evict, stats.drains_reject,
                             stats.drains_quantum, stats.gap_passes, stats.gap_joins)
                     .c_str());
    stats = {};
  }

  Counters stats;

private:
  struct Hash {
    size_t operator()(const Key &key) const {
      size_t hash = std::hash<uint64_t>{}(key.pc ^ (uint64_t{key.vmid} << 32));
      hash ^= size_t{key.vgprs} << 1;
      hash ^= key.acc;
      hash ^= size_t{key.wave_size} << 9;
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.address_space.generation);
      hash ^= size_t{key.cache_namespace.address_space.slot} << 2;
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.translation_epoch);
      return hash;
    }
  };
  size_t capacity_;
  std::unordered_map<Key, Plan, Hash> plans_;
  std::vector<Materialized> materialized_;
  std::vector<PendingJobs> pending_;
};

} // namespace rocjitsu::amdgpu::mma_plan
