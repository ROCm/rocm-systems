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
#include <vector>

namespace rocjitsu::amdgpu::mma_plan {

// Long enough for unrolled K steps with their epilogues; joins use helper slots, not steps.
inline constexpr unsigned kMaxSteps = 64;
inline constexpr unsigned kSlots = async_execution::kMaxPendingMma;
inline constexpr unsigned kMaxChain = matrix_coexecution::Helper::kMaxChain;
static_assert(kSlots <= 8, "wait masks are one byte");

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
  uint8_t a_regs = 0, b_regs = 0;
};

struct Step {
  StepKind kind = StepKind::Ordinary;
  uint8_t size = 0;
  uint16_t offset = 0; ///< Bytes from the block entry.
  uint8_t slot = 0;
  uint8_t wait = 0; ///< Helper slots joined before the step.
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
  if (inst.is_memory_op() || inst.is_waitcnt() || inst.data())
    return false;
  return operation == Operation::S_SETPRIO || operation == Operation::S_MOVK_I32 ||
         operation == Operation::S_ADDK_I32 ||
         async_mma_policy::inline_operation_kind(operation) ==
             async_mma_policy::InlineOperationKind::Arithmetic;
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
/// RJ_MMA_PLAN_INLINE=1 replays plans entirely on the issuer, isolating the
/// saved fetch, decode and scoreboard work from helper parallelism.
inline bool inline_only() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_MMA_PLAN_INLINE");
    return value && *value && *value != '0';
  }();
  return enabled;
}

using Access = async_execution::Access;
inline bool overlaps(const Access &a, const Access &b) { return a.conflicts(b) || b.conflicts(a); }

/// @brief Register operands of one chain member, split into factors and the rest.
struct MemberOperands {
  unsigned step = 0;
  uint16_t a_base = 0, b_base = 0;
  uint8_t a_regs = 0, b_regs = 0;
  bool a_snap = false, b_snap = false;
  Access other; ///< Accumulator, scale and result registers.

  decltype(Access::reads) factors(bool snapped_only) const {
    decltype(Access::reads) bits;
    for (unsigned r = 0; r != a_regs; ++r)
      if (!snapped_only || a_snap)
        bits.set(a_base + r);
    for (unsigned r = 0; r != b_regs; ++r)
      if (!snapped_only || b_snap)
        bits.set(b_base + r);
    return bits;
  }
};

inline std::optional<MemberOperands> member_operands(const Instruction &inst, unsigned step,
                                                     uint32_t num_vgprs, bool has_accvgprs) {
  MemberOperands member;
  member.step = step;
  const auto range = [&](const Operand *operand, uint16_t &base,
                         uint8_t &regs) -> std::optional<bool> {
    const auto ref = operand ? operand->to_register_ref() : std::nullopt;
    if (!ref || (ref->cls != RegClass::VGPR && ref->cls != RegClass::ACC_VGPR))
      return false;
    const bool acc = ref->cls == RegClass::ACC_VGPR;
    const uint32_t limit = acc ? (has_accvgprs ? Access::kRegistersPerBank : 0)
                               : std::min(Access::kRegistersPerBank, num_vgprs);
    if (ref->index >= limit || ref->width > limit - ref->index)
      return std::nullopt;
    base = uint16_t(ref->index + (acc ? Access::kRegistersPerBank : 0));
    regs = uint8_t(ref->width);
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
  for (unsigned r = 0; r != regs; ++r)
    member.other.writes.set(base + r);
  return member;
}

struct Plan {
  using Words = std::array<uint32_t, 4>;
  struct Snapshot {
    uint64_t offset;
    Words words;
  };
  bool accept = false;
  uint64_t epoch = 0;
  uint64_t length = 0; ///< Bytes covered by the steps.
  Words first{};
  std::vector<Snapshot> code;
  std::vector<Step> steps;
  std::vector<std::unique_ptr<Instruction>> owned;
  unsigned chains = 0, chained = 0, issuer_mma = 0, alu = 0, ordinary = 0;

  Plan() = default;
  Plan(Plan &&) = default;
  Plan &operator=(Plan &&other) {
    release();
    accept = other.accept;
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
/// Scheduling first forms logical chains with no slot limit, so a block never
/// joins a running chain early just to free a slot. A post-pass then places
/// them on physical helpers and the issuer.
template <typename Fetch>
void compile(Plan &plan, Decoder &decoder, Fetch &&fetch_words, uint64_t pc, uint32_t num_vgprs,
             bool has_accvgprs, uint64_t &decodes) {
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
                             std::initializer_list<int> own) {
    for (unsigned j = from + 1; j < at; ++j) {
      if (std::find(own.begin(), own.end(), step_chain[j]) != own.end())
        continue;
      if (plan.steps[j].kind == StepKind::Ordinary || overlaps(accesses[j], access))
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
    // Stay in the fetchable code page of the block entry.
    if ((pc % GpuMemory::PAGE_SIZE) + offset + sizeof(Plan::Words) > GpuMemory::PAGE_SIZE)
      break;
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
    const auto access = async_execution::footprint(*inst, num_vgprs, has_accvgprs);
    if (!access)
      break;
    plan.code.push_back({offset, words});
    Step step;
    step.size = static_cast<uint8_t>(inst->size());
    step.offset = static_cast<uint16_t>(offset);
    std::optional<MemberOperands> operands;
    if (offloadable(*inst, has_accvgprs))
      operands = member_operands(*inst, k, num_vgprs, has_accvgprs);
    // Placeholders keep step-indexed tables aligned while this step is decided.
    accesses.push_back(*access);
    step_chain.push_back(-1);
    joins.emplace_back();
    const auto conflicts = conflicts_of(*access);
    if (operands) {
      // A same-accumulator successor joins its producer's chain: one handoff.
      int bundle = -1;
      if (chaining_enabled() && conflicts.size() == 1) {
        const unsigned c = conflicts.front();
        if (chains[c].open && chains[c].count < kMaxChain &&
            hoistable(*access, chains[c].head, k, {int(c)}))
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
      if (step.kind == StepKind::Ordinary)
        for (unsigned c : live)
          chains[c].open = false;
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

  // Profitability: offloading pays only while another chain runs at the same
  // time. A serial chain (even one split at kMaxChain) never qualifies.
  unsigned max_concurrent = 0;
  for (unsigned t = 0; t != steps; ++t) {
    unsigned alive = 0;
    for (const auto &chain : chains)
      alive += chain.head <= t && t < chain.joined;
    max_concurrent = std::max(max_concurrent, alive);
  }
  // Placement. Within each group of chains with overlapping lifetimes, the
  // last-started chain stays on the issuer so it does useful matrix work, as
  // does any chain that overlaps no other. The rest take free helpers in head
  // order; a chain finding none is appended to a running helper job when it
  // can run that early, and otherwise also stays on the issuer.
  constexpr int kIssuer = -1;
  std::vector<int> placed(chains.size(), kIssuer); // helper slot, or kIssuer
  std::vector<int> packed_into(chains.size(), -1);  // host chain when appended
  std::vector<unsigned> job_size(chains.size(), 0); // members of each host's helper job
  if (max_concurrent >= 2) {
    std::vector<bool> issuer(chains.size(), false);
    for (unsigned c = 0; c != chains.size();) {
      unsigned end = chains[c].joined, last = c, size = 0;
      unsigned g = c;
      for (; g != chains.size() && chains[g].head < end; ++g) {
        end = std::max(end, chains[g].joined);
        last = g;
        ++size;
      }
      issuer[last] = true;
      (void)size;
      c = g;
    }
    std::array<int, kSlots> occupant;
    occupant.fill(-1);
    for (unsigned c = 0; c != chains.size(); ++c) {
      if (issuer[c])
        continue;
      int slot = -1;
      for (unsigned s = 0; s != kSlots && slot < 0; ++s)
        if (occupant[s] < 0 || chains[occupant[s]].joined <= chains[c].head)
          slot = int(s);
      if (slot >= 0) {
        occupant[slot] = int(c);
        placed[c] = slot;
        job_size[c] = chains[c].count;
        continue;
      }
      // Pack onto a running helper job: its members must all be able to run at
      // that job's submission.
      for (unsigned s = 0; s != kSlots; ++s) {
        const int h = occupant[s];
        const unsigned host = packed_into[h] >= 0 ? unsigned(packed_into[h]) : unsigned(h);
        if (job_size[host] + chains[c].count > kMaxChain || chains[h].joined <= chains[c].head)
          continue;
        bool ok = true;
        for (unsigned i = 0; ok && i != chains[c].count; ++i) {
          const unsigned at = chains[c].members[i].step;
          ok = hoistable(accesses[at], chains[host].head, at, {int(host), int(c)});
        }
        if (!ok)
          continue;
        packed_into[c] = int(host);
        placed[c] = int(s);
        job_size[host] += chains[c].count;
        break;
      }
    }
  }

  // Materialize steps: helper jobs, issuer MMAs and physical join masks.
  unsigned helper_chains = 0;
  for (unsigned c = 0; c != chains.size(); ++c)
    helper_chains += placed[c] != kIssuer && packed_into[c] < 0;
  plan.accept = helper_chains >= 1;
  if (!plan.accept)
    return;
  for (unsigned k = 0; k != steps; ++k) {
    auto &step = plan.steps[k];
    const int c = step_chain[k];
    if (c >= 0) {
      if (placed[c] == kIssuer) {
        step.kind = StepKind::IssuerMma;
      } else {
        const unsigned host = packed_into[c] >= 0 ? unsigned(packed_into[c]) : unsigned(c);
        step.slot = uint8_t(placed[c]);
        if (unsigned(c) == host && k == chains[c].head) {
          step.kind = StepKind::Offload;
          step.members = 0;
        } else {
          step.kind = StepKind::Chained;
        }
      }
    }
    for (unsigned j : joins[k])
      if (placed[j] != kIssuer)
        step.wait |= uint8_t(1u << placed[j]);
  }
  // Fill each helper job's member list in program order of its host chain,
  // then of the chains packed onto it.
  for (unsigned c = 0; c != chains.size(); ++c) {
    if (placed[c] == kIssuer)
      continue;
    const unsigned host = packed_into[c] >= 0 ? unsigned(packed_into[c]) : c;
    auto &head = plan.steps[chains[host].head];
    for (unsigned i = 0; i != chains[c].count; ++i) {
      const auto &member = chains[c].members[i];
      const unsigned at = member.step;
      auto &spec = head.snapshot[head.members];
      if (member.a_snap)
        spec.a_base = member.a_base, spec.a_regs = member.a_regs;
      if (member.b_snap)
        spec.b_base = member.b_base, spec.b_regs = member.b_regs;
      head.snapshots |= member.a_snap || member.b_snap;
      head.member_step[head.members] = uint8_t(at);
      head.chain[head.members++] = plan.steps[at].inst;
    }
  }
  for (const auto &step : plan.steps) {
    plan.chains += step.kind == StepKind::Offload;
    plan.chained += step.kind == StepKind::Chained;
    plan.issuer_mma += step.kind == StepKind::IssuerMma;
    plan.alu += step.kind == StepKind::Alu;
    plan.ordinary += step.kind == StepKind::Ordinary;
  }
}

/// @brief Helper chains of one replayed plan, owned by the issuing thread.
/// @details Every claimed helper is joined before the wave can leave the
/// issuer: at the end of the block, on an early exit, and before fault paths.
class Replay {
public:
  explicit Replay(matrix_coexecution::SharedPool &pool) : pool_(pool) {}
  ~Replay() { (void)drain(); }
  Replay(const Replay &) = delete;
  Replay &operator=(const Replay &) = delete;

  /// Claims a helper first; `fill` copies snapshot factors into that helper's
  /// own buffer only after the claim succeeded.
  template <typename Fill>
  bool submit(unsigned slot, const Step &step, void *wave, Fill &&fill) {
    tickets_[slot] = pool_.claim();
    if (!tickets_[slot])
      return false;
    const MatrixInputSnapshot *const *snapshots =
        step.snapshots ? fill(matrix_coexecution::SharedPool::snapshot_buffer(tickets_[slot]))
                       : nullptr;
    matrix_coexecution::SharedPool::start(tickets_[slot], step.chain.data(), step.members, wave,
                                          snapshots);
    heads_[slot] = &step;
    active_ |= uint8_t(1u << slot);
    return true;
  }
  /// Returns false if a helper was unavailable (after fork).
  bool join(uint8_t mask) {
    bool ok = true;
    mask &= active_;
    while (mask) {
      const unsigned slot = std::countr_zero(mask);
      HelperErrorSink error;
      unsigned member = 0;
      ok &= pool_.finish(tickets_[slot], &error, &member).succeeded();
      if (error.error && !failure_)
        failure_ = Failure{heads_[slot], member, error};
      active_ &= uint8_t(~(1u << slot));
      mask &= uint8_t(mask - 1);
    }
    return ok;
  }
  /// The first instruction a helper failed, kept until the issuer reports it.
  struct Failure {
    const Step *head;
    unsigned member;
    HelperErrorSink error;
  };
  const std::optional<Failure> &failure() const { return failure_; }
  bool drain() { return join(active_); }
  bool pending() const { return active_ != 0; }

private:
  matrix_coexecution::SharedPool &pool_;
  std::array<matrix_coexecution::SharedPool::Ticket, kSlots> tickets_{};
  std::array<const Step *, kSlots> heads_{};
  std::optional<Failure> failure_;
  uint8_t active_ = 0;
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
  };

  explicit PlanCache(size_t capacity = 4096) : capacity_(capacity) {}
  ~PlanCache() { flush(); }

  struct Key {
    uint64_t pc;
    uint32_t vmid, vgprs;
    bool acc;
    VmCacheNamespace cache_namespace;
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
      if (it == plans_.end() && plans_.size() == capacity_) {
        plans_.clear();
        it = plans_.end();
      }
      if (it == plans_.end())
        it = plans_.try_emplace(key).first;
      Plan &plan = it->second;
      plan = Plan{};
      plan.epoch = icache.epoch();
      ++stats.builds;
      compile(plan, decoder, fetch_words, key.pc, key.vgprs, key.acc, stats.decodes);
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

  /// @brief Destroy every retained decode, e.g. when the decoder is replaced.
  void clear() { plans_.clear(); }

  void flush() {
    if (!(stats.builds || stats.replays || stats.entry_rejects || stats.pool_full) ||
        !async_execution::report_enabled())
      return;
    std::fprintf(stderr, "%s\n",
                 std::format("RJ_PLAN builds={} accept={} reject={} hits={} validations={} "
                             "decodes={} replays={} offloads={} chained={} issuer_mma={} alu={} "
                             "ordinary={} pool_full={} early_exit={} entry_rejects={} "
                             "snapshot_chains={} snapshot_bytes={}",
                             stats.builds, stats.accept, stats.reject, stats.hits,
                             stats.validations, stats.decodes, stats.replays, stats.offloads,
                             stats.chained, stats.issuer_mma, stats.alu, stats.ordinary,
                             stats.pool_full, stats.early_exit, stats.entry_rejects,
                             stats.snapshot_chains, stats.snapshot_bytes)
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
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.address_space.generation);
      hash ^= size_t{key.cache_namespace.address_space.slot} << 2;
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.translation_epoch);
      return hash;
    }
  };
  size_t capacity_;
  std::unordered_map<Key, Plan, Hash> plans_;
  std::vector<Materialized> materialized_;
};

} // namespace rocjitsu::amdgpu::mma_plan
