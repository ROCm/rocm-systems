// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_plan.h
/// @brief Straight-line MMA issue plans with helper chains, compiled once per block.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
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

inline constexpr unsigned kMaxSteps = 32;
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

struct Step {
  StepKind kind = StepKind::Ordinary;
  uint8_t size = 0;
  uint8_t slot = 0;
  uint8_t wait = 0; ///< Helper slots joined before the step.
  uint8_t members = 0;
  Instruction *inst = nullptr;
  std::array<Instruction *, kMaxChain> chain{};
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
template <typename Fetch>
void compile(Plan &plan, Decoder &decoder, Fetch &&fetch_words, uint64_t pc, uint32_t num_vgprs,
             bool has_accvgprs, uint64_t &decodes) {
  Instruction::ScopedHeapAllocation heap;
  struct Chain {
    bool active = false, open = false;
    unsigned head = 0, id = 0;
    Access access;
  };
  std::array<Chain, kSlots> slots{};
  // Every helper chain ever started, with the step that joined it. A bundled
  // member runs early, concurrently with any chain alive at or after its head.
  struct History {
    Access access;
    unsigned head = 0, joined = ~0u;
  };
  std::vector<History> history;
  std::vector<Access> accesses;
  std::vector<int8_t> step_slot; // helper slot executing each step, -1 on the issuer
  uint64_t offset = 0;
  unsigned age = 0;
  std::array<unsigned, kSlots> started{};
  unsigned current = 0;
  const auto join = [&](uint8_t mask) {
    for (unsigned s = 0; s != kSlots; ++s)
      if ((mask & (1u << s)) && slots[s].active) {
        history[slots[s].id].joined = current;
        slots[s] = {};
      }
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
    uint8_t conflicts = 0;
    for (unsigned s = 0; s != kSlots; ++s)
      if (slots[s].active && overlaps(slots[s].access, *access))
        conflicts |= uint8_t(1u << s);
    if (offloadable(*inst, has_accvgprs)) {
      // A same-accumulator successor joins a running chain only when its other
      // inputs already exist at the chain's head: no step in between may touch
      // its registers, and none may be able to stall or fault, since an early
      // exit would otherwise re-execute an accumulate the helper already ran.
      int bundle = -1;
      if (chaining_enabled() && std::has_single_bit(conflicts)) {
        const unsigned h = std::countr_zero(conflicts);
        bool ok = slots[h].open;
        for (unsigned j = slots[h].head + 1; ok && j != k; ++j) {
          if (step_slot[j] == int8_t(h))
            continue;
          ok = plan.steps[j].kind != StepKind::Ordinary && !overlaps(accesses[j], *access);
        }
        for (unsigned c = 0; ok && c != history.size(); ++c)
          ok = c == slots[h].id || history[c].joined <= slots[h].head ||
               !overlaps(history[c].access, *access);
        auto &head = plan.steps[slots[h].head];
        if (ok && head.members < kMaxChain)
          bundle = int(h);
      }
      if (bundle >= 0) {
        auto &chain = slots[bundle];
        auto &head = plan.steps[chain.head];
        head.chain[head.members++] = inst.get();
        chain.access.reads |= access->reads;
        chain.access.writes |= access->writes;
        history[chain.id].access = chain.access;
        step.kind = StepKind::Chained;
        step.slot = uint8_t(bundle);
        step_slot.push_back(int8_t(bundle));
      } else {
        step.wait = conflicts;
        join(conflicts);
        unsigned slot = kSlots;
        for (unsigned s = 0; s != kSlots && slot == kSlots; ++s)
          if (!slots[s].active)
            slot = s;
        if (slot == kSlots) {
          // Reuse the oldest chain's slot after joining it.
          slot = 0;
          for (unsigned s = 1; s != kSlots; ++s)
            if (started[s] < started[slot])
              slot = s;
          step.wait |= uint8_t(1u << slot);
          join(uint8_t(1u << slot));
        }
        slots[slot] = {true, true, k, unsigned(history.size()), *access};
        history.push_back({*access, k});
        started[slot] = ++age;
        step.kind = StepKind::Offload;
        step.slot = uint8_t(slot);
        step.chain[0] = inst.get();
        step.members = 1;
        step_slot.push_back(int8_t(slot));
      }
      step.inst = inst.get();
    } else {
      step.wait = conflicts;
      join(conflicts);
      // Retain only stateless arithmetic. Memory instructions keep per-issue
      // state and go through ordinary issue and its decoded-instruction cache.
      if (alu_enabled() && pure_alu(*inst)) {
        step.kind = StepKind::Alu;
        step.inst = inst.get();
      }
      // An issuer step that could stall or fault closes every running chain
      // to later members (see above).
      if (step.kind == StepKind::Ordinary)
        for (auto &slot : slots)
          slot.open = false;
      step_slot.push_back(-1);
    }
    if (step.inst)
      plan.owned.push_back(std::move(inst));
    accesses.push_back(*access);
    plan.steps.push_back(step);
    offset += step.size;
  }
  plan.length = offset;
  // Keep the last independent chain on the issuer so it does useful matrix
  // work instead of only waiting. A block with fewer than two chains offloads nothing.
  unsigned chains = 0, last = 0;
  for (unsigned k = 0; k != plan.steps.size(); ++k)
    if (plan.steps[k].kind == StepKind::Offload) {
      ++chains;
      last = k;
    }
  plan.accept = chains >= 2;
  if (!plan.accept)
    return;
  const uint8_t issuer_slot = plan.steps[last].slot;
  for (unsigned k = last; k != plan.steps.size(); ++k) {
    auto &step = plan.steps[k];
    if ((step.kind == StepKind::Offload && k == last) ||
        (step.kind == StepKind::Chained && step.slot == issuer_slot && k > last))
      step.kind = StepKind::IssuerMma;
    // Later waits on the issuer's chain are already satisfied in program order.
    if (k > last && step.kind != StepKind::Offload)
      step.wait &= uint8_t(~(1u << issuer_slot));
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

  bool submit(unsigned slot, Instruction *const *chain, unsigned count, void *wave) {
    tickets_[slot] = pool_.submit(chain, count, wave);
    if (!tickets_[slot])
      return false;
    active_ |= uint8_t(1u << slot);
    return true;
  }
  /// Returns false if a helper was unavailable (after fork).
  bool join(uint8_t mask) {
    bool ok = true;
    mask &= active_;
    while (mask) {
      const unsigned slot = std::countr_zero(mask);
      ok &= pool_.finish(tickets_[slot]).succeeded();
      active_ &= uint8_t(~(1u << slot));
      mask &= uint8_t(mask - 1);
    }
    return ok;
  }
  bool drain() { return join(active_); }
  bool pending() const { return active_ != 0; }

private:
  matrix_coexecution::SharedPool &pool_;
  std::array<matrix_coexecution::SharedPool::Ticket, kSlots> tickets_{};
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

  /// @brief Find or build the plan for a block entry; nullptr if it offloads nothing.
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
    return it->second.accept ? &it->second : nullptr;
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
                             "ordinary={} pool_full={} early_exit={} entry_rejects={}",
                             stats.builds, stats.accept, stats.reject, stats.hits,
                             stats.validations, stats.decodes, stats.replays, stats.offloads,
                             stats.chained, stats.issuer_mma, stats.alu, stats.ordinary,
                             stats.pool_full, stats.early_exit, stats.entry_rejects)
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
};

} // namespace rocjitsu::amdgpu::mma_plan
