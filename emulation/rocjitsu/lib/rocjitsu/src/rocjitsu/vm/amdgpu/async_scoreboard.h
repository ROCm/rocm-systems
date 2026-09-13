// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"

#include <bitset>

namespace rocjitsu::amdgpu {
namespace async_execution {
namespace mc = matrix_coexecution;
inline unsigned window_limit() {
  static const unsigned value = [] {
    const char *text = std::getenv("RJ_ASYNC_WINDOW");
    return static_cast<unsigned>(std::clamp(text ? std::atoi(text) : 32, 1, 256));
  }();
  return value;
}
struct Access {
  std::bitset<512> reads, writes;
  bool conflicts(const Access &next) const {
    return (writes & (next.reads | next.writes)).any() || (reads & next.writes).any();
  }
};
inline bool ordinary_memory(std::string_view name) {
  return name.starts_with("global_load_") || name.starts_with("global_store_") ||
         name.starts_with("buffer_load_") || name.starts_with("buffer_store_");
}
inline bool safe_inline(const Instruction &inst) {
  if (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR |
                      BARRIER | WRITES_EXEC))
    return false;
  const auto name = inst.mnemonic();
  if (name == "s_delay_alu" || name == "s_nop" || name == "v_nop")
    return true;
  if (mc::async_candidate(name) || ordinary_memory(name))
    return true;
  if (inst.is_waitcnt())
    return !name.starts_with("s_wait_alu");
  // Explicitly exclude instructions with hidden EXEC, MODE, cache, scheduling,
  // register-allocation or cross-lane register-index side effects.
  constexpr std::string_view prefixes[] = {
      "s_mov_",     "s_add_",      "s_addc_",     "s_sub_",     "s_subb_", "s_mul_",    "s_mad_",
      "s_and_",     "s_or_",       "s_xor_",      "s_lshl_",    "s_lshr_", "s_ashr_",   "s_cmp_",
      "s_cselect_", "v_mov_",      "v_add_",      "v_addc_",    "v_sub_",  "v_subrev_", "v_mul_",
      "v_mad_",     "v_fma_",      "v_lshl",      "v_lshr",     "v_ashr",  "v_and_",    "v_or_",
      "v_xor_",     "v_cvt_",      "v_cmp_",      "v_cndmask_", "v_max_",  "v_min_",    "v_bfe_",
      "v_bfi_",     "v_alignbit_", "v_alignbyte_"};
  if (!std::ranges::any_of(prefixes, [&](auto prefix) { return name.starts_with(prefix); }))
    return false;
  for (int i = 0; i != inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    // Ordinary SGPRs only. Fieldless SCC/VCC are permitted by the whitelist;
    // WMMA workers do not read them. Explicit special-register writes drain.
    if (op && !op->is_vgpr() && !op->is_fieldless() &&
        (op->encoding_value() < 0 || op->encoding_value() + (op->size_bits() + 31) / 32 > 102))
      return false;
  }
  return true;
}
inline std::optional<Access> footprint(const Instruction &inst, uint32_t num_vgprs,
                                       bool has_accvgprs = false) {
  Access access;
  for (bool dst : {false, true}) {
    const int count = dst ? inst.num_dst_operands() : inst.num_src_operands();
    for (int i = 0; i != count; ++i) {
      const auto *op = dst ? inst.dst_operand(i) : inst.src_operand(i);
      if (!op || !op->is_vgpr())
        continue;
      // is_vgpr() describes selector capability, including scalar and inline
      // encodings. Resolve the actual register and packed-half aliases using
      // the ISA adapter; the queue itself stays independent of the ISA.
      const auto ref = op->to_register_ref();
      if (!ref || (ref->cls != RegClass::VGPR && ref->cls != RegClass::ACC_VGPR))
        continue;
      const bool acc = ref->cls == RegClass::ACC_VGPR;
      const uint32_t limit = acc ? (has_accvgprs ? 256 : 0) : std::min(256u, num_vgprs);
      const uint32_t index = ref->index, width = ref->width;
      if (index >= limit || width > limit - index)
        return std::nullopt;
      const uint32_t reg = index + (acc ? 256 : 0);
      for (uint32_t r = reg; r != reg + width; ++r) {
        (dst ? access.writes : access.reads).set(r);
        // A store's encoding can describe its data as a destination operand.
        if (inst.is_memory_op())
          access.reads.set(r);
      }
    }
  }
  return access;
}
struct Stats {
  uint64_t windows = 0, mma = 0, inline_overlap = 0;
  uint64_t hazards = 0, boundaries = 0, full = 0, retired_mma = 0;
  ~Stats() { flush(); }
  void flush() {
    if (windows)
      std::fprintf(stderr,
                   "RJ_ASYNC windows=%llu mma=%llu inline_overlap=%llu "
                   "hazards=%llu boundaries=%llu full=%llu retired_mma=%llu\n",
                   (unsigned long long)windows, (unsigned long long)mma,
                   (unsigned long long)inline_overlap, (unsigned long long)hazards,
                   (unsigned long long)boundaries, (unsigned long long)full,
                   (unsigned long long)retired_mma);
    windows = mma = inline_overlap = hazards = boundaries = full = retired_mma = 0;
  }
};
inline thread_local Stats stats;
} // namespace async_execution

// Publication follows issue order. Completion can require an ordered prefix
// or release independent arithmetic dependencies individually. Retirement
// always runs on the instruction allocator's owner.
class AsyncInstructionQueue {
public:
  using Access = async_execution::Access;
  using Pool = matrix_coexecution::SharedPool;
  using Retire = void (*)(void *, Instruction *, bool);
  enum class Completion { Ordered, Independent };
  AsyncInstructionQueue(void *owner, Retire retire, Completion completion,
                        Pool &pool = matrix_coexecution::shared_pool())
      : owner_(owner), retire_(retire), completion_(completion), pool_(pool) {}
  ~AsyncInstructionQueue() { assert(empty()); }
  bool empty() const { return count_ == 0; }
  size_t size() const { return count_; }
  bool submit(Instruction *owned, Instruction &work, void *context, const Access &access) {
    if (count_ == entries_.size())
      return false;
    auto ticket = pool_.submit(work, context);
    if (!ticket)
      return false;
    entries_[count_++] = {owned, ticket, access};
    return true;
  }
  void poll() {
    for (size_t i = 0; i != count_;) {
      if (Pool::ready(entries_[i].ticket))
        retire(i);
      else if (completion_ == Completion::Ordered)
        break;
      else
        ++i;
    }
  }
  bool wait_conflicts(const Access &access) {
    bool waited = false;
    if (completion_ == Completion::Ordered) {
      size_t through = 0;
      for (size_t i = 0; i != count_; ++i)
        if (entries_[i].access.conflicts(access))
          through = i + 1;
      waited = through != 0;
      while (through--)
        retire(0);
    } else {
      for (size_t i = 0; i != count_;) {
        if (entries_[i].access.conflicts(access)) {
          waited = true;
          retire(i);
        } else
          ++i;
      }
    }
    return waited;
  }
  void drain() {
    while (!empty())
      retire(0);
  }
  std::exception_ptr take_error() { return std::exchange(error_, {}); }

private:
  struct Entry {
    Instruction *inst = nullptr;
    Pool::Ticket ticket;
    Access access;
  };
  void retire(size_t index) {
    auto &entry = entries_[index];
    auto error = pool_.finish(entry.ticket);
    try {
      retire_(owner_, entry.inst, bool(error));
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
    if (error && !error_)
      error_ = error;
    for (size_t i = index + 1; i != count_; ++i)
      entries_[i - 1] = entries_[i];
    entries_[--count_] = {};
  }
  void *owner_;
  Retire retire_;
  Completion completion_;
  Pool &pool_;
  std::array<Entry, 8> entries_{};
  size_t count_ = 0;
  std::exception_ptr error_;
};

// A bounded same-wave execution window. Instruction-family policy chooses
// queues and work; AsyncInstructionQueue supplies the common execution/retirement
// protocol. Draining before CU rescheduling keeps decoder pools, wave storage
// and data caches owned by the issuing thread for this initial prototype.
class AsyncInstructionWindow {
  using Access = async_execution::Access;

public:
  AsyncInstructionWindow(ComputeUnitCore &cu, Wavefront &wf)
      : cu_(cu), wf_(wf), has_accvgprs_(cu.arch() == ROCJITSU_CODE_ARCH_CDNA3 ||
                                        cu.arch() == ROCJITSU_CODE_ARCH_CDNA4) {}
  bool pending() const { return arithmetic_ && !arithmetic_->empty(); }
  bool stopped() const { return stopped_; }

  void before(const Instruction &inst) {
    poll();
    if (!pending())
      return;
    auto access = async_execution::footprint(inst, wf_.num_vgprs(), has_accvgprs_);
    if (!async_execution::safe_inline(inst) || !access) {
      if (pending())
        ++async_execution::stats.boundaries;

      drain();
      stopped_ = true;
      return;
    }
    if (arithmetic_ && arithmetic_->wait_conflicts(*access))
      ++async_execution::stats.hazards;
    check_errors();
    if (pending() && !matrix_coexecution::async_candidate(inst.mnemonic()) && !inst.is_memory_op())
      ++async_execution::stats.inline_overlap;
  }

  bool submit_mma(Instruction *inst) {
    namespace ae = async_execution;
    if (stopped_ || !matrix_coexecution::async_candidate(inst->mnemonic()))
      return false;
    auto access = ae::footprint(*inst, wf_.num_vgprs(), has_accvgprs_);
    const int sources = inst->num_src_operands();
    if (!access || inst->num_dst_operands() != 1 || (sources != 3 && sources != 5))
      return false;
    for (int i = 0; i <= sources; ++i) {
      const auto *operand = i == sources ? inst->dst_operand(0) : inst->src_operand(i);
      const auto ref = operand->to_register_ref();
      if (ref && (ref->cls == RegClass::VGPR || (has_accvgprs_ && ref->cls == RegClass::ACC_VGPR)))
        continue;
      // Inline accumulator/scale constants are immutable. SGPRs and special
      // registers are not covered by the vector-register scoreboard.
      if (i == sources || !operand->const_value())
        return false;
    }
    const size_t limit = matrix_coexecution::width() - 1;
    if (!limit)
      return false;
    auto &pool = matrix_coexecution::shared_pool();
    if (!pool.available()) {
      ++ae::stats.full;
      return false;
    }
    materialize();
    if (!arithmetic_)
      arithmetic_.emplace(this, retire_arithmetic, AsyncInstructionQueue::Completion::Independent);
    if (arithmetic_->size() < limit && arithmetic_->submit(inst, *inst, &wf_, *access)) {
      ++ae::stats.mma;
      started();
      return true;
    }
    // before() has resolved true dependencies. The ordinary synchronous path
    // can complete this independent MMA without joining older arithmetic jobs.
    ++ae::stats.full;
    return false;
  }

  void poll() {
    if (arithmetic_)
      arithmetic_->poll();
    check_errors();
  }
  void drain() {
    if (arithmetic_)
      arithmetic_->drain();
    check_errors();
  }
  void abandon() noexcept {
    try {
      drain();
    } catch (...) {
    }
  }

private:
  static void retire_arithmetic(void *, Instruction *inst, bool) {
    delete inst;
    ++async_execution::stats.retired_mma;
  }
  void started() {
    if (!started_) {
      started_ = true;
      ++async_execution::stats.windows;
    }
  }
  void materialize() {
    if (materialized_)
      return;
    // Inline instructions can touch new registers before jobs finish. Allocate
    // all lazy chunks before that can race with worker register-file access.
    for (uint32_t reg = 0; reg != wf_.num_vgprs(); ++reg)
      (void)cu_.raw_vgpr_data(wf_.vgpr_alloc().base + reg);
    if (has_accvgprs_)
      for (uint32_t reg = 256; reg != 512; ++reg)
        (void)cu_.raw_vgpr_data(wf_.vgpr_alloc().base + reg);
    materialized_ = true;
  }
  void check_errors() {
    auto error = arithmetic_ ? arithmetic_->take_error() : std::exception_ptr{};
    if (error) {
      abandon();
      std::rethrow_exception(error);
    }
  }
  ComputeUnitCore &cu_;
  Wavefront &wf_;
  std::optional<AsyncInstructionQueue> arithmetic_;
  bool has_accvgprs_;
  bool stopped_ = false, started_ = false, materialized_ = false;
};

// An issuer with no eligible instruction initializes only the optional's tag.
// Queue state is constructed after the first eligible instruction has been
// decoded, and still lives on the issuing thread's stack.
struct AsyncInstructionWindowStorage {
  std::optional<AsyncInstructionWindow> window;
};
} // namespace rocjitsu::amdgpu
