// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
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
inline unsigned memory_min_bytes() {
  static const unsigned value = [] {
    const char *text = std::getenv("RJ_ASYNC_MEM_MIN_BYTES");
    return static_cast<unsigned>(std::max(0, text ? std::atoi(text) : 512));
  }();
  return value;
}
struct Access {
  std::bitset<256> reads, writes;
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
  if (mc::candidate(name) || ordinary_memory(name))
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
inline std::optional<Access> footprint(const Instruction &inst, uint32_t num_vgprs) {
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
      const auto ref = static_cast<const cdna5::Operand *>(op)->cdna5::Operand::to_register_ref();
      if (!ref || ref->cls != RegClass::VGPR)
        continue;
      const uint32_t reg = ref->index, width = ref->width;
      if (reg >= 256 || width > 256 - reg || reg >= num_vgprs || width > num_vgprs - reg)
        return std::nullopt;
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
  uint64_t windows = 0, mma = 0, loads = 0, stores = 0, inline_overlap = 0;
  uint64_t hazards = 0, boundaries = 0, full = 0, memory_order = 0, waits = 0;
  uint64_t retired_mma = 0, retired_memory = 0;
  ~Stats() { flush(); }
  void flush() {
    if (windows)
      std::fprintf(stderr,
                   "RJ_ASYNC windows=%llu mma=%llu loads=%llu stores=%llu inline_overlap=%llu "
                   "hazards=%llu boundaries=%llu full=%llu memory_order=%llu waits=%llu "
                   "retired_mma=%llu retired_memory=%llu\n",
                   (unsigned long long)windows, (unsigned long long)mma, (unsigned long long)loads,
                   (unsigned long long)stores, (unsigned long long)inline_overlap,
                   (unsigned long long)hazards, (unsigned long long)boundaries,
                   (unsigned long long)full, (unsigned long long)memory_order,
                   (unsigned long long)waits, (unsigned long long)retired_mma,
                   (unsigned long long)retired_memory);
    windows = mma = loads = stores = inline_overlap = hazards = boundaries = full = 0;
    memory_order = waits = retired_mma = retired_memory = 0;
  }
};
inline thread_local Stats stats;
} // namespace async_execution

// Publication follows issue order. Completion policy is separate: memory can
// require an ordered prefix, while independent arithmetic releases only its
// own dependencies. Retirement always runs on the instruction allocator's owner.
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
  AsyncInstructionWindow(ComputeUnitCore &cu, Wavefront &wf) : cu_(cu), wf_(wf) {}
  bool pending() const {
    return (arithmetic_ && !arithmetic_->empty()) || (memory_ && !memory_->empty());
  }
  bool stopped() const { return stopped_; }

  void before(const Instruction &inst) {
    poll();
    if (!pending())
      return;
    auto access = async_execution::footprint(inst, wf_.num_vgprs());
    if (!async_execution::safe_inline(inst) || !access) {
      if (pending())
        ++async_execution::stats.boundaries;

      drain();
      stopped_ = true;
      return;
    }
    // One data-cache access at a time per CU. Even a synchronous next access
    // waits: this preserves L1 ownership and same-address load/store ordering.
    // Separate load/store queues can replace this stronger ordering once cache
    // ownership and inter-queue memory dependencies have been implemented.
    if (inst.is_memory_op() && memory_ && !memory_->empty()) {
      ++async_execution::stats.memory_order;
      memory_->drain();
    }
    if (memory_ && memory_->wait_conflicts(*access))
      ++async_execution::stats.hazards;
    if (arithmetic_ && arithmetic_->wait_conflicts(*access))
      ++async_execution::stats.hazards;
    check_errors();
    if (pending() && !matrix_coexecution::candidate(inst.mnemonic()) && !inst.is_memory_op())
      ++async_execution::stats.inline_overlap;
  }

  bool submit_mma(Instruction *inst) {
    namespace ae = async_execution;
    if (stopped_ || matrix_coexecution::mode() == 5 ||
        !matrix_coexecution::candidate(inst->mnemonic()))
      return false;
    auto access = ae::footprint(*inst, wf_.num_vgprs());
    if (!access || inst->num_dst_operands() != 1 || inst->num_src_operands() != 3)
      return false;
    for (int i = 0; i != 4; ++i) {
      const auto *operand = i == 3 ? inst->dst_operand(0) : inst->src_operand(i);
      const auto ref =
          static_cast<const cdna5::Operand *>(operand)->cdna5::Operand::to_register_ref();
      if (!ref || ref->cls != RegClass::VGPR)
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

  bool submit_memory(Instruction *inst) {
    namespace ae = async_execution;
    if (stopped_ || matrix_coexecution::mode() == 4 || !ae::ordinary_memory(inst->mnemonic()) ||
        !inst->data() || inst->data()->tag() != GLOBAL_MEM)
      return false;
    auto &d = *inst->data_as<VectorMemState>();
    if (d.atomic_op != AtomicOp::NONE || d.lds_dst || d.transpose || d.scratch_swizzle ||
        d.d16_hi || d.d16_lo || d.wf_size * d.elem_size * d.num_elems < ae::memory_min_bytes())
      return false;
    assert(!memory_ || memory_->empty());
    if (!wf_.wait_counters().empty())
      return false;
    if (!matrix_coexecution::shared_pool().available()) {
      ++ae::stats.full;
      return false;
    }
    materialize();
    Access access;
    if (d.is_load) {
      const uint32_t count = std::max(1u, (d.elem_size * d.num_elems + 3) / 4);
      if (d.dst_reg_base < wf_.vgpr_alloc().base)
        return false;
      const auto reg = d.dst_reg_base - wf_.vgpr_alloc().base;
      if (reg >= 256 || count > 256 - reg)
        return false;
      for (uint32_t i = 0; i != count; ++i)
        access.writes.set(reg + i);
    }
    memory_inst_ = inst;
    // ISA execution already captured addresses, masks and store bytes. Only
    // access runs on a helper; writeback and counter release happen at retirement.
    if (!memory_)
      memory_.emplace(this, retire_memory, AsyncInstructionQueue::Completion::Ordered);
    if (!memory_->submit(inst, memory_callback_, this, access)) {
      memory_inst_ = nullptr;
      ++ae::stats.full;
      return false;
    }
    cu_.global_mem_pipeline_.begin_async_access(*inst, wf_);
    if (d.is_load)
      ++ae::stats.loads;
    else
      ++ae::stats.stores;
    started();
    return true;
  }

  void poll() {
    if (arithmetic_)
      arithmetic_->poll();
    if (memory_)
      memory_->poll();
    check_errors();
  }
  void wait_memory() {
    if (memory_)
      memory_->drain();
    check_errors();
  }
  void drain() {
    if (arithmetic_)
      arithmetic_->drain();
    if (memory_)
      memory_->drain();
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
  static void retire_memory(void *owner, Instruction *inst, bool failed) {
    auto &window = *static_cast<AsyncInstructionWindow *>(owner);
    window.memory_inst_ = nullptr;
    window.cu_.global_mem_pipeline_.retire_async_access(inst, window.wf_, failed);
    ++async_execution::stats.retired_memory;
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
    materialized_ = true;
  }
  void check_errors() {
    auto error = arithmetic_ ? arithmetic_->take_error() : std::exception_ptr{};
    auto memory_error = memory_ ? memory_->take_error() : std::exception_ptr{};
    if (!error)
      error = memory_error;
    if (error) {
      abandon();
      std::rethrow_exception(error);
    }
  }
  ComputeUnitCore &cu_;
  Wavefront &wf_;
  std::optional<AsyncInstructionQueue> arithmetic_, memory_;
  Instruction *memory_inst_ = nullptr;
  Instruction memory_callback_{"host_memory_access", [](Instruction &, void *context) {
                                 auto &window = *static_cast<AsyncInstructionWindow *>(context);
                                 window.cu_.global_mem_pipeline_.run_async_access(
                                     *window.memory_inst_, window.wf_);
                               }};
  bool stopped_ = false, started_ = false, materialized_ = false;
};
} // namespace rocjitsu::amdgpu
