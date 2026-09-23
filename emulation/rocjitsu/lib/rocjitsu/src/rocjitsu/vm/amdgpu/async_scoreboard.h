// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file async_scoreboard.h
/// @brief Bounded instruction queue and register dependencies for MMA helpers.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <optional>

namespace rocjitsu::amdgpu {
class ComputeUnitCore;
class MmaAdmissionCache;
namespace async_execution {
inline constexpr unsigned issue_limit() { return 32; }
inline constexpr unsigned kMaxPendingMma = 7;
using Access = async_mma_policy::Access;
using async_mma_policy::footprint;
inline bool safe_inline(const Instruction &inst) { return async_mma_policy::safe_inline(inst); }
class Stats {
public:
  uint64_t windows = 0;
  uint64_t mma = 0;
  uint64_t inline_overlap = 0;
  uint64_t hazards = 0;
  uint64_t boundaries = 0;
  uint64_t full = 0;
  uint64_t retired_mma = 0;
  ~Stats() { flush(); }
  void flush() {
    if (windows)
      util::Logger::vm([&](auto &os) {
        os << std::format("RJ_ASYNC windows={} mma={} inline_overlap={} hazards={} "
                          "boundaries={} full={} retired_mma={}",
                          windows, mma, inline_overlap, hazards, boundaries, full, retired_mma);
      });
    windows = mma = inline_overlap = hazards = boundaries = full = retired_mma = 0;
  }
};
inline thread_local Stats stats;
} // namespace async_execution

/// @brief Pending MMA instructions owned by their issuing thread.
/// @details Completion releases independent register dependencies individually.
/// Instructions are destroyed on the decoder allocator's owning thread.
class AsyncInstructionQueue {
public:
  using Access = async_execution::Access;
  using Pool = matrix_coexecution::SharedPool;
  explicit AsyncInstructionQueue(Pool &pool) : pool_(pool) {}
  ~AsyncInstructionQueue() { assert(empty()); }
  bool empty() const { return count_ == 0; }
  size_t size() const { return count_; }
  /// Transfer ownership only when a helper accepts the instruction.
  bool submit(std::unique_ptr<Instruction> &inst, void *context, const Access &access) {
    if (count_ == entries_.size())
      return false;
    auto ticket = pool_.submit(*inst, context);
    if (!ticket)
      return false;
    entries_[count_++] = {std::move(inst), ticket, access};
    return true;
  }
  void poll() {
    for (size_t i = 0; i != count_;) {
      if (Pool::ready(entries_[i].ticket))
        retire(i);
      else
        ++i;
    }
  }
  bool wait_conflicts(const Access &access) {
    bool waited = false;
    for (size_t i = 0; i != count_;) {
      if (entries_[i].access.conflicts(access)) {
        waited = true;
        retire(i);
      } else
        ++i;
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
    std::unique_ptr<Instruction> inst;
    Pool::Ticket ticket;
    Access access;
  };
  void retire(size_t index) {
    auto &entry = entries_[index];
    auto error = pool_.finish(entry.ticket);
    entry.inst.reset();
    ++async_execution::stats.retired_mma;
    if (error && !error_)
      error_ = error;
    for (size_t i = index + 1; i != count_; ++i)
      entries_[i - 1] = std::move(entries_[i]);
    entries_[--count_] = {};
  }
  Pool &pool_;
  std::array<Entry, async_execution::kMaxPendingMma> entries_{};
  size_t count_ = 0;
  std::exception_ptr error_;
};

/// @brief A bounded same-wave execution window.
/// @details Draining before CU rescheduling keeps decoder pools, wave storage
/// and data caches owned by the issuing thread.
class AsyncInstructionWindow {
  using Access = async_execution::Access;

public:
  AsyncInstructionWindow(ComputeUnitCore &cu, Wavefront &wf, bool has_accvgprs = false);
  bool pending() const { return arithmetic_ && !arithmetic_->empty(); }
  bool stopped() const { return stopped_; }
  void reserve_issuer(uint64_t pc) { issuer_pc_ = std::min(issuer_pc_, pc); }
  bool take_issuer(uint64_t pc) {
    if (issuer_pc_ != pc)
      return false;
    issuer_pc_ = ~uint64_t{0};
    return true;
  }

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

  bool submit_mma(std::unique_ptr<Instruction> &inst) {
    if (stopped_ || !matrix_coexecution::async_candidate(inst->mnemonic()))
      return false;
    auto access = async_execution::footprint(*inst, wf_.num_vgprs(), has_accvgprs_);
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
    if (!pool_.available()) {
      ++async_execution::stats.full;
      return false;
    }
    materialize();
    if (!arithmetic_)
      arithmetic_.emplace(pool_);
    if (arithmetic_->submit(inst, &wf_, *access)) {
      ++async_execution::stats.mma;
      started();
      return true;
    }
    // before() has resolved true dependencies. The ordinary synchronous path
    // can complete this independent MMA without joining older arithmetic jobs.
    ++async_execution::stats.full;
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
  void started() {
    if (!started_) {
      started_ = true;
      ++async_execution::stats.windows;
    }
  }
  void materialize();
  void check_errors() {
    auto error = arithmetic_ ? arithmetic_->take_error() : std::exception_ptr{};
    if (error) {
      abandon();
      std::rethrow_exception(error);
    }
  }
  ComputeUnitCore &cu_;
  Wavefront &wf_;
  matrix_coexecution::SharedPool &pool_;
  std::optional<AsyncInstructionQueue> arithmetic_;
  bool has_accvgprs_;
  bool stopped_ = false, started_ = false, materialized_ = false;
  uint64_t issuer_pc_ = ~uint64_t{0};
};

// An issuer with no eligible instruction initializes only the optional's tag.
// Queue state is constructed after the first eligible instruction has been
// decoded, and still lives on the issuing thread's stack.
struct AsyncInstructionWindowStorage {
  std::optional<AsyncInstructionWindow> window;
  MmaAdmissionCache *admission = nullptr;
  bool has_accvgprs = false;
};
} // namespace rocjitsu::amdgpu
