// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/register_set.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {
class Instruction;
namespace amdgpu {

/// A register shadow read by issuers and MMA helpers. Only the issuer writes it.
/// Helpers use it solely as a filter before consulting their thread-local scope;
/// they never inspect the mutable dependency records. Relaxed atomics therefore
/// suffice: the shadow publishes no other state to a helper.
class MemoryWaitShadow {
public:
  // Reserve scalar encoding slots separately from allocatable SGPRs. ACC
  // registers are not destinations of the modeled memory producers.
  static constexpr size_t kSgprEncodingSlots = 128;
  static constexpr size_t kTtmpBase = REGISTER_SET_MAX_VGPRS + kSgprEncodingSlots;
  static constexpr size_t kSpecialBase = kTtmpBase + REGISTER_SET_MAX_TTMPS;
  static constexpr size_t kScalarShadowSlots = 256;
  static constexpr size_t kRegisters = REGISTER_SET_MAX_VGPRS + kScalarShadowSlots;
  static size_t index(RegisterRef reg) {
    if (reg.cls == RegClass::VGPR && reg.index < REGISTER_SET_MAX_VGPRS)
      return reg.index;
    if (reg.cls == RegClass::SGPR && reg.index < kSgprEncodingSlots)
      return REGISTER_SET_MAX_VGPRS + reg.index;
    if (reg.cls == RegClass::TTMP && reg.index < REGISTER_SET_MAX_TTMPS)
      return kTtmpBase + reg.index;
    if (is_special_reg_class(reg.cls))
      return kSpecialBase + static_cast<size_t>(reg.cls);
    return kRegisters;
  }
  static constexpr uint8_t kResult = 1;
  static constexpr uint8_t kReplaySource = 2;
  bool test(size_t i, bool write = false) const {
    return i < kRegisters && (bytes_[i].load(std::memory_order_relaxed) &
                              (write ? kResult | kReplaySource : kResult));
  }
  bool pending(RegisterRef reg, bool write = false) const {
    for (unsigned r = 0; r < reg.width; ++r) {
      auto element = reg;
      element.index += r;
      if (test(index(element), write))
        return true;
    }
    return false;
  }
  void set(size_t i, uint8_t bits = kResult) {
    // Only the issuer writes; helpers only read this filter.
    bytes_[i].store(bytes_[i].load(std::memory_order_relaxed) | bits, std::memory_order_relaxed);
  }
  void clear(size_t i) { bytes_[i].store(0, std::memory_order_relaxed); }
  void reset() {
    for (auto &byte : bytes_)
      byte.store(0, std::memory_order_relaxed);
  }

private:
  std::array<std::atomic<uint8_t>, kRegisters> bytes_{};
};

/// Tracks software-visible memory dependencies independently of eager writeback.
/// No memory payloads, decoder objects, or helper threads are retained here.
class MemoryWaitScoreboard {
public:
  static constexpr uint8_t kFullDwordByteMask = 0xf;
  static constexpr uint16_t kUnordered = UINT16_MAX;
  explicit MemoryWaitScoreboard(MemoryWaitShadow &shadow) : pending_(shadow) {
    last_order_.fill(kUnordered);
  }
  struct Event {
    uint64_t sequence;
    uint64_t pc;
    uint64_t lanes;
    RegisterRef reg;
    WaitCounterKind counter;
    uint8_t bytes;
    bool reported = false;
    uint16_t order = kUnordered;
    uint64_t order_sequence = 0;
  };
  struct Hazard {
    Event producer;
    uint64_t consumer_pc;
    RegisterRef reg;
    bool write;
    uint32_t required_wait;
  };
  using Reporter = void (*)(void *, const Hazard &);

  bool empty() const { return events_.empty(); }
  void clear();
  void before(const Instruction &inst, rj_code_arch_t arch);
  /// X has one translation group at a time, independent of completion queues.
  void xcnt_group(bool scalar);
  /// Map translation to this instruction's completion position, if it received one.
  uint64_t issue_xcnt(std::optional<WaitCounterKind> completion, bool scalar);
  /// A VMEM destination orders translation of older overlapping VMEM sources.
  void xcnt_ordered_write(RegisterRef reg, uint64_t lanes, uint8_t bytes);
  /// Count every operation, including stores without a register destination.
  uint64_t issue(WaitCounterKind counter, bool unordered = false, uint32_t ordering_kind = 0,
                 uint32_t units = 1, bool backpressure_ordered = true);
  uint64_t issue(const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch,
                 uint32_t units = 1);
  /// Find the incoming producer's FIFO class without issuing its completion.
  uint16_t ordered_write_order(const waitcheck_detail::ClassifiedEvent &event,
                               rj_code_arch_t arch) const;
  /// Admission can force completion before the incoming instruction reads its operands.
  void backpressure(WaitCounterKind counter, uint32_t capacity);
  void add(Event event);
  void wait(WaitCounterKind counter, uint32_t threshold);
  uint64_t outstanding(WaitCounterKind counter) const {
    const auto i = static_cast<size_t>(counter);
    return issued_[i] - retired_[i];
  }
  void access(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
              uint16_t ordered_write_order = kUnordered) {
    if (!lanes || !bytes)
      return;
    if (!pending_.pending(reg, write))
      return;
    access_pending(reg, lanes, bytes, write, ordered_write_order);
  }

  void bind(uint64_t pc, void *context, Reporter reporter) {
    pc_ = pc;
    context_ = context;
    reporter_ = reporter;
  }
  const std::vector<Event> &events() const { return events_; }

private:
  static constexpr size_t kCounters = static_cast<size_t>(WaitCounterKind::Count);
  static constexpr size_t kRegisters = MemoryWaitShadow::kRegisters;
  static size_t index(RegisterRef reg) { return MemoryWaitShadow::index(reg); }

  void access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                      uint16_t ordered_write_order);
  void rebuild_mask();
  void clear_destination(const Event &event);
  void retire_completed();
  bool completed(WaitCounterKind counter, uint64_t sequence, uint16_t order,
                 uint64_t order_sequence) const;
  void stamp_order(Event &event) const;
  uint32_t required_wait(const Event &event) const;
  std::array<uint64_t, kCounters> issued_{};
  std::array<uint64_t, kCounters> retired_{};
  std::array<bool, kCounters> unordered_{};
  std::array<uint32_t, kCounters> ordering_kinds_{};
  struct Order {
    WaitCounterKind counter;
    uint32_t kind;
    uint64_t issued = 0;
    uint64_t retired = 0;
  };
  // Counter membership is not completion order. Only operations in the same
  // ordered class can prove that an old result must have completed.
  std::vector<Order> orders_;
  std::array<uint16_t, kCounters> last_order_{};
  bool xcnt_scalar_ = false;
  struct Translation {
    uint64_t sequence;
    uint64_t completion_sequence;
    WaitCounterKind completion;
    uint16_t order;
    uint64_t order_sequence;
  };
  std::vector<Translation> translations_;
  MemoryWaitShadow &pending_;
  std::vector<Event> events_;
  uint64_t pc_ = 0;
  void *context_ = nullptr;
  Reporter reporter_ = nullptr;
};

// These TLS scopes belong to core, not the observer ABI. Separately loaded
// observers use register accessors and the exported helper below; they must not
// install a scope in their own DSO. Only the issuing CPU installs this scope.
// MMA helpers are checked before publication and never access mutable state.
inline thread_local MemoryWaitScoreboard *active_memory_wait_check = nullptr;
// Keep TLS lookup behind the shadow branch; inlining lets the compiler hoist
// __tls_get_addr onto accesses to registers that have no pending dependency.
// Inline register accessors are also used by separately loaded observers.
RJ_API_EXPORT RJ_NOINLINE void check_active_memory_wait(RegisterRef reg, uint64_t lanes,
                                                        uint8_t bytes, bool write);

// Legacy SDWA compares temporarily compute their explicit SGPR result in VCC
// and restore VCC afterwards. Those internal writes are not architectural.
inline thread_local bool suppress_memory_wait_vcc_write = false;
class ScopedMemoryWaitVccWriteSuppression {
public:
  ScopedMemoryWaitVccWriteSuppression(const ScopedMemoryWaitVccWriteSuppression &) = delete;
  ScopedMemoryWaitVccWriteSuppression &
  operator=(const ScopedMemoryWaitVccWriteSuppression &) = delete;
  explicit ScopedMemoryWaitVccWriteSuppression(bool suppress) : suppress_(suppress) {
    if (suppress_) {
      previous_ = suppress_memory_wait_vcc_write;
      suppress_memory_wait_vcc_write = true;
    }
  }
  ~ScopedMemoryWaitVccWriteSuppression() {
    if (suppress_)
      suppress_memory_wait_vcc_write = previous_;
  }

private:
  bool suppress_;
  bool previous_ = false;
};

class ScopedMemoryWaitCheck {
public:
  ScopedMemoryWaitCheck(const ScopedMemoryWaitCheck &) = delete;
  ScopedMemoryWaitCheck &operator=(const ScopedMemoryWaitCheck &) = delete;
  explicit ScopedMemoryWaitCheck(MemoryWaitScoreboard *state) {
    if (state && !state->empty()) {
      installed_ = true;
      previous_ = active_memory_wait_check;
      active_memory_wait_check = state;
    }
  }
  ~ScopedMemoryWaitCheck() {
    if (installed_)
      active_memory_wait_check = previous_;
  }

private:
  MemoryWaitScoreboard *previous_ = nullptr;
  bool installed_ = false;
};

/// Observer snapshots must not be mistaken for an instruction's register use.
class SuspendedMemoryWaitCheck {
public:
  SuspendedMemoryWaitCheck(const SuspendedMemoryWaitCheck &) = delete;
  SuspendedMemoryWaitCheck &operator=(const SuspendedMemoryWaitCheck &) = delete;
  SuspendedMemoryWaitCheck() : previous_(active_memory_wait_check) {
    active_memory_wait_check = nullptr;
  }
  ~SuspendedMemoryWaitCheck() { active_memory_wait_check = previous_; }

private:
  MemoryWaitScoreboard *previous_;
};
} // namespace amdgpu
} // namespace rocjitsu
