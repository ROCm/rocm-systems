// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/register_set.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {
class Instruction;
namespace amdgpu {
class Wavefront;
struct RegisterModifiers;

/// A per-wave register filter owned exclusively by the instruction issuer.
/// Instruction-boundary validation completes before async work is published;
/// helper threads and plugin observers never access this filter.
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
    return i < kRegisters && (bytes_[i] & (write ? kResult | kReplaySource : kResult));
  }
  bool pending_vgpr(uint16_t index, uint8_t width) const {
    assert(index + width <= REGISTER_SET_MAX_VGPRS);
    if (width == 1)
      return bytes_[index] != 0;
    for (unsigned i = 0; i < width; ++i)
      if (bytes_[index + i])
        return true;
    return false;
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
  void set(size_t i, uint8_t bits = kResult) { bytes_[i] |= bits; }
  void clear(size_t i) { bytes_[i] = 0; }

  // Result ranges are normally contiguous physical VGPRs. Resolve their file
  // once and combine the overlap probe with marking the same shadow bytes.
  bool mark(RegisterRef reg, uint8_t bits) {
    bool overlap = false;
    if (reg.cls == RegClass::VGPR && reg.index + reg.width <= REGISTER_SET_MAX_VGPRS) {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto &byte = bytes_[reg.index + r];
        overlap |= byte != 0;
        byte |= bits;
      }
    } else {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto element = reg;
        element.index += r;
        if (const auto i = index(element); i < kRegisters) {
          overlap |= bytes_[i] != 0;
          bytes_[i] |= bits;
        }
      }
    }
    return overlap;
  }
  void clear(RegisterRef reg) {
    if (reg.cls == RegClass::VGPR && reg.index + reg.width <= REGISTER_SET_MAX_VGPRS) {
      for (unsigned r = 0; r < reg.width; ++r)
        bytes_[reg.index + r] = 0;
    } else {
      for (unsigned r = 0; r < reg.width; ++r) {
        auto element = reg;
        element.index += r;
        if (const auto i = index(element); i < kRegisters)
          bytes_[i] = 0;
      }
    }
  }
  void reset() { bytes_.fill(0); }

private:
  std::array<uint8_t, kRegisters> bytes_{};
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
  void check_instruction(const Instruction &inst, Wavefront &wf);
  static uint64_t result_lanes(const Instruction &inst, Wavefront &wf);
  static bool result_is_written(const Instruction &inst, Wavefront &wf);
  struct FlatLanes {
    uint64_t requests;
    uint64_t shared;
  };
  static FlatLanes flat_lanes(const Instruction &inst, Wavefront &wf, uint64_t shared_base,
                              uint64_t shared_limit);
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
  /// Use the same decoded counter units for admission and issue accounting.
  static uint32_t issue_units(const Instruction &inst,
                              const waitcheck_detail::ClassifiedEvent &event, rj_code_arch_t arch);
  /// Find the incoming producer's FIFO class without issuing its completion.
  uint16_t ordered_write_order(const waitcheck_detail::ClassifiedEvent &event,
                               rj_code_arch_t arch) const;
  /// Admission can force completion before the incoming instruction reads its operands.
  void backpressure(WaitCounterKind counter, uint32_t capacity, uint32_t incoming_units = 1);
  void add(Event event);
  void wait(WaitCounterKind counter, uint32_t threshold);
  uint64_t outstanding(WaitCounterKind counter) const {
    const auto i = static_cast<size_t>(counter);
    return issued_[i] - retired_[i];
  }
  [[gnu::always_inline]] void access(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
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

  [[gnu::cold]] RJ_NOINLINE void check_instruction_pending(const Instruction &inst, Wavefront &wf);
  void check_memory_result(const Instruction &inst, Wavefront &wf, uint64_t vector_lanes,
                           const RegisterModifiers &modifiers);
  void access_pending(RegisterRef reg, uint64_t lanes, uint8_t bytes, bool write,
                      uint16_t ordered_write_order);
  void rebuild_mask();
  void clear_destination(const Event &event);
  void retire_completed();
  bool apply_wait(WaitCounterKind counter, uint32_t threshold);
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
  bool pending_scalar_ = false;
  // Once two events share a shadow slot, retirement must restore overlapping
  // entries until the wave has no pending events. Lane/byte disjointness does
  // not make their shared register byte independently clearable.
  bool may_overlap_ = false;
  MemoryWaitShadow &pending_;
  std::vector<Event> events_;
  uint64_t pc_ = 0;
  void *context_ = nullptr;
  Reporter reporter_ = nullptr;
};

} // namespace amdgpu
} // namespace rocjitsu
