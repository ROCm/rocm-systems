// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file WaitCounters.h
/// @brief Cycle-domain wait-counter mirror of the funcsim's WaitCounters.
///
/// rocjitsu's functional sim drains memory synchronously, so its own
/// wf.wait_counters() return to zero immediately. The cycle model maintains its
/// OWN counters in the cycle domain: increment when the scheduler commits a memop
/// to a pipe, decrement when the model-computed completion retires. A positional
/// WaitcntGate then blocks until WaitTarget::satisfied(outstanding) holds.
///
/// Representation is a slot array (NOT named fields) so an arch with a different
/// counter set is a data change, not a struct edit. Slot semantics mirror
/// rocjitsu/vm/amdgpu/wait_counters.h exactly — including the GFX11 quirk that
/// DSCNT/KMCNT also bump the aggregate LGKMCNT. The cycle model is an observer of
/// the funcsim, so its routing (which slot a memop increments) must match the
/// funcsim's MemoryPipeline mapping or gate checks diverge.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cycle_model {

/// Canonical hardware wait-counter slots. Order is the array index order; the
/// adapter copies rocjitsu WaitTarget's same-named fields into these slots.
enum class WaitCounter : uint8_t {
  VMCNT,    ///< vector memory (global/flat loads + stores on CDNA; loads on GFX11+)
  LGKMCNT,  ///< LDS/GDS/scalar/message aggregate
  EXPCNT,   ///< export
  VSCNT,    ///< vector stores (GFX10+; dormant on CDNA)
  DSCNT,    ///< DS subset of LGKMCNT (GFX11+)
  KMCNT,    ///< scalar/constant subset of LGKMCNT (GFX11+)
  COUNT,
};

inline constexpr std::size_t kNumWaitCounters = static_cast<std::size_t>(WaitCounter::COUNT);

inline constexpr std::size_t idx(WaitCounter w) { return static_cast<std::size_t>(w); }

/// Outstanding cycle-domain counts, one per slot. Saturates like hardware.
struct WaitCounters {
  std::array<uint8_t, kNumWaitCounters> c{};   // all slots zero-initialized

  static constexpr uint8_t SAT = 63;           // VMCNT/LGKMCNT/VSCNT/DSCNT max

  void increment(WaitCounter w) {
    bump(w);
    // DSCNT/KMCNT are subsets of the aggregate LGKMCNT (mirrors funcsim).
    if (w == WaitCounter::DSCNT || w == WaitCounter::KMCNT) bump(WaitCounter::LGKMCNT);
  }

  void decrement(WaitCounter w) {
    drop(w);
    if (w == WaitCounter::DSCNT || w == WaitCounter::KMCNT) drop(WaitCounter::LGKMCNT);
  }

  void clear() { c.fill(0); }
  bool empty() const {
    for (uint8_t v : c) if (v) return false;
    return true;
  }

 private:
  void bump(WaitCounter w) { uint8_t& v = c[idx(w)]; if (v < SAT) ++v; }
  void drop(WaitCounter w) { uint8_t& v = c[idx(w)]; if (v) --v; }
};

/// Per-slot thresholds. A slot at MAX means "don't wait on this counter".
struct WaitTarget {
  std::array<uint8_t, kNumWaitCounters> t;

  WaitTarget() { t.fill(0xFF); }   // default: wait on nothing

  /// True once every outstanding count is at or below its threshold.
  bool satisfied(const WaitCounters& o) const {
    for (std::size_t i = 0; i < kNumWaitCounters; ++i)
      if (o.c[i] > t[i]) return false;
    return true;
  }
};

}  // namespace cycle_model
