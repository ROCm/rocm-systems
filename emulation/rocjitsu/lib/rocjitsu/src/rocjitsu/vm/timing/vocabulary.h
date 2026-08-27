// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vocabulary.h
/// @brief The small shared nouns the timing plane speaks in.
///
/// @details These are the names a functional component uses to describe what it
/// just did, before anything has been costed: which hardware counter an
/// operation will be waited on, which address space it went to, and how far an
/// `s_waitcnt` drains those counters. They are separated from the components
/// that consume them because they are the only part of the plane a functional
/// component has to know about, and because a test can build a sequence out of
/// them with no simulator, no compiled kernel and no GPU attached. Most of what
/// is worth testing about a timing model is how stalls compose, and building the
/// exact sequence by hand tests that far more precisely than hoping a kernel
/// reaches the case.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocjitsu::timing {

/// @brief The hardware counters a wavefront can wait on.
///
/// @details Which counter an operation posts to is a per-target ISA decision,
/// not something derivable from the instruction class, so it is reported by the
/// functional side rather than inferred here. Vector stores are the case that
/// matters: older compute targets have no separate store counter and post
/// stores to the load counter, so a model that assumed a store counter would
/// park those completions in a queue no wait instruction on that target can
/// name, and the wait would cost nothing.
enum class WaitCounter : std::uint8_t {
  VectorLoad,   ///< vmcnt / loadcnt. Also stores, on targets with no store counter.
  VectorStore,  ///< vscnt / storecnt, where the target separates them.
  LgkmCombined, ///< lgkmcnt, the pre-GFX11 counter shared by LDS, scalar and messages.
  LdsAndGds,    ///< dscnt, where the target splits LDS out of lgkmcnt.
  ScalarMemory, ///< kmcnt, where the target splits scalar memory out of lgkmcnt.
  Export,       ///< expcnt.
  Tensor,       ///< tensorcnt.
  Async,        ///< asynccnt.
  Count,        ///< Number of counters, and "none reported"; not a counter itself.
};

/// @brief Number of wait counters, for the per-counter arrays a wavefront keeps.
inline constexpr std::size_t kNumWaitCounters = static_cast<std::size_t>(WaitCounter::Count);

/// @brief Threshold high enough that a counter is never the constraint.
///
/// @details Most waits name a single counter, so this is the value every other
/// entry of a wait carries. It is deliberately not zero: zero is the *strictest*
/// threshold there is, and a default-constructed wait that read as zero would
/// turn every unnamed counter into a full drain.
inline constexpr std::uint32_t kUnconstrained = 0xFFFFFFFFu;

/// @brief Where a memory access went, which decides the path it is charged.
enum class MemorySpace : std::uint8_t {
  None,           ///< Not a memory access.
  Global,         ///< Global, buffer, scratch, or flat resolved to memory.
  LocalDataShare, ///< Local data share.
  Scalar,         ///< Scalar (constant) path.
  Tensor,         ///< Tensor data mover, between global memory and the LDS.
};

/// @brief Report name for a counter. Stable; it is part of the report's
///        vocabulary and of the architecture config file's.
inline constexpr const char *wait_counter_name(WaitCounter value) {
  switch (value) {
  case WaitCounter::VectorLoad:
    return "vector_load";
  case WaitCounter::VectorStore:
    return "vector_store";
  case WaitCounter::LgkmCombined:
    return "lgkm_combined";
  case WaitCounter::LdsAndGds:
    return "lds_and_gds";
  case WaitCounter::ScalarMemory:
    return "scalar_memory";
  case WaitCounter::Export:
    return "export";
  case WaitCounter::Tensor:
    return "tensor";
  case WaitCounter::Async:
    return "async";
  case WaitCounter::Count:
    break;
  }
  return "none";
}

/// @brief Report name for an address space.
inline constexpr const char *memory_space_name(MemorySpace value) {
  switch (value) {
  case MemorySpace::None:
    return "none";
  case MemorySpace::Global:
    return "global";
  case MemorySpace::LocalDataShare:
    return "local_data_share";
  case MemorySpace::Scalar:
    return "scalar";
  case MemorySpace::Tensor:
    return "tensor";
  }
  return "none";
}

/// @brief Thresholds an `s_waitcnt`-family instruction waits down to.
///
/// @details One entry per counter, holding the count the wavefront is willing
/// to leave outstanding. This is a thin helper over the raw array a wavefront
/// carries rather than a type of its own, so that a caller holding the array
/// directly and a caller holding this see the same values in the same order.
struct WaitThresholds {
  /// @brief Same as the namespace-scope constant; repeated so a reader holding
  ///        only this type does not have to go looking for it.
  static constexpr std::uint32_t kNone = kUnconstrained;

  /// @brief Per-counter thresholds, indexed by WaitCounter.
  std::array<std::uint32_t, kNumWaitCounters> values{};

  /// @brief Construct a wait that constrains nothing.
  WaitThresholds() { values.fill(kUnconstrained); }

  /// @brief Wait @p counter down to at most @p threshold outstanding.
  void set(WaitCounter counter, std::uint32_t threshold) {
    if (counter != WaitCounter::Count)
      values[static_cast<std::size_t>(counter)] = threshold;
  }

  /// @brief The threshold for @p counter, or kUnconstrained.
  std::uint32_t get(WaitCounter counter) const {
    return counter == WaitCounter::Count ? kUnconstrained
                                         : values[static_cast<std::size_t>(counter)];
  }

  /// @brief Whether any counter is named at all.
  ///
  /// @details Lets a wavefront skip the drain entirely for the waits a compiler
  /// emits that name nothing, which are common enough to be worth the check.
  bool constrains_anything() const {
    for (std::uint32_t value : values)
      if (value != kUnconstrained)
        return true;
    return false;
  }
};

} // namespace rocjitsu::timing
