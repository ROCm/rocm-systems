// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace simdojo {
enum class ExitCause : uint8_t;
}

namespace rocjitsu::concurrent_dispatch_fuzz {

/// @brief Bytes reserved for dispatch controls before the input-carried kernel.
inline constexpr std::size_t kHeaderSize = 16;
/// @brief Largest input accepted by the focused harness.
inline constexpr std::size_t kMaxInputSize = 1024;
/// @brief Size of unconstrained inputs produced by random_input().
inline constexpr std::size_t kGeneratedInputSize = 64;
/// @brief Byte-1 control bits shared by decoding and corpus construction.
inline constexpr uint8_t kPrimeCachesBit = 1u << 0;
inline constexpr uint8_t kBarrierBit = 1u << 1;
inline constexpr uint8_t kAgentAcquireBit = 1u << 2;
inline constexpr uint8_t kAgentReleaseBit = 1u << 3;
inline constexpr uint8_t kXcdFanoutBit = 1u << 4;

/// @brief Work performed and outcomes observed while executing one or more inputs.
struct RunStats {
  uint64_t inputs = 0;                  ///< Inputs accepted by the size contract.
  uint64_t targets = 0;                 ///< Inputs selecting a supported GPU target.
  uint64_t modes = 0;                   ///< Distinct scheduling modes executed.
  uint64_t completed_modes = 0;         ///< Modes completing every submitted packet.
  uint64_t budget_limited_modes = 0;    ///< Modes reaching the simulated tick budget.
  uint64_t primary_stalled_modes = 0;   ///< Modes stopping with a quiescent active primary.
  uint64_t fatal_exit_modes = 0;        ///< Modes stopping on a nonzero exit request.
  uint64_t terminal_faulted_modes = 0;  ///< Modes observing a terminal queue fault.
  uint64_t published_completions = 0;   ///< Packet completions published across all modes.
  uint64_t instructions = 0;            ///< Wave instructions observed across all modes.
  uint64_t packets = 0;                 ///< AQL packets submitted per input.
  uint64_t workgroups = 0;              ///< Workgroups submitted per input.
  uint64_t workitems = 0;               ///< Work-items submitted per input.
  uint64_t kernel_bytes = 0;            ///< Input-carried kernel bytes, excluding termination.
  uint64_t completed_inputs = 0;        ///< Inputs completing in every scheduling mode.
  uint64_t incomplete_inputs = 0;       ///< Inputs incomplete in at least one mode.
  uint64_t budget_limited_inputs = 0;   ///< Inputs reaching the budget in at least one mode.
  uint64_t multi_xcd_inputs = 0;        ///< Inputs using a representative multi-XCD topology.
  uint64_t cache_primed_inputs = 0;     ///< Inputs that prepopulate each XCD's L2 cache.
  uint64_t fanout_inputs = 0;           ///< Inputs enabling cross-XCD queue fan-out.
  uint64_t differential_inputs = 0;     ///< Inputs using the canonical exact-result oracle.
  uint64_t pooled_dispatch_threads = 0; ///< Largest effective shared dispatch-pool width.
};

/// @brief Run an input-selected kernel across engine and CPU dispatch worker combinations.
/// @details Single-XCD targets use one engine thread with either serial or pooled dispatch.
/// The canonical cache kernel has an exact-result and cross-mode oracle. Each mode checks
/// per-packet completion publication and primary lifetime. For raw kernels, modes must agree on
/// terminal queue fault state. In the absence of a terminal fault, the exit reason and code must
/// also agree. Racy control flow can lead to stalls or budget exhaustion in some modes.
/// Retirement is compared when every mode completes.
[[nodiscard]] RunStats run_input(std::span<const uint8_t> input);

/// @brief Produce a deterministic input with a target-specific instruction stream.
[[nodiscard]] std::vector<uint8_t> random_input(uint64_t seed);

/// @brief Return one terminating input-carried kernel for every supported execution target.
[[nodiscard]] std::vector<std::vector<uint8_t>> seed_corpus();

namespace detail {

/// @brief Check whether the primary's exit cause is consistent with the queue outcome.
[[nodiscard]] bool primary_lifecycle_is_valid(bool published_all_completions, bool terminal_faulted,
                                              simdojo::ExitCause cause);

} // namespace detail

} // namespace rocjitsu::concurrent_dispatch_fuzz
