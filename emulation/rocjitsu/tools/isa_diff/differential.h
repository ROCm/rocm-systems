// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file differential.h
/// @brief Paired-input differential testing of one region pair, plus the result model.
///
/// @details The result keeps the evidence levels separate:
///   - test_status: NO_MISMATCH_FOUND | LOCAL_MISMATCH | INCONCLUSIVE | UNSUPPORTED |
///     HARNESS_ERROR. NO_MISMATCH_FOUND means only that no counterexample was found under
///     the given seed, budget and contract; it is not a proof of equivalence.
///   - contract_status: ESTABLISHED | ASSUMED | UNRESOLVED.
///   - witness_scope: NONE | LOCAL_ONLY | REACHABLE_AT_CUT | OBSERVABLE_KERNEL_DIVERGENCE.
///     P0 can only produce LOCAL_ONLY; establishing reachability is pending.
///   - coverage_scope: COMPLETE_FOR_DECLARED_SCOPE | PARTIAL | NOT_ESTABLISHED.
///   - proof_status: always NOT_RUN in P0.
/// Zero valid cases never count as a pass.
#pragma once

#include "input_policy.h"
#include "machine_state.h"
#include "region.h"
#include "region_runner.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

inline constexpr const char *kToolName = "rj_isa_diff";
inline constexpr const char *kToolVersion = "0.1.0-p0";
inline constexpr const char *kSchemaVersion = "rj_isa_diff.v1";

enum class TestStatus {
  NoMismatchFound,
  LocalMismatch,
  Inconclusive,
  Unsupported,
  HarnessError,
  NoSemanticDiff, ///< Aggregate only: text and metadata identical, nothing to test.
};
enum class WitnessScope { None, LocalOnly, ReachableAtCut, ObservableKernelDivergence };
enum class CoverageScope { CompleteForDeclaredScope, Partial, NotEstablished };

[[nodiscard]] const char *test_status_name(TestStatus s);
[[nodiscard]] const char *witness_scope_name(WitnessScope s);
[[nodiscard]] const char *coverage_scope_name(CoverageScope s);

struct TestOptions {
  uint64_t seed = 1;
  /// Random cases (accepted samples) run after the mandatory boundary sweep.
  uint64_t cases = 5000;
  uint64_t max_attempts = 0; ///< Random-phase attempts; 0 = 4 * cases.
  /// Safety cap on the boundary sweep. The sweep cannot be disabled; reaching the cap
  /// is reported as truncation and partial coverage.
  size_t max_boundary_cases = 250000;
  size_t max_counterexamples = 3; ///< Bundles saved per region.
  std::string counterexample_dir; ///< Empty: keep in memory only.
  /// Shrink saved counterexamples (handoff 10): input values first, then seed dependence.
  bool reduce = true;
  size_t reduction_budget = 4096; ///< Paired executions per counterexample.
  /// Per-case JSONL log directory (handoff 8 learning data). Empty: no log.
  std::string case_log_dir;
  /// Artifacts and provenance copied into every counterexample bundle (handoff 10).
  Json artifact_info;
  /// Result cache directory (handoff 11); empty disables the cache.
  std::string cache_dir;
  /// Compare NaNs bit for bit. By default a NaN produced by a floating-point instruction
  /// equals any NaN of the same format produced on the other side (LLVM leaves payload,
  /// sign and quietness unspecified).
  bool strict_nan = false;
  /// PoisonKind mask: cases whose region instructions meet one of these machine analogues
  /// of IR poison conditions, on either side, are excluded as poison inputs.
  uint32_t poison_filter = 0;
};

struct PhaseCounts {
  uint64_t cases = 0;    ///< Samples tried in this phase.
  uint64_t accepted = 0; ///< Samples satisfying the constraints.
  uint64_t valid = 0;    ///< Pairs that exited normally on both sides.
  uint64_t mismatches = 0;
};

struct Mismatch {
  uint64_t case_index = 0;
  std::string phase; ///< "boundary" or "random".
  std::string kind;  ///< observable | effect_model | frame_preservation | exit_status_divergence
  LocSet locs;
  VSet vlocs; ///< VGPRs involved (VALU support).
  /// Memory model: stored bytes differ; the first differing byte addresses.
  bool memory = false;
  std::vector<uint64_t> mem_addrs;
  std::vector<std::string> frame_fields;
  std::string detail;
  MachineState init_old, init_new, final_old, final_new;
  RunOutcome out_old, out_new;
  std::vector<std::string> events;
  /// Reduction record (null when not reduced): runs, accepted steps, whether the
  /// mismatch no longer depends on the seed-derived canaries, and the original state.
  Json reduction;
  /// The comparison it was found under: NaN policy and poison filter (PoisonKind mask).
  bool strict_nan = false;
  uint32_t poison_filter = 0;
  /// The new side ran with an alternative NaN choice (RegionRunner::set_nan_choice) when
  /// it showed the difference: 0 for none, else the alternative and its seed. Reduction,
  /// the bundle trace and replay run the new side with it.
  unsigned new_nan_choice = 0;
  uint64_t new_nan_seed = 0;
};

struct Timings {
  uint64_t setup_ns = 0;
  uint64_t propose_ns = 0;
  uint64_t load_ns = 0;
  uint64_t old_exec_ns = 0;
  uint64_t new_exec_ns = 0;
  uint64_t compare_ns = 0;
  uint64_t triage_ns = 0;
};

struct RegionResult {
  std::string region_id;
  bool synthetic = false;
  TestStatus test_status = TestStatus::Inconclusive;
  ContractStatus contract_status = ContractStatus::Unresolved;
  WitnessScope witness_scope = WitnessScope::None;
  CoverageScope coverage_scope = CoverageScope::NotEstablished;
  std::string reason;
  std::string policy;
  uint64_t seed = 0;
  uint64_t attempted = 0;      ///< Samples proposed.
  uint64_t accepted = 0;       ///< Samples satisfying the constraints.
  uint64_t executed_pairs = 0; ///< Pairs where both sides were run.
  uint64_t valid_pairs = 0;    ///< Pairs where both sides exited normally and were compared.
  uint64_t mismatches = 0;
  uint64_t harness_errors = 0;
  /// Valid pairs that compare equal only through NaN equivalence / undefined-bit masking.
  uint64_t nan_tolerated = 0;
  uint64_t undef_masked = 0;
  /// Valid pairs equal only through the per-bit don't-care mask (bits that changed across
  /// old's re-runs with undefined values refilled or NaNs replaced). Per-bit masking loses
  /// the relation between bits (old `(u & 1) * 3` is 0 or 3; masking both bits accepts 1),
  /// so a region that needed it is not a plain pass.
  uint64_t dont_care_masked = 0;
  bool strict_nan = false;
  uint32_t poison_filter = 0;
  PhaseCounts boundary; ///< Mandatory boundary sweep.
  PhaseCounts random;   ///< Policy-driven random phase.
  uint64_t boundary_generated = 0;
  uint64_t boundary_duplicates = 0;
  bool boundary_truncated = false;
  /// Valid boundary cases per catalog family ("int", "f32", "shift", "pair.carry", ...).
  std::map<std::string, uint64_t> boundary_families;
  std::map<std::string, uint64_t> rejections; ///< Constraint rejections and run exclusions.
  std::map<std::string, uint64_t> mismatch_kinds;
  uint64_t old_insts_executed = 0;
  uint64_t new_insts_executed = 0;
  size_t old_len = 0, new_len = 0;
  Timings t;
  std::vector<Mismatch> examples;
  std::vector<std::string> counterexample_files;
  std::string prepared_key;
  /// Automatic context expansion after a local mismatch (handoff 5.2):
  /// "" (not needed), "RESOLVED_BY_CONTEXT", "PERSISTS_WITH_CONTEXT", "NO_EXPANSION_AVAILABLE".
  std::string resolution;
  std::string resolved_by; ///< Variant id with no mismatch, if any.
  ContractStatus resolved_contract = ContractStatus::Unresolved;
  CoverageScope resolved_coverage = CoverageScope::NotEstablished;
  Json expansion_tried = Json::array(); ///< Summary of each variant tested.
  /// Handoff 13: a region whose contract carries derived prefix constraints is also run
  /// unrestricted; this summarises that run (null when not applicable).
  Json unrestricted;
  uint64_t novelty_buckets = 0; ///< Distinct bounded outcome classes seen.
  uint64_t novel_boundary = 0, novel_random = 0;
  std::string case_log; ///< Path of the per-case JSONL log, if written.
  /// Union of the locations of every observable mismatch.
  LocSet mismatch_locs;
  VSet mismatch_vlocs;
  bool mismatch_memory = false;
  /// Set by test_case: every mismatching location is read by changed code downstream of
  /// the region, which may compensate it; the mismatch is then a witness, not a verdict.
  bool compensable = false;
  std::string pending_effects; ///< From the contract (handoff 6.1).
  std::string composition;     ///< From the contract (handoff 5.2).
  /// Set when the result came from the cache: to_json() returns it verbatim.
  Json cached;

  [[nodiscard]] Json to_json() const;
};

/// @brief Build both initial states for one sample.
void build_initial_states(const RegionContract &contract, const Sample &sample, const Frame &frame,
                          uint64_t seed, uint64_t case_index, MachineState &old_state,
                          MachineState &new_state);

/// @brief Run the differential test of one READY region.
[[nodiscard]] RegionResult test_region(const RegionPair &region, const TestOptions &options,
                                       InputPolicy &policy);

/// @brief Self-contained counterexample bundle (region, contract, states, traces, identity).
[[nodiscard]] Json counterexample_json(const RegionPair &region, const Mismatch &m, uint64_t seed,
                                       const std::string &policy,
                                       const std::vector<TraceStep> &trace_old,
                                       const std::vector<TraceStep> &trace_new);

struct ReplayOutcome {
  bool reproduced = false;
  std::string summary;
  Json detail;
};
/// @brief Re-execute a counterexample bundle from its recorded words (no artifact, no LLVM).
[[nodiscard]] ReplayOutcome replay_counterexample(const Json &bundle, std::string &error);

/// @brief Tool, schema and RocJITsu build identity.
[[nodiscard]] Json build_identity();

} // namespace rocjitsu::isa_diff
