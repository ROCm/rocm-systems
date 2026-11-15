// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file result_cache.h
/// @brief Result and analysis caches with conservative invalidation (handoff 11).
///
/// @details Layers and what each key binds:
///   - analysis: both artifacts' complete input hashes, the kernel names, every planning
///     option, the tool/schema versions and the build fingerprint;
///   - test result: the prepared region (both instruction streams with PCs, the complete
///     contract including observable/frame sets, mapping, constraints and exit mapping),
///     the prepared keys of its expansion variants, the policy, seed and every budget, the
///     tool/schema versions and the build fingerprint.
/// Identical region bytes alone never hit: a liveness change upstream or downstream
/// changes the contract and with it the prepared key. The build fingerprint hashes the
/// running executable and every executable file mapping, so any rebuild of RocJITsu or
/// this tool invalidates everything; P0 does not track finer helper dependencies.
#pragma once

#include "artifact.h"
#include "differential.h"
#include "region.h"
#include "region_planner.h"

#include <optional>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

/// @brief SHA-256 over the running executable and its executable file mappings. Computed
/// once per process; "unavailable" if /proc/self cannot be read (then nothing is cached).
[[nodiscard]] const std::string &build_fingerprint();

[[nodiscard]] std::string result_cache_key(const RegionPair &region,
                                           const std::vector<RegionPair> &variants,
                                           const TestOptions &options, const std::string &policy);

[[nodiscard]] std::string analysis_cache_key(const Artifact &old_artifact,
                                             const Artifact &new_artifact,
                                             const PlanOptions &options);

/// @brief Load a cached region result. The returned result carries the stored JSON
/// (annotated with the cache hit) and the fields that aggregation needs; counterexample
/// paths point at the copies kept inside the cache entry.
[[nodiscard]] std::optional<RegionResult> load_cached_result(const std::string &cache_dir,
                                                             const std::string &key);

/// @brief Store a result together with copies of its counterexample bundles.
bool store_cached_result(const std::string &cache_dir, const std::string &key,
                         const RegionResult &result);

} // namespace rocjitsu::isa_diff
