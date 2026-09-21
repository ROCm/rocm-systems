// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file checkpoint.h
/// @brief Simulation checkpoint save and restore via FlatBuffers.

#ifndef ROCJITSU_CONFIG_CHECKPOINT_H_
#define ROCJITSU_CONFIG_CHECKPOINT_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/soc.h"

#include <cstdint>
#include <span>
#include <string>

namespace rocjitsu {
namespace config {

/// @brief Save the current simulation state to a binary FlatBuffer file.
///
/// Captures the full state of all wavefronts, compute units, the command
/// processor, and GPU memory pages. The resulting file can be loaded with
/// restore_checkpoint() to resume simulation from this point.
/// @param path Output file path for the binary checkpoint.
/// @param soc The SoC to serialize.
/// @param tick Current simulation tick.
/// @param engine_config Engine configuration to persist alongside the SoC state.
/// @param cpu_dispatch_threads Original functional dispatch-width request. Zero
/// selects automatic sizing; nonzero values are explicit per-SoC widths.
/// @param cpu_thread_budget Original total budget request; zero uses receiving-host affinity.
/// @param thread_allocations Preferred allocations to re-evaluate when restoring.
/// @param legacy_auto_dispatch Preserve an old automatic-dispatch checkpoint's
/// absent allocation vector when saving it again; false for new configurations.
/// @param async_helper_threads Original helper request; -1 selects the table.
/// Zero preserves disabled-helper behavior for existing save callers.
void save_checkpoint(const std::string &path, const SoC &soc, uint64_t tick,
                     const simdojo::SimulationEngine::Config &engine_config,
                     uint32_t cpu_dispatch_threads, uint32_t cpu_thread_budget = 0,
                     std::span<const ExecutionThreadChoice> thread_allocations = {},
                     bool legacy_auto_dispatch = false, int32_t async_helper_threads = 0);

/// @brief Restore simulation state from a binary FlatBuffer checkpoint.
///
/// Rebuilds the component tree from the stored config, then restores all
/// wavefront register state and GPU memory pages to match the checkpointed
/// values. Scheduling indices (CU round-robin position, CP dispatch cursor)
/// are saved in the checkpoint but not yet restored; simulation resumes with
/// the same correctness guarantees but may schedule wavefronts in a different
/// order than the original run.
/// @param path Path to the binary checkpoint file.
/// @returns LoadedConfig with engine parameters and a topology with restored state.
LoadedConfig restore_checkpoint(const std::string &path);

} // namespace config
} // namespace rocjitsu

#endif // ROCJITSU_CONFIG_CHECKPOINT_H_
