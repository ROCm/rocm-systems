// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file checkpoint.h
/// @brief Simulation checkpoint save and restore via FlatBuffers.

#ifndef ROCJITSU_CONFIG_CHECKPOINT_H_
#define ROCJITSU_CONFIG_CHECKPOINT_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/soc.h"

#include <cstdint>
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
void save_checkpoint(const std::string &path, const SoC &soc, uint64_t tick,
                     const simdojo::SimulationEngine::Config &engine_config);

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
