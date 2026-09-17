// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_sync_metadata.h
/// @brief Metadata contracts for ordinary-memory synchronization sequences.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>

namespace rocjitsu::consan {

/// The control-flow and target-form context in which acquire metadata is
/// being screened. Instruction-run and path proofs remain the caller's
/// responsibility.
enum class OrdinaryAcquireMetadataPolicy : uint8_t {
  SameBlockSingleFence,
  BoundedPathSingleFence,
  SameBlockCachePairMember,
  BoundedPathCachePairMember,
};

/// Classify the cache-maintenance endpoints that can carry an atomic or
/// ordinary-memory release/acquire edge. Ordered multi-event target forms are
/// deliberately handled by their exact sequence matchers instead.
[[nodiscard]] bool is_release_cache_event(std::span<const ProgramSite> program_sites,
                                          const SyncEvent *event);
[[nodiscard]] bool is_acquire_cache_event(std::span<const ProgramSite> program_sites,
                                          const SyncEvent *event);

[[nodiscard]] bool ordinary_acquire_metadata_compatible(std::span<const ProgramSite> program_sites,
                                                        const SyncEvent &load,
                                                        const SyncSequence &load_sequence,
                                                        const SyncEvent &cache,
                                                        const SyncSequence &cache_sequence,
                                                        OrdinaryAcquireMetadataPolicy policy);

[[nodiscard]] bool ordinary_acquire_metadata_compatible(std::span<const ProgramSite> program_sites,
                                                        const SyncEvent &load,
                                                        const SyncSequence &load_sequence,
                                                        const SyncEvent &cache,
                                                        const SyncSequence &cache_sequence);

[[nodiscard]] bool ordinary_release_metadata_compatible(std::span<const ProgramSite> program_sites,
                                                        const SyncEvent &cache,
                                                        const SyncSequence &cache_sequence,
                                                        const SyncEvent &store,
                                                        const SyncSequence &store_sequence);

} // namespace rocjitsu::consan
