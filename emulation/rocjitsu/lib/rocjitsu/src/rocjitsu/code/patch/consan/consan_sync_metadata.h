// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_sync_metadata.h
/// @brief Metadata contracts for ordinary-memory synchronization sequences.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>

namespace rocjitsu {

/// The control-flow and target-form context in which acquire metadata is
/// being screened. Instruction-run and path proofs remain the caller's
/// responsibility.
enum class ConSanOrdinaryAcquireMetadataPolicy : uint8_t {
  SameBlockSingleFence,
  BoundedPathSingleFence,
  SameBlockRdna3CachePairMember,
  BoundedPathRdna3CachePairMember,
};

[[nodiscard]] bool consan_ordinary_acquire_metadata_compatible(
    const ConSanSyncEvent &load, const ConSanSyncSequence &load_sequence,
    const ConSanSyncEvent &cache, const ConSanSyncSequence &cache_sequence,
    ConSanOrdinaryAcquireMetadataPolicy policy);

} // namespace rocjitsu
