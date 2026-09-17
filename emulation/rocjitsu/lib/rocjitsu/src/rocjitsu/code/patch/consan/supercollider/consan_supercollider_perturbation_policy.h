// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_perturbation_policy.h
/// @brief Target-neutral admission policy for SuperCollider perturbations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

namespace rocjitsu::consan {

/// Classify whether one exact synchronization sequence is eligible for the
/// requested perturbation edge. Planning and validation branch on this typed
/// reason; diagnostic code renders it separately.
[[nodiscard]] SuperColliderPerturbationRejectionReason supercollider_perturbation_rejection_reason(
    const SynchronizationInventoryView &events, const SyncSequence &sequence,
    SuperColliderPerturbationKind kind, SuperColliderPerturbationEdge edge);

} // namespace rocjitsu::consan
