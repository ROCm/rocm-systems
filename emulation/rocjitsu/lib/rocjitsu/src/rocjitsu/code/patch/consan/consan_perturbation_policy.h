// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_perturbation_policy.h
/// @brief Target-neutral admission policy for SuperCollider perturbations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

namespace rocjitsu {

/// Classify whether one exact synchronization sequence is eligible for the
/// requested perturbation edge. Planning and validation branch on this typed
/// reason; diagnostic code renders it separately.
[[nodiscard]] ConSanPerturbationRejectionReason
perturbation_rejection_reason(const SyncEventSemanticIndex &events,
                              const ConSanSyncSequence &sequence, ConSanPerturbationKind kind,
                              ConSanPerturbationEdge edge);

} // namespace rocjitsu
