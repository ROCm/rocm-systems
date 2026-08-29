// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_perturbation_policy.h
/// @brief Target-neutral admission policy for SuperCollider perturbations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"

#include <string>

namespace rocjitsu {

/// Return an empty string when one exact synchronization sequence is eligible
/// for the requested perturbation edge, otherwise a stable typed-in-practice
/// reason token consumed by planning and proof rederivation.
[[nodiscard]] std::string perturbation_rejection_reason(const SyncEventSemanticIndex &events,
                                                        const ConSanSyncSequence &sequence,
                                                        ConSanPerturbationKind kind,
                                                        ConSanPerturbationEdge edge);

} // namespace rocjitsu
