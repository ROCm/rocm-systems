// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_unmatched_barrier_abort.h
/// @brief Opt-in fail-closed mutation of statically unmatched barrier waits.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <cstdint>
#include <span>

namespace rocjitsu {

/// Replace selected statically unmatched barrier waits with a terminating
/// instruction. The resulting bytes and patch proof remain part of the same
/// lowering transaction and are independently validated before publication.
void try_apply_unmatched_barrier_wait_abort(std::span<const uint8_t> original_bytes,
                                            const ConSanOptions &options,
                                            ConSanTransformArtifacts &result);

} // namespace rocjitsu
