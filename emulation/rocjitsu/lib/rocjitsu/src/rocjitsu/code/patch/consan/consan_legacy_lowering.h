// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_legacy_lowering.h
/// @brief Internal entry to the prototype ConSan lowering implementation.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

namespace rocjitsu {

/// Invoke the prototype lowerer with its historical mutable option bundle.
///
/// Production callers must enter through `transform_consan` or
/// `transform_consan_with_mutation`. This declaration is intentionally kept
/// out of `consan.h`: only the typed pipeline implementation, the HSA hook's
/// temporary test-override seam, the mechanism-level host tests, and the
/// transform fuzzer may bypass the production entry while the prototype
/// lowerer is incrementally decomposed. The returned `ConSanResult` is the
/// compatibility mechanism record, not the production control-plane contract.
[[nodiscard]] ConSanResult try_patch_consan(std::span<const uint8_t> code_object_bytes,
                                            const ConSanOptions &options);

} // namespace rocjitsu
