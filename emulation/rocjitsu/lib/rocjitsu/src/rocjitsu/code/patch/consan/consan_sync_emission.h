// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_sync_emission.h
/// @brief Shared typed synchronization planning and native emission contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_probe_contracts.h"

namespace rocjitsu {
class CodeObjectPatcher;
}

namespace rocjitsu::consan {}

namespace rocjitsu::consan::detail {

[[nodiscard]] bool
append_atomic_scalar_clause_patch(std::span<const uint8_t> text, const AtomicSite &site,
                                  std::optional<uint64_t> scalar_clause_text_offset,
                                  rj_code_arch_t arch, std::vector<PatchInfo> &patches,
                                  std::vector<std::string> &errors);

} // namespace rocjitsu::consan::detail
