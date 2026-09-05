// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sync_emission.h
/// @brief Shared typed synchronization planning and native emission contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

namespace rocjitsu {
class CodeObjectPatcher;
}

namespace rocjitsu::consan_moi_impl {

enum class SampledAtomicSemanticsReason : uint8_t {
  None,
  UnqualifiedSharedSyncSequence,
  UnsupportedQualifiedMemoryRole,
  MissingQualifiedScope,
  UnsupportedQualifiedScope,
  UnsupportedQualifiedByteRange,
  CompareExchangeDynamicOutcomeUnavailable,
  UnsupportedQualifiedRmwOutcome,
  SampledSyncAbiRejectedQualifiedSequence,
  SampledSyncAbiRejectedCasFailure,
  Count,
};

struct SampledAtomicSemanticsResult {
  std::optional<consan_detail::SampledAtomicSemantics> semantics;
  SampledAtomicSemanticsReason reason = SampledAtomicSemanticsReason::None;
};

[[nodiscard]] SampledAtomicSemanticsResult
sampled_atomic_semantics_for_source(const MoiAtomicEvidenceSourceView &source);

[[nodiscard]] std::string_view
sampled_atomic_semantics_reason_name(SampledAtomicSemanticsReason reason);

[[nodiscard]] uint16_t fence_record_scratch_count(const ConSanAtomicLoweringForm &form);
[[nodiscard]] uint16_t atomic_record_scratch_count(const ConSanAtomicLoweringForm &form);

[[nodiscard]] bool append_atomic_scalar_clause_patch(
    std::span<const uint8_t> text, const ConSanAtomicSite &site,
    std::optional<uint64_t> scalar_clause_text_offset, rj_code_arch_t arch,
    std::vector<ConSanPatchInfo> &patches, std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl
