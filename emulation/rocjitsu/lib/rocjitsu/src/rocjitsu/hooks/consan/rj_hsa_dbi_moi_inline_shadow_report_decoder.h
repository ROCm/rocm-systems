// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiExactShadowEvidence {
  uint32_t index = 0;
  ConSanMoiExactShadowEntry entry;
  ConSanMoiExactByteCellProvenance byte_provenance;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
};

struct AutoMoiInlineAtomicReleaseEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAtomicReleaseSlot slot;
  ConSanMoiInlineCausalSnapshot snapshot;
};

struct AutoMoiInlineAcquiredTokenEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAcquiredEpochTokenSlot token;
};

struct AutoMoiInlineShadowDecodedReport {
  uint32_t deferred_token_qualified_diagnostic_count = 0;
  std::vector<AutoMoiExactShadowEvidence> exact_shadow;
  std::vector<AutoMoiInlineAtomicReleaseEvidence> atomic_releases;
  std::vector<AutoMoiInlineAcquiredTokenEvidence> acquired_tokens;
};

} // namespace rocjitsu::consan_hook
