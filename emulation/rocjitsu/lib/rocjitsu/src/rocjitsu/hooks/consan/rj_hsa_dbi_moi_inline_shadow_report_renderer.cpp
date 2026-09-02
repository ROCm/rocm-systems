// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_inline_shadow_report_renderer.h"

#include <algorithm>

namespace rocjitsu::consan_hook {

AutoMoiModeRendering render_auto_moi_inline_shadow_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiInlineShadowDecodedReport &mode) {
  AutoMoiModeRendering result;
  result.effective_diagnostic_count =
      header.diagnostic_count >= mode.deferred_token_qualified_diagnostic_count
          ? header.diagnostic_count - mode.deferred_token_qualified_diagnostic_count
          : 0u;
  result.summary_fields = format_auto_moi_report_text(
      " visible_inline_publications=%llu deferred_token_qualified_diagnostics=%u "
      "exact_shadow_capacity=%u visible_exact_shadow=%zu exact_incomplete_snapshots=%llu "
      "exact_changed_snapshots=%llu exact_malformed_snapshots=%llu "
      "inline_atomic_release_capacity=%u visible_inline_atomic_releases=%zu "
      "release_incomplete_snapshots=%llu release_changed_snapshots=%llu "
      "release_overflow_snapshots=%llu release_source_incomplete_snapshots=%llu "
      "release_malformed_snapshots=%llu inline_acquired_token_capacity=%u "
      "visible_inline_acquired_tokens=%zu "
      "token_incomplete_snapshots=%llu token_changed_snapshots=%llu "
      "token_malformed_snapshots=%llu inline_undercoverage=%llu inline_overflow=%llu "
      "inline_unsupported=%llu inline_malformed=%llu",
      static_cast<unsigned long long>(summary.visible_inline_publication_count),
      mode.deferred_token_qualified_diagnostic_count, header.exact_shadow_entry_capacity,
      mode.exact_shadow.size(),
      static_cast<unsigned long long>(summary.exact_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.exact_changed_snapshot_count),
      static_cast<unsigned long long>(summary.exact_malformed_snapshot_count),
      header.inline_atomic_release_capacity, mode.atomic_releases.size(),
      static_cast<unsigned long long>(summary.release_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.release_changed_snapshot_count),
      static_cast<unsigned long long>(summary.release_overflow_snapshot_count),
      static_cast<unsigned long long>(summary.release_source_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.release_malformed_snapshot_count),
      header.inline_acquired_epoch_token_capacity, mode.acquired_tokens.size(),
      static_cast<unsigned long long>(summary.token_incomplete_snapshot_count),
      static_cast<unsigned long long>(summary.token_changed_snapshot_count),
      static_cast<unsigned long long>(summary.token_malformed_snapshot_count),
      static_cast<unsigned long long>(summary.inline_undercoverage_count),
      static_cast<unsigned long long>(summary.inline_overflow_count),
      static_cast<unsigned long long>(summary.inline_unsupported_count),
      static_cast<unsigned long long>(summary.inline_malformed_count));

  constexpr AutoMoiReportDiagnosticKind kEvidence = AutoMoiReportDiagnosticKind::Evidence;
  constexpr AutoMoiReportDiagnosticKind kDetail = AutoMoiReportDiagnosticKind::Detail;
  if (mode.deferred_token_qualified_diagnostic_count != 0) {
    result.evidence.push_back(
        {kEvidence,
         format_auto_moi_report_text(
             "ConSan MOI deferred-token qualification reader=%llu ordered_diagnostics=%u",
             static_cast<unsigned long long>(input.reader),
             mode.deferred_token_qualified_diagnostic_count)});
  }
  bool rendered_first_exact_malformed = false;
  for (const AutoMoiInlineShadowEvidenceIssue &issue : mode.issues) {
    switch (issue.reason) {
    case AutoMoiInlineShadowEvidenceReason::ExactMalformed:
      if (!rendered_first_exact_malformed) {
        result.evidence.push_back(
            {kEvidence,
             format_auto_moi_report_text(
                 "ConSan MOI first malformed exact snapshot reader=%llu index=%u "
                 "version_before=%u packed_access=0x%016llx dispatch_id=0x%016llx "
                 "byte_provenance=0x%08x version_after=%u",
                 static_cast<unsigned long long>(input.reader), issue.index,
                 static_cast<uint32_t>(issue.words[0]),
                 static_cast<unsigned long long>(issue.words[1]),
                 static_cast<unsigned long long>(issue.words[2]),
                 static_cast<uint32_t>(issue.words[3]), static_cast<uint32_t>(issue.words[4]))});
        rendered_first_exact_malformed = true;
      }
      break;
    case AutoMoiInlineShadowEvidenceReason::ReleasePublishing:
      result.evidence.push_back(
          {kEvidence,
           format_auto_moi_report_text(
               "ConSan MOI auto incomplete-inline-atomic-release reader=%llu index=%u "
               "version=%u owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx "
               "dispatch=0x%llx",
               static_cast<unsigned long long>(input.reader), issue.index,
               static_cast<uint32_t>(issue.words[0]), static_cast<uint32_t>(issue.words[1]),
               static_cast<uint32_t>(issue.words[2]), static_cast<uint32_t>(issue.words[3]),
               static_cast<unsigned long long>(issue.words[4]),
               static_cast<unsigned long long>(issue.words[5]))});
      break;
    case AutoMoiInlineShadowEvidenceReason::CompactDiagnosticTokenUnresolved:
      result.evidence.push_back(
          {kEvidence,
           format_auto_moi_report_text(
               "ConSan MOI compact diagnostic token unresolved reader=%llu current=0x%x "
               "tagged=0x%x well_formed=%s current_ambiguous=%s prior_ambiguous=%s",
               static_cast<unsigned long long>(input.reader), issue.index,
               static_cast<uint32_t>(issue.words[0]), issue.words[1] != 0 ? "true" : "false",
               issue.words[2] != 0 ? "true" : "false", issue.words[3] != 0 ? "true" : "false")});
      break;
    }
  }

  for (const AutoMoiInlineAtomicReleaseEvidence &release : mode.atomic_releases) {
    const ConSanMoiInlineAtomicReleaseSlot &slot = release.slot;
    const ConSanMoiInlineCausalSnapshot &snapshot = release.snapshot;
    result.details.push_back(
        {kDetail,
         format_auto_moi_report_text(
             "ConSan MOI auto inline-atomic-release reader=%llu index=%u version=%u "
             "owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx dispatch=0x%llx "
             "snapshot_flags=%u snapshot_count=%u snapshot0_owner=%u "
             "snapshot0_epoch_plus_one=%u snapshot1_owner=%u snapshot1_epoch_plus_one=%u "
             "snapshot2_owner=%u snapshot2_epoch_plus_one=%u snapshot3_owner=%u "
             "snapshot3_epoch_plus_one=%u",
             static_cast<unsigned long long>(input.reader), release.index, slot.version,
             slot.owner_id, slot.epoch_plus_one, slot.workgroup_key,
             static_cast<unsigned long long>(slot.atomic_address),
             static_cast<unsigned long long>(slot.dispatch_id), snapshot.flags,
             snapshot.entry_count, snapshot.entries[0].ancestor_owner_id,
             snapshot.entries[0].ancestor_epoch_plus_one, snapshot.entries[1].ancestor_owner_id,
             snapshot.entries[1].ancestor_epoch_plus_one, snapshot.entries[2].ancestor_owner_id,
             snapshot.entries[2].ancestor_epoch_plus_one, snapshot.entries[3].ancestor_owner_id,
             snapshot.entries[3].ancestor_epoch_plus_one)});
  }
  for (const AutoMoiInlineAcquiredTokenEvidence &entry_token : mode.acquired_tokens) {
    const ConSanMoiInlineAcquiredEpochTokenSlot &token = entry_token.token;
    result.details.push_back(
        {kDetail,
         format_auto_moi_report_text(
             "ConSan MOI auto inline-acquired-token reader=%llu index=%u version=%u "
             "kind=%s consumer=%u producer=%u producer_epoch_plus_one=%u "
             "consumer_epoch_plus_one=%u workgroup=%u dispatch=0x%llx "
             "source_address=0x%llx source_version=%u",
             static_cast<unsigned long long>(input.reader), entry_token.index, token.version,
             token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct)
                 ? "direct"
             : token.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited)
                 ? "inherited"
                 : "release-sequence",
             token.consumer_owner_id, token.producer_owner_id, token.producer_epoch_plus_one,
             token.consumer_epoch_plus_one, token.workgroup_key,
             static_cast<unsigned long long>(token.dispatch_id),
             static_cast<unsigned long long>(token.source_release_address),
             token.source_release_version)});
  }
  for (uint32_t i = 0; i < std::min<size_t>(mode.exact_shadow.size(), 4u); ++i) {
    const AutoMoiExactShadowEvidence &entry = mode.exact_shadow[i];
    result.details.push_back(
        {kEvidence,
         format_auto_moi_report_text(
             "ConSan MOI auto exact-shadow reader=%llu index=%u kind=%u owner=%u epoch=%u "
             "generation=%u inst=0x%x bytes=[%u,%u) lane=%u dispatch=0x%llx version=%u",
             static_cast<unsigned long long>(input.reader), entry.index,
             static_cast<uint32_t>(entry.entry.kind), entry.entry.owner_id, entry.entry.epoch,
             entry.entry.generation, entry.entry.instruction_offset,
             entry.byte_provenance.byte_offset,
             entry.byte_provenance.byte_offset + entry.byte_provenance.byte_count,
             entry.byte_provenance.representative_lane,
             static_cast<unsigned long long>(entry.dispatch_id), entry.version)});
  }
  return result;
}

} // namespace rocjitsu::consan_hook
