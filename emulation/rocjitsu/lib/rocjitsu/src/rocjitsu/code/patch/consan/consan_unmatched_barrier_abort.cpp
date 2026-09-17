// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_unmatched_barrier_abort.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_text_relocation.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::consan {
namespace {

struct UnmatchedBarrierWait {
  const BarrierSite *site = nullptr;
  std::optional<uint64_t> owner_descriptor_file_offset;
};

void append_unmatched_barrier_wait(const ProgramSite &decoded, const Request &request,
                                   const DebugOverrides &debug, const ProgramInventory &inventory,
                                   std::vector<UnmatchedBarrierWait> &waits) {
  const BarrierSite *site = decoded.get_if<BarrierSite>();
  const ProgramContainer *container = inventory.container(decoded.container);
  if (site == nullptr || container == nullptr ||
      !container_selected(request, debug, container->name))
    return;
  const SynchronizationInventoryView synchronization = inventory.sync();
  if (site->operation != BarrierSite::Operation::Wait || site->size != sizeof(uint32_t) ||
      !site->barrier_id || site->operand_source != BarrierSite::OperandSource::Immediate)
    return;
  const SyncEvent *event = synchronization.find_event(decoded.id);
  if (event == nullptr || event->operation != SyncOperation::BarrierWait ||
      synchronization.source(*event) != &decoded)
    return;
  const SyncEventId member_identity = synchronization.event_id(*event);
  const bool belongs_to_sequence =
      std::ranges::any_of(synchronization.sync_sequences, [&](const SyncSequence &sequence) {
        return sequence.kind == SyncKind::Barrier && sequence.member_event_ids.size() > 1u &&
               std::ranges::find(sequence.member_event_ids, member_identity) !=
                   sequence.member_event_ids.end();
      });
  if (!belongs_to_sequence) {
    waits.push_back({site, container->is_kernel() ? std::optional{container->descriptor_file_offset}
                                                  : std::nullopt});
  }
}

} // namespace

void try_apply_unmatched_barrier_wait_abort(std::span<const uint8_t> original_bytes,
                                            const Request &request, const DebugOverrides &debug,
                                            const MutationRequest &mutation,
                                            const TransformPolicy &transform_policy,
                                            TransformArtifacts &result) {
  if (!debug.abort_unmatched_barrier_wait || mutation.fault_dry_run || !result.errors.empty())
    return;
  std::vector<UnmatchedBarrierWait> waits;
  for (const ProgramSite &site : result.program_inventory.program_sites())
    append_unmatched_barrier_wait(site, request, debug, result.program_inventory, waits);
  if (waits.empty())
    return;

  const std::span<const uint8_t> active_bytes =
      result.replacement.empty() ? original_bytes : std::span<const uint8_t>(result.replacement);
  const rj_code_arch_t arch = result.program_inventory.arch();
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    return;
  const uint32_t abort_word = build_s_endpgm(arch);
  std::vector<TextFragment> fragments;
  fragments.reserve(waits.size());
  for (const UnmatchedBarrierWait &wait : waits) {
    const BarrierSite &site = *wait.site;
    PatchInfo patch;
    patch.kind = PatchKind::InlineMalformedBarrierAbort;
    patch.anchor_offset = site.text_offset;
    patch.trampoline_offset = site.text_offset;
    patch.original_size = site.size;
    if (wait.owner_descriptor_file_offset)
      patch.owner_descriptor_file_offsets.push_back(*wait.owner_descriptor_file_offset);
    fragments.push_back(TextFragment::replacement({abort_word}, std::move(patch)));
  }
  if (!stage_text_fragments(std::move(fragments), result) ||
      !finalize_text_rewrites(active_bytes, arch, transform_policy.patched_image_growth_limit,
                              "unmatched barrier abort", result)) {
    result.discard_candidate_modification();
    return;
  }
  result.warnings.emplace_back("ConSan diagnosed and aborted " + std::to_string(waits.size()) +
                               " statically unmatched barrier wait(s)");
}

} // namespace rocjitsu::consan
