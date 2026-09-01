// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_unmatched_barrier_abort.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

struct UnmatchedBarrierWait {
  const ConSanBarrierSite *site = nullptr;
  std::optional<uint64_t> owner_descriptor_file_offset;
};

template <typename Container>
void append_unmatched_barrier_waits(const Container &container, bool in_kernel,
                                    const ConSanOptions &options,
                                    const ConSanTransformArtifacts &result,
                                    std::vector<UnmatchedBarrierWait> &waits) {
  if (!consan_container_selected(options, container.name))
    return;
  for (const ConSanBarrierSite &site : container.barrier_sites) {
    if (site.operation != ConSanBarrierSite::Operation::Wait || site.size != sizeof(uint32_t) ||
        !site.barrier_id || site.operand_source != ConSanBarrierSite::OperandSource::Immediate)
      continue;
    const ConSanSyncEvent *event = nullptr;
    size_t event_count = 0;
    for (const ConSanSyncEvent &candidate : result.program_inventory.sync().sync_events) {
      if (candidate.kind != ConSanSyncEventKind::Barrier ||
          candidate.operation != ConSanSyncOperation::BarrierWait ||
          candidate.container_name != container.name || candidate.in_kernel != in_kernel ||
          candidate.text_offset != site.text_offset || candidate.size != site.size)
        continue;
      event = &candidate;
      ++event_count;
    }
    if (event_count != 1u || event == nullptr)
      continue;
    const bool belongs_to_sequence = std::ranges::any_of(
        result.program_inventory.sync().sync_sequences, [&](const ConSanSyncSequence &sequence) {
          return sequence.kind == ConSanSyncSequenceKind::Barrier &&
                 sequence.member_event_identities.size() > 1u &&
                 std::ranges::find(sequence.member_event_identities, event->identity) !=
                     sequence.member_event_identities.end();
        });
    if (belongs_to_sequence)
      continue;
    std::optional<uint64_t> owner;
    if constexpr (std::is_same_v<Container, ConSanKernelInfo>)
      owner = container.descriptor_file_offset;
    waits.push_back({&site, owner});
  }
}

} // namespace

void try_apply_unmatched_barrier_wait_abort(std::span<const uint8_t> original_bytes,
                                            const ConSanOptions &options,
                                            ConSanTransformArtifacts &result) {
  if (!options.abort_unmatched_barrier_wait || options.fault_dry_run || !result.errors.empty())
    return;
  std::vector<UnmatchedBarrierWait> waits;
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
    append_unmatched_barrier_waits(kernel, true, options, result, waits);
  for (const ConSanFunctionInfo &function : result.program_inventory.functions())
    append_unmatched_barrier_waits(function, false, options, result, waits);
  if (waits.empty())
    return;

  const std::span<const uint8_t> active_bytes =
      result.replacement.empty() ? original_bytes : std::span<const uint8_t>(result.replacement);
  AmdGpuCodeObject active(active_bytes.data(), active_bytes.size());
  const rj_code_arch_t arch = consan_arch_for_target(active.target_id());
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    return;
  CodeObjectPatcher patcher(active);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan unmatched barrier abort found no executable text");
    return;
  }
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  const uint32_t abort_word = build_s_endpgm(arch);
  std::vector<ConSanPatchInfo> patches;
  for (const UnmatchedBarrierWait &wait : waits) {
    const ConSanBarrierSite &site = *wait.site;
    if (site.text_offset > new_text.size() || site.size > new_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan unmatched barrier abort found an out-of-range wait");
      return;
    }
    const bool overlaps_existing =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          if (patch.original_size == 0)
            return false;
          const uint64_t site_end = site.text_offset + site.size;
          const uint64_t patch_end = patch.anchor_offset + patch.original_size;
          return site.text_offset < patch_end && patch.anchor_offset < site_end;
        });
    if (overlaps_existing) {
      result.errors.emplace_back(
          "ConSan unmatched barrier abort overlaps existing instrumentation");
      return;
    }
    std::memcpy(new_text.data() + site.text_offset, &abort_word, sizeof(abort_word));
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::InlineMalformedBarrierAbort;
    patch.anchor_offset = site.text_offset;
    patch.trampoline_offset = site.text_offset;
    patch.original_size = site.size;
    if (wait.owner_descriptor_file_offset)
      patch.owner_descriptor_file_offsets.push_back(*wait.owner_descriptor_file_offset);
    patches.push_back(std::move(patch));
  }
  if (!replace_consan_text(patcher, new_text, options.patched_image_growth_limit,
                           "unmatched barrier abort", result.program_inventory.code_object_id(),
                           result.errors, &result.transform_failure_cause)) {
    return;
  }
  result.replacement = std::move(patcher).emit();
  result.patches.insert(result.patches.end(), std::make_move_iterator(patches.begin()),
                        std::make_move_iterator(patches.end()));
  result.mark_modified();
  result.warnings.emplace_back("ConSan diagnosed and aborted " + std::to_string(waits.size()) +
                               " statically unmatched barrier wait(s)");
}

} // namespace rocjitsu
