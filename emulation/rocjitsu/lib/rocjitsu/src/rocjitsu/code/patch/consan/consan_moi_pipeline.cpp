// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"

#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiFenceEvidenceSitePlan;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_pipeline.inc"

namespace {

using K = ConSanPatchKind;

struct Summary {
  K kind;
  const char *engine;
  bool counted;
  std::string_view text;
};

// A null engine is mode-independent; an empty engine names the active mode.
constexpr std::array kMoiLoweringSummaries{
    Summary{K::InlineMoiAccessRecordStore, "", false, "emitted a first-light access record probe"},
    Summary{K::TrampolineMoiAccessRecordStore, "", false,
            "emitted an appended-cave first-light access record probe"},
    Summary{K::InlineMoiExactShadowStore, "inline-shadow", false,
            "emitted an exact-shadow publish probe"},
    Summary{K::TrampolineMoiExactShadowStore, "inline-shadow", false,
            "emitted an appended-cave exact-shadow publish probe"},
    Summary{K::InlineMoiSampledWatchpointStore, "sampled", false,
            "emitted a direct sampled watchpoint probe"},
    Summary{K::TrampolineMoiSampledWatchpointStore, "sampled", false,
            "emitted an appended-cave direct sampled watchpoint probe"},
    Summary{K::KernelEntryMoiOwnerEpochPrologue, nullptr, false,
            "initialized owner/epoch VGPRs with a kernel-entry prologue"},
    Summary{K::KernelEntryMoiPrivateEpochPrologue, nullptr, false,
            "initialized private epoch state with a kernel-entry prologue"},
    Summary{K::TrampolineMoiBarrierRecord, nullptr, true, "barrier record probe(s)"},
    Summary{K::TrampolineMoiInlineEpochBarrier, "", true, "barrier epoch probe(s)"},
    Summary{K::TrampolineMoiInlineAtomicOrdering, "inline-shadow", true,
            "inline atomic ordering probe(s)"},
    Summary{K::TrampolineMoiAtomicRecord, nullptr, true, "atomic record probe(s)"},
    Summary{K::TrampolineMoiSampledSyncMetadata, "sampled", true, "typed synchronization probe(s)"},
    Summary{K::TrampolineMoiFenceRecord, nullptr, true, "fence record probe(s)"},
};

} // namespace

std::vector<std::string> summarize_moi_lowering(ConSanMoiEngine engine, bool modified,
                                                std::span<const ConSanPatchKind> patch_kinds) {
  if (!modified)
    return {std::string("ConSan MOI ") + consan_moi_engine_name(engine) +
            " engine is an inventory-only stub"};

  std::vector<std::string> warnings;
  for (const Summary &entry : kMoiLoweringSummaries) {
    const auto count = static_cast<uint32_t>(std::ranges::count(patch_kinds, entry.kind));
    if (count == 0)
      continue;
    std::string warning = "ConSan MOI ";
    if (entry.engine != nullptr) {
      warning += *entry.engine == '\0' ? consan_moi_engine_name(engine) : entry.engine;
      warning += " engine ";
    }
    if (entry.counted)
      warning += "emitted " + std::to_string(count) + " ";
    warning += entry.text;
    warnings.push_back(std::move(warning));
  }
  return warnings;
}

} // namespace consan_moi_impl
} // namespace rocjitsu
