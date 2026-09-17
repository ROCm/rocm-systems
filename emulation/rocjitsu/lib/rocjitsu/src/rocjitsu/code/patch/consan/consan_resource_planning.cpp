// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_resource_planning.h"

#include "rocjitsu/code/patch/consan/consan_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_sync_emission.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::consan {

using detail::AtomicEvidenceSitePlan;
using detail::BarrierEvidenceSitePlan;

namespace detail {

#include "rocjitsu/code/patch/consan/consan_resource_planning.inc"

namespace {

using K = PatchKind;

struct Summary {
  K kind;
  bool counted;
  std::string_view text;
};

constexpr std::array kLoweringSummaries{
    Summary{K::InlineWatchpointStore, false, "emitted a direct sampled watchpoint probe"},
    Summary{K::TrampolineWatchpointStore, false,
            "emitted an appended-cave direct sampled watchpoint probe"},
    Summary{K::KernelEntryOwnerEpochPrologue, false,
            "initialized owner/epoch VGPRs with a kernel-entry prologue"},
    Summary{K::KernelEntryPrivateEpochPrologue, false,
            "initialized private epoch state with a kernel-entry prologue"},
    Summary{K::TrampolineSyncMetadata, true, "typed synchronization probe(s)"},
};

} // namespace

std::vector<std::string> summarize_lowering(bool modified, std::span<const PatchKind> patch_kinds) {
  if (!modified)
    return {"ConSan emitted no instrumentation"};

  std::vector<std::string> warnings;
  for (const Summary &entry : kLoweringSummaries) {
    const auto count = static_cast<uint32_t>(std::ranges::count(patch_kinds, entry.kind));
    if (count == 0)
      continue;
    std::string warning = "ConSan ";
    if (entry.counted)
      warning += "emitted " + std::to_string(count) + " ";
    warning += entry.text;
    warnings.push_back(std::move(warning));
  }
  return warnings;
}

} // namespace detail
} // namespace rocjitsu::consan
