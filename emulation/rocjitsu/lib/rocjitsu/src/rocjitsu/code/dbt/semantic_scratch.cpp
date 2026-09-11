// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic_scratch.h"

#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "util/bit.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rocjitsu {

SemanticSpillFrame::SemanticSpillFrame(TranslationContext &context)
    : context_(context),
      cursor_(util::align_up(context.semantic_spill_persistent_end, uint32_t{16})) {}

std::optional<SemanticSpillRange> SemanticSpillFrame::allocate_dwords(uint16_t dword_count,
                                                                      uint32_t byte_alignment,
                                                                      uint32_t max_dword_offset) {
  if (dword_count == 0 || byte_alignment == 0)
    return std::nullopt;

  const uint32_t bytes = static_cast<uint32_t>(dword_count) * sizeof(uint32_t);
  auto base = cursor_.preview(bytes, byte_alignment);
  if (!base || static_cast<uint64_t>(*base) + bytes - sizeof(uint32_t) > max_dword_offset)
    return std::nullopt;

  base = cursor_.allocate(bytes, byte_alignment);
  context_.require_private_segment_bytes(cursor_.high_water_mark());
  return SemanticSpillRange{.byte_offset = *base, .dword_count = dword_count};
}

SemanticScratchAllocator::SemanticScratchAllocator(const Instruction &inst,
                                                   const LivenessAnalysis &liveness,
                                                   TranslationContext &context,
                                                   SemanticScratchPolicy policy)
    : inst_(inst), liveness_(liveness), context_(context), policy_(policy), spill_frame_(context) {}

bool SemanticScratchAllocator::window_is_allowed(uint16_t base,
                                                 const SemanticScratchRequest &request,
                                                 uint16_t available_count) const {
  if (request.count == 0 || request.alignment == 0 || base % request.alignment != 0 ||
      static_cast<uint32_t>(base) + request.count > available_count)
    return false;

  RegisterSet candidate;
  candidate.expand(RegisterRef{RegClass::VGPR, base, static_cast<uint8_t>(request.count)});
  return !candidate.intersects(request.forbidden);
}

SemanticScratchResult
SemanticScratchAllocator::acquire_vgprs(const SemanticScratchRequest &request) {
  if (request.count == 0 || request.count > std::numeric_limits<uint8_t>::max() ||
      request.alignment == 0 || request.count > policy_.max_vgprs) {
    return {.lease = std::nullopt, .failure = SemanticScratchFailure::InvalidRequest};
  }

  // A register unused by the decoded kernel is stronger than a point-liveness
  // result: it is dead at every source rewrite boundary. Prefer such a tuple
  // inside the descriptor allocation so the common case avoids CFG live-before
  // computation and descriptor growth.
  const uint16_t allocated_vgprs =
      static_cast<uint16_t>(std::min<uint32_t>(context_.num_vgprs, policy_.max_vgprs));
  uint16_t unused_search_start = 0;
  while (auto unused = liveness_.find_globally_unused_vgpr_run(
             &inst_, request.count, unused_search_start, request.alignment, allocated_vgprs)) {
    if (window_is_allowed(*unused, request, allocated_vgprs)) {
      context_.require_vgprs(static_cast<uint32_t>(*unused) + request.count);
      return {.lease = SemanticScratchLease{.reg_class = RegClass::VGPR,
                                            .base = *unused,
                                            .count = request.count,
                                            .spilled = false,
                                            .spill_offset = 0},
              .failure = SemanticScratchFailure::None};
    }
    unused_search_start = static_cast<uint16_t>(*unused + request.alignment);
  }

  uint16_t search_start = 0;
  while (auto free =
             liveness_.find_free_run(&inst_, request.count, search_start, request.alignment)) {
    if (static_cast<uint32_t>(*free) + request.count > policy_.max_vgprs)
      break;
    if (window_is_allowed(*free, request, policy_.max_vgprs)) {
      context_.require_vgprs(static_cast<uint32_t>(*free) + request.count);
      return {.lease = SemanticScratchLease{.reg_class = RegClass::VGPR,
                                            .base = *free,
                                            .count = request.count,
                                            .spilled = false,
                                            .spill_offset = 0},
              .failure = SemanticScratchFailure::None};
    }
    search_start = static_cast<uint16_t>(*free + request.alignment);
  }

  if (!request.allow_spill)
    return {.lease = std::nullopt, .failure = SemanticScratchFailure::NoRegisterWindow};

  std::optional<uint16_t> victim;
  if (request.preferred_victim_base &&
      window_is_allowed(*request.preferred_victim_base, request, allocated_vgprs)) {
    victim = request.preferred_victim_base;
  } else {
    for (uint16_t base = 0; static_cast<uint32_t>(base) + request.count <= allocated_vgprs;
         ++base) {
      if (window_is_allowed(base, request, allocated_vgprs)) {
        victim = base;
        break;
      }
    }
  }

  if (!victim)
    return {.lease = std::nullopt, .failure = SemanticScratchFailure::NoRegisterWindow};

  if (context_.uses_dynamic_stack) {
    return {.lease = std::nullopt, .failure = SemanticScratchFailure::DynamicStackUnsupported};
  }

  auto spill = allocate_spill_dwords(request.count);
  if (!spill)
    return {.lease = std::nullopt, .failure = SemanticScratchFailure::SpillOffsetUnencodable};

  return {.lease = SemanticScratchLease{.reg_class = RegClass::VGPR,
                                        .base = *victim,
                                        .count = request.count,
                                        .spilled = true,
                                        .spill_offset = spill->byte_offset},
          .failure = SemanticScratchFailure::None};
}

std::optional<SemanticSpillRange>
SemanticScratchAllocator::allocate_spill_dwords(uint16_t dword_count, uint32_t byte_alignment) {
  return spill_frame_.allocate_dwords(dword_count, byte_alignment, policy_.max_spill_dword_offset);
}

} // namespace rocjitsu
