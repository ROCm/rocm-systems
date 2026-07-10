// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/spill_manager.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "util/bit.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility> // for std::pair

namespace rocjitsu {

namespace {

/// Per-class hardware bound. Indices >= this are not representable in
/// RegisterSet's bitsets and indicate either a programming error or a class
/// that RegisterSet doesn't track (EXEC, VCC, etc.).
[[nodiscard]] size_t per_class_max(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    return REGISTER_SET_MAX_SGPRS;
  case RegClass::VGPR:
    return REGISTER_SET_MAX_VGPRS;
  case RegClass::ACC_VGPR:
    return REGISTER_SET_MAX_ACC_VGPRS;
  default:
    return 0; // class not tracked — every index rejected
  }
}

} // namespace

SpillManager::SpillManager(uint32_t original_private_bytes, uint32_t per_lane_scratch_limit)
    : base_offset_(util::align_up(original_private_bytes, kDbiZoneAlignment)),
      total_bytes_(base_offset_), limit_(per_lane_scratch_limit), next_offset_(base_offset_) {}

std::optional<uint32_t> SpillManager::allocate_slot(RegisterRef reg) {
  // Reject indices past the per-class hardware bound (or unsupported classes
  // like EXEC/VCC). The cache lookup happens first so an idempotent re-alloc
  // of an already-cached register cannot fail this check (it never could have
  // been cached without passing the check the first time).
  const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
  auto it = reg_to_offset_.find(key);
  if (it != reg_to_offset_.end()) {
    return it->second;
  }
  if (reg.index >= per_class_max(reg.cls)) {
    return std::nullopt;
  }
  // Overflow-safe equivalent of `next_offset_ + kSlotBytes > limit_`.
  if (static_cast<uint64_t>(next_offset_) + kSlotBytes > limit_) {
    return std::nullopt;
  }
  const uint32_t offset = next_offset_;
  next_offset_ += kSlotBytes;
  total_bytes_ = next_offset_;
  reg_to_offset_.emplace(key, offset);
  return offset;
}

std::optional<uint32_t> SpillManager::allocate_slots(RegisterRef reg, unsigned width) {
  if (width == 0)
    return std::nullopt;
  // Reject ranges that would wrap the uint16_t register index space.
  if (static_cast<uint32_t>(reg.index) + width - 1 > std::numeric_limits<uint16_t>::max()) {
    return std::nullopt;
  }
  // RegisterRef::width is uint8_t; reject anything that wouldn't fit.
  if (width > std::numeric_limits<uint8_t>::max()) {
    return std::nullopt;
  }
  // Reject ranges that would extend past the per-class hardware bound —
  // RegisterSet::expand would silently truncate them, leaving the caller
  // with a short allocation and no error.
  if (static_cast<size_t>(reg.index) + width > per_class_max(reg.cls)) {
    return std::nullopt;
  }

  // Build a width-N range and reserve
  RegisterSet set;
  set.expand(RegisterRef{reg.cls, reg.index, static_cast<uint8_t>(width)});
  if (!reserve(set))
    return std::nullopt;
  // width=1 is irrelevant here; offset_for keys on (cls, index) only.
  return offset_for(RegisterRef{reg.cls, reg.index, 1});
}

bool SpillManager::reserve(const RegisterSet &set) {
  // Count NEW registers (cache misses) so we can size-check upfront.
  unsigned num_new = 0;
  set.for_each([&](RegisterRef reg) {
    const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
    if (!reg_to_offset_.contains(key))
      ++num_new;
  });
  if (num_new > 0 && static_cast<uint64_t>(next_offset_) + kSlotBytes * num_new > limit_) {
    return false;
  }

  // Capacity check passed — no failure possible from here. Every reg from
  // for_each is within per-class bounds (the bitset itself enforces that),
  // and we just verified there's enough room.
  set.for_each([this](RegisterRef reg) {
    [[maybe_unused]] auto off = allocate_slot(reg);
    assert(off.has_value() && "allocate_slot failed after capacity check");
  });
  return true;
}

std::optional<uint32_t> SpillManager::offset_for(RegisterRef reg) const {
  const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
  auto it = reg_to_offset_.find(key);
  if (it == reg_to_offset_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<VgprSpillSequence> build_vgpr_spill_sequence(SpillManager &manager,
                                                           uint16_t vgpr_base, uint16_t vgpr_count,
                                                           rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || vgpr_count == 0 ||
      static_cast<uint32_t>(vgpr_base) + vgpr_count > REGISTER_SET_MAX_VGPRS) {
    return std::nullopt;
  }

  SpillManager planned_manager = manager;
  const auto first_offset =
      planned_manager.allocate_slots(RegisterRef{RegClass::VGPR, vgpr_base, 1}, vgpr_count);
  if (!first_offset || planned_manager.total_private_bytes() > kMaxAddressFreeScratchPrivateBytes) {
    return std::nullopt;
  }

  const auto wait_store = build_s_wait_storecnt0(arch);
  const auto wait_load = build_s_wait_loadcnt0(arch);
  if (!wait_store || !wait_load)
    return std::nullopt;

  VgprSpillSequence sequence;
  sequence.vgpr_base = vgpr_base;
  sequence.vgpr_count = vgpr_count;
  sequence.slot_offsets.reserve(vgpr_count);
  sequence.save_words.reserve(static_cast<size_t>(vgpr_count) * 3u + 1u);
  sequence.restore_words.reserve(static_cast<size_t>(vgpr_count) * 3u + 1u);
  for (uint16_t i = 0; i < vgpr_count; ++i) {
    const uint16_t vgpr = static_cast<uint16_t>(vgpr_base + i);
    const auto offset = planned_manager.offset_for(RegisterRef{RegClass::VGPR, vgpr, 1});
    if (!offset)
      return std::nullopt;
    const auto store = build_address_free_scratch_store_b32(vgpr, *offset, arch);
    const auto load = build_address_free_scratch_load_b32(vgpr, *offset, arch);
    if (!store || !load)
      return std::nullopt;
    sequence.slot_offsets.push_back(*offset);
    sequence.save_words.insert(sequence.save_words.end(), store->begin(), store->end());
    sequence.restore_words.insert(sequence.restore_words.end(), load->begin(), load->end());
  }
  sequence.save_words.push_back(*wait_store);
  sequence.restore_words.push_back(*wait_load);
  sequence.total_private_bytes = planned_manager.total_private_bytes();
  manager = std::move(planned_manager);
  return sequence;
}

SpillDescriptorUpdate update_kernel_descriptor_for_spills(std::span<uint8_t> image,
                                                          uint64_t descriptor_file_offset,
                                                          uint32_t required_private_bytes,
                                                          bool uses_dynamic_stack) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;
  if (uses_dynamic_stack)
    return SpillDescriptorUpdate::DynamicStack;
  if (required_private_bytes == 0 || required_private_bytes > kMaxAddressFreeScratchPrivateBytes) {
    return SpillDescriptorUpdate::InvalidPrivateSize;
  }
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    return SpillDescriptorUpdate::InvalidDescriptor;
  }

  KD descriptor{};
  std::memcpy(&descriptor, image.data() + descriptor_file_offset, sizeof(descriptor));
  const uint32_t grown_private_bytes =
      std::max(descriptor.private_segment_fixed_size, required_private_bytes);
  const bool private_segment_enabled =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) !=
      0;
  if (grown_private_bytes == descriptor.private_segment_fixed_size && private_segment_enabled)
    return SpillDescriptorUpdate::Unchanged;

  descriptor.private_segment_fixed_size = grown_private_bytes;
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  std::memcpy(image.data() + descriptor_file_offset, &descriptor, sizeof(descriptor));
  return SpillDescriptorUpdate::Updated;
}

} // namespace rocjitsu
