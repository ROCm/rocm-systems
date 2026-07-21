// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"

#include <cstdint>

namespace rocjitsu {
namespace {

constexpr uint16_t kCdnaWaitcntAll0 = 0;

[[nodiscard]] bool valid_lease(const SemanticScratchLease &lease) {
  if (!lease.spilled)
    return true;
  if (lease.reg_class != RegClass::VGPR || lease.count == 0 ||
      static_cast<uint32_t>(lease.base) + lease.count > 256)
    return false;
  return Cdna3ScratchEmitter::can_address(
      SemanticSpillRange{.byte_offset = lease.spill_offset, .dword_count = lease.count});
}

} // namespace

bool Cdna3ScratchEmitter::can_address(const SemanticSpillRange &range) {
  return range.dword_count != 0 && range.last_dword_offset() <= kMaxDwordOffset;
}

void Cdna3ScratchEmitter::append_store_dword(std::vector<uint32_t> &words, uint8_t vgpr,
                                             uint32_t byte_offset) {
  auto encoded = cdna3::build_flat(
      cdna3::kFlatStoreDwordFlat,
      {.offset = static_cast<uint16_t>(byte_offset), .seg = 1, .data = vgpr, .saddr = 0x7F});
  // ENC_FLAT's XML layout exposes the low 12 offset bits. Scratch addresses
  // additionally use unnamed bit 12, so preserve that bit after packing.
  encoded[0] |= byte_offset & 0x1000u;
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void Cdna3ScratchEmitter::append_load_dword(std::vector<uint32_t> &words, uint8_t vgpr,
                                            uint32_t byte_offset) {
  auto encoded = cdna3::build_flat(
      cdna3::kFlatLoadDwordFlat,
      {.offset = static_cast<uint16_t>(byte_offset), .seg = 1, .saddr = 0x7F, .vdst = vgpr});
  encoded[0] |= byte_offset & 0x1000u;
  words.insert(words.end(), encoded.begin(), encoded.end());
}

void Cdna3ScratchEmitter::append_wait(std::vector<uint32_t> &words) {
  words.push_back(cdna3::build_sopp(cdna3::kSWaitcntSopp, {.simm16 = kCdnaWaitcntAll0})[0]);
}

bool Cdna3ScratchEmitter::append_save(std::vector<uint32_t> &words,
                                      const SemanticScratchLease &lease) {
  if (!valid_lease(lease))
    return false;
  if (!lease.spilled)
    return true;

  for (uint16_t i = 0; i < lease.count; ++i) {
    append_store_dword(words, static_cast<uint8_t>(lease.base + i),
                       lease.spill_offset + static_cast<uint32_t>(i) * sizeof(uint32_t));
  }
  append_wait(words);
  return true;
}

bool Cdna3ScratchEmitter::append_restore(std::vector<uint32_t> &words,
                                         const SemanticScratchLease &lease) {
  if (!valid_lease(lease))
    return false;
  if (!lease.spilled)
    return true;

  for (uint16_t i = 0; i < lease.count; ++i) {
    append_load_dword(words, static_cast<uint8_t>(lease.base + i),
                      lease.spill_offset + static_cast<uint32_t>(i) * sizeof(uint32_t));
  }
  append_wait(words);
  return true;
}

} // namespace rocjitsu
