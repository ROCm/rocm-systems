// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/semantic/cdna3_scratch.h"

#include "rocjitsu/code/dbt/semantic/cdna3_emitter.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"

#include <bit>
#include <cstdint>

namespace rocjitsu {
namespace {

constexpr uint16_t kCdnaWaitcntAll0 = 0;

[[nodiscard]] constexpr uint32_t pack_sopp(uint16_t op, uint16_t simm16) {
  cdna3::SoppMachineInst dst{};
  dst.encoding = 0x17F;
  dst.op = op & 0x7F;
  dst.simm16 = simm16;
  return std::bit_cast<uint32_t>(dst);
}

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
  auto [w0, w1] =
      Cdna3Emitter::flat_scratch_dword(cdna3::kFlatStoreDwordFlat, vgpr, byte_offset, false);
  words.push_back(w0);
  words.push_back(w1);
}

void Cdna3ScratchEmitter::append_load_dword(std::vector<uint32_t> &words, uint8_t vgpr,
                                            uint32_t byte_offset) {
  auto [w0, w1] =
      Cdna3Emitter::flat_scratch_dword(cdna3::kFlatLoadDwordFlat, vgpr, byte_offset, true);
  words.push_back(w0);
  words.push_back(w1);
}

void Cdna3ScratchEmitter::append_wait(std::vector<uint32_t> &words) {
  words.push_back(pack_sopp(cdna3::kSWaitcntSopp, kCdnaWaitcntAll0));
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
