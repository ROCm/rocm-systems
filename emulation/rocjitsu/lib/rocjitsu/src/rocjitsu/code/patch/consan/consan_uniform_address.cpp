// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_uniform_address.h"

#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <optional>
#include <string_view>

namespace rocjitsu {
void ConSanUniformAddressTracker::observe(const Instruction &inst) {
  // Relative destination writes can clobber a different VGPR from the
  // encoded destination; ordinary def/use facts do not identify that register.
  if (inst.mnemonic().starts_with("v_movrel")) {
    uniform_ = {};
    return;
  }
  bool changes_exec =
      (inst.flags() & (WRITES_EXEC | BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL)) != 0;
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *operand = inst.dst_operand(i);
    const auto ref = operand ? operand->to_register_ref() : std::nullopt;
    changes_exec |= ref && ref->cls == RegClass::EXEC;
  }
  if (changes_exec) {
    uniform_ = {};
    return;
  }
  const auto uniform_source = [&](const Operand *operand) {
    if (!operand)
      return false;
    if (operand->const_value())
      return true;
    const auto ref = operand->to_register_ref();
    return ref && ref->width == 1 &&
           (ref->cls == RegClass::SGPR || (ref->cls == RegClass::VGPR && contains(ref->index)));
  };
  std::array<std::optional<uint16_t>, 2> copies{};
  const auto copy = [&](int slot, const Operand *source) {
    const auto *destination = inst.dst_operand(slot);
    const auto ref = destination ? destination->to_register_ref() : std::nullopt;
    if (ref && ref->cls == RegClass::VGPR && ref->width == 1 && ref->index < kTrackedVgprs &&
        uniform_source(source))
      copies[slot] = ref->index;
  };
  const std::string_view name = inst.mnemonic();
  if (!(inst.flags() & PREDICATED_DEF)) {
    // Exclude DPP/SDWA and other forms that may select or preserve lanes.
    const uint32_t src0 = inst.raw_encoding() ? inst.raw_encoding()[0] & 0x1ffu : amdgpu::SRC_DPP;
    const bool plain_move =
        src0 != amdgpu::SRC_DPP && src0 != amdgpu::SRC_SDWA && !amdgpu::dpp::is_src_dpp8(src0);
    if ((name == "v_mov_b32" || name == "v_mov_b32_e32") && plain_move &&
        inst.num_src_operands() == 1 && inst.num_dst_operands() == 1)
      copy(0, inst.src_operand(0));
    if (inst.num_dst_operands() == 2 && inst.num_src_operands() >= 2) {
      if (name.starts_with("v_dual_mov_b32"))
        copy(0, inst.src_operand(0));
      if (name.ends_with("v_dual_mov_b32"))
        copy(1, inst.src_operand(inst.num_src_operands() - 1));
    }
  }
  const InstDefUse effects(inst);
  // Subtract the definition bitsets directly: enumerating every possible
  // register here adds a full register-file scan for each instruction.
  uniform_ -= effects.defs;
  // Both VOPD inputs describe the state before either destination is written.
  for (auto destination : copies)
    if (destination)
      uniform_.expand({RegClass::VGPR, *destination, 1});
}
} // namespace rocjitsu
