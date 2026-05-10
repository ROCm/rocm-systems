// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {

namespace {

constexpr uint16_t kAmdgpuMubufEncodingId = 0x1C0;
constexpr uint32_t kAmdgpuMubufLdsMask = 1u << 16;

// TODO: Replace this class-based approximation with instruction metadata that
// identifies vector defs whose inactive lanes are preserved under EXEC, and pair
// it with program-point EXEC state so full-EXEC writes can be treated as normal
// kills.
[[nodiscard]] bool is_exec_masked_def(RegisterRef ref) {
  return ref.cls == RegClass::VGPR || ref.cls == RegClass::ACC_VGPR;
}

[[nodiscard]] bool has_mubuf_lds_destination(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  return raw != nullptr && inst.size() >= 8 && inst.encoding_id() == kAmdgpuMubufEncodingId &&
         (raw[0] & kAmdgpuMubufLdsMask) != 0;
}

void add_def(InstDefUse &du, RegisterRef ref) {
  du.defs.expand(ref);
  if (is_exec_masked_def(ref))
    du.has_exec_masked_vector_def = true;
}

} // namespace

InstDefUse::InstDefUse(const Instruction &inst) {
  has_predicated_def = inst.flags() & PREDICATED_DEF;

  // MUBUF direct-to-LDS loads encode their memory payload register field in the
  // same slot used by ordinary buffer-load VGPR destinations. When LDS=1 that
  // field selects the memory payload shape/addressing metadata; the loaded data
  // is written to LDS, not to VGPRs. Treating it as a VGPR def lets scratch
  // allocation clobber values that are still live after the async LDS copy.
  if (!has_mubuf_lds_destination(inst)) {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const auto *op = inst.dst_operand(i);
      if (op == nullptr)
        continue;
      if (auto ref = op->to_register_ref())
        add_def(*this, *ref);
    }
  }
  inst.implicit_defs(defs);

  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    if (auto ref = op->to_register_ref())
      uses.expand(*ref);
  }
  inst.implicit_uses(uses);
}

} // namespace rocjitsu
