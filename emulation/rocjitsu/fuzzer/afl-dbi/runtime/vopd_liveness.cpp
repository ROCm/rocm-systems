#include "vopd_liveness.h"

#include <optional>

namespace rocjitsu::fuzzer::afl_dbi {
namespace {

std::optional<RegisterRef> vopd_src0_register_ref(uint16_t src) {
  if (src <= 105)
    return RegisterRef{RegClass::SGPR, src, 1};
  if (src >= 256 && src <= 511)
    return RegisterRef{RegClass::VGPR, static_cast<uint16_t>(src - 256), 1};
  return std::nullopt;
}

void add_optional_register(RegisterSet &set, std::optional<RegisterRef> ref) {
  if (ref)
    set.expand(*ref);
}

void add_vgpr(RegisterSet &set, uint16_t reg) {
  set.expand({RegClass::VGPR, reg, 1});
}

bool vopd_component_reads_vdst(uint16_t op) {
  // VOPD uses compact component opcodes, not the standalone VOP2 opcode
  // values. These accumulate forms have a `$vdst = $src2` constraint in LLVM's
  // AMDGPU descriptions, so the encoded destination is also an input.
  switch (op) {
  case 0x0: // v_fmac_f32
  case 0xc: // v_dot2acc_f32_f16
  case 0xd: // v_dot2acc_f32_bf16
    return true;
  default:
    return false;
  }
}

bool vopd_x_component_opcode_is_modeled(uint16_t op) {
  // LLVM's VOPD table defines X opcodes as 0..12: the common Y opcode
  // DOT2ACC_F32_BF16 is intentionally omitted from X. Keep Y-only and future
  // opcodes conservative so opaque-instruction liveness cannot justify fresh
  // registers.
  return op <= 0xc;
}

bool vopd_y_component_opcode_is_modeled(uint16_t op) {
  // Y accepts the common 0..13 opcode set plus Y-only integer forms 16..18.
  return op <= 0xd || (op >= 0x10 && op <= 0x12);
}

bool vopd_component_uses_vsrc1(uint16_t op) {
  // v_mov_b32 is the src0-only VOPD component in the gfx11/gfx12 VOPD64 list.
  // Its vsrc1 field is an encoding placeholder, not an operand.
  return op != 0x8;
}

} // namespace

std::optional<Vopd64LivenessModel> decode_vopd64_liveness_model(uint32_t word0,
                                                                uint32_t word1) {
  if (((word0 >> 26) & 0x3fu) != 0x32u)
    return std::nullopt;

  const uint16_t src0_x = static_cast<uint16_t>(word0 & 0x1ffu);
  const uint16_t vsrc1_x = static_cast<uint16_t>((word0 >> 9) & 0xffu);
  const uint16_t op_x = static_cast<uint16_t>((word0 >> 22) & 0xfu);
  const uint16_t op_y = static_cast<uint16_t>((word0 >> 17) & 0x1fu);
  const uint16_t src0_y = static_cast<uint16_t>(word1 & 0x1ffu);
  const uint16_t vsrc1_y = static_cast<uint16_t>((word1 >> 9) & 0xffu);
  const uint16_t vdst_x = static_cast<uint16_t>((word1 >> 24) & 0xffu);
  const uint16_t vdst_y_hi = static_cast<uint16_t>((word1 >> 17) & 0x7fu);
  const uint16_t vdst_y =
      static_cast<uint16_t>((vdst_y_hi << 1) | ((~vdst_x) & 1u));

  if (!vopd_x_component_opcode_is_modeled(op_x) ||
      !vopd_y_component_opcode_is_modeled(op_y))
    return std::nullopt;

  Vopd64LivenessModel model;
  add_optional_register(model.uses, vopd_src0_register_ref(src0_x));
  add_optional_register(model.uses, vopd_src0_register_ref(src0_y));
  if (vopd_component_uses_vsrc1(op_x))
    add_vgpr(model.uses, vsrc1_x);
  if (vopd_component_uses_vsrc1(op_y))
    add_vgpr(model.uses, vsrc1_y);
  if (vopd_component_reads_vdst(op_x))
    add_vgpr(model.uses, vdst_x);
  if (vopd_component_reads_vdst(op_y))
    add_vgpr(model.uses, vdst_y);
  add_vgpr(model.defs, vdst_x);
  add_vgpr(model.defs, vdst_y);
  return model;
}

uint32_t decode_vopd64_word_count(uint32_t word0, uint32_t word1) {
  if (((word0 >> 26) & 0x3fu) != 0x32u)
    return 0;

  const uint32_t src0_x = word0 & 0x1ffu;
  const uint32_t src0_y = word1 & 0x1ffu;
  return src0_x == 255u || src0_y == 255u ? 3u : 2u;
}

} // namespace rocjitsu::fuzzer::afl_dbi
