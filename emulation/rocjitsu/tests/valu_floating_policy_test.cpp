// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace rocjitsu;

namespace {

class InstructionPolicyMachine {
public:
  explicit InstructionPolicyMachine(rj_code_arch_t arch)
      : memory_("policy_memory"), cache_("policy_cache") {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    compute_unit_ =
        amdgpu::ComputeUnitCore::create("instruction_policy", config, &memory_, &cache_);
    decoder_ = Decoder::create(arch);
    wave_ = compute_unit_->dispatch_wf(0, 0, 106, 256);
    wave_->set_exec(1);
    base_ = wave_->vgpr_alloc().base;
  }
  ~InstructionPolicyMachine() { wave_->halt(); }
  template <size_t N>
  uint32_t run(const std::array<uint32_t, N> &words, uint32_t a, uint32_t b = 0, uint32_t c = 0,
               uint32_t mode = 0xf0) {
    wave_->set_mode_raw(mode);
    compute_unit_->write_vgpr(base_, 0, a);
    compute_unit_->write_vgpr(base_ + 1, 0, b);
    compute_unit_->write_vgpr(base_ + 2, 0, c);
    compute_unit_->write_vgpr(base_ + 6, 0, 0xfacebeef);
    std::array<uint32_t, 4> padded{};
    std::copy(words.begin(), words.end(), padded.begin());
    DecodeResult decoded = decoder_->decode(padded.data());
    if (decoded.failed())
      throw std::runtime_error("Instruction encoding rejected by decoder_");
    std::unique_ptr<Instruction> instruction(std::move(decoded).value());
    compute_unit_->execute_instruction(instruction.get(), *wave_);
    return compute_unit_->read_vgpr(base_ + 6, 0);
  }

private:
  amdgpu::GpuMemory memory_;
  amdgpu::L2Cache cache_;
  std::unique_ptr<amdgpu::ComputeUnitCore> compute_unit_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
  uint32_t base_ = 0;
};
void witness(const char *name, rj_code_arch_t arch, uint32_t got, uint32_t expected) {
  EXPECT_EQ(got, expected) << name << " arch=" << unsigned(arch);
}
constexpr std::array<rj_code_arch_t, 10> kAllArchitectures = {
    ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
    ROCJITSU_CODE_ARCH_RDNA4};
std::array<uint32_t, 2> packed_words(rj_code_arch_t arch, uint16_t op, uint8_t clamp = 0) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return cdna1::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_CDNA2:
    return cdna2::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_CDNA3:
    return cdna3::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_CDNA4:
    return cdna4::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_CDNA5:
    return cdna5::build_vop3p(op, {.vdst = 6,
                                   .opsel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .opsel_hi = 3});
  case ROCJITSU_CODE_ARCH_RDNA1:
    return rdna1::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_RDNA2:
    return rdna2::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_RDNA3:
    return rdna3::build_vop3p(op, {.vdst = 6,
                                   .op_sel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return rdna3_5::build_vop3p(op, {.vdst = 6,
                                     .op_sel_hi_2 = 1,
                                     .clamp = clamp,
                                     .src0 = 256,
                                     .src1 = 257,
                                     .src2 = 258,
                                     .op_sel_hi = 3});
  case ROCJITSU_CODE_ARCH_RDNA4:
    return rdna4::build_vop3p(op, {.vdst = 6,
                                   .opsel_hi_2 = 1,
                                   .clamp = clamp,
                                   .src0 = 256,
                                   .src1 = 257,
                                   .src2 = 258,
                                   .opsel_hi = 3});
  default:
    return {};
  }
}
std::array<uint32_t, 1> unary_words(rj_code_arch_t arch, size_t op) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1: {
    const std::array<uint16_t, 5> ops = {cdna1::kVRcpF32Vop1, cdna1::kVSqrtF32Vop1,
                                         cdna1::kVLogF32Vop1, cdna1::kVExpF32Vop1,
                                         cdna1::kVRsqF32Vop1};
    return cdna1::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_CDNA2: {
    const std::array<uint16_t, 5> ops = {cdna2::kVRcpF32Vop1, cdna2::kVSqrtF32Vop1,
                                         cdna2::kVLogF32Vop1, cdna2::kVExpF32Vop1,
                                         cdna2::kVRsqF32Vop1};
    return cdna2::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const std::array<uint16_t, 5> ops = {cdna3::kVRcpF32Vop1, cdna3::kVSqrtF32Vop1,
                                         cdna3::kVLogF32Vop1, cdna3::kVExpF32Vop1,
                                         cdna3::kVRsqF32Vop1};
    return cdna3::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const std::array<uint16_t, 5> ops = {cdna4::kVRcpF32Vop1, cdna4::kVSqrtF32Vop1,
                                         cdna4::kVLogF32Vop1, cdna4::kVExpF32Vop1,
                                         cdna4::kVRsqF32Vop1};
    return cdna4::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_CDNA5: {
    const std::array<uint16_t, 5> ops = {cdna5::kVRcpF32Vop1, cdna5::kVSqrtF32Vop1,
                                         cdna5::kVLogF32Vop1, cdna5::kVExpF32Vop1,
                                         cdna5::kVRsqF32Vop1};
    return cdna5::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_RDNA1: {
    const std::array<uint16_t, 5> ops = {rdna1::kVRcpF32Vop1, rdna1::kVSqrtF32Vop1,
                                         rdna1::kVLogF32Vop1, rdna1::kVExpF32Vop1,
                                         rdna1::kVRsqF32Vop1};
    return rdna1::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_RDNA2: {
    const std::array<uint16_t, 5> ops = {rdna2::kVRcpF32Vop1, rdna2::kVSqrtF32Vop1,
                                         rdna2::kVLogF32Vop1, rdna2::kVExpF32Vop1,
                                         rdna2::kVRsqF32Vop1};
    return rdna2::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_RDNA3: {
    const std::array<uint16_t, 5> ops = {rdna3::kVRcpF32Vop1, rdna3::kVSqrtF32Vop1,
                                         rdna3::kVLogF32Vop1, rdna3::kVExpF32Vop1,
                                         rdna3::kVRsqF32Vop1};
    return rdna3::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_RDNA3_5: {
    const std::array<uint16_t, 5> ops = {rdna3_5::kVRcpF32Vop1, rdna3_5::kVSqrtF32Vop1,
                                         rdna3_5::kVLogF32Vop1, rdna3_5::kVExpF32Vop1,
                                         rdna3_5::kVRsqF32Vop1};
    return rdna3_5::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const std::array<uint16_t, 5> ops = {rdna4::kVRcpF32Vop1, rdna4::kVSqrtF32Vop1,
                                         rdna4::kVLogF32Vop1, rdna4::kVExpF32Vop1,
                                         rdna4::kVRsqF32Vop1};
    return rdna4::build_vop1(ops[op], {.src0 = 256, .vdst = 6});
  }
  default:
    return {};
  }
}
TEST(ValuFloatingPolicy, PackedMinimumPropagatesNaN) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode : {cdna5::kVPkMinNumF16Vop3p, cdna5::kVPkMaxNumF16Vop3p,
                            cdna5::kVPkMinimumF16Vop3p, cdna5::kVPkMaximumF16Vop3p}) {
      const std::array<uint32_t, 2> words = packed_words(arch, opcode);
      const uint32_t result = machine.run(words, 0x7e017e01, 0x3c003c00, 0, 0x2f0);
      if (opcode >= 29)
        witness("packed_minimum_nan", arch, result & 0x7e007e00, 0x7e007e00);
      else
        witness("packed_min_num_nan_control", arch, result, 0x3c003c00);
    }
  }
}
TEST(ValuFloatingPolicy, PackedF32Mode) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                              ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    InstructionPolicyMachine machine(arch);
    // src1 is v1:v2; testing the low result makes the high component immaterial.
    const std::array<uint32_t, 2> words =
        packed_words(arch, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 41 : 50);
    witness("packed_f32_denorm", arch, machine.run(words, 1, 1, 0, 0), 0);
    witness("packed_f32_round_up", arch, machine.run(words, 0x3f800000, 0x33800000, 0, 0xf1),
            0x3f800001);
    const int saved = std::fegetround();
    std::fesetround(FE_UPWARD);
    const uint32_t got = machine.run(words, 0x3f800000, 0x33800000, 0, 0xf0);
    std::fesetround(saved);
    witness("packed_f32_host_round", arch, got, 0x3f800000);
  }
}
TEST(ValuFloatingPolicy, PackedF16Clamp) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode :
         {cdna5::kVPkFmaF16Vop3p, cdna5::kVPkAddF16Vop3p, cdna5::kVPkMulF16Vop3p}) {
      const std::array<uint32_t, 2> words = packed_words(arch, opcode, 1);
      witness("packed_f16_clamp", arch, machine.run(words, 0xc0004000, 0x40004000, 0), 0x00003c00);
    }
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA4)
      continue;
    for (uint16_t opcode : {cdna5::kVPkMinNumF16Vop3p, cdna5::kVPkMaxNumF16Vop3p,
                            cdna5::kVPkMinimumF16Vop3p, cdna5::kVPkMaximumF16Vop3p}) {
      const std::array<uint32_t, 2> words = packed_words(arch, opcode, 1);
      witness("packed_f16_minmax_clamp", arch, machine.run(words, 0xc0004000, 0x40004000, 0),
              opcode == cdna5::kVPkMaxNumF16Vop3p || opcode == cdna5::kVPkMaximumF16Vop3p
                  ? 0x3c003c00
                  : 0x00003c00);
    }
  }
}
TEST(ValuFloatingPolicy, PackedF16Mode) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint32_t, 2> words = packed_words(arch, cdna5::kVPkAddF16Vop3p);
    witness("packed_f16_denorm", arch, machine.run(words, 0x00010001, 0x00010001, 0, 0), 0);
    witness("packed_f16_round_up", arch, machine.run(words, 0x3c003c00, 0x10001000, 0, 0xf4),
            0x3c013c01);
    const std::array<uint32_t, 2> fma = packed_words(arch, cdna5::kVPkFmaF16Vop3p);
    witness("packed_f16_fma_denorm_control", arch, machine.run(fma, 0x00010001, 0x3c003c00, 0, 0),
            0);
    witness("packed_f16_fma_round_control", arch,
            machine.run(fma, 0x3c003c00, 0x3c003c00, 0x10001000, 0xf4), 0x3c013c01);
  }
}
TEST(ValuFloatingPolicy, PackedF16FmaZeroSignUsesGuestRounding) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint32_t, 2> words = packed_words(arch, cdna5::kVPkFmaF16Vop3p);
    for (uint32_t guest_round = 0; guest_round < 4; ++guest_round) {
      for (int host_round : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
        const int saved_round = std::fegetround();
        ASSERT_EQ(std::fesetround(host_round), 0);
        const uint32_t mode = 0xf0u | (guest_round << 2);
        const uint32_t cancellation = machine.run(words, 0x3c003c00, 0x3c003c00, 0xbc00bc00, mode);
        const uint32_t mixed_zeros = machine.run(words, 0x80000000, 0x3c003c00, 0x00008000, mode);
        const uint32_t negative_zeros =
            machine.run(words, 0x80008000, 0x3c003c00, 0x80008000, mode);
        const int restored_round = std::fegetround();
        std::fesetround(saved_round);
        SCOPED_TRACE(arch);
        SCOPED_TRACE(guest_round);
        SCOPED_TRACE(host_round);
        EXPECT_EQ(cancellation, guest_round == 2 ? 0x80008000u : 0u);
        EXPECT_EQ(mixed_zeros, guest_round == 2 ? 0x80008000u : 0u);
        EXPECT_EQ(negative_zeros, 0x80008000u);
        EXPECT_EQ(restored_round, host_round);
      }
    }
  }
}

TEST(ValuFloatingPolicy, F16Dot2HostIndependence) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint32_t, 2> words = rdna4::build_vop3(
        rdna4::kVDot2F16F16Vop3, {.vdst = 6, .src0 = 256, .src1 = 257, .src2 = 258});
    const int saved = std::fegetround();
    std::fesetround(FE_TONEAREST);
    const uint32_t nearest = machine.run(words, 0x10003c00, 0x3c003c00, 0x0001);
    std::fesetround(FE_UPWARD);
    const uint32_t upward = machine.run(words, 0x10003c00, 0x3c003c00, 0x0001);
    std::fesetround(saved);
    witness("dot2_f16_host_independence", arch, upward, nearest);
  }
}
TEST(ValuFloatingPolicy, F32UnaryDenormControls) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    struct Case {
      size_t op;
      uint32_t input;
      uint32_t expected;
    };
    for (const Case &c :
         {Case{0, 0x7f000000, 0}, Case{1, 1, 0}, Case{2, 1, 0xff800000}, Case{3, 0xc3000000, 0}}) {
      const std::array<uint32_t, 1> words = unary_words(arch, c.op);
      witness("unary_f32_denorm", arch, machine.run(words, c.input), c.expected);
    }
  }
}
TEST(ValuFloatingPolicy, F32UnaryQuietsSignalingNaNs) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    for (size_t op = 0; op < 5; ++op) {
      const std::array<uint32_t, 1> words = unary_words(arch, op);
      const uint32_t result = machine.run(words, 0x7f800001, 0, 0, 0x2f0);
      witness("unary_f32_quiet_snan", arch, result & 0x7fc00000, 0x7fc00000);
    }
  }
}
TEST(ValuFloatingPolicy, Dot2InlineReplication) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode : {rdna4::kVDot2F16F16Vop3, rdna4::kVDot2Bf16Bf16Vop3}) {
      const std::array<uint32_t, 2> words =
          rdna4::build_vop3(opcode, {.vdst = 6, .src0 = 242, .src1 = 242, .src2 = 128});
      const uint32_t expected = 0x4000;
      witness("dot2_inline_replication", arch, machine.run(words, 0) & 0xffff, expected);
    }
  }
}
TEST(ValuFloatingPolicy, PackedBf16Clamp) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_CDNA5);
  for (uint16_t opcode : {cdna5::kVPkFmaBf16Vop3p, cdna5::kVPkAddBf16Vop3p, cdna5::kVPkMulBf16Vop3p,
                          cdna5::kVPkMinNumBf16Vop3p, cdna5::kVPkMaxNumBf16Vop3p}) {
    const std::array<uint32_t, 2> words = packed_words(ROCJITSU_CODE_ARCH_CDNA5, opcode, 1);
    witness("packed_bf16_clamp", ROCJITSU_CODE_ARCH_CDNA5,
            machine.run(words, 0xc0004000, 0x40004000, 0),
            opcode == cdna5::kVPkMaxNumBf16Vop3p ? 0x3f803f80 : 0x00003f80);
  }
}
TEST(ValuFloatingPolicy, Bf16UnaryQuietsSignalingNaNs) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_CDNA5);
  for (uint16_t opcode : {cdna5::kVRcpBf16Vop1, cdna5::kVSqrtBf16Vop1, cdna5::kVRsqBf16Vop1,
                          cdna5::kVLogBf16Vop1, cdna5::kVExpBf16Vop1}) {
    const std::array<uint32_t, 1> words = cdna5::build_vop1(opcode, {.src0 = 256, .vdst = 6});
    const uint32_t snan = machine.run(words, 0x7f81, 0, 0, 0x2f0) & 0xffff;
    witness("unary_bf16_quiet_snan", ROCJITSU_CODE_ARCH_CDNA5, snan & 0x7fc0, 0x7fc0);
  }
}

TEST(ValuFloatingPolicy, PackedF32MultiplyAndFmaUseMode) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                              ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    InstructionPolicyMachine machine(arch);
    const uint16_t multiply =
        arch == ROCJITSU_CODE_ARCH_CDNA5 ? cdna5::kVPkMulF32Vop3p : cdna2::kVPkMulF32Vop3p;
    const uint16_t fused =
        arch == ROCJITSU_CODE_ARCH_CDNA5 ? cdna5::kVPkFmaF32Vop3p : cdna2::kVPkFmaF32Vop3p;
    for (uint16_t opcode : {multiply, fused}) {
      const std::array<uint32_t, 2> words = packed_words(arch, opcode);
      // Exactly 1 + 2^-22 + 2^-46 rounds upward to the next float.
      EXPECT_EQ(machine.run(words, 0x3f800001, 0x3f800001, 0, 0xf1), 0x3f800003u);
      EXPECT_EQ(machine.run(words, 1, 0x3f800000, 0, 0), 0u);
      EXPECT_EQ(machine.run(words, 0x00800000, 0x3f000000, 0, 0xd0), 0u);
      EXPECT_EQ(machine.run(words, 0x00800000, 0x3f000000, 0, 0xf0), 0x00400000u);
    }
  }
}

TEST(ValuFloatingPolicy, PackedF16OutputFlushAndSignedZero) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint32_t, 2> multiply = packed_words(arch, cdna5::kVPkMulF16Vop3p);
    EXPECT_EQ(machine.run(multiply, 0x04000400, 0x38003800, 0, 0x70), 0u);
    EXPECT_EQ(machine.run(multiply, 0x04000400, 0x38003800, 0, 0xf0), 0x02000200u);
    EXPECT_EQ(machine.run(multiply, 0x80008000, 0x3c003c00), 0x80008000u);
    const std::array<uint32_t, 2> add = packed_words(arch, cdna5::kVPkAddF16Vop3p);
    EXPECT_EQ(machine.run(add, 0x3c003c00, 0xbc00bc00, 0, 0xf8), 0x80008000u);
    EXPECT_EQ(machine.run(add, 0x3c003c00, 0xbc00bc00, 0, 0xf4), 0u);
  }
}

TEST(ValuFloatingPolicy, Dot2ReplicatesEachInlineSourceButPreservesLiteralHalves) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    InstructionPolicyMachine machine(arch);
    for (uint16_t opcode : {rdna4::kVDot2F16F16Vop3, rdna4::kVDot2Bf16Bf16Vop3}) {
      const bool bf16 = opcode == rdna4::kVDot2Bf16Bf16Vop3;
      const uint32_t one = bf16 ? 0x3f803f80u : 0x3c003c00u;
      const uint32_t expected = bf16 ? 0x3f80u : 0x3c00u;
      for (bool first : {false, true}) {
        const std::array<uint32_t, 2> words =
            rdna4::build_vop3(opcode, {.vdst = 6,
                                       .src0 = static_cast<uint16_t>(first ? 240 : 256),
                                       .src1 = static_cast<uint16_t>(first ? 257 : 240),
                                       .src2 = 128});
        EXPECT_EQ(machine.run(words, one, one) & 0xffffu, expected);
      }
      const std::array<uint32_t, 2> prefix =
          rdna4::build_vop3(opcode, {.vdst = 6, .src0 = 255, .src1 = 257, .src2 = 128});
      const std::array<uint32_t, 3> literal{prefix[0], prefix[1], expected};
      EXPECT_EQ(machine.run(literal, 0, one) & 0xffffu, expected);
    }
    const std::array<uint32_t, 2> integer = rdna4::build_vop3(
        rdna4::kVDot2F16F16Vop3, {.vdst = 6, .src0 = 129, .src1 = 257, .src2 = 128});
    EXPECT_EQ(machine.run(integer, 0, 0x3c003c00) & 0xffffu, 2u);
  }
}

TEST(ValuFloatingPolicy, SqrtFlushesNegativeDenormalsToNegativeZero) {
  for (rj_code_arch_t arch : kAllArchitectures) {
    InstructionPolicyMachine machine(arch);
    EXPECT_EQ(machine.run(unary_words(arch, 1), 0x80000001u), 0x80000000u);
  }
}

TEST(ValuFloatingPolicy, Bf16MixInlineConstantsRespectSelectedHalf) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_CDNA5);
  for (uint16_t opcode :
       {cdna5::kVFmaMixF32Bf16Vop3p, cdna5::kVFmaMixloBf16Vop3p, cdna5::kVFmaMixhiBf16Vop3p}) {
    for (uint32_t source = 0; source < 3; ++source) {
      for (bool high_half : {false, true}) {
        const std::array<uint32_t, 2> words = cdna5::build_vop3p(
            opcode, {.vdst = 6,
                     .opsel = static_cast<uint8_t>(high_half ? 1u << source : 0),
                     .opsel_hi_2 = static_cast<uint8_t>(source == 2),
                     .src0 = 242,
                     .src1 = 242,
                     .src2 = 242,
                     .opsel_hi = static_cast<uint8_t>(source < 2 ? 1u << source : 0)});
        const uint32_t result = machine.run(words, 0);
        if (opcode == cdna5::kVFmaMixF32Bf16Vop3p)
          EXPECT_EQ(result, high_half ? 0x40000000u : 0x3f800000u);
        else {
          const uint32_t half =
              opcode == cdna5::kVFmaMixhiBf16Vop3p ? result >> 16 : result & 0xffffu;
          EXPECT_EQ(half, high_half ? 0x4000u : 0x3f80u);
        }
      }
    }
  }
}

TEST(ValuFloatingPolicy, TanhQuietsSignalingNaNsAcrossFormats) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_CDNA5);
  struct Case {
    uint16_t opcode;
    uint32_t source;
    uint32_t quiet_nan;
  };
  for (const Case &test : {Case{cdna5::kVTanhF32Vop1, 0x7f800001u, 0x7fc00001u},
                           Case{cdna5::kVTanhF16Vop1, 0x7c01u, 0x7e01u},
                           Case{cdna5::kVTanhBf16Vop1, 0x7f81u, 0x7fc1u}}) {
    const std::array<uint32_t, 1> words = cdna5::build_vop1(test.opcode, {.src0 = 256, .vdst = 6});
    const uint32_t mask = test.opcode == cdna5::kVTanhF32Vop1 ? 0xffffffffu : 0xffffu;
    EXPECT_EQ(machine.run(words, test.source, 0, 0, 0x2f0) & mask, test.quiet_nan);
  }
}

TEST(ValuFloatingPolicy, PackedF32ClampAcrossTargets) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                              ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    InstructionPolicyMachine machine(arch);
    const std::array<uint16_t, 3> opcodes =
        arch == ROCJITSU_CODE_ARCH_CDNA5
            ? std::array<uint16_t, 3>{cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p,
                                      cdna5::kVPkFmaF32Vop3p}
            : std::array<uint16_t, 3>{cdna2::kVPkAddF32Vop3p, cdna2::kVPkMulF32Vop3p,
                                      cdna2::kVPkFmaF32Vop3p};
    for (uint16_t opcode : opcodes) {
      const std::array<uint32_t, 2> words = packed_words(arch, opcode, 1);
      EXPECT_EQ(machine.run(words, 0x40000000, 0x40000000), 0x3f800000u);
      EXPECT_EQ(machine.run(words, 0xc0800000, 0x40000000), 0u);
      EXPECT_EQ(machine.run(words, 0x7fc00001, 0x40000000, 0, 0x1f0), 0u);
    }
  }
}

TEST(ValuFloatingPolicy, PackedF16ThreeInputSelectionsUseClampAndDenormMode) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    InstructionPolicyMachine machine(arch);
    const std::vector<uint16_t> opcodes =
        arch == ROCJITSU_CODE_ARCH_CDNA5
            ? std::vector<uint16_t>{cdna5::kVPkMinimum3F16Vop3p, cdna5::kVPkMaximum3F16Vop3p,
                                    cdna5::kVPkMin3NumF16Vop3p, cdna5::kVPkMax3NumF16Vop3p}
            : std::vector<uint16_t>{cdna4::kVPkMinimum3F16Vop3p, cdna4::kVPkMaximum3F16Vop3p};
    for (uint16_t opcode : opcodes) {
      EXPECT_EQ(machine.run(packed_words(arch, opcode, 1), 0x40004000, 0x40004000, 0x40004000),
                0x3c003c00u);
      EXPECT_EQ(machine.run(packed_words(arch, opcode), 0x00010001, 0x00010001, 0x00010001, 0), 0u);
      EXPECT_EQ(machine.run(packed_words(arch, opcode), 0x00010001, 0x00010001, 0x00010001, 0xf0),
                0x00010001u);
    }
  }
}

TEST(ValuFloatingPolicy, PackedBf16MinimumAndMaximumOrderSignedZeros) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_CDNA5);
  for (bool minimum : {false, true}) {
    const uint16_t opcode = minimum ? cdna5::kVPkMinNumBf16Vop3p : cdna5::kVPkMaxNumBf16Vop3p;
    EXPECT_EQ(machine.run(packed_words(ROCJITSU_CODE_ARCH_CDNA5, opcode), 0x80000000, 0x00008000),
              minimum ? 0x80008000u : 0u);
  }
}

TEST(ValuFloatingPolicy, Rdna4F32DotInlineSourcesReplicateBothHalves) {
  InstructionPolicyMachine machine(ROCJITSU_CODE_ARCH_RDNA4);
  for (uint16_t opcode : {rdna4::kVDot2F32F16Vop3p, rdna4::kVDot2F32Bf16Vop3p}) {
    const std::array<uint32_t, 2> words = rdna4::build_vop3p(
        opcode, {.vdst = 6, .src0 = 242, .src1 = 242, .src2 = 128, .opsel_hi = 3});
    EXPECT_EQ(machine.run(words, 0), 0x40000000u);
  }
}
} // namespace
