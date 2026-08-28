// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../tools/waitcheck_fixture.h"
#include "decode_test_util.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text"), data_(std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  const char *data() const override { return data_.get(); }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::unique_ptr<char[]> data_;
  std::size_t size_;
};

class TestCodeObject : public CodeObject {
public:
  explicit TestCodeObject(const std::vector<uint32_t> &words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

template <typename T> void append_inst(std::vector<uint32_t> &words, const T &inst) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0);
  std::array<uint32_t, sizeof(T) / sizeof(uint32_t)> encoded =
      std::bit_cast<std::array<uint32_t, sizeof(T) / sizeof(uint32_t)>>(inst);
  words.reserve(words.size() + encoded.size());
  for (uint32_t word : encoded)
    words.push_back(word);
}

[[nodiscard]] rdna4::SoppMachineInst sopp(uint32_t op, uint32_t simm16 = 0) {
  rdna4::SoppMachineInst inst{};
  inst.encoding = 0x17F;
  inst.op = op;
  inst.simm16 = simm16;
  return inst;
}

[[nodiscard]] rdna4::SoppMachineInst s_endpgm() { return sopp(48, 0); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_idle() { return sopp(10, 0); }

[[nodiscard]] rdna4::SoppMachineInst s_set_vgpr_msb(uint32_t mode) { return sopp(6, mode); }

[[nodiscard]] constexpr uint32_t hwreg(uint32_t id, uint32_t offset, uint32_t size) {
  return (id & 0x3fu) | ((offset & 0x1fu) << 6u) | (((size - 1u) & 0x1fu) << 11u);
}

[[nodiscard]] rdna4::SopkInstLiteralMachineInst s_setreg_imm32_b32(uint32_t reg, uint32_t value) {
  rdna4::SopkInstLiteralMachineInst inst{};
  inst.encoding = 0xb;
  inst.op = 19;
  inst.simm16 = reg;
  inst.simm32 = value;
  return inst;
}

[[nodiscard]] constexpr uint32_t vgpr_msb_mode(uint32_t src0, uint32_t src1, uint32_t src2,
                                               uint32_t dst) {
  return (src0 & 0x3u) | ((src1 & 0x3u) << 2u) | ((src2 & 0x3u) << 4u) | ((dst & 0x3u) << 6u);
}

[[nodiscard]] std::string access_name(WaitcheckAccessKind access) {
  switch (access) {
  case WaitcheckAccessKind::Use:
    return "use";
  case WaitcheckAccessKind::Def:
    return "def";
  case WaitcheckAccessKind::MemoryOrder:
    return "memory-order";
  case WaitcheckAccessKind::ProgramEnd:
    return "program-end";
  case WaitcheckAccessKind::ControlTransfer:
    return "control-transfer";
  }
  return "unknown";
}

[[nodiscard]] std::string diagnostic_summary(const WaitcheckReport &report) {
  std::string result;
  for (const auto &diag : report.diagnostics) {
    result += '\n';
    result += std::string(wait_counter_name(diag.counter));
    result += '/';
    result += access_name(diag.access);
    result += " required=";
    result += std::to_string(diag.required_count);
    result += " at ";
    result += diag.instruction;
    result += " from ";
    result += diag.producer_instruction;
    if (!diag.message.empty()) {
      result += ": ";
      result += diag.message;
    }
  }
  return result;
}

[[nodiscard]] WaitcheckReport analyze_gfx1250_normal(const std::vector<uint32_t> &program) {
  TestCodeObject code_object(program);
  return analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);
}

constexpr std::array<uint32_t, 4> kGfx1250ScaledWmma{
    0xcc3a0200u,
    0x42020d04u,
    0xcc336008u,
    0x04223110u,
};

TEST(WaitcheckTest, ReachableGfx1250DecodePreservesFourWordInstruction) {
  std::vector<uint32_t> program(kGfx1250ScaledWmma.begin(), kGfx1250ScaledWmma.end());
  program.push_back(0xbfb00000u); // s_endpgm.
  TestCodeObject code_object(program);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint64_t, 1> entries{0};
  const std::array<uint64_t, 1> entry_sizes{program.size() * sizeof(uint32_t)};
  auto blocks = BasicBlock::build_reachable(code_object, *decoder, ROCJITSU_CODE_ARCH_CDNA5,
                                            entries, entry_sizes, 32);

  ASSERT_FALSE(blocks.empty());
  const Instruction &instruction = *blocks.front()->instructions().begin();
  ASSERT_EQ(instruction.size(), 4 * sizeof(uint32_t));
  ASSERT_EQ(instruction.num_dst_operands(), 1);
  ASSERT_EQ(instruction.num_src_operands(), 5);
  EXPECT_EQ(instruction.dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::VGPR, 8, 8}));
  EXPECT_EQ(instruction.src_operand(0)->to_register_ref(), (RegisterRef{RegClass::VGPR, 16, 8}));
  EXPECT_EQ(instruction.src_operand(1)->to_register_ref(), (RegisterRef{RegClass::VGPR, 24, 8}));
  EXPECT_EQ(instruction.src_operand(2)->to_register_ref(), (RegisterRef{RegClass::VGPR, 8, 8}));
  ASSERT_NE(instruction.raw_encoding(), nullptr);
  EXPECT_EQ(instruction.raw_encoding()[3], program[3]);
}

[[nodiscard]] rdna4::Vop1MachineInst v_mov_b32(uint32_t vdst, uint32_t src_vgpr) {
  rdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 1;
  inst.vdst = vdst;
  inst.src0 = 256 + src_vgpr;
  return inst;
}

[[nodiscard]] rdna4::Vop1MachineInst v_sqrt_f32(uint32_t vdst, uint32_t src_vgpr) {
  rdna4::Vop1MachineInst inst{};
  inst.encoding = 0x3F;
  inst.op = 51;
  inst.vdst = vdst;
  inst.src0 = 256 + src_vgpr;
  return inst;
}

[[nodiscard]] rdna4::Vop2MachineInst v_add_f32_e32(uint32_t vdst, uint32_t src0, uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::Vop2MachineInst>(0x06000000U);
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::Vop2MachineInst v_cndmask_b32_e32(uint32_t vdst, uint32_t src0,
                                                       uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::Vop2MachineInst>(0x02000000U);
  inst.vdst = vdst;
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::VopcMachineInst v_cmp_gt_u32_e32(uint32_t src0, uint32_t vsrc1) {
  auto inst = std::bit_cast<rdna4::VopcMachineInst>(0x7C980000U);
  inst.src0 = src0;
  inst.vsrc1 = vsrc1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_mov_b32(uint32_t sdst, uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800000U);
  inst.sdst = sdst;
  inst.ssrc0 = ssrc0;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_or_saveexec_b32(uint32_t sdst, uint32_t ssrc0) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE802200U);
  inst.sdst = sdst;
  inst.ssrc0 = ssrc0;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_mov_b64_exec_from_s0() {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800100U);
  inst.sdst = 126; // exec_lo
  inst.ssrc0 = 0;
  return inst;
}

[[nodiscard]] cdna4::Sop2MachineInst s_or_b32_exec_lo(uint32_t ssrc1) {
  cdna4::Sop2MachineInst inst{};
  inst.encoding = 2;
  inst.op = 14;
  inst.sdst = 126;  // EXEC_LO
  inst.ssrc0 = 126; // EXEC_LO
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_barrier_signal_isfirst(uint32_t barrier_id) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804F00U);
  inst.ssrc0 = barrier_id + 128u;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_sendmsg_rtn_b32(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804C00U);
  inst.sdst = sdst;
  inst.ssrc0 = 1;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_sendmsg_rtn_b64(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804D00U);
  inst.sdst = sdst;
  inst.ssrc0 = 1;
  return inst;
}

[[nodiscard]] rdna4::Sop2MachineInst s_cselect_b32(uint32_t sdst) {
  auto inst = std::bit_cast<rdna4::Sop2MachineInst>(0x98000000U);
  inst.sdst = sdst;
  inst.ssrc0 = 129; // literal 1
  inst.ssrc1 = 128; // literal 0
  return inst;
}

[[nodiscard]] rdna4::SopcMachineInst s_cmp_eq_u32(uint32_t ssrc0, uint32_t ssrc1) {
  auto inst = std::bit_cast<rdna4::SopcMachineInst>(0xBF060000U);
  inst.ssrc0 = ssrc0;
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::SopcMachineInst s_cmp_ge_u32(uint32_t ssrc0, uint32_t ssrc1) {
  auto inst = std::bit_cast<rdna4::SopcMachineInst>(0xBF090000U);
  inst.ssrc0 = ssrc0;
  inst.ssrc1 = ssrc1;
  return inst;
}

[[nodiscard]] rdna4::SmemMachineInst s_load_b32(uint32_t sdata, uint32_t sbase) {
  auto inst = std::bit_cast<rdna4::SmemMachineInst>(std::array<uint32_t, 2>{0xF4000000U, 0});
  inst.sdata = sdata;
  inst.sbase = sbase;
  return inst;
}

void append_s_buffer_load_b128_s8_s12_imm48(std::vector<uint32_t> &program) {
  program.push_back(0xF4124206u);
  program.push_back(0xF8000030u);
}

void append_s_buffer_load_b64_s20_s12_imm64(std::vector<uint32_t> &program) {
  program.push_back(0xF4122506u);
  program.push_back(0xF8000040u);
}

void append_s_buffer_load_b128_s16_s8_imm32(std::vector<uint32_t> &program) {
  program.push_back(0xF4024404u);
  program.push_back(0xF8000020u);
}

void append_s_buffer_load_b128_s0_s8_imm0(std::vector<uint32_t> &program) {
  program.push_back(0xF4024004u);
  program.push_back(0xF8000000u);
}

void append_s_load_b64_s2_s16_imm196(std::vector<uint32_t> &program) {
  program.push_back(0xF4002088u);
  program.push_back(0xF80000C4u);
}

void append_v_cndmask_b32_e64_v6_0_1_s(std::vector<uint32_t> &program, uint32_t selector) {
  program.push_back(0xD5010006u);
  program.push_back(0x00010280u | (selector << 18u));
}

void append_v_sub_co_u32_v0_s0_v0_v8(std::vector<uint32_t> &program) {
  program.push_back(0xD7010000u);
  program.push_back(0x02021100u);
}

void append_v_sub_co_ci_u32_v1_null_v1_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xD5217C01u);
  program.push_back(0x00021901u);
}

void append_v_div_scale_f32_v5_s2_s8_s9_s8(std::vector<uint32_t> &program) {
  program.push_back(0xD6FC0205u);
  program.push_back(0x00201208u);
}

void append_v_cmp_gt_u32_sdst_s5_v12(std::vector<uint32_t> &program, uint32_t sdst) {
  program.push_back(0xD44C0000u | sdst);
  program.push_back(0x02021805u);
}

void append_v_cmp_gt_u32_s2_s5_v12(std::vector<uint32_t> &program) {
  append_v_cmp_gt_u32_sdst_s5_v12(program, 2);
}

void append_v_mad_co_s2(std::vector<uint32_t> &program, uint16_t opcode) {
  const auto inst = rdna4::build_vop3_sdst_enc(
      opcode, {.vdst = 0, .sdst = 2, .src0 = 256, .src1 = 257, .src2 = 258});
  program.insert(program.end(), inst.begin(), inst.end());
}

void append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(std::vector<uint32_t> &program) {
  program.push_back(0xCA500501u);
  program.push_back(0x02000080u);
}

void append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xCF409081u);
  program.push_back(0x000A0108u);
  program.push_back(0x08000C0Bu);
}

void append_v_cndmask_b32_v8_v8_v12_s0(std::vector<uint32_t> &program) {
  program.push_back(0xD5010008u);
  program.push_back(0x00021908u);
}

void append_v_cndmask_b16_v1_v1_v14_s2(std::vector<uint32_t> &program) {
  program.push_back(0xD65D0001u);
  program.push_back(0x000A1D01u);
}

void append_v_and_b16_v134_ff_v134(std::vector<uint32_t> &program) {
  program.push_back(0xD7620086u);
  program.push_back(0x02030CFFu);
  program.push_back(0x000000FFu);
}

void append_v_dual_lshlrev_b32_v17_2_v9_dual_mov_b32_v9_s11(std::vector<uint32_t> &program) {
  program.push_back(0xCF448082u);
  program.push_back(0x0009000Bu);
  program.push_back(0x09000011u);
}

void append_buffer_load_b128_v32_v7_s4_offen(std::vector<uint32_t> &program) {
  program.push_back(0xC405C07Cu);
  program.push_back(0x40800820u);
  program.push_back(0x00000007u);
}

[[nodiscard]] rdna4::VglobalMachineInst global_load_b32(uint32_t vdst, uint32_t vaddr = 8,
                                                        uint32_t saddr = 0) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xEE;
  inst.op = 20;
  inst.vdst = vdst;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  return inst;
}

void append_global_load_b64_v8_v7_s0_offset16(std::vector<uint32_t> &program) {
  program.push_back(0xEE054080u);
  program.push_back(0x00000008u);
  program.push_back(0x00001007u);
}

void append_global_load_b128_v2_v7_s0(std::vector<uint32_t> &program) {
  program.push_back(0xEE05C080u);
  program.push_back(0x00000002u);
  program.push_back(0x00000007u);
}

void append_global_load_b128_v6_v137_s2(std::vector<uint32_t> &program) {
  program.push_back(0xEE05C002u);
  program.push_back(0x00000006u);
  program.push_back(0x00000089u);
}

void append_global_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdst) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, global_load_b32(first_vdst + i));
}

[[nodiscard]] rdna4::VscratchMachineInst scratch_load_b32(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VscratchMachineInst>(std::array<uint32_t, 3>{0xED050000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 8;
  inst.saddr = 0;
  return inst;
}

[[nodiscard]] rdna4::VscratchMachineInst scratch_load_b128_off(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VscratchMachineInst>(std::array<uint32_t, 3>{0xED05C000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 0;
  inst.saddr = 64;
  return inst;
}

[[nodiscard]] rdna4::VglobalMachineInst global_store_b32(uint32_t vsrc, uint32_t vaddr = 8,
                                                         uint32_t saddr = 0) {
  auto inst = std::bit_cast<rdna4::VglobalMachineInst>(std::array<uint32_t, 3>{0xEE068000U, 0, 0});
  inst.vsrc = vsrc;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  return inst;
}

[[nodiscard]] rdna4::VglobalMachineInst global_atomic_add_u32(uint32_t vsrc, uint32_t vaddr = 8,
                                                              uint32_t saddr = 124,
                                                              bool returns_value = false) {
  auto inst = std::bit_cast<rdna4::VglobalMachineInst>(std::array<uint32_t, 3>{0xEE0D4000U, 0, 0});
  inst.vdst = vsrc;
  inst.vsrc = vsrc;
  inst.vaddr = vaddr;
  inst.saddr = saddr;
  inst.th = returns_value ? 1 : 0;
  return inst;
}

[[nodiscard]] std::array<uint32_t, 2> s_prefetch_data() { return {0xF404C000U, 0}; }

void append_global_stores(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vsrc) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, global_store_b32(first_vsrc + i));
}

[[nodiscard]] std::array<uint32_t, 3> global_inv() { return {0xEE0AC07CU, 0, 0}; }

[[nodiscard]] std::array<uint32_t, 3> global_wb() { return {0xEE0B007CU, 0, 0}; }

[[nodiscard]] std::array<uint32_t, 3> global_wbinv() { return {0xEE13C07CU, 0, 0}; }

[[nodiscard]] rdna4::VflatMachineInst flat_load_b32(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VflatMachineInst>(std::array<uint32_t, 3>{0xEC050000U, 0, 0});
  inst.vdst = vdst;
  inst.vaddr = 8;
  return inst;
}

[[nodiscard]] rdna4::VflatMachineInst flat_store_b32(uint32_t vsrc) {
  auto inst = std::bit_cast<rdna4::VflatMachineInst>(std::array<uint32_t, 3>{0xEC068000U, 0, 0});
  inst.vsrc = vsrc;
  inst.vaddr = 8;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_load_b32(uint32_t vdst, uint32_t addr) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8D80000U, 0});
  inst.vdst = vdst;
  inst.addr = addr;
  return inst;
}

void append_ds_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t addr) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, ds_load_b32(i, addr));
}

[[nodiscard]] rdna4::VdsMachineInst ds_store_b32(uint32_t addr, uint32_t data0) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8340000U, 0});
  inst.addr = addr;
  inst.data0 = data0;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_swizzle_b32(uint32_t vdst, uint32_t addr) {
  auto inst = std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8D40000U, 0});
  inst.vdst = vdst;
  inst.addr = addr;
  return inst;
}

[[nodiscard]] rdna4::VdsMachineInst ds_nop() {
  return std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8500000U, 0});
}

void append_global_load_async_to_lds_b32(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xEE180000u, 0x00000000u, 0x00000800u});
}

void append_global_store_async_from_lds_b32(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xEE190000u, 0x00000000u, 0x00000800u});
}

void append_ds_atomic_async_barrier_arrive_b64(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD9580000u, 0x00000000u});
}

void append_tensor_load_to_lds(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD0710001u, 0x7C000000u, 0x7C7C0C00u});
}

void append_tensor_store_from_lds(std::vector<uint32_t> &program) {
  program.insert(program.end(), {0xD0714001u, 0x7C000000u, 0x7C7C0C08u});
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_asynccnt(uint32_t count) { return sopp(74, count); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_tensorcnt(uint32_t count) { return sopp(75, count); }

[[nodiscard]] rdna4::VdsdirMachineInst ds_param_load(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VdsdirMachineInst>(0xCE000000U);
  inst.vdst = vdst;
  inst.wait_va_vdst = 15;
  inst.wait_vm_vsrc = 1;
  return inst;
}

[[nodiscard]] rdna4::VdsdirMachineInst
ds_param_load_with_waits(uint32_t vdst, uint32_t wait_va_vdst, uint32_t wait_vm_vsrc) {
  auto inst = ds_param_load(vdst);
  inst.wait_va_vdst = wait_va_vdst;
  inst.wait_vm_vsrc = wait_vm_vsrc;
  return inst;
}

[[nodiscard]] rdna4::VdsdirMachineInst ds_direct_load(uint32_t vdst) {
  auto inst = std::bit_cast<rdna4::VdsdirMachineInst>(0xCE100000U);
  inst.vdst = vdst;
  inst.wait_va_vdst = 15;
  inst.wait_vm_vsrc = 1;
  return inst;
}

void append_ds_direct_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdst) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, ds_direct_load(first_vdst + i));
}

[[nodiscard]] rdna4::VinterpMachineInst v_interp_p10_f32(uint32_t vdst, uint32_t src0,
                                                         uint32_t wait_exp) {
  auto inst = std::bit_cast<rdna4::VinterpMachineInst>(std::array<uint32_t, 2>{0xCD000000U, 0});
  inst.vdst = vdst;
  inst.src0 = 256 + src0;
  inst.src1 = 256;
  inst.src2 = 256;
  inst.wait_exp = wait_exp;
  return inst;
}

[[nodiscard]] rdna4::VsampleMachineInst image_sample(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VsampleMachineInst>(std::array<uint32_t, 3>{0xE406C000U, 0, 0});
  inst.vdata = vdata;
  return inst;
}

[[nodiscard]] rdna4::VsampleMachineInst image_msaa_load(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VsampleMachineInst>(std::array<uint32_t, 3>{0xE4060000U, 0, 0});
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

void append_image_samples(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdata) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, image_sample(first_vdata + i));
}

[[nodiscard]] rdna4::VimageMachineInst image_load(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0000000U, 0, 0});
  inst.dmask = 0xf;
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

[[nodiscard]] rdna4::VimageMachineInst image_atomic_add_uint(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0030000U, 0, 0});
  inst.dmask = 1;
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

[[nodiscard]] rdna4::VimageMachineInst image_bvh_intersect_ray(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0064000U, 0, 0});
  inst.vdata = vdata;
  return inst;
}

void append_image_bvhs(std::vector<uint32_t> &program, uint32_t count, uint32_t first_vdata) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, image_bvh_intersect_ray(first_vdata + i));
}

[[nodiscard]] rdna4::VimageMachineInst image_store_b32(uint32_t vdata) {
  auto inst = std::bit_cast<rdna4::VimageMachineInst>(std::array<uint32_t, 3>{0xD0018000U, 0, 0});
  inst.dmask = 1;
  inst.vdata = vdata;
  inst.rsrc = 0;
  return inst;
}

[[nodiscard]] rdna4::VexportMachineInst export_mrt0_v0() {
  rdna4::VexportMachineInst inst{};
  inst.encoding = 0x3e;
  inst.en = 1;
  inst.tgt = 0;
  inst.done = 1;
  inst.vsrc0 = 0;
  return inst;
}

[[nodiscard]] std::vector<uint32_t> words(std::initializer_list<rdna4::SoppMachineInst> insts) {
  std::vector<uint32_t> result;
  for (const auto &inst : insts)
    append_inst(result, inst);
  return result;
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_sa_sdst_0() {
  return sopp(8, 0xff9e); // depctr_sa_sdst(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_va_vcc_0() {
  return sopp(8, 0xff9d); // depctr_va_vcc(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_va_sdst_0() {
  return sopp(8, 0xf19f); // depctr_va_sdst(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_wait_alu_vm_vsrc_0() {
  return sopp(8, 0xff83); // depctr_vm_vsrc(0)
}

[[nodiscard]] rdna4::SoppMachineInst s_delay_alu(uint32_t simm16) { return sopp(7, simm16); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_xcnt(uint32_t count) { return sopp(69, count); }

[[nodiscard]] rdna4::SoppMachineInst s_wait_xcnt_0() { return s_wait_xcnt(0); }

[[nodiscard]] std::vector<uint32_t> gfx1201_swappc_valu_sgpr_boundary_program(bool include_wait) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4));
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  if (!readlane)
    return {};
  program.insert(program.end(), readlane->begin(), readlane->end());
  if (include_wait)
    append_inst(program, s_wait_alu_va_sdst_0());
  program.push_back(build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  const uint64_t helper_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(helper_offset - kGetpcResultOffset);
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  return program;
}

[[nodiscard]] std::vector<uint32_t> gfx1201_call_valu_sgpr_boundary_program(bool include_wait) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  if (!readlane)
    return {};
  program.insert(program.end(), readlane->begin(), readlane->end());
  if (include_wait)
    append_inst(program, s_wait_alu_va_sdst_0());
  program.push_back(build_s_call_b64(kReturnSreg, /*offset_dwords=*/1, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  return program;
}

void append_gfx950_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F70u);
}

void append_gfx950_s_waitcnt_vmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F71u);
}

void append_gfx950_s_waitcnt_vmcnt_15(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F7Fu);
}

void append_gfx950_s_waitcnt_vmcnt_16(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C4F70u);
}

void append_gfx950_s_waitcnt_lgkmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC07Fu);
}

void append_gfx950_s_waitcnt_lgkmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC17Fu);
}

void append_gfx950_s_waitcnt_lgkmcnt_2(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC27Fu);
}

void append_gfx950_s_waitcnt_lgkmcnt_15(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CCF7Fu);
}

void append_gfx950_s_waitcnt_expcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CCF0Fu);
}

void append_gfx950_buffer_load_dword_v0_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000008u);
}

void append_gfx950_buffer_load_dword(std::vector<uint32_t> &program, uint32_t vdst,
                                     uint32_t vaddr = 8) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000000u | (vdst << 8u) | vaddr);
}

void append_gfx950_buffer_store_dword(std::vector<uint32_t> &program, uint32_t vdata) {
  program.push_back(0xE0700000u);
  program.push_back(vdata << 8u);
}

void append_gfx950_flat_load_dword(std::vector<uint32_t> &program, uint32_t vdst,
                                   uint32_t vaddr = 8) {
  program.push_back(0xDC500000u);
  program.push_back((vdst << 24u) | vaddr);
}

void append_gfx950_global_atomic_add_x2(std::vector<uint32_t> &program, uint32_t vdst,
                                        uint32_t vaddr, uint32_t data, bool return_old) {
  cdna4::FlatMachineInst inst{};
  inst.encoding = 0x37;
  inst.seg = 2;
  inst.op = 98;
  inst.sc0 = return_old;
  inst.addr = vaddr;
  inst.data = data;
  inst.saddr = 0x7f;
  inst.vdst = vdst;
  append_inst(program, inst);
}

void append_gfx950_buffer_atomic_add(std::vector<uint32_t> &program, uint32_t vdata,
                                     bool return_old) {
  cdna4::MubufMachineInst inst{};
  inst.encoding = 0x38;
  inst.op = 66;
  inst.offen = 1;
  inst.sc0 = return_old;
  inst.vaddr = 8;
  inst.vdata = vdata;
  inst.srsrc = 0;
  inst.soffset = 0;
  append_inst(program, inst);
}

void append_gfx950_buffer_load_dword_v1_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000108u);
}

void append_gfx950_buffer_load_dword_v1_off_s0(std::vector<uint32_t> &program) {
  program.push_back(0xE0500000u);
  program.push_back(0x80000100u);
}

void append_gfx950_buffer_load_dword_v2_v0_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000200u);
}

void append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(std::vector<uint32_t> &program) {
  program.push_back(0xE05D1000u);
  program.push_back(0x80100008u);
}

void append_gfx950_s_mov_b32_m0_0(std::vector<uint32_t> &program) {
  program.push_back(0xBEFC0080u);
}

void append_gfx950_v_mov_b32_v12_0(std::vector<uint32_t> &program) {
  program.push_back(0x7E180280u);
}

void append_gfx950_vmem_age_events(std::vector<uint32_t> &program, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
}

void append_gfx950_ds_read_b128_v18_v12_offset64(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0040u);
  program.push_back(0x1200000Cu);
}

void append_gfx950_ds_read_b128_v18_v12_offset1056(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0420u);
  program.push_back(0x1200000Cu);
}

void append_gfx950_ds_read_b128_v82_v13_offset64(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0040u);
  program.push_back(0x5200000Du);
}

void append_gfx950_ds_read_b128_v148_v13_offset1056(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0420u);
  program.push_back(0x9400000Du);
}

void append_gfx950_ds_read_b128_v90_v13_offset1120(std::vector<uint32_t> &program) {
  program.push_back(0xD9FE0460u);
  program.push_back(0x5A00000Du);
}

void append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(std::vector<uint32_t> &program) {
  program.push_back(0xD2680056u);
  program.push_back(0x00032B94u);
}

void append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(std::vector<uint32_t> &program) {
  program.push_back(0xD2680058u);
  program.push_back(0x0002B75Au);
}

void append_gfx950_v_mfma_f32_16x16x32_bf16_v16_acc_cd_0(std::vector<uint32_t> &program) {
  program.push_back(0xD3B50010u);
  program.push_back(0x04425D66u);
}

void append_gfx950_v_mfma_f32_16x16x32_bf16_acc16_acc_cd_1(std::vector<uint32_t> &program) {
  program.push_back(0xD3B58010u);
  program.push_back(0x04425D66u);
}

void append_gfx950_buffer_load_dword_v1_v1_s24_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80060101u);
}

void append_gfx950_s_buffer_load_dwordx8_s0_s28_imm16(std::vector<uint32_t> &program) {
  program.push_back(0xC02E000Eu);
  program.push_back(0x00000010u);
}

void append_gfx950_s_buffer_load_dwordx8_s24_s28_imm16(std::vector<uint32_t> &program) {
  program.push_back(0xC02E060Eu);
  program.push_back(0x00000010u);
}

void append_gfx950_ds_read_b32_v0_v4(std::vector<uint32_t> &program) {
  program.push_back(0xD86C0000u);
  program.push_back(0x00000004u);
}

void append_gfx950_ds_read_b32(std::vector<uint32_t> &program, uint32_t vdst, uint32_t addr = 4,
                               bool gds = false) {
  program.push_back(0xD86C0000u | (static_cast<uint32_t>(gds) << 16u));
  program.push_back((vdst << 24u) | addr);
}

void append_gfx950_s_load_dword_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xC0020100u);
  program.push_back(0x00000000u);
}

void append_gfx950_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx950_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx950_v_mov_b32_v0_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E000302u);
}

void append_gfx950_v_mov_b32_v8_v10(std::vector<uint32_t> &program) {
  program.push_back(0x7E10030Au);
}

void append_gfx942_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8C0F70u);
}

void append_gfx942_s_waitcnt_lgkmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8CC07Fu);
}

void append_gfx942_buffer_load_dword_v0_v8_s0_offen(std::vector<uint32_t> &program) {
  program.push_back(0xE0501000u);
  program.push_back(0x80000008u);
}

void append_gfx942_buffer_load_dword(std::vector<uint32_t> &program, uint8_t vdst, uint8_t vaddr) {
  program.push_back(0xE0501000u);
  program.push_back(0x80010000u | (static_cast<uint32_t>(vdst) << 8u) | vaddr);
}

void append_gfx942_s_load_dword_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xC0020100u);
  program.push_back(0x00000000u);
}

void append_gfx942_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx942_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx942_v_mov_b32_v4_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E080302u);
}

void append_gfx942_global_load_dword_v4_v2(std::vector<uint32_t> &program) {
  program.push_back(0xDC508000u);
  program.push_back(0x047F0002u);
}

void append_gfx942_ds_read_b32_v4_v0(std::vector<uint32_t> &program) {
  program.push_back(0xD86C0000u);
  program.push_back(0x04000000u);
}

void append_gfx942_ds_read_b32(std::vector<uint32_t> &program, uint32_t vdst, uint32_t addr = 0) {
  program.push_back(0xD86C0000u);
  program.push_back((vdst << 24u) | addr);
}

void append_gfx942_v_mov_b32_v5_v4(std::vector<uint32_t> &program) {
  program.push_back(0x7E0A0304u);
}

void append_gfx1100_s_waitcnt_vmcnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBF8903F7u);
}

void append_gfx1100_s_waitcnt_vscnt_0(std::vector<uint32_t> &program) {
  program.push_back(0xBC7C0000u);
}

void append_gfx1100_s_waitcnt_vmcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBCFC0000u);
}

void append_gfx1100_s_waitcnt_vmcnt_sopk_1(std::vector<uint32_t> &program) {
  program.push_back(0xBCFC0001u);
}

void append_gfx1100_s_waitcnt_expcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBD7C0000u);
}

void append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(std::vector<uint32_t> &program) {
  program.push_back(0xBDFC0000u);
}

void append_gfx1100_global_load_b32_v0_v8_s0(std::vector<uint32_t> &program) {
  program.push_back(0xDC520000u);
  program.push_back(0x00000008u);
}

void append_gfx1100_global_load_b32_v2_v8_s0(std::vector<uint32_t> &program) {
  program.push_back(0xDC520000u);
  program.push_back(0x02000008u);
}

void append_gfx1150_global_load_b32(std::vector<uint32_t> &program, uint8_t vdst) {
  program.push_back(0xDC520000u);
  program.push_back((static_cast<uint32_t>(vdst) << 24u) | 0x00000008u);
}

void append_gfx1100_s_load_b32_s4_s0(std::vector<uint32_t> &program) {
  program.push_back(0xF4000100u);
  program.push_back(0xF8000000u);
}

void append_gfx1100_s_mov_b32_s8_s4(std::vector<uint32_t> &program) {
  program.push_back(0xBE880004u);
}

void append_gfx1100_v_add_co_u32_v0_s1_s4_v0(std::vector<uint32_t> &program) {
  program.push_back(0xD7000100u);
  program.push_back(0x02020004u);
}

void append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(std::vector<uint32_t> &program) {
  program.push_back(0xD5207C02u);
  program.push_back(0x00050019u);
}

void append_gfx1100_v_mov_b32_v1_v0(std::vector<uint32_t> &program) {
  program.push_back(0x7E020300u);
}

void append_gfx1100_v_mov_b32_v3_v2(std::vector<uint32_t> &program) {
  program.push_back(0x7E060302u);
}

void append_gfx1150_ds_load_u8_d16_hi_v1_v1(std::vector<uint32_t> &program) {
  program.push_back(0xDA8C0000u);
  program.push_back(0x01000001u);
}

void append_gfx1150_ds_load_u8_d16_hi_v1_v6(std::vector<uint32_t> &program) {
  program.push_back(0xDA8C0000u);
  program.push_back(0x01000006u);
}

void append_gfx1150_ds_load_u8_d16_v1_v5_offset512(std::vector<uint32_t> &program) {
  program.push_back(0xDA880200u);
  program.push_back(0x01000005u);
}

void append_gfx1150_s_waitcnt_lgkmcnt_1(std::vector<uint32_t> &program) {
  program.push_back(0xBF89FC17u);
}

void append_gfx1150_global_store_d16_hi_b8_v2_v1(std::vector<uint32_t> &program) {
  program.push_back(0xDC920000u);
  program.push_back(0x007C0102u);
}

constexpr uint32_t kOverflowQueueSize = 40;
constexpr uint32_t kOverflowRequiredCount = kOverflowQueueSize - 1;
constexpr uint32_t kOverflowBaseVgpr = 32;
constexpr uint32_t kOverflowConsumerVgpr = 96;
constexpr uint32_t kOverflowKmcntBaseSgpr = 0;
constexpr uint32_t kOverflowKmcntSbase = 48;
constexpr uint32_t kOverflowKmcntConsumerSgpr = 80;

void append_s_loads(std::vector<uint32_t> &program, uint32_t count, uint32_t first_sdata,
                    uint32_t sbase) {
  for (uint32_t i = 0; i < count; ++i)
    append_inst(program, s_load_b32(first_sdata + i, sbase));
}

void expect_single_overflow_diagnostic(const WaitcheckReport &report, WaitCounterKind counter,
                                       WaitcheckAccessKind access, RegClass reg_class,
                                       uint32_t reg_index,
                                       uint32_t required_count = kOverflowRequiredCount) {
  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, counter);
  EXPECT_EQ(report.diagnostics[0].access, access);
  EXPECT_EQ(report.diagnostics[0].reg.cls, reg_class);
  EXPECT_EQ(report.diagnostics[0].reg.index, reg_index);
  EXPECT_EQ(report.diagnostics[0].required_count, required_count);
}

TEST(WaitcheckTest, ReportsUnsupportedArchitectures) {
  auto program = words({sopp(0)});

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RV32I);

  EXPECT_FALSE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, MapsGfx942TargetToCdna3) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX942), ROCJITSU_CODE_ARCH_CDNA3);
}

TEST(WaitcheckTest, MapsGfx950TargetToCdna4) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX950), ROCJITSU_CODE_ARCH_CDNA4);
}

TEST(WaitcheckTest, MapsGfx1100TargetToRdna3) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1100), ROCJITSU_CODE_ARCH_RDNA3);
}

TEST(WaitcheckTest, MapsGfx1150AndGfx1151TargetsToRdna35) {
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1150), ROCJITSU_CODE_ARCH_RDNA3_5);
  EXPECT_EQ(waitcheck_arch_for_target(ROCJITSU_CODE_TARGET_GFX1151), ROCJITSU_CODE_ARCH_RDNA3_5);
}

TEST(WaitcheckTest, DecodesRdna4CombinedStoreDsWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(73, 0));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, program.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_wait_storecnt_dscnt");
  EXPECT_TRUE(inst->is_waitcnt());
}

TEST(WaitcheckTest, DecodesCombinedStoreDsWaitSequenceAtExpectedOffset) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0));
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  struct ExpectedInst {
    const char *mnemonic;
    int size;
  };
  const ExpectedInst expected[] = {
      {"global_store_b32", 12}, {"ds_load_b32", 8}, {"s_wait_storecnt_dscnt", 4},
      {"v_mov_b32_e32", 4},     {"s_endpgm", 4},
  };

  size_t word_index = 0;
  for (const auto &tc : expected) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &program[word_index]));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(std::string(inst->mnemonic()), tc.mnemonic);
    EXPECT_EQ(inst->size(), tc.size) << inst->mnemonic();
    word_index += static_cast<size_t>(inst->size() / sizeof(uint32_t));
  }
  EXPECT_EQ(word_index, program.size());
}

TEST(WaitcheckTest, ReportsMissingLoadcntBeforeUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::WaitCounter);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250StreamDecodePreservesFourWordInstructionAtEnd) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16));
  program.insert(program.end(), kGfx1250ScaledWmma.begin(), kGfx1250ScaledWmma.end());

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  const auto diagnostic =
      std::ranges::find_if(report.diagnostics, [](const WaitcheckDiagnostic &candidate) {
        return candidate.instruction.starts_with("v_wmma_scale16_f32_16x16x128_f8f6f4");
      });
  ASSERT_NE(diagnostic, report.diagnostics.end()) << diagnostic_summary(report);
  EXPECT_EQ(diagnostic->counter, WaitCounterKind::Load);
  EXPECT_EQ(diagnostic->access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic->reg, (RegisterRef{RegClass::VGPR, 16, 1}));
}

TEST(WaitcheckTest, AcceptsLoadcntZeroBeforeUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942ReportsMissingVmcntBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942AcceptsSWaitcntVmcntZeroBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_s_waitcnt_vmcnt_0(program);
  append_gfx942_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942OffenBufferAddressDoesNotUseAdjacentVgpr) {
  // Reduced from PyTorch's CK kernel at code object 202, .text+0x43500.
  // An offen-only MUBUF address is one VGPR wide: the final load uses v67,
  // not v[67:68], so the earlier asynchronous definition of v68 is unrelated.
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword(program, 68, 63);
  append_gfx942_buffer_load_dword(program, 69, 64);
  append_gfx942_buffer_load_dword(program, 70, 65);
  append_gfx942_buffer_load_dword(program, 71, 67);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942BufferDescriptorUsesEncodedSgprQuad) {
  // SRSRC=1 names s[4:7], not the unscaled register range s[1:4].
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_buffer_load_dword(program, 0, 8);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 4, 1}));
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_global_load_dword_v4_v2(program);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, Gfx942ReportsMissingLgkmcntBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, Gfx942CounterCapacityRetiresOldestOrderedDsRead) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32(program, 4);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx942_ds_read_b32(program, i + 6);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942CounterCapacityRetiresOldestOrderedBufferLoad) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword(program, 4, 100);
  for (uint32_t i = 0; i < 63; ++i)
    append_gfx942_buffer_load_dword(program, static_cast<uint8_t>(i + 6), 100);
  append_gfx942_v_mov_b32_v5_v4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942ReportsMissingLgkmcntBeforeDsReadOverwrite) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_ds_read_b32_v4_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx942ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx942AcceptsSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx942_s_load_dword_s4_s0(program);
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942AcceptsOldVgprUseWhileDsReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v4_v2(program);   // Establish the committed v4 generation.
  append_gfx942_ds_read_b32_v4_v0(program); // Start an asynchronous replacement.
  append_gfx942_v_mov_b32_v5_v4(program);   // Consume the old v4 generation.
  append_gfx942_v_mov_b32_v4_v2(program);   // Replace that old generation in place.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // Consume the replacement generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942AcceptsLiveInVgprUseWhileReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v5_v4(program);   // Consume the ABI/live-in v4 generation.
  append_gfx942_ds_read_b32_v4_v0(program); // Start an asynchronous replacement.
  append_gfx942_v_mov_b32_v5_v4(program);   // Consume the old v4 generation.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // Consume the replacement generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942AcceptsSynchronousOverlayCreatedAfterDsRead) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program); // Start the future generation.
  append_gfx942_v_mov_b32_v4_v2(program);   // Create the immediately visible generation.
  append_gfx942_v_mov_b32_v5_v4(program);   // Consume that visible generation.
  append_gfx942_s_waitcnt_lgkmcnt_0(program);
  append_gfx942_v_mov_b32_v5_v4(program); // Consume the retired DS generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx942StillReportsUncommittedVgprUseWithOtherReadyValues) {
  std::vector<uint32_t> program;
  append_gfx942_v_mov_b32_v4_v2(program); // An unrelated committed generation.
  append_gfx942_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx942_v_mov_b32_v4_v2(program); // Another unrelated definition.
  append_gfx942_v_mov_b32_v1_v0(program); // Consume the uncommitted load result.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx942StillReportsUncommittedCrossCounterAsyncReplacement) {
  std::vector<uint32_t> program;
  append_gfx942_ds_read_b32_v4_v0(program);
  append_gfx942_global_load_dword_v4_v2(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1100DecodesLegacyAndSopkWaitcnts) {
  std::vector<uint32_t> program;
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  append_gfx1100_s_waitcnt_vscnt_0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_0(program);
  append_gfx1100_s_waitcnt_expcnt_sopk_0(program);
  append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(program);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);

  const std::array<std::string_view, 5> expected = {
      "s_waitcnt", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt"};
  size_t word_index = 0;
  for (const std::string_view mnemonic : expected) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &program[word_index]));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->mnemonic(), mnemonic);
    EXPECT_TRUE(inst->is_waitcnt());
    word_index += static_cast<size_t>(inst->size() / sizeof(uint32_t));
  }
  EXPECT_EQ(word_index, program.size());
}

TEST(WaitcheckTest, Gfx1100ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Rdna35ReportsMissingVmcntBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Rdna35AcceptsPendingD16LoadToOppositeVgprHalf) {
  std::vector<uint32_t> program;
  append_gfx1150_ds_load_u8_d16_hi_v1_v1(program);
  append_gfx1150_ds_load_u8_d16_v1_v5_offset512(program);
  append_gfx1150_s_waitcnt_lgkmcnt_1(program);
  append_gfx1150_global_store_d16_hi_b8_v2_v1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Rdna35ReportsPendingD16LoadToSameVgprHalf) {
  std::vector<uint32_t> program;
  append_gfx1150_ds_load_u8_d16_v1_v5_offset512(program);
  append_gfx1150_ds_load_u8_d16_hi_v1_v6(program);
  append_gfx1150_s_waitcnt_lgkmcnt_1(program);
  append_gfx1150_global_store_d16_hi_b8_v2_v1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1100AcceptsLegacySWaitcntVmcntZeroBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100AcceptsSopkSWaitcntVmcntZeroBeforeGlobalLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100SopkSWaitcntVmcntOneLeavesYoungerLoadPending) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_global_load_b32_v2_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_1(program);
  append_gfx1100_v_mov_b32_v1_v0(program);
  append_gfx1100_v_mov_b32_v3_v2(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1100ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_s_load_b32_s4_s0(program);
  append_gfx1100_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1100AcceptsSopkSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx1100_s_load_b32_s4_s0(program);
  append_gfx1100_s_waitcnt_lgkmcnt_sopk_0(program);
  append_gfx1100_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave32Vop3CarryOutDoesNotDefineAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_u32_v0_s1_s4_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave64Vop3CarryOutDefinesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_u32_v0_s1_s4_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1100Wave32Vop3CarryInDoesNotUseAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1100Wave64Vop3CarryInUsesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_gfx1100_v_add_co_ci_u32_v2_null_s25_0_s1(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1100);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx950ReportsMissingVmcntBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt vmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntVmcntZeroBeforeBufferLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReportsOldestDsReadBeforeLgkmCounterCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 14; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 14u);
}

TEST(WaitcheckTest, Gfx950LgkmcntAllOnesDoesNotRetireBeforeCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 14; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_gfx950_s_waitcnt_lgkmcnt_15(program);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 14u);
}

TEST(WaitcheckTest, Gfx950CounterCapacityRetiresOldestOrderedDsRead) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950LgkmBackpressurePrecedesDependencyCheck) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 14; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  // This is the sixteenth outstanding LDS operation and consumes v0 as its
  // address. Issue stalls until the oldest read into v0 has completed.
  append_gfx950_ds_read_b32(program, 20, 0);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950LgkmBelowCapacityDoesNotRetireBeforeDependencyCheck) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 13; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  // This is only the fifteenth outstanding LDS operation, so it can issue
  // while the read into v0 remains pending.
  append_gfx950_ds_read_b32(program, 20, 0);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950LgkmcntAllOnesIsRedundantAfterCapacityRetirement) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_gfx950_s_waitcnt_lgkmcnt_15(program);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950CounterCapacityRetiresOldestDsReadWithPendingSmem) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950MixedCounterCapacityDoesNotGuaranteeOldestDsRead) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 14; ++i)
    append_gfx950_ds_read_b32(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950CounterCapacityDoesNotOrderScalarMemory) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx950_ds_read_b32(program, i);
  append_gfx950_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950GdsOperationsDoNotAdvanceLdsCounterCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32(program, 0);
  for (uint32_t i = 0; i < 15; ++i)
    append_gfx950_ds_read_b32(program, i + 1, /*addr=*/4, /*gds=*/true);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsOldestBufferLoadBeforeVmCounterCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 62; ++i)
    append_gfx950_buffer_load_dword(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 62u);
}

TEST(WaitcheckTest, Gfx950CounterCapacityRetiresOldestOrderedBufferLoad) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 63; ++i)
    append_gfx950_buffer_load_dword(program, i + 1);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950LegacyVmemStoresAdvanceVmCounterCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 63; ++i)
    append_gfx950_buffer_store_dword(program, /*vdata=*/127);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferLoadRemainsPendingBeforeStoreCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 62; ++i)
    append_gfx950_buffer_store_dword(program, /*vdata=*/127);
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950VmBackpressurePrecedesDependencyCheck) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 62; ++i)
    append_gfx950_buffer_load_dword(program, i + 1);
  // This is the sixty-fourth outstanding VMEM load and consumes v0 as its
  // address. Issue stalls until the oldest load into v0 has completed.
  append_gfx950_buffer_load_dword(program, 70, 0);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950VmBelowCapacityDoesNotRetireBeforeDependencyCheck) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 61; ++i)
    append_gfx950_buffer_load_dword(program, i + 1);
  // This is only the sixty-third outstanding VMEM load, so it can issue while
  // the load into v0 remains pending.
  append_gfx950_buffer_load_dword(program, 70, 0);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950VmcntAllOnesDoesNotRetireBeforeCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 62; ++i)
    append_gfx950_buffer_load_dword(program, i + 1);
  append_gfx950_s_waitcnt_lgkmcnt_15(program); // All wait fields are one.
  append_inst(program, v_mov_b32(100, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 62u);
}

TEST(WaitcheckTest, Gfx950FlatLoadsDoNotAdvanceOrderedVmCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword(program, 0);
  for (uint32_t i = 0; i < 63; ++i)
    append_gfx950_flat_load_dword(program, i + 1, 100);
  append_inst(program, v_mov_b32(101, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950NonReturningFlatAtomicDoesNotDefineVdst) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_global_atomic_add_x2(program, 0, 40, 36, false);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReturningFlatAtomicDefinesVdst) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_global_atomic_add_x2(program, 0, 40, 36, true);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950BufferAtomicAlwaysReadsPayload) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_atomic_add(program, 0, false);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950DoesNotTreatDirectToLdsBufferLoadAsVgprProducer) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950DirectToLdsBufferLoadAgesVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ExactDirectToLdsVmcnt15BoundaryIsClean) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_15(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.analysis_complete) << report.incomplete_reason;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_TRUE(report.passed());
}

TEST(WaitcheckTest, Gfx950ExactDirectToLdsVmcnt16BoundaryReportsHazard) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_TRUE(report.analysis_complete) << report.incomplete_reason;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::PC);
  EXPECT_EQ(report.diagnostics[0].required_count, 15u);
  EXPECT_NE(report.diagnostics[0].producer_instruction.find("buffer_load_dwordx4"),
            std::string::npos);
  EXPECT_NE(report.diagnostics[0].instruction.find("ds_read_b128"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950DirectToLdsRemainsPendingBelowVmCounterCapacity) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 62);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_TRUE(report.analysis_complete) << report.incomplete_reason;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 62u);
}

TEST(WaitcheckTest, Gfx950VmCounterCapacityCompletesDirectToLdsBeforeBarrier) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 63);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.analysis_complete) << report.incomplete_reason;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_TRUE(report.passed());
}

TEST(WaitcheckTest, Gfx950ExactDisjointDirectToLdsRangeIsClean) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset1056(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.analysis_complete) << report.incomplete_reason;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_TRUE(report.passed());
}

TEST(WaitcheckTest, Gfx950UnknownDirectToLdsProducerRangeIsIncomplete) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_FALSE(report.analysis_complete);
  EXPECT_NE(report.incomplete_reason.find("producer range"), std::string::npos);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_FALSE(report.passed());
}

TEST(WaitcheckTest, Gfx950UnknownDirectToLdsConsumerRangeIsIncomplete) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_FALSE(report.analysis_complete);
  EXPECT_NE(report.incomplete_reason.find("consumer range"), std::string::npos);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950WaitAfterBarrierDoesNotRetroactivelyPublishDirectToLds) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  append_gfx950_s_waitcnt_vmcnt_15(program);
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.analysis_complete) << report.incomplete_reason;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].required_count, 15u);
}

TEST(WaitcheckTest, Gfx950SecondBarrierPublishesCompletedDirectToLds) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_vmem_age_events(program, 15);
  append_gfx950_s_waitcnt_vmcnt_16(program);
  program.push_back(0xBF8A0000u); // First barrier is too early.
  append_gfx950_s_waitcnt_vmcnt_15(program);
  program.push_back(0xBF8A0000u); // Second barrier publishes the completed load.
  append_gfx950_v_mov_b32_v12_0(program);
  append_gfx950_ds_read_b128_v18_v12_offset64(program);

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.analysis_complete) << report.incomplete_reason;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_TRUE(report.passed());
}

TEST(WaitcheckTest, Gfx950DirectToLdsAcrossNontrivialCfgIsIncomplete) {
  std::vector<uint32_t> program;
  append_gfx950_s_mov_b32_m0_0(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  program.push_back(0xBF820000u); // s_branch to the next basic block.
  program.push_back(0xBF810000u); // s_endpgm.
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_FALSE(report.analysis_complete);
  EXPECT_NE(report.incomplete_reason.find("nontrivial control flow"), std::string::npos);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

// Counter-parity fixtures retain final-ISA shapes from LLVM's
// waitcnt-overflow.mir, waitcnt-flat.ll, waitcnt-sample-{out-order,waw}.mir,
// waitcnt-bvh.mir, waitcnt-gfx1250.mir, lds-direct-hazards-gfx12.mir,
// memory-legalizer-barriers.ll, and waitcnt-loop-*.mir tests, plus the
// checked-in LLVM-produced Triton fixtures and extracted ROCm PyTorch objects.
// The encoded waits are stable inputs: normal tests neither invoke LLVM nor
// weaken a kernel before analysis.
TEST(WaitcheckTest, Gfx950CounterParityMatchesRequiredVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950CounterParityMatchesPendingReplacementDefinitionsLikeLlvm) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program); // Establish committed v0.
  append_gfx950_v_mov_b32_v1_v0(program); // Establish committed v1.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  program.push_back(0x68040300u); // v_add_u32_e32 v2, v0, v1.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950CounterParityForcesZeroForDsWithPendingScalarMemory) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950RejectsPartialLgkmcntForDsWithPendingScalarMemory) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_1(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[0].producer_instruction, "ds_read_b32 v0, v4");
}

TEST(WaitcheckTest, Gfx950CounterParityReportsStrongerEmittedVmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_TRUE(diag.has_required_dependency);
  EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(diag.wait_section_offset, 16u);
  EXPECT_EQ(diag.consumer_section_offset, 20u);
  EXPECT_EQ(diag.producer_section_offset, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityUsesNoWaitSentinelWhenNoDependencyIsModeled) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 63u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx950CounterParityCombinesConsecutiveWaits) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_NE(diag.wait_instruction.find(';'), std::string::npos);
}

TEST(WaitcheckTest, Gfx950CounterParityPreservesKernelIdentityInCodeObject) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_s_waitcnt_vmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_TRUE(diag.has_kernel);
  EXPECT_EQ(diag.kernel_name, "waitcheck");
  EXPECT_EQ(diag.kernel_entry_offset, 0u);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
}

TEST(WaitcheckTest, Gfx950CounterParityReportsStrongerEmittedLgkmcnt) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Ds);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_TRUE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx950CounterParityDecodesExpcntField) {
  std::vector<uint32_t> program;
  append_gfx950_s_waitcnt_expcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Exp);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 7u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx950CounterParityFollowsUnconditionalBranchToGuardedUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  program.push_back(0xBF820001u); // s_branch over the dead instruction.
  program.push_back(0xBF800000u); // s_nop 0.
  append_gfx950_s_mov_b32_s8_s4(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityContinuesAcrossFallthroughBlockLabel) {
  std::vector<uint32_t> program;
  program.push_back(0xBF840003u); // s_cbranch_scc0 to the guarded use.
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  // This instruction starts a basic block because it is also the branch
  // target, while the wait reaches it through the fallthrough predecessor.
  append_gfx950_s_mov_b32_s8_s4(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityFindsGuardedUseAfterIndependentInstruction) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xD86C0000u); // ds_read_b32 v2, v4.
  program.push_back(0x02000004u);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v8_v10(program);
  append_gfx950_v_mov_b32_v0_v2(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityCollectsAllUsesBeforeNextWait) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xD86C0000u); // ds_read_b32 v2, v4.
  program.push_back(0x02000004u);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(0xBF8A0000u); // s_barrier consumes the younger DS event.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityAttributesLegacyLgkmDrainToBarrier) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950DoesNotRequireBarrierWaitForRegisterOnlyDsLoad) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xBF8A0000u); // s_barrier does not consume the VGPR result.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950CounterParityTracksPartialDirectToLdsWaitBeforeBarrier) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dwordx4_v0_s64_offen_lds(program);
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_s_waitcnt_vmcnt_1(program);
  program.push_back(0xBF8A0000u); // s_barrier.
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityForcesZeroWithPendingFlatEvent) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  program.push_back(0xDC500000u); // flat_load_dword v109, v[12:13].
  program.push_back(0x6D00000Cu);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);
  program.push_back(0xBF810000u); // s_endpgm.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx942CounterParityMatchesLegacyVmcnt) {
  std::vector<uint32_t> program;
  append_gfx942_buffer_load_dword(program, 0, 8);
  append_gfx942_buffer_load_dword(program, 2, 8);
  program.push_back(0xBF8C0F71u); // s_waitcnt vmcnt(1).
  append_gfx942_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityMatchesSopkVmcnt) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_global_load_b32_v2_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_sopk_1(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityLooksPastAdjacentDepctrWait) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  append_gfx1100_s_waitcnt_vmcnt_0(program);
  program.push_back(0xBF880000u); // s_waitcnt_depctr 0.
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityNormalizesWaitIdle) {
  std::vector<uint32_t> program;
  append_gfx1100_global_load_b32_v0_v8_s0(program);
  program.push_back(0xBF8A0000u); // s_wait_idle.
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 4u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 3u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 3u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1100CounterParityDecodesVscnt) {
  std::vector<uint32_t> program;
  append_gfx1100_s_waitcnt_vscnt_0(program);
  append_gfx1100_v_mov_b32_v1_v0(program);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Store);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 63u);
  EXPECT_FALSE(diag.has_required_dependency);
}

TEST(WaitcheckTest, Gfx1100CounterParityCatalogsConservativePairedD16Wait) {
  // Final ISA from PyTorch 2.9.1 code object 139, entry 0, .text+0x4dc.
  // The packaged compiler emits lgkmcnt(0) before the paired high-half load;
  // current LLVM emits lgkmcnt(1) for the later v0 use, matching waitcheck.
  std::vector<uint32_t> program{
      0xDA880000u, 0x0000000Eu, // ds_load_u8_d16 v0, v14.
      0xBF89FC07u,              // s_waitcnt lgkmcnt(0).
      0xDA8C0002u, 0x0000000Eu, // ds_load_u8_d16_hi v0, v14 offset:2.
      0xDA880000u, 0x0100000Fu, // ds_load_u8_d16 v1, v15.
      0xBF89FC07u,              // s_waitcnt lgkmcnt(0).
      0xDA8C0002u, 0x0100000Fu, // ds_load_u8_d16_hi v1, v15 offset:2.
      0xD7620012u, 0x02020081u, // v_and_b16 v18.l, 1, v0.l.
      0xBF810000u,              // s_endpgm.
  };

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 2u);
  const auto diag_it =
      std::ranges::find_if(report.counter_underaccounting_diagnostics,
                           [](const WaitcheckCounterUnderaccountingDiagnostic &diag) {
                             return diag.has_required_dependency;
                           });
  ASSERT_NE(diag_it, report.counter_underaccounting_diagnostics.end());
  const auto &diag = *diag_it;
  EXPECT_EQ(diag.counter, WaitCounterKind::Ds);
  EXPECT_EQ(diag.emitted_count, 0u);
  EXPECT_EQ(diag.required_count, 1u);
  EXPECT_EQ(diag.producer_instruction, "ds_load_u8_d16_hi v0, v14");
  EXPECT_EQ(diag.consumer_instruction, "v_and_b16 v18, 1, v0");
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesSplitLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(2));
  append_inst(program, sopp(64, 1)); // s_wait_loadcnt 1.
  append_inst(program, v_mov_b32(1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityNormalizesCombinedLoadAndDsWait) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, global_load_b32(2));
  append_inst(program, ds_load_b32(4, 12));
  append_inst(program, sopp(72, 0x0101)); // s_wait_loadcnt_dscnt 1, 1.
  append_inst(program, v_mov_b32(1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesPartialDsWaitBeforeResultOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, ds_load_b32(2, 12));
  append_inst(program, ds_load_b32(4, 12));
  append_inst(program, sopp(70, 2)); // s_wait_dscnt 2 retires the oldest result.
  append_inst(program, ds_load_b32(0, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesXcntBeforeImplicitCmpxExecDef) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  // v_cmpx_ne_u16_e32 0, v1.h implicitly defines EXEC, but the final encoding
  // has no explicit destination operand.
  append_inst(program, 0x7D7B0280u);

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeXcntClauseDrain) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(16, 2));
  append_inst(program, global_load_b32(17, 7));
  append_inst(program, global_load_b32(18, 10));
  append_inst(program, global_load_b32(19, 12));
  // LLVM's global-load clause policy emits one all-zero XCNT drain before a
  // staged sequence of partial loadcnt waits and address-register reuses.
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(2, 20));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 0u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 3u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeScratchStoreXcnt) {
  std::vector<uint32_t> program = {
      // Four scratch_store_b128 instructions from the final gfx1250 corpus.
      0xED0740FCu, 0x20000000u, 0x00002000u, 0xED0740FCu, 0x22000000u, 0x00003000u,
      0xED0740FCu, 0x24000000u, 0x00004000u, 0xED0740FCu, 0x26000000u, 0x00005000u,
  };
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(68, 100));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 0u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 2u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg,
            (RegisterRef{RegClass::VGPR, 68, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityCatalogsConservativeScratchLoadReplacement) {
  std::vector<uint32_t> program;
  for (uint32_t vdst = 0; vdst != 64; vdst += 4)
    append_inst(program, scratch_load_b128_off(vdst));
  append_inst(program, sopp(64, 8)); // s_wait_loadcnt 8.
  append_inst(program, v_mov_b32(0, 100));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_TRUE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].emitted_count, 8u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 15u);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesVmVsrc) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_wait_groups, 1u);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesKmcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0.
  append_inst(program, s_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesSamplecnt) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0.
  append_inst(program, v_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesBvhcnt) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(4));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0.
  append_inst(program, v_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0.
  append_inst(program, s_mov_b64_exec_from_s0());

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesAsynccnt) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, ds_load_b32(4, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityMatchesTensorcnt) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, s_wait_tensorcnt(0));
  append_inst(program, ds_load_b32(4, 12));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVmVsrc) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVaVdst) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesEmbeddedVinterpExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_interp_p10_f32(2, 1, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesImpliedVmVsrcToLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses v8 as the vector offset.
  append_inst(program, sopp(64, 0));        // s_wait_loadcnt 0 also waits VM_VSRC.
  append_inst(program, v_mov_b32(8, 10));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityAttributesImpliedXcntToLoadcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // Uses s[0:1] as the scalar address.
  append_inst(program, sopp(64, 0));        // s_wait_loadcnt 0 also waits VMEM X_CNT.
  append_inst(program, s_mov_b32(0, 128));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1250CounterParityAttributesImpliedXcntToKmcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0)); // Uses s[0:1] as the scalar address.
  append_inst(program, sopp(71, 0));      // s_wait_kmcnt 0 also waits SMEM X_CNT.
  append_inst(program, s_mov_b32(0, 128));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesStoreDataWar) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, global_store_b32(10));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(10, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesUncertainLoadOrderAtCfgMerge) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path.
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to the join.
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0.
  append_inst(program, v_mov_b32(2, 0));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesLoopCarriedDsDependency) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(70, 0)); // Loop header: s_wait_dscnt 0.
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-5))); // Back to the loop header.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesDsDrainToSplitBarrierSignal) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0.
  program.push_back(0xBE804EC1u);    // s_barrier_signal -1.
  program.push_back(0xBF810000u);    // s_endpgm.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityAttributesAcquireFenceWaitBeforeGlobalInv) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(1));
  append_inst(program, ds_load_b32(0, 12));
  append_inst(program, ds_store_b32(13, 2));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt 0.
  append_inst(program, global_inv());
  append_inst(program, v_mov_b32(3, 1));
  append_inst(program, ds_load_b32(0, 12));
  program.push_back(0xBF810000u); // s_endpgm.

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201CounterParityHandlesConsecutiveWaitsAcrossCfgLabel) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // Branch target is the second wait below.
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 12));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0.
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 at a CFG label.
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 2u);
  EXPECT_EQ(report.counter_parity_exact, 2u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  EXPECT_EQ(report.counter_parity_indeterminate_groups, 0u);
}

TEST(WaitcheckTest, Gfx1150CounterParityMatchesPartialLoopCarriedDsDependency) {
  std::vector<uint32_t> program;
  append_gfx1150_s_waitcnt_lgkmcnt_1(program); // 0x00 loop header.
  program.push_back(0x7E040300u);              // 0x04: v_mov_b32_e32 v2, v0.
  program.push_back(0xD8D80000u);              // 0x08: ds_load_b32 v0, v4.
  program.push_back(0x00000004u);
  program.push_back(0xD8340000u); // 0x10: ds_store_b32 v4, v2.
  program.push_back(0x00000204u);
  program.push_back(0xBFA0FFF9u); // 0x18: s_branch to loop header.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
  if (!report.counter_underaccounting_diagnostics.empty()) {
    ADD_FAILURE() << report.counter_underaccounting_diagnostics.front().message << " from "
                  << report.counter_underaccounting_diagnostics.front().producer_instruction;
  }
}

// Reduced final ISA from a gfx1150 PyTorch CK kernel.  The compiler-emitted
// vmcnt(2) is one step stronger than the vmcnt(3) required before the VOPD
// reads v[24:25].  Keep the encoded instruction stream here so parity remains
// anchored to the artifact rather than to an assembler or a synthetic opcode.
TEST(WaitcheckTest, Gfx1150CounterParityCapturesConservativeScratchClauseWait) {
  const std::vector<uint32_t> program = {
      0xBF850003u,              // s_clause 0x3.
      0xDC5D0640u, 0x187C0000u, // scratch_load_b128 v[24:27].
      0xDC5D0650u, 0x1C7C0000u, // scratch_load_b128 v[28:31].
      0xDC5D0040u, 0x007C0000u, // scratch_load_b128 v[0:3].
      0xDC5D0050u, 0x047C0000u, // scratch_load_b128 v[4:7].
      0xBF890BF7u,              // s_waitcnt vmcnt(2).
      0xCA100118u, 0x10100119u, // v_dual_mov_b32 v16, v24 :: v17, v25.
      0xBF810000u,              // s_endpgm.
  };

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 0u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  const auto &diag = report.counter_underaccounting_diagnostics.front();
  EXPECT_EQ(diag.counter, WaitCounterKind::Load);
  EXPECT_EQ(diag.emitted_count, 2u);
  EXPECT_EQ(diag.required_count, 3u);
  EXPECT_TRUE(diag.has_required_dependency);
  EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diag.reg, (RegisterRef{RegClass::VGPR, 24, 1}));
  EXPECT_EQ(diag.wait_section_offset, 36u);
  EXPECT_EQ(diag.consumer_section_offset, 40u);
  EXPECT_EQ(diag.producer_section_offset, 4u);
}

TEST(WaitcheckTest, Gfx1150CounterParityCatalogsStrongerCyclicWaitAsUnmodeled) {
  std::vector<uint32_t> program;
  program.push_back(0xBF89FC07u); // 0x00 loop header: s_waitcnt lgkmcnt(0).
  program.push_back(0x7E040300u); // 0x04: v_mov_b32_e32 v2, v0.
  program.push_back(0xD8D80000u); // 0x08: ds_load_b32 v0, v4.
  program.push_back(0x00000004u);
  program.push_back(0xD8340000u); // 0x10: ds_store_b32 v4, v2.
  program.push_back(0x00000204u);
  program.push_back(0xBFA0FFF9u); // 0x18: s_branch to loop header.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1150);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA3_5, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_FALSE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
}

TEST(WaitcheckTest, Gfx1201CounterParityMatchesSaturatedCounterAge) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(64, kOverflowRequiredCount)); // s_wait_loadcnt 39.
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  TestCodeObject code_object(program);
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityFollowsResolvedCallReturn) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                     // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x08.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x0c.
  program.push_back(24);                                        // 0x10 -> helper 0x24.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x14.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x18.
  append_gfx950_v_mov_b32_v1_v0(program);                                      // 0x1c.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x20.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                    // 0x24 helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x28 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx950CounterParityCatalogsStrongerFunctionPrologWaitAsUnmodeled) {
  constexpr uint8_t kTargetSreg = 26;
  constexpr uint8_t kReturnSreg = 30;
  constexpr uint8_t kLiteralOperand = 0xff;
  constexpr uint8_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                     // 0x00.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);                     // 0x08.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x10.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x14.
  program.push_back(20);                                        // 0x18 -> helper 0x28.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x1c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x20.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x24.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                    // 0x28 helper.
  program.push_back(0x7E040300u); // 0x2c: v_mov_b32_e32 v2, v0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_FALSE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 0x3fu);
}

TEST(WaitcheckTest, Gfx950CounterParityCatalogsStrongerFunctionReturnWaitAsUnmodeled) {
  constexpr uint8_t kTargetSreg = 26;
  constexpr uint8_t kReturnSreg = 30;
  constexpr uint8_t kLiteralOperand = 0xff;
  constexpr uint8_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x00.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x04.
  program.push_back(24);                                        // PC 0x04 -> helper 0x1c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x0c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x10.
  program.push_back(0x7E040300u);                              // 0x14: v_mov_b32_e32 v2, v0.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)); // 0x18.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);     // 0x1c helper.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);     // 0x24.
  append_gfx950_s_waitcnt_vmcnt_0(program);                    // 0x2c return wait.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  WaitcheckOptions options;
  options.check_counter_parity = true;
  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 0u);
  EXPECT_EQ(report.counter_underaccounting_observed, 1u);
  EXPECT_EQ(report.counter_unmodeled_wait_observed, 1u);
  ASSERT_EQ(report.counter_underaccounting_diagnostics.size(), 1u);
  EXPECT_FALSE(report.counter_underaccounting_diagnostics[0].has_required_dependency);
  EXPECT_EQ(report.counter_underaccounting_diagnostics[0].required_count, 0x3fu);
}

TEST(WaitcheckTest, Gfx950DoesNotTreatMfmaAccCdAccumulatorAsVgprUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v18_v12_offset64(program);
  append_gfx950_v_mfma_f32_16x16x32_bf16_acc16_acc_cd_1(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950StillTracksMfmaVgprAccumulatorUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v18_v12_offset64(program);
  append_gfx950_v_mfma_f32_16x16x32_bf16_v16_acc_cd_0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 18);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950ReportsKnownInc0001LgkmcntTwoHazardInIsolation) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_ds_read_b128_v148_v13_offset1056(program);
  append_gfx950_ds_read_b128_v90_v13_offset1120(program);
  append_gfx950_s_waitcnt_lgkmcnt_2(program);
  append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(program);
  append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 148u);
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 90u);
  EXPECT_EQ(report.diagnostics[1].required_count, 0u);
}

TEST(WaitcheckTest, Gfx950AcceptsKnownInc0001LgkmcntZeroFixInIsolation) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b128_v82_v13_offset64(program);
  append_gfx950_ds_read_b128_v148_v13_offset1056(program);
  append_gfx950_ds_read_b128_v90_v13_offset1120(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_cvt_pk_bf16_f32_v86_v148_v149(program);
  append_gfx950_v_cvt_pk_bf16_f32_v88_v90_v91(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950DoesNotTrackGfx12VmVsrcSourceOverwrite) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v8_v10(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferOffenUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v2_v0_s0_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferOffsetModeDoesNotReadEncodedVaddr) {
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);
  append_gfx950_buffer_load_dword_v1_off_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950BufferResourceUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_gfx950_s_buffer_load_dwordx8_s0_s28_imm16(program);
  append_gfx950_buffer_load_dword_v1_v1_s24_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforePhysicalBufferResourceUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_buffer_load_dwordx8_s24_s28_imm16(program);
  append_gfx950_buffer_load_dword_v1_v1_s24_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 24u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeVmemSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::WaitCounter);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_vm_vsrc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250GlobalLoadWithSaddrUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_global_load_b64_v8_v7_s0_offset16(program);
  append_global_load_b128_v2_v7_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeVmemSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitXcntZeroBeforeVmemSourceOverwriteOnGfx1250) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_wait_xcnt_0());
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250CodeObjectTracksXcntInNormalMode) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 8, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeExecOverwriteAfterVmemLoad) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_or_b32_exec_lo(18));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeImplicitCmpxExecDef) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  // v_cmpx_ne_u16_e32 0, v1.h implicitly defines EXEC.
  append_inst(program, 0x7D7B0280u);

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeExecOverwriteAfterVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4));
  append_inst(program, s_or_b32_exec_lo(29));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::EXEC, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeSmemBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AcceptsXcntZeroBeforeSmemBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_wait_xcnt(0));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntRetiresOldestVmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntKeepsYoungestVmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(10, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 10, 1}));
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250NonzeroXcntCannotRetireSmemTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_load_b32(5, 4));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250SmemAfterVmemImplicitlyDrainsVmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250VmemAfterSmemImplicitlyDrainsSmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(12, 0));
  append_inst(program, global_load_b32(0, 8, 4));
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LoadcntWaitAlsoRetiresVmemLoadXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250LoadcntWaitDoesNotRetireVmemStoreXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
}

TEST(WaitcheckTest, Gfx1250KmcntZeroAlsoRetiresSmemXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250SetVgprMsbImplicitlyDrainsXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_set_vgpr_msb(0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ModeSetregUpdatesVgprMsbAndDropsImmediateSetVgprMsb) {
  constexpr uint32_t kHwRegMode = 1;
  // MODE stores dst/src0/src1/src2, so 0x04 corresponds to the
  // S_SET_VGPR_MSB layout src0=1, src1=src2=dst=0.
  constexpr uint32_t kModeSrc0V256 = 0x04u << 12u;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegMode, 0, 16), kModeSrc0V256));
  append_inst(program, s_set_vgpr_msb(0)); // Dropped by the gfx1250 hazard.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250NopMakesSetVgprMsbAfterModeSetregEffective) {
  constexpr uint32_t kHwRegMode = 1;
  constexpr uint32_t kModeSrc0V256 = 0x04u << 12u;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegMode, 0, 16), kModeSrc0V256));
  append_inst(program, sopp(0, 0)); // s_nop 0
  append_inst(program, s_set_vgpr_msb(0));
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 8, 1}));
}

TEST(WaitcheckTest, Gfx1250SetregImplicitlyDrainsXcnt) {
  constexpr uint32_t kHwRegStatus = 2;
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegStatus, 0, 1), 0));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ConditionalBranchImplicitlyDrainsXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, sopp(33, 0)); // s_cbranch_scc0 to the fallthrough.
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeVmemStoreDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, v_mov_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeVmemAtomicDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_atomic_add_u32(4));
  append_inst(program, v_mov_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 4, 1}));
}

TEST(WaitcheckTest, Gfx1250LoadcntRetiresReturningVmemAtomicXcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_atomic_add_u32(4, 8, 124, true));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250ReportsXcntBeforeScalarPrefetchBaseOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_prefetch_data());
  append_inst(program, s_mov_b32(0, 128));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1250VmemTranslationOrderingHandlesVmemDestinationOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(4, 8));
  append_inst(program, global_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250NullSaddrGlobalTracksBothVaddrRegisters) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(4, 2, 124));
  append_inst(program, v_mov_b32(3, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 3, 1}));
}

TEST(WaitcheckTest, Gfx1250VmemOrderingHandlesNullSaddrVaddrHighOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(4, 2, 124));
  append_inst(program, global_load_b32(3, 10, 124));
  append_inst(program, v_mov_b32(2, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntAcrossFallthroughJoinRetiresOldestTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // s_cbranch_scc0 over both VMEM operations.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(8, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialXcntAcrossFallthroughJoinKeepsYoungestTranslation) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 6)); // s_cbranch_scc0 over both VMEM operations.
  append_inst(program, global_load_b32(0, 8));
  append_inst(program, global_load_b32(1, 10));
  append_inst(program, s_wait_xcnt(1));
  append_inst(program, v_mov_b32(10, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 10, 1}));
}

TEST(WaitcheckTest, Gfx1250ReportsAsynccntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AcceptsAsynccntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250PartialAsynccntKeepsYoungestOperation) {
  std::vector<uint32_t> program;
  append_global_load_async_to_lds_b32(program);
  append_global_store_async_from_lds_b32(program);
  append_inst(program, s_wait_asynccnt(1));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250AsyncLoadAdvancesLoadcntAge) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(20, 16));
  append_global_load_async_to_lds_b32(program);
  append_inst(program, sopp(64, 1)); // s_wait_loadcnt 1
  append_inst(program, v_mov_b32(4, 20));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250DoesNotInventAsynccntBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_global_store_async_from_lds_b32(program);
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250AsyncBarrierArriveRequiresWaitBeforeBarrierWait) {
  std::vector<uint32_t> program;
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Async);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, Gfx1250ReportsMissingVmVsrcWaitBeforeAsyncBarrierArrive) {
  std::vector<uint32_t> program;
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::AsyncBarrierPreWait);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_NE(report.diagnostics[0].message.find("immediately before"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ReportsMissingVmVsrcWaitAfterAsyncBarrierArrive) {
  std::vector<uint32_t> program;
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_ds_atomic_async_barrier_arrive_b64(program);
  append_inst(program, s_wait_asynccnt(0));
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::AsyncBarrierPostWait);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_NE(report.diagnostics[0].message.find("immediately after"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ReportsTensorcntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Tensor);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, Gfx1250AcceptsTensorcntBeforeLdsUse) {
  std::vector<uint32_t> program;
  append_tensor_load_to_lds(program);
  append_inst(program, s_wait_tensorcnt(0));
  append_inst(program, ds_load_b32(4, 12));

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250DoesNotInventTensorcntBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_tensor_store_from_lds(program);
  append_inst(program, s_endpgm());

  auto report = analyze_gfx1250_normal(program);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201CodeObjectTracksSgprHazardsInNormalMode) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201DisablingExpertModeDoesNotClearSgprHazards) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 0));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201CodeObjectTracksAluDependenciesInExpertMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250CodeObjectTracksVmVsrcInExpertMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 8u);
}

TEST(WaitcheckTest, Gfx1250WaitIdlePreservesExpertSchedulingMode) {
  constexpr uint32_t kHwRegWaveSchedMode = 26;
  std::vector<uint32_t> program;
  append_inst(program, s_setreg_imm32_b32(hwreg(kHwRegWaveSchedMode, 0, 2), 2));
  append_inst(program, s_wait_idle());
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(8, 10));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::X);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 8u);
}

TEST(WaitcheckTest, ScratchOffDoesNotReadVaddrZero) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, scratch_load_b128_off(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, LoadcntZeroAlsoClearsVmemSourceRead) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWrongLoadcntForNewestEvent) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // leaves the newest load pending
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntOneForOlderEvent) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // completes the older load, leaves v1 pending
  append_inst(program, v_mov_b32(2, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, LoadcntOneRetiresOnlyOlderVmemSourceRead) {
  std::vector<uint32_t> program;
  auto older = global_load_b32(0);
  older.vaddr = 8;
  auto newer = global_load_b32(1);
  newer.vaddr = 9;
  append_inst(program, older);
  append_inst(program, newer);
  append_inst(program, sopp(64, 1)); // leaves only the newer load and its source read pending
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsOrderedVmemLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsOrderedImageSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsFlatToVmemLoadOverwriteOnUnresolvedDsCounter) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsGfx942FlatToScratchOverwriteAfterLgkmcntWaitLikeLlvm) {
  // LLVM SIInsertWaitcnts relies on same-class VMEM loads writing VGPR
  // results in order.  The lgkmcnt wait retires the older generic FLAT load's
  // possible LDS completion; no vmcnt wait is needed before the scratch load.
  std::vector<uint32_t> program{
      0xDC500000u, 0x6D00000Cu, // flat_load_dword v109, v[12:13]
      0xBF8CC07Fu,              // s_waitcnt lgkmcnt(0)
      0xDC504000u, 0x6D210000u, // scratch_load_dword v109, off, s33
  };

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA3);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsFlatToDsLoadOverwriteOnBothPossibleFlatCounters) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, ds_load_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, ReportsDsToFlatLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, flat_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 0, 1}));
}

TEST(WaitcheckTest, ReportsVmemToFlatLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, flat_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntBeforeImageLoadSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_load(4));
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntBeforeImageLoadSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_load(4));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, image_sample(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsSamplecntBeforeImageSampleLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, image_load(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntBeforeImageSampleLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(4));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, image_load(4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsLoadcntAndDscntBeforeFlatLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsDscntWhenOnlyLoadcntWaitsFlatLoad) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntWhenOnlyDscntWaitsFlatLoad) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsCombinedLoadcntDscntBeforeFlatLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesLowLoadFromHighStoreData) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 0)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 1, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesLowLoadFromV512StoreData) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 0)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameHighStoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 1)));
  append_inst(program, flat_load_b32(255));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 1, 0, 0)));
  append_inst(program, flat_store_b32(255));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 511u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 511u);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeAcceptsCombinedWaitBeforeSameHighStoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 2)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameV768StoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 3)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 3, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 768u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 768u);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesDifferentHighBanks) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 3)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250WaitIdlePreservesHighVgprMode) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 2)));
  append_inst(program, s_wait_idle());
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 512u);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 512u);
}

TEST(WaitcheckTest, ReportsMissingWaitAluSaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0)); // VALU reads s102, tracking its SGPR pair.
  append_inst(program, s_mov_b32(102, 128));      // SALU writes the tracked SGPR.
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 102u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, SaluConsumerDoesNotRequireWaitAluSaSdst) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_mov_b32(0, 102));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluSaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluSaluCycleBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_delay_alu(9)); // SALU_CYCLE_1 for the next instruction.
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsDelayAluSkippedSaluCycleBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_delay_alu((3u << 4u) | (9u << 7u)));
  append_inst(program, s_mov_b32(0, 128));
  append_inst(program, s_mov_b32(1, 128));
  append_inst(program, s_mov_b32(2, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWaitAluSaSdstAfterOnlyThreeDsNops) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, FourDsNopsCullTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, ds_nop());
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, NonFlatVmemCullClearsTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, global_load_b32(4));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ScalarMemoryCullClearsTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ScratchLoadDoesNotCullTrackedSgprHazardState) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 102, 0));
  append_inst(program, s_mov_b32(102, 128));
  append_inst(program, scratch_load_b32(4));
  append_inst(program, v_add_f32_e32(1, 102, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 102u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0)); // VALU reads s2, tracking its SGPR pair.
  append_v_cmp_gt_u32_s2_s5_v12(program);       // VALU writes s2 as a scalar mask.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, UnrelatedSaluRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0)); // Track the s[2:3] pair.
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  append_inst(program, s_or_saveexec_b32(/*sdst=*/68, /*ssrc0=*/193)); // 193 is inline constant -1.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SpecialSaluRegisterOperandsWaitForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  program.push_back(0xBEFC01EBu); // s_mov_b64 null, src_shared_base
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ImplicitSaluRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  append_inst(program, sopp(/*op=*/33, /*simm16=*/0)); // s_cbranch_scc0
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, HiddenModeRegisterOperandWaitsForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  append_inst(program, s_setreg_imm32_b32(hwreg(/*id=*/1, /*offset=*/0, /*size=*/32), 0));
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SaluWithoutRegisterOperandDoesNotWaitForValuSgprWrites) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, sopp(/*op=*/0, /*simm16=*/0)); // s_nop 0
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201Wave32MadCarryDoesNotDefineEncodedHighSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4)); // Track the s[2:3] pair.
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 3, 5));

    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

    EXPECT_TRUE(report.supported) << report.analysis_error;
    EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  }
}

TEST(WaitcheckTest, Gfx1201Wave32MadCarryStillRequiresWaitForLowSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4));
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 2, 5));

    auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
    EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
  }
}

TEST(WaitcheckTest, Gfx1201Wave64MadCarryDefinesEncodedHighSgpr) {
  for (const uint16_t opcode : {rdna4::kVMadCoU64U32Vop3SdstEnc, rdna4::kVMadCoI64I32Vop3SdstEnc}) {
    SCOPED_TRACE(opcode);
    std::vector<uint32_t> program;
    append_inst(program, v_add_f32_e32(4, 2, 4));
    append_v_mad_co_s2(program, opcode);
    append_inst(program, v_add_f32_e32(5, 3, 5));

    const auto image =
        rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
    AmdGpuCodeObject code_object(image.data(), image.size());
    auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

    ASSERT_TRUE(report.supported) << report.analysis_error;
    ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
    EXPECT_EQ(report.diagnostics[0].reg.index, 3u);
    EXPECT_NE(report.diagnostics[0].message.find("depctr_va_sdst(0)"), std::string::npos);
  }
}

TEST(WaitcheckTest, AcceptsWaitAluVaSdstBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, s_wait_alu_va_sdst_0());
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluValuDepBeforeValuReadsTrackedSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);
  append_inst(program, s_delay_alu(1)); // VALU_DEP_1 for the next instruction.
  append_inst(program, v_add_f32_e32(1, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVaVccBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, v_cmp_gt_u32_e32(5, 12));      // VALU writes VCC.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VCC);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250VopdCndmaskDoesNotRequireWaitAluVaVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, v_cmp_gt_u32_e32(5, 12));      // VALU writes VCC.
  append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitAluVaVccBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_wait_alu_va_vcc_0());
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsDelayAluValuDepBeforeValuReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_delay_alu(1)); // VALU_DEP_1 for the next instruction.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluSaSdstBeforeValuReadsSaluWrittenVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, s_cselect_b32(106));           // SALU writes vcc_lo.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VCC);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, AcceptsWaitAluSaSdstBeforeValuReadsSaluWrittenVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, s_cselect_b32(106));
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, SaluVccReadClearsValuVccHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1));
  append_inst(program, v_cmp_gt_u32_e32(5, 12));
  append_inst(program, s_mov_b32(0, 106)); // SALU reads vcc_lo, clearing the VALU VCC hazard.
  append_inst(program, v_cndmask_b32_e32(2, 128, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingDscntBeforeDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsDscntZeroBeforeDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DscntPartialWaitCountsYoungerDsSwizzles) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(29, 4));
  append_inst(program, ds_swizzle_b32(28, 4));
  append_inst(program, ds_swizzle_b32(5, 19));
  append_inst(program, sopp(70, 2)); // s_wait_dscnt 2 retires the oldest result.
  append_inst(program, v_mov_b32(29, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsDscntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, v_mov_b32(40, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Ds, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, 0);
}

TEST(WaitcheckTest, AcceptsDscntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, sopp(70, kOverflowRequiredCount)); // s_wait_dscnt 39
  append_inst(program, v_mov_b32(40, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, DscntMaxCountLeavesSecondOldestOverflowSizedQueuePending) {
  std::vector<uint32_t> program;
  append_ds_loads(program, 40, 99);
  append_inst(program, sopp(70, kOverflowRequiredCount)); // s_wait_dscnt 39
  append_inst(program, v_mov_b32(40, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, kOverflowRequiredCount - 1);
}

TEST(WaitcheckTest, ReportsLoadcntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Load, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsLoadcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_global_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(64, kOverflowRequiredCount)); // s_wait_loadcnt 39
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

// Minimized from a gfx1150 CK kernel in the PyTorch nightly corpus.  The
// producer has 61 younger VMEM events at vmcnt(62), so that wait may leave it
// pending.  Eight subsequent loads saturate its age at 62 before the use.
TEST(WaitcheckTest, Gfx1150ReportsOldLoadAfterMaximumPartialWaitAndNewLoads) {
  std::vector<uint32_t> program;
  append_gfx1150_global_load_b32(program, 32);
  for (uint32_t i = 0; i < 61; ++i)
    append_gfx1150_global_load_b32(program, static_cast<uint8_t>(64 + i));
  program.push_back(0xBF89FBF7u); // s_waitcnt vmcnt(62)
  for (uint32_t i = 0; i < 8; ++i)
    append_gfx1150_global_load_b32(program, static_cast<uint8_t>(160 + i));
  append_inst(program, v_mov_b32(200, 32));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA3_5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 32u);
  EXPECT_EQ(report.diagnostics[0].required_count, 62u);
}

TEST(WaitcheckTest, AcceptsStorecntOverflowSizedQueueBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_global_stores(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, s_endpgm());
  WaitcheckOptions options;
  options.max_diagnostics = 1;

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics_observed, 0u);
  EXPECT_FALSE(report.diagnostics_truncated);
}

TEST(WaitcheckTest, ReportsKmcntZeroForOverflowSizedSmemQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, kOverflowKmcntBaseSgpr);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_wait_kmcnt <= 0"), std::string::npos);
}

TEST(WaitcheckTest, NonzeroKmcntDoesNotRetireOldestSmemResult) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, sopp(71, kOverflowRequiredCount)); // s_wait_kmcnt 39
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, KmcntZeroRetiresOverflowSizedSmemQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsSamplecntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_samples(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Sample, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsSamplecntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_samples(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(66, kOverflowRequiredCount)); // s_wait_samplecnt 39
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsBvhcntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_bvhs(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Bvh, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr, 6);
}

TEST(WaitcheckTest, AcceptsBvhcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_bvhs(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(67, 6)); // Largest non-sentinel 3-bit wait value.
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsExpcntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_direct_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Exp, WaitcheckAccessKind::Use,
                                    RegClass::VGPR, kOverflowBaseVgpr, 6);
}

TEST(WaitcheckTest, AcceptsExpcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_direct_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(68, 6)); // Largest non-sentinel 3-bit wait value.
  append_inst(program, v_mov_b32(kOverflowConsumerVgpr, kOverflowBaseVgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeDsSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, DscntZeroAlsoClearsDsSourceRead) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0
  append_inst(program, v_mov_b32(4, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, VopdMovDoesNotReadUnusedVsrc1Encoding) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(0, 10));
  append_v_dual_lshlrev_b32_v17_2_v9_dual_mov_b32_v9_s11(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, BufferOffenUsesSingleVaddrRegister) {
  std::vector<uint32_t> program;
  append_inst(program, ds_load_b32(8, 10));
  append_buffer_load_b128_v32_v7_s4_offen(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsStoreDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeDsStoreAddressOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(4, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeDsStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 1));
  append_inst(program, v_mov_b32(4, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingKmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsKmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1201CndmaskDoesNotReadAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_s_load_b64_s2_s16_imm196(program);
  append_v_cndmask_b32_e64_v6_0_1_s(program, 1);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201CndmaskReportsPendingMaskSgpr) {
  std::vector<uint32_t> program;
  append_s_load_b64_s2_s16_imm196(program);
  append_v_cndmask_b32_e64_v6_0_1_s(program, 2);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201Vop3CarryOutDoesNotDefineAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_sub_co_u32_v0_s0_v0_v8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingVop3CarryOutSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(0, 16));
  append_v_sub_co_u32_v0_s0_v0_v8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx1201DivScaleDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingDivScaleMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201Wave32CodeObjectDivScaleDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  const auto kernels = waitcheck_kernels(code_object);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].wavefront_size, 32u);
  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201Wave64DivScaleDefinesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_div_scale_f32_v5_s2_s8_s9_s8(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 3, 1}));
}

TEST(WaitcheckTest, Gfx1250Wave32VopCompareDoesNotDefineAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_cmp_gt_u32_sdst_s5_v12(program, 1);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave64VopCompareDefinesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 16));
  append_v_cmp_gt_u32_sdst_s5_v12(program, 1);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1250Wave32VopdCndmaskDoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave64VopdCndmaskUsesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_dual_add_nc_u32_v11_1_v10_cndmask_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 1, 1}));
}

TEST(WaitcheckTest, Gfx1250Wave32Vop3CndmaskDoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_cndmask_b32_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave64Vop3CndmaskUsesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_cndmask_b32_v8_v8_v12_s0(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 1, 1}));
}

TEST(WaitcheckTest, Gfx1250Wave32Vop3CndmaskB16DoesNotUseAdjacentMaskSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_cndmask_b16_v1_v1_v14_s2(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250Wave64Vop3CndmaskB16UsesBothMaskSgprs) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 16));
  append_v_cndmask_b16_v1_v1_v14_s2(program);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 3, 1}));
}

TEST(WaitcheckTest, Gfx1250Vop3True16DestinationUsesFullEightBitVgprIndex) {
  std::vector<uint32_t> program;
  append_global_load_b128_v6_v137_s2(program);
  append_v_and_b16_v134_ff_v134(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201Vop3CarryInDoesNotReadAdjacentSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(1, 16));
  append_v_sub_co_ci_u32_v1_null_v1_v12_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsPendingVop3CarryInSgpr) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(0, 16));
  append_v_sub_co_ci_u32_v1_null_v1_v12_s0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 0, 1}));
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntLgkmcntZeroBeforeScalarLoadUse) {
  std::vector<uint32_t> program;
  append_gfx950_s_load_dword_s4_s0(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_s_mov_b32_s8_s4(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ReportsMissingLgkmcntBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("s_waitcnt lgkmcnt(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx950AcceptsSWaitcntLgkmcntZeroBeforeDsReadUse) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program);
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950AcceptsOldVgprUseWhileDsReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program);   // Establish the committed v0 generation.
  append_gfx950_ds_read_b32_v0_v4(program); // Start an asynchronous replacement.
  append_gfx950_v_mov_b32_v1_v0(program);   // Consume the old v0 generation.
  append_gfx950_v_mov_b32_v0_v2(program);   // Replace that old generation in place.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // Consume the replacement generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950AcceptsLiveInVgprUseWhileReplacementIsPending) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v1_v0(program);   // Consume the ABI/live-in v0 generation.
  append_gfx950_ds_read_b32_v0_v4(program); // Start an asynchronous replacement.
  append_gfx950_v_mov_b32_v1_v0(program);   // Consume the old v0 generation.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // Consume the replacement generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950AcceptsSynchronousOverlayCreatedAfterDsRead) {
  std::vector<uint32_t> program;
  append_gfx950_ds_read_b32_v0_v4(program); // Start the future generation.
  append_gfx950_v_mov_b32_v0_v2(program);   // Create the immediately visible generation.
  append_gfx950_v_mov_b32_v1_v0(program);   // Consume that visible generation.
  append_gfx950_s_waitcnt_lgkmcnt_0(program);
  append_gfx950_v_mov_b32_v1_v0(program); // Consume the retired DS generation.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950StillReportsUncommittedVgprUseWithOtherReadyValues) {
  std::vector<uint32_t> program;
  append_gfx950_v_mov_b32_v0_v2(program); // An unrelated committed generation.
  append_gfx950_buffer_load_dword_v1_v8_s0_offen(program);
  append_gfx950_v_mov_b32_v8_v10(program); // Another unrelated definition.
  program.push_back(0x7E040301u);          // v_mov_b32_e32 v2, v1.

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::VGPR, 1, 1}));
}

TEST(WaitcheckTest, ReportsKmcntBeforeScalarLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, s_load_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Gfx1250SBufferLoadSbaseUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s8_s12_imm48(program);
  append_s_buffer_load_b64_s20_s12_imm64(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201SBufferLoadSbaseUsesPhysicalDescriptorRegister) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s16_s8_imm32(program);
  append_s_buffer_load_b128_s0_s8_imm0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsKmcntBeforeSBufferDescriptorUse) {
  std::vector<uint32_t> program;
  append_s_buffer_load_b128_s8_s12_imm48(program);
  append_s_buffer_load_b128_s0_s8_imm0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg, (RegisterRef{RegClass::SGPR, 8, 1}));
}

TEST(WaitcheckTest, AcceptsKmcntBeforeScalarLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(4, 0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_load_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsKmcntBeforeSendmsgRtnB32Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsKmcntBeforeSendmsgRtnB32Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1201CounterParityForcesZeroForMixedKmcntEventKinds) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0.
  append_inst(program, s_mov_b32(8, 4));

  WaitcheckOptions options;
  options.check_counter_parity = true;
  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.passed()) << diagnostic_summary(report);
  EXPECT_EQ(report.counter_parity_fields_checked, 1u);
  EXPECT_EQ(report.counter_parity_exact, 1u);
  EXPECT_EQ(report.counter_underaccounting_observed, 0u);
}

TEST(WaitcheckTest, Gfx1201RejectsPartialWaitForMixedKmcntEventKinds) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b32(4));
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 1)); // s_wait_kmcnt 1 cannot advance a mixed counter.
  append_inst(program, s_mov_b32(8, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_EQ(report.diagnostics[0].producer_instruction, "s_sendmsg_rtn_b32 s4, 1");
}

TEST(WaitcheckTest, ReportsKmcntBeforeSendmsgRtnB64Use) {
  std::vector<uint32_t> program;
  append_inst(program, s_sendmsg_rtn_b64(4));
  append_inst(program, s_mov_b32(8, 5));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 5u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, Vop3CompareSdstDoesNotOverlapAdjacentScalarLoadResult) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(3, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsKmcntWhenVop3CompareOverwritesLoadedScalarMask) {
  std::vector<uint32_t> program;
  append_inst(program, s_load_b32(2, 0));
  append_v_cmp_gt_u32_s2_s5_v12(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
}

TEST(WaitcheckTest, ReportsKmcntBeforeSccUseAfterBarrierSignalIsfirst) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsKmcntBeforeSccUseAfterBarrierSignalIsfirst) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(71, 0)); // s_wait_kmcnt 0
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, MatchingBarrierWaitClearsSccWrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DifferentBarrierWaitDoesNotClearSccWrite) {
  std::vector<uint32_t> program;
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 1)); // s_barrier_wait 1
  append_inst(program, s_cselect_b32(1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
}

TEST(WaitcheckTest, ReportsSccWriteFromOtherBlockBeforeSccUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 128)); // s_cmp_eq_u32 s0, 0
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over signal
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, s_cselect_b32(1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SCC);
}

TEST(WaitcheckTest, BarrierWaitClearsSccWriteFromOtherBlock) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 128)); // s_cmp_eq_u32 s0, 0
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over signal
  append_inst(program, s_barrier_signal_isfirst(0));
  append_inst(program, sopp(20, 0)); // s_barrier_wait 0
  append_inst(program, s_cselect_b32(1));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingSamplecntBeforeImageSampleUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntZeroBeforeImageSampleUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_sample(0));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingSamplecntBeforeImageMsaaLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_msaa_load(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Sample);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsSamplecntZeroBeforeImageMsaaLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_msaa_load(0));
  append_inst(program, sopp(66, 0)); // s_wait_samplecnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingLoadcntBeforeImageAtomicResultUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntZeroBeforeImageAtomicResultUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsImageAtomicWaitsBeforeImageMsaaOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, image_msaa_load(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 3u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[2].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[2].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[2].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[2].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsImageAtomicWaitsBeforeImageMsaaOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_atomic_add_uint(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, image_msaa_load(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingBvhcntBeforeImageBvhUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsBvhcntZeroBeforeImageBvhUse) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0
  append_inst(program, v_mov_b32(4, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsBvhcntBeforeVmemOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsBvhcntBeforeVmemOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, sopp(67, 0)); // s_wait_bvhcnt 0
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsBvhcntBeforeImageSampleOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_bvh_intersect_ray(0));
  append_inst(program, image_sample(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Bvh);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsLoadcntBeforeBvhOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, image_bvh_intersect_ray(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsLoadcntBeforeBvhOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, image_bvh_intersect_ray(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, StoreSourceReadDoesNotNeedExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalWbDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_wb());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalWbinvDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_wbinv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalInvDoesNotInventMemoryDependency) {
  std::vector<uint32_t> program;
  append_inst(program, global_inv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, GlobalInvContributesToLoadcntThreshold) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_inv());
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
}

TEST(WaitcheckTest, AcceptsAdjustedLoadcntThresholdWithGlobalInv) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_inv());
  append_inst(program, sopp(64, 1)); // s_wait_loadcnt 1
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx1250LoadcntAgeSaturatesAcrossCounterOnlyEvents) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  for (size_t i = 0; i < 80; ++i)
    append_inst(program, global_inv());
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].required_count, 62u);
}

TEST(WaitcheckTest, Gfx1250MaximumLoadcntRetiresSaturatedEvent) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  for (size_t i = 0; i < 80; ++i)
    append_inst(program, global_inv());
  append_inst(program, sopp(64, 62));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsEndpgmAfterPendingVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, EndpgmClearsPendingStateBeforeNextEntry) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_endpgm());
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsStorecntBeforeEndpgmAfterVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, sopp(65, 0)); // s_wait_storecnt 0
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  EXPECT_EQ(report.memory_events_tracked, 2u);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsWaitAluVmVsrcBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsParamLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_param_load(1));
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, ReportsMissingWaitVmVsrcBeforeDsParamLoadSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
}

TEST(WaitcheckTest, AcceptsDsParamLoadWaitVmVsrcBeforeSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v8 as the vector offset.
  append_inst(program, ds_param_load_with_waits(8, 15, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitVaVdstBeforeDsParamLoadAfterImmediateValuRead) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].kind, WaitcheckDiagnosticKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstZeroBeforeDsParamLoadAfterImmediateValuRead) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingWaitVaVdstWithOneInterveningValu) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_add_f32_e32(2, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].required_count, 1u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstOneWithOneInterveningValu) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_add_f32_e32(2, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsWaitVaVdstZeroForTransMixedDsParamLoadHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_sqrt_f32(2, 2));
  append_inst(program, v_add_f32_e32(3, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VaVdst);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstZeroForTransMixedDsParamLoadHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, v_sqrt_f32(2, 2));
  append_inst(program, v_add_f32_e32(3, 258, 2));
  append_inst(program, ds_param_load_with_waits(1, 0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, AcceptsWaitVaVdstFifteenAfterVmemExpiry) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(0, 257, 1)); // VALU reads v1.
  append_inst(program, global_load_b32(4));
  append_inst(program, ds_param_load_with_waits(1, 15, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsDirectLoadOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_mov_b32(1, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, AcceptsExpcntBeforeDsDirectLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, v_mov_b32(2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, AcceptsVinterpWaitExpBeforeDsDirectLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, v_interp_p10_f32(2, 1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsVinterpWaitExpOneLeavesNewestDsDirectLoadPending) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, ds_direct_load(2));
  append_inst(program, v_interp_p10_f32(3, 2, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 2u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsVinterpWaitExpOneForOlderDsDirectLoad) {
  std::vector<uint32_t> program;
  append_inst(program, ds_direct_load(1));
  append_inst(program, ds_direct_load(2));
  append_inst(program, v_interp_p10_f32(3, 1, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeExecOverwriteAfterExport) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, s_mov_b64_exec_from_s0());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::EXEC);
}

TEST(WaitcheckTest, AcceptsExpcntZeroBeforeExecOverwriteAfterExport) {
  std::vector<uint32_t> program;
  append_inst(program, export_mrt0_v0());
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, s_mov_b64_exec_from_s0());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, CombinedLoadcntDscntWaitsBothCounters) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 4));
  append_inst(program, sopp(72, 0)); // s_wait_loadcnt_dscnt loadcnt(0), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, CombinedLoadcntDscntLeavesNewestLoadPending) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, ds_load_b32(2, 4));
  append_inst(program, sopp(72, 0x100)); // loadcnt(1), dscnt(0)
  append_inst(program, v_mov_b32(3, 1));
  append_inst(program, v_mov_b32(4, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 1u);
}

TEST(WaitcheckTest, CombinedStorecntDscntWaitsBothCounters) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0)); // s_wait_storecnt_dscnt storecnt(0), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, CombinedStorecntDscntLeavesStorePending) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 0x100)); // storecnt(1), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, CombinedStorecntDscntLeavesDsPending) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(10));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(73, 1)); // storecnt(0), dscnt(1)
  append_inst(program, v_mov_b32(2, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, CombinedLoadcntDscntDecodesSixBitFields) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, ds_load_b32(1, 4));
  append_inst(program, sopp(72, 0x3F01)); // loadcnt(63), dscnt(1)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, v_mov_b32(3, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].reg.index, 1u);
}

TEST(WaitcheckTest, ReportsOverwriteBeforeLoadCompletes) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, v_mov_b32(0, 2));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisIgnoresUnreachableSkippedUse) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(32, 1)); // s_branch over the next instruction
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, Gfx950KernelAnalysisIgnoresHazardInUnreachableBlock) {
  const std::vector<uint32_t> program{
      0xBF820004u,              // 0x00: s_branch to s_endpgm.
      0xC00E0A00u, 0x00000050u, // 0x04: dead s_load_dwordx8 s[40:47], s[0:1], 0x50.
      0xC0020B40u, 0x0000008Cu, // 0x0c: dead s_load_dword s45, s[0:1], 0x8c.
      0xBF810000u,              // 0x14: s_endpgm.
  };
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"dead_block", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.instructions_analyzed, 2u);
  EXPECT_EQ(report.memory_events_tracked, 0u);
}

TEST(WaitcheckTest, Gfx950KernelAnalysisAttributesRank1HazardToReachableProducer) {
  const std::vector<uint32_t> program{
      0xBF068205u,              // 0x00: s_cmp_eq_u32 s5, 2.
      0xBF850003u,              // 0x04: s_cbranch_scc1 to the alternate path.
      0xC00E0A00u, 0x00000050u, // 0x08: fallthrough s_load_dwordx8 s[40:47].
      0xBF820004u,              // 0x10: s_branch to the join.
      0xC00E0A00u, 0x00000050u, // 0x14: reachable s_load_dwordx8 s[40:47].
      0xC0020B40u, 0x0000008Cu, // 0x1c: s_load_dword s45 overlaps the reachable load.
      0xBF8CC07Fu,              // 0x24: s_waitcnt lgkmcnt(0).
      0xBF810000u,              // 0x28: s_endpgm.
  };
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"rank1_branch", program}}, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 45u);
  EXPECT_EQ(report.diagnostics[0].producer_section_offset, 0x14u);
  EXPECT_EQ(report.diagnostics[0].section_offset, 0x1cu);
}

TEST(WaitcheckTest, ObjectAnalysisReportsPathThatSkipsWaitAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(33, 1)); // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 on fallthrough only
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisIgnoresBranchInfeasibleSkippedWaitPath) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 2));          // s_cbranch_scc1 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 != 1
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisPreservesInfeasiblePathWithReachabilityCacheDisabled) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 2));          // s_cbranch_scc1 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 != 1
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  WaitcheckOptions options;
  options.max_reachability_cache_bytes = 0;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisPreservesFeasiblePathWithReachabilityCacheDisabled) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(33, 1)); // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0 on fallthrough only
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  WaitcheckOptions options;
  options.max_reachability_cache_bytes = 0;
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisCorrelatesUnsignedRangeChecksAroundWait) {
  std::vector<uint32_t> program;
  append_inst(program, s_cmp_ge_u32(0, 129)); // s0 >= 1
  append_inst(program, sopp(33, 2));          // s_cbranch_scc0 over the load
  append_inst(program, global_load_b32(0));   // load executes only when s0 >= 1
  append_inst(program, s_cmp_ge_u32(0, 129)); // s0 >= 1
  append_inst(program, sopp(33, 1));          // s_cbranch_scc0 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisKeepsPathFeasibleAfterScalarRewrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 3));          // s_cbranch_scc1 to the wait
  append_inst(program, s_mov_b32(0, 129));    // redefine s0, invalidating prior path fact
  append_inst(program, s_cmp_eq_u32(0, 129)); // s0 == 1
  append_inst(program, sopp(34, 1));          // s_cbranch_scc1 over the wait
  append_inst(program, sopp(64, 0));          // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(1, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, ObjectAnalysisRequiresZeroWaitForMixedLoadOrderAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 1)); // one predecessor still has v0 as newest load
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  for (const auto &diag : report.diagnostics) {
    EXPECT_EQ(diag.counter, WaitCounterKind::Load);
    EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
    EXPECT_EQ(diag.reg.cls, RegClass::VGPR);
    EXPECT_EQ(diag.reg.index, 0u);
    EXPECT_EQ(diag.required_count, 0u);
  }
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsConsistentOlderLoadAtJoinAfterPartialWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(64, 1)); // both predecessors have v0 older than v1
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsZeroWaitForMixedLoadOrderAtJoin) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(33, 7)); // s_cbranch_scc0 to the else path
  append_inst(program, global_load_b32(0));
  append_inst(program, global_load_b32(1));
  append_inst(program, sopp(32, 6)); // s_branch to join
  append_inst(program, global_load_b32(1));
  append_inst(program, global_load_b32(0));
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, v_mov_b32(2, 0));

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ObjectAnalysisReportsLoopCarriedDsLoadUse) {
  std::vector<uint32_t> program;
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-4))); // s_cbranch_scc1 to loop header

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  size_t uses = 0;
  size_t defs = 0;
  for (const auto &diagnostic : report.diagnostics) {
    EXPECT_EQ(diagnostic.counter, WaitCounterKind::Ds);
    EXPECT_EQ(diagnostic.reg.cls, RegClass::VGPR);
    EXPECT_EQ(diagnostic.reg.index, 0u);
    uses += diagnostic.access == WaitcheckAccessKind::Use;
    defs += diagnostic.access == WaitcheckAccessKind::Def;
  }
  EXPECT_EQ(uses, 1u);
  EXPECT_EQ(defs, 1u);
}

TEST(WaitcheckTest, ObjectAnalysisAcceptsLoopCarriedDsLoadUseAfterWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(70, 0)); // s_wait_dscnt 0 at loop header
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-5))); // s_cbranch_scc1 to loop header

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DescriptorEntryAnalysisIgnoresPaddingAfterEndpgm) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_padded_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());

  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty());
  EXPECT_EQ(report.instructions_analyzed, 2u);
}

TEST(WaitcheckTest, DescriptorEntryAnalysisStopsAtUnterminatedFunctionEnd) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1250_unterminated_stub_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  const auto kernels = waitcheck_kernels(code_object);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].code_size, 2 * sizeof(uint32_t));

  auto report = analyze_waitcnts_for_kernel(code_object, ROCJITSU_CODE_ARCH_CDNA5, kernels[0]);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_EQ(report.instructions_analyzed, 1u);
}

TEST(WaitcheckTest, FunctionEntryAnalysisIgnoresInterFunctionPadding) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_function_only_padded_code_object();
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());

  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.instructions_analyzed, 5u);
}

TEST(WaitcheckTest, SharedKernelEntryDiagnosticsAreReportedOnce) {
  std::vector<uint32_t> words;
  rocjitsu::waitcheck_test::append_inst(words, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(words, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  words.push_back(0xBFB00000U); // s_endpgm

  auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", words}, {"alias", words}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kDescriptorSize = 64;
  constexpr uint64_t kEntryOffsetField = 16;
  const uint64_t text_size = 2 * words.size() * sizeof(uint32_t);
  const uint64_t rodata_offset = kTextOffset + text_size;
  const uint64_t rodata_vaddr = kTextVaddr + text_size + 0x1000;
  const uint64_t alias_descriptor_vaddr = rodata_vaddr + kDescriptorSize;
  const int64_t alias_entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(alias_descriptor_vaddr);
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kEntryOffsetField,
              &alias_entry_offset, sizeof(alias_entry_offset));

  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernels[0].entry_offset, 0u);
  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
}

TEST(WaitcheckTest, SharedKernelEntryPreservesDistinctWaveModes) {
  const std::vector<uint32_t> words{0xBFB00000U}; // s_endpgm.
  auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"wave64", words}, {"wave32", words}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextVaddr = 0x1100;
  constexpr uint64_t kDescriptorSize = 64;
  constexpr uint64_t kEntryOffsetField = 16;
  constexpr uint64_t kPropertiesField = 56;
  constexpr uint16_t kWave32 = 1u << 10u;
  const uint64_t text_size = 2 * words.size() * sizeof(uint32_t);
  const uint64_t rodata_offset = kTextOffset + text_size;
  const uint64_t rodata_vaddr = kTextVaddr + text_size + 0x1000;
  const uint64_t wave32_descriptor_vaddr = rodata_vaddr + kDescriptorSize;
  const int64_t shared_entry_offset =
      static_cast<int64_t>(kTextVaddr) - static_cast<int64_t>(wave32_descriptor_vaddr);
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kEntryOffsetField,
              &shared_entry_offset, sizeof(shared_entry_offset));
  std::memcpy(image.data() + rodata_offset + kDescriptorSize + kPropertiesField, &kWave32,
              sizeof(kWave32));

  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);

  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(kernels[0].entry_offset, 0u);
  EXPECT_EQ(kernels[1].entry_offset, 0u);
  EXPECT_NE(kernels[0].wavefront_size, kernels[1].wavefront_size);
}

TEST(WaitcheckTest, KnownKernelBatchReportsPerKernelCompletion) {
  std::vector<uint32_t> hazardous;
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous.push_back(0xBFB00000U); // s_endpgm
  const std::vector<uint32_t> clean{0xBFB00000U};
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"hazardous", hazardous}, {"clean", clean}}, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  AmdGpuCodeObject code_object(image.data(), image.size());
  ASSERT_TRUE(code_object.is_valid());
  const auto kernels = waitcheck_kernels(code_object);
  ASSERT_EQ(kernels.size(), 2u);

  std::vector<std::string> completed;
  WaitcheckOptions options;
  size_t completion_count = 0;
  options.kernel_analyzed_callback = [&] { ++completion_count; };
  options.kernel_timing_callback = [&](const WaitcheckKernelInfo &kernel,
                                       std::chrono::nanoseconds elapsed) {
    EXPECT_GE(elapsed.count(), 0);
    completed.push_back(kernel.name);
  };
  const auto report =
      analyze_waitcnts_for_kernels(code_object, ROCJITSU_CODE_ARCH_RDNA4, kernels, options);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_EQ(report.kernels_discovered, 2u);
  EXPECT_EQ(report.kernels_analyzed, 2u);
  EXPECT_EQ(completion_count, 2u);
  EXPECT_EQ(completed.size(), 2u);
  EXPECT_EQ(completed[0], kernels[0].name);
  EXPECT_EQ(completed[1], kernels[1].name);
  EXPECT_EQ(report.diagnostics_observed, 1u) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ResolvedSwappcHelperWaitAppliesBeforeContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // The kernel leaves a VMEM result pending, builds the helper address relative
  // to s_getpc, and uses s_swappc to call it. The helper's entry wait is the only
  // wait before the continuation consumes v0. Runtime waitcheck must decode the
  // proven helper target without walking unrelated .text bytes and propagate
  // the helper state back through its matching s[30:31] return.
  std::vector<uint32_t> program;
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                     // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x08.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x0c.
  program.push_back(24);                                        // 0x10 -> 0x24.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x14.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x18.
  append_gfx950_v_mov_b32_v1_v0(program);                                      // 0x1c.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x20.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                    // 0x24 helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x28 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ResolvedSwappcPreservesCanonicalizedPcHighHalf) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // PyTorch's gfx1201 address builder canonicalizes the upper 16 bits of the
  // getpc result before applying its PC-relative delta. The helper entry wait
  // must still flow back to the continuation that consumes v0.
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                    // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4)); // 0x0c.
  program.push_back(0xBE890F09u); // 0x10: s_sext_i32_i16 s9, s9.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_RDNA4)); // 0x14.
  program.push_back(28);                                        // Base 0x10 -> helper 0x2c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_RDNA4)); // 0x1c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4)); // 0x20.
  append_inst(program, v_mov_b32(1, 0));                                       // 0x24.
  append_inst(program, s_endpgm());                                            // 0x28.
  append_inst(program, sopp(64, 0));                                           // 0x2c helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_RDNA4, 0x48, 0,
                                        kReturnSreg)); // 0x30 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuSgprHazardBeforeResolvedSwappcCall) {
  const std::vector<uint32_t> program =
      gfx1201_swappc_valu_sgpr_boundary_program(/*include_wait=*/false);
  ASSERT_FALSE(program.empty());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.kind, WaitcheckDiagnosticKind::SgprDepctr);
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_swappc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_va_sdst(0)"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("control transfer"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsValuSgprWaitBeforeResolvedSwappcCall) {
  const std::vector<uint32_t> program =
      gfx1201_swappc_valu_sgpr_boundary_program(/*include_wait=*/true);
  ASSERT_FALSE(program.empty());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuSgprHazardBeforeResolvedDirectCall) {
  const std::vector<uint32_t> program =
      gfx1201_call_valu_sgpr_boundary_program(/*include_wait=*/false);
  ASSERT_FALSE(program.empty());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_call_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_va_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsValuSgprWaitBeforeResolvedDirectCall) {
  const std::vector<uint32_t> program =
      gfx1201_call_valu_sgpr_boundary_program(/*include_wait=*/true);
  ASSERT_FALSE(program.empty());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsSaluSgprHazardBeforeResolvedFunctionReturn) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("s_setpc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201AcceptsSaluSgprWaitBeforeResolvedFunctionReturn) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, s_wait_alu_sa_sdst_0());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ReportsValuVccHazardBeforeResolvedSwappcCall) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;
  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4));
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(program, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  program.push_back(build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  const uint64_t helper_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(helper_offset - kGetpcResultOffset);
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSetpcStillChecksSgprBoundaryHazards) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_NE(report.diagnostics.front().message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSwappcConservativelySeedsItsContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_swappc_b64(kReturnSreg, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201UnresolvedSwappcLinkRegisterIsPendingAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_swappc_b64(kReturnSreg, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, kReturnSreg, 1}));
  EXPECT_NE(diagnostic.producer_instruction.find("s_swappc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ResolvedCallLinkRegisterIsPendingAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, /*offset_dwords=*/2, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, kReturnSreg, 1}));
  EXPECT_NE(diagnostic.producer_instruction.find("s_call_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201WaitClearsResolvedCallLinkRegisterAtContinuation) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, /*offset_dwords=*/3, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_wait_alu_sa_sdst_0());
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/kReturnSreg, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  program.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ResolvedIndirectBranchPreservesOrdinarySgprState) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;
  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  program.push_back(
      build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand, ROCJITSU_CODE_ARCH_RDNA4));
  const size_t delta_word = program.size();
  program.push_back(0);
  program.push_back(
      build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0, ROCJITSU_CODE_ARCH_RDNA4));
  program.push_back(build_s_setpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  const uint64_t target_offset = program.size() * sizeof(uint32_t);
  constexpr uint64_t kGetpcResultOffset = sizeof(uint32_t);
  program[delta_word] = static_cast<uint32_t>(target_offset - kGetpcResultOffset);
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  EXPECT_GE(report.instructions_analyzed, 7u);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201DirectBranchPreservesPendingSgprHazard) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_branch(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_RDNA4));
  append_inst(program, s_endpgm());
  append_inst(program, v_add_f32_e32(/*vdst=*/1, /*src0=*/2, /*vsrc1=*/1));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.instruction.find("v_add_f32"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201SymbolLessSectionConservativelySeedsEveryCfgRoot) {
  std::vector<uint32_t> program;
  append_inst(program, s_endpgm());
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_endpgm());
  const auto image = rocjitsu::waitcheck_test::make_gfx_symbol_less_code_object(
      program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics.front().access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics.front().reg, (RegisterRef{RegClass::SGPR, 2, 1}));
}

TEST(WaitcheckTest, Gfx1201ReportsSaluVccHazardAtControlTransfer) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/106, /*ssrc0=*/128)); // vcc_lo.
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ControlTransferReportsOneRepresentativePerDepctrField) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/1, /*src0=*/0, /*vsrc1=*/1));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(program, s_mov_b32(/*sdst=*/106, /*ssrc0=*/128)); // vcc_lo.
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201ScalarIssueStallClearsValuSgprHazardBeforeTransfer) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/2, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  // Any register-bearing SALU instruction supplies the hardware issue stall
  // modeled by AMDGPUInsertDelayAlu, even when its operands are unrelated.
  append_inst(program, s_mov_b32(/*sdst=*/4, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250DoesNotApplyRdna4SgprControlTransferRules) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_CDNA5));
  const auto image = rocjitsu::waitcheck_test::make_gfx1250_code_object(program);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1201ControlTransferReportsEveryOutstandingSgprWaitField) {
  std::vector<uint32_t> program;
  append_inst(program, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(program, v_add_f32_e32(/*vdst=*/1, /*src0=*/4, /*vsrc1=*/1));
  append_inst(program, v_cndmask_b32_e32(/*vdst=*/2, /*src0=*/0, /*vsrc1=*/2));
  append_inst(program, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  const auto readlane =
      build_rdna4_v_readlane_b32(/*sdst=*/4, /*vsrc=*/40, /*lane=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(readlane);
  program.insert(program.end(), readlane->begin(), readlane->end());
  append_inst(program, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  program.push_back(build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 3u) << diagnostic_summary(report);
  // A single all-zero depctr wait repairs every register covered by a field.
  // Report one representative register per required field instead of emitting
  // a duplicate diagnostic for every outstanding lane.
  std::set<std::string> fields;
  for (const WaitcheckDiagnostic &diagnostic : report.diagnostics) {
    EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::ControlTransfer);
    if (diagnostic.message.find("depctr_sa_sdst(0)") != std::string::npos)
      fields.insert("sa_sdst");
    if (diagnostic.message.find("depctr_va_sdst(0)") != std::string::npos)
      fields.insert("va_sdst");
    if (diagnostic.message.find("depctr_va_vcc(0)") != std::string::npos)
      fields.insert("va_vcc");
  }
  EXPECT_EQ(fields, (std::set<std::string>{"sa_sdst", "va_sdst", "va_vcc"}));
}

TEST(WaitcheckTest, Gfx1201NonEntryFunctionConservativelyTracksEverySgprPair) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> function;
  append_inst(function, s_mov_b32(/*sdst=*/2, /*ssrc0=*/128));
  append_inst(function, v_add_f32_e32(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0));
  append_inst(function, s_wait_alu_sa_sdst_0());
  function.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx1201_function_only_code_object({{"helper", function}});
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_sa_sdst(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1201NonEntryFunctionConservativelyTracksVcc) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> function;
  append_inst(function, v_cmp_gt_u32_e32(/*src0=*/0, /*vsrc1=*/0));
  append_inst(function, v_cndmask_b32_e32(/*vdst=*/0, /*src0=*/0, /*vsrc1=*/0));
  append_inst(function, s_wait_alu_va_vcc_0());
  function.push_back(build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_RDNA4));
  const auto image =
      rocjitsu::waitcheck_test::make_gfx1201_function_only_code_object({{"helper", function}});
  AmdGpuCodeObject code_object(image.data(), image.size());

  const auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  const WaitcheckDiagnostic &diagnostic = report.diagnostics.front();
  EXPECT_EQ(diagnostic.access, WaitcheckAccessKind::Use);
  EXPECT_EQ(diagnostic.reg, (RegisterRef{RegClass::VCC, 0, 1}));
  EXPECT_NE(diagnostic.message.find("depctr_va_vcc(0)"), std::string::npos);
}

TEST(WaitcheckTest, Gfx1250ResolvedSwappcLiteral64HelperWaitAppliesBeforeContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                    // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5)); // 0x0c.
  program.push_back(0xA988FE08u); // 0x10: s_add_nc_u64 s[8:9], s[8:9], literal64.
  program.push_back(24);          // getpc next 0x10 + 24 -> helper 0x28.
  program.push_back(0);
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5)); // 0x1c.
  append_inst(program, v_mov_b32(1, 0));                                       // 0x20.
  append_inst(program, s_endpgm());                                            // 0x24.
  append_inst(program, sopp(64, 0)); // 0x28 helper: s_wait_loadcnt 0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0x48, 0,
                                        kReturnSreg)); // 0x2c return.

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  std::unique_ptr<Instruction> add_inst(decode_valid(*decoder, &program[4], 0x10));
  ASSERT_NE(add_inst, nullptr);
  EXPECT_EQ(add_inst->mnemonic(), "s_add_nc_u64");
  ASSERT_EQ(add_inst->num_dst_operands(), 1);
  ASSERT_EQ(add_inst->num_src_operands(), 2);
  EXPECT_EQ(add_inst->dst_operand(0)->to_register_ref(),
            (RegisterRef{RegClass::SGPR, kTargetSreg, 2}));
  EXPECT_EQ(add_inst->src_operand(0)->to_register_ref(),
            (RegisterRef{RegClass::SGPR, kTargetSreg, 2}));
  EXPECT_EQ(add_inst->src_operand(1)->literal64_value(), 24u);

  std::vector<std::unique_ptr<Instruction>> decoded;
  std::vector<const Instruction *> decoded_ptrs;
  for (uint64_t offset = 0; offset < program.size() * sizeof(uint32_t);) {
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, &program[offset / sizeof(uint32_t)], offset));
    ASSERT_NE(inst, nullptr);
    offset += static_cast<uint64_t>(inst->size());
    decoded_ptrs.push_back(inst.get());
    decoded.push_back(std::move(inst));
  }
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(program.data()),
                                             program.size() * sizeof(uint32_t));
  const auto fixups = discover_indirect_branch_edges(decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_EQ(fixups[0].source_target_offset, 0x28u);
  EXPECT_TRUE(fixups[0].source_is_call);

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_GE(report.instructions_analyzed, 8u);
}

TEST(WaitcheckTest, Gfx1250Wave32VopcDoesNotClobberAdjacentSwappcTarget) {
  constexpr uint16_t kTargetSreg = 4;
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0));                                    // 0x00.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5)); // 0x0c.
  program.push_back(0xA984FE04u); // 0x10: s_add_nc_u64 s[4:5], s[4:5], literal64.
  program.push_back(32);          // getpc next 0x10 + 32 -> helper 0x30.
  program.push_back(0);
  program.push_back(0xD4410003u); // 0x1c: v_cmp_lt_i32_e64 s3, v20, v101.
  program.push_back(0x0202CB14u);
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5)); // 0x24.
  append_inst(program, v_mov_b32(1, 0));                                       // 0x28.
  append_inst(program, s_endpgm());                                            // 0x2c.
  append_inst(program, sopp(72, 0)); // 0x30 helper: s_wait_loadcnt_dscnt 0.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA5, 0x48, 0,
                                        kReturnSreg)); // 0x34 return.

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  std::unique_ptr<Instruction> cmp_inst(decode_valid(*decoder, &program[7], 0x1c));
  ASSERT_NE(cmp_inst, nullptr);
  EXPECT_EQ(cmp_inst->mnemonic(), "v_cmp_lt_i32");
  ASSERT_EQ(cmp_inst->num_dst_operands(), 1);
  EXPECT_EQ(cmp_inst->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::SGPR, 3, 1}));

  std::vector<std::unique_ptr<Instruction>> decoded;
  std::vector<const Instruction *> decoded_ptrs;
  for (uint64_t offset = 0; offset < program.size() * sizeof(uint32_t);) {
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, &program[offset / sizeof(uint32_t)], offset));
    ASSERT_NE(inst, nullptr);
    offset += static_cast<uint64_t>(inst->size());
    decoded_ptrs.push_back(inst.get());
    decoded.push_back(std::move(inst));
  }
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(program.data()),
                                             program.size() * sizeof(uint32_t));
  const auto fixups =
      discover_indirect_branch_edges(decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5, {}, 32);
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_EQ(fixups[0].source_target_offset, 0x30u);
  EXPECT_TRUE(fixups[0].source_is_call);
  EXPECT_TRUE(
      discover_indirect_branch_edges(decoded_ptrs, text, ROCJITSU_CODE_ARCH_CDNA5, {}, 64).empty());

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX1250, true);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA5);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950ResolvedSwappcPropagatesHelperLoadToContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  std::vector<uint32_t> program;
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x00.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x04.
  program.push_back(24);                                        // 0x08 -> 0x1c.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x0c.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x10.
  append_gfx950_v_mov_b32_v1_v0(program);                                      // 0x14.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x18.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                     // 0x1c helper.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x24 return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
}

TEST(WaitcheckTest, Gfx950SharedSwappcReturnsToMatchingContinuation) {
  constexpr uint16_t kTargetSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kLiteralOperand = 255;
  constexpr uint16_t kInlineInt0 = 128;

  // The branch selects exactly one call to the shared helper. The B1 path has
  // a pending VMEM load but waits in its own continuation. Pairing the helper's
  // return with the B0 continuation would create an impossible path to the B0
  // v0 consumer and a false missing-wait diagnostic.
  std::vector<uint32_t> program;
  program.push_back(0xBF840007u); // 0x00: s_cbranch_scc0 to B1 at 0x20.

  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x04.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x08.
  program.push_back(60);                                        // Base 0x08 -> helper 0x44.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x10.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x14.
  append_gfx950_v_mov_b32_v1_v0(program);                                      // 0x18 B0.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x1c.

  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                     // 0x20 B1.
  program.push_back(build_s_getpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x28.
  program.push_back(build_s_add_u32(kTargetSreg, kTargetSreg, kLiteralOperand,
                                    ROCJITSU_CODE_ARCH_CDNA4)); // 0x2c.
  program.push_back(24);                                        // Base 0x2c -> helper 0x44.
  program.push_back(build_s_addc_u32(kTargetSreg + 1, kTargetSreg + 1, kInlineInt0,
                                     ROCJITSU_CODE_ARCH_CDNA4)); // 0x34.
  program.push_back(
      build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA4)); // 0x38.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                    // 0x3c B1.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                 // 0x40.
  program.push_back(build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0,
                                        kReturnSreg)); // 0x44 shared return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx950RecursiveHelperUsesFiniteCallContextSummary) {
  constexpr uint16_t kReturnSreg = 30;

  // The helper recursively calls its own entry before waiting for a VMEM load.
  // Expanding every finite recursive stack produces exponentially many
  // analysis nodes even though the helper body and pending-state lattice are
  // finite. The recursive edge is summarized at its continuation; the outer
  // call still returns to the exact main-kernel continuation.
  std::vector<uint32_t> program;
  program.push_back(build_s_call_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_CDNA4));  // 0x00 -> 0x0c.
  append_gfx950_v_mov_b32_v1_v0(program);                                         // 0x04.
  program.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));                    // 0x08.
  append_gfx950_buffer_load_dword_v0_v8_s0_offen(program);                        // 0x0c helper.
  program.push_back(build_s_call_b64(kReturnSreg, -3, ROCJITSU_CODE_ARCH_CDNA4)); // 0x14 -> 0x0c.
  append_gfx950_s_waitcnt_vmcnt_0(program);                                       // 0x18.
  program.push_back(
      build_sop1_encoding(ROCJITSU_CODE_ARCH_CDNA4, 0x1d, 0, kReturnSreg)); // 0x1c return.

  const auto image =
      rocjitsu::waitcheck_test::make_gfx_code_object(program, EF_AMDGPU_MACH_AMDGCN_GFX950);
  AmdGpuCodeObject code_object(image.data(), image.size());
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_TRUE(report.supported) << report.analysis_error;
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
  EXPECT_LT(report.instructions_analyzed, 64u);
}

} // namespace
