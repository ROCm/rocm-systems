// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../tools/waitcheck_fixture.h"
#include "rocjitsu/analysis/waitcheck.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
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
    code_sections_.push_back(sections_.back().get());
  }
};

template <typename T> void append_inst(std::vector<uint32_t> &words, const T &inst) {
  static_assert(sizeof(T) % sizeof(uint32_t) == 0);
  std::array<uint32_t, sizeof(T) / sizeof(uint32_t)> encoded =
      std::bit_cast<std::array<uint32_t, sizeof(T) / sizeof(uint32_t)>>(inst);
  words.insert(words.end(), encoded.begin(), encoded.end());
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

[[nodiscard]] rdna4::Sop1MachineInst s_mov_b64_exec_from_s0() {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE800100U);
  inst.sdst = 126; // exec_lo
  inst.ssrc0 = 0;
  return inst;
}

[[nodiscard]] rdna4::Sop1MachineInst s_barrier_signal_isfirst(uint32_t barrier_id) {
  auto inst = std::bit_cast<rdna4::Sop1MachineInst>(0xBE804F00U);
  inst.ssrc0 = barrier_id;
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

[[nodiscard]] rdna4::SmemMachineInst s_load_b32(uint32_t sdata, uint32_t sbase) {
  auto inst = std::bit_cast<rdna4::SmemMachineInst>(std::array<uint32_t, 2>{0xF4000000U, 0});
  inst.sdata = sdata;
  inst.sbase = sbase;
  return inst;
}

void append_v_cmp_gt_u32_s2_s5_v12(std::vector<uint32_t> &program) {
  program.push_back(0xD44C0002u);
  program.push_back(0x02021805u);
}

void append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(std::vector<uint32_t> &program) {
  program.push_back(0xCA500501u);
  program.push_back(0x02000080u);
}

[[nodiscard]] rdna4::VglobalMachineInst global_load_b32(uint32_t vdst) {
  rdna4::VglobalMachineInst inst{};
  inst.encoding = 0xEE;
  inst.op = 20;
  inst.vdst = vdst;
  inst.vaddr = 8;
  inst.saddr = 0;
  return inst;
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

[[nodiscard]] rdna4::VglobalMachineInst global_store_b32(uint32_t vsrc) {
  auto inst = std::bit_cast<rdna4::VglobalMachineInst>(std::array<uint32_t, 3>{0xEE068000U, 0, 0});
  inst.vsrc = vsrc;
  inst.vaddr = 8;
  inst.saddr = 0;
  return inst;
}

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

[[nodiscard]] rdna4::VdsMachineInst ds_nop() {
  return std::bit_cast<rdna4::VdsMachineInst>(std::array<uint32_t, 2>{0xD8500000U, 0});
}

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
                                       uint32_t reg_index) {
  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, counter);
  EXPECT_EQ(report.diagnostics[0].access, access);
  EXPECT_EQ(report.diagnostics[0].reg.cls, reg_class);
  EXPECT_EQ(report.diagnostics[0].reg.index, reg_index);
  EXPECT_EQ(report.diagnostics[0].required_count, kOverflowRequiredCount);
}

TEST(WaitcheckTest, ReportsUnsupportedArchitectures) {
  auto program = words({sopp(0)});

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_CDNA4);

  EXPECT_FALSE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, DecodesRdna4CombinedStoreDsWait) {
  std::vector<uint32_t> program;
  append_inst(program, sopp(73, 0));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decoder->decode(program.data()));
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
    std::unique_ptr<Instruction> inst(decoder->decode(&program[word_index]));
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
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
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

TEST(WaitcheckTest, ReportsMissingWaitAluVmVsrcBeforeVmemSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_load_b32(0)); // uses v[8:9] as the address.
  append_inst(program, v_mov_b32(8, 10));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported) << report.analysis_error;
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 8u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_vm_vsrc(0)"), std::string::npos);
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

TEST(WaitcheckTest, ReportsFlatToVmemLoadOverwriteOnBothPossibleFlatCounters) {
  std::vector<uint32_t> program;
  append_inst(program, flat_load_b32(0));
  append_inst(program, global_load_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
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

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeSeparatesLowLoadFromV512StoreData) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 0)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 2, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameHighStoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 1)));
  append_inst(program, flat_load_b32(255));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 1, 0, 0)));
  append_inst(program, flat_store_b32(255));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

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

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, Gfx1250HighVgprModeReportsSameV768StoreDataUse) {
  std::vector<uint32_t> program;
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 0, 0, 3)));
  append_inst(program, flat_load_b32(0));
  append_inst(program, s_set_vgpr_msb(vgpr_msb_mode(0, 3, 0, 0)));
  append_inst(program, flat_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

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

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

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

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

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
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::SGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 102u);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_sa_sdst(0)"), std::string::npos);
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

TEST(WaitcheckTest, ReportsMissingWaitAluVaVccBeforeVopdCndmaskReadsTrackedVcc) {
  std::vector<uint32_t> program;
  append_inst(program, v_cndmask_b32_e32(0, 128, 1)); // VALU reads VCC, tracking it.
  append_inst(program, v_cmp_gt_u32_e32(5, 12));      // VALU writes VCC.
  append_v_dual_cndmask_b32_v2_v1_v2_dual_mov_b32_v1_0(program);

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_GFX1250);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Depctr);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VCC);
  EXPECT_NE(report.diagnostics[0].message.find("depctr_va_vcc(0)"), std::string::npos);
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

TEST(WaitcheckTest, ReportsStorecntMaxCountForOverflowSizedQueueBeforeProgramEnd) {
  std::vector<uint32_t> program;
  append_global_stores(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, s_endpgm());
  WaitcheckOptions options;
  options.max_diagnostics = 1;

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4, options);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Store, WaitcheckAccessKind::ProgramEnd,
                                    RegClass::PC, 0);
  EXPECT_EQ(report.diagnostics_observed, kOverflowQueueSize);
  EXPECT_TRUE(report.diagnostics_truncated);
}

TEST(WaitcheckTest, ReportsKmcntMaxCountForOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, s_mov_b32(kOverflowKmcntConsumerSgpr, kOverflowKmcntBaseSgpr));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  expect_single_overflow_diagnostic(report, WaitCounterKind::Km, WaitcheckAccessKind::Use,
                                    RegClass::SGPR, kOverflowKmcntBaseSgpr);
}

TEST(WaitcheckTest, AcceptsKmcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_s_loads(program, kOverflowQueueSize, kOverflowKmcntBaseSgpr, kOverflowKmcntSbase);
  append_inst(program, sopp(71, kOverflowRequiredCount)); // s_wait_kmcnt 39
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
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsBvhcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_image_bvhs(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(67, kOverflowRequiredCount)); // s_wait_bvhcnt 39
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
                                    RegClass::VGPR, kOverflowBaseVgpr);
}

TEST(WaitcheckTest, AcceptsExpcntMaxCountForOldestOverflowSizedQueue) {
  std::vector<uint32_t> program;
  append_ds_direct_loads(program, kOverflowQueueSize, kOverflowBaseVgpr);
  append_inst(program, sopp(68, kOverflowRequiredCount)); // s_wait_expcnt 39
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

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsStoreDataOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(0, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeDsStoreAddressOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, v_mov_b32(4, 1));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 4u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 4u);
}

TEST(WaitcheckTest, AcceptsExpcntZeroBeforeDsStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, ds_store_b32(4, 0));
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
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

TEST(WaitcheckTest, ReportsMissingExpcntBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, StoreSourceReadDoesNotNeedExpcnt) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, v_mov_b32(1, 0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingStorecntAfterGlobalWbBeforeMemoryOp) {
  std::vector<uint32_t> program;
  append_inst(program, global_wb());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Store);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
}

TEST(WaitcheckTest, AcceptsStorecntAfterGlobalWbBeforeMemoryOp) {
  std::vector<uint32_t> program;
  append_inst(program, global_wb());
  append_inst(program, sopp(65, 0)); // s_wait_storecnt 0
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingStorecntAfterGlobalWbinvBeforeMemoryOp) {
  std::vector<uint32_t> program;
  append_inst(program, global_wbinv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Store);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, ReportsMissingLoadcntAfterGlobalInvBeforeMemoryOp) {
  std::vector<uint32_t> program;
  append_inst(program, global_inv());
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::MemoryOrder);
}

TEST(WaitcheckTest, AcceptsLoadcntAfterGlobalInvBeforeMemoryOp) {
  std::vector<uint32_t> program;
  append_inst(program, global_inv());
  append_inst(program, sopp(64, 0)); // s_wait_loadcnt 0
  append_inst(program, global_store_b32(0));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty());
}

TEST(WaitcheckTest, ReportsMissingStorecntBeforeEndpgmAfterVmemStore) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Store);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::ProgramEnd);
  EXPECT_EQ(report.diagnostics[0].required_count, 0u);
  EXPECT_NE(report.diagnostics[0].message.find("program end"), std::string::npos);
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

TEST(WaitcheckTest, AcceptsExpcntZeroBeforeVmemStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, global_store_b32(0));
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
  append_inst(program, s_wait_alu_vm_vsrc_0());
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_TRUE(report.supported);
  EXPECT_TRUE(report.diagnostics.empty()) << diagnostic_summary(report);
}

TEST(WaitcheckTest, ReportsMissingExpcntBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, v_mov_b32(0, 4));

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  EXPECT_EQ(report.memory_events_tracked, 3u);
  ASSERT_EQ(report.diagnostics.size(), 2u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Exp);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
}

TEST(WaitcheckTest, AcceptsExpcntZeroBeforeImageStoreSourceOverwrite) {
  std::vector<uint32_t> program;
  append_inst(program, image_store_b32(0));
  append_inst(program, sopp(68, 0)); // s_wait_expcnt 0
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
  append_inst(program, global_load_b32(0)); // uses v[8:9] as the address.
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
  append_inst(program, global_load_b32(0)); // uses v[8:9] as the address.
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
  append_inst(program, sopp(72, 0x10)); // loadcnt(1), dscnt(0)
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
  append_inst(program, sopp(73, 0x10)); // storecnt(1), dscnt(0)
  append_inst(program, v_mov_b32(2, 0));
  append_inst(program, s_endpgm());

  auto report = analyze_waitcnts(program, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 1u) << diagnostic_summary(report);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Store);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::ProgramEnd);
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
  ASSERT_EQ(report.diagnostics.size(), 2u);
  for (const auto &diag : report.diagnostics) {
    EXPECT_EQ(diag.counter, WaitCounterKind::Load);
    EXPECT_EQ(diag.access, WaitcheckAccessKind::Use);
    EXPECT_EQ(diag.reg.cls, RegClass::VGPR);
    EXPECT_EQ(diag.reg.index, 0u);
    EXPECT_EQ(diag.required_count, 0u);
  }
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

TEST(WaitcheckTest, ObjectAnalysisReportsLoopCarriedDsLoadHazards) {
  std::vector<uint32_t> program;
  append_inst(program, v_mov_b32(1, 0));
  append_inst(program, ds_load_b32(0, 4));
  append_inst(program, sopp(34, static_cast<uint16_t>(-4))); // s_cbranch_scc1 to loop header

  TestCodeObject code_object(program);
  auto report = analyze_waitcnts(code_object, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(report.supported);
  ASSERT_EQ(report.diagnostics.size(), 2u);
  EXPECT_EQ(report.diagnostics[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[0].access, WaitcheckAccessKind::Use);
  EXPECT_EQ(report.diagnostics[0].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[0].reg.index, 0u);
  EXPECT_EQ(report.diagnostics[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(report.diagnostics[1].access, WaitcheckAccessKind::Def);
  EXPECT_EQ(report.diagnostics[1].reg.cls, RegClass::VGPR);
  EXPECT_EQ(report.diagnostics[1].reg.index, 0u);
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

} // namespace
