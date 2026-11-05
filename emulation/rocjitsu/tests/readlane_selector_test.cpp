// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

class ReadlaneSelectorTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("readlane", cfg, &mem, &l2);
    decoder = Decoder::create(GetParam());
    wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu->write_vgpr(wf->vgpr_alloc().base, lane, 0xa5000000u + lane);
  }

  void TearDown() override {
    if (wf)
      wf->halt();
  }

  amdgpu::GpuMemory mem{"readlane_mem"};
  amdgpu::L2Cache l2{"readlane_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, ReadlaneSelectorTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST_P(ReadlaneSelectorTest, ReadlaneMasksInline64AndLiteralIndices) {
  wf->set_exec(0); // READLANE ignores EXEC, even for inactive source lanes.
  for (uint32_t selector : {192u, 255u}) {
    if (selector == 255 && GetParam() != ROCJITSU_CODE_ARCH_RDNA3 &&
        GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
      continue; // Literal lane indices are only qualified on these two targets.
    // v_readlane_b32 s4, v0, 64 / literal 95, followed by s_endpgm.
    std::array<uint32_t, 4> words{gfx9() ? 0xd2890004u : 0xd7600004u, 256u | (selector << 9), 95,
                                  0xbfb00000};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->size(), selector == 255 ? 12u : 8u);
    EXPECT_EQ(inst->disassemble(),
              selector == 255 ? "v_readlane_b32 s4, v0, 0x5f" : "v_readlane_b32 s4, v0, 64");
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    EXPECT_EQ(cu->read_sgpr(wf->sgpr_alloc().base + 4),
              0xa5000000u + ((selector == 255 ? 95u : 64u) & (wf->wf_size() - 1)));
  }
}

TEST_P(ReadlaneSelectorTest, ReadlaneWritesBothVccWordsWithEmptyExec) {
  wf->set_exec(0);
  // The lane index comes from s4, rather than its selector encoding.
  cu->write_sgpr(wf->sgpr_alloc().base + 4, 5);
  for (uint32_t dst : {106u, 107u}) {
    SCOPED_TRACE(dst);
    constexpr uint64_t initial = 0x1122334455667788ull;
    wf->set_vcc_raw(initial);
    std::array<uint32_t, 2> words{(gfx9() ? 0xd2890000u : 0xd7600000u) | dst, 256u | (4u << 9)};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(
        inst->disassemble(),
        GetParam() == ROCJITSU_CODE_ARCH_CDNA5
            ? (dst == 106 ? "v_readlane_b32 VCC_LO, v0, s4" : "v_readlane_b32 VCC_HI, v0, s4")
            : (dst == 106 ? "v_readlane_b32 vcc_lo, v0, s4" : "v_readlane_b32 vcc_hi, v0, s4"));
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    const uint64_t value = 0xa5000005u;
    EXPECT_EQ(wf->vcc(), dst == 106 ? (initial & 0xffffffff00000000ull) | value
                                    : (initial & 0xffffffffull) | (value << 32));
    EXPECT_EQ(wf->exec(), 0u);
  }
}

TEST_P(ReadlaneSelectorTest, ReadfirstlaneWritesBothVccWords) {
  for (bool e64 : {false, true}) {
    for (uint64_t exec : {uint64_t{0}, uint64_t{1} << 5 | uint64_t{1} << 7}) {
      for (uint32_t dst : {106u, 107u}) {
        SCOPED_TRACE(e64);
        SCOPED_TRACE(exec);
        SCOPED_TRACE(dst);
        constexpr uint64_t initial = 0x1122334455667788ull;
        wf->set_vcc_raw(initial);
        wf->set_exec(exec);
        // VOP1 and VOP3 forms from the architecture encoding tables.
        std::array<uint32_t, 2> words =
            e64 ? std::array<uint32_t, 2>{(gfx9() ? 0xd1420000u : 0xd5820000u) | dst, 256u}
                : std::array<uint32_t, 2>{0x7e000500u | (dst << 17), 0u};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        EXPECT_EQ(inst->size(), e64 ? 8u : 4u);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        const uint64_t value = 0xa5000000u + (exec ? 5u : 0u);
        EXPECT_EQ(wf->vcc(), dst == 106 ? (initial & 0xffffffff00000000ull) | value
                                        : (initial & 0xffffffffull) | (value << 32));
        EXPECT_EQ(wf->exec(), exec);
      }
    }
  }
}

TEST_P(ReadlaneSelectorTest, WritelaneUsesVccSourceAndLaneIndexWithEmptyExec) {
  wf->set_exec(0);
  for (uint32_t src : {106u, 107u}) {
    SCOPED_TRACE(src);
    constexpr uint64_t initial = 0xdeadbeefcafebabeull;
    wf->set_vcc_raw(initial);
    // v_writelane_b32 v0, vcc_lo/hi, 5.
    std::array<uint32_t, 2> words{gfx9() ? 0xd28a0000u : 0xd7610000u, src | (133u << 9)};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, lane),
                lane == 5 ? (src == 106 ? 0xcafebabeu : 0xdeadbeefu) : 0xa5000000u + lane);
    EXPECT_EQ(wf->vcc(), initial);
    // v_writelane_b32 v0, 42, vcc_lo/hi. Each encoding uses one scalar source.
    const uint64_t index = wf->wf_size() + 5u;
    wf->set_vcc_raw(src == 106 ? index : index << 32);
    words[1] = 170u | (src << 9);
    inst.reset(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      EXPECT_EQ(cu->read_vgpr(wf->vgpr_alloc().base, lane), lane == 5 ? 42u : 0xa5000000u + lane);
    EXPECT_EQ(wf->vcc(), src == 106 ? index : index << 32);
    EXPECT_EQ(wf->exec(), 0u);
  }
}

TEST_P(ReadlaneSelectorTest, LaneReadsRejectUnqualifiedScalarDestinations) {
  const bool rdna12 =
      GetParam() == ROCJITSU_CODE_ARCH_RDNA1 || GetParam() == ROCJITSU_CODE_ARCH_RDNA2;
  const uint32_t m0 = gfx9() || rdna12 ? 124u : 125u;
  // LLVM assembles EXEC destinations, but their write behavior is not qualified.
  // Keep them explicitly unsupported alongside the architectural M0 exclusion.
  for (uint32_t dst : {m0, 126u, 127u}) {
    SCOPED_TRACE(dst);
    const std::array<uint32_t, 2> readlane{(gfx9() ? 0xd2890000u : 0xd7600000u) | dst,
                                           256u | (133u << 9)};
    const std::array<uint32_t, 2> firstlane_e32{0x7e000500u | (dst << 17), 0u};
    const std::array<uint32_t, 2> firstlane_e64{(gfx9() ? 0xd1420000u : 0xd5820000u) | dst, 256u};
    EXPECT_TRUE(decode_fails(*decoder, readlane.data()));
    EXPECT_TRUE(decode_fails(*decoder, firstlane_e32.data()));
    EXPECT_TRUE(decode_fails(*decoder, firstlane_e64.data()));
  }
}

TEST(CdnaReadlaneTest, RestoresBothVccHalvesFromInactiveSpillLanes) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    amdgpu::GpuMemory mem("readlane_mem");
    amdgpu::L2Cache l2("readlane_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 102;
    cfg.vgprs_per_wf = 128;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("readlane", cfg, &mem, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, 102, 128);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(0);
    wf->set_vcc(0xaabbccdd11223344ull);
    cu->write_vgpr(wf->vgpr_alloc().base + 126, 33, 0x76543210u);
    cu->write_vgpr(wf->vgpr_alloc().base + 126, 34, 0xfedcba98u);
    // LLVM's tinygrad cumprod gradient restores VCC from VGPR spill lanes.
    const std::array<std::array<uint32_t, 2>, 2> words{{
        {0xd289006au, 0x0001437eu}, // v_readlane_b32 vcc_lo, v126, 33
        {0xd289006bu, 0x0001457eu}, // v_readlane_b32 vcc_hi, v126, 34
    }};
    for (unsigned i = 0; i < words.size(); ++i) {
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words[i].data()));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->disassemble(),
                i == 0 ? "v_readlane_b32 vcc_lo, v126, 33" : "v_readlane_b32 vcc_hi, v126, 34");
      ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
      EXPECT_EQ(wf->vcc(), i == 0 ? 0xaabbccdd76543210ull : 0xfedcba9876543210ull);
      EXPECT_EQ(wf->exec(), 0u);
    }
    wf->halt();
  }
}

} // namespace
