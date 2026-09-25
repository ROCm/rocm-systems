// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "async_mma_test_util.h"
#include "decode_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
#include "rocjitsu/vm/amdgpu/mma_admission.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <barrier>
#include <bit>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <vector>

namespace {
using namespace rocjitsu;
namespace mc = amdgpu::matrix_coexecution;

TEST(MmaAdmissionCacheTest, ReusesAcceptedAndRejectedPlansAcrossLoopsAndInvalidation) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  decoder->enable_pool();
  amdgpu::MmaAdmissionCache cache;
  amdgpu::GpuMemory memory("admission");
  amdgpu::InstructionCache icache;
  constexpr uint64_t pc = 0x480000;
  const auto a =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  const auto b =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3});
  for (unsigned i = 0; i != 2; ++i) {
    memory.write32(pc + 4 * i, a[i]);
    memory.write32(pc + 8 + 4 * i, b[i]);
  }
  memory.write32(pc + 16, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
  amdgpu::MmaAdmissionCache::Words first;
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
  for (unsigned i = 0; i != 1000; ++i)
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  EXPECT_EQ(cache.stats.decodes, 3u);
  EXPECT_EQ(cache.stats.plans, 1u);
  EXPECT_EQ(cache.stats.hits, 999u);
  icache.invalidate_all();
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
  EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  EXPECT_EQ(cache.stats.decodes, 3u);
  EXPECT_EQ(cache.stats.plans, 1u);
  EXPECT_EQ(cache.stats.validations, 1u);

  // A changed future word outside the initial fetch must invalidate the plan.
  memory.write32(pc + 16, 0xbf800001); // s_nop, changes the saved decode window.
  icache.invalidate_all();
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
  EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first), pc + 8);
  EXPECT_EQ(cache.stats.plans, 2u);
  EXPECT_GT(cache.stats.decodes, 3u);

  memory.write32(pc + 8, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
  icache.invalidate_all();
  icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
  EXPECT_FALSE(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first));
  const auto decodes = cache.stats.decodes;
  const auto plans = cache.stats.plans;
  for (unsigned i = 0; i != 1000; ++i)
    EXPECT_FALSE(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first));
  EXPECT_EQ(cache.stats.decodes, decodes);
  EXPECT_EQ(cache.stats.plans, plans);
}

TEST(MmaAdmissionCacheTest, RejectsDependenciesBoundsAndUnknownInstructions) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  constexpr uint64_t pc = 0x490000;
  for (unsigned hazard = 0; hazard != 5; ++hazard) {
    amdgpu::MmaAdmissionCache cache;
    amdgpu::GpuMemory memory("admission_hazard");
    amdgpu::InstructionCache icache;
    const auto a =
        cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                           {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
    const uint8_t dst = hazard == 2 ? 0 : hazard == 3 ? 64 : 96;
    const auto b = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                                      {.vdst = dst,
                                       .src0 = static_cast<uint16_t>(hazard == 1 ? 320 : 256),
                                       .src1 = 288,
                                       .src2 = static_cast<uint16_t>(256 + dst),
                                       .opsel_hi = 3});
    for (unsigned i = 0; i != 2; ++i) {
      memory.write32(pc + i * 4, a[i]);
      memory.write32(pc + 8 + i * 4, hazard == 4 ? 0xffffffffu : b[i]);
    }
    amdgpu::MmaAdmissionCache::Words first;
    icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first).has_value(),
              hazard == 0);
    if (hazard == 0) {
      EXPECT_FALSE(cache.inspect(*decoder, icache, memory, pc, 0, 96, false, first));
      EXPECT_FALSE(cache.inspect(*decoder, icache, memory, pc + amdgpu::GpuMemory::PAGE_SIZE - 8, 0,
                                 128, false, first));
    }
  }
}

TEST(MmaAdmissionCacheTest, KeepsWiderGroupsAndStopsAtANonAdjacentHazard) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  constexpr uint64_t pc = 0x4a0000;
  for (bool hazard : {false, true}) {
    amdgpu::MmaAdmissionCache cache;
    amdgpu::GpuMemory memory("admission_group");
    amdgpu::InstructionCache icache;
    for (unsigned i = 0; i != 4; ++i) {
      const uint8_t dst = 64 + 16 * i;
      const auto words =
          cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                             {.vdst = dst,
                              .src0 = static_cast<uint16_t>(hazard && i == 3 ? 320 : 256),
                              .src1 = 288,
                              .src2 = static_cast<uint16_t>(256 + dst),
                              .opsel_hi = 3});
      for (unsigned j = 0; j != 2; ++j)
        memory.write32(pc + i * 8 + j * 4, words[j]);
    }
    memory.write32(pc + 32, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
    amdgpu::MmaAdmissionCache::Words first;
    icache.fetch(memory, pc, reinterpret_cast<uint8_t *>(first.data()));
    EXPECT_EQ(cache.inspect(*decoder, icache, memory, pc, 0, 128, false, first),
              pc + (hazard ? 16 : 24));
  }
}
static_assert(HasAsyncMma<cdna5::Isa> && HasAsyncMma<cdna4::Isa>);
static_assert(!HasAsyncMma<cdna3::Isa> && !HasAsyncMma<rdna3::Isa> && !HasAsyncMma<rdna4::Isa>);
static_assert(
    std::is_same_v<
        decltype(&amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna4::Isa>::step),
        bool (amdgpu::ComputeUnitCore::*)()>);
static_assert(std::is_same_v<
              decltype(&amdgpu::IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna5::Isa>::step),
              bool (amdgpu::ComputeUnitCore::*)()>);
static_assert(amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL,
                                         cdna5::Isa>::supports_async_execution);
static_assert(
    !amdgpu::IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna5::Isa>::supports_async_execution);
static_assert(amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL,
                                         cdna4::Isa>::supports_async_execution);
static_assert(!amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL,
                                          cdna3::Isa>::supports_async_execution);
static_assert(!amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL,
                                          rdna4::Isa>::supports_async_execution);

TEST(MatrixCoexecutionTest, SelectsOnlySupportedLargeMatrixInstructions) {
  EXPECT_TRUE(mc::async_candidate("v_wmma_f32_16x16x64_fp8_fp8"));
  EXPECT_TRUE(mc::async_candidate("v_wmma_f32_16x16x128_bf8_bf8"));
  EXPECT_TRUE(mc::async_candidate("v_wmma_f32_32x16x128_f4"));
  EXPECT_TRUE(mc::async_candidate("v_wmma_f32_16x16x32_f16"));
  EXPECT_FALSE(mc::async_candidate("v_wmma_f32_16x16x128_f8f6f4"));
  EXPECT_FALSE(mc::async_candidate("v_add_f32"));
}

TEST(MatrixCoexecutionTest, BatchOverlapsEightCallbacksAndJoinsBeforeReturning) {
  struct Context {
    std::barrier<> rendezvous{8};
    std::atomic<unsigned> finished{0};
  } context;
  Instruction instruction("test", [](Instruction &, void *opaque) {
    auto &state = *static_cast<Context *>(opaque);
    state.rendezvous.arrive_and_wait();
    state.finished.fetch_add(1);
  });
  std::array<Instruction *, 8> instructions;
  instructions.fill(&instruction);
  mc::SharedPool pool(7);
  mma_test::execute_independent(pool, instructions, &context);
  EXPECT_EQ(context.finished.load(), 8u);
  mma_test::execute_independent(pool, instructions, &context);
  EXPECT_EQ(context.finished.load(), 16u);
}

// Compare actual CU issue with the ordinary generated callbacks, bit for bit.
// Explicit VM resources exercise asynchronous issue without environment controls.
TEST(MatrixCoexecutionTest, IssueMatchesSerialForLargeShapesAndHazards) {
  constexpr uint64_t pc = 0x200000;
  for (const auto opcode :
       {cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p, cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
        cdna5::kVWmmaF3232x16x128F4Vop3p}) {
    for (int hazard = 0; hazard != 4; ++hazard) {
      SCOPED_TRACE(hazard);
      amdgpu::GpuMemory memory("matrix_test_memory");
      amdgpu::L2Cache l2("matrix_test_l2");
      amdgpu::ComputeUnitCore::Config config{};
      config.async_resources = std::make_shared<mc::ExecutionResources>(4);
      config.arch = ROCJITSU_CODE_ARCH_CDNA5;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 106;
      config.vgprs_per_wf = 128;
      auto cu = amdgpu::ComputeUnitCore::create("matrix_test_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, pc, 106, 128);
      ASSERT_NE(wf, nullptr);
      wf->set_exec(0xFFFFFFFFu);
      const uint32_t base = wf->vgpr_alloc().base;
      const bool fp4 = opcode == cdna5::kVWmmaF3232x16x128F4Vop3p;
      const uint32_t ones = fp4 ? 0x22222222u : 0x38383838u;
      auto seed = [&] {
        for (uint32_t r = 0; r != 128; ++r)
          for (uint32_t lane = 0; lane != 32; ++lane)
            cu->write_vgpr(base + r, lane, r < 48 ? ones : std::bit_cast<uint32_t>(1.0f));
      };
      auto snapshot = [&] {
        std::vector<uint32_t> values;
        for (uint32_t r = 0; r != 128; ++r)
          for (uint32_t lane = 0; lane != 32; ++lane)
            values.push_back(cu->read_vgpr(base + r, lane));
        return values;
      };
      const auto a = cdna5::build_vop3p(
          opcode, {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
      const uint32_t dst = hazard == 2 ? 0 : hazard == 3 ? 64 : 96;
      const auto b =
          cdna5::build_vop3p(opcode, {.vdst = static_cast<uint8_t>(dst),
                                      .src0 = static_cast<uint16_t>(hazard == 1 ? 320 : 256),
                                      .src1 = 288,
                                      .src2 = static_cast<uint16_t>(256u + dst),
                                      .opsel_hi = 3});
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
      std::unique_ptr<Instruction> ai(decode_valid(*decoder, a.data()));
      std::unique_ptr<Instruction> bi(decode_valid(*decoder, b.data()));
      seed();
      ASSERT_TRUE(cu->execute_instruction(ai.get(), *wf).succeeded());
      ASSERT_TRUE(cu->execute_instruction(bi.get(), *wf).succeeded());
      const auto expected = snapshot();
      seed();
      for (uint32_t i = 0; i != 2; ++i) {
        memory.write32(pc + i * 4, a[i]);
        memory.write32(pc + 8 + i * 4, b[i]);
      }
      memory.write32(pc + 16, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
      const auto pairs = amdgpu::async_execution::stats.mma;
      for (int steps = 0; wf->pc < pc + 16 && steps != 2; ++steps)
        cu->step();
      EXPECT_EQ(wf->pc, pc + 16);
      EXPECT_EQ(snapshot(), expected);
      EXPECT_EQ(amdgpu::async_execution::stats.mma - pairs, hazard == 0 ? 1u : 0u);
    }
  }
}
TEST(MatrixCoexecutionTest, WiderBatchChecksNonAdjacentDependenciesAndAllocatesBeforePublication) {
  constexpr uint64_t pc = 0x210000;
  for (bool clocked : {false, true}) {
    for (bool hazard : {false, true}) {
      std::array<std::array<uint32_t, 2>, 3> words;
      for (unsigned i = 0; i != words.size(); ++i) {
        const auto dst = static_cast<uint8_t>(64 + 16 * i);
        const auto encoding =
            cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                               {.vdst = dst,
                                .src0 = static_cast<uint16_t>(i == 2 && hazard ? 320 : 256),
                                .src1 = 288,
                                .src2 = static_cast<uint16_t>(256 + dst),
                                .opsel_hi = 3});
        words[i] = {encoding[0], encoding[1]};
      }
      auto execute = [&](bool issue) {
        amdgpu::GpuMemory memory("wide_memory");
        amdgpu::L2Cache l2("wide_l2");
        amdgpu::ComputeUnitCore::Config config{};
        config.async_resources = std::make_shared<mc::ExecutionResources>(4);
        config.arch = ROCJITSU_CODE_ARCH_CDNA5;
        config.num_wf_slots = 1;
        config.sgprs_per_wf = 106;
        config.vgprs_per_wf = 128;
        auto cu = amdgpu::ComputeUnitCore::create("wide_cu", config, &memory, &l2,
                                                  clocked ? simdojo::ExecMode::CLOCKED
                                                          : simdojo::ExecMode::FUNCTIONAL);
        auto *wf = cu->dispatch_wf(0, pc, 106, 128);
        wf->set_exec(0xFFFFFFFFu);
        const uint32_t base = wf->vgpr_alloc().base;
        // Leave the output chunks lazy, including their zero accumulators.
        for (unsigned r = 0; r != 48; ++r)
          for (unsigned lane = 0; lane != 32; ++lane)
            cu->write_vgpr(base + r, lane, 0x01234567u ^ (r * 0x07654321u + lane * 0x01234567u));
        auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
        const auto submitted = amdgpu::async_execution::stats.mma;
        for (unsigned i = 0; i != words.size(); ++i) {
          if (issue) {
            memory.write32(pc + 8 * i, words[i][0]);
            memory.write32(pc + 8 * i + 4, words[i][1]);
          } else {
            std::unique_ptr<Instruction> inst(decode_valid(*decoder, words[i].data()));
            EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          }
        }
        if (issue) {
          memory.write32(pc + 24, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
          for (int step = 0; wf->pc < pc + 24 && step != 3; ++step)
            cu->step();
          EXPECT_EQ(wf->pc, pc + 24);
          const auto jobs = amdgpu::async_execution::stats.mma - submitted;
          if (clocked)
            EXPECT_EQ(jobs, 0u);
          else
            EXPECT_GT(jobs, 0u);
        }
        std::vector<uint32_t> output;
        for (unsigned r = 64; r != 112; ++r)
          for (unsigned lane = 0; lane != 32; ++lane)
            output.push_back(cu->read_vgpr(base + r, lane));
        return output;
      };
      const auto expected = execute(false);
      EXPECT_EQ(execute(true), expected);
    }
  }
}

TEST(AsyncInstructionQueueTest, FootprintResolvesInlineConstantsAndPackedHalfAliases) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  const auto encoding = cdna5::build_vop3(cdna5::kVAlignbitB32Vop3,
                                          {.vdst = 8, .src0 = 264, .src1 = 264, .src2 = 128 + 27});
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, encoding.data()));
  const auto access = amdgpu::async_execution::footprint(*inst, 16);
  ASSERT_TRUE(access);
  EXPECT_EQ(access->reads.count(), 1u);
  EXPECT_TRUE(access->reads.test(8));
  EXPECT_EQ(access->writes.count(), 1u);
  EXPECT_TRUE(access->writes.test(8));
  // Packed f16 selectors encode the upper half using bit 7 of the register
  // selector. Both halves must conflict through the same physical VGPR.
  cdna5::Operand low(16, cdna5::OperandType::OPR_VGPR, 7, false, true);
  cdna5::Operand high(16, cdna5::OperandType::OPR_VGPR, 135, false, true);
  ASSERT_TRUE(low.to_register_ref());
  EXPECT_EQ(low.to_register_ref(), high.to_register_ref());
}

TEST(AsyncInstructionQueueTest, InterleavedMemoryMmaAndRegisterReuseMatchSerial) {
  constexpr uint64_t pc = 0x230000, address = 0x300000;
  std::vector<uint32_t> program;
  auto append = [&](auto words) { program.insert(program.end(), words.begin(), words.end()); };
  append(
      cdna5::build_vglobal(cdna5::kGlobalLoadB128Vglobal, {.saddr = 0, .vdst = 0, .vaddr = 120}));
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128 + 1, .vdst = 112}));
  append(cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
  append(cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                            {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3}));
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 368, .vdst = 113}));
  append(cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                            {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3}));
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 352, .vdst = 114}));
  append(
      cdna5::build_vglobal(cdna5::kGlobalStoreB128Vglobal, {.saddr = 0, .vsrc = 64, .vaddr = 120}));
  // The store already captured its source; this reuse must remain safe.
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 64}));
  append(
      cdna5::build_vglobal(cdna5::kGlobalLoadB128Vglobal, {.saddr = 0, .vdst = 116, .vaddr = 120}));
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 372, .vdst = 115}));
  append(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
  auto execute = [&](bool issue) {
    amdgpu::GpuMemory memory("interleave_memory");
    amdgpu::L2Cache l2("interleave_l2");
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.async_resources = std::make_shared<mc::ExecutionResources>(4);
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 128;
    auto cu = amdgpu::ComputeUnitCore::create("interleave_cu", config, &memory, &l2);
    auto *wf = cu->dispatch_wf(0, pc, 106, 128);
    wf->set_exec(0xFFFFFFFFu);
    const auto base = wf->vgpr_alloc().base;
    cu->write_sgpr(wf->sgpr_alloc().base, address);
    cu->write_sgpr(wf->sgpr_alloc().base + 1, 0);
    for (unsigned reg = 0; reg != 128; ++reg)
      for (unsigned lane = 0; lane != 32; ++lane)
        cu->write_vgpr(base + reg, lane, reg < 48 ? 0x38383838 : 0);
    for (unsigned lane = 0; lane != 32; ++lane) {
      cu->write_vgpr(base + 120, lane, 16 * lane);
      for (unsigned elem = 0; elem != 4; ++elem)
        memory.write32(address + 16 * lane + elem * 4, 0x38383838);
    }
    const uint64_t end = pc + program.size() * 4;
    for (unsigned i = 0; i != program.size(); ++i)
      memory.write32(pc + i * 4, program[i]);
    memory.write32(end, cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff})[0]);
    if (issue) {
      for (unsigned step = 0; wf->pc < end && step != program.size(); ++step)
        cu->step();
      EXPECT_EQ(wf->pc, end);
    } else {
      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
      amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);
      for (unsigned i = 0; i != program.size();) {
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, program.data() + i));
        i += inst->size() / 4;
        EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        if (inst->is_memory_op())
          pipeline.issue(inst.release(), *wf);
      }
    }
    EXPECT_TRUE(wf->wait_counters().empty());
    std::vector<uint32_t> result;
    for (unsigned reg = 0; reg != 128; ++reg)
      for (unsigned lane = 0; lane != 32; ++lane)
        result.push_back(cu->read_vgpr(base + reg, lane));
    for (unsigned i = 0; i != 128; ++i)
      result.push_back(memory.read32(address + 4 * i));
    return result;
  };
  const auto submitted = amdgpu::async_execution::stats.mma;
  const auto retired = amdgpu::async_execution::stats.retired_mma;
  EXPECT_EQ(execute(true), execute(false));
  const auto offloads = amdgpu::async_execution::stats.mma - submitted;
  EXPECT_EQ(amdgpu::async_execution::stats.retired_mma - retired, offloads);
  EXPECT_GT(offloads, 0u);
}

TEST(MatrixCoexecutionTest, SharedPoolDoesNotWaitForBusyHelpersAndReusesThemAcrossIssuers) {
  mc::SharedPool pool(1);
  struct Context {
    std::binary_semaphore started{0};
    std::binary_semaphore unblock{0};
    std::atomic<unsigned> finished{0};
  } context;
  Instruction blocked("block", [](Instruction &, void *opaque) {
    auto &ctx = *static_cast<Context *>(opaque);
    ctx.started.release();
    ctx.unblock.acquire();
    ++ctx.finished;
  });
  Instruction increment(
      "increment", [](Instruction &, void *opaque) { ++static_cast<Context *>(opaque)->finished; });
  std::array<Instruction *, 2> first{&blocked, &increment};
  std::thread issuer([&] { mma_test::execute_independent(pool, first, &context); });
  context.started.acquire();
  std::array<Instruction *, 2> fallback{&increment, &increment};
  EXPECT_EQ(mma_test::execute_independent(pool, fallback, &context), 0u);
  // This call returned while the other issuer's helper is still blocked.
  EXPECT_GE(context.finished.load(), 2u);
  context.unblock.release();
  issuer.join();
  EXPECT_EQ(context.finished.load(), 4u);
  EXPECT_EQ(mma_test::execute_independent(pool, fallback, &context), 1u);
  EXPECT_EQ(context.finished.load(), 6u);
}

TEST(MatrixCoexecutionTest, SharedPoolReleasesClaimsAfterHelperAndInlineFailures) {
  mc::SharedPool pool(1);
  Instruction failure("failure", [](Instruction &, void *) {
    throw std::runtime_error("injected shared helper failure");
  });
  Instruction increment("increment", [](Instruction &, void *opaque) {
    ++*static_cast<std::atomic<unsigned> *>(opaque);
  });
  std::atomic<unsigned> finished{0};
  std::array<Instruction *, 2> instructions{&failure, &increment};
  EXPECT_THROW(mma_test::execute_independent(pool, instructions, &finished), std::runtime_error);
  EXPECT_EQ(finished.load(), 1u);
  std::swap(instructions[0], instructions[1]);
  EXPECT_THROW(mma_test::execute_independent(pool, instructions, &finished), std::runtime_error);
  EXPECT_EQ(finished.load(), 2u);
  instructions.fill(&increment);
  EXPECT_EQ(mma_test::execute_independent(pool, instructions, &finished), 1u);
  EXPECT_EQ(finished.load(), 4u);
}

TEST(MatrixCoexecutionTest, SharedPoolHandlesConcurrentIssuersAndZeroCapacity) {
  for (unsigned capacity : {0u, 1u, 4u}) {
    mc::SharedPool pool(capacity);
    struct Context {
      unsigned input = 0;
      std::array<unsigned, 8> output{};
    };
    // Disjoint non-atomic outputs exercise publication and completion ordering,
    // including changing an issuer's inputs after every joined batch.
    std::array<std::unique_ptr<Instruction>, 8> owned;
    std::array<Instruction *, 8> instructions;
    for (size_t i = 0; i != instructions.size(); ++i) {
      owned[i] = std::make_unique<Instruction>(
          "copy",
          [](Instruction &self, void *opaque) {
            auto &ctx = *static_cast<Context *>(opaque);
            ctx.output[self.src_loc()] = ctx.input + self.src_loc();
          },
          i);
      instructions[i] = owned[i].get();
    }
    std::atomic<bool> matched{true};
    std::array<std::thread, 8> issuers;
    for (size_t i = 0; i != issuers.size(); ++i)
      issuers[i] = std::thread([&, i] {
        Context ctx;
        for (unsigned iteration = 0; iteration != 100; ++iteration) {
          ctx.input = i * 100 + iteration;
          ctx.output.fill(0);
          mma_test::execute_independent(pool, instructions, &ctx);
          for (size_t j = 0; j != ctx.output.size(); ++j)
            if (ctx.output[j] != ctx.input + j)
              matched.store(false, std::memory_order_relaxed);
        }
      });
    for (auto &issuer : issuers)
      issuer.join();
    EXPECT_TRUE(matched.load());
  }
}

class QueueRetirementProbe final : public DynamicInstState {
public:
  QueueRetirementProbe(std::vector<unsigned> &retired, unsigned id)
      : retired_(retired), id_(id), issuer_(std::this_thread::get_id()) {}
  ~QueueRetirementProbe() override {
    EXPECT_EQ(std::this_thread::get_id(), issuer_);
    retired_.push_back(id_);
  }

private:
  std::vector<unsigned> &retired_;
  unsigned id_;
  std::thread::id issuer_;
};

TEST(AsyncInstructionQueueTest, IndependentCompletionReleasesOnlyItsOwnDependencies) {
  using Queue = amdgpu::AsyncInstructionQueue;
  mc::SharedPool pool(2);
  struct State {
    std::binary_semaphore started{0}, unblock{0};
    std::vector<unsigned> retired;
  } state;
  auto first = std::make_unique<Instruction>(
      "blocked",
      [](Instruction &, void *ctx) {
        auto &s = *static_cast<State *>(ctx);
        s.started.release();
        s.unblock.acquire();
      },
      0);
  auto second = std::make_unique<Instruction>("ready", [](Instruction &, void *) {}, 1);
  first->set_data(std::make_unique<QueueRetirementProbe>(state.retired, 0));
  second->set_data(std::make_unique<QueueRetirementProbe>(state.retired, 1));
  Queue queue(pool);
  amdgpu::async_execution::Access a, b, consumer;
  a.reads.set(0);
  a.writes.set(64);
  b.reads.set(0);
  b.writes.set(96);
  ASSERT_TRUE(queue.submit(first, &state, a));
  EXPECT_FALSE(first);
  state.started.acquire();
  EXPECT_TRUE(queue.submit(second, &state, b));
  EXPECT_FALSE(second);
  EXPECT_FALSE(pool.available());
  // This dependency must not join the still-blocked earlier instruction.
  consumer.reads.set(96);
  EXPECT_TRUE(queue.wait_conflicts(consumer));
  EXPECT_EQ(state.retired, std::vector<unsigned>({1}));
  EXPECT_EQ(queue.size(), 1u);
  EXPECT_TRUE(pool.available());
  EXPECT_FALSE(queue.wait_conflicts(consumer));
  consumer.reads.reset();
  consumer.writes.set(0);
  EXPECT_TRUE(a.conflicts(consumer)); // The first job still owns this input.
  state.unblock.release();
  queue.drain();
  EXPECT_EQ(state.retired, std::vector<unsigned>({1, 0}));
}

TEST(AsyncInstructionQueueTest, RejectsFullPoolWithoutTakingInstructionOwnership) {
  using Queue = amdgpu::AsyncInstructionQueue;
  mc::SharedPool pool(2);
  struct State {
    std::binary_semaphore started{0}, unblock{0}, second_done{0};
    std::vector<unsigned> retired;
  } state;
  auto first = std::make_unique<Instruction>(
      "blocked",
      [](Instruction &, void *ctx) {
        auto &s = *static_cast<State *>(ctx);
        s.started.release();
        s.unblock.acquire();
      },
      0);
  auto second = std::make_unique<Instruction>(
      "ready", [](Instruction &, void *ctx) { static_cast<State *>(ctx)->second_done.release(); },
      1);
  auto third = std::make_unique<Instruction>("fallback", [](Instruction &, void *) {}, 2);
  first->set_data(std::make_unique<QueueRetirementProbe>(state.retired, 0));
  second->set_data(std::make_unique<QueueRetirementProbe>(state.retired, 1));
  third->set_data(std::make_unique<QueueRetirementProbe>(state.retired, 2));
  Queue queue(pool);
  ASSERT_TRUE(queue.submit(first, &state, {}));
  state.started.acquire();
  EXPECT_TRUE(queue.submit(second, &state, {}));
  state.second_done.acquire();
  EXPECT_TRUE(state.retired.empty());
  EXPECT_EQ(queue.size(), 2u);
  EXPECT_FALSE(pool.available());
  auto *retained = third.get();
  EXPECT_FALSE(queue.submit(third, &state, {}));
  EXPECT_EQ(third.get(), retained);
  // Synchronous fallback is available even though the first job is blocked.
  third->execute(*third, &state);
  state.unblock.release();
  queue.drain();
  EXPECT_EQ(state.retired, std::vector<unsigned>({0, 1}));
  EXPECT_TRUE(pool.available());
  EXPECT_FALSE(queue.take_error());
  third.reset();
  EXPECT_EQ(state.retired, std::vector<unsigned>({0, 1, 2}));
}

struct ExtendedMmaCase {
  rj_code_arch_t arch;
  std::vector<uint32_t> words;
  uint32_t input;
  bool mfma;
  unsigned wmma_k;
  unsigned mfma_family;
};
std::vector<ExtendedMmaCase> extended_mma_cases() {
  std::vector<ExtendedMmaCase> cases;
  auto add = [&](rj_code_arch_t arch, auto words, uint32_t input, unsigned wmma_k,
                 unsigned mfma_family = 0) {
    cases.push_back(
        {arch, {words.begin(), words.end()}, input, mfma_family != 0, wmma_k, mfma_family});
  };
  for (auto opcode : {cdna5::kVWmmaF3216x16x32F16Vop3p, cdna5::kVWmmaF3216x16x32Bf16Vop3p})
    add(ROCJITSU_CODE_ARCH_CDNA5,
        cdna5::build_vop3p(opcode,
                           {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3}),
        opcode == cdna5::kVWmmaF3216x16x32F16Vop3p ? 0x3c003c00u : 0x3f803f80u, 32);
  for (unsigned format = 0; format != 5; ++format)
    for (bool inline_scale : {false, true})
      add(ROCJITSU_CODE_ARCH_CDNA5,
          mma_test::make_cdna5_wmma_scale_words(format, format, inline_scale ? 128 : 448,
                                                inline_scale ? 128 : 449),
          format == 4 ? 0x22222222u : 0x38383838u, 128);
  for (auto opcode : {rdna4::kVWmmaF3216x16x16F16Vop3p, rdna4::kVWmmaF3216x16x16Bf16Vop3p})
    add(ROCJITSU_CODE_ARCH_RDNA4,
        rdna4::build_vop3p(opcode,
                           {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3}),
        opcode == rdna4::kVWmmaF3216x16x16F16Vop3p ? 0x3c003c00u : 0x3f803f80u, 16);
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4})
    for (auto opcode : {cdna4::kVMfmaF3232x32x42bF16Vop3pMfma,
                        cdna4::kVMfmaF3216x16x44bF16Vop3pMfma, cdna4::kVMfmaF324x4x416bF16Vop3pMfma,
                        cdna4::kVMfmaF3232x32x8F16Vop3pMfma, cdna4::kVMfmaF3216x16x16F16Vop3pMfma})
      for (uint8_t acc : {0, 1})
        add(arch,
            cdna4::build_vop3p_mfma(
                opcode, {.vdst = 64, .acc_cd = acc, .src0 = 256, .src1 = 288, .src2 = 320}),
            0x3c003c00, 0,
            opcode == cdna4::kVMfmaF3232x32x8F16Vop3pMfma ||
                    opcode == cdna4::kVMfmaF3216x16x16F16Vop3pMfma
                ? 4
                : 1);
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4})
    for (auto opcode :
         {cdna4::kVMfmaF3232x32x12bF32Vop3pMfma, cdna4::kVMfmaF3216x16x14bF32Vop3pMfma})
      for (uint8_t acc : {0, 1})
        add(arch,
            cdna4::build_vop3p_mfma(
                opcode, {.vdst = 64, .acc_cd = acc, .src0 = 256, .src1 = 288, .src2 = 320}),
            0x3f800000, 0, 1);
  for (auto opcode : {cdna4::kVMfmaF3232x32x16F16Vop3pMfma, cdna4::kVMfmaF3216x16x32F16Vop3pMfma})
    for (uint8_t acc : {0, 1})
      add(ROCJITSU_CODE_ARCH_CDNA4,
          cdna4::build_vop3p_mfma(
              opcode, {.vdst = 64, .acc_cd = acc, .src0 = 256, .src1 = 288, .src2 = 320}),
          0x3c003c00, 0, 4);
  // Ordinary FP8/BF8, including mixed inputs and both accumulator banks.
  for (unsigned opcode = 112; opcode != 120; ++opcode)
    for (uint8_t acc : {0, 1})
      add(ROCJITSU_CODE_ARCH_CDNA4,
          cdna4::build_vop3p_mfma(
              opcode, {.vdst = 64, .acc_cd = acc, .src0 = 256, .src1 = 288, .src2 = 320}),
          0x38383838, 0, 8);
  // The unscaled mixed-format opcodes have the same register-only contract.
  for (unsigned opcode : {45, 46})
    for (uint8_t format = 0; format != 5; ++format)
      for (uint8_t acc : {0, 1})
        add(ROCJITSU_CODE_ARCH_CDNA4,
            cdna4::build_vop3p_mfma(opcode, {.vdst = 64,
                                             .cbsz = format,
                                             .acc_cd = acc,
                                             .src0 = 256,
                                             .src1 = 288,
                                             .src2 = 320,
                                             .blgp = format}),
            0x38383838, 0, 8);
  for (unsigned opcode : {45, 46})
    for (bool inline_scale : {false, true})
      add(ROCJITSU_CODE_ARCH_CDNA4,
          mma_test::make_cdna4_mfma_scale_words(opcode, 1, inline_scale ? 242 : 448,
                                                inline_scale ? 242 : 449, 0, 0, 4, 4, 64, 256, 288,
                                                320),
          0x22222222, 0, 2);
  return cases;
}

TEST(AsyncInstructionQueueTest, SmallerWmmaAndMultiBlockMfmaMatchSerialAcrossRegisterHazards) {
  // Unsupported targets and families must remain inline.
  // Eligibility is recorded with each case instead of reusing the candidate
  // predicate: dropping an opcode from that predicate must fail this test.
  constexpr uint64_t pc = 0x250000;
  for (const auto &c : extended_mma_cases()) {
    auto decoder = Decoder::create(c.arch);
    std::unique_ptr<Instruction> first(decode_valid(*decoder, c.words.data()));
    SCOPED_TRACE(first->mnemonic());
    SCOPED_TRACE(c.arch);
    const bool expect_async = c.mfma
                                  ? c.arch == ROCJITSU_CODE_ARCH_CDNA4 && (14u & c.mfma_family) != 0
                                  : c.arch == ROCJITSU_CODE_ARCH_CDNA5 && c.wmma_k >= 32;
    const size_t suffix = c.words.size() - 2;
    const bool acc = c.mfma && ((c.words[suffix] >> 15) & 1);
    for (unsigned hazard = 0; hazard != 5; ++hazard) {
      SCOPED_TRACE(hazard);
      auto second_words = c.words;
      const unsigned dst = hazard == 2 ? 0 : hazard == 3 ? 64 : hazard == 4 ? 192 : 128;
      const unsigned src0 = hazard == 1 ? 320 : 256;
      second_words[suffix] = (second_words[suffix] & ~255u) | dst;
      second_words[suffix + 1] =
          (second_words[suffix + 1] & ~(511u | (511u << 18))) | src0 | ((256u + dst) << 18);
      if (acc && hazard == 1)
        second_words[suffix + 1] |= 1u << 27; // Read the first MMA's AccVGPR output.
      std::unique_ptr<Instruction> second(decode_valid(*decoder, second_words.data()));
      const auto access = amdgpu::async_execution::footprint(*first, 256, c.mfma);
      ASSERT_TRUE(access);
      EXPECT_TRUE(access->writes.test((acc ? 256 : 0) + 64));
      if (c.words.size() == 4 && (c.words[1] & 511) == 448) {
        EXPECT_TRUE(access->reads.test(192));
      }
      amdgpu::GpuMemory memory("extended_matrix_memory");
      amdgpu::L2Cache l2("extended_matrix_l2");
      amdgpu::ComputeUnitCore::Config config{};
      config.async_resources = std::make_shared<mc::ExecutionResources>(4);
      config.arch = c.arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 106;
      config.vgprs_per_wf = 256;
      auto cu = amdgpu::ComputeUnitCore::create("extended_matrix_cu", config, &memory, &l2);
      auto *wf = cu->dispatch_wf(0, pc, 106, 256);
      ASSERT_NE(wf, nullptr);
      const unsigned wave_size = c.mfma ? 64 : 32;
      const unsigned registers = c.mfma ? 512 : 256;
      wf->set_exec(c.mfma ? ~uint64_t{0} : 0xffffffff);
      const auto base = wf->vgpr_alloc().base;
      auto seed = [&] {
        for (unsigned reg = 0; reg != registers; ++reg)
          for (unsigned lane = 0; lane != wave_size; ++lane)
            cu->write_vgpr(base + reg, lane,
                           reg < 48                   ? c.input
                           : reg == 192 || reg == 193 ? 0x7f7f7f7fu
                                                      : 0);
      };
      auto snapshot = [&] {
        std::vector<uint32_t> result;
        for (unsigned reg = 0; reg != registers; ++reg)
          for (unsigned lane = 0; lane != wave_size; ++lane)
            result.push_back(cu->read_vgpr(base + reg, lane));
        return result;
      };
      seed();
      ASSERT_TRUE(cu->execute_instruction(first.get(), *wf).succeeded());
      ASSERT_TRUE(cu->execute_instruction(second.get(), *wf).succeeded());
      const auto expected = snapshot();
      seed();
      std::vector<uint32_t> program = c.words;
      program.insert(program.end(), second_words.begin(), second_words.end());
      for (size_t i = 0; i != program.size(); ++i)
        memory.write32(pc + 4 * i, program[i]);
      const auto end = pc + 4 * program.size();
      const auto branch = c.mfma ? cdna4::build_sopp(cdna4::kSBranchSopp, {.simm16 = 0xffff})
                          : c.arch == ROCJITSU_CODE_ARCH_RDNA4
                              ? rdna4::build_sopp(rdna4::kSBranchSopp, {.simm16 = 0xffff})
                              : cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0xffff});
      memory.write32(end, branch[0]);
      const auto mma_before = amdgpu::async_execution::stats.mma;
      const auto windows_before = amdgpu::async_execution::stats.windows;
      const auto retired_before = amdgpu::async_execution::stats.retired_mma;
      for (unsigned steps = 0; wf->pc < end && steps != 3; ++steps)
        cu->step();
      EXPECT_EQ(wf->pc, end);
      EXPECT_EQ(snapshot(), expected);
      const auto &after = amdgpu::async_execution::stats;
      const auto offloads = after.mma - mma_before;
      EXPECT_EQ(after.retired_mma - retired_before, offloads);
      if (!expect_async) {
        EXPECT_EQ(offloads, 0u);
        EXPECT_EQ(after.windows - windows_before, 0u);
      } else if (hazard == 0) {
        // Admission may reject dependent pairs, but this pair is independent
        // for every shape, register bank, and scale representation above.
        EXPECT_GT(offloads, 0u);
        EXPECT_GT(after.windows - windows_before, 0u);
      }
    }
  }
}

} // namespace
