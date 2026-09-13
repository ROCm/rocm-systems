// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/matrix_coexecution.h"
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

TEST(MatrixCoexecutionTest, RejectsAllRegisterHazardsButAllowsSharedInputs) {
  const mc::Footprint a{{64, 16}, {{{0, 16}, {32, 8}, {64, 16}}}};
  mc::Footprint b{{96, 16}, {{{0, 16}, {32, 8}, {96, 16}}}};
  EXPECT_TRUE(mc::independent(a, b));
  b.inputs[0] = {79, 8};
  EXPECT_FALSE(mc::independent(a, b)); // RAW at the final register.
  b.inputs[0] = {0, 16};
  b.output = {15, 16};
  EXPECT_FALSE(mc::independent(a, b)); // WAR at the final input register.
  b.output = {79, 16};
  EXPECT_FALSE(mc::independent(a, b)); // WAW at the final output register.
  b.output = {80, 16};
  EXPECT_TRUE(mc::independent(a, b));
}

TEST(MatrixCoexecutionTest, SelectsOnlySupportedLargeMatrixInstructions) {
  EXPECT_TRUE(mc::candidate("v_wmma_f32_16x16x64_fp8_fp8"));
  EXPECT_TRUE(mc::candidate("v_wmma_f32_16x16x128_bf8_bf8"));
  EXPECT_TRUE(mc::candidate("v_wmma_f32_32x16x128_f4"));
  EXPECT_FALSE(mc::candidate("v_wmma_f32_16x16x32_f16"));
  EXPECT_FALSE(mc::candidate("v_wmma_f32_16x16x128_f8f6f4"));
  EXPECT_FALSE(mc::candidate("v_add_f32"));
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
  mc::execute_private_batch(instructions, &context);
  EXPECT_EQ(context.finished.load(), 8u);
  mc::execute_private_batch(instructions, &context);
  EXPECT_EQ(context.finished.load(), 16u);
}

TEST(MatrixCoexecutionTest, HelperJoinsBothCallbacksWhenOneThrows) {
  Instruction a("throw",
                [](Instruction &, void *) { throw std::runtime_error("injected helper failure"); });
  Instruction b("finish", [](Instruction &, void *opaque) {
    static_cast<std::atomic<bool> *>(opaque)->store(true);
  });
  std::atomic<bool> finished{false};
  mc::Helper helper;
  EXPECT_THROW(helper.execute_pair(a, b, &finished), std::runtime_error);
  EXPECT_TRUE(finished.load());
  finished.store(false);
  EXPECT_THROW(helper.execute_pair(b, a, &finished), std::runtime_error);
  EXPECT_TRUE(finished.load());
}

// Compare actual CU issue with the ordinary generated callbacks, bit for bit.
// The process environment selects scan, serial-batch or parallel-batch mode.
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
      const auto pairs = mc::stats.batches;
      for (int steps = 0; wf->pc < pc + 16 && steps != 2; ++steps)
        cu->step();
      EXPECT_EQ(wf->pc, pc + 16);
      EXPECT_EQ(snapshot(), expected);
      EXPECT_EQ(mc::stats.batches - pairs,
                mc::mode() >= 2 && mc::width() > 1 && hazard == 0 ? 1u : 0u);
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
        const auto batches = mc::stats.batches;
        const auto triples = mc::stats.widths[3];
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
          for (int step = 0; wf->pc < pc + 24 && step != 3; ++step)
            cu->step();
          EXPECT_EQ(wf->pc, pc + 24);
          const bool batches_enabled = !clocked && mc::mode() >= 2 && mc::width() > 1;
          EXPECT_EQ(mc::stats.batches - batches, batches_enabled ? 1u : 0u);
          EXPECT_EQ(mc::stats.widths[3] - triples,
                    batches_enabled && mc::width() >= 3 && !hazard ? 1u : 0u);
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
  std::thread issuer([&] { pool.execute(first, &context); });
  context.started.acquire();
  std::array<Instruction *, 2> fallback{&increment, &increment};
  const auto full = mc::stats.shared_full;
  pool.execute(fallback, &context);
  EXPECT_EQ(mc::stats.shared_full - full, 1u);
  // This call returned while the other issuer's helper is still blocked.
  EXPECT_GE(context.finished.load(), 2u);
  context.unblock.release();
  issuer.join();
  EXPECT_EQ(context.finished.load(), 4u);
  const auto submitted = mc::stats.shared_submitted;
  pool.execute(fallback, &context);
  EXPECT_EQ(mc::stats.shared_submitted - submitted, 1u);
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
  EXPECT_THROW(pool.execute(instructions, &finished), std::runtime_error);
  EXPECT_EQ(finished.load(), 1u);
  std::swap(instructions[0], instructions[1]);
  EXPECT_THROW(pool.execute(instructions, &finished), std::runtime_error);
  EXPECT_EQ(finished.load(), 2u);
  instructions.fill(&increment);
  const auto submitted = mc::stats.shared_submitted;
  pool.execute(instructions, &finished);
  EXPECT_EQ(mc::stats.shared_submitted - submitted, 1u);
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
          pool.execute(instructions, &ctx);
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

} // namespace
