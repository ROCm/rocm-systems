// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_plan_test.cpp
/// @brief Straight-line MMA plan scheduling: profitability, placement and chaining legality.

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/mma_plan.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
namespace plan = rocjitsu::amdgpu::mma_plan;

constexpr uint64_t kPc = 0x10000;

struct Program {
  std::vector<uint32_t> words;
  // f16 16x16x32: A and B are four VGPRs, C/D four VGPRs (acc < 48) or AccVGPRs (acc >= 48).
  void mfma(unsigned acc, unsigned a = 0, unsigned b = 16) {
    const bool use_acc = acc >= 48;
    const uint8_t reg = uint8_t(use_acc ? 4 * (acc - 48) : 64 + 4 * acc);
    const auto w = cdna4::build_vop3p_mfma(
        cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
        {.vdst = reg, .acc_cd = uint8_t(use_acc ? 1 : 0), .src0 = uint16_t(256 + a),
         .src1 = uint16_t(256 + b), .src2 = uint16_t(256 + reg)});
    words.insert(words.end(), w.begin(), w.end());
  }
  void ds_read_b128(unsigned vdst) {
    const auto w = cdna4::build_ds(cdna4::kDsReadB128, {.addr = 200, .vdst = uint8_t(vdst)});
    words.insert(words.end(), w.begin(), w.end());
  }
  void ds_write_b128(unsigned data) {
    const auto w = cdna4::build_ds(cdna4::kDsWriteB128, {.addr = 200, .data0 = uint8_t(data)});
    words.insert(words.end(), w.begin(), w.end());
  }
  void buffer_load_dwordx4(unsigned vdata) {
    const auto w = cdna4::build_mubuf(cdna4::kBufferLoadDwordx4Mubuf,
                                      {.offen = 1, .vaddr = 201, .vdata = uint8_t(vdata), .srsrc = 4});
    words.insert(words.end(), w.begin(), w.end());
  }
  void s_add_u32_m0() {
    const auto w = cdna4::build_sop2(cdna4::kSAddU32Sop2, {.ssrc0 = 4, .ssrc1 = 129, .sdst = 124});
    words.insert(words.end(), w.begin(), w.end());
  }
  unsigned helper_members() const {
    unsigned n = 0;
    for (const auto &step : compiled.steps)
      if (step.kind == plan::StepKind::Offload)
        n += step.members;
    return n;
  }
  plan::Plan compiled;
  plan::Plan compile() {
    words.push_back(cdna4::build_sopp(cdna4::kSBranchSopp, {.simm16 = 0xffff})[0]);
    words.resize(words.size() + 4, 0);
    auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna4);
    plan::Plan result;
    uint64_t decodes = 0;
    const auto fetch = [&](uint64_t address, plan::Plan::Words &out) {
      const size_t index = (address - entry_pc) / 4;
      for (size_t i = 0; i != out.size(); ++i)
        out[i] = index + i < words.size() ? words[index + i] : 0;
      return true;
    };
    plan::compile(result, *decoder, fetch, entry_pc, 256, true, decodes, cross_block, pack_limit,
                  min_chains, min_weight);
    return result;
  }
  uint64_t entry_pc = kPc;
  bool cross_block = false;
  unsigned pack_limit = plan::batch_pack_limit();
  unsigned min_chains = plan::batch_min_chains();
  unsigned min_weight = plan::min_job_weight();
  const plan::Plan &build() { return compiled = compile(); }
};

unsigned count(const plan::Plan &p, plan::StepKind kind) {
  unsigned n = 0;
  for (const auto &step : p.steps)
    n += step.kind == kind;
  return n;
}

TEST(MmaPlanTest, SerialChainIsRejectedEvenWhenSplitAtTheJobLimit) {
  Program p;
  for (unsigned i = 0; i != 16; ++i)
    p.mfma(0);
  const auto result = p.compile();
  EXPECT_EQ(result.steps.size(), 16u);
  EXPECT_FALSE(result.accept);
}

TEST(MmaPlanTest, EightAccumulatorsKeepEveryPairingWithoutEarlyJoins) {
  Program p;
  for (unsigned k = 0; k != 2; ++k)
    for (unsigned acc = 0; acc != 8; ++acc)
      p.mfma(acc);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  // Eight helper jobs of two same-accumulator MFMAs; at replay a job that
  // finds no helper runs on the issuer.
  EXPECT_EQ(count(result, plan::StepKind::Offload), 8u);
  EXPECT_EQ(count(result, plan::StepKind::Chained), 8u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
  for (const auto &step : result.steps) {
    EXPECT_EQ(step.wait, 0u) << "no step may wait for a helper inside the block";
    if (step.kind == plan::StepKind::Offload)
      EXPECT_EQ(step.members, 2u);
  }
}

TEST(MmaPlanTest, EveryIndependentChainIsItsOwnJob) {
  Program p;
  p.pack_limit = 1;
  for (unsigned acc = 0; acc != 16; ++acc)
    p.mfma(acc);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  unsigned helper_members = 0, jobs = 0;
  for (const auto &step : result.steps) {
    EXPECT_EQ(step.wait, 0u);
    if (step.kind == plan::StepKind::Offload) {
      ++jobs;
      EXPECT_LE(step.members, plan::kMaxChain);
      helper_members += step.members;
    }
  }
  EXPECT_EQ(jobs, 16u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
  EXPECT_EQ(helper_members, 16u);
}

TEST(MmaPlanTest, ChainsPastTheJobLimitAppendToRunningJobs) {
  Program p;
  p.pack_limit = 1;
  for (unsigned acc = 0; acc != 72; ++acc)
    p.mfma(acc);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  ASSERT_EQ(result.steps.size(), 72u);
  unsigned jobs = 0, members = 0;
  for (const auto &step : result.steps)
    if (step.kind == plan::StepKind::Offload) {
      ++jobs;
      members += step.members;
    }
  EXPECT_EQ(jobs, plan::kMaxJobs);
  EXPECT_EQ(members, 72u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
}

TEST(MmaPlanTest, SuccessorIsChainedAcrossAnLdsReadOfOtherRegisters) {
  // LDS reads cannot stall or fault, so a same-accumulator successor whose
  // factors are already live may run on its producer's helper before them.
  Program p;
  p.mfma(0);
  p.mfma(1);
  p.ds_read_b128(40);
  p.ds_read_b128(44);
  p.mfma(0);
  p.mfma(1);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  ASSERT_EQ(result.steps.size(), 6u);
  EXPECT_EQ(result.steps[2].kind, plan::StepKind::Lds);
  EXPECT_EQ(result.steps[4].kind, plan::StepKind::Chained);
  EXPECT_EQ(result.steps[5].kind, plan::StepKind::Chained);
  EXPECT_EQ(result.steps[0].members, 2u);
  EXPECT_EQ(result.steps[1].members, 2u);
}

TEST(MmaPlanTest, SuccessorIsNotChainedAcrossVectorMemory) {
  // A buffer load may retry or fault and re-enter the block, so nothing runs ahead of it.
  Program p;
  p.mfma(0);
  p.mfma(1);
  p.buffer_load_dwordx4(40);
  p.mfma(0);
  p.mfma(1);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  ASSERT_EQ(result.steps.size(), 5u);
  EXPECT_EQ(result.steps[2].kind, plan::StepKind::Ordinary);
  EXPECT_NE(result.steps[3].kind, plan::StepKind::Chained);
  EXPECT_NE(result.steps[4].kind, plan::StepKind::Chained);
  // The successors join their producers before accumulating.
  EXPECT_NE(result.steps[3].wait, 0u);
  EXPECT_NE(result.steps[4].wait, 0u);
}

TEST(MmaPlanTest, LdsWritesAndScalarM0UpdatesStayInsideTheBlock) {
  Program p;
  p.mfma(0);
  p.ds_write_b128(40);
  p.s_add_u32_m0();
  p.mfma(1);
  const auto result = p.compile();
  ASSERT_EQ(result.steps.size(), 4u);
  EXPECT_EQ(result.steps[1].kind, plan::StepKind::Lds);
  EXPECT_EQ(result.steps[2].kind, plan::StepKind::Alu);
  EXPECT_TRUE(result.accept);
}

TEST(MmaPlanTest, LdsReadsBetweenIndependentChainsDoNotEndJobs) {
  // Ten independent accumulators interleaved with LDS reads of other registers
  // all hand off; the reads stay issuer steps inside the block.
  Program p;
  p.pack_limit = 1;
  for (unsigned acc = 0; acc != 10; ++acc) {
    p.mfma(acc);
    p.ds_read_b128(40 + 4 * (acc % 4));
  }
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  unsigned jobs = 0, members = 0;
  for (const auto &step : result.steps)
    if (step.kind == plan::StepKind::Offload) {
      ++jobs;
      members += step.members;
    }
  EXPECT_EQ(jobs, 10u);
  EXPECT_EQ(members, 10u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
  EXPECT_EQ(count(result, plan::StepKind::Lds), 10u);
  EXPECT_EQ(count(result, plan::StepKind::Ordinary), 0u);
}

TEST(MmaPlanTest, SuccessorWhoseFactorIsLoadedLaterIsNotChained) {
  // acc0 K=0, then an LDS load of the next B tile, then acc0 K=1 reading it:
  // K=1 cannot run at K=0's submission. Independent acc1 work keeps the block profitable.
  Program p;
  p.mfma(0);
  p.mfma(1);
  p.ds_read_b128(32);
  p.mfma(0, 0, 32);
  p.mfma(1, 0, 32);
  const auto result = p.compile();
  ASSERT_GE(result.steps.size(), 5u);
  EXPECT_NE(result.steps[3].kind, plan::StepKind::Chained);
  EXPECT_NE(result.steps[4].kind, plan::StepKind::Chained);
}
TEST(MmaPlanTest, TailJobsSnapshotAllFactorsForTheNextBlock) {
  // Eight independent accumulators, one MFMA each, then a long run of LDS reads so
  // that only the last jobs fall inside the tail window.
  Program p;
  p.cross_block = true;
  for (unsigned acc = 0; acc != 8; ++acc)
    p.mfma(acc, 0, 16);
  for (unsigned i = 0; i != 20; ++i)
    p.ds_read_b128(40 + 4 * (i % 4));
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  // Steps 0..7 are the heads; the block has 28 steps, so heads at steps >= 12
  // are tail jobs. None of these are, so no snapshot is flagged...
  for (unsigned k = 0; k != 8; ++k)
    EXPECT_FALSE(result.steps[k].snapshots) << k;
  // ...whereas with the reads first, every head is in the tail.
  Program q;
  for (unsigned i = 0; i != 20; ++i)
    q.ds_read_b128(40 + 4 * (i % 4));
  for (unsigned acc = 0; acc != 8; ++acc)
    q.mfma(acc, 0, 16);
  // A plan starts at a candidate MMA, so the reads are not part of this block;
  // place the MFMAs after a leading MFMA-free prefix is impossible here. Use a
  // second independent block instead: MFMAs then a few reads, all in the tail.
  Program r;
  r.cross_block = true;
  for (unsigned acc = 0; acc != 8; ++acc)
    r.mfma(acc, 0, 16);
  r.ds_read_b128(40);
  const auto tail = r.compile();
  ASSERT_TRUE(tail.accept);
  for (unsigned k = 0; k != 8; ++k) {
    EXPECT_TRUE(tail.steps[k].snapshots) << k;
    EXPECT_EQ(tail.steps[k].snapshot[0].a_regs, 4u);
    EXPECT_EQ(tail.steps[k].snapshot[0].b_regs, 4u);
    // Snapshotted factors leave the job's read set; the result stays in its write set.
    EXPECT_FALSE(tail.job_accesses[tail.steps[k].job].reads.test(0));
    EXPECT_TRUE(tail.job_accesses[tail.steps[k].job].writes.test(64 + 4 * k));
  }
  // First-touch index: register 64 (acc0 result) is first touched at step 0,
  // v40 is first written by the trailing read.
  EXPECT_EQ(tail.first_access[64], 0u);
  EXPECT_EQ(tail.first_write[40], 8u);
  EXPECT_EQ(tail.first_write[64], 0u);
  EXPECT_TRUE(tail.footprint.writes.test(64 + 28));
}

TEST(MmaPlanTest, ScalarOnlyEncodingsNeedNoRegisterCheck) {
  EXPECT_TRUE(plan::scalar_only_encoding(cdna4::build_sopp(cdna4::kSBranchSopp, {.simm16 = 1})[0]));
  EXPECT_TRUE(plan::scalar_only_encoding(cdna4::build_sopp(cdna4::kSWaitcntSopp, {.simm16 = 0})[0]));
  EXPECT_TRUE(plan::scalar_only_encoding(
      cdna4::build_sop2(cdna4::kSAddU32Sop2, {.ssrc0 = 4, .ssrc1 = 129, .sdst = 124})[0]));
  EXPECT_TRUE(plan::scalar_only_encoding(
      cdna4::build_sopk(cdna4::kSMovkI32Sopk, {.simm16 = 7, .sdst = 4})[0]));
  EXPECT_FALSE(plan::scalar_only_encoding(
      cdna4::build_ds(cdna4::kDsReadB128, {.addr = 200, .vdst = 40})[0]));
  EXPECT_FALSE(plan::scalar_only_encoding(
      cdna4::build_vop1(cdna4::kVExpF32Vop1, {.src0 = 256 + 3, .vdst = 5})[0]));
  EXPECT_FALSE(plan::scalar_only_encoding(cdna4::build_mubuf(
      cdna4::kBufferLoadDwordx4Mubuf, {.offen = 1, .vaddr = 201, .vdata = 40, .srsrc = 4})[0]));
}

TEST(MmaPlanTest, FirstTouchIndexUsesTheJobHeadForChainedMembers) {
  // acc0 K1 (step 3) is bundled onto acc0's job at step 0 and reads v[32:35],
  // which nothing else touches: a running job writing v32 must be joined at
  // step 0, where the member actually executes, not at step 3.
  Program p;
  p.mfma(0, 0, 16);
  p.mfma(1, 0, 16);
  p.ds_read_b128(40);
  p.mfma(0, 0, 32);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  ASSERT_EQ(result.steps[3].kind, plan::StepKind::Chained);
  EXPECT_EQ(result.first_access[32], 0u);
  EXPECT_EQ(result.first_access[40], 2u);
  EXPECT_EQ(result.first_write[40], 2u);
  EXPECT_EQ(result.first_access[16], 0u);
}

struct MockJobInstruction : Instruction {
  explicit MockJobInstruction(const char *name, ExecuteFn fn) : Instruction(name, fn) {}
};

plan::Plan make_synthetic_job_plan(unsigned read_reg, unsigned write_reg, Instruction *inst) {
  plan::Plan p;
  p.accept = true;
  p.first_access.fill(plan::Plan::kNoStep);
  p.first_write.fill(plan::Plan::kNoStep);
  plan::Step step;
  step.kind = plan::StepKind::Offload;
  step.job = 0;
  step.offset = 8;
  step.members = 1;
  step.inst = inst;
  step.chain[0] = inst;
  step.member_step[0] = 0;
  p.steps.push_back(step);
  plan::Access access;
  access.reads.set(read_reg);
  access.writes.set(write_reg);
  p.job_accesses.push_back(access);
  p.footprint = access;
  p.first_access[read_reg] = 0;
  p.first_access[write_reg] = 0;
  p.first_write[write_reg] = 0;
  return p;
}

TEST(MmaPlanTest, ReapFinishedExceptFreesSiblingHelpersAndPreservesFailure) {
  matrix_coexecution::SharedPool pool(1);
  plan::PlanCache cache;
  std::atomic<bool> wave0_done{false};
  std::atomic<bool> wave1_done{false};
  MockJobInstruction fail_inst("mock_fail_mfma", [](Instruction &, void *ctx) {
    if (helper_error_sink)
      *helper_error_sink = {1u, "synthetic failure"};
    static_cast<std::atomic<bool> *>(ctx)->store(true, std::memory_order_release);
  });
  MockJobInstruction ok_inst("mock_ok_mfma", [](Instruction &, void *ctx) {
    static_cast<std::atomic<bool> *>(ctx)->store(true, std::memory_order_release);
  });
  const plan::Plan p0 = make_synthetic_job_plan(0, 64, &fail_inst);
  const plan::Plan p1 = make_synthetic_job_plan(16, 68, &ok_inst);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;

  auto &w0 = cache.pending(0, pool);
  (void)w0.begin_block(p0, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait, filtered);
  const auto no_fill = [](matrix_coexecution::SnapshotBuffer &) { return nullptr; };
  ASSERT_TRUE(w0.submit(0, p0.steps[0], &wave0_done, no_fill));

  // Wait for wave 0's helper to finish while wave 0 still holds the slot.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!wave0_done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_TRUE(wave0_done.load(std::memory_order_acquire));

  // Creating wave 1 may reallocate cache.pending_; re-fetch wave 0 via pending_if_any below.
  auto &w1 = cache.pending(1, pool);
  (void)w1.begin_block(p1, kPc + 0x100, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait,
                       filtered);
  bool submitted = false;
  while (!submitted && std::chrono::steady_clock::now() < deadline) {
    submitted =
        w1.submit(0, p1.steps[0], &wave1_done, no_fill, [&] { return cache.reap_finished_except(1); },
                  /*bounded_wait=*/false);
    if (!submitted)
      std::this_thread::yield();
  }
  ASSERT_TRUE(submitted);
  ASSERT_TRUE(w1.drain());
  EXPECT_TRUE(wave1_done.load(std::memory_order_acquire));

  // Wave 0 has no active jobs left, yet pending_if_any(0) must still surface its stored error.
  auto *w0_after = cache.pending_if_any(0);
  ASSERT_NE(w0_after, nullptr);
  EXPECT_FALSE(w0_after->pending());
  ASSERT_TRUE(w0_after->failure().has_value());
  EXPECT_EQ(w0_after->failure()->mnemonic, "mock_fail_mfma");
  EXPECT_EQ(w0_after->failure()->pc, kPc + 8);
}

TEST(MmaPlanTest, ReusedSlotDropsOutOfInheritedMask) {
  matrix_coexecution::SharedPool pool(1);
  std::atomic<bool> done{false};
  MockJobInstruction inst("mock_mfma", [](Instruction &, void *ctx) {
    static_cast<std::atomic<bool> *>(ctx)->store(true, std::memory_order_release);
  });
  const plan::Plan prev = make_synthetic_job_plan(0, 64, &inst);

  // Next block has step 0 touching v68 (acc1) and step 2 touching v64 (acc0).
  plan::Plan next = make_synthetic_job_plan(16, 68, &inst);
  next.footprint.reads.set(64);
  next.footprint.writes.set(64);
  next.first_access[64] = 2;
  next.first_write[64] = 2;

  plan::PendingJobs jobs(&pool);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;
  const auto no_fill = [](matrix_coexecution::SnapshotBuffer &) { return nullptr; };
  (void)jobs.begin_block(prev, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait,
                         filtered);
  ASSERT_TRUE(jobs.submit(0, prev.steps[0], &done, no_fill));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_TRUE(done.load(std::memory_order_acquire));

  // Enter next block: slot 0 is inherited and recorded in result_wait[2].
  ASSERT_TRUE(jobs.begin_block(next, kPc + 0x40, matrix_coexecution::SharedPool::kUnlimited, rwait,
                               fwait, filtered));
  EXPECT_EQ(jobs.inherited(), 1u);
  EXPECT_EQ(rwait[2], 1u);

  // Reclaiming slot 0 and reusing it for a new job must drop slot 0 from inherited().
  done.store(false, std::memory_order_relaxed);
  bool submitted = false;
  while (!submitted && std::chrono::steady_clock::now() < deadline) {
    submitted = jobs.submit(0, next.steps[0], &done, no_fill, [] { return false; },
                            /*bounded_wait=*/false);
    if (!submitted)
      std::this_thread::yield();
  }
  ASSERT_TRUE(submitted);
  EXPECT_EQ(jobs.active(), 1u);
  EXPECT_EQ(jobs.inherited(), 0u);
  EXPECT_EQ(rwait[2] & jobs.inherited(), 0u);
  EXPECT_TRUE(jobs.drain());
}

TEST(MmaPlanTest, BeginBlockSplitsResultAndFactorWaits) {
  matrix_coexecution::SharedPool pool(1);
  MockJobInstruction inst("mock_mfma", [](Instruction &, void *) {});
  const plan::Plan prev = make_synthetic_job_plan(0, 64, &inst);

  // Next block reads/writes v64 at step 1 (result wait) and overwrites v0 at step 3 (factor wait).
  plan::Plan next;
  next.accept = true;
  next.first_access.fill(plan::Plan::kNoStep);
  next.first_write.fill(plan::Plan::kNoStep);
  next.footprint.reads.set(64);
  next.footprint.writes.set(64);
  next.footprint.writes.set(0);
  next.first_access[64] = 1;
  next.first_write[64] = 1;
  next.first_access[0] = 3;
  next.first_write[0] = 3;

  plan::PendingJobs jobs(&pool);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;
  const auto no_fill = [](matrix_coexecution::SnapshotBuffer &) { return nullptr; };
  (void)jobs.begin_block(prev, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait,
                         filtered);
  ASSERT_TRUE(jobs.submit(0, prev.steps[0], nullptr, no_fill));

  ASSERT_TRUE(jobs.begin_block(next, kPc + 0x40, matrix_coexecution::SharedPool::kUnlimited, rwait,
                               fwait, filtered));
  EXPECT_TRUE(filtered);
  EXPECT_EQ(rwait[1], 1u);
  EXPECT_EQ(fwait[1], 0u);
  EXPECT_EQ(rwait[3], 0u);
  EXPECT_EQ(fwait[3], 1u);
  EXPECT_TRUE(jobs.drain());
}

TEST(MmaPlanTest, BorrowCapDoesNotSubtractWarmSpinningHelpers) {
  const int saved_reserved =
      matrix_coexecution::reserved_threads.exchange(9, std::memory_order_relaxed);
  struct Restore {
    int value;
    ~Restore() { matrix_coexecution::reserved_threads.store(value, std::memory_order_relaxed); }
  } restore{saved_reserved};

  matrix_coexecution::SharedPool pool(7);
  // Claim and release all 7 helpers so they are awake and spinning with claimed() == 0.
  MockJobInstruction inst("mock_mfma", [](Instruction &, void *) {});
  std::array<matrix_coexecution::SharedPool::Ticket, 7> tickets{};
  for (unsigned i = 0; i != 7; ++i)
    tickets[i] = pool.submit(inst, nullptr);
  for (auto t : tickets)
    ASSERT_TRUE(pool.finish(t).succeeded());

  // 16-core budget minus 9 reserved (8 engines + host) must leave 7 claimable helpers,
  // even while all 7 are awake and spinning idle.
  EXPECT_EQ(pool.claimed(), 0u);
  EXPECT_EQ(pool.borrow_cap_for(16), 7u);
  EXPECT_EQ(pool.borrow_cap_for(9), 0u);
}

TEST(MmaPlanTest, BorrowClaimReusesWarmHelperBeforeWakingSleeper) {
  matrix_coexecution::SharedPool pool(4);
  // Let all 4 helpers settle into futex sleep first.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (pool.awake() != 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (pool.awake() != 0)
    GTEST_SKIP() << "Helpers did not enter sleep within timeout.";

  MockJobInstruction inst("mock_mfma", [](Instruction &, void *) {});
  // A different thread (different issuer hash) already running claims immediately
  // after finish(t1) while that helper is warm, and must not wake a second helper.
  std::atomic<bool> go{false};
  std::atomic<bool> ready{false};
  unsigned max_awake = 0;
  std::thread other([&] {
    ready.store(true, std::memory_order_release);
    while (!go.load(std::memory_order_acquire))
      std::this_thread::yield();
    auto t2 = pool.claim(1);
    ASSERT_TRUE(t2);
    max_awake = pool.awake();
    Instruction *p = &inst;
    matrix_coexecution::SharedPool::start(t2, &p, 1, nullptr);
    ASSERT_TRUE(pool.finish(t2).succeeded());
    max_awake = std::max(max_awake, pool.awake());
  });
  while (!ready.load(std::memory_order_acquire))
    std::this_thread::yield();

  // Wake exactly one helper under cap=1, then immediately hand off to `other`.
  auto t1 = pool.claim(1);
  ASSERT_TRUE(t1);
  Instruction *ptr = &inst;
  matrix_coexecution::SharedPool::start(t1, &ptr, 1, nullptr);
  ASSERT_TRUE(pool.finish(t1).succeeded());
  EXPECT_LE(pool.awake(), 1u);
  go.store(true, std::memory_order_release);
  other.join();
  EXPECT_LE(max_awake, 1u);
}

TEST(MmaPlanTest, IndependentMfmasBatchBySharedFactor) {
  // Four A-tile groups of 4 independent MFMAs each (Tensile outer-product pattern):
  // pack_limit=4 should produce 4 jobs of 4 MFMAs grouped by shared A registers.
  Program p;
  p.pack_limit = 4;
  p.min_chains = 9;
  for (unsigned tile_a = 0; tile_a != 4; ++tile_a)
    for (unsigned tile_b = 0; tile_b != 4; ++tile_b)
      p.mfma(tile_a * 4 + tile_b, tile_a * 4, 16 + tile_b * 4);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  EXPECT_EQ(count(result, plan::StepKind::Offload), 4u);
  EXPECT_EQ(count(result, plan::StepKind::Chained), 12u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
  for (unsigned tile_a = 0; tile_a != 4; ++tile_a) {
    const auto &head = result.steps[tile_a * 4];
    EXPECT_EQ(head.kind, plan::StepKind::Offload);
    EXPECT_EQ(head.members, 4u);
    for (unsigned m = 0; m != 4; ++m)
      EXPECT_EQ(head.member_step[m], tile_a * 4 + m);
  }
}

TEST(MmaPlanTest, IndependentBatchingDoesNotMergeDifferentJoinPoints) {
  // acc0 is consumed mid-block by ds_write; acc1..acc9 live to block end.
  // acc0 must not absorb unrelated MFMAs that would delay the ds_write join.
  Program p;
  p.pack_limit = 4;
  p.min_chains = 9;
  p.mfma(0, 0, 16);
  p.mfma(1, 0, 20);
  p.mfma(2, 0, 24);
  p.ds_write_b128(64); // consumes acc0 (v[64:67])
  for (unsigned acc = 3; acc != 10; ++acc)
    p.mfma(acc, 0, 16 + (acc % 4) * 4);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  EXPECT_EQ(result.steps[0].kind, plan::StepKind::Offload);
  EXPECT_EQ(result.steps[0].members, 1u);
  EXPECT_EQ(result.steps[3].wait, uint32_t{1} << result.steps[0].job);
  EXPECT_LT(count(result, plan::StepKind::Offload), 10u);
}

TEST(MmaPlanTest, IndependentBatchingStartsFreshJobsAfterVectorMemoryWall) {
  Program p;
  p.pack_limit = 4;
  p.min_chains = 9;
  for (unsigned acc = 0; acc != 8; ++acc)
    p.mfma(acc, (acc / 4) * 4, 16 + (acc % 4) * 4);
  p.buffer_load_dwordx4(220);
  for (unsigned acc = 8; acc != 16; ++acc)
    p.mfma(acc, (acc / 4) * 4, 16 + (acc % 4) * 4);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  // Pre-wall 8 MFMAs -> 2 jobs of 4; post-wall 8 MFMAs -> 2 fresh jobs of 4.
  EXPECT_EQ(count(result, plan::StepKind::Offload), 4u);
  EXPECT_EQ(count(result, plan::StepKind::Chained), 12u);
  EXPECT_EQ(count(result, plan::StepKind::IssuerMma), 0u);
}

TEST(MmaPlanTest, PageBoundaryPlanCompilesAcrossPageAndCachesZeroStepReject) {
  // Block starting 8 bytes before a 4 KiB page boundary still compiles cleanly.
  Program p;
  p.entry_pc = kPc + GpuMemory::PAGE_SIZE - 8;
  for (unsigned acc = 0; acc != 4; ++acc)
    p.mfma(acc);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  EXPECT_EQ(result.steps.size(), 4u);

  // A 0-step rejected lookup (e.g. step 0 fails fetch inside compile) must cache
  // `first` so repeated lookups hit cleanly without calling drain_all(drains_evict).
  matrix_coexecution::SharedPool pool(1);
  plan::PlanCache cache;
  InstructionCache icache;
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna4);
  MockJobInstruction inst("mock_mfma", [](Instruction &, void *) {});
  const plan::Plan active_plan = make_synthetic_job_plan(0, 64, &inst);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;
  auto &w0 = cache.pending(0, pool);
  (void)w0.begin_block(active_plan, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait,
                       filtered);
  ASSERT_TRUE(w0.submit(0, active_plan.steps[0], nullptr,
                        [](matrix_coexecution::SnapshotBuffer &) { return nullptr; }));

  const plan::PlanCache::Key key{kPc + 0xff8, 0, 256, true, {}};
  const plan::Plan::Words first{0xd3ac0040u, 0x04020000u, 0u, 0u};
  const auto fail_fetch = [](uint64_t, plan::Plan::Words &) { return false; };
  const auto *p1 = cache.lookup(*decoder, icache, fail_fetch, key, first);
  ASSERT_NE(p1, nullptr);
  EXPECT_FALSE(p1->accept);
  const auto *p2 = cache.lookup(*decoder, icache, fail_fetch, key, first);
  EXPECT_EQ(p2, p1);
  EXPECT_EQ(cache.stats.builds, 1u);
  EXPECT_EQ(cache.stats.hits, 1u);
  EXPECT_EQ(cache.stats.drains_evict, 0u);
  EXPECT_TRUE(w0.drain());
}

TEST(MmaPlanTest, MinJobWeightRejectsUnderweightBlocks) {
  Program p;
  p.min_weight = 20; // 4 * f16 (weight 9 each = 36) < 2 * 20 = 40
  for (unsigned acc = 0; acc != 4; ++acc)
    p.mfma(acc);
  EXPECT_FALSE(p.compile().accept);
  p.min_weight = 18; // 36 >= 2 * 18
  EXPECT_TRUE(p.compile().accept);
}

TEST(MmaPlanTest, BorrowCapSkipsUndersizedPoolWhenDispatchWorkersAreActive) {
  const int saved_reserved =
      matrix_coexecution::reserved_threads.exchange(9, std::memory_order_relaxed);
  struct Restore {
    int value;
    ~Restore() { matrix_coexecution::reserved_threads.store(value, std::memory_order_relaxed); }
  } restore{saved_reserved};

  matrix_coexecution::SharedPool pool(7);
  matrix_coexecution::ScopedOccupancyShard shard;
  // 8 engines + 4 dispatch workers = 12 active CUs on 16 cores (room = 3 < 6):
  // do not thrash a tiny 3-helper pool across 12 CUs.
  shard.occupy(4);
  EXPECT_EQ(pool.borrow_cap_for(16), 0u);
  // With ample spare cores (32-core host, room = 19 >= 6), borrowing stays active.
  EXPECT_EQ(pool.borrow_cap_for(32), 19u);
  shard.occupy(0);
  EXPECT_EQ(pool.borrow_cap_for(16), 7u);
}

TEST(MmaPlanTest, PartialBatchSubmitReportsTailMemberErrorPc) {
  matrix_coexecution::SharedPool pool(1);
  MockJobInstruction ok_inst("mock_ok", [](Instruction &, void *) {});
  MockJobInstruction fail_inst("mock_tail_fail", [](Instruction &, void *) {
    if (helper_error_sink)
      *helper_error_sink = {1u, "tail failure"};
  });
  plan::Plan p;
  p.accept = true;
  p.first_access.fill(plan::Plan::kNoStep);
  p.first_write.fill(plan::Plan::kNoStep);
  plan::Step s0;
  s0.kind = plan::StepKind::Offload;
  s0.job = 0;
  s0.offset = 0;
  s0.members = 2;
  s0.inst = &ok_inst;
  s0.chain[0] = &ok_inst;
  s0.chain[1] = &fail_inst;
  s0.member_step[0] = 0;
  s0.member_step[1] = 1;
  plan::Step s1;
  s1.kind = plan::StepKind::Chained;
  s1.offset = 24;
  s1.inst = &fail_inst;
  p.steps.push_back(s0);
  p.steps.push_back(s1);
  plan::Access access;
  access.writes.set(64);
  p.job_accesses.push_back(access);

  plan::PendingJobs jobs(&pool);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;
  (void)jobs.begin_block(p, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait, filtered);
  ASSERT_TRUE(jobs.submit(
      0, p.steps[0], nullptr, [](matrix_coexecution::SnapshotBuffer &) { return nullptr; },
      [] { return false; }, /*bounded_wait=*/false, /*start_member=*/1));
  ASSERT_TRUE(jobs.drain());
  ASSERT_TRUE(jobs.failure().has_value());
  EXPECT_EQ(jobs.failure()->mnemonic, "mock_tail_fail");
  EXPECT_EQ(jobs.failure()->pc, kPc + 24);
}

TEST(MmaPlanTest, Rdna4PlanAdmitsWaitAluClauseAndVopdAndSnapshotsWar) {
  auto decoder = Decoder::create(IsaFamily::Rdna4);
  GpuMemory memory("mem");
  InstructionCache icache;
  uint64_t addr = kPc;
  const auto emit = [&](const auto &words) {
    for (uint32_t w : words) {
      memory.write32(addr, w);
      addr += 4;
    }
  };
  // Four independent Wave32 FP16 WMMAs (A=v[0:3], B=v[4:7], C/D=v[16..47]),
  // separated by s_wait_alu, s_clause, and v_dual_mov_b32 :: v_dual_mov_b32
  // (which overwrites A's v[0:1] -> WAR snapshot instead of join).
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16F16Vop3p,
                          {.vdst = 16, .src0 = 256 + 0, .src1 = 256 + 4, .src2 = 256 + 16}));
  emit(rdna4::build_sopp(rdna4::kSWaitAluSopp, {.simm16 = 0x0f9f}));
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16F16Vop3p,
                          {.vdst = 24, .src0 = 256 + 0, .src1 = 256 + 8, .src2 = 256 + 24}));
  emit(rdna4::build_sopp(rdna4::kSClauseSopp, {.simm16 = 1}));
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16F16Vop3p,
                          {.vdst = 32, .src0 = 256 + 0, .src1 = 256 + 4, .src2 = 256 + 32}));
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16F16Vop3p,
                          {.vdst = 40, .src0 = 256 + 0, .src1 = 256 + 8, .src2 = 256 + 40}));
  // v_dual_mov_b32 v0, v12 :: v_dual_mov_b32 v1, v13 (overwrites factor A v[0:1]).
  const std::array<uint32_t, 2> vopd = {
      (0x32u << 26) | (8u << 22) | (8u << 17) | (0u << 9) | (256u + 12u),
      (0u << 24) | (0u << 17) | (0u << 9) | (256u + 13u)};
  emit(vopd);
  emit(rdna4::build_sopp(rdna4::kSEndpgmSopp, {.simm16 = 0}));

  plan::Plan p;
  uint64_t decodes = 0;
  plan::compile(
      p, *decoder,
      [&](uint64_t a, plan::Plan::Words &w) {
        icache.fetch(memory, a, reinterpret_cast<uint8_t *>(w.data()));
        return true;
      },
      kPc, 64, /*has_accvgprs=*/false, decodes, /*tail_snapshot=*/false, /*pack_limit=*/4,
      /*min_chains=*/4, /*min_weight=*/0, /*wave_size=*/32);
  ASSERT_TRUE(p.accept);
  // s_wait_alu, s_clause, and v_dual_* must all stay inside the block as Alu steps.
  EXPECT_EQ(p.steps.size(), 7u);
  EXPECT_EQ(p.alu, 3u);
  EXPECT_EQ(p.ordinary, 0u);
  // Lightweight RDNA4 packing merges the 4 shared-factor chains into 1 job,
  // and v_dual_mov overwriting v[0:1] is resolved by WAR snapshotting (no join).
  EXPECT_EQ(p.chains, 1u);
  EXPECT_EQ(p.chained, 3u);
  EXPECT_TRUE(p.steps[0].snapshots);
  EXPECT_EQ(p.steps[6].wait, 0u);
}

TEST(MmaPlanTest, Rdna4Wave64AndSwmmacFootprintsAndScalarFilter) {
  auto decoder = Decoder::create(IsaFamily::Rdna4);
  // Wave64 halves RDNA4 WMMA VGPR widths: v[0:3] (4 regs), not v[0:7] (8 regs).
  const auto wmma_words = rdna4::build_vop3p(
      rdna4::kVWmmaF3216x16x16F16Vop3p,
      {.vdst = 60, .src0 = 256 + 0, .src1 = 256 + 2, .src2 = 256 + 60});
  std::unique_ptr<Instruction> wmma(decode_valid(*decoder, wmma_words.data()));
  EXPECT_FALSE(async_mma_policy::footprint(*wmma, 64, false, 32).has_value());
  const auto w64_fp = async_mma_policy::footprint(*wmma, 64, false, 64);
  ASSERT_TRUE(w64_fp.has_value());
  EXPECT_EQ(w64_fp->writes.count(), 4u);
  EXPECT_TRUE(w64_fp->writes.test(60) && w64_fp->writes.test(63));
  EXPECT_TRUE(w64_fp->reads.test(0) && w64_fp->reads.test(1) && !w64_fp->reads.test(4));

  // Sparse SWMMAC reads and writes its tied accumulator vdst.
  const auto swmmac_words = rdna4::build_vop3p(
      rdna4::kVSwmmacF3216x16x32Fp8Fp8Vop3p,
      {.vdst = 16, .src0 = 256 + 0, .src1 = 256 + 4, .src2 = 256 + 8});
  std::unique_ptr<Instruction> swmmac(decode_valid(*decoder, swmmac_words.data()));
  const auto sw_fp = async_mma_policy::footprint(*swmmac, 64, false, 32);
  ASSERT_TRUE(sw_fp.has_value());
  EXPECT_EQ(sw_fp->writes.count(), 8u);
  EXPECT_TRUE(sw_fp->reads.test(16) && sw_fp->reads.test(23));
  EXPECT_TRUE(sw_fp->reads.test(8));

  // RDNA4 scalar_only_encoding must reject MODE-modifying ops (fp16_ovfl).
  const auto round_mode = rdna4::build_sopp(rdna4::kSRoundModeSopp, {.simm16 = 0});
  const auto setreg = rdna4::build_sopk(rdna4::kSSetregB32Sopk, {.simm16 = 0, .sdst = 0});
  const auto s_add = rdna4::build_sop2(rdna4::kSAddF32Sop2, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 2});
  EXPECT_FALSE(plan::scalar_only_encoding(IsaFamily::Rdna4, round_mode[0]));
  EXPECT_FALSE(plan::scalar_only_encoding(IsaFamily::Rdna4, setreg[0]));
  EXPECT_TRUE(plan::scalar_only_encoding(IsaFamily::Rdna4, s_add[0]));
}

TEST(MmaPlanTest, Rdna4SplitWaitsUseStepKindWaitAndPureBurstKeepsFanOut) {
  auto decoder = Decoder::create(IsaFamily::Rdna4);
  GpuMemory memory("rdna4_wait_burst_mem");
  InstructionCache icache;
  constexpr uint64_t kPc = 0x1000;
  uint64_t addr = kPc;
  auto emit = [&](const auto &words) {
    for (uint32_t w : words) {
      memory.write32(addr, w);
      addr += 4;
    }
  };
  // Four 2-WMMA chains separated by s_wait_dscnt (pure burst, 0 issuer ALU/mem):
  // s_wait_dscnt must compile to StepKind::Wait (ordinary == 0), and the 4
  // multi-member chains must stay 4 independent jobs for helper fan-out.
  for (uint32_t k = 0; k != 2; ++k) {
    emit(rdna4::build_sopp(rdna4::kSWaitDscntSopp, {.simm16 = static_cast<uint16_t>(1 - k)}));
    for (uint32_t c = 0; c != 4; ++c) {
      emit(rdna4::build_vop3p(
          rdna4::kVWmmaF3216x16x16F16Vop3p,
          {.vdst = static_cast<uint8_t>(c * 8),
           .src0 = static_cast<uint16_t>(256 + 32 + k * 8 + (c & 1) * 4),
           .src1 = static_cast<uint16_t>(256 + 48 + k * 8 + (c >> 1) * 4),
           .src2 = static_cast<uint16_t>(256 + c * 8)}));
    }
  }
  emit(rdna4::build_sopp(rdna4::kSEndpgmSopp, {.simm16 = 0}));

  plan::Plan p;
  uint64_t decodes = 0;
  // Entry at the first WMMA (skip leading s_wait_dscnt at kPc).
  plan::compile(
      p, *decoder,
      [&](uint64_t a, plan::Plan::Words &w) {
        icache.fetch(memory, a, reinterpret_cast<uint8_t *>(w.data()));
        return true;
      },
      kPc + 4, 80, /*has_accvgprs=*/false, decodes, /*tail_snapshot=*/false,
      plan::batch_pack_limit(), plan::batch_min_chains(), /*min_weight=*/0, /*wave_size=*/32);
  ASSERT_TRUE(p.accept);
  EXPECT_EQ(p.wait, 1u);
  EXPECT_EQ(p.ordinary, 0u);
  EXPECT_EQ(p.chains, 4u);
  EXPECT_EQ(p.chained, 4u);
}

TEST(MmaPlanTest, SingleMultiOpChainWithIssuerOverlapIsAccepted) {
  auto decoder = Decoder::create(IsaFamily::Rdna4);
  GpuMemory memory("rdna4_single_overlap_mem");
  InstructionCache icache;
  constexpr uint64_t kPc = 0x2000;
  uint64_t addr = kPc;
  auto emit = [&](const auto &words) {
    for (uint32_t w : words) {
      memory.write32(addr, w);
      addr += 4;
    }
  };
  // One 4-WMMA chain into v[0:7] followed by 8 independent VALU ops:
  // single-chain overlap profitability should accept and offload it.
  for (uint32_t k = 0; k != 4; ++k) {
    emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16Fp8Fp8Vop3p,
                            {.vdst = 0,
                             .src0 = static_cast<uint16_t>(256 + 16 + k * 2),
                             .src1 = static_cast<uint16_t>(256 + 24 + k * 2),
                             .src2 = static_cast<uint16_t>(256 + 0)}));
  }
  for (uint32_t i = 0; i != 8; ++i) {
    emit(rdna4::build_vop1(rdna4::kVMovB32Vop1,
                           {.src0 = static_cast<uint16_t>(256 + 40 + i),
                            .vdst = static_cast<uint8_t>(48 + i)}));
  }
  emit(rdna4::build_sopp(rdna4::kSEndpgmSopp, {.simm16 = 0}));

  plan::Plan p;
  uint64_t decodes = 0;
  plan::compile(
      p, *decoder,
      [&](uint64_t a, plan::Plan::Words &w) {
        icache.fetch(memory, a, reinterpret_cast<uint8_t *>(w.data()));
        return true;
      },
      kPc, 64, /*has_accvgprs=*/false, decodes, /*tail_snapshot=*/false, /*pack_limit=*/4,
      /*min_chains=*/9, /*min_weight=*/0, /*wave_size=*/32);
  ASSERT_TRUE(p.accept);
  EXPECT_EQ(p.chains, 1u);
  EXPECT_EQ(p.chained, 3u);
  EXPECT_EQ(p.alu, 8u);
}

TEST(MmaPlanTest, WithinBlockReusedSlotClearsRetiredJobMapping) {
  // Job 0 occupies slot 0, finishes, and is reaped when independent job 1
  // reuses slot 0. A later planned wait for job 0 must not falsely wait on job 1.
  matrix_coexecution::SharedPool pool(1);
  std::atomic<bool> job0_done{false};
  std::atomic<bool> release_job1{false};
  MockJobInstruction inst0("mock_mfma_0", [](Instruction &, void *ctx) {
    static_cast<std::atomic<bool> *>(ctx)->store(true, std::memory_order_release);
  });
  MockJobInstruction inst1("mock_mfma_1", [](Instruction &, void *ctx) {
    auto *gate = static_cast<std::atomic<bool> *>(ctx);
    while (!gate->load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  plan::Plan p;
  p.accept = true;
  p.steps.resize(2);
  p.job_accesses.resize(2);
  p.steps[0].kind = plan::StepKind::Offload;
  p.steps[0].job = 0;
  p.steps[0].members = 1;
  p.steps[0].chain[0] = &inst0;
  p.steps[0].member_step[0] = 0;
  p.job_accesses[0].writes.set(0);
  p.steps[1].kind = plan::StepKind::Offload;
  p.steps[1].job = 1;
  p.steps[1].members = 1;
  p.steps[1].chain[0] = &inst1;
  p.steps[1].member_step[0] = 1;
  p.job_accesses[1].writes.set(8);

  plan::PendingJobs pending(&pool);
  std::array<plan::JobMask, plan::kMaxSteps> rwait, fwait;
  bool filtered = false;
  (void)pending.begin_block(p, kPc, matrix_coexecution::SharedPool::kUnlimited, rwait, fwait,
                            filtered);
  const auto no_fill = [](matrix_coexecution::SnapshotBuffer &) { return nullptr; };

  ASSERT_TRUE(pending.submit(0, p.steps[0], &job0_done, no_fill));
  ASSERT_EQ(pending.slots_of(plan::JobMask{1} << 0), plan::JobMask{1} << 0);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!job0_done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_TRUE(job0_done.load(std::memory_order_acquire));

  // Submitting job 1 reaps completed job 0 and reuses physical slot 0.
  bool submitted = false;
  while (!submitted && std::chrono::steady_clock::now() < deadline) {
    submitted = pending.submit(1, p.steps[1], &release_job1, no_fill, [] { return false; },
                               /*bounded_wait=*/false);
    if (!submitted)
      std::this_thread::yield();
  }
  ASSERT_TRUE(submitted);
  EXPECT_EQ(pending.slots_of(plan::JobMask{1} << 0), 0u);
  EXPECT_EQ(pending.slots_of(plan::JobMask{1} << 1), plan::JobMask{1} << 0);

  release_job1.store(true, std::memory_order_release);
  ASSERT_TRUE(pending.drain());
  EXPECT_EQ(pending.slots_of(plan::JobMask{1} << 1), 0u);
}

TEST(MmaPlanTest, LocalOnlyFiniteCapRespectsBudget) {
  // With 16 helpers across <= 8 CCDs and cap=8, cap >= active_shards_ exercises
  // the strictly shard-local budget path (zero cross-CCD shard scans) and never
  // exceeds cap.
  matrix_coexecution::SharedPool pool(16);
  EXPECT_FALSE(pool.claim(0, /*local_only=*/true));
  std::vector<matrix_coexecution::SharedPool::Ticket> held;
  for (unsigned i = 0; i != 16; ++i) {
    auto t = pool.claim(8, /*local_only=*/true);
    if (!t)
      break;
    held.push_back(t);
  }
  EXPECT_GE(held.size(), 1u);
  EXPECT_LE(held.size(), 8u);
  EXPECT_LE(pool.claimed(), 8u);
  MockJobInstruction inst("mock_mfma", [](Instruction &, void *) {});
  for (auto t : held) {
    Instruction *ptr = &inst;
    matrix_coexecution::SharedPool::start(t, &ptr, 1, nullptr);
    ASSERT_TRUE(pool.finish(t).succeeded());
  }
  EXPECT_EQ(pool.claimed(), 0u);
}

TEST(MmaPlanTest, Rdna4ExplicitVccAndHighSgprStayInOnePlanAbove128Steps) {
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Rdna4);
  amdgpu::GpuMemory memory("mem");
  amdgpu::InstructionCache icache;
  uint64_t addr = kPc;
  const auto emit = [&](const auto &enc) {
    for (uint32_t w : enc) {
      memory.write32(addr, w);
      addr += 4;
    }
  };
  // WMMA #0 -> s_and_b32 vcc_lo -> s_mov_b32 s104 -> 140 ALUs -> WMMA #1 -> s_endpgm.
  // Verifies explicit VCC/s104 writes do not split the block and >128 steps fit.
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16Bf16Vop3p,
                          {.vdst = 0, .src0 = 256 + 16, .src1 = 256 + 20, .src2 = 256 + 0}));
  emit(rdna4::build_sop2(rdna4::kSAndB32Sop2, {.ssrc0 = 0, .ssrc1 = 106, .sdst = 106}));
  emit(rdna4::build_sop1(rdna4::kSMovB32Sop1, {.ssrc0 = 1, .sdst = 104}));
  for (unsigned i = 0; i != 140; ++i) {
    emit(rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = 256 + 32, .vdst = 33}));
  }
  emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16Bf16Vop3p,
                          {.vdst = 8, .src0 = 256 + 24, .src1 = 256 + 28, .src2 = 256 + 8}));
  emit(rdna4::build_sopp(rdna4::kSEndpgmSopp, {.simm16 = 0}));

  plan::Plan p;
  uint64_t decodes = 0;
  plan::compile(
      p, *decoder,
      [&](uint64_t a, plan::Plan::Words &w) {
        icache.fetch(memory, a, reinterpret_cast<uint8_t *>(w.data()));
        return true;
      },
      kPc, 64, /*has_accvgprs=*/false, decodes, /*tail_snapshot=*/false, /*pack_limit=*/4,
      /*min_chains=*/9, /*min_weight=*/0, /*wave_size=*/32, rocjitsu::IsaFamily::Rdna4);
  ASSERT_TRUE(p.accept);
  EXPECT_FALSE(p.truncated);
  EXPECT_EQ(p.steps.size(), 144u);
  EXPECT_EQ(p.chains, 2u);
  EXPECT_EQ(p.alu, 142u);
}

TEST(MmaPlanTest, RejectedSerialChainSetsNonTruncatedSpanAndInvalidatesOnEpochChange) {
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Rdna4);
  amdgpu::GpuMemory memory("mem");
  amdgpu::InstructionCache icache;
  uint64_t addr = kPc;
  const auto emit = [&](const auto &enc) {
    for (uint32_t w : enc) {
      memory.write32(addr, w);
      addr += 4;
    }
  };
  // 4 serial WMMAs into v[0:7] followed immediately by a consumer -> rejected, non-truncated.
  std::array<uint64_t, 4> wmma_pcs{};
  for (unsigned i = 0; i != 4; ++i) {
    wmma_pcs[i] = addr;
    emit(rdna4::build_vop3p(rdna4::kVWmmaF3216x16x16Fp8Fp8Vop3p,
                            {.vdst = 0,
                             .src0 = static_cast<uint16_t>(256 + 16 + 2 * i),
                             .src1 = static_cast<uint16_t>(256 + 32 + 2 * i),
                             .src2 = 256 + 0}));
  }
  emit(rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = 256 + 0, .vdst = 40}));
  emit(rdna4::build_sopp(rdna4::kSEndpgmSopp, {.simm16 = 0}));

  plan::PlanCache cache;
  const auto fetch_words = [&](uint64_t a, plan::Plan::Words &w) {
    icache.fetch(memory, a, reinterpret_cast<uint8_t *>(w.data()));
    return true;
  };
  plan::Plan::Words first{};
  ASSERT_TRUE(fetch_words(wmma_pcs[0], first));
  const plan::PlanCache::Key key{wmma_pcs[0], 0, 64, false, {}, 32};
  const plan::Plan *p =
      cache.lookup(*decoder, icache, fetch_words, key, first, rocjitsu::IsaFamily::Rdna4);
  ASSERT_FALSE(p->accept);
  EXPECT_FALSE(p->truncated);
  EXPECT_GT(p->length, 0u);

  int fake_wave = 0;
  auto &span = cache.rejected_span(0);
  span = {&fake_wave, 1, icache.epoch(), 64, wmma_pcs[0], wmma_pcs[0] + p->length};
  for (uint64_t pc : wmma_pcs)
    EXPECT_TRUE(span.contains(&fake_wave, 1, icache.epoch(), 64, pc));
  EXPECT_FALSE(span.contains(&fake_wave, 1, icache.epoch() + 1, 64, wmma_pcs[1]));
  EXPECT_FALSE(span.contains(&fake_wave, 1, icache.epoch(), 128, wmma_pcs[1]));
  EXPECT_FALSE(span.contains(&fake_wave, 2, icache.epoch(), 64, wmma_pcs[1]));
  cache.clear();
  EXPECT_FALSE(cache.rejected_span(0).contains(&fake_wave, 1, icache.epoch(), 64, wmma_pcs[0]));
}

TEST(MmaPlanTest, ScratchRenameEliminatesRepeatedDsReadWarSnapshots) {
  Program p;
  p.pack_limit = 1;
  // v0: two independent MFMAs reading shared A=v[0:3].
  p.mfma(0, /*a=*/0, /*b=*/16);
  p.mfma(1, /*a=*/0, /*b=*/20);
  // v1: ds_read_b128 overwrites v[0:3] (step 2).
  p.ds_read_b128(0);
  p.mfma(2, /*a=*/0, /*b=*/16);
  p.mfma(3, /*a=*/0, /*b=*/20);
  // Read acc 0 (v64) & acc 1 (v68) via ALU so v0_readers are joined before v2 write.
  for (uint8_t src : {uint8_t(64), uint8_t(68)}) {
    const auto w = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = uint16_t(256 + src), .vdst = 50});
    p.words.insert(p.words.end(), w.begin(), w.end());
  }
  // v2 (final write, step 7): v0_readers already joined -> final_uses_arch, zero commits.
  p.ds_read_b128(0);
  p.mfma(2, /*a=*/0, /*b=*/16);
  p.mfma(3, /*a=*/0, /*b=*/20);

  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  EXPECT_EQ(result.rename_scratch_regs, 4u);
  EXPECT_TRUE(result.rename_commits.empty());
  // Step 2 (v1 load) writes scratch slot 0; step 7 (v2 final load) writes arch directly.
  EXPECT_EQ(result.steps[2].rename_dst_reg, 0u);
  EXPECT_EQ(result.steps[2].rename_arch_base, 0u);
  EXPECT_EQ(result.steps[2].rename_dst_regs, 4u);
  EXPECT_EQ(result.steps[7].rename_dst_reg, plan::SnapshotSpec::kNoSlot);
  // v0 readers (steps 0, 1) have snapshots cleared; v1 readers (steps 3, 4) read scratch slot 0.
  EXPECT_EQ(result.steps[0].snapshot[0].a_regs, 0u);
  EXPECT_EQ(result.steps[1].snapshot[0].a_regs, 0u);
  EXPECT_NE(result.steps[3].snapshot[0].rename_mask & plan::SnapshotSpec::kRenameA, 0u);
  EXPECT_EQ(result.steps[3].snapshot[0].a_slot, 0u);
  EXPECT_NE(result.steps[4].snapshot[0].rename_mask & plan::SnapshotSpec::kRenameA, 0u);
  EXPECT_EQ(result.steps[4].snapshot[0].a_slot, 0u);
}

TEST(MmaPlanTest, ScratchRenameCutoffHandlesDeadOverwriteAndLiveMidBlockCommit) {
  // Case 1: Late non-load ALU overwrites v[0:3] completely (dead at cutoff -> 0 commits,
  // but v0_readers must be joined before the first architectural write at step 5).
  {
    Program p;
    p.pack_limit = 1;
    p.mfma(0, /*a=*/0, /*b=*/16);
    p.mfma(1, /*a=*/0, /*b=*/20);
    p.ds_read_b128(0); // step 2 -> renamed
    p.mfma(2, /*a=*/0, /*b=*/16);
    p.mfma(3, /*a=*/0, /*b=*/20);
    for (uint8_t r = 0; r != 4; ++r) {
      const auto w = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = uint16_t(256 + 40 + r), .vdst = r});
      p.words.insert(p.words.end(), w.begin(), w.end());
    }
    const auto result = p.compile();
    ASSERT_TRUE(result.accept);
    EXPECT_EQ(result.rename_scratch_regs, 4u);
    EXPECT_EQ(result.steps[2].rename_dst_reg, 0u);
    EXPECT_TRUE(result.rename_commits.empty());
    // Step 5 (first v_mov into v0) must join both v0_readers (jobs 0 and 1).
    const plan::JobMask v0_mask = (plan::JobMask{1} << result.steps[0].job) |
                                  (plan::JobMask{1} << result.steps[1].job);
    EXPECT_EQ(result.steps[5].wait & v0_mask, v0_mask);
  }
  // Case 2: Late ds_write_b128 reads v[0:3] (live at cutoff -> mid-block RenameCommit).
  {
    Program p;
    p.pack_limit = 1;
    p.mfma(0, /*a=*/0, /*b=*/16);
    p.mfma(1, /*a=*/0, /*b=*/20);
    p.ds_read_b128(0); // step 2 -> renamed
    p.mfma(2, /*a=*/0, /*b=*/16);
    p.mfma(3, /*a=*/0, /*b=*/20);
    p.ds_write_b128(0); // step 5 -> reads v[0:3] -> triggers mid-block commit
    const auto result = p.compile();
    ASSERT_TRUE(result.accept);
    EXPECT_EQ(result.rename_scratch_regs, 4u);
    EXPECT_EQ(result.steps[2].rename_dst_reg, 0u);
    EXPECT_TRUE(result.steps[5].has_rename_commit_before);
    ASSERT_EQ(result.rename_commits.size(), 1u);
    EXPECT_EQ(result.rename_commits[0].before_step, 5u);
    EXPECT_EQ(result.rename_commits[0].arch_base, 0u);
    EXPECT_EQ(result.rename_commits[0].scratch_reg, 0u);
    EXPECT_EQ(result.rename_commits[0].regs, 4u);
  }
}

TEST(MmaPlanTest, ScratchRenameHandlesSplitTransposeLoads) {
  Program p;
  p.pack_limit = 1;
  p.mfma(0, /*a=*/0, /*b=*/16);
  p.mfma(1, /*a=*/0, /*b=*/20);
  const auto tr_lo = cdna4::build_ds(cdna4::kDsReadB64TrB16, {.addr = 200, .vdst = 0});
  const auto tr_hi = cdna4::build_ds(cdna4::kDsReadB64TrB16, {.addr = 200, .vdst = 2});
  p.words.insert(p.words.end(), tr_lo.begin(), tr_lo.end());
  p.words.insert(p.words.end(), tr_hi.begin(), tr_hi.end());
  p.mfma(2, /*a=*/0, /*b=*/16);
  p.mfma(3, /*a=*/0, /*b=*/20);

  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  EXPECT_EQ(result.rename_scratch_regs, 4u);
  EXPECT_EQ(result.steps[2].rename_dst_reg, 0u);
  EXPECT_EQ(result.steps[2].rename_dst_regs, 2u);
  EXPECT_EQ(result.steps[3].rename_dst_reg, 2u);
  EXPECT_EQ(result.steps[3].rename_dst_regs, 2u);
  EXPECT_NE(result.steps[4].snapshot[0].rename_mask & plan::SnapshotSpec::kRenameA, 0u);
  EXPECT_EQ(result.steps[4].snapshot[0].a_slot, 0u);
  EXPECT_NE(result.steps[5].snapshot[0].rename_mask & plan::SnapshotSpec::kRenameA, 0u);
  EXPECT_EQ(result.steps[5].snapshot[0].a_slot, 0u);
}

TEST(MmaPlanTest, Cdna4DenseF64F32Bf16I8AndSparseSmfmacPlanAndRenameCorrectly) {
  auto decoder = Decoder::create(IsaFamily::Cdna4);
  for (uint16_t opcode :
       {cdna4::kVMfmaF6416x16x4F64Vop3pMfma, cdna4::kVMfmaF3232x32x2F32Vop3pMfma,
        cdna4::kVMfmaF3216x16x16Bf16Vop3pMfma, cdna4::kVMfmaI3216x16x64I8Vop3pMfma}) {
    Program p;
    for (unsigned acc = 0; acc != 2; ++acc) {
      const uint8_t dst = static_cast<uint8_t>(64 + 16 * acc);
      const auto w = cdna4::build_vop3p_mfma(
          opcode, {.vdst = dst, .src0 = 256 + 0, .src1 = 256 + 8, .src2 = uint16_t(256 + dst)});
      p.words.insert(p.words.end(), w.begin(), w.end());
    }
    const auto compiled = p.compile();
    ASSERT_TRUE(compiled.accept) << "opcode=" << opcode;
    EXPECT_EQ(count(compiled, plan::StepKind::Offload), 2u) << "opcode=" << opcode;
  }

  const auto smfmac0_w = cdna4::build_vop3p_mfma(
      cdna4::kVSmfmacF3216x16x64F16Vop3pMfma,
      {.vdst = 64, .src0 = 256 + 0, .src1 = 256 + 16, .src2 = 256 + 40});
  std::unique_ptr<Instruction> smfmac0(decode_valid(*decoder, smfmac0_w.data()));
  ASSERT_TRUE(plan::offloadable(*smfmac0, /*has_accvgprs=*/true));
  const auto mo = plan::member_operands(*smfmac0, 0, 256, /*has_accvgprs=*/true, 64);
  ASSERT_TRUE(mo.has_value());
  EXPECT_EQ(mo->a_base, 0u);
  EXPECT_EQ(mo->a_regs, 4u);
  EXPECT_EQ(mo->b_base, 16u);
  EXPECT_EQ(mo->b_regs, 8u);
  EXPECT_EQ(mo->sa_regs, 0u);
  EXPECT_EQ(mo->sb_regs, 0u);
  EXPECT_TRUE(mo->other.reads.test(64) && mo->other.reads.test(67));
  EXPECT_TRUE(mo->other.writes.test(64) && mo->other.writes.test(67));
  EXPECT_TRUE(mo->other.reads.test(40));

  Program sp;
  sp.pack_limit = 1;
  sp.words.insert(sp.words.end(), smfmac0_w.begin(), smfmac0_w.end());
  const auto smfmac1_w = cdna4::build_vop3p_mfma(
      cdna4::kVSmfmacF3216x16x64F16Vop3pMfma,
      {.vdst = 72, .src0 = 256 + 0, .src1 = 256 + 24, .src2 = 256 + 41});
  sp.words.insert(sp.words.end(), smfmac1_w.begin(), smfmac1_w.end());
  sp.ds_read_b128(0); // overwrites A=v[0:3] -> resolved by scratch rename (no snapshot, no join)
  const auto mov_idx = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256 + 42, .vdst = 40});
  sp.words.insert(sp.words.end(), mov_idx.begin(), mov_idx.end());
  const auto compiled_sp = sp.compile();
  ASSERT_TRUE(compiled_sp.accept);
  EXPECT_EQ(count(compiled_sp, plan::StepKind::Offload), 2u);
  EXPECT_EQ(compiled_sp.steps[2].rename_dst_reg, 0u);
  EXPECT_EQ(compiled_sp.steps[0].snapshot[0].a_regs, 0u);
  EXPECT_EQ(compiled_sp.steps[2].wait, 0u) << "A overwrite must rename, not wait";
  EXPECT_EQ(compiled_sp.steps[3].wait, plan::JobMask{1} << compiled_sp.steps[0].job)
      << "sparse index overwrite must wait for job 0";
  EXPECT_EQ(compiled_sp.steps[2].kind, plan::StepKind::Lds);
  EXPECT_TRUE(compiled_sp.steps[0].direct_vgpr_wait);
}

} // namespace
