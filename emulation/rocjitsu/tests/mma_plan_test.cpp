// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_plan_test.cpp
/// @brief Straight-line MMA plan scheduling: profitability, placement and chaining legality.

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/mma_plan.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
namespace plan = rocjitsu::amdgpu::mma_plan;

constexpr uint64_t kPc = 0x10000;

struct Program {
  std::vector<uint32_t> words;
  // f16 16x16x32: A and B are four VGPRs, C/D four VGPRs. acc selects the accumulator tuple.
  void mfma(unsigned acc, unsigned a = 0, unsigned b = 16) {
    const auto w = cdna4::build_vop3p_mfma(
        cdna4::kVMfmaF3216x16x32F16Vop3pMfma,
        {.vdst = uint8_t(64 + 4 * acc), .src0 = uint16_t(256 + a), .src1 = uint16_t(256 + b),
         .src2 = uint16_t(256 + 64 + 4 * acc)});
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
      const size_t index = (address - kPc) / 4;
      for (size_t i = 0; i != out.size(); ++i)
        out[i] = index + i < words.size() ? words[index + i] : 0;
      return true;
    };
    plan::compile(result, *decoder, fetch, kPc, 256, true, decodes);
    return result;
  }
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
  for (unsigned acc = 0; acc != 40; ++acc)
    p.mfma(acc);
  const auto result = p.compile();
  ASSERT_TRUE(result.accept);
  ASSERT_EQ(result.steps.size(), 40u);
  unsigned jobs = 0, members = 0;
  for (const auto &step : result.steps)
    if (step.kind == plan::StepKind::Offload) {
      ++jobs;
      members += step.members;
    }
  EXPECT_EQ(jobs, plan::kMaxJobs);
  EXPECT_EQ(members, 40u);
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
  EXPECT_EQ(result.steps[2].kind, plan::StepKind::Ordinary);
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
  EXPECT_EQ(result.steps[1].kind, plan::StepKind::Ordinary);
  EXPECT_EQ(result.steps[2].kind, plan::StepKind::Alu);
  EXPECT_TRUE(result.accept);
}

TEST(MmaPlanTest, LdsReadsBetweenIndependentChainsDoNotEndJobs) {
  // Ten independent accumulators interleaved with LDS reads of other registers
  // all hand off; the reads stay issuer steps inside the block.
  Program p;
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
  EXPECT_EQ(count(result, plan::StepKind::Ordinary), 10u);
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
} // namespace
