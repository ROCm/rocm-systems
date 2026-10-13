// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <memory>
#include <semaphore>
#include <thread>

namespace {
using namespace rocjitsu;
namespace policy = amdgpu::async_mma_policy;
namespace mc = amdgpu::matrix_coexecution;

TEST(AsyncMmaPolicyTest, EncodingHintDoesNotExcludeEnabledDecodedCandidates) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  // Exhaust the VOP3P opcode space so future decoded candidates cannot silently
  // become ineligible only after their instruction-cache line is warm.
  for (unsigned opcode = 0; opcode != 512; ++opcode) {
    const auto encoding = cdna5::build_vop3p(
        opcode, {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
    // An opcode may decode as an extension prefix; provide the full decode window.
    const std::array<uint32_t, cdna5::Decoder::kMaxInstructionWords> words = {encoding[0],
                                                                              encoding[1]};
    auto decoded = decoder->decode(words.data());
    if (decoded.failed())
      continue;
    const auto &inst = *decoded.value();
    SCOPED_TRACE(inst.mnemonic());
    if (policy::candidate(inst.mnemonic())) {
      EXPECT_TRUE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA5, words[0]));
    }
  }
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_RDNA4})
    EXPECT_TRUE(policy::encoding_may_be_candidate(arch, 0));
  EXPECT_FALSE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA5, 0));
}

TEST(AsyncMmaPolicyTest, Cdna4EncodingHintCoversOrdinaryAndExtendedMfma) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  for (unsigned opcode = 0; opcode != 128; ++opcode) {
    const auto encoding =
        cdna4::build_vop3p_mfma(opcode, {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320});
    // An opcode may decode as an extension prefix; provide the maximum size.
    const std::array<uint32_t, 4> words = {encoding[0], encoding[1], 0, 0};
    auto result = decoder->decode(words.data());
    if (result.succeeded() && policy::candidate(result.value()->mnemonic())) {
      EXPECT_TRUE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA4, words[0]));
    }
  }
  for (unsigned opcode : {45, 46}) {
    const auto words = mma_test::make_cdna4_mfma_scale_words(opcode, 1, 448, 449);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_TRUE(policy::candidate(inst->mnemonic()));
    EXPECT_TRUE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA4, words[0]));
  }
  const auto add = cdna4::build_sop2(cdna4::kSAddU32Sop2, {.ssrc0 = 1, .ssrc1 = 2, .sdst = 0});
  EXPECT_FALSE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA4, add[0]));
}

TEST(AsyncMmaPolicyTest, TargetSelectionMatchesTheAcceptedInstructionShapes) {
  EXPECT_TRUE(policy::supported(ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_TRUE(policy::supported(ROCJITSU_CODE_ARCH_CDNA5));
  EXPECT_FALSE(policy::supported(ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(policy::supported(ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_FALSE(policy::supported(ROCJITSU_CODE_ARCH_RDNA4));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  for (auto opcode : {rdna4::kVWmmaF3216x16x16F16Vop3p, rdna4::kVWmmaI3216x16x32Iu4Vop3p,
                      rdna4::kVSwmmacF3216x16x32F16Vop3p}) {
    const auto words = rdna4::build_vop3p(
        opcode, {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    EXPECT_FALSE(policy::candidate(inst->mnemonic()));
  }
}

TEST(AsyncMmaPolicyTest, QualifiedMfmaFamiliesExcludeUnmeasuredShapes) {
  for (auto name :
       {"v_mfma_f32_16x16x32_f16", "v_mfma_f32_32x32x16_f16", "v_mfma_f32_16x16x32_fp8_fp8",
        "v_mfma_f32_32x32x16_bf8_fp8", "v_mfma_f32_16x16x128_f8f6f4", "v_mfma_f32_32x32x64_f8f6f4",
        "v_mfma_scale_f32_16x16x128_f8f6f4"}) {
    EXPECT_TRUE(policy::candidate(name));
  }
  EXPECT_FALSE(policy::candidate("v_mfma_f32_32x32x4_2b_f16"));
  EXPECT_FALSE(policy::candidate("v_mfma_i32_16x16x64_i8"));
}

TEST(AsyncMmaPolicyTest, Cdna5DefaultFamiliesIncludeK32AndScaledWmma) {
  for (auto name :
       {"v_wmma_f32_16x16x32_f16", "v_wmma_f32_16x16x32_bf16", "v_wmma_f32_16x16x64_fp8_fp8",
        "v_wmma_f32_16x16x128_fp8_fp8", "v_wmma_scale_f32_16x16x128_f8f6f4"}) {
    EXPECT_TRUE(policy::candidate(name));
  }
  EXPECT_FALSE(policy::candidate("v_wmma_f32_16x16x16_f16"));
}

TEST(AsyncMmaPolicyTest, ScaledWmmaHintAndFootprintIncludeBothScaleRegisters) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  for (unsigned format_a = 0; format_a != 5; ++format_a) {
    for (unsigned format_b = 0; format_b != 5; ++format_b) {
      const auto words = mma_test::make_cdna5_wmma_scale_words(format_a, format_b, 448, 449);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_EQ(inst->mnemonic(), "v_wmma_scale_f32_16x16x128_f8f6f4");
      EXPECT_TRUE(policy::candidate(inst->mnemonic()));
      EXPECT_TRUE(policy::encoding_may_be_candidate(ROCJITSU_CODE_ARCH_CDNA5, words[0]));
      auto access = policy::footprint(*inst, 256);
      ASSERT_TRUE(access);
      EXPECT_TRUE(access->reads.test(192));
      EXPECT_TRUE(access->reads.test(193));
      policy::Access overwrite;
      overwrite.writes.set(192);
      EXPECT_TRUE(access->conflicts(overwrite));
      overwrite.writes.reset(192);
      overwrite.writes.set(193);
      EXPECT_TRUE(access->conflicts(overwrite));
      EXPECT_FALSE(policy::footprint(*inst, 193));
    }
  }
}

TEST(AsyncMmaPolicyTest, ScaledWmmaOffloadsVectorAndInlineScalesButKeepsScalarScalesInline) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  amdgpu::GpuMemory memory("scaled_wmma_memory");
  amdgpu::L2Cache l2("scaled_wmma_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("scaled_wmma_cu", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);
  for (unsigned scale_a : {128u, 448u, 0u}) {
    for (unsigned scale_b : {128u, 449u, 1u}) {
      SCOPED_TRACE(scale_a);
      SCOPED_TRACE(scale_b);
      const auto words = mma_test::make_cdna5_wmma_scale_words(4, 4, scale_a, scale_b);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      amdgpu::AsyncInstructionWindow window(*cu, *wave);
      const bool submitted = window.submit_mma(inst);
      EXPECT_EQ(submitted, scale_a >= 128 && scale_b >= 128);
      EXPECT_EQ(inst == nullptr, submitted);
      window.drain();
    }
  }
}

struct BlockedMmaState {
  std::binary_semaphore started{0};
  std::binary_semaphore unblock{0};
};

// Retain real decoded operands while replacing arithmetic with a controlled
// delay. The window must honor architectural waits regardless of MMA latency.
class BlockedMma final : public Instruction {
public:
  BlockedMma(const Instruction &decoded, BlockedMmaState &state)
      : Instruction(decoded.mnemonic(),
                    [](Instruction &self, void *) {
                      auto &state = static_cast<BlockedMma &>(self).state_;
                      state.started.release();
                      state.unblock.acquire();
                    }),
        state_(state) {
    num_src_ = decoded.num_src_operands();
    num_dst_ = decoded.num_dst_operands();
    size_ = decoded.size();
    for (unsigned index = 0; index != num_src_; ++index)
      src_operands_[index] = const_cast<Operand *>(decoded.src_operand(index));
    for (unsigned index = 0; index != num_dst_; ++index)
      dst_operands_[index] = const_cast<Operand *>(decoded.dst_operand(index));
  }

private:
  BlockedMmaState &state_;
};

TEST(AsyncMmaPolicyTest, WaitIdleDrainsBlockedMmaBeforeReturningToIssue) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  const auto words =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  std::unique_ptr<Instruction> decoded(decode_valid(*decoder, words.data()));
  amdgpu::GpuMemory memory("wait_idle_memory");
  amdgpu::L2Cache l2("wait_idle_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 128;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("wait_idle_cu", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 106, 128);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);

  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(int(arch));
    auto wait_decoder = Decoder::create(arch);
    const auto wait_words = arch == ROCJITSU_CODE_ARCH_CDNA5
                                ? cdna5::build_sopp(cdna5::kSWaitIdleSopp, {})
                                : rdna4::build_sopp(rdna4::kSWaitIdleSopp, {});
    std::unique_ptr<Instruction> wait(decode_valid(*wait_decoder, wait_words.data()));
    BlockedMmaState state;
    amdgpu::AsyncInstructionWindow window(*cu, *wave);
    std::unique_ptr<Instruction> blocked = std::make_unique<BlockedMma>(*decoded, state);
    ASSERT_TRUE(window.submit_mma(blocked));
    state.started.acquire();

    std::binary_semaphore returned{0};
    bool returned_before_completion = false;
    std::thread releaser([&] {
      returned_before_completion = returned.try_acquire_for(std::chrono::milliseconds(50));
      state.unblock.release();
    });
    window.before(*wait);
    returned.release();
    releaser.join();
    EXPECT_FALSE(returned_before_completion);
    EXPECT_FALSE(window.pending());
    EXPECT_TRUE(window.stopped());
    window.drain();
  }
}
} // namespace
