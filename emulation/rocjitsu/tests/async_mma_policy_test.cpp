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
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/target_provider.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/pm4/pm4.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

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

TEST(AsyncMmaPolicyTest, UnsupportedAsyncWmmaHaltsWithoutHostException) {
  constexpr uint64_t kPc = 0x200000;
  amdgpu::GpuMemory memory("unsupported_async_wmma_memory");
  amdgpu::L2Cache l2("unsupported_async_wmma_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("unsupported_async_wmma_cu", config, &memory, &l2);
  const auto &production_backend = cdna5::execution_backend();
  std::vector<Instruction::ExecuteFn> callbacks(production_backend.instruction_callbacks,
                                                production_backend.instruction_callbacks +
                                                    production_backend.instruction_callback_count);
  callbacks[static_cast<size_t>(cdna5::InstructionExecutionId::VWmmaF3232x16x128F4Vop3p)] =
      [](Instruction &, void *) { throw util::UnimplementedInst("test async execution failure"); };
  const IsaExecutionBackend throwing_backend{
      .instruction_callbacks = callbacks.data(),
      .instruction_callback_count = callbacks.size(),
      .operand_backend = production_backend.operand_backend,
  };
  cu->replace_decoder_for_test(make_isa_decoder<cdna5::Isa>(&throwing_backend));
  auto *wave = cu->dispatch_wf(0, kPc, 106, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);

  const auto first =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  const auto second =
      cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                         {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3});
  for (size_t index = 0; index != first.size(); ++index) {
    memory.write32(kPc + index * sizeof(uint32_t), first[index]);
    memory.write32(kPc + (index + first.size()) * sizeof(uint32_t), second[index]);
  }
  memory.write32(kPc + (first.size() + second.size()) * sizeof(uint32_t),
                 cdna5::build_sopp(cdna5::kSEndpgmSopp)[0]);

  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(cu->step());
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(wave->is_halted());
  EXPECT_NE(output.find("asynchronous v_wmma"), std::string::npos) << output;
  EXPECT_NE(output.find("pc=0x200000"), std::string::npos) << output;
  EXPECT_NE(output.find("unimplemented instruction"), std::string::npos) << output;
  EXPECT_NE(output.find("test async execution failure"), std::string::npos) << output;
}

struct FaultDrainState {
  std::binary_semaphore worker_started{0};
  std::binary_semaphore release_worker{0};
  std::binary_semaphore fault_reached{0};
  std::binary_semaphore failure_published{0};
  std::atomic<bool> worker_done{false};
};

FaultDrainState *fault_drain_state = nullptr;

TEST(AsyncMmaPolicyTest, TerminalMemoryFaultDrainsPendingMmaBeforePublishingFailure) {
  constexpr uint64_t kPc = 0x200000;
  FaultDrainState state;
  amdgpu::GpuMemory memory("fault_drain_memory");
  amdgpu::L2Cache l2("fault_drain_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("fault_drain_cu", config, &memory, &l2);
  const auto &production_backend = cdna5::execution_backend();
  std::vector<Instruction::ExecuteFn> callbacks(production_backend.instruction_callbacks,
                                                production_backend.instruction_callbacks +
                                                    production_backend.instruction_callback_count);
  callbacks[static_cast<size_t>(cdna5::InstructionExecutionId::VWmmaF3232x16x128F4Vop3p)] =
      [](Instruction &, void *) {
        fault_drain_state->worker_started.release();
        fault_drain_state->release_worker.acquire();
        fault_drain_state->worker_done.store(true, std::memory_order_release);
      };
  callbacks[static_cast<size_t>(cdna5::InstructionExecutionId::VWmmaF3216x16x64Fp8Fp8Vop3p)] =
      [](Instruction &, void *) {};
  callbacks[static_cast<size_t>(cdna5::InstructionExecutionId::GlobalLoadB32Vglobal)] =
      [](Instruction &inst, void *) {
        auto data = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
        data->lane_mask = data->exec_mask = 1;
        data->wf_size = 32;
        data->elem_size = 4;
        data->num_elems = 1;
        data->scratch_swizzle = data->requires_scratch_backing = true;
        data->scratch_lane_mask = 1;
        inst.set_data(std::move(data));
        fault_drain_state->fault_reached.release();
      };
  const IsaExecutionBackend fault_backend{
      .instruction_callbacks = callbacks.data(),
      .instruction_callback_count = callbacks.size(),
      .operand_backend = production_backend.operand_backend,
  };
  cu->replace_decoder_for_test(make_isa_decoder<cdna5::Isa>(&fault_backend));
  auto *wave = cu->dispatch_wf(0, kPc, 106, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);
  auto failure = std::make_shared<amdgpu::Pm4FailureState>();
  bool finished_before_failure = false;
  failure->wake = [&] {
    finished_before_failure = state.worker_done.load(std::memory_order_acquire);
    state.failure_published.release();
  };
  wave->set_pm4_failure(failure);

  uint64_t address = kPc;
  const auto emit = [&](auto words) {
    for (uint32_t word : words) {
      memory.write32(address, word);
      address += sizeof(word);
    }
  };
  emit(cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                          {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3}));
  emit(cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                          {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3}));
  emit(cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal, {.saddr = 124, .vdst = 128, .vaddr = 0}));
  emit(cdna5::build_sopp(cdna5::kSEndpgmSopp));

  bool observed_fault = false;
  bool published_before_release = false;
  fault_drain_state = &state;
  std::thread release([&] {
    state.worker_started.acquire();
    observed_fault = state.fault_reached.try_acquire_for(std::chrono::seconds(2));
    published_before_release =
        state.failure_published.try_acquire_for(std::chrono::milliseconds(200));
    state.release_worker.release();
  });
  EXPECT_NO_THROW(cu->step());
  release.join();
  fault_drain_state = nullptr;
  EXPECT_TRUE(observed_fault);
  EXPECT_FALSE(published_before_release);
  EXPECT_TRUE(failure->failed.load());
  EXPECT_TRUE(finished_before_failure);
  EXPECT_TRUE(wave->is_halted());
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

TEST(AsyncMmaPolicyTest, Dpp16InstructionsDrainAndStopAnAsyncWindow) {
  auto mma_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  const auto mma_words =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  std::unique_ptr<Instruction> mma(decode_valid(*mma_decoder, mma_words.data()));

  auto cdna4_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto vop1_base =
      cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = amdgpu::SRC_DPP, .vdst = 202});
  auto vop1_raw =
      std::bit_cast<cdna4::Vop1VopDppMachineInst>(std::array<uint32_t, 2>{vop1_base[0], 0});
  vop1_raw.vsrc0 = 200;
  vop1_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  vop1_raw.bank_mask = vop1_raw.row_mask = 0xf;

  const auto vop2_base =
      cdna4::build_vop2(cdna4::kVAddF32Vop2, {.src0 = amdgpu::SRC_DPP, .vsrc1 = 201, .vdst = 202});
  auto vop2_raw =
      std::bit_cast<cdna4::Vop2VopDppMachineInst>(std::array<uint32_t, 2>{vop2_base[0], 0});
  vop2_raw.vsrc0 = 200;
  vop2_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  vop2_raw.bank_mask = vop2_raw.row_mask = 0xf;

  auto cdna5_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  const auto vop3_base = cdna5::build_vop3(
      cdna5::kVAddF32Vop3,
      {.vdst = 202, .src0 = amdgpu::SRC_DPP, .src1 = 256 + 201, .src2 = 256 + 202});
  auto vop3_raw = std::bit_cast<cdna5::Vop3VopDpp16MachineInst>(
      std::array<uint32_t, 3>{vop3_base[0], vop3_base[1], 0});
  vop3_raw.vsrc0 = 200;
  vop3_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  vop3_raw.bank_mask = vop3_raw.row_mask = 0xf;

  std::vector<std::unique_ptr<Instruction>> boundaries;
  boundaries.emplace_back(
      decode_valid(*cdna4_decoder, reinterpret_cast<const uint32_t *>(&vop1_raw)));
  boundaries.emplace_back(
      decode_valid(*cdna4_decoder, reinterpret_cast<const uint32_t *>(&vop2_raw)));
  boundaries.emplace_back(
      decode_valid(*cdna5_decoder, reinterpret_cast<const uint32_t *>(&vop3_raw)));

  amdgpu::GpuMemory memory("dpp_boundary_memory");
  amdgpu::L2Cache l2("dpp_boundary_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("dpp_boundary_cu", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);

  for (const auto &boundary : boundaries) {
    SCOPED_TRACE(boundary->disassemble());
    amdgpu::RegisterModifiers modifiers;
    boundary->amdgpu_register_modifiers(modifiers);
    ASSERT_TRUE(modifiers.dpp);
    ASSERT_FALSE(modifiers.dpp8);

    BlockedMmaState state;
    amdgpu::AsyncInstructionWindow window(*cu, *wave);
    std::unique_ptr<Instruction> blocked = std::make_unique<BlockedMma>(*mma, state);
    ASSERT_TRUE(window.submit_mma(blocked));
    state.started.acquire();

    std::binary_semaphore returned{0};
    bool returned_before_completion = false;
    std::thread releaser([&] {
      returned_before_completion = returned.try_acquire_for(std::chrono::milliseconds(50));
      state.unblock.release();
    });
    window.before(*boundary);
    returned.release();
    releaser.join();

    EXPECT_FALSE(returned_before_completion);
    EXPECT_FALSE(window.pending());
    EXPECT_TRUE(window.stopped());
    window.drain();
  }
}

TEST(AsyncMmaPolicyTest, IndependentPlainAndDpp8InstructionsKeepTheAsyncWindowOpen) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  const auto mma_words =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  std::unique_ptr<Instruction> mma(decode_valid(*decoder, mma_words.data()));

  const auto plain_words = cdna5::build_vop3(
      cdna5::kVAddF32Vop3, {.vdst = 202, .src0 = 256 + 200, .src1 = 256 + 201, .src2 = 256 + 202});
  std::unique_ptr<Instruction> plain(decode_valid(*decoder, plain_words.data()));
  const auto dpp8_base =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = amdgpu::SRC_DPP8_FI_0, .vdst = 202});
  auto dpp8_raw =
      std::bit_cast<cdna5::Vop1VopDpp8MachineInst>(std::array<uint32_t, 2>{dpp8_base[0], 0});
  dpp8_raw.vsrc0 = 200;
  dpp8_raw.lane_sel_0 = 0;
  dpp8_raw.lane_sel_1 = 1;
  dpp8_raw.lane_sel_2 = 2;
  dpp8_raw.lane_sel_3 = 3;
  dpp8_raw.lane_sel_4 = 4;
  dpp8_raw.lane_sel_5 = 5;
  dpp8_raw.lane_sel_6 = 6;
  dpp8_raw.lane_sel_7 = 7;
  std::unique_ptr<Instruction> dpp8(
      decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&dpp8_raw)));
  amdgpu::RegisterModifiers dpp8_modifiers;
  dpp8->amdgpu_register_modifiers(dpp8_modifiers);
  ASSERT_FALSE(dpp8_modifiers.dpp);
  ASSERT_TRUE(dpp8_modifiers.dpp8);

  amdgpu::GpuMemory memory("dpp_control_memory");
  amdgpu::L2Cache l2("dpp_control_l2");
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_CDNA5;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.async_resources = std::make_shared<mc::ExecutionResources>(1);
  auto cu = amdgpu::ComputeUnitCore::create("dpp_control_cu", config, &memory, &l2);
  auto *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  wave->set_exec(0xffffffff);

  for (const auto *control : {plain.get(), dpp8.get()}) {
    SCOPED_TRACE(control->disassemble());
    BlockedMmaState state;
    amdgpu::AsyncInstructionWindow window(*cu, *wave);
    std::unique_ptr<Instruction> blocked = std::make_unique<BlockedMma>(*mma, state);
    ASSERT_TRUE(window.submit_mma(blocked));
    state.started.acquire();

    window.before(*control);
    EXPECT_TRUE(window.pending());
    EXPECT_FALSE(window.stopped());
    state.unblock.release();
    window.drain();
  }
}

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
