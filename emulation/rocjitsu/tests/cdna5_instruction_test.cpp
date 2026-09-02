// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "decode_test_util.h"

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::cdna5;

class BarrierSpanPlugin final : public ExecutionPlugin {
public:
  BarrierSpanPlugin() : ExecutionPlugin("barrier_span") {}

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override {
    span_sizes.push_back(wavefronts.size());
    std::vector<uint32_t> workgroups;
    for (const auto *wf : wavefronts)
      workgroups.push_back(wf->wg_id());
    std::ranges::sort(workgroups);
    workgroups.erase(std::unique(workgroups.begin(), workgroups.end()), workgroups.end());
    workgroup_ids.push_back(std::move(workgroups));
  }

  std::vector<size_t> span_sizes;
  std::vector<std::vector<uint32_t>> workgroup_ids;
};

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

private:
  bool original_;
};

TEST(Gfx1250SimulationTest, VClsI32CountsLeadingSignBits) {
  struct Case {
    uint32_t input;
    uint32_t expected;
  };
  constexpr std::array cases{
      Case{0x00000000u, 0xFFFFFFFFu}, Case{0xFFFFFFFFu, 0xFFFFFFFFu}, Case{0x00000001u, 31u},
      Case{0x00000002u, 30u},         Case{0x00000003u, 30u},         Case{0x00000007u, 29u},
      Case{0x0000FFFFu, 16u},         Case{0x40000000u, 1u},          Case{0x7FFFFFFFu, 1u},
      Case{0x80000000u, 1u},          Case{0xFFFFFFFEu, 31u},         Case{0xFFFFFC00u, 22u},
  };
  constexpr uint32_t kSrc = 0;
  constexpr uint32_t kDst = 2;
  const auto vop1 = cdna5::build_vop1(cdna5::kVClsI32Vop1, {.src0 = 256 + kSrc, .vdst = kDst});
  const auto vop3 = cdna5::build_vop3(cdna5::kVClsI32Vop3, {.vdst = kDst, .src0 = 256 + kSrc});

  ForceScalarGuard force_scalar_guard;
  for (const bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");

    Gfx1250Sim sim;
    auto *wf = sim.dispatch_scratch_wf();
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 32u);
    wf->set_exec((uint64_t{1} << cases.size()) - 1);

    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < cases.size(); ++lane)
      sim.cu()->write_vgpr(vgpr_base + kSrc, lane, cases[lane].input);

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto check_encoding = [&](const auto &words, std::string_view expected_mnemonic) {
      SCOPED_TRACE(expected_mnemonic);
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(std::string_view(inst->mnemonic()), expected_mnemonic);

      for (uint32_t lane = 0; lane < cases.size(); ++lane)
        sim.cu()->write_vgpr(vgpr_base + kDst, lane, 0xDEADBEEFu);
      sim.cu()->execute_instruction(inst.get(), *wf);

      for (uint32_t lane = 0; lane < cases.size(); ++lane) {
        EXPECT_EQ(sim.cu()->read_vgpr(vgpr_base + kDst, lane), cases[lane].expected)
            << "lane " << lane << ", input 0x" << std::hex << cases[lane].input;
      }
    };

    check_encoding(vop1, "v_cls_i32_e32");
    check_encoding(vop3, "v_cls_i32");
  }
}

TEST(Gfx1250SimulationTest, SGetPcI64ReturnsNextInstructionAddress) {
  constexpr uint64_t kKernelAddr = 0x10000;
  const uint32_t code[] = {
      0xBE844700u, // s_get_pc_i64 s[4:5]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code, std::size(code));
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  EXPECT_EQ(wf.sgpr64(4), entry_pc + sizeof(uint32_t));
}

TEST(Gfx1250SimulationTest, SAddPcI64SkipsRelativeToNextPc) {
  const uint32_t code[] = {
      0xBE804B84u, // s_add_pc_i64 4
      0xBE840081u, // s_mov_b32 s4, 1
      0xBE840082u, // s_mov_b32 s4, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(4), 2u);
}

TEST(Gfx1250SimulationTest, SAddPcI64NegativeLiteralJumpsBackwardFromNextPc) {
  const auto initial_branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 2});
  const auto set_result = cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 130, .sdst = 4});
  const auto add_pc = cdna5::build_sop1(cdna5::kSAddPcI64Sop1, {.ssrc0 = 255});
  const std::array code{
      initial_branch[0], // Branch forward to s_add_pc_i64.
      set_result[0],     // Backward-jump target: s_mov_b32 s4, 2.
      S_ENDPGM_GFX12,
      add_pc[0],  // s_add_pc_i64 0xfffffff0.
      0xfffffff0u // -16 bytes from the next PC reaches set_result.
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code.data(), code.size());
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(4), 2u);
}

TEST(Gfx1250SimulationTest, SAddPcI64WrapsAtUnsignedBoundaries) {
  const auto add_one = cdna5::build_sop1(cdna5::kSAddPcI64Sop1, {.ssrc0 = 129});
  const auto add_minus_one = cdna5::build_sop1(cdna5::kSAddPcI64Sop1, {.ssrc0 = 193});

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> increment(decode_valid(*decoder, add_one.data()));
  std::unique_ptr<Instruction> decrement(decode_valid(*decoder, add_minus_one.data()));
  ASSERT_NE(increment, nullptr);
  ASSERT_NE(decrement, nullptr);

  Gfx1250Sim sim;
  auto *wf = sim.cu()->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf, nullptr);

  wf->pc = 0x7fffffffffffffffULL;
  sim.cu()->execute_instruction(increment.get(), *wf);
  EXPECT_EQ(wf->pc, 0x8000000000000000ULL);

  wf->pc = 0;
  sim.cu()->execute_instruction(decrement.get(), *wf);
  EXPECT_EQ(wf->pc, 0xffffffffffffffffULL);

  wf->pc = 0x8000000000000000ULL;
  sim.cu()->execute_instruction(decrement.get(), *wf);
  EXPECT_EQ(wf->pc, 0x7fffffffffffffffULL);
}

TEST(Gfx1250SimulationTest, SSetPcI64JumpsToScalarAddress) {
  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint32_t kTargetWord = 5;
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  uint64_t target_pc = entry_pc + kTargetWord * sizeof(uint32_t);
  std::vector<uint32_t> code = {
      0xBE8400FFu,    static_cast<uint32_t>(target_pc), // s_mov_b32 s4, target_pc[31:0]
      0xBE850080u,                                      // s_mov_b32 s5, 0
      0xBE804804u,                                      // s_set_pc_i64 s[4:5]
      0xBE860081u,                                      // s_mov_b32 s6, 1
      0xBE860082u,                                      // s_mov_b32 s6, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size());
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  EXPECT_EQ(sim.snapshot->snapshots().front().sgpr(6), 2u);
}

TEST(Gfx1250SimulationTest, SSwapPcI64StoresReturnAddressAndJumps) {
  constexpr uint64_t kKernelAddr = 0x10000;
  constexpr uint32_t kReturnWord = 4;
  constexpr uint32_t kTargetWord = 5;
  uint64_t entry_pc = kKernelAddr + sizeof(rocr::llvm::amdhsa::kernel_descriptor_t);
  uint64_t return_pc = entry_pc + kReturnWord * sizeof(uint32_t);
  uint64_t target_pc = entry_pc + kTargetWord * sizeof(uint32_t);
  std::vector<uint32_t> code = {
      0xBE8400FFu,    static_cast<uint32_t>(target_pc), // s_mov_b32 s4, target_pc[31:0]
      0xBE850080u,                                      // s_mov_b32 s5, 0
      0xBE864904u,                                      // s_swap_pc_i64 s[6:7], s[4:5]
      0xBE880081u,                                      // s_mov_b32 s8, 1
      0xBE880082u,                                      // s_mov_b32 s8, 2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code.data(), code.size());
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 32, 32);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 1u);
  const auto &wf = sim.snapshot->snapshots().front();
  EXPECT_EQ(wf.sgpr64(6), return_pc);
  EXPECT_EQ(wf.sgpr(8), 2u);
}

TEST(Gfx1250SimulationTest, SBitreplicateB64B32DuplicatesEachSourceBit) {
  const uint32_t code[] = {
      0xBE8200FFu,
      0x80000001u, // s_mov_b32 s2, 0x80000001
      0xBE841402u, // s_bitreplicate_b64_b32 s[4:5], s2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr64(4), 0xC000000000000003ULL);
}

TEST(Gfx1250SimulationTest, SGetShaderCyclesU64ReadsSimulationTime) {
  const uint32_t code[] = {
      0xBE840600u, // s_get_shader_cycles_u64 s[4:5]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  sim.engine->schedule_event_async(sim.cp()->doorbell_event(), 17);
  ASSERT_TRUE(sim.engine->step());
  ASSERT_EQ(sim.engine->global_time(), 17u);

  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  const auto observed_time = wf->sgpr64(4);
  EXPECT_GE(observed_time, 17u);
  EXPECT_LE(observed_time, sim.engine->global_time());
}

TEST(Gfx1250SimulationTest, SSendmsgRtnB64ReadsRealtimeAndB32UsesPlaceholder) {
  const uint32_t code[] = {
      0xBE844D83u, // s_sendmsg_rtn_b64 s[4:5], sendmsg(MSG_RTN_GET_REALTIME)
      0xBE864C80u, // s_sendmsg_rtn_b32 s6, sendmsg(MSG_RTN_GET_DOORBELL)
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  sim.engine->schedule_event_async(sim.cp()->doorbell_event(), 23);
  ASSERT_TRUE(sim.engine->step());
  ASSERT_EQ(sim.engine->global_time(), 23u);

  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  const auto observed_time = wf->sgpr64(4);
  EXPECT_GE(observed_time, 23u);
  EXPECT_LE(observed_time, sim.engine->global_time());
  EXPECT_EQ(wf->sgpr(6), 0u);
}

TEST(Gfx1250SimulationTest, SMovrelsReadsM0IndexedScalarSources) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8200FFu,    0x11111111u, // s_mov_b32 s2, 0x11111111
      0xBE8300FFu,    0x22222222u, // s_mov_b32 s3, 0x22222222
      0xBE8400FFu,    0x33333333u, // s_mov_b32 s4, 0x33333333
      0xBE8500FFu,    0x44444444u, // s_mov_b32 s5, 0x44444444
      0xBE884002u,                 // s_movrels_b32 s8, s2
      0xBE8A4102u,                 // s_movrels_b64 s[10:11], s[2:3]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(8), 0x22222222u);
  EXPECT_EQ(wf->sgpr64(10), 0x4444444433333333ULL);
}

TEST(Gfx1250SimulationTest, SMovreldWritesM0IndexedScalarDestinations) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8200FFu,    0x55555555u, // s_mov_b32 s2, 0x55555555
      0xBE8300FFu,    0x66666666u, // s_mov_b32 s3, 0x66666666
      0xBE884202u,                 // s_movreld_b32 s8, s2
      0xBE8A4302u,                 // s_movreld_b64 s[10:11], s[2:3]
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(9), 0x55555555u);
  EXPECT_EQ(wf->sgpr64(12), 0x6666666655555555ULL);
}

TEST(Gfx1250SimulationTest, SMovrelsd2B32UsesSeparatePackedM0Offsets) {
  const uint32_t code[] = {
      0xBEFD00FFu,    0x00000201u, // s_mov_b32 m0, 0x201
      0xBE8200FFu,    0x77777777u, // s_mov_b32 s2, 0x77777777
      0xBE8300FFu,    0x88888888u, // s_mov_b32 s3, 0x88888888
      0xBE884402u,                 // s_movrelsd_2_b32 s8, s2
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(10), 0x88888888u);
}

TEST(Gfx1250SimulationTest, SplitNamedBarrierOpsReportIdleState) {
  const uint32_t code[] = {
      0xBEFD0081u,                 // s_mov_b32 m0, 1
      0xBE8400FFu,    0xFFFFFFFFu, // s_mov_b32 s4, -1
      0xBE80517Du,                 // s_barrier_init m0
      0xBE80527Du,                 // s_barrier_join m0
      0xBE80577Du,                 // s_wakeup_barrier m0
      0xBE84507Du,                 // s_get_barrier_state s4, m0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code));
  ASSERT_NE(wf, nullptr);
  EXPECT_EQ(wf->sgpr(4), 0u);
}

TEST(Gfx1250SimulationTest, NamedBarrierSignalIsfirstPreservesSccWithoutAllocation) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> signal_words = {0xBE804F7Du};
  std::unique_ptr<Instruction> signal(decode_valid(*decoder, signal_words.data()));
  ASSERT_NE(signal, nullptr);

  wf->set_m0((2u << 16) | 1u);
  wf->write_scc(true);
  sim.cu()->execute_instruction(signal.get(), *wf);
  EXPECT_TRUE(wf->read_scc());
}

void expect_barrier_init_reads_implicit_m0(uint32_t init_encoding, uint32_t init_m0) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2, 4);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> init_words = {init_encoding};
  const std::array<uint32_t, 1> join_words = {0xBE805281u};   // s_barrier_join 1
  const std::array<uint32_t, 1> signal_words = {0xBE804F81u}; // s_barrier_signal_isfirst 1
  const std::array<uint32_t, 1> state_words = {0xBE845081u};  // s_get_barrier_state s4, 1
  std::unique_ptr<Instruction> init(decode_valid(*decoder, init_words.data()));
  std::unique_ptr<Instruction> join(decode_valid(*decoder, join_words.data()));
  std::unique_ptr<Instruction> signal(decode_valid(*decoder, signal_words.data()));
  std::unique_ptr<Instruction> state(decode_valid(*decoder, state_words.data()));
  ASSERT_NE(init, nullptr);
  ASSERT_NE(join, nullptr);
  ASSERT_NE(signal, nullptr);
  ASSERT_NE(state, nullptr);

  wf0->set_m0(init_m0);
  cu->execute_instruction(init.get(), *wf0);
  for (auto *wf : {wf0, wf1})
    cu->execute_instruction(join.get(), *wf);

  cu->execute_instruction(signal.get(), *wf0);
  EXPECT_TRUE(wf0->read_scc());
  cu->execute_instruction(state.get(), *wf0);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf0, 4), 0x01010021u);
  wf0->barrier_wait(1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);

  cu->execute_instruction(signal.get(), *wf1);
  EXPECT_FALSE(wf1->read_scc());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  cu->execute_instruction(state.get(), *wf1);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf1, 4), 0x01000021u);
}

TEST(Gfx1250SimulationTest, NamedBarrierInitImmediateReadsMemberCountFromImplicitM0) {
  expect_barrier_init_reads_implicit_m0(0xBE805181u, 2u << 16); // s_barrier_init 1
}

TEST(Gfx1250SimulationTest, NamedBarrierInitM0ReadsMemberCountFromImplicitM0) {
  expect_barrier_init_reads_implicit_m0(0xBE80517Du, (2u << 16) | 1u); // s_barrier_init m0
}

TEST(Gfx1250SimulationTest, NamedBarrierSynchronizesJoinedWavesAcrossPhases) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2, 4);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> join_words = {0xBE80527Du};
  const std::array<uint32_t, 1> signal_words = {0xBE804F7Du};
  const std::array<uint32_t, 1> wait_words = {0xBF940001u};
  const std::array<uint32_t, 1> state_words = {0xBE84507Du};
  std::unique_ptr<Instruction> join(decode_valid(*decoder, join_words.data()));
  std::unique_ptr<Instruction> signal(decode_valid(*decoder, signal_words.data()));
  std::unique_ptr<Instruction> wait(decode_valid(*decoder, wait_words.data()));
  std::unique_ptr<Instruction> state(decode_valid(*decoder, state_words.data()));
  ASSERT_NE(join, nullptr);
  ASSERT_NE(signal, nullptr);
  ASSERT_NE(wait, nullptr);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(std::string_view(join->mnemonic()), "s_barrier_join");
  ASSERT_EQ(std::string_view(signal->mnemonic()), "s_barrier_signal_isfirst");
  ASSERT_EQ(std::string_view(wait->mnemonic()), "s_barrier_wait");
  ASSERT_EQ(std::string_view(state->mnemonic()), "s_get_barrier_state");

  for (auto *wf : {wf0, wf1}) {
    wf->set_m0(1);
    cu->execute_instruction(join.get(), *wf);
    wf->set_m0((2u << 16) | 1u);
  }

  cu->execute_instruction(signal.get(), *wf0);
  EXPECT_TRUE(wf0->read_scc());
  cu->execute_instruction(wait.get(), *wf0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);

  cu->execute_instruction(signal.get(), *wf1);
  EXPECT_FALSE(wf1->read_scc());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  cu->execute_instruction(wait.get(), *wf1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);

  cu->execute_instruction(state.get(), *wf0);
  EXPECT_EQ(read_wave_sgpr(*cu, *wf0, 4), 0x01000021u);

  cu->execute_instruction(signal.get(), *wf1);
  EXPECT_TRUE(wf1->read_scc());
  cu->execute_instruction(wait.get(), *wf1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::BARRIER);

  cu->execute_instruction(signal.get(), *wf0);
  EXPECT_FALSE(wf0->read_scc());
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
  cu->execute_instruction(wait.get(), *wf0);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
}

TEST(Gfx1250SimulationTest, WorkgroupBarrierSignalAndWaitUseDedicatedCompletionState) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2, 4);

  EXPECT_TRUE(wf0->barrier_signal(-1, 0));
  EXPECT_EQ(wf0->barrier_state(-1), 0x01010021u);
  wf0->barrier_wait(-1);
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);

  EXPECT_FALSE(wf1->barrier_signal(-1, 0));
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf1->barrier_state(-1), 0x01000021u);
  wf1->barrier_wait(-1);
  EXPECT_EQ(wf1->state(), amdgpu::WfState::RUNNING);
}

TEST(Gfx1250SimulationTest, WorkgroupBarrierCompletesAfterUnsignaledWaveTerminates) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *waiting = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *terminating = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(waiting, nullptr);
  ASSERT_NE(terminating, nullptr);
  cu->begin_workgroup(0, 0, 2);

  EXPECT_TRUE(waiting->barrier_signal(-1, 0));
  waiting->barrier_wait(-1);
  EXPECT_EQ(waiting->state(), amdgpu::WfState::BARRIER);

  terminating->halt();
  EXPECT_EQ(waiting->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(waiting->barrier_state(-1), 0x00000011u);
}

TEST(Gfx1250SimulationTest, NamedBarrierMembershipPersistsAfterJoinedWaveTerminates) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *waiting = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *terminating = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(waiting, nullptr);
  ASSERT_NE(terminating, nullptr);
  cu->begin_workgroup(0, 0, 2, 4);

  waiting->barrier_init(1, 2);
  waiting->barrier_join(1);
  terminating->barrier_join(1);
  EXPECT_TRUE(waiting->barrier_signal(1, 0));
  waiting->barrier_wait(1);
  EXPECT_EQ(waiting->state(), amdgpu::WfState::BARRIER);

  terminating->halt();
  EXPECT_EQ(waiting->state(), amdgpu::WfState::BARRIER);
  EXPECT_EQ(waiting->barrier_state(1), 0x01010021u);
}

TEST(Gfx1250SimulationTest, WorkgroupAndTrapBarriersKeepIndependentSignalCounts) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2);

  EXPECT_TRUE(wf0->barrier_signal(-1, 0));
  EXPECT_EQ(wf0->barrier_state(-1), 0x00010021u);
  EXPECT_EQ(wf0->barrier_state(-2), 0u);

  wf0->set_status_raw(wf0->status_raw() | (1u << 5));
  EXPECT_TRUE(wf0->barrier_signal(-2, 0));
  EXPECT_EQ(wf0->barrier_state(-2), 0x00010021u);
  EXPECT_EQ(wf0->barrier_state(-1), 0x00010021u);
}

TEST(Gfx1250SimulationTest, WorkgroupSignalIsfirstAcceptsNegativeInlineBarrierId) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> signal_words = {0xBE804FC1u}; // s_barrier_signal_isfirst -1
  std::unique_ptr<Instruction> signal(decode_valid(*decoder, signal_words.data()));
  ASSERT_NE(signal, nullptr);

  cu->execute_instruction(signal.get(), *wf0);
  EXPECT_TRUE(wf0->read_scc());
  cu->execute_instruction(signal.get(), *wf1);
  EXPECT_FALSE(wf1->read_scc());
}

TEST(Gfx1250SimulationTest, NamedBarrierLeaveCompletesReducedMembership) {
  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf0 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  auto *wf1 = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(wf0, nullptr);
  ASSERT_NE(wf1, nullptr);
  cu->begin_workgroup(0, 0, 2, 4);

  wf0->barrier_init(1, 2);
  EXPECT_EQ(wf0->barrier_state(1), 0x01000021u);
  wf0->barrier_join(1);
  wf1->barrier_join(1);
  EXPECT_TRUE(wf0->barrier_signal(1, 0));
  wf0->barrier_wait(7); // Any nonnegative immediate waits on the joined named barrier.
  EXPECT_EQ(wf0->state(), amdgpu::WfState::BARRIER);

  EXPECT_FALSE(wf1->barrier_leave());
  EXPECT_EQ(wf0->state(), amdgpu::WfState::RUNNING);
  EXPECT_EQ(wf0->barrier_state(1), 0x01000011u);
  EXPECT_TRUE(wf0->barrier_leave());
  EXPECT_EQ(wf0->barrier_state(1), 0x01000001u);
}

TEST(Gfx1250SimulationTest, AqlDescriptorAllocatesNamedBarriersForTwoWaveKernel) {
  constexpr uint64_t kKernelAddr = 0x10000;
  const uint32_t code[] = {
      0xBEFD0081u, // s_mov_b32 m0, 1
      0xBE80527Du, // s_barrier_join m0
      0xBEFD00FFu,
      0x00020001u, // s_mov_b32 m0, (2 << 16) | 1
      0xBE804E7Du, // s_barrier_signal m0
      0xBF940007u, // s_barrier_wait 7 (the joined named barrier)
      0xBE84507Du, // s_get_barrier_state s4, m0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(kKernelAddr, code, std::size(code), 104, 32, 2, false,
                                            false, false, 0, 0, 0, 0, 0, 1);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel_object, 64, 64);
  step_until_halted(*sim.engine, *sim.cu());

  ASSERT_EQ(sim.snapshot->snapshots().size(), 2u);
  for (const auto &wf : sim.snapshot->snapshots())
    EXPECT_EQ(wf.sgpr(4), 0x01000021u);
}

TEST(Gfx1250SimulationTest, ClusterBarrierSynchronizesWorkgroupsAcrossComputeUnits) {
  constexpr auto read_first_workitem =
      cdna5::build_vop1(cdna5::kVReadfirstlaneB32Vop1, {.src0 = 256, .vdst = 4});
  constexpr auto is_first_wave =
      cdna5::build_sopc(cdna5::kSCmpEqU32Sopc, {.ssrc0 = 4, .ssrc1 = 128});
  constexpr auto skip_signal = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 1});
  const uint32_t code[] = {
      read_first_workitem[0], // v_readfirstlane_b32 s4, v0
      is_first_wave[0],       // s_cmp_eq_u32 s4, 0
      skip_signal[0],         // s_cbranch_scc0 wait
      0xBE804EC3u,            // s_barrier_signal -3
      0xBF94FFFDu,            // s_barrier_wait -3
      0xBE8450C3u,            // s_get_barrier_state s4, -3
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  auto plugin = std::make_unique<BarrierSpanPlugin>();
  auto *barrier_span = plugin.get();
  ASSERT_TRUE(sim.plugin_group->add(std::move(plugin)));
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code), 104, 32, 2, false,
                                            false, false, 0, 0, 0, 0, 0, 1);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/64);
  step_until_xcd_halted(sim);

  ASSERT_EQ(sim.snapshot->snapshots().size(), 4u);
  for (const auto &wf : sim.snapshot->snapshots()) {
    const uint32_t state = wf.sgpr(4);
    const uint32_t member_count = (state >> 4) & 0x7fu;
    EXPECT_EQ(state & 0x07000001u, 0x01000001u);
    EXPECT_EQ((state >> 16) & 0x7fu, 0u);
    EXPECT_TRUE(member_count == 1 || member_count == 2);
  }
  ASSERT_EQ(barrier_span->span_sizes.size(), 1u);
  EXPECT_EQ(barrier_span->span_sizes[0], 4u);
  EXPECT_EQ(barrier_span->workgroup_ids[0], (std::vector<uint32_t>{0, 1}));
}

TEST(Gfx1250SimulationTest, ClusterBarrierCompletesAfterWorkgroupTerminatesEarly) {
  constexpr auto is_first_workgroup =
      cdna5::build_sopc(cdna5::kSCmpEqU32Sopc, {.ssrc0 = 2, .ssrc1 = 128});
  constexpr auto exit_first_workgroup = cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 2});
  const uint32_t code[] = {
      is_first_workgroup[0],   // s_cmp_eq_u32 s2, 0
      exit_first_workgroup[0], // s_cbranch_scc1 end
      0xBE804EC3u,             // s_barrier_signal -3
      0xBF94FFFDu,             // s_barrier_wait -3
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  uint64_t kernel_object = sim.write_kernel(0x10000, code, std::size(code));
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch_clustered(kernel_object, /*cluster_count_x=*/1, /*cluster_size_x=*/2,
                           /*workgroup_size_x=*/32);
  step_until_xcd_halted(sim);

  ASSERT_EQ(sim.snapshot->snapshots().size(), 2u);
}

TEST(Gfx1250SimulationTest, VgprMsbModeTracksModeRegisterLayout) {
  Gfx1250Sim sim;
  // Resident wave: this test mutates and reads live wavefront MODE state.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);

  constexpr uint8_t kSetLayout = 0xB9;  // src0=1, src1=2, src2=3, dst=2.
  constexpr uint8_t kModeLayout = 0xE6; // dst,src0,src1,src2 packed in MODE.
  static_assert(amdgpu::set_vgpr_msb_to_mode_layout(kSetLayout) == kModeLayout);
  static_assert(amdgpu::mode_layout_to_set_vgpr_msb(kModeLayout) == kSetLayout);
  for (uint32_t layout = 0; layout <= 0xFF; ++layout) {
    EXPECT_EQ(amdgpu::mode_layout_to_set_vgpr_msb(
                  amdgpu::set_vgpr_msb_to_mode_layout(static_cast<uint8_t>(layout))),
              layout);
    EXPECT_EQ(amdgpu::set_vgpr_msb_to_mode_layout(
                  amdgpu::mode_layout_to_set_vgpr_msb(static_cast<uint8_t>(layout))),
              layout);
  }

  wf->set_vgpr_msb_mode(kSetLayout);
  EXPECT_EQ(wf->vgpr_msb_mode(), kSetLayout);
  EXPECT_EQ((wf->mode_raw() & amdgpu::VGPR_MSB_MODE_MASK) >> amdgpu::VGPR_MSB_MODE_SHIFT,
            kModeLayout);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src0), 1u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src1), 2u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Src2), 3u);
  EXPECT_EQ(wf->vgpr_msb_for_role(amdgpu::VgprMsbRole::Dst), 2u);

  wf->set_mode_raw(kModeLayout << amdgpu::VGPR_MSB_MODE_SHIFT);
  EXPECT_EQ(wf->vgpr_msb_mode(), kSetLayout);
}

TEST(Gfx1250SimulationTest, SSetVgprMsbUpdatesWavefrontMode) {
  constexpr uint8_t kSetLayout = 0xB9;
  constexpr uint8_t kModeLayout = amdgpu::set_vgpr_msb_to_mode_layout(kSetLayout);
  Gfx1250Sim sim;
  const uint32_t code[] = {S_SET_VGPR_MSB | kSetLayout, S_ENDPGM_GFX12};

  // The s_set_vgpr_msb result lives in the wavefront MODE register; capture it at
  // halt so it survives the wave freeing itself.
  auto *snapshot = sim.snapshot;
  uint64_t kernel_object =
      sim.write_kernel(0x10000, code, std::size(code), 104, kGfx1250Wave32VgprAllocation);
  {
    test::AqlQueue queue(sim.memory, sim.cp());
    queue.dispatch(kernel_object, 32, 32);
  }
  step_until_halted(*sim.engine, *sim.cu());
  ASSERT_EQ(snapshot->snapshots().size(), 1u);
  const auto &wf = snapshot->snapshots().front();

  EXPECT_EQ(wf.vgpr_msb_mode, kSetLayout);
  EXPECT_EQ((wf.mode_raw & amdgpu::VGPR_MSB_MODE_MASK) >> amdgpu::VGPR_MSB_MODE_SHIFT, kModeLayout);
}

TEST(Gfx1250SimulationTest, VgprMsbRolesSelectHighVgprBanks) {
  Gfx1250Sim sim;
  // Resident wave: this test drives Operand read/write against live VGPR banks.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kLane = 0;
  constexpr uint8_t kSetLayout = 0xB9; // src0=1, src1=2, src2=3, dst=2.
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(kSetLayout);
  cu.write_vgpr(vb + 5, kLane, 0xDEADBEEFu);
  cu.write_vgpr(vb + 1 * 256 + 2, kLane, 0x11111111u);
  cu.write_vgpr(vb + 2 * 256 + 3, kLane, 0x22222222u);
  cu.write_vgpr(vb + 3 * 256 + 4, kLane, 0x33333333u);

  cdna5::Operand src0(32, cdna5::OperandType::OPR_SRC, 256 + 2);
  cdna5::Operand src1(32, cdna5::OperandType::OPR_VGPR, 3);
  cdna5::Operand src2(32, cdna5::OperandType::OPR_SRC_VGPR, 256 + 4);
  cdna5::Operand dst(32, cdna5::OperandType::OPR_VGPR, 5);
  src0.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
  src1.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src1);
  src2.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src2);
  dst.set_vgpr_msb_role(amdgpu::VgprMsbRole::Dst);

  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(src0, kLane), 0x11111111u);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(src1, kLane), 0x22222222u);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(src2, kLane), 0x33333333u);

  amdgpu::RegisterAccess(*wf).write_lane(dst, kLane, 0x44444444u);
  EXPECT_EQ(cu.read_vgpr(vb + 2 * 256 + 5, kLane), 0x44444444u);
  EXPECT_EQ(cu.read_vgpr(vb + 5, kLane), 0xDEADBEEFu);
}

TEST(Gfx1250SimulationTest, VMovDppReadsSelectedHighVgprBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kSrc = 7;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kHighBank = 256;
  constexpr uint32_t kHighValueBase = 0x42000000u;
  constexpr uint32_t kLowAliasBase = 0xDEAD0000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  for (const uint8_t mode : {uint8_t{0x01}, uint8_t{0x41}}) {
    SCOPED_TRACE(testing::Message() << "vgpr_msb_mode=" << static_cast<uint32_t>(mode));
    wf->set_vgpr_msb_mode(mode); // SRC0=bank 1; DST=bank 0 or bank 1.
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu.write_vgpr(vb + kSrc, lane, kLowAliasBase + lane);
      cu.write_vgpr(vb + kHighBank + kSrc, lane, kHighValueBase + lane);
      cu.write_vgpr(vb + kDst, lane, 0u);
      cu.write_vgpr(vb + kHighBank + kDst, lane, 0u);
    }

    cdna5::VMovB32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
    inst.execute_impl(*wf);

    const uint32_t dst_bank = (mode >> 6) & 0x3u;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      EXPECT_EQ(cu.read_vgpr(vb + dst_bank * kHighBank + kDst, lane), kHighValueBase + (lane ^ 1u))
          << "lane " << lane;
    }
  }
}

TEST(Gfx1250SimulationTest, VMovDpp8ReadsSelectedHighVgprBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kSrc = 7;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kHighBank = 256;
  constexpr uint32_t kHighValueBase = 0x43000000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x01); // SRC0=bank 1; DST=bank 0.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kSrc, lane, 0u);
    cu.write_vgpr(vb + kHighBank + kSrc, lane, kHighValueBase + lane);
  }

  cdna5::Vop1VopDpp8MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP8_FI_1;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.lane_sel_0 = 1;
  raw.lane_sel_1 = 0;
  raw.lane_sel_2 = 3;
  raw.lane_sel_3 = 2;
  raw.lane_sel_4 = 5;
  raw.lane_sel_5 = 4;
  raw.lane_sel_6 = 7;
  raw.lane_sel_7 = 6;

  cdna5::VMovB32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kHighValueBase + (lane ^ 1u)) << "lane " << lane;
}

TEST(Gfx1250SimulationTest, VMovDppPreservesMaskedHighDestinationLanes) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSrc = 7;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kHighBank = 256;
  constexpr uint32_t kSourceBase = 0x44000000u;
  constexpr uint32_t kOldHighDstBase = 0x55000000u;
  constexpr uint32_t kLowDstBase = 0x66000000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x41); // SRC0=bank 1; DST=bank 1.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kHighBank + kSrc, lane, kSourceBase + lane);
    cu.write_vgpr(vb + kHighBank + kDst, lane, kOldHighDstBase + lane);
    cu.write_vgpr(vb + kDst, lane, kLowDstBase + lane);
  }

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0x1; // Only lanes 0-15 may be updated.

  cdna5::VMovB32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint32_t expected = lane < 16 ? kSourceBase + (lane ^ 1u) : kOldHighDstBase + lane;
    EXPECT_EQ(cu.read_vgpr(vb + kHighBank + kDst, lane), expected) << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kLowDstBase + lane) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VAddDppReadsHighBanksAndPreservesMaskedHighDst) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kSrc0Base = 1000;
  constexpr uint32_t kSrc1Base = 2000;
  constexpr uint32_t kOldHighDstBase = 3000;
  constexpr uint32_t kLowDstBase = 4000;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x49); // SRC0=bank 1; SRC1=bank 2; DST=bank 1.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kSrc0, lane, 10u);
    cu.write_vgpr(vb + kSrc1, lane, 20u);
    cu.write_vgpr(vb + kBankStride + kSrc0, lane, kSrc0Base + lane);
    cu.write_vgpr(vb + 2 * kBankStride + kSrc1, lane, kSrc1Base + lane);
    cu.write_vgpr(vb + kBankStride + kDst, lane, kOldHighDstBase + lane);
    cu.write_vgpr(vb + kDst, lane, kLowDstBase + lane);
  }

  cdna5::Vop2VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc0;
  raw.vsrc1 = kSrc1;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0x1;

  cdna5::VAddNcU32Vop2 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint32_t expected =
        lane < 16 ? kSrc0Base + (lane ^ 1u) + kSrc1Base + lane : kOldHighDstBase + lane;
    EXPECT_EQ(cu.read_vgpr(vb + kBankStride + kDst, lane), expected) << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kLowDstBase + lane) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VCvtF64DppPreservesBothMaskedHighDstDwords) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSrc = 7;
  constexpr uint32_t kDst = 20;
  constexpr uint32_t kHighBank = 256;
  constexpr uint32_t kSourceBase = 100;
  constexpr uint64_t kOldDstBase = 0x5500000066000000ULL;
  constexpr uint32_t kLowDstLoBase = 0x77000000u;
  constexpr uint32_t kLowDstHiBase = 0x88000000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x41); // SRC0=bank 1; DST=bank 1.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint64_t old_dst = kOldDstBase + lane;
    cu.write_vgpr(vb + kHighBank + kSrc, lane, kSourceBase + lane);
    cu.write_vgpr(vb + kHighBank + kDst, lane, static_cast<uint32_t>(old_dst));
    cu.write_vgpr(vb + kHighBank + kDst + 1, lane, static_cast<uint32_t>(old_dst >> 32));
    cu.write_vgpr(vb + kDst, lane, kLowDstLoBase + lane);
    cu.write_vgpr(vb + kDst + 1, lane, kLowDstHiBase + lane);
  }

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SELECT_BASE; // row-select lane 0
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0x1;

  cdna5::VCvtF64I32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint64_t actual =
        static_cast<uint64_t>(cu.read_vgpr(vb + kHighBank + kDst, lane)) |
        (static_cast<uint64_t>(cu.read_vgpr(vb + kHighBank + kDst + 1, lane)) << 32);
    const uint64_t expected =
        lane < 16 ? std::bit_cast<uint64_t>(static_cast<double>(kSourceBase)) : kOldDstBase + lane;
    EXPECT_EQ(actual, expected) << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kLowDstLoBase + lane) << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, lane), kLowDstHiBase + lane) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VAddF32Vop3DppPreservesMaskedHighDestinationLanes) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kOldHighDstBase = 0x55000000u;
  constexpr uint32_t kLowDstBase = 0x66000000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x49); // SRC0=bank 1; SRC1=bank 2; DST=bank 1.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kBankStride + kSrc0, lane,
                  std::bit_cast<uint32_t>(static_cast<float>(lane + 1)));
    cu.write_vgpr(vb + 2 * kBankStride + kSrc1, lane, std::bit_cast<uint32_t>(100.0f));
    cu.write_vgpr(vb + kBankStride + kDst, lane, kOldHighDstBase + lane);
    cu.write_vgpr(vb + kDst, lane, kLowDstBase + lane);
  }

  cdna5::Vop3VopDpp16MachineInst raw{};
  raw.vdst = kDst;
  raw.src0 = amdgpu::SRC_DPP;
  raw.src1 = 256 + kSrc1;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0x1;

  cdna5::VAddF32Vop3 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const uint32_t expected = lane < 16
                                  ? std::bit_cast<uint32_t>(static_cast<float>((lane ^ 1u) + 101))
                                  : kOldHighDstBase + lane;
    EXPECT_EQ(cu.read_vgpr(vb + kBankStride + kDst, lane), expected) << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kLowDstBase + lane) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, ModeBit27DoesNotIndexVmovDppOperands) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kSrc = 7;
  constexpr uint32_t kDst = 8;
  constexpr uint32_t kHighBank = 256;
  constexpr uint32_t kGprIdxOffset = 16;
  constexpr uint32_t kExpectedBase = 0x77000000u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x01); // SRC0=bank 1; DST=bank 0.
  wf->set_mode_raw(wf->mode_raw() | amdgpu::Wavefront::GPR_IDX_EN_BIT);
  wf->set_m0((1u << 12u) | kGprIdxOffset); // Index SRC0 only.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kHighBank + kSrc, lane, kExpectedBase + lane);
    cu.write_vgpr(vb + kHighBank + kGprIdxOffset + kSrc, lane, 0u);
  }

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xB1; // quad_perm:[1,0,3,2]
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  cdna5::VMovB32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kExpectedBase + (lane ^ 1u)) << "lane " << lane;
}

TEST(Gfx1250SimulationTest, PackedTrue16SourcesIgnoreModeBit27) {
  Gfx1250Sim sim;
  // Resident wave: this test drives Operand reads against live VGPR banks.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf();
  ASSERT_NE(wf, nullptr);
  ASSERT_GE(wf->vgpr_alloc().count, kGfx1250Wave32VgprAllocation);

  constexpr uint32_t kLane = 0;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_mode_raw(amdgpu::Wavefront::GPR_IDX_EN_BIT);
  wf->set_m0((1u << 12u) | 16u);
  cu.write_vgpr(vb + 2, kLane, 0xAAAA1111u);
  cu.write_vgpr(vb + 18, kLane, 0xBBBB2222u);

  cdna5::Operand lo(16, cdna5::OperandType::OPR_VGPR, 2, true);
  cdna5::Operand hi(16, cdna5::OperandType::OPR_VGPR, 128 + 2, true);
  lo.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
  hi.set_vgpr_msb_role(amdgpu::VgprMsbRole::Src0);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(lo, kLane), 0x1111u);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(hi, kLane), 0xAAAAu);
}

TEST(Gfx1250SimulationTest, PackedTrue16InstructionIgnoresModeBit27) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 3;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kGprIdxOffset = 16;
  constexpr uint32_t kLowAlias = 0x55556666u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x41); // SRC0=bank 1; DST=bank 1.
  wf->set_mode_raw(wf->mode_raw() | amdgpu::Wavefront::GPR_IDX_EN_BIT);
  wf->set_m0((0x9u << 12u) | kGprIdxOffset); // Index SRC0 and DST.
  cu.write_vgpr(vb + kSrc, 0, kLowAlias);
  cu.write_vgpr(vb + kDst, 0, kLowAlias);
  cu.write_vgpr(vb + kBankStride + kSrc, 0, 0xABCD1234u);
  cu.write_vgpr(vb + kBankStride + kDst, 0, 0xEEEE1111u);
  cu.write_vgpr(vb + kBankStride + kGprIdxOffset + kSrc, 0, kLowAlias);
  cu.write_vgpr(vb + kBankStride + kGprIdxOffset + kDst, 0, 0xEEEE1111u);

  cdna5::Vop1MachineInst raw{};
  raw.src0 = 256 + 128 + kSrc; // High half of the encoded VGPR source.
  raw.vdst = kDst;
  cdna5::VMovB16Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu.read_vgpr(vb + kBankStride + kDst, 0), 0xEEEEABCDu);
  EXPECT_EQ(cu.read_vgpr(vb + kBankStride + kGprIdxOffset + kDst, 0), 0xEEEE1111u);
  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), kLowAlias);
}

TEST(Gfx1250SimulationTest, DsPermuteUsesIndependentHighOperandBanks) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData = 2;
  constexpr uint32_t kDst = 3;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kValueBase = 0x12340000u;
  constexpr uint32_t kLowAlias = 0xDEADBEEFu;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0xC9); // SRC0=bank 1; SRC1=bank 2; DST=bank 3.
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu.write_vgpr(vb + kBankStride + kAddr, lane, lane * 4);
    cu.write_vgpr(vb + 2 * kBankStride + kData, lane, kValueBase + lane);
    cu.write_vgpr(vb + 3 * kBankStride + kDst, lane, 0u);
    cu.write_vgpr(vb + kAddr, lane, kLowAlias);
    cu.write_vgpr(vb + kData, lane, kLowAlias);
    cu.write_vgpr(vb + kDst, lane, kLowAlias);
  }

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData;
  raw.vdst = kDst;
  cdna5::DsPermuteB32Vds inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(cu.read_vgpr(vb + 3 * kBankStride + kDst, lane), kValueBase + lane)
        << "lane " << lane;
    EXPECT_EQ(cu.read_vgpr(vb + kDst, lane), kLowAlias) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, DsStore2addrUsesSrc2HighBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kData1 = 3;
  constexpr uint32_t kSrc0Bank = 1;
  constexpr uint32_t kSrc1Bank = 2;
  constexpr uint32_t kSrc2Bank = 3;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint32_t kExpected0 = 0x12345678u;
  constexpr uint32_t kExpected1 = 0x9ABCDEF0u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(kSrc0Bank | (kSrc1Bank << 2) | (kSrc2Bank << 4));
  cu.write_vgpr(vb + kSrc0Bank * kBankStride + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kSrc1Bank * kBankStride + kData0, 0, kExpected0);
  cu.write_vgpr(vb + kSrc2Bank * kBankStride + kData1, 0, kExpected1);
  cu.write_vgpr(vb + kAddr, 0, 0x100u);
  cu.write_vgpr(vb + kData0, 0, 0xDEADBEEFu);
  cu.write_vgpr(vb + kData1, 0, 0xDEADBEEFu);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.data1 = kData1;
  raw.offset0 = 1;
  raw.offset1 = 2;
  cdna5::DsStore2addrB32Vds inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  auto *state = inst.data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], wf->lds_base() + kAddress + 4);
  EXPECT_EQ(state->ds2_per_lane_addr[0], wf->lds_base() + kAddress + 8);
  uint32_t actual0 = 0;
  uint32_t actual1 = 0;
  std::memcpy(&actual0, state->store_data.data(), sizeof(actual0));
  std::memcpy(&actual1, state->ds2_store_data.data(), sizeof(actual1));
  EXPECT_EQ(actual0, kExpected0);
  EXPECT_EQ(actual1, kExpected1);
}

TEST(Gfx1250SimulationTest, DsStorexchg2addrB64ExchangesBothAddresses) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(256));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kData1 = 4;
  constexpr uint32_t kDst = 6;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint64_t kOld0 = 0x1122334455667788ull;
  constexpr uint64_t kOld1 = 0x99AABBCCDDEEFF00ull;
  constexpr uint64_t kNew0 = 0x0123456789ABCDEFull;
  constexpr uint64_t kNew1 = 0xFEDCBA9876543210ull;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData0, 0, static_cast<uint32_t>(kNew0));
  cu.write_vgpr(vb + kData0 + 1, 0, static_cast<uint32_t>(kNew0 >> 32));
  cu.write_vgpr(vb + kData1, 0, static_cast<uint32_t>(kNew1));
  cu.write_vgpr(vb + kData1 + 1, 0, static_cast<uint32_t>(kNew1 >> 32));
  cu.lds().write64(wf->lds_base() + kAddress + 8, kOld0);
  cu.lds().write64(wf->lds_base() + kAddress + 24, kOld1);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.data1 = kData1;
  raw.vdst = kDst;
  raw.offset0 = 1;
  raw.offset1 = 3;
  auto *inst =
      new cdna5::DsStorexchg2addrRtnB64Vds(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->wf_size, wf->wf_size());
  EXPECT_EQ(state->per_lane_addr[0], wf->lds_base() + kAddress + 8);
  EXPECT_EQ(state->ds2_per_lane_addr[0], wf->lds_base() + kAddress + 24);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read64(wf->lds_base() + kAddress + 8), kNew0);
  EXPECT_EQ(cu.lds().read64(wf->lds_base() + kAddress + 24), kNew1);
  const uint64_t returned0 =
      cu.read_vgpr(vb + kDst, 0) | (static_cast<uint64_t>(cu.read_vgpr(vb + kDst + 1, 0)) << 32);
  const uint64_t returned1 = cu.read_vgpr(vb + kDst + 2, 0) |
                             (static_cast<uint64_t>(cu.read_vgpr(vb + kDst + 3, 0)) << 32);
  EXPECT_EQ(returned0, kOld0);
  EXPECT_EQ(returned1, kOld1);
}

TEST(Gfx1250SimulationTest, NonReturningDsAddU32DoesNotWriteVgpr) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(64));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData = 2;
  constexpr uint32_t kUnusedVdst = 4;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint32_t kInitial = 0x1020'3040u;
  constexpr uint32_t kAddend = 0x0102'0304u;
  constexpr uint32_t kVgprSentinel = 0xA1A2'A3A4u;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData, 0, kAddend);
  cu.write_vgpr(vb + kUnusedVdst, 0, kVgprSentinel);
  cu.lds().write32(wf->lds_base() + kAddress, kInitial);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData;
  raw.vdst = kUnusedVdst;
  auto *inst = new cdna5::DsAddU32Vds(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(state->is_load);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read32(wf->lds_base() + kAddress), kInitial + kAddend);
  EXPECT_EQ(cu.read_vgpr(vb + kUnusedVdst, 0), kVgprSentinel);
}

TEST(Gfx1250SimulationTest, NonReturningDsAddU64DoesNotWriteVgpr) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(64));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData = 2;
  constexpr uint32_t kUnusedVdst = 4;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint64_t kInitial = 0x0102'0304'0506'0708ULL;
  constexpr uint64_t kAddend = 0x1011'1213'1415'1617ULL;
  constexpr uint64_t kVgprSentinel = 0xA1A2'A3A4'B1B2'B3B4ULL;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData, 0, static_cast<uint32_t>(kAddend));
  cu.write_vgpr(vb + kData + 1, 0, static_cast<uint32_t>(kAddend >> 32));
  cu.write_vgpr(vb + kUnusedVdst, 0, static_cast<uint32_t>(kVgprSentinel));
  cu.write_vgpr(vb + kUnusedVdst + 1, 0, static_cast<uint32_t>(kVgprSentinel >> 32));
  cu.lds().write64(wf->lds_base() + kAddress, kInitial);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData;
  raw.vdst = kUnusedVdst;
  auto *inst = new cdna5::DsAddU64Vds(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_FALSE(state->is_load);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read64(wf->lds_base() + kAddress), kInitial + kAddend);
  const uint64_t vgpr_value = cu.read_vgpr(vb + kUnusedVdst, 0) |
                              (static_cast<uint64_t>(cu.read_vgpr(vb + kUnusedVdst + 1, 0)) << 32);
  EXPECT_EQ(vgpr_value, kVgprSentinel);
}

TEST(Gfx1250SimulationTest, DsStorexchg2addrStride64B32UsesScaledOffsetsForActiveLanes) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x5u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(1024));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kData1 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kLane0Address = 0x20;
  constexpr uint32_t kLane2Address = 0x40;
  constexpr uint32_t kOffset0 = 256;
  constexpr uint32_t kOffset1 = 768;
  constexpr uint32_t kLane0New0 = 0x01020304;
  constexpr uint32_t kLane0New1 = 0x11121314;
  constexpr uint32_t kLane2New0 = 0x21222324;
  constexpr uint32_t kLane2New1 = 0x31323334;
  constexpr uint32_t kLane0Old0 = 0x41424344;
  constexpr uint32_t kLane0Old1 = 0x51525354;
  constexpr uint32_t kLane2Old0 = 0x61626364;
  constexpr uint32_t kLane2Old1 = 0x71727374;
  constexpr uint32_t kInactiveSentinel = 0xDEADBEEF;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kLane0Address);
  cu.write_vgpr(vb + kAddr, 2, kLane2Address);
  cu.write_vgpr(vb + kData0, 0, kLane0New0);
  cu.write_vgpr(vb + kData1, 0, kLane0New1);
  cu.write_vgpr(vb + kData0, 2, kLane2New0);
  cu.write_vgpr(vb + kData1, 2, kLane2New1);
  cu.write_vgpr(vb + kDst, 1, kInactiveSentinel);
  cu.write_vgpr(vb + kDst + 1, 1, kInactiveSentinel);

  const uint32_t lane0_addr0 = wf->lds_base() + kLane0Address + kOffset0;
  const uint32_t lane0_addr1 = wf->lds_base() + kLane0Address + kOffset1;
  const uint32_t lane2_addr0 = wf->lds_base() + kLane2Address + kOffset0;
  const uint32_t lane2_addr1 = wf->lds_base() + kLane2Address + kOffset1;
  cu.lds().write32(lane0_addr0, kLane0Old0);
  cu.lds().write32(lane0_addr1, kLane0Old1);
  cu.lds().write32(lane2_addr0, kLane2Old0);
  cu.lds().write32(lane2_addr1, kLane2Old1);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.data1 = kData1;
  raw.vdst = kDst;
  raw.offset0 = 1;
  raw.offset1 = 3;
  auto *inst = new cdna5::DsStorexchg2addrStride64RtnB32Vds(
      reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  auto *state = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], lane0_addr0);
  EXPECT_EQ(state->ds2_per_lane_addr[0], lane0_addr1);
  EXPECT_EQ(state->per_lane_addr[2], lane2_addr0);
  EXPECT_EQ(state->ds2_per_lane_addr[2], lane2_addr1);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.lds().read32(lane0_addr0), kLane0New0);
  EXPECT_EQ(cu.lds().read32(lane0_addr1), kLane0New1);
  EXPECT_EQ(cu.lds().read32(lane2_addr0), kLane2New0);
  EXPECT_EQ(cu.lds().read32(lane2_addr1), kLane2New1);
  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), kLane0Old0);
  EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, 0), kLane0Old1);
  EXPECT_EQ(cu.read_vgpr(vb + kDst, 2), kLane2Old0);
  EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, 2), kLane2Old1);
  EXPECT_EQ(cu.read_vgpr(vb + kDst, 1), kInactiveSentinel);
  EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, 1), kInactiveSentinel);
}

TEST(Gfx1250SimulationTest, DsStorexchg2addrSameAddressAppliesBothExchangesInOrder) {
  Gfx1250Sim sim;
  auto *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(64));
  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kData0 = 2;
  constexpr uint32_t kData1 = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kAddress = 0x20;
  constexpr uint32_t kOld = 0x11223344;
  constexpr uint32_t kNew0 = 0x55667788;
  constexpr uint32_t kNew1 = 0x99AABBCC;
  const uint32_t vb = wf->vgpr_alloc().base;

  cu.write_vgpr(vb + kAddr, 0, kAddress);
  cu.write_vgpr(vb + kData0, 0, kNew0);
  cu.write_vgpr(vb + kData1, 0, kNew1);
  cu.lds().write32(wf->lds_base() + kAddress, kOld);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.data0 = kData0;
  raw.data1 = kData1;
  raw.vdst = kDst;
  auto *inst =
      new cdna5::DsStorexchg2addrRtnB32Vds(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst->execute_impl(*wf);

  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(inst, *wf);

  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), kOld);
  EXPECT_EQ(cu.read_vgpr(vb + kDst + 1, 0), kNew0);
  EXPECT_EQ(cu.lds().read32(wf->lds_base() + kAddress), kNew1);
}

TEST(Gfx1250SimulationTest, DsTransposeLoadUsesHighDestinationBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kAddr = 1;
  constexpr uint32_t kDst = 20;
  constexpr uint32_t kDstBank = 2;
  constexpr uint32_t kBankStride = 256;
  const uint32_t vb = wf->vgpr_alloc().base;
  sim.cu()->write_vgpr(vb + kAddr, 0, 0u);
  wf->set_vgpr_msb_mode(kDstBank << 6);

  cdna5::VdsMachineInst raw{};
  raw.addr = kAddr;
  raw.vdst = kDst;
  cdna5::DsLoadTr4B64Vds inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  auto *state = inst.data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->dst_reg_base, vb + kDstBank * kBankStride + kDst);
}

TEST(Gfx1250SimulationTest, Vop2FmamkUsesSrc2HighBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kAddend = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kSrc2Bank = 2;
  constexpr uint32_t kBankStride = 256;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(kSrc2Bank << 4);
  cu.write_vgpr(vb + kAddend, 0, std::bit_cast<uint32_t>(100.0f));
  cu.write_vgpr(vb + kSrc2Bank * kBankStride + kAddend, 0, std::bit_cast<uint32_t>(4.0f));

  cdna5::Vop2InstLiteralMachineInst raw{};
  raw.src0 = 242; // Inline 1.0f.
  raw.vsrc1 = kAddend;
  raw.vdst = kDst;
  raw.simm32 = std::bit_cast<uint32_t>(2.0f);
  cdna5::VFmamkF32Vop2 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), std::bit_cast<uint32_t>(6.0f));
}

TEST(Gfx1250SimulationTest, VopdFmamkUsesSrc2HighBank) {
  constexpr uint32_t kSrc0InlineOne = 242;
  constexpr uint32_t kAddend = 3;
  constexpr uint32_t kDst = 4;
  constexpr uint32_t kSrc2Bank = 2;
  constexpr uint32_t kBankStride = 256;
  const std::array<uint32_t, 3> words = {
      (0x32u << 26) | (static_cast<uint32_t>(VopdOp::FmamkF32) << 22) |
          (static_cast<uint32_t>(VopdOp::MovB32) << 17) | (kAddend << 9) | kSrc0InlineOne,
      (kDst << 24) | 128u,
      std::bit_cast<uint32_t>(2.0f),
  };

  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  wf->set_vgpr_msb_mode(kSrc2Bank << 4);
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kAddend, 0, std::bit_cast<uint32_t>(100.0f));
  cu->write_vgpr(vb + kSrc2Bank * kBankStride + kAddend, 0, std::bit_cast<uint32_t>(4.0f));

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_dual_fmamk_f32 :: v_dual_mov_b32");
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr(vb + kDst, 0), std::bit_cast<uint32_t>(6.0f));
}

TEST(Gfx1250SimulationTest, VSwapUsesIndependentSourceAndDestinationBanks) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kSrc = 1;
  constexpr uint32_t kDst = 2;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kOldDst = 0x11112222u;
  constexpr uint32_t kHighSrc = 0x33334444u;
  constexpr uint32_t kLowSrc = 0x55556666u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();

  wf->set_vgpr_msb_mode(0x01); // SRC0=bank 1; DST=bank 0.
  cu.write_vgpr(vb + kSrc, 0, kLowSrc);
  cu.write_vgpr(vb + kBankStride + kSrc, 0, kHighSrc);
  cu.write_vgpr(vb + kDst, 0, kOldDst);

  cdna5::Vop1MachineInst raw{};
  raw.src0 = 256 + kSrc;
  raw.vdst = kDst;
  cdna5::VSwapB32Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  EXPECT_EQ(cu.read_vgpr(vb + kDst, 0), kHighSrc);
  EXPECT_EQ(cu.read_vgpr(vb + kBankStride + kSrc, 0), kOldDst);
  EXPECT_EQ(cu.read_vgpr(vb + kSrc, 0), kLowSrc);
}

TEST(Gfx1250SimulationTest, AsyncToLdsUsesDstAndSrc0HighBanks) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kVaddr = 10;
  constexpr uint32_t kLdsAddr = 20;
  constexpr uint32_t kSrc0Bank = 1;
  constexpr uint32_t kDstBank = 2;
  constexpr uint32_t kBankStride = 256;
  constexpr uint64_t kGlobalAddr = 0x1000;
  constexpr uint32_t kLdsOffset = 0x40;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(256));
  wf->set_lds_size(256);
  wf->set_vgpr_msb_mode((kDstBank << 6) | kSrc0Bank);
  cu.write_vgpr(vb + kSrc0Bank * kBankStride + kVaddr, 0, static_cast<uint32_t>(kGlobalAddr));
  cu.write_vgpr(vb + kSrc0Bank * kBankStride + kVaddr + 1, 0,
                static_cast<uint32_t>(kGlobalAddr >> 32));
  cu.write_vgpr(vb + kDstBank * kBankStride + kLdsAddr, 0, kLdsOffset);
  cu.write_vgpr(vb + kLdsAddr, 0, 0x80u);

  cdna5::VglobalMachineInst raw{};
  raw.saddr = 124; // null
  raw.vaddr = kVaddr;
  raw.vdst = kLdsAddr;
  cdna5::GlobalLoadAsyncToLdsB8Vglobal inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));
  inst.execute_impl(*wf);

  auto *state = inst.data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], kGlobalAddr);
  EXPECT_EQ(state->per_lane_lds_addr[0], wf->lds_base() + kLdsOffset);
}

TEST(Gfx1250SimulationTest, AddtidStoresUseSrc1HighBank) {
  Gfx1250Sim sim;
  // Resident wave: these tests inject instructions directly (execute_impl) and read
  // live register state, so they need a live wavefront, not a run-to-halt snapshot.
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);

  constexpr uint32_t kData = 5;
  constexpr uint32_t kSrc1Bank = 2;
  constexpr uint32_t kBankStride = 256;
  constexpr uint32_t kExpected = 0x12345678u;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto &cu = *sim.cu();
  wf->set_vgpr_msb_mode(kSrc1Bank << 2);
  cu.write_vgpr(vb + kData, 0, 0xDEADBEEFu);
  cu.write_vgpr(vb + kSrc1Bank * kBankStride + kData, 0, kExpected);

  cdna5::VdsMachineInst ds_raw{};
  ds_raw.data0 = kData;
  cdna5::DsStoreAddtidB32Vds ds_inst(reinterpret_cast<const cdna5::MachineInst *>(&ds_raw));
  ds_inst.execute_impl(*wf);
  auto *ds_state = ds_inst.data_as<amdgpu::VectorMemState>();
  ASSERT_NE(ds_state, nullptr);
  uint32_t ds_value = 0;
  std::memcpy(&ds_value, ds_state->store_data.data(), sizeof(ds_value));
  EXPECT_EQ(ds_value, kExpected);

  write_wave_sgpr(cu, *wf, 0, 0u);
  write_wave_sgpr(cu, *wf, 1, 0u);
  cdna5::VglobalMachineInst global_raw{};
  global_raw.saddr = 0;
  global_raw.vsrc = kData;
  cdna5::GlobalStoreAddtidB32Vglobal global_inst(
      reinterpret_cast<const cdna5::MachineInst *>(&global_raw));
  global_inst.execute_impl(*wf);
  auto *global_state = global_inst.data_as<amdgpu::VectorMemState>();
  ASSERT_NE(global_state, nullptr);
  uint32_t global_value = 0;
  std::memcpy(&global_value, global_state->store_data.data(), sizeof(global_value));
  EXPECT_EQ(global_value, kExpected);
}

TEST(Gfx1250SimulationTest, DsLoadTransposeIgnoresExecForAddressesAndMasks) {
  for (const uint64_t architectural_exec : {uint64_t{0}, uint64_t{1} << 7}) {
    SCOPED_TRACE(architectural_exec);
    Gfx1250Sim sim;
    amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(architectural_exec);

    constexpr uint32_t kAddr = 5;
    constexpr uint32_t kDst = 10;
    constexpr uint32_t kOffset = 16;
    const uint32_t vgpr_base = wf->vgpr_alloc().base;
    auto &cu = *sim.cu();
    wf->set_lds_base(cu.allocate_lds(512));
    wf->set_lds_size(512);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      cu.write_vgpr(vgpr_base + kAddr, lane, lane * 8u);

    cdna5::VdsMachineInst raw{};
    raw.addr = kAddr;
    raw.vdst = kDst;
    raw.offset0 = kOffset;
    cdna5::DsLoadTr8B64Vds instruction(reinterpret_cast<const cdna5::MachineInst *>(&raw));
    instruction.execute_impl(*wf);

    auto *state = instruction.data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->lane_mask, 0xFFFF'FFFFu);
    EXPECT_EQ(state->exec_mask, 0xFFFF'FFFFu);
    EXPECT_EQ(wf->exec(), architectural_exec);
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
      EXPECT_EQ(state->per_lane_addr[lane], wf->lds_base() + kOffset + lane * 8u);
  }
}

TEST(Gfx1250SimulationTest, DsAddtidLoadAndStoreUseM0ByteBaseAddresses) {
  Gfx1250Sim sim;
  amdgpu::Wavefront *wf = sim.dispatch_scratch_wf(kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x3u);

  auto &cu = *sim.cu();
  wf->set_lds_base(cu.allocate_lds(0x4000));
  constexpr uint32_t kM0ByteBase = 0x1000;
  wf->set_m0(kM0ByteBase);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t kLane0Data = 0x12345678u;
  constexpr uint32_t kLane1Data = 0x9ABCDEF0u;
  cu.write_vgpr(vb + 5, 0, kLane0Data);
  cu.write_vgpr(vb + 5, 1, kLane1Data);
  const uint32_t expected0 = wf->lds_base() + kM0ByteBase + 0x1234;
  const uint32_t expected1 = expected0 + sizeof(uint32_t);

  const uint32_t store_words[] = {0xDAC01234u, 0x00000500u};
  std::unique_ptr<Instruction> store(decode_valid(*decoder, store_words));
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(std::string_view(store->mnemonic()), "ds_store_addtid_b32");
  cu.execute_instruction(store.get(), *wf);
  auto *store_state = store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(store_state, nullptr);
  EXPECT_EQ(store_state->per_lane_addr[0], expected0);
  EXPECT_EQ(store_state->per_lane_addr[1], expected1);
  uint32_t lane0_data = 0;
  uint32_t lane1_data = 0;
  std::memcpy(&lane0_data, &store_state->store_data[0], sizeof(lane0_data));
  std::memcpy(&lane1_data, &store_state->store_data[4], sizeof(lane1_data));
  EXPECT_EQ(lane0_data, kLane0Data);
  EXPECT_EQ(lane1_data, kLane1Data);
  amdgpu::LocalMemPipeline local_pipeline;
  local_pipeline.issue(store.release(), *wf);
  EXPECT_EQ(cu.lds().read32(expected0), kLane0Data);
  EXPECT_EQ(cu.lds().read32(expected1), kLane1Data);

  const uint32_t load_words[] = {0xDAC41234u, 0x08000000u};
  std::unique_ptr<Instruction> load(decode_valid(*decoder, load_words));
  ASSERT_NE(load, nullptr);
  ASSERT_EQ(std::string_view(load->mnemonic()), "ds_load_addtid_b32");
  cu.execute_instruction(load.get(), *wf);
  auto *load_state = load->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(load_state, nullptr);
  EXPECT_EQ(load_state->per_lane_addr[0], expected0);
  EXPECT_EQ(load_state->per_lane_addr[1], expected1);
  const uint32_t load_dst_base = load_state->dst_reg_base;
  local_pipeline.issue(load.release(), *wf);
  EXPECT_EQ(cu.read_vgpr(load_dst_base, 0), kLane0Data);
  EXPECT_EQ(cu.read_vgpr(load_dst_base, 1), kLane1Data);

  wf->set_m0(0);
  std::unique_ptr<Instruction> default_m0_store(decode_valid(*decoder, store_words));
  ASSERT_NE(default_m0_store, nullptr);
  cu.execute_instruction(default_m0_store.get(), *wf);
  auto *default_m0_state = default_m0_store->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(default_m0_state, nullptr);
  EXPECT_EQ(default_m0_state->per_lane_addr[0], wf->lds_base() + 0x1234);
  EXPECT_EQ(default_m0_state->per_lane_addr[1], wf->lds_base() + 0x1238);
}

TEST(Gfx1250SimulationTest, VMovrelsReadsM0RelativeVgpr) {
  const uint32_t code[] = {
      0xBEFD0082u, // s_mov_b32 m0, 2
      0x7E1402FFu,
      0x00000063u, // v_mov_b32_e32 v10, 99
      0x7E028708u, // v_movrels_b32_e32 v1, v8
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code), 16);
  ASSERT_NE(wf, nullptr);

  for (uint32_t lane = 0; lane < wf->wf_size; ++lane)
    EXPECT_EQ(wf->vgpr(1, lane), 99u) << "lane " << lane;
}

TEST(Gfx1250SimulationTest, VopdMulDx9ZeroOverridesNanProducts) {
  constexpr auto dx9_mul = make_vopd3_pair(
      {.op = VopdOp::MulDx9ZeroF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 0, .dst = 4},
      {.op = VopdOp::MulDx9ZeroF32, .src0 = vopd_src0_vgpr(0), .src1 = 2, .src2 = 0, .dst = 5});
  constexpr auto ieee_mul = make_vopd3_pair(
      {.op = VopdOp::MulF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 0, .dst = 6},
      {.op = VopdOp::MulF32, .src0 = vopd_src0_vgpr(0), .src1 = 2, .src2 = 0, .dst = 7});

  std::vector<uint32_t> code;
  append_instruction(code, make_vmov_b32_literal(0, 0x7FC00000u)); // quiet NaN
  append_instruction(code, make_vmov_b32_literal(1, 0x00000000u)); // +0.0f
  append_instruction(code, make_vmov_b32_literal(2, 0x80000000u)); // -0.0f
  append_instruction(code, dx9_mul);
  append_instruction(code, ieee_mul);
  append_instruction(code, S_ENDPGM_GFX12);

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code.data(), code.size(), 16);
  ASSERT_NE(wf, nullptr);

  for (uint32_t lane = 0; lane < wf->wf_size; ++lane) {
    EXPECT_EQ(wf->vgpr(4, lane), 0x00000000u) << "lane " << lane;
    EXPECT_EQ(wf->vgpr(5, lane), 0x00000000u) << "lane " << lane;
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(wf->vgpr(6, lane)))) << "lane " << lane;
    EXPECT_TRUE(std::isnan(std::bit_cast<float>(wf->vgpr(7, lane)))) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VopdFmaUsesSingleRounding) {
  constexpr uint32_t kSrc0 = 0x3F800001u;
  constexpr uint32_t kSrc1 = 0x3F7FFFFFu;
  constexpr uint32_t kSrc2 = 0xBF800000u;
  constexpr auto fma = make_vopd3_pair(
      {.op = VopdOp::FmaF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 2, .dst = 4},
      {.op = VopdOp::FmaF32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 2, .dst = 5});
  const uint32_t expected = std::bit_cast<uint32_t>(std::fma(
      std::bit_cast<float>(kSrc0), std::bit_cast<float>(kSrc1), std::bit_cast<float>(kSrc2)));
  ASSERT_EQ(expected, 0x337FFFFEu);

  std::vector<uint32_t> code;
  append_instruction(code, make_vmov_b32_literal(0, kSrc0));
  append_instruction(code, make_vmov_b32_literal(1, kSrc1));
  append_instruction(code, make_vmov_b32_literal(2, kSrc2));
  append_instruction(code, fma);
  append_instruction(code, S_ENDPGM_GFX12);

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code.data(), code.size(), 16);
  ASSERT_NE(wf, nullptr);

  for (uint32_t lane = 0; lane < wf->wf_size; ++lane) {
    EXPECT_EQ(wf->vgpr(4, lane), expected) << "lane " << lane;
    EXPECT_EQ(wf->vgpr(5, lane), expected) << "lane " << lane;
  }
}

TEST(Gfx1250SimulationTest, VopdFmacUsesDestinationAccumulator) {
  const uint32_t code[] = {
      0x7E0002FFu,
      0x40000000u, // v_mov_b32_e32 v0, 2.0f
      0x7E1202FFu,
      0x3F800000u, // v_mov_b32_e32 v9, 1.0f
      0x7E1402FFu,
      0x40400000u, // v_mov_b32_e32 v10, 3.0f
      0x7E1C02FFu,
      0x40800000u, // v_mov_b32_e32 v14, 4.0f
      0xC900150Eu,    0x0A0800FFu,
      0x3F000000u, // v_dual_add_f32 v10, v14, v10 :: v_dual_fmac_f32 v9, 0.5f, v0
      S_ENDPGM_GFX12,
  };

  Gfx1250Sim sim;
  const auto *wf = dispatch_one_wave(sim, code, std::size(code), 16);
  ASSERT_NE(wf, nullptr);

  for (uint32_t lane = 0; lane < wf->wf_size; ++lane) {
    EXPECT_EQ(wf->vgpr(10, lane), 0x40E00000u) << "lane " << lane;
    EXPECT_EQ(wf->vgpr(9, lane), 0x40000000u) << "lane " << lane;
  }
}

TEST(Gfx1250ExecutionTest, Vopd3CndmaskAppliesB32NegModifiers) {
  constexpr auto cndmask = make_vopd3_pair(
      {.op = VopdOp::CndmaskB32, .src0 = vopd_src0_vgpr(0), .src1 = 1, .src2 = 106, .dst = 2},
      {.op = VopdOp::CndmaskB32, .src0 = vopd_src0_vgpr(3), .src1 = 4, .src2 = 106, .dst = 5},
      /*negx=*/0x1, /*negy=*/0x2);

  Gfx1250Sim sim;
  auto *cu = sim.cu();
  auto *wf = cu->dispatch_wf(0, 0, kGfx1250ScalarSlots, kGfx1250Wave32VgprAllocation);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(0x3u);
  wf->set_vcc(0x1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, cndmask.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_dual_cndmask_b32 :: v_dual_cndmask_b32");

  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + 0, 0, 0x3F800000u);
  cu->write_vgpr(vb + 0, 1, 0x3F000000u);
  cu->write_vgpr(vb + 1, 0, 0x11223344u);
  cu->write_vgpr(vb + 1, 1, 0x55667788u);
  cu->write_vgpr(vb + 3, 0, 0x40000000u);
  cu->write_vgpr(vb + 3, 1, 0x40400000u);
  cu->write_vgpr(vb + 4, 0, 0x3F800000u);
  cu->write_vgpr(vb + 4, 1, 0x3F000000u);

  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr(vb + 2, 0), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr(vb + 2, 1), 0xBF000000u);
  EXPECT_EQ(cu->read_vgpr(vb + 5, 0), 0xBF800000u);
  EXPECT_EQ(cu->read_vgpr(vb + 5, 1), 0x40400000u);
}

} // namespace
