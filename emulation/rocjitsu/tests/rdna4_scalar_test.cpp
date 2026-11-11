// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <gtest/gtest.h>

namespace rocjitsu {
namespace {
class ScalarExecution : public ::testing::Test {
protected:
  void initialize(rj_code_arch_t arch) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 2;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "scalar", config, &memory, &cache));
    decoder = Decoder::create(config.arch);
    ASSERT_NE(decoder, nullptr);
    wave = cu->dispatch_wf(0, 0, 106, 16);
    ASSERT_NE(wave, nullptr);
  }

  void execute(amdgpu::Wavefront &wf, uint32_t word) {
    std::array<uint32_t, 4> words{word};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr) << std::hex << word;
    ASSERT_TRUE(cu->execute_instruction(inst.get(), wf).succeeded());
  }

  amdgpu::GpuMemory memory{"scalar_memory"};
  amdgpu::L2Cache cache{"scalar_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
};

class Rdna4Scalar : public ScalarExecution {
protected:
  void SetUp() override { initialize(ROCJITSU_CODE_ARCH_RDNA4); }

  void check_workgroup_barrier(std::span<const uint32_t> code, std::array<uint32_t, 2> expected) {
    constexpr uint64_t pc = 0x1000;
    for (size_t i = 0; i < code.size(); ++i)
      memory.write32(pc + i * 4, code[i]);
    cu->begin_workgroup(0, 0, 2);
    wave->pc = pc;
    auto *peer = cu->dispatch_wf(0, pc, 106, 16);
    ASSERT_NE(peer, nullptr);
    const std::array waves{wave, peer};
    std::array<uint32_t, 2> results{0xffffffffu, 0xffffffffu};
    for (auto *member : waves)
      cu->write_sgpr(member->sgpr_alloc().base, 0xffffffffu);
    for (unsigned tick = 0; tick < 64 && !cu->is_idle(); ++tick) {
      cu->step();
      // Wave retirement releases and clears its registers. Capture s0 at ENDPGM.
      for (size_t i = 0; i < waves.size(); ++i)
        if (!waves[i]->is_halted() && waves[i]->pc == pc + code.size_bytes() - sizeof(uint32_t))
          results[i] = cu->read_sgpr(waves[i]->sgpr_alloc().base);
    }
    EXPECT_TRUE(wave->is_halted());
    EXPECT_TRUE(peer->is_halted());
    std::ranges::sort(results);
    EXPECT_EQ(results, expected);
    EXPECT_EQ(cu->memory_wait_diagnostic_count(), 0u);
  }
};

class ScalarFloatingPoint : public ScalarExecution,
                            public ::testing::WithParamInterface<rj_code_arch_t> {
protected:
  void SetUp() override { initialize(GetParam()); }

  void set_mode(uint32_t mode) {
    // GFX11.5 has a selectable IEEE arithmetic mode. GFX12 uses IEEE NaN propagation.
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      mode |= amdgpu::Wavefront::IEEE_BIT;
    wave->set_mode_raw(mode);
  }
};

// Expected bit patterns were measured on gfx1201. Exercise the shared GFX11.5 implementation too.
INSTANTIATE_TEST_SUITE_P(Architectures, ScalarFloatingPoint,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4));

TEST_P(ScalarFloatingPoint, UnsignedConversionUsesAll32BitsAndFlagsNegativeInputs) {
  struct Case {
    uint32_t input, result, flags;
  };
  const uint32_t base = wave->sgpr_alloc().base;
  set_mode(0xf0);
  for (const auto &[input, result, flags] :
       {Case{0x4f000000, 0x80000000, 0}, Case{0x4f000001, 0x80000100, 0},
        Case{0x4f32d05e, 0xb2d05e00, 0}, Case{0x4f7fffff, 0xffffff00, 0}, Case{0xabcd0000, 0, 1},
        Case{0x807fffff, 0, 1}, Case{0x80000000, 0, 0}}) {
    SCOPED_TRACE(testing::Message() << std::hex << input);
    cu->write_sgpr(base, input);
    wave->set_trapsts(0);
    ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbe826700)); // s_cvt_u32_f32 s2,s0
    EXPECT_EQ(cu->read_sgpr(base + 2), result);
    EXPECT_EQ(wave->trapsts(), flags);
  }
}

TEST_P(ScalarFloatingPoint, SubtractNegatesNanSubtrahendsAndPreservesMinuendPayload) {
  constexpr uint32_t sub_f32 = 0xa0820100, sub_f16 = 0xa5020100; // s2,s0,s1
  struct Case {
    uint32_t opcode, lhs, rhs, result;
  };
  const uint32_t base = wave->sgpr_alloc().base;
  for (uint32_t mode : {0x00u, 0xf0u, 0xc5u}) {
    set_mode(mode);
    for (const auto &[opcode, lhs, rhs, result] :
         {Case{sub_f32, 0, 0x7fc00000, 0xffc00000}, Case{sub_f32, 0, 0xffc00123, 0x7fc00123},
          Case{sub_f32, 0x3f800000, 0x7f800001, 0xffc00001},
          Case{sub_f32, 0x7fc00000, 0xffc00123, 0x7fc00000},
          Case{sub_f16, 0, 0x807fffff, 0x7fff}}) {
      SCOPED_TRACE(testing::Message()
                   << std::hex << opcode << ' ' << mode << ' ' << lhs << ' ' << rhs);
      cu->write_sgpr(base, lhs);
      cu->write_sgpr(base + 1, rhs);
      ASSERT_NO_FATAL_FAILURE(execute(*wave, opcode));
      EXPECT_EQ(cu->read_sgpr(base + 2), result);
    }
  }
}

TEST_P(ScalarFloatingPoint, DenormalFlagsDistinguishArithmeticFromCompareAndSelect) {
  struct Opcodes {
    uint32_t mode, add, min, compare, fmac, qnan, snan, infinity, one;
  };
  const uint32_t base = wave->sgpr_alloc().base;
  for (const auto &[mode, add, min, compare, fmac, qnan, snan, infinity, one] :
       {Opcodes{0xf0, 0xa0020100, 0xa1020100, 0xbf410100, 0xa3820100, 0x7fc00000, 0x7f800001,
                0x7f800000, 0x3f800000},
        Opcodes{0xc5, 0xa4820100, 0xa5820100, 0xbf520100, 0xa7020100, 0x7e00, 0x7c01, 0x7c00,
                0x3c00}}) {
    set_mode(mode);
    struct Case {
      uint32_t opcode, lhs, rhs, accumulator, flags;
    };
    for (const auto &[opcode, lhs, rhs, accumulator, flags] :
         {Case{add, qnan, 1, 0, 0}, Case{add, 1, snan, 0, 1}, Case{add, infinity, 1, 0, 2},
          Case{fmac, 0, infinity, 1, 1}, Case{min, infinity, 1, 0, 0}, Case{compare, qnan, 1, 0, 0},
          Case{fmac, qnan, snan, 1, 1}, Case{compare, one, 1, 0, 2}}) {
      SCOPED_TRACE(testing::Message()
                   << std::hex << opcode << ' ' << lhs << ' ' << rhs << ' ' << accumulator);
      cu->write_sgpr(base, lhs);
      cu->write_sgpr(base + 1, rhs);
      cu->write_sgpr(base + 2, accumulator);
      wave->set_trapsts(0);
      ASSERT_NO_FATAL_FAILURE(execute(*wave, opcode));
      EXPECT_EQ(wave->trapsts(), flags);
    }
  }
}

TEST_F(Rdna4Scalar, ArithmeticUsesGuestModeWithEmptyExec) {
  wave->set_exec(0);
  const auto base = wave->sgpr_alloc().base;
  cu->write_sgpr(base, 0x3f800000);
  cu->write_sgpr(base + 1, 0x33800000);
  wave->set_mode_raw(1);                               // F32 toward positive infinity.
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xa0020100)); // s_add_f32 s2,s0,s1
  EXPECT_EQ(cu->read_sgpr(base + 2), 0x3f800001u);
  EXPECT_NE(wave->trapsts() & (1u << 5), 0u);

  cu->write_sgpr(base, 1);
  cu->write_sgpr(base + 1, 0);
  wave->set_mode_raw(0);                               // Flush input denormals.
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbf420100)); // s_cmp_eq_f32 s0,s1
  EXPECT_TRUE(wave->read_scc());
  wave->set_mode_raw(0xf0);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbf420100));
  EXPECT_FALSE(wave->read_scc());

  cu->write_sgpr(base, 0x0561);
  cu->write_sgpr(base + 1, 0x4df3);
  cu->write_sgpr(base + 2, 0x46f2);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xa7020100)); // s_fmac_f16 s2,s0,s1
  EXPECT_EQ(cu->read_sgpr(base + 2), 0x46f3u);
  cu->write_sgpr(base, 0x7f800000);
  cu->write_sgpr(base + 1, 0);
  wave->set_trapsts(0);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xa2020100)); // s_mul_f32 s2,s0,s1
  EXPECT_NE(wave->trapsts() & 1u, 0u);
}

TEST_F(Rdna4Scalar, ScalarConversionExceptionMasksMatchGfx1201) {
  const auto base = wave->sgpr_alloc().base;
  wave->set_mode_raw(0xf0);
  cu->write_sgpr(base, 0x3fc00000); // 1.5
  wave->set_trapsts(0);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbe826600)); // s_cvt_i32_f32 s2,s0
  EXPECT_EQ(cu->read_sgpr(base + 2), 1u);
  EXPECT_EQ(wave->trapsts(), 0u);

  cu->write_sgpr(base, 0x7f000000);                    // finite overflow converting to F16
  cu->write_sgpr(base + 1, 0x7f800005);                // signaling NaN
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xa4020100)); // s_cvt_pk_rtz_f16_f32 s2,s0,s1
  EXPECT_EQ(cu->read_sgpr(base + 2), 0x7e007bffu);
  EXPECT_EQ(wave->trapsts(), 0u);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbe826800)); // s_cvt_f16_f32 s2,s0
  EXPECT_EQ(cu->read_sgpr(base + 2), 0x7c00u);
  EXPECT_EQ(wave->trapsts(), 0x28u);
}

TEST_F(Rdna4Scalar, Wave64NarrowMaskUpdatesPreserveHighWordAndTestLowWord) {
  auto *wide = cu->dispatch_wf(1, 0, 106, 16, 64);
  ASSERT_NE(wide, nullptr);
  constexpr uint64_t initial = 0x123400004567ULL;
  const auto base = wide->sgpr_alloc().base;
  for (const auto word : {0xbe822080u, 0xbe8234c1u}) {
    wide->set_exec_raw(initial);
    wide->write_scc(true);
    ASSERT_NO_FATAL_FAILURE(execute(*wide, word));
    EXPECT_EQ(wide->exec_raw(), initial & 0xffffffff00000000ULL);
    EXPECT_FALSE(wide->read_scc());
    // AND_SAVEEXEC with zero returns old EXEC_LO; AND_NOT0_WREXEC with -1 returns result.
    EXPECT_EQ(cu->read_sgpr(base + 2), word == 0xbe822080u ? 0x4567u : 0u);
  }
}

TEST_F(Rdna4Scalar, WorkgroupSignalAndWaitReleaseBothWaves) {
  const uint32_t code[] = {
      0xbe804ec1, // s_barrier_signal -1
      0xbf94ffff, // s_barrier_wait -1
      0xbe800081, // s_mov_b32 s0, 1
      0xbfb00000, // s_endpgm
  };
  ASSERT_NO_FATAL_FAILURE(check_workgroup_barrier(code, {1, 1}));
}

TEST_F(Rdna4Scalar, SplitWaitsReplacePreviousTargetAndMaskReservedBits) {
  auto &counts = wave->wait_counters();
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbfc00000)); // s_wait_loadcnt 0
  counts.increment(amdgpu::WaitCounterType::LOADCNT);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbfc30000)); // s_wait_bvhcnt 0
  EXPECT_EQ(wave->state(), amdgpu::WfState::RUNNING);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbfc00040)); // LOADCNT threshold [5:0] = 0
  EXPECT_EQ(wave->state(), amdgpu::WfState::WAITCNT);
  counts.decrement(amdgpu::WaitCounterType::LOADCNT);
  EXPECT_TRUE(wave->wait_satisfied());

  wave->set_state(amdgpu::WfState::RUNNING);
  counts.increment(amdgpu::WaitCounterType::BVHCNT);
  ASSERT_NO_FATAL_FAILURE(execute(*wave, 0xbfc30008)); // BVHCNT threshold [2:0] = 0
  EXPECT_EQ(wave->state(), amdgpu::WfState::WAITCNT);
  counts.decrement(amdgpu::WaitCounterType::BVHCNT);
  EXPECT_TRUE(wave->wait_satisfied());
}
} // namespace
} // namespace rocjitsu
