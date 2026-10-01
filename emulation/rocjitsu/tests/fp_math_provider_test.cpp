// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_provider.h"
#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iterator>
#include <memory>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu::fp_math;

constexpr uint32_t kSentinel = 0xcafef00du;
constexpr unsigned kSgprs = 106;
constexpr unsigned kVgprs = 256;

// RDNA4 VOP1: encoding[31:25]=0x3f, vdst[24:17], op[16:9], src0[8:0].
// The generated gfx1201 encodings use opcode 37 for EXP and 39 for LOG.
constexpr uint32_t encode_vop1(Operation operation, unsigned destination, unsigned source,
                               bool vector_source = true) {
  const unsigned opcode = operation == Operation::Exp ? 37u : 39u;
  return 0x7e000000u | (destination << 17) | (opcode << 9) | ((vector_source ? 256u : 0u) + source);
}

struct Witness {
  uint32_t input;
  uint32_t result;
};

// Independent gfx1201 physical instruction captures; the scalar model is
// checked separately over much wider ranges in transcendental_test.cpp.
constexpr Witness kExpWitnesses[] = {
    {0x33800000u, 0x3f800000u}, {0xb3800000u, 0x3f7fffffu}, {0x42ffffffu, 0x7f7fffa7u},
    {0x43000000u, 0x7f800000u}, {0xc2fc0000u, 0x00800000u}, {0xc2fc0001u, 0x00000000u},
    {0x3f0567ecu, 0x3fb7b03du}, {0xc114ed44u, 0x3acecc1eu}, {0x3e000090u, 0x3f8b95d0u},
    {0x3ec0001au, 0x3fa5fedcu},
};

constexpr Witness kLogWitnesses[] = {
    {0x3f7bffffu, 0xbcba1fa3u}, {0x3f7c0000u, 0xbcba1f74u}, {0x3f7e0000u, 0xbc3963ddu},
    {0x3f7fffffu, 0xb3b8aa3cu}, {0x3f800000u, 0x00000000u}, {0x3f800001u, 0x3438aa3bu},
    {0x3f810000u, 0x3c37f286u}, {0x3f820000u, 0x3cb73cb4u}, {0x3f840000u, 0x3d35d69cu},
    {0x3fffffffu, 0x3f7fffffu},
};

uint32_t model(Operation operation, uint32_t input) {
  return operation == Operation::Exp ? util::detail::exp::evaluate(input)
                                     : util::detail::log::evaluate(input);
}

FpQualificationKey qualified_key(Operation operation) {
  return {ROCJITSU_CODE_TARGET_GFX1201,
          operation,
          InstructionForm::Vop1,
          F32Format::F32,
          F32Format::F32,
          0,
          0,
          0,
          0,
          false};
}

struct ForceScalarGuard {
  bool saved = util::force_scalar();
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(saved); }
};

struct HostFenvGuard {
  fenv_t saved;
  HostFenvGuard() { std::fegetenv(&saved); }
  ~HostFenvGuard() { std::fesetenv(&saved); }
};

struct Fixture {
  amdgpu::GpuMemory gpu_mem{"fp_math_provider_mem"};
  amdgpu::L2Cache l2{"fp_math_provider_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;

  explicit Fixture(unsigned wave_size, rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201,
                   std::shared_ptr<ExecutionPluginGroup> plugins = {}) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA4;
    config.target = target;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = kSgprs;
    config.vgprs_per_wf = kVgprs;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("fp_math_provider_cu", config, &gpu_mem, &l2);
    if (cu && plugins)
      cu->set_plugin_group(std::move(plugins));
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    if (cu)
      wave = cu->dispatch_wf(0, 0, kSgprs, kVgprs, wave_size);
  }

  std::unique_ptr<Instruction> decode(Operation operation, unsigned destination, unsigned source,
                                      bool vector_source = true) const {
    const uint32_t words[4] = {encode_vop1(operation, destination, source, vector_source), 0, 0, 0};
    return std::unique_ptr<Instruction>(decode_valid(*decoder, words));
  }

  void seed(const std::array<uint32_t, 64> &inputs, unsigned source, unsigned destination,
            uint64_t exec) {
    const uint32_t base = wave->vgpr_alloc().base;
    for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
      cu->write_vgpr(base + source, lane, inputs[lane]);
      if (source != destination)
        cu->write_vgpr(base + destination, lane, kSentinel);
    }
    wave->set_exec(exec);
  }

  std::array<uint32_t, 64> snapshot(unsigned destination) const {
    std::array<uint32_t, 64> output{};
    const uint32_t base = wave->vgpr_alloc().base;
    for (unsigned lane = 0; lane < wave->wf_size(); ++lane)
      output[lane] = cu->read_vgpr(base + destination, lane);
    return output;
  }

  std::array<uint32_t, 64> run(Instruction &instruction, unsigned destination) {
    EXPECT_TRUE(cu->execute_instruction(&instruction, *wave).succeeded());
    return snapshot(destination);
  }
};

std::array<uint32_t, 64> inputs_for(Operation operation) {
  std::array<uint32_t, 64> input{};
  for (unsigned lane = 0; lane < input.size(); ++lane) {
    input[lane] = 0x3e000000u + ((lane * 0x10123u) & 0x01ffffffu);
    if (lane & 1)
      input[lane] ^= 0x80000000u;
  }
  const auto &witnesses = operation == Operation::Exp ? kExpWitnesses : kLogWitnesses;
  for (unsigned lane = 0; lane < std::size(kExpWitnesses); ++lane)
    input[lane] = witnesses[lane].input;
  input[std::size(kExpWitnesses)] = 0x7fa12345u;
  input[std::size(kExpWitnesses) + 1] = 0xff800000u;
  input[std::size(kExpWitnesses) + 2] = 0x80000001u;
  return input;
}

void check_decoded(Operation operation, unsigned wave_size, unsigned source, unsigned destination,
                   uint64_t exec) {
  ForceScalarGuard guard;
  const auto input = inputs_for(operation);
  const FpQualificationKey key = qualified_key(operation);
  ASSERT_TRUE(is_qualified(key));
  if (!fp_provider().unary_f32(key)) {
    // Auto does not adopt EXP/LOG until the measured default-entry gate passes.
    // gfx1200 uses the same RDNA4 decoded legacy handler without a qualified entry.
    util::set_force_scalar_for_testing(false);
    std::array<uint32_t, 64> automatic{};
    {
      Fixture fixture(wave_size);
      ASSERT_NE(fixture.wave, nullptr);
      auto instruction = fixture.decode(operation, destination, source);
      ASSERT_NE(instruction, nullptr);
      fixture.wave->set_mode_raw(0);
      fixture.seed(input, source, destination, exec);
      automatic = fixture.run(*instruction, destination);
    }
    {
      Fixture legacy(wave_size, ROCJITSU_CODE_TARGET_GFX1200);
      ASSERT_NE(legacy.wave, nullptr);
      auto instruction = legacy.decode(operation, destination, source);
      ASSERT_NE(instruction, nullptr);
      legacy.wave->set_mode_raw(0);
      legacy.seed(input, source, destination, exec);
      EXPECT_EQ(automatic, legacy.run(*instruction, destination));
    }
    for (unsigned lane = 0; lane < wave_size; ++lane) {
      if (!(exec & (uint64_t{1} << lane))) {
        EXPECT_EQ(automatic[lane], source == destination ? input[lane] : kSentinel)
            << "inactive lane=" << lane;
      }
    }
    return;
  }

  Fixture fixture(wave_size);
  ASSERT_NE(fixture.cu, nullptr);
  ASSERT_NE(fixture.wave, nullptr);
  auto instruction = fixture.decode(operation, destination, source);
  ASSERT_NE(instruction, nullptr);
  fixture.wave->set_mode_raw(0);
  util::set_force_scalar_for_testing(true);
  fixture.seed(input, source, destination, exec);
  const auto scalar = fixture.run(*instruction, destination);
  util::set_force_scalar_for_testing(false);
  fixture.seed(input, source, destination, exec);
  const auto automatic = fixture.run(*instruction, destination);
  EXPECT_EQ(automatic, scalar);

  const auto &witnesses = operation == Operation::Exp ? kExpWitnesses : kLogWitnesses;
  for (unsigned lane = 0; lane < wave_size; ++lane) {
    const uint32_t expected = (exec & (uint64_t{1} << lane)) ? model(operation, input[lane])
                              : source == destination        ? input[lane]
                                                             : kSentinel;
    EXPECT_EQ(automatic[lane], expected) << "lane=" << lane;
    if (lane < std::size(kExpWitnesses) && (exec & (uint64_t{1} << lane))) {
      EXPECT_EQ(automatic[lane], witnesses[lane].result) << "physical witness lane=" << lane;
    }
  }
}

void check_scalar_broadcast(Operation operation, unsigned wave_size, uint64_t exec) {
  ForceScalarGuard guard;
  constexpr uint32_t input = 0x3f0567ecu;
  std::array<uint32_t, 64> dummy{};
  if (!fp_provider().unary_f32(qualified_key(operation))) {
    util::set_force_scalar_for_testing(false);
    std::array<uint32_t, 64> automatic{};
    {
      Fixture fixture(wave_size);
      ASSERT_NE(fixture.wave, nullptr);
      auto instruction = fixture.decode(operation, 2, 0, /*vector_source=*/false);
      ASSERT_NE(instruction, nullptr);
      fixture.wave->set_mode_raw(0);
      fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base, input);
      fixture.seed(dummy, 1, 2, exec);
      automatic = fixture.run(*instruction, 2);
    }
    {
      Fixture legacy(wave_size, ROCJITSU_CODE_TARGET_GFX1200);
      ASSERT_NE(legacy.wave, nullptr);
      auto instruction = legacy.decode(operation, 2, 0, /*vector_source=*/false);
      ASSERT_NE(instruction, nullptr);
      legacy.wave->set_mode_raw(0);
      legacy.cu->write_sgpr(legacy.wave->sgpr_alloc().base, input);
      legacy.seed(dummy, 1, 2, exec);
      EXPECT_EQ(automatic, legacy.run(*instruction, 2));
    }
    return;
  }

  Fixture fixture(wave_size);
  ASSERT_NE(fixture.cu, nullptr);
  ASSERT_NE(fixture.wave, nullptr);
  auto instruction = fixture.decode(operation, 2, 0, /*vector_source=*/false);
  ASSERT_NE(instruction, nullptr);
  fixture.wave->set_mode_raw(0);
  fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base, input);
  util::set_force_scalar_for_testing(true);
  fixture.seed(dummy, 1, 2, exec);
  const auto scalar = fixture.run(*instruction, 2);
  util::set_force_scalar_for_testing(false);
  fixture.seed(dummy, 1, 2, exec);
  const auto automatic = fixture.run(*instruction, 2);
  EXPECT_EQ(automatic, scalar);
  for (unsigned lane = 0; lane < wave_size; ++lane)
    EXPECT_EQ(automatic[lane], (exec & (uint64_t{1} << lane)) ? model(operation, input) : kSentinel)
        << "lane=" << lane;
}

class FpMathAccessRecorder final : public ExecutionPlugin {
public:
  enum class Kind { ReadVgpr, WriteVgpr, ReadSgpr };
  struct Event {
    Kind kind;
    uint32_t reg;
    uint64_t lanes;
    uint8_t bytes;
  };

  FpMathAccessRecorder() : ExecutionPlugin("fp_math_access_recorder") {}
  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    events.push_back({Kind::ReadVgpr, physical_reg - wf->vgpr_alloc().base, lane_mask, byte_mask});
  }
  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask, uint8_t byte_mask) override {
    events.push_back({Kind::WriteVgpr, physical_reg - wf->vgpr_alloc().base, lane_mask, byte_mask});
  }
  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef reg) override {
    if (reg.cls == RegClass::SGPR)
      events.push_back({Kind::ReadSgpr, reg.index, 0, 0});
  }

  std::vector<Event> events;
};

TEST(FpMathProviderTest, Gfx1201DecodedVop1WitnessesMasksAndAliasing) {
  constexpr uint64_t masks[] = {~uint64_t{0}, 0xf0f0000fff00ff0full, 0x8000000100000081ull, 0};
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    for (unsigned wave_size : {32u, 64u}) {
      for (uint64_t exec : masks) {
        check_decoded(operation, wave_size, 0, 2, exec);
        check_decoded(operation, wave_size, 0, 0, exec);
      }
    }
  }
}

TEST(FpMathProviderTest, Gfx1201ScalarRegisterSourceBroadcast) {
  for (Operation operation : {Operation::Exp, Operation::Log})
    check_scalar_broadcast(operation, 64, 0xf0f0f0f00f0f0f0full);
}

TEST(FpMathProviderTest, ForceScalarUsesQualifiedModelBeforeAutoAdoption) {
  ForceScalarGuard guard;
  util::set_force_scalar_for_testing(true);
  Fixture fixture(64);
  ASSERT_NE(fixture.wave, nullptr);
  fixture.wave->set_mode_raw(0);
  constexpr Witness cases[] = {{0xc114ed44u, 0x3acecc1eu}, {0x3f800001u, 0x3438aa3bu}};
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    auto instruction = fixture.decode(operation, 2, 0);
    ASSERT_NE(instruction, nullptr);
    const Witness witness = cases[operation == Operation::Exp ? 0 : 1];
    std::array<uint32_t, 64> input{};
    input.fill(witness.input);
    fixture.seed(input, 0, 2, 1);
    const auto result = fixture.run(*instruction, 2);
    EXPECT_EQ(result[0], witness.result);
    EXPECT_EQ(result[1], kSentinel);
  }
}

TEST(FpMathProviderTest, Gfx1201DecodedVop1SparseAndEmptyExec) {
  constexpr unsigned one_hot_lanes[] = {0, 31, 32, 63};
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    for (unsigned wave_size : {32u, 64u}) {
      for (unsigned lane : one_hot_lanes) {
        if (lane >= wave_size)
          continue;
        const uint64_t exec = uint64_t{1} << lane;
        check_decoded(operation, wave_size, 0, 2, exec);
        check_decoded(operation, wave_size, 0, 0, exec);
        check_scalar_broadcast(operation, wave_size, exec);
      }
      check_decoded(operation, wave_size, 0, 2, 0);
      check_decoded(operation, wave_size, 0, 0, 0);
      check_scalar_broadcast(operation, wave_size, 0);
    }
  }
}

TEST(FpMathProviderTest, Gfx1201SparseAndEmptyExecPreserveRegisterObservations) {
  const auto *kernel = fp_provider().unary_f32(qualified_key(Operation::Exp));
  if (!kernel || kernel->kind == MathKernelKind::Scalar)
    GTEST_SKIP() << "This test requires an adopted packed backend";
  ForceScalarGuard guard;
  util::set_force_scalar_for_testing(false);
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto recorder = std::make_unique<FpMathAccessRecorder>();
  auto *observed = recorder.get();
  ASSERT_TRUE(plugins->add(std::move(recorder)));
  Fixture fixture(64, ROCJITSU_CODE_TARGET_GFX1201, plugins);
  ASSERT_NE(fixture.wave, nullptr);
  fixture.wave->set_mode_raw(0);

  for (Operation operation : {Operation::Exp, Operation::Log}) {
    auto vgpr_instruction = fixture.decode(operation, 2, 0);
    auto sgpr_instruction = fixture.decode(operation, 2, 0, /*vector_source=*/false);
    ASSERT_NE(vgpr_instruction, nullptr);
    ASSERT_NE(sgpr_instruction, nullptr);
    const auto input = inputs_for(operation);
    for (unsigned lane : {0u, 31u, 32u, 63u}) {
      const uint64_t exec = uint64_t{1} << lane;
      fixture.seed(input, 0, 2, exec);
      observed->events.clear();
      EXPECT_TRUE(
          fixture.cu->execute_instruction(vgpr_instruction.get(), *fixture.wave).succeeded());
      ASSERT_EQ(observed->events.size(), 2u);
      EXPECT_EQ(observed->events[0].kind, FpMathAccessRecorder::Kind::ReadVgpr);
      EXPECT_EQ(observed->events[0].reg, 0u);
      EXPECT_EQ(observed->events[0].lanes, exec);
      EXPECT_EQ(observed->events[0].bytes, ExecutionPlugin::kFullByteMask);
      EXPECT_EQ(observed->events[1].kind, FpMathAccessRecorder::Kind::WriteVgpr);
      EXPECT_EQ(observed->events[1].reg, 2u);
      EXPECT_EQ(observed->events[1].lanes, exec);
      EXPECT_EQ(observed->events[1].bytes, ExecutionPlugin::kFullByteMask);
      EXPECT_EQ(fixture.snapshot(2)[lane], model(operation, input[lane]));
    }

    fixture.cu->write_sgpr(fixture.wave->sgpr_alloc().base, 0x3f0567ecu);
    std::array<uint32_t, 64> dummy{};
    fixture.seed(dummy, 1, 2, 0);
    observed->events.clear();
    EXPECT_TRUE(fixture.cu->execute_instruction(sgpr_instruction.get(), *fixture.wave).succeeded());
    ASSERT_EQ(observed->events.size(), 1u);
    EXPECT_EQ(observed->events[0].kind, FpMathAccessRecorder::Kind::ReadSgpr);
    EXPECT_EQ(observed->events[0].reg, 0u);
  }
}

TEST(FpMathProviderTest, UnqualifiedTargetsKeepOriginalHandler) {
  ForceScalarGuard guard;
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX942}) {
    for (Operation operation : {Operation::Exp, Operation::Log}) {
      const FpQualificationKey key{
          target, operation, InstructionForm::Vop1, F32Format::F32, F32Format::F32, 0, 0, 0,
          0,      false};
      EXPECT_FALSE(is_qualified(key));
      EXPECT_EQ(fp_provider().unary_f32(key), nullptr);
    }
  }
  // Also execute gfx1200's same decoded VOP1 form: the absence of an entry
  // cannot turn the host tier into a qualified GPU implementation.
  Fixture fixture(64, ROCJITSU_CODE_TARGET_GFX1200);
  ASSERT_NE(fixture.wave, nullptr);
  std::array<uint32_t, 64> input{};
  input.fill(0x3f0567ecu);
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    auto instruction = fixture.decode(operation, 2, 0);
    ASSERT_NE(instruction, nullptr);
    fixture.wave->set_mode_raw(0);
    util::set_force_scalar_for_testing(true);
    fixture.seed(input, 0, 2, ~uint64_t{0});
    const auto legacy_scalar = fixture.run(*instruction, 2);
    util::set_force_scalar_for_testing(false);
    fixture.seed(input, 0, 2, ~uint64_t{0});
    EXPECT_EQ(fixture.run(*instruction, 2), legacy_scalar);
  }
}

TEST(FpMathProviderTest, ModeTransitionDoesNotRetainQualifiedEntry) {
  if (!fp_provider().unary_f32(qualified_key(Operation::Exp)))
    GTEST_SKIP() << "Auto leaves EXP/LOG on the legacy handler";
  ForceScalarGuard guard;
  util::set_force_scalar_for_testing(false);
  Fixture fixture(64);
  ASSERT_NE(fixture.wave, nullptr);
  constexpr uint32_t input = 0x3f0567ecu;
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    auto instruction = fixture.decode(operation, 2, 0);
    ASSERT_NE(instruction, nullptr);
    std::array<uint32_t, 64> inputs{};
    inputs.fill(input);
    fixture.wave->set_mode_raw(0);
    fixture.seed(inputs, 0, 2, ~uint64_t{0});
    EXPECT_EQ(fixture.run(*instruction, 2)[0], model(operation, input));

    fixture.wave->set_mode_raw(1); // Non-default FP32 rounding is not qualified.
    const FpQualificationKey changed{ROCJITSU_CODE_TARGET_GFX1201,
                                     operation,
                                     InstructionForm::Vop1,
                                     F32Format::F32,
                                     F32Format::F32,
                                     0,
                                     0,
                                     1,
                                     0,
                                     false};
    EXPECT_FALSE(is_qualified(changed));
    fixture.seed(inputs, 0, 2, ~uint64_t{0});
    const auto legacy = fixture.run(*instruction, 2);
    const float value = std::bit_cast<float>(input);
    const uint32_t legacy_expected = std::bit_cast<uint32_t>(
        operation == Operation::Exp ? amdgpu::transcendental::exp_f32(value)
                                    : amdgpu::transcendental::log_f32(value));
    EXPECT_EQ(legacy[0], legacy_expected);

    fixture.wave->set_mode_raw(0);
    fixture.seed(inputs, 0, 2, ~uint64_t{0});
    EXPECT_EQ(fixture.run(*instruction, 2)[0], model(operation, input));
  }
}

TEST(FpMathProviderTest, QualifiedInstructionPreservesHostFenv) {
  if (!fp_provider().unary_f32(qualified_key(Operation::Exp)))
    GTEST_SKIP() << "Auto leaves EXP/LOG on the legacy handler";
  HostFenvGuard fenv_guard;
  ForceScalarGuard scalar_guard;
  util::set_force_scalar_for_testing(false);
  Fixture fixture(64);
  ASSERT_NE(fixture.wave, nullptr);
  auto exp_instruction = fixture.decode(Operation::Exp, 2, 0);
  auto log_instruction = fixture.decode(Operation::Log, 2, 0);
  ASSERT_NE(exp_instruction, nullptr);
  ASSERT_NE(log_instruction, nullptr);
  const auto exp_input = inputs_for(Operation::Exp);
  const auto log_input = inputs_for(Operation::Log);
  EXPECT_EQ(std::fesetround(FE_UPWARD), 0);
  EXPECT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
  EXPECT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
  const int round_before = std::fegetround();
  const int status_before = std::fetestexcept(FE_ALL_EXCEPT);
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    fixture.wave->set_mode_raw(0);
    fixture.seed(operation == Operation::Exp ? exp_input : log_input, 0, 2, ~uint64_t{0});
    fixture.run(operation == Operation::Exp ? *exp_instruction : *log_instruction, 2);
    EXPECT_EQ(std::fegetround(), round_before);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), status_before);
  }
}

TEST(FpMathKernelTest, PackedBackendsMatchScalarModel) {
#if defined(ROCJITSU_FP_MATH_X86_BACKENDS) && ROCJITSU_FP_MATH_X86_BACKENDS
  const HostCpuTier tier = fp_provider().selected_tier();
  if (tier == HostCpuTier::Scalar)
    GTEST_SKIP() << "No selected x86 SIMD backend";

  auto check = [&](F32Bits16 input, bool quiet_snan) {
    F32Bits8 halves[2]{};
    for (unsigned lane = 0; lane < 16; ++lane)
      halves[lane / 8].lane[lane % 8] = input.lane[lane];
    for (Operation operation : {Operation::Exp, Operation::Log}) {
      const auto model_bits = [&](uint32_t bits) {
        return operation == Operation::Exp ? util::detail::exp::evaluate(bits, quiet_snan)
                                           : util::detail::log::evaluate(bits, quiet_snan);
      };
      for (unsigned half = 0; half < 2; ++half) {
        const F32Bits8 actual = operation == Operation::Exp ? exp_v3(halves[half], quiet_snan)
                                                            : log_v3(halves[half], quiet_snan);
        for (unsigned lane = 0; lane < 8; ++lane) {
          const uint32_t expected = model_bits(halves[half].lane[lane]);
          if (actual.lane[lane] != expected) {
            ADD_FAILURE() << "v3 op=" << static_cast<unsigned>(operation) << " quiet=" << quiet_snan
                          << " bits=0x" << std::hex << halves[half].lane[lane] << " actual=0x"
                          << actual.lane[lane] << " expected=0x" << expected;
            return false;
          }
        }
      }
      if (tier != HostCpuTier::X86V4)
        continue;
      for (unsigned half = 0; half < 2; ++half) {
        const F32Bits8 actual = operation == Operation::Exp ? exp_v4x8(halves[half], quiet_snan)
                                                            : log_v4x8(halves[half], quiet_snan);
        for (unsigned lane = 0; lane < 8; ++lane) {
          const uint32_t expected = model_bits(halves[half].lane[lane]);
          if (actual.lane[lane] != expected) {
            ADD_FAILURE() << "v4x8 op=" << static_cast<unsigned>(operation)
                          << " quiet=" << quiet_snan << " bits=0x" << std::hex
                          << halves[half].lane[lane] << " actual=0x" << actual.lane[lane]
                          << " expected=0x" << expected;
            return false;
          }
        }
      }
      const F32Bits16 actual =
          operation == Operation::Exp ? exp_v4x16(input, quiet_snan) : log_v4x16(input, quiet_snan);
      for (unsigned lane = 0; lane < 16; ++lane) {
        const uint32_t expected = model_bits(input.lane[lane]);
        if (actual.lane[lane] != expected) {
          ADD_FAILURE() << "v4x16 op=" << static_cast<unsigned>(operation)
                        << " quiet=" << quiet_snan << " bits=0x" << std::hex << input.lane[lane]
                        << " actual=0x" << actual.lane[lane] << " expected=0x" << expected;
          return false;
        }
      }
    }
    return true;
  };

  constexpr uint32_t edges[] = {
      0u,          1u,          0x007fffffu, 0x00800000u, 0x33800000u, 0xb3800000u, 0x3f7bffffu,
      0x3f7c0000u, 0x3f7fffffu, 0x3f800000u, 0x3f800001u, 0x3f83ffffu, 0x3f840000u, 0x42fc0000u,
      0x43000000u, 0x7f800000u, 0x7f800001u, 0x7fa12345u, 0x7fc12345u, 0x80000000u, 0x80000001u,
      0xc2fc0000u, 0xc2fc0001u, 0xff800000u, 0xff800001u, 0xffa12345u, 0xffc12345u, 0xffffffffu,
      0x3f0567ecu, 0xc114ed44u, 0x3e000090u, 0x3ec0001au,
  };
  for (bool quiet_snan : {false, true}) {
    for (unsigned base = 0; base < std::size(edges); base += 16) {
      F32Bits16 input{};
      for (unsigned lane = 0; lane < 16; ++lane)
        input.lane[lane] = edges[base + lane];
      if (!check(input, quiet_snan))
        return;
    }
    for (uint32_t base = 0x3f7bfff0u; base < 0x3f840010u; base += 16) {
      F32Bits16 input{};
      for (unsigned lane = 0; lane < 16; ++lane)
        input.lane[lane] = base + lane;
      if (!check(input, quiet_snan))
        return;
    }
    uint32_t state = 0x95bdb72au;
    for (unsigned iteration = 0; iteration < 20000; ++iteration) {
      F32Bits16 input{};
      for (auto &lane : input.lane) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        lane = state;
      }
      if (!check(input, quiet_snan))
        return;
    }
  }
#else
  GTEST_SKIP() << "x86 SIMD backends are not built on this host";
#endif
}

// Run this in separate processes with RJ_MATH_BACKEND=auto|scalar|v3|v4.
// It times the active decoded host-simulator dispatch, not GPU execution and
// not an end-to-end workload. The same seeded wave runs after a warmup.
TEST(FpMathProviderBenchmark, DISABLED_ActiveDecodedGfx1201Vop1) {
  constexpr unsigned warmup = 512;
  constexpr unsigned iterations = 16384;
  constexpr uint64_t masks[] = {~uint64_t{0}, 0xf0f0000fff00ff0full, 1, 0x3, 0xf, 0xff};
  for (Operation operation : {Operation::Exp, Operation::Log}) {
    for (uint64_t exec : masks) {
      const auto input = inputs_for(operation);
      std::array<uint32_t, 64> result{};
      double ns_per_dispatch = 0;
      {
        Fixture fixture(64);
        ASSERT_NE(fixture.wave, nullptr);
        auto instruction = fixture.decode(operation, 2, 0);
        ASSERT_NE(instruction, nullptr);
        fixture.wave->set_mode_raw(0);
        fixture.seed(input, 0, 2, exec);
        for (unsigned i = 0; i < warmup; ++i)
          ASSERT_TRUE(
              fixture.cu->execute_instruction(instruction.get(), *fixture.wave).succeeded());
        const auto start = std::chrono::steady_clock::now();
        bool success = true;
        for (unsigned i = 0; i < iterations; ++i)
          success &= fixture.cu->execute_instruction(instruction.get(), *fixture.wave).succeeded();
        const auto stop = std::chrono::steady_clock::now();
        ASSERT_TRUE(success);
        result = fixture.snapshot(2);
        ns_per_dispatch =
            std::chrono::duration<double, std::nano>(stop - start).count() / iterations;
      }
      const auto *kernel = fp_provider().unary_f32(qualified_key(operation));
      const bool force_scalar = util::force_scalar();
      if (kernel || force_scalar) {
        for (unsigned lane = 0; lane < 64; ++lane)
          EXPECT_EQ(result[lane],
                    (exec & (uint64_t{1} << lane)) ? model(operation, input[lane]) : kSentinel);
      } else {
        Fixture legacy(64, ROCJITSU_CODE_TARGET_GFX1200);
        ASSERT_NE(legacy.wave, nullptr);
        auto legacy_instruction = legacy.decode(operation, 2, 0);
        ASSERT_NE(legacy_instruction, nullptr);
        legacy.wave->set_mode_raw(0);
        legacy.seed(input, 0, 2, exec);
        EXPECT_EQ(result, legacy.run(*legacy_instruction, 2));
      }
      const bool sparse_scalar =
          kernel && exec != 0 && std::popcount(exec) <= kernel->scalar_threshold;
      // -1 identifies the original handler, not a qualified scalar kernel.
      const int executed_kind =
          !kernel && !force_scalar
              ? -1
              : static_cast<int>(force_scalar || sparse_scalar ? MathKernelKind::Scalar
                                                               : kernel->kind);
      const int descriptor_kind = kernel ? static_cast<int>(kernel->kind) : -1;
      std::printf("host simulator gfx1201 %s exec=%016llx tier=%u kind=%d descriptor_kind=%d "
                  "force_scalar=%d "
                  "%.2f ns/dispatch\n",
                  operation == Operation::Exp ? "EXP" : "LOG",
                  static_cast<unsigned long long>(exec),
                  static_cast<unsigned>(fp_provider().selected_tier()), executed_kind,
                  descriptor_kind, force_scalar, ns_per_dispatch);
    }
  }
}

} // namespace
