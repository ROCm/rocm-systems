// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/patch/consan/probes.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_live_in.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <string_view>

namespace rocjitsu::consan {
namespace {

struct CallableProbeCase {
  uint32_t target;
  uint32_t wave_size;
};

class ConSanCallableProbe : public ::testing::TestWithParam<CallableProbeCase> {};

TEST_P(ConSanCallableProbe, CompiledBodiesAreSelfContainedAndUseOnlyTheirArguments) {
  const CallableProbeCase mode = GetParam();
  const std::span<const uint8_t> image = lds_probe_image(mode.target, mode.wave_size);
  ASSERT_FALSE(image.empty());
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  for (const auto &[name, arguments] : {std::pair{"rj_consan_entry", uint8_t{5}},
                                        {"rj_consan_access", uint8_t{12}},
                                        {"rj_consan_exit", uint8_t{5}},
                                        {"rj_consan_barrier", uint8_t{8}}}) {
    SCOPED_TRACE(name);
    std::string error;
    const std::optional<ResolvedProbeSymbol> symbol = resolve_probe_symbol(object, name, &error);
    ASSERT_TRUE(symbol) << error;
    const std::optional<ProbeCallable> body =
        build_probe_callable(object, *symbol, target_profile(mode.target)->arch, arguments, &error);
    ASSERT_TRUE(body) << error;
    const std::optional<RegisterSet> live =
        analyze_probe_live_ins(object, *symbol, body->arch, body->abi, mode.wave_size, &error,
                               ProbeEntryState::ZeroVgprBanks);
    ASSERT_TRUE(live) << error;
    EXPECT_TRUE(live->none()) << format_register_set(*live);
    const std::optional<ProbeClobberSummary> clobbers =
        build_probe_clobber_summary(*body, mode.wave_size, &error);
    ASSERT_TRUE(clobbers) << error;
    EXPECT_FALSE(clobbers->uses_private_segment);
    EXPECT_FALSE(clobbers->touches_flat_scratch);
  }
}

TEST_P(ConSanCallableProbe, IntegerBookkeepingHasNoFloatingPointInstructions) {
  const CallableProbeCase mode = GetParam();
  const TargetProfile &target = *target_profile(mode.target);
  const std::span<const uint8_t> image = lds_probe_image(mode.target, mode.wave_size);
  ASSERT_FALSE(image.empty());
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target.public_id);
  ASSERT_TRUE(decoder);
  for (const auto &[name, arguments] : {std::pair{"rj_consan_entry", uint8_t{5}},
                                        {"rj_consan_access", uint8_t{12}},
                                        {"rj_consan_exit", uint8_t{5}},
                                        {"rj_consan_barrier", uint8_t{8}}}) {
    SCOPED_TRACE(name);
    std::string error;
    const std::optional<ResolvedProbeSymbol> symbol = resolve_probe_symbol(object, name, &error);
    ASSERT_TRUE(symbol) << error;
    const std::optional<ProbeCallable> body =
        build_probe_callable(object, *symbol, target.arch, arguments, &error);
    ASSERT_TRUE(body) << error;
    for (size_t cursor = 0; cursor < body->body_words.size();) {
      const FailureOr<std::unique_ptr<Instruction>> instruction =
          decoder->decode_window(std::span(body->body_words).subspan(cursor), cursor * 4);
      ASSERT_TRUE(instruction.succeeded());
      const std::string_view mnemonic = instruction.value()->mnemonic();
      // The call envelope does not save FP exception state. Check the compiled
      // body because integer source expressions can still lower to FP helpers.
      for (const std::string_view format : {"_f4", "_f8", "_f16", "_f32", "_f64", "_bf8", "_bf16"})
        EXPECT_EQ(mnemonic.find(format), std::string_view::npos) << mnemonic;
      cursor += instruction.value()->size() / sizeof(uint32_t);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ConSanProbeModes, ConSanCallableProbe,
                         ::testing::Values(CallableProbeCase{942, 64}, CallableProbeCase{950, 64},
                                           CallableProbeCase{1100, 32}, CallableProbeCase{1100, 64},
                                           CallableProbeCase{1201, 32}, CallableProbeCase{1201, 64},
                                           CallableProbeCase{1250, 32}),
                         [](const ::testing::TestParamInfo<CallableProbeCase> &info) {
                           return "gfx" + std::to_string(info.param.target) + "Wave" +
                                  std::to_string(info.param.wave_size);
                         });

} // namespace
} // namespace rocjitsu::consan
