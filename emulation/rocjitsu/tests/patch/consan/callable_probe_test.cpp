// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/patch/consan/probes.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_live_in.h"
#include "rocjitsu/code/patch/probe_symbol.h"

#include <gtest/gtest.h>

namespace rocjitsu::consan {
namespace {

struct CallableProbeCase {
  uint32_t target;
  uint32_t wave;
};

class ConSanCallableProbe : public ::testing::TestWithParam<CallableProbeCase> {};

TEST_P(ConSanCallableProbe, CompiledBodiesAreSelfContainedAndUseOnlyTheirArguments) {
  const auto mode = GetParam();
  const auto image = lds_probe_image(mode.target, mode.wave);
  ASSERT_FALSE(image.empty());
  AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  for (const auto &[name, arguments] : {std::pair{"rj_consan_entry", uint8_t{4}},
                                        {"rj_consan_access", uint8_t{9}},
                                        {"rj_consan_exit", uint8_t{4}}}) {
    SCOPED_TRACE(name);
    std::string error;
    const auto symbol = resolve_probe_symbol(object, name, &error);
    ASSERT_TRUE(symbol) << error;
    const auto body =
        build_probe_callable(object, *symbol, target_profile(mode.target)->arch, arguments, &error);
    ASSERT_TRUE(body) << error;
    const auto live = analyze_probe_live_ins(object, *symbol, body->arch, body->abi, &error,
                                             ProbeEntryState::ZeroVgprBanks);
    ASSERT_TRUE(live) << error;
    EXPECT_TRUE(live->none());
    const auto clobbers = build_probe_clobber_summary(*body, &error);
    ASSERT_TRUE(clobbers) << error;
    EXPECT_FALSE(clobbers->uses_private_segment);
    EXPECT_FALSE(clobbers->touches_flat_scratch);
  }
}

INSTANTIATE_TEST_SUITE_P(ConSanProbeModes, ConSanCallableProbe,
                         ::testing::Values(CallableProbeCase{942, 64}, CallableProbeCase{950, 64},
                                           CallableProbeCase{1100, 32}, CallableProbeCase{1100, 64},
                                           CallableProbeCase{1201, 32}, CallableProbeCase{1201, 64},
                                           CallableProbeCase{1250, 32}),
                         [](const ::testing::TestParamInfo<CallableProbeCase> &info) {
                           return "gfx" + std::to_string(info.param.target) + "Wave" +
                                  std::to_string(info.param.wave);
                         });

} // namespace
} // namespace rocjitsu::consan
