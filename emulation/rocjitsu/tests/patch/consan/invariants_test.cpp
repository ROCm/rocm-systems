// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "target_fixture.h"

#include "rocjitsu/code/patch/consan/prepare.h"

#include <gtest/gtest.h>

#ifndef NDEBUG
namespace rocjitsu::consan {
namespace {

class ConSanProbeInvariantDeathTest : public ::testing::Test {
protected:
  void SetUp() override {
#ifndef CONSAN_FIXTURE_DIR
    GTEST_SKIP() << "gfx1100 assembler/linker fixtures unavailable";
#else
    const std::vector<uint8_t> image =
        test::read_fixture(std::filesystem::path(CONSAN_FIXTURE_DIR) / "race.hsaco");
    const FailureOr<InputProgram> program =
        build_input_program(image, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(program.succeeded()) << error_.message;
    program_ = program.value();
    const FailureOr<LdsRaceProbePlan> plan = plan_lds_race_probes(program_, error_);
    ASSERT_TRUE(plan.succeeded()) << error_.message;
    plan_ = plan.value();
#endif
  }
  InputProgram program_;
  LdsRaceProbePlan plan_;
  InputError error_;
  std::vector<uint8_t> text_;
};

TEST_F(ConSanProbeInvariantDeathTest, PlannerRequiresAQualifiedTarget) {
  program_.target = nullptr;
  EXPECT_DEATH((void)plan_lds_race_probes(program_, error_), "qualified input requires a target");
}

TEST_F(ConSanProbeInvariantDeathTest, PlannerRequiresASupportedWave) {
  program_.wave_size = 0;
  EXPECT_DEATH((void)plan_lds_race_probes(program_, error_),
               "qualified input requires a supported wave mode");
}

TEST_F(ConSanProbeInvariantDeathTest, ArchitectureIsCheckedBeforeOperandHelpers) {
  plan_.arch = ROCJITSU_CODE_ARCH_INVALID;
  EXPECT_DEATH((void)emit_lds_race_entry(text_, plan_, error_),
               "native LDS probes require a supported architecture");
}

TEST_F(ConSanProbeInvariantDeathTest, ExitRequiresASupportedWave) {
  plan_.wave_size = 0;
  EXPECT_DEATH((void)emit_lds_race_exit(text_, plan_, error_),
               "native LDS probes require a supported wave mode");
}

TEST_F(ConSanProbeInvariantDeathTest, AccessScratchIsCheckedBeforeNarrowing) {
  plan_.scalar_base = 256;
  EXPECT_DEATH((void)emit_lds_race_access(text_, plan_, program_.accesses.front(), error_),
               "native LDS probes require bounded SGPR scratch");
}

} // namespace
} // namespace rocjitsu::consan
#endif
