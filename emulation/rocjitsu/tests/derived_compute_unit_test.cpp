// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file derived_compute_unit_test.cpp
/// @brief create_compute_unit(): a model's class derived from ExecComputeUnit,
/// completed with the ISA of the configured architecture.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/compute_unit_isa.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/diagnostic.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint64_t kProgram = 0x1000;
constexpr uint32_t kMovS5 = 0xBE850087u;  // s_mov_b32 s5, 7
constexpr uint32_t kEndpgm = 0xBFB00000u; // s_endpgm

/// A clocked CU model: state of its own, built ahead of the ISA parts.
class ClockedComputeUnit : public ExecComputeUnit<simdojo::ExecMode::CLOCKED> {
public:
  template <typename... CoreArgs>
  ClockedComputeUnit(uint32_t tag, CoreArgs &&...core)
      : ExecComputeUnit<simdojo::ExecMode::CLOCKED>(std::forward<CoreArgs>(core)...), tag_(tag),
        slots_at_construction_(num_wf_slots()) {}

  uint32_t tag() const { return tag_; }
  uint32_t slots_at_construction() const { return slots_at_construction_; }

  /// A clocked pipeline learns of new work here, so count the wakeups that
  /// reach this layer rather than stopping at the shell or the ISA parts.
  void schedule_work() override {
    ++schedule_work_calls_;
    ExecComputeUnit<simdojo::ExecMode::CLOCKED>::schedule_work();
  }
  uint32_t schedule_work_calls() const { return schedule_work_calls_; }

private:
  const uint32_t tag_;
  const uint32_t slots_at_construction_;
  uint32_t schedule_work_calls_ = 0;
};

ComputeUnitCore::Config config_for(rj_code_arch_t arch, uint32_t slots = 2) {
  ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = slots;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  return config;
}

class DerivedComputeUnitTest : public ::testing::Test {
protected:
  void SetUp() override { l2.set_backing_memory(&memory); }

  GpuMemory memory{"derived_memory"};
  L2Cache l2{"derived_l2"};
};

TEST_F(DerivedComputeUnitTest, CompletesTheModelWithTheConfiguredIsa) {
  std::unique_ptr<ClockedComputeUnit> cu = create_compute_unit<ClockedComputeUnit>(
      "clocked_cu", config_for(ROCJITSU_CODE_ARCH_RDNA4), &memory, &l2, {}, 42u);
  ASSERT_NE(cu, nullptr);
  EXPECT_EQ(cu->tag(), 42u);
  // The core is complete before ClockedComputeUnit's own state is built.
  EXPECT_EQ(cu->slots_at_construction(), 2u);
  EXPECT_EQ(cu->name(), "clocked_cu");
  using Rdna4Cu = IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, rdna4::Isa, ClockedComputeUnit>;
  EXPECT_NE(dynamic_cast<Rdna4Cu *>(cu.get()), nullptr);
  EXPECT_EQ(cu->wf_size(), rdna4::Isa::WF_SIZE);
}

TEST_F(DerivedComputeUnitTest, ExecutesThroughTheModelClass) {
  std::unique_ptr<ClockedComputeUnit> cu = create_compute_unit<ClockedComputeUnit>(
      "clocked_cu", config_for(ROCJITSU_CODE_ARCH_RDNA4), &memory, &l2, {}, 0u);
  memory.write32(kProgram, kMovS5);
  for (uint32_t i = 1; i <= 4; ++i)
    memory.write32(kProgram + i * 4, kEndpgm);
  EXPECT_EQ(cu->schedule_work_calls(), 0u);
  Wavefront *wave = cu->dispatch_wf(0, kProgram, 104, 256);
  ASSERT_NE(wave, nullptr);
  // The test drives each cycle itself below, so only this count shows that
  // dispatch woke the model's override, as a clocked pipeline needs it to.
  EXPECT_EQ(cu->schedule_work_calls(), 1u);
  cu->write_vgpr(wave->vgpr_alloc().base, 3, 0x1234u);
  EXPECT_EQ(cu->read_vgpr(wave->vgpr_alloc().base, 3), 0x1234u);

  EXPECT_EQ(cu->run_issue_cycle(0b1), 0b1u);
  EXPECT_EQ(cu->read_sgpr(wave->sgpr_alloc().base + 5), 7u);
  EXPECT_EQ(cu->run_issue_cycle(0b1), 0b1u);
  EXPECT_TRUE(wave->is_halted());
}

TEST_F(DerivedComputeUnitTest, SelectsEachArchitecturesIsa) {
  std::unique_ptr<ClockedComputeUnit> cu = create_compute_unit<ClockedComputeUnit>(
      "clocked_cu", config_for(ROCJITSU_CODE_ARCH_CDNA3), &memory, &l2, {}, 0u);
  using Cdna3Cu = IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, cdna3::Isa, ClockedComputeUnit>;
  EXPECT_NE(dynamic_cast<Cdna3Cu *>(cu.get()), nullptr);
  EXPECT_EQ(cu->wf_size(), cdna3::Isa::WF_SIZE);
}

TEST_F(DerivedComputeUnitTest, ValidatesTheConfigurationLikeCreate) {
  const auto too_many_slots = config_for(ROCJITSU_CODE_ARCH_RDNA4, rdna4::Isa::MAX_WF_SLOTS + 1);
  util::StringDiagnostic derived;
  EXPECT_EQ(create_compute_unit<ClockedComputeUnit>("clocked_cu", too_many_slots, &memory, &l2,
                                                    derived.emitter(), 0u),
            nullptr);
  EXPECT_THAT(derived.message(), testing::HasSubstr("num_wf_slots exceeds the ISA maximum"));
  util::StringDiagnostic created;
  EXPECT_EQ(ComputeUnitCore::create("cu", too_many_slots, &memory, &l2,
                                    simdojo::ExecMode::FUNCTIONAL, created.emitter()),
            nullptr);
  EXPECT_EQ(derived.message(), created.message());
}

} // namespace
