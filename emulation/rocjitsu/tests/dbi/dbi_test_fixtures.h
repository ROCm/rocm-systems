// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_test_fixtures.h
/// @brief Target parameterization and the gtest scaffolding shared by the two
///        HSA DBI smoke tests: hsa_dbi_nop_asm_test.cpp (inline-nop path) and
///        hsa_dbi_nop_probe_test.cpp (probe-call path).
///
/// The two tests differ in how they patch the kernel and in the sabotage case
/// that proves the patch executed on the GPU. Everything else is the same test
/// twice: load the patched ELF into an HSA executable and validate it, then
/// dispatch patched-vs-original and require identical output. Those two bodies
/// live here so they cannot drift apart.
///
/// hsa_dispatch_util.h is the assertion-free HSA plumbing this builds on; it
/// stays free of gtest so its dispatch helper can keep reporting failure by
/// returning an empty buffer.

#pragma once

#include "hsa_dispatch_util.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::dbi_test {

/// @brief Everything that differs between one target's run of these tests and
///        another's. Each test file defines one constant per supported target.
struct DbiTargetParams {
  rj_code_arch_t arch;
  rj_code_target_id_t target;
  /// kernel_path() stem of the kernel to instrument.
  const char *kernel_fixture;
  /// kernel_hsaco_path() stem of the compiled rj_nop_probe device ELF. Null in
  /// the inline-nop test, which calls no probe.
  const char *probe_fixture;
  /// Substring identifying the agent in the HSA ISA name. Matching on the ISA
  /// name is what lets one binary bind either to a real GPU or to whichever
  /// agent the rocjitsu CLI supplies.
  const char *isa_substring;
};

/// @brief The vector_add inputs both smoke tests dispatch, with the CPU-side
///        expected result. The seed is fixed so the golden is reproducible and
///        the sabotage cases can compare against the same values.
class GoldenVectorAddInputs {
public:
  static constexpr uint32_t kSize = 1024;

  GoldenVectorAddInputs() : a_(kSize), b_(kSize), golden_(kSize) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (uint32_t i = 0; i < kSize; ++i) {
      a_[i] = dist(rng);
      b_[i] = dist(rng);
      golden_[i] = a_[i] + b_[i];
    }
  }

  const std::vector<float> &a() const { return a_; }
  const std::vector<float> &b() const { return b_; }
  const std::vector<float> &golden() const { return golden_; }

private:
  std::vector<float> a_;
  std::vector<float> b_;
  std::vector<float> golden_;
};

/// @brief Count elements of @p out within tolerance of @p golden.
///
/// The sabotage cases use this to assert the *absence* of a match: a terminated
/// kernel must not reproduce the golden buffer.
inline uint32_t count_matching_golden(const std::vector<float> &out,
                                      const std::vector<float> &golden) {
  uint32_t matches = 0;
  for (uint32_t i = 0; i < out.size(); ++i) {
    if (std::abs(out[i] - golden[i]) < 1e-5f)
      ++matches;
  }
  return matches;
}

/// @brief True if any element of @p out is non-zero.
///
/// dispatch_vector_add zeroes the C buffer before every dispatch, so a non-zero
/// result proves the kernel actually ran and wrote into it rather than merely
/// that the surrounding HSA plumbing succeeded.
inline bool kernel_wrote_output(const std::vector<float> &out) {
  for (float v : out) {
    if (v != 0.0f)
      return true;
  }
  return false;
}

// The expect_* helpers below contain ASSERT_*, which returns only from the
// function it appears in. Call each as the last statement of its caller so a
// failed assertion ends the test rather than falling through -- the
// DbiHardwareBase run_* methods do exactly that.

/// @brief Load @p elf_bytes into an HSA executable on @p gpu, validate it, and
///        confirm the vector_add kernel symbol survived patching. No dispatch.
///
/// Needs a real agent because hsa_executable_load_agent_code_object requires an
/// agent whose ISA matches the code object.
inline void expect_elf_loads_and_validates(std::span<const uint8_t> elf_bytes, hsa_agent_t gpu) {
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(hsa_code_object_reader_create_from_memory(elf_bytes.data(), elf_bytes.size(), &reader),
            HSA_STATUS_SUCCESS)
      << "Patched ELF rejected by hsa_code_object_reader_create_from_memory";

  hsa_executable_t executable{};
  ASSERT_EQ(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_executable_load_agent_code_object(executable, gpu, reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS)
      << "hsa_executable_load_agent_code_object rejected the patched ELF";
  ASSERT_EQ(hsa_executable_freeze(executable, nullptr), HSA_STATUS_SUCCESS);

  uint32_t validate_result = 0;
  ASSERT_EQ(hsa_executable_validate(executable, &validate_result), HSA_STATUS_SUCCESS);
  EXPECT_EQ(validate_result, 0u) << "hsa_executable_validate reported error: " << validate_result;

  // Sanity: the kernel symbol is still findable post-patch.
  hsa_executable_symbol_t symbol{};
  EXPECT_EQ(hsa_executable_get_symbol_by_name(executable, "vector_add.kd", &gpu, &symbol),
            HSA_STATUS_SUCCESS)
      << "kernel symbol vector_add.kd missing after patching";

  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(reader);
}

/// @brief Dispatch @p original_elf and @p patched_elf with identical inputs and
///        require bit-identical output.
///
/// Both DBI paths are semantically no-ops, so any difference means the patch
/// corrupted execution. @p patch_description names the mechanism in the failure
/// message (e.g. "the inline-nop placeholder").
///
/// This is a regression check; it does NOT prove the instrumentation executed.
/// Each test's sabotage case is what proves that.
inline void expect_patched_dispatch_matches_original(std::span<const uint8_t> original_elf,
                                                     std::span<const uint8_t> patched_elf,
                                                     hsa_agent_t gpu, hsa_agent_t cpu,
                                                     std::string_view patch_description) {
  const GoldenVectorAddInputs inputs;
  constexpr uint32_t kN = GoldenVectorAddInputs::kSize;

  // Sanity: the original (unpatched) dispatch matches the CPU golden. If this
  // fails the fixture is bad, not the instrumentation.
  auto orig_out = dispatch_vector_add(original_elf, gpu, cpu, inputs.a(), inputs.b(), kN);
  ASSERT_EQ(orig_out.size(), kN) << "original dispatch failed (empty result)";
  ASSERT_TRUE(kernel_wrote_output(orig_out))
      << "original dispatch left output buffer all zeros (kernel didn't run?)";
  ASSERT_EQ(count_matching_golden(orig_out, inputs.golden()), kN)
      << "original vector_add dispatch does not match the CPU golden (fixture problem)";

  // The real check: the patched dispatch must produce the same buffer.
  auto patched_out = dispatch_vector_add(patched_elf, gpu, cpu, inputs.a(), inputs.b(), kN);
  ASSERT_EQ(patched_out.size(), kN) << "patched dispatch failed (empty result)";
  ASSERT_TRUE(kernel_wrote_output(patched_out))
      << "patched dispatch left output buffer all zeros (kernel didn't run?)";
  EXPECT_EQ(patched_out, orig_out) << "patched kernel output differs from original -- "
                                   << patch_description << " should be semantically transparent";
}

/// @brief Per-target HSA setup and the two target-agnostic hardware bodies,
///        shared by both smoke tests' hardware suites.
///
/// hsa_init / hsa_shut_down run once per suite (HSA tolerates per-test
/// init/shutdown but it isn't free) and the agent is enumerated once and cached.
///
/// Params is a template parameter rather than a constructor argument so that
/// each target's suite gets its own s_gpu_ / s_init_ok_: SetUpTestSuite is
/// static and cannot reach an instance member.
///
/// @tparam Fixture the per-test fixture, which must expose original_elf_bytes_,
///         patched_elf_bytes_, a DbiTargetParams constructor, and a
///         kPatchDescription naming its instrumentation mechanism.
template <typename Fixture, const DbiTargetParams &Params> class DbiHardwareBase : public Fixture {
protected:
  DbiHardwareBase() : Fixture(Params) {}

  static void SetUpTestSuite() {
    s_init_ok_ = (hsa_init() == HSA_STATUS_SUCCESS);
    if (s_init_ok_)
      s_gpu_ = find_gpu_agent(Params.isa_substring);
  }
  static void TearDownTestSuite() {
    if (s_init_ok_)
      hsa_shut_down();
    s_init_ok_ = false;
    s_gpu_ = {};
  }

  // Skip before Fixture::SetUp() rather than after. Fixture::SetUp() asserts its
  // way through the ELF load and Instrumentor::patch, and ASSERT_* returns only
  // from the function it appears in -- so running it first would need a
  // HasFatalFailure() guard here, or a genuine patch failure would be reported
  // as a skip.
  void SetUp() override {
    if (!s_init_ok_)
      GTEST_SKIP() << "hsa_init failed (no HSA runtime at runtime)";
    if (s_gpu_.handle == 0)
      GTEST_SKIP() << "No " << Params.isa_substring << " agent present";
    Fixture::SetUp();
  }

  void run_patched_elf_loads_and_validates() {
    expect_elf_loads_and_validates(this->patched_elf_bytes_, s_gpu_);
  }

  void run_patched_kernel_dispatch_matches_original() {
    hsa_agent_t cpu = find_cpu_agent();
    ASSERT_NE(cpu.handle, 0u) << "No CPU agent found";
    expect_patched_dispatch_matches_original(this->original_elf_bytes_, this->patched_elf_bytes_,
                                             s_gpu_, cpu, Fixture::kPatchDescription);
  }

  static inline bool s_init_ok_ = false;
  static inline hsa_agent_t s_gpu_{};
};

} // namespace rocjitsu::dbi_test
