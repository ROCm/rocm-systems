// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validate the probe symbol resolver against a real amdclang++-compiled
// fixture. Loads the standalone gfx90a device ELF built by
// rj_add_probe_object(), confirms the target, resolves rj_nop_probe, and
// decodes its body to confirm it returns through s[30:31].
//
// Gated by CMake on HAS_DEVICE_KERNELS + HAS_PROBE_FIXTURES (needs amdclang++
// and clang-offload-bundler at build time). No GPU is required; this only
// loads and parses the object.

#include "../test_paths.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

using test::kernel_hsaco_path;

TEST(ProbeFixture, NopProbeResolvesAndReturns) {
  Executable exec(kernel_hsaco_path("rj_nop_probe_gfx90a"));
  ASSERT_TRUE(exec.is_valid()) << "failed to load rj_nop_probe_gfx90a.hsaco";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX90A), 0u);
  const AmdGpuCodeObject *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
  ASSERT_NE(co, nullptr);
  EXPECT_EQ(co->target_id(), ROCJITSU_CODE_TARGET_GFX90A);

  std::string err;
  const auto resolved = resolve_probe_symbol(*co, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->name, "rj_nop_probe");
  EXPECT_GT(resolved->body_size, 0u);
  EXPECT_EQ(resolved->body_size % sizeof(uint32_t), 0u);

  // Copy the body into an aligned word buffer before decoding (the image buffer
  // is only byte-aligned). One extra zero word of slack so the decoder can
  // succeed in the event of a malformed input
  const auto *base = reinterpret_cast<const uint8_t *>(co->image_data());
  const size_t num_words = resolved->body_size / sizeof(uint32_t);
  std::vector<uint32_t> body(num_words + 1, 0);
  std::memcpy(body.data(), base + resolved->body_file_offset, resolved->body_size);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  // Decode the whole body; the last instruction must be the return.
  std::string last_mnemonic;
  size_t w = 0;
  while (w < num_words) {
    std::unique_ptr<Instruction> inst(decoder->decode(&body[w]));
    ASSERT_NE(inst, nullptr);
    const int size = inst->size();
    ASSERT_TRUE(size == 4 || size == 8) << "unexpected instruction size in probe body";
    last_mnemonic = std::string(inst->mnemonic());
    w += static_cast<size_t>(size) / sizeof(uint32_t);
  }
  EXPECT_EQ(last_mnemonic, "s_setpc_b64") << "probe body should return via s_setpc_b64 s[30:31]";
}

TEST(ProbeFixture, NopProbeBuildsCallable) {
  Executable exec(kernel_hsaco_path("rj_nop_probe_gfx90a"));
  ASSERT_TRUE(exec.is_valid()) << "failed to load rj_nop_probe_gfx90a.hsaco";
  const AmdGpuCodeObject *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
  ASSERT_NE(co, nullptr);

  std::string err;
  const auto resolved = resolve_probe_symbol(*co, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;

  const auto callable = build_probe_callable(*co, *resolved, ROCJITSU_CODE_ARCH_CDNA2, &err);
  ASSERT_TRUE(callable.has_value()) << err;
  EXPECT_EQ(callable->symbol, "rj_nop_probe");
  EXPECT_EQ(callable->arch, ROCJITSU_CODE_ARCH_CDNA2);
  EXPECT_EQ(callable->cc, ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31);
  EXPECT_EQ(callable->body_words.size(), resolved->body_size / sizeof(uint32_t));
  EXPECT_EQ(callable->output_text_offset, 0u);
}

TEST(ProbeFixture, NopProbeClobberSummaryIsEmpty) {
  Executable exec(kernel_hsaco_path("rj_nop_probe_gfx90a"));
  ASSERT_TRUE(exec.is_valid()) << "failed to load rj_nop_probe_gfx90a.hsaco";
  const AmdGpuCodeObject *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX90A, 0);
  ASSERT_NE(co, nullptr);

  std::string err;
  const auto resolved = resolve_probe_symbol(*co, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  const auto callable = build_probe_callable(*co, *resolved, ROCJITSU_CODE_ARCH_CDNA2, &err);
  ASSERT_TRUE(callable.has_value()) << err;

  const auto summary = build_probe_clobber_summary(*callable, &err);
  ASSERT_TRUE(summary.has_value()) << err;
  EXPECT_TRUE(summary->ordinary_clobbers.none());
  EXPECT_FALSE(summary->touches_exec);
  EXPECT_FALSE(summary->touches_vcc);
  EXPECT_FALSE(summary->touches_scc);
  EXPECT_FALSE(summary->touches_m0);
  EXPECT_FALSE(summary->touches_flat_scratch);
  EXPECT_FALSE(summary->uses_private_segment);
}

} // namespace
} // namespace rocjitsu
