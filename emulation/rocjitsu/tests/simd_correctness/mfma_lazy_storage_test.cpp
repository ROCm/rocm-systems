// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_lazy_storage_test.cpp
/// @brief Focused MFMA coverage for logical VGPR reads across lazy chunks.

#include "mma_exact_test_support.h"
#include "simdojo/components/register_file.h"
#include "simdojo/components/vector_reg.h"

namespace {

using namespace rocjitsu;
using namespace mma_exact;

constexpr uint32_t WF_SIZE = 64;
using MfmaVgpr = simdojo::VectorReg<WF_SIZE, uint32_t>;
using MfmaLazyStorage = simdojo::detail::SoftwareLazyRegisterStorage<MfmaVgpr, 1024>;

TEST(MfmaLazyStorageTest, F16SpecInputCrossesChunkBoundary) {
  SKIP_IF_NO_SIMD();
  constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
  if (!amdgpu::mma_f32_native_width_supported(32, width))
    GTEST_SKIP() << "f16 MFMA shape is not divisible by the native SIMD width";

  constexpr uint32_t regs_per_chunk = MfmaLazyStorage::registers_per_chunk();
  constexpr uint32_t source_a = regs_per_chunk - 1;
  constexpr uint32_t source_b = 32;
  constexpr uint32_t accumulator = 64;
  constexpr uint32_t destination = 128;
  constexpr uint32_t input_regs = 4;
  static_assert(source_a / regs_per_chunk != (source_a + input_regs - 1) / regs_per_chunk);

  ExactFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
  ASSERT_NE(fixture.wf, nullptr);
  fixture.seed(source_a, input_regs, Fmt::F16, Mode::RandomInt, 88);
  fixture.seed(source_b, input_regs, Fmt::F16, Mode::RandomInt, 99);
  auto reseed_accumulator = [&] { fixture.seed(accumulator, 32, Fmt::F32, Mode::RandomInt, 111); };
  auto kernel = [&] {
    amdgpu::exec_f32_mfma_f16_spec<32, 32, 16>(
        *fixture.cu, fixture.vbase + destination, fixture.vbase + source_a,
        fixture.vbase + source_b, fixture.vbase + accumulator, amdgpu::ACC_FROM_VGPR, 0, 0, 0);
  };

  expect_bit_exact("f16 cross-chunk input", Mode::RandomInt, fixture, reseed_accumulator, kernel,
                   destination, 32);
}

} // namespace
