// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_lazy_storage_test.cpp
/// @brief Focused MFMA coverage for VGPR reads and writes across lazy chunks.

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
  // Overwrite the scalar result so that a missing SIMD store cannot match it.
  auto reseed = [&] {
    fixture.seed_words(destination, 32, 777);
    fixture.seed(accumulator, 32, Fmt::F32, Mode::RandomInt, 111);
  };
  auto kernel = [&] {
    amdgpu::exec_f32_mfma_f16_spec<32, 32, 16>(
        *fixture.cu, fixture.vbase + destination, fixture.vbase + source_a,
        fixture.vbase + source_b, fixture.vbase + accumulator, amdgpu::ACC_FROM_VGPR, 0, 0, 0);
  };

  expect_bit_exact("f16 cross-chunk input", Mode::RandomInt, fixture, reseed, kernel, destination,
                   32);
}

TEST(MfmaLazyStorageTest, F16SpecOutputCrossesChunksAndOverlapsInput) {
  SKIP_IF_NO_SIMD();
  constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
  if (!amdgpu::mma_f32_native_width_supported(32, width))
    GTEST_SKIP() << "f16 MFMA shape is not divisible by the native SIMD width";

  constexpr uint32_t destination_and_source_a = MfmaLazyStorage::registers_per_chunk() - 1;
  constexpr uint32_t source_b = 64;
  constexpr uint32_t accumulator = 128;
  constexpr uint32_t CONST_ONE = 0x3f800000u;
  ExactFixture fixture(ROCJITSU_CODE_ARCH_CDNA4, WF_SIZE);
  ASSERT_NE(fixture.wf, nullptr);
  // Overwrite the whole scalar result before restoring the overlapping source.
  auto reseed = [&] {
    fixture.seed_words(destination_and_source_a, 32, 777);
    fixture.seed(destination_and_source_a, 2, Fmt::F16, Mode::RandomInt, 88);
    fixture.seed(source_b, 2, Fmt::F16, Mode::RandomInt, 99);
    fixture.seed(accumulator, 32, Fmt::F32, Mode::RandomInt, 111);
  };
  for (uint32_t const_acc : {amdgpu::ACC_FROM_VGPR, CONST_ONE}) {
    auto kernel = [&] {
      amdgpu::exec_f32_mfma_f16_spec<32, 32, 4, 2>(
          *fixture.cu, fixture.vbase + destination_and_source_a,
          fixture.vbase + destination_and_source_a, fixture.vbase + source_b,
          fixture.vbase + accumulator, const_acc, 0, 0, 0);
    };
    expect_bit_exact("f16 cross-chunk output and source overlap", Mode::RandomInt, fixture, reseed,
                     kernel, destination_and_source_a, 32);
  }
}

} // namespace
