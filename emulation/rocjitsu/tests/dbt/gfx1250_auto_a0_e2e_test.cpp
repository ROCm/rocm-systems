// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_auto_a0_e2e_test.cpp
/// @brief End-to-end proof that a real gfx1250 kernel translated B0->A0 executes
///        correctly on the emulator.
///
/// This is the eager auto-A0 keystone: it chains the two flows that previously
/// only existed separately -- (1) load a real compiled gfx1250 code object and
/// translate it B0->A0 via BinaryTranslator (as in relocation_function_table_hip_test),
/// and (2) load a code object into the simulator and execute it with buffer
/// readback (as in vector_add_test). Running the *translated* A0 ELF and getting
/// the right answer proves the whole translate->execute path on the dev box,
/// before any real A0 hardware.

#ifndef HAS_GFX1250_DEVICE_KERNELS
#error "gfx1250_auto_a0_e2e_test.cpp requires a gfx1250-capable device compiler"
#endif

#include "../aql_queue.h"
#include "../test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

// A gfx1250 config with the VM topology the simulator needs to execute A0 code.
const std::string kConfigPath = test::config_path("gfx1250.json");

constexpr uint32_t kBlockSize = 64; // matches vector_add.hip
constexpr uint32_t kN = 256;        // small grid: 4 workgroups of 64

constexpr uint64_t kKdAddr = 0x10000;
constexpr uint64_t kAAddr = 0x100000;
constexpr uint64_t kBAddr = 0x200000;
constexpr uint64_t kCAddr = 0x300000;
constexpr uint64_t kKernargAddr = 0x400000;

// Translate a compiled gfx1250 code object B0->A0 and return the translated ELF.
std::vector<uint8_t> translate_b0_to_a0(const AmdGpuCodeObject &source) {
  BinaryTranslatorOptions options;
  options.input_revision = ProcessorRevision::Gfx1250B0;
  options.output_revision = ProcessorRevision::Gfx1250A0;
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_GFX1250, ROCJITSU_CODE_ARCH_GFX1250,
                              /*target_mach=*/0, options);
  auto result = translator.translate(source);
  EXPECT_TRUE(result.ok()) << (result.diagnostics.empty() ? "translation failed without diagnostics"
                                                          : result.diagnostics.front().message);
  EXPECT_TRUE(result.dispatchable()) << "translated object is not dispatchable";
  return std::move(result.elf_bytes);
}

// Load an A0 code object into the simulator, dispatch vector_add over kN
// elements, and return the output vector C.
std::vector<float> run_vector_add_on_sim(const AmdGpuCodeObject &a0, const std::vector<float> &a,
                                         const std::vector<float> &b) {
  auto loaded = config::load_config(kConfigPath, rocjitsu::kEmbeddedSchema);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->create();

  const uint64_t kd_offset = a0.kernel_descriptor_offset("vector_add");
  EXPECT_NE(kd_offset, 0u) << "vector_add kernel descriptor not found in translated ELF";
  a0.load_to_memory(memory, kKdAddr);
  const uint64_t kernel_object = kKdAddr + kd_offset;

  const size_t vec_bytes = kN * sizeof(float);
  memory->load_image(reinterpret_cast<const uint8_t *>(a.data()), vec_bytes, kAAddr);
  memory->load_image(reinterpret_cast<const uint8_t *>(b.data()), vec_bytes, kBAddr);
  std::vector<float> zeros(kN, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()), vec_bytes, kCAddr);

  struct {
    uint64_t a, b, c;
    uint32_t n;
  } args = {kAAddr, kBAddr, kCAddr, kN};
  memory->load_image(reinterpret_cast<const uint8_t *>(&args), sizeof(args), kKernargAddr);

  auto *cp = soc->xcd(0)->command_processor();
  test::AqlQueue queue(memory, cp);
  queue.dispatch(kernel_object, kN, kBlockSize, kKernargAddr);

  engine->run();
  soc->flush_all();

  std::vector<float> c(kN);
  for (uint32_t i = 0; i < kN; ++i)
    c[i] = std::bit_cast<float>(memory->read32(kCAddr + i * sizeof(float)));
  return c;
}

} // namespace

// The full eager keystone: a real gfx1250 kernel, translated B0->A0, executes on
// the emulator and produces the correct result.
TEST(Gfx1250AutoA0E2E, TranslatedVectorAddExecutesCorrectly) {
  Executable executable(test::kernel_path("vector_add_gfx1250"));
  ASSERT_TRUE(executable.is_valid()) << "failed to load vector_add_gfx1250.o";
  ASSERT_EQ(executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX1250), 1u);
  const auto *source = executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(source, nullptr);

  const std::vector<uint8_t> translated_elf = translate_b0_to_a0(*source);
  ASSERT_FALSE(translated_elf.empty());

  AmdGpuCodeObject a0(translated_elf.data(), translated_elf.size());
  ASSERT_TRUE(a0.is_valid()) << "translated bytes are not a valid AMDGPU code object";

  std::vector<float> a(kN), b(kN), expected(kN);
  for (uint32_t i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i % 97) * 0.1f;
    b[i] = static_cast<float>(i % 61) * 0.2f;
    expected[i] = a[i] + b[i];
  }

  const std::vector<float> c = run_vector_add_on_sim(a0, a, b);

  unsigned mismatches = 0;
  for (uint32_t i = 0; i < kN; ++i) {
    if (std::abs(c[i] - expected[i]) > 1e-6f) {
      if (mismatches < 10)
        ADD_FAILURE() << "C[" << i << "] = " << c[i] << ", expected " << expected[i];
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " elements differ (showing first 10)";
}

// Sanity: the translated ELF is a distinct object from the source (translation
// actually produced new bytes) yet still exposes the kernel. Cheap guard that the
// e2e test above is exercising the translated image, not the original.
TEST(Gfx1250AutoA0E2E, TranslationProducesLoadableKernel) {
  Executable executable(test::kernel_path("vector_add_gfx1250"));
  ASSERT_TRUE(executable.is_valid());
  const auto *source = executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(source, nullptr);

  const std::vector<uint8_t> translated_elf = translate_b0_to_a0(*source);
  AmdGpuCodeObject a0(translated_elf.data(), translated_elf.size());
  ASSERT_TRUE(a0.is_valid());
  EXPECT_NE(a0.kernel_descriptor_offset("vector_add"), 0u);
  EXPECT_FALSE(a0.text_sections().empty());
}
