// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <span>
#include <vector>

#ifdef HAS_GFX1251_DEVICE_KERNELS

namespace {

using namespace rocjitsu;

constexpr uint64_t kCodeObjectBase = 0x10000;
constexpr uint64_t kInputAAddress = 0x100000;
constexpr uint64_t kInputBAddress = 0x200000;
constexpr uint64_t kOutputAddress = 0x300000;
constexpr uint64_t kKernargAddress = 0x400000;
constexpr uint64_t kCompletionSignal = 0x500000;
constexpr uint32_t kSignalValueOffset = 8;
constexpr uint32_t kWorkgroupSize = 64;

class Gfx1251CompiledKernelTest : public testing::Test {
protected:
  void SetUp() override {
    config::LoadedConfig loaded =
        config::load_config(test::config_path("gfx1251_synthetic.json"), kEmbeddedSchema);
    ASSERT_EQ(loaded.target, ROCJITSU_CODE_TARGET_GFX1251);
    soc_ = loaded.soc();
    memory_ = loaded.memory();
    ASSERT_NE(soc_, nullptr);
    ASSERT_NE(memory_, nullptr);
    engine_ = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine_->topology().set_root(loaded.take_root());
    loaded.wire_links(engine_->topology());
    engine_->create();
  }

  void load_compiled_kernel(const char *fixture_name, const char *kernel_name,
                            uint64_t &kernel_object) {
    Executable executable(test::kernel_path(fixture_name));
    ASSERT_TRUE(executable.is_valid()) << "Failed to load " << fixture_name << ".o";
    ASSERT_EQ(executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX1251), 1u);
    AmdGpuCodeObject *code_object = executable.code_object(ROCJITSU_CODE_TARGET_GFX1251, 0);
    ASSERT_NE(code_object, nullptr);
    ASSERT_EQ(code_object->target_id(), ROCJITSU_CODE_TARGET_GFX1251);
    ASSERT_FALSE(code_object->text_sections().empty());

    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1251);
    ASSERT_NE(decoder, nullptr);
    size_t instruction_count = 0;
    bool all_instructions_executable = true;
    for (const Section *section : code_object->text_sections()) {
      ASSERT_EQ(section->size() % sizeof(uint32_t), 0u);
      const auto *words = reinterpret_cast<const uint32_t *>(section->data());
      std::span<const uint32_t> remaining(words, section->size() / sizeof(uint32_t));
      uint64_t pc = section->vaddr();
      while (!remaining.empty()) {
        DecodeResult decoded = decoder->decode_window(remaining, pc);
        ASSERT_TRUE(decoded.succeeded()) << "decode failed at PC 0x" << std::hex << pc;
        ASSERT_NE(decoded.value(), nullptr);
        all_instructions_executable &= decoded.value()->execute != nullptr;
        EXPECT_NE(decoded.value()->execute, nullptr)
            << decoded.value()->mnemonic() << " at PC 0x" << std::hex << pc;
        const size_t instruction_words = decoded.value()->size() / sizeof(uint32_t);
        ASSERT_GT(instruction_words, 0u);
        ASSERT_LE(instruction_words, remaining.size());
        remaining = remaining.subspan(instruction_words);
        pc += decoded.value()->size();
        ++instruction_count;
      }
    }
    ASSERT_GT(instruction_count, 0u);
    ASSERT_TRUE(all_instructions_executable);

    code_object->load_to_memory(memory_, kCodeObjectBase);
    const uint64_t descriptor_offset = code_object->kernel_descriptor_offset(kernel_name);
    ASSERT_NE(descriptor_offset, 0u) << "Kernel descriptor symbol not found";
    kernel_object = kCodeObjectBase + descriptor_offset;
  }

  void dispatch(uint64_t kernel_object, uint32_t grid_size, uint64_t kernarg_address) {
    memory_->write64(kCompletionSignal + kSignalValueOffset, 1);

    hsa_kernel_dispatch_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    packet.setup = 1;
    packet.workgroup_size_x = kWorkgroupSize;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = grid_size;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
    packet.kernel_object = kernel_object;
    packet.kernarg_address = reinterpret_cast<void *>(kernarg_address);
    packet.completion_signal.handle = kCompletionSignal;

    auto *command_processor = soc_->xcd(0)->command_processor();
    test::AqlQueue queue(memory_, command_processor);
    queue.submit(packet);
    engine_->run();
    soc_->flush_all();

    EXPECT_EQ(command_processor->dispatched_count(), 1u);
    EXPECT_EQ(memory_->read64(test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
    EXPECT_EQ(memory_->read64(kCompletionSignal + kSignalValueOffset), 0u);
  }

  SoC *soc_ = nullptr;
  amdgpu::GpuMemory *memory_ = nullptr;
  std::unique_ptr<simdojo::SimulationEngine> engine_;
};

TEST_F(Gfx1251CompiledKernelTest, ExecutesVectorAddAcrossTwoWave32Wavefronts) {
  constexpr uint32_t kElementCount = 64;
  uint64_t kernel_object = 0;
  ASSERT_NO_FATAL_FAILURE(load_compiled_kernel("vector_add_gfx1251", "vector_add", kernel_object));

  std::vector<float> input_a(kElementCount);
  std::vector<float> input_b(kElementCount);
  std::vector<float> expected(kElementCount);
  for (uint32_t index = 0; index < kElementCount; ++index) {
    input_a[index] = static_cast<float>(index);
    input_b[index] = static_cast<float>(index * 2);
    expected[index] = static_cast<float>(index * 3);
  }
  const size_t vector_bytes = kElementCount * sizeof(float);
  memory_->load_image(reinterpret_cast<const uint8_t *>(input_a.data()), vector_bytes,
                      kInputAAddress);
  memory_->load_image(reinterpret_cast<const uint8_t *>(input_b.data()), vector_bytes,
                      kInputBAddress);
  std::vector<float> output(kElementCount, -1.0f);
  memory_->load_image(reinterpret_cast<const uint8_t *>(output.data()), vector_bytes,
                      kOutputAddress);

  struct {
    uint64_t input_a;
    uint64_t input_b;
    uint64_t output;
    uint32_t element_count;
  } arguments{kInputAAddress, kInputBAddress, kOutputAddress, kElementCount};
  memory_->load_image(reinterpret_cast<const uint8_t *>(&arguments), sizeof(arguments),
                      kKernargAddress);

  dispatch(kernel_object, kElementCount, kKernargAddress);

  for (uint32_t index = 0; index < kElementCount; ++index) {
    const float actual =
        std::bit_cast<float>(memory_->read32(kOutputAddress + index * sizeof(float)));
    EXPECT_FLOAT_EQ(actual, expected[index]) << "element " << index;
  }
}

TEST_F(Gfx1251CompiledKernelTest, ExecutesNaiveMatmulAndMatchesCpuGolden) {
  constexpr uint32_t kDimension = 4;
  constexpr uint32_t kElementCount = kDimension * kDimension;
  uint64_t kernel_object = 0;
  ASSERT_NO_FATAL_FAILURE(
      load_compiled_kernel("matmul_naive_gfx1251", "matmul_naive", kernel_object));

  std::vector<float> input_a(kElementCount);
  std::vector<float> input_b(kElementCount);
  for (uint32_t index = 0; index < kElementCount; ++index) {
    input_a[index] = static_cast<float>((index % 5) + 1);
    input_b[index] = static_cast<float>((index % 3) + 1);
  }
  std::vector<float> expected(kElementCount, 0.0f);
  for (uint32_t row = 0; row < kDimension; ++row)
    for (uint32_t column = 0; column < kDimension; ++column)
      for (uint32_t inner = 0; inner < kDimension; ++inner)
        expected[row * kDimension + column] +=
            input_a[row * kDimension + inner] * input_b[inner * kDimension + column];

  const size_t matrix_bytes = kElementCount * sizeof(float);
  memory_->load_image(reinterpret_cast<const uint8_t *>(input_a.data()), matrix_bytes,
                      kInputAAddress);
  memory_->load_image(reinterpret_cast<const uint8_t *>(input_b.data()), matrix_bytes,
                      kInputBAddress);
  std::vector<float> output(kElementCount, -1.0f);
  memory_->load_image(reinterpret_cast<const uint8_t *>(output.data()), matrix_bytes,
                      kOutputAddress);

  struct {
    uint64_t input_a;
    uint64_t input_b;
    uint64_t output;
    uint32_t dimension;
  } arguments{kInputAAddress, kInputBAddress, kOutputAddress, kDimension};
  memory_->load_image(reinterpret_cast<const uint8_t *>(&arguments), sizeof(arguments),
                      kKernargAddress);

  dispatch(kernel_object, kWorkgroupSize, kKernargAddress);

  for (uint32_t index = 0; index < kElementCount; ++index) {
    const float actual =
        std::bit_cast<float>(memory_->read32(kOutputAddress + index * sizeof(float)));
    EXPECT_NEAR(actual, expected[index], 1e-6f) << "element " << index;
  }
}

} // namespace

#else

TEST(Gfx1251CompiledKernelAvailabilityTest, DeviceKernelFixturesUnavailable) {
  GTEST_SKIP() << "gfx1251 device-kernel fixtures are unavailable; see CMake configure output";
}

#endif // HAS_GFX1251_DEVICE_KERNELS
