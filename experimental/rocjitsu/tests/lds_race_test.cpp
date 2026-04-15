// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file End-to-end test for LDS race detection.
///
/// Dispatches lds_race.hip (2 wavefronts, no barrier between LDS write and
/// read) with race detection enabled and verifies that violations are reported.

#include "aql_queue.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/kernel_metadata.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/race_detection_plugin.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include <gtest/gtest.h>

#include <vector>

#ifdef HAS_DEVICE_KERNELS

namespace {

using namespace rocjitsu;
using rocjitsu::amdgpu::RaceDetectionPlugin;

const std::string kSchemaPath = std::string(SCHEMA_DIR) + "/simulation_config.fbs";
const std::string kConfigPath = std::string(CONFIG_DIR) + "/amdgpu_cdna4.json";

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

constexpr uint64_t KD_ADDR = 0x800000;
constexpr uint64_t OUT_ADDR = 0x100000;
constexpr uint64_t KERNARG_ADDR = 0x400000;

// Dispatch lds_race kernel with race detection enabled.
// The kernel has 2 wavefronts per workgroup writing/reading the same LDS
// locations without a barrier. The race detector should report violations.
TEST(LdsRaceTest, DetectsRace) {
  Executable exec(kernel_path("lds_race"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load lds_race.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("lds_race");
  ASSERT_NE(kernel_object, KD_ADDR) << "kernel descriptor not found";

  // Write output buffer (zeroed).
  std::vector<int> zeros(64, 0);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()),
                     zeros.size() * sizeof(int), OUT_ADDR);

  // Write kernarg: (int* out).
  uint64_t out_ptr = OUT_ADDR;
  memory->load_image(reinterpret_cast<const uint8_t *>(&out_ptr), sizeof(out_ptr),
                     KERNARG_ADDR);

  // Write hidden args for blockDim.x = 128 (2 wavefronts of 64).
  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "lds_race");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  if (group_idx) {
    uint16_t group_size = 128;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*group_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&group_size),
                       sizeof(group_size), addr);
  }
  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  if (block_idx) {
    uint32_t block_count = 1;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*block_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&block_count),
                       sizeof(block_count), addr);
  }

  // Dispatch with race detection enabled.
  auto *cp = soc->xcd(0)->command_processor();
  auto plugin_group = std::make_unique<amdgpu::ExecutionPluginGroup>();
  auto race_plugin = std::make_unique<RaceDetectionPlugin>();
  auto *rp = race_plugin.get();
  plugin_group->add(std::move(race_plugin));
  soc->set_plugin_group(std::move(plugin_group));

  test::AqlQueue queue(memory, cp);
  // grid_size = 128 (1 workgroup of 128 threads = 2 wavefronts).
  queue.dispatch(kernel_object, 128, 128, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  // Verify: race detector should have found violations with diagnostics.
  auto &diagnostics = rp->diagnostics();
  EXPECT_FALSE(diagnostics.empty())
      << "Expected LDS race diagnostics but none were generated";

  size_t total = 0;
  for (const auto &[wg_id, messages] : diagnostics) {
    total += messages.size();
  }
  fprintf(stderr, "\n%zu race diagnostic(s) detected:\n", total);
  for (const auto &[wg_id, messages] : diagnostics) {
    for (size_t i = 0; i < messages.size(); ++i) {
      fprintf(stderr, "  %s\n", messages[i].c_str());
    }
  }
}

constexpr uint64_t IN_ADDR = 0x200000;

// Dispatch vgpr_race kernel with race detection enabled.
// The kernel issues a global_load_dword and immediately reads the destination
// VGPR without s_waitcnt. The race detector should report VGPR violations.
TEST(VgprRaceTest, DetectsRace) {
  Executable exec(kernel_path("vgpr_race"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vgpr_race.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("vgpr_race");
  ASSERT_NE(kernel_object, KD_ADDR) << "kernel descriptor not found";

  // Write input buffer.
  std::vector<int> input(64);
  for (int i = 0; i < 64; ++i)
    input[i] = i + 1;
  memory->load_image(reinterpret_cast<const uint8_t *>(input.data()),
                     input.size() * sizeof(int), IN_ADDR);

  // Write output buffer (zeroed).
  std::vector<int> zeros(64, 0);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()),
                     zeros.size() * sizeof(int), OUT_ADDR);

  // Write kernarg: (int *out, const int *in).
  uint64_t kernarg_data[2] = {OUT_ADDR, IN_ADDR};
  memory->load_image(reinterpret_cast<const uint8_t *>(kernarg_data),
                     sizeof(kernarg_data), KERNARG_ADDR);

  // Write hidden args for blockDim.x = 64 (1 wavefront).
  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "vgpr_race");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  if (group_idx) {
    uint16_t group_size = 64;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*group_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&group_size),
                       sizeof(group_size), addr);
  }
  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  if (block_idx) {
    uint32_t block_count = 1;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*block_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&block_count),
                       sizeof(block_count), addr);
  }

  // Dispatch with race detection enabled.
  auto *cp = soc->xcd(0)->command_processor();
  auto plugin_group = std::make_unique<amdgpu::ExecutionPluginGroup>();
  auto race_plugin = std::make_unique<RaceDetectionPlugin>();
  auto *rp = race_plugin.get();
  plugin_group->add(std::move(race_plugin));
  soc->set_plugin_group(std::move(plugin_group));

  test::AqlQueue queue(memory, cp);
  // grid_size = 64 (1 workgroup of 64 threads = 1 wavefront).
  queue.dispatch(kernel_object, 64, 64, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  // Verify: race detector should have found VGPR violations.
  auto &diagnostics = rp->diagnostics();
  ASSERT_FALSE(diagnostics.empty())
      << "Expected VGPR race diagnostics but none were generated";

  // Collect all messages across workgroups.
  std::vector<std::string> all_messages;
  for (const auto &[wg_id, messages] : diagnostics) {
    all_messages.insert(all_messages.end(), messages.begin(), messages.end());
  }

  fprintf(stderr, "\n%zu VGPR race diagnostic(s) detected:\n", all_messages.size());
  for (const auto &msg : all_messages) {
    fprintf(stderr, "  %s\n", msg.c_str());
  }

  // Every diagnostic should mention VGPR race and the conflicting load.
  for (const auto &msg : all_messages) {
    EXPECT_NE(msg.find("VGPR race on v"), std::string::npos) << msg;
    EXPECT_NE(msg.find("conflicts with pending load"), std::string::npos) << msg;
  }
}

// Dispatch dtl_load kernel (direct-to-LDS buffer load) without s_waitcnt
// vmcnt(0) between the buffer_load_dword...lds and ds_read_b32. The race
// detector should report LDS race violations for the missing synchronization.
TEST(DtlTest, DetectsRace) {
  Executable exec(kernel_path("dtl_basic"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load dtl_basic.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("dtl_load");
  ASSERT_NE(kernel_object, KD_ADDR) << "kernel descriptor not found";

  // Write input buffer (float values 1.0, 2.0, ..., 64.0).
  std::vector<float> input(64);
  for (int i = 0; i < 64; ++i)
    input[i] = static_cast<float>(i + 1);
  memory->load_image(reinterpret_cast<const uint8_t *>(input.data()),
                     input.size() * sizeof(float), IN_ADDR);

  // Write output buffer (zeroed).
  std::vector<float> zeros(64, 0.0f);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()),
                     zeros.size() * sizeof(float), OUT_ADDR);

  // Write kernarg: (const float *input, float *output).
  uint64_t kernarg_data[2] = {IN_ADDR, OUT_ADDR};
  memory->load_image(reinterpret_cast<const uint8_t *>(kernarg_data),
                     sizeof(kernarg_data), KERNARG_ADDR);

  // Write hidden args for blockDim.x = 64 (1 wavefront).
  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "dtl_load");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  if (group_idx) {
    uint16_t group_size = 64;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*group_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&group_size),
                       sizeof(group_size), addr);
  }
  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  if (block_idx) {
    uint32_t block_count = 1;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*block_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&block_count),
                       sizeof(block_count), addr);
  }

  auto *cp = soc->xcd(0)->command_processor();
  auto plugin_group = std::make_unique<amdgpu::ExecutionPluginGroup>();
  auto race_plugin = std::make_unique<RaceDetectionPlugin>();
  auto *rp = race_plugin.get();
  plugin_group->add(std::move(race_plugin));
  soc->set_plugin_group(std::move(plugin_group));

  test::AqlQueue queue(memory, cp);
  queue.dispatch(kernel_object, 64, 64, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  // Verify: race detector should have found LDS violations from the missing
  // s_waitcnt vmcnt(0) between buffer_load_dword...lds and ds_read_b32.
  auto &diagnostics = rp->diagnostics();
  ASSERT_FALSE(diagnostics.empty())
      << "Expected LDS race diagnostics but none were generated";

  std::vector<std::string> all_messages;
  for (const auto &[wg_id, messages] : diagnostics) {
    all_messages.insert(all_messages.end(), messages.begin(), messages.end());
  }

  fprintf(stderr, "\n%zu DTL race diagnostic(s) detected:\n", all_messages.size());
  for (const auto &msg : all_messages) {
    fprintf(stderr, "  %s\n", msg.c_str());
  }

  for (const auto &msg : all_messages) {
    EXPECT_NE(msg.find("LDS race at byte"), std::string::npos) << msg;
    EXPECT_NE(msg.find("conflicts with wave 0 write"), std::string::npos) << msg;
  }

  // Numerical validation: output[i] should be input[i] + 1.0f.
  // (Functional mode loads synchronously, so results are correct despite the race.)
  for (int i = 0; i < 64; ++i) {
    uint32_t bits = memory->read32(OUT_ADDR + i * 4);
    float val;
    memcpy(&val, &bits, sizeof(float));
    float expected = static_cast<float>(i + 1) + 1.0f;
    EXPECT_FLOAT_EQ(val, expected)
        << "output[" << i << "] = " << val << ", expected " << expected;
  }
}

// Reverse array via LDS: out[i] = in[63-i].
// Step 1: validate numerical correctness without race detection.
TEST(LgkmVgprRaceTest, DetectsRace) {
  Executable exec(kernel_path("lgkm_vgpr_race"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load lgkm_vgpr_race.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("lgkm_vgpr_race");
  ASSERT_NE(kernel_object, KD_ADDR) << "kernel descriptor not found";

  // Write input buffer: in[i] = i.
  constexpr int N = 64;
  std::vector<int> input(N);
  for (int i = 0; i < N; ++i)
    input[i] = i;
  memory->load_image(reinterpret_cast<const uint8_t *>(input.data()),
                     input.size() * sizeof(int), IN_ADDR);

  // Write output buffer (zeroed).
  std::vector<int> zeros(N, 0);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()),
                     zeros.size() * sizeof(int), OUT_ADDR);

  // Write kernarg: (const int *in, int *out).
  uint64_t kernarg_data[2] = {IN_ADDR, OUT_ADDR};
  memory->load_image(reinterpret_cast<const uint8_t *>(kernarg_data),
                     sizeof(kernarg_data), KERNARG_ADDR);

  // Write hidden args for blockDim.x = 64 (1 wavefront).
  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "lgkm_vgpr_race");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  if (group_idx) {
    uint16_t group_size = N;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*group_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&group_size),
                       sizeof(group_size), addr);
  }
  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  if (block_idx) {
    uint32_t block_count = 1;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*block_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&block_count),
                       sizeof(block_count), addr);
  }

  auto *cp = soc->xcd(0)->command_processor();
  auto plugin_group = std::make_unique<amdgpu::ExecutionPluginGroup>();
  auto race_plugin = std::make_unique<RaceDetectionPlugin>();
  auto *rp = race_plugin.get();
  plugin_group->add(std::move(race_plugin));
  soc->set_plugin_group(std::move(plugin_group));

  test::AqlQueue queue(memory, cp);
  queue.dispatch(kernel_object, N, N, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  // Numerical validation: out[i] should be in[63-i] = 63-i.
  for (int i = 0; i < N; ++i) {
    int val = static_cast<int>(memory->read32(OUT_ADDR + i * 4));
    EXPECT_EQ(val, 63 - i) << "output[" << i << "]";
  }

  // Verify VGPR race diagnostics from the missing lgkmcnt on the inline asm
  // ds_read_b32. The v_add_u32 reads the VGPR before the ds_read completes.
  auto &diagnostics = rp->diagnostics();
  ASSERT_FALSE(diagnostics.empty())
      << "Expected VGPR race diagnostics (lgkmcnt) but none were generated";

  std::vector<std::string> all_messages;
  for (const auto &[wg_id, messages] : diagnostics) {
    all_messages.insert(all_messages.end(), messages.begin(), messages.end());
  }

  fprintf(stderr, "\n%zu LDS_TO_VGPR race diagnostic(s) detected:\n", all_messages.size());
  for (const auto &msg : all_messages) {
    fprintf(stderr, "  %s\n", msg.c_str());
  }

  for (const auto &msg : all_messages) {
    EXPECT_NE(msg.find("VGPR race on v"), std::string::npos) << msg;
    EXPECT_NE(msg.find("conflicts with pending load"), std::string::npos) << msg;
  }
}

// Dispatch sgpr_race kernel: s_load_dword loads into an SGPR, then that SGPR
// is used without s_waitcnt lgkmcnt(0). Tests GLOBAL_TO_SGPR race detection.
TEST(SgprRaceTest, DetectsRace) {
  Executable exec(kernel_path("sgpr_race"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load sgpr_race.o";
  auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto loaded = config::load_config(kConfigPath, kSchemaPath);
  auto *soc = loaded.soc();
  auto *memory = loaded.memory();
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  loaded.wire_links(engine->topology());
  engine->build();

  co->load_to_memory(memory, KD_ADDR);
  uint64_t kernel_object = KD_ADDR + co->kernel_descriptor_offset("sgpr_race");
  ASSERT_NE(kernel_object, KD_ADDR) << "kernel descriptor not found";

  // Write input buffer.
  int input_val = 42;
  memory->load_image(reinterpret_cast<const uint8_t *>(&input_val),
                     sizeof(input_val), IN_ADDR);

  // Write output buffer (zeroed).
  std::vector<int> zeros(64, 0);
  memory->load_image(reinterpret_cast<const uint8_t *>(zeros.data()),
                     zeros.size() * sizeof(int), OUT_ADDR);

  // Write kernarg: (int *out, const int *in).
  uint64_t kernarg_data[2] = {OUT_ADDR, IN_ADDR};
  memory->load_image(reinterpret_cast<const uint8_t *>(kernarg_data),
                     sizeof(kernarg_data), KERNARG_ADDR);

  // Write hidden args for blockDim.x = 64 (1 wavefront).
  auto parsed = KernelArgs::parse(
      reinterpret_cast<const uint8_t *>(co->image_data()), co->image_size(),
      "sgpr_race");
  ASSERT_TRUE(std::holds_alternative<KernelArgs>(parsed))
      << std::get<std::string>(parsed);
  auto &args = std::get<KernelArgs>(parsed);
  auto group_idx = args.getIndexFromValueKind("hidden_group_size_x");
  if (group_idx) {
    uint16_t group_size = 64;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*group_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&group_size),
                       sizeof(group_size), addr);
  }
  auto block_idx = args.getIndexFromValueKind("hidden_block_count_x");
  if (block_idx) {
    uint32_t block_count = 1;
    uint64_t addr = KERNARG_ADDR + args.getOffsetFromIndex(*block_idx);
    memory->load_image(reinterpret_cast<const uint8_t *>(&block_count),
                       sizeof(block_count), addr);
  }

  auto *cp = soc->xcd(0)->command_processor();
  auto plugin_group = std::make_unique<amdgpu::ExecutionPluginGroup>();
  auto race_plugin = std::make_unique<RaceDetectionPlugin>();
  auto *rp = race_plugin.get();
  plugin_group->add(std::move(race_plugin));
  soc->set_plugin_group(std::move(plugin_group));

  test::AqlQueue queue(memory, cp);
  queue.dispatch(kernel_object, 64, 64, KERNARG_ADDR);

  engine->run();
  soc->flush_all();

  auto &diagnostics = rp->diagnostics();
  ASSERT_FALSE(diagnostics.empty())
      << "Expected SGPR race diagnostics (lgkmcnt) but none were generated";

  std::vector<std::string> all_messages;
  for (const auto &[wg_id, messages] : diagnostics) {
    all_messages.insert(all_messages.end(), messages.begin(), messages.end());
  }

  fprintf(stderr, "\n%zu SGPR race diagnostic(s) detected:\n", all_messages.size());
  for (const auto &msg : all_messages) {
    fprintf(stderr, "  %s\n", msg.c_str());
  }

  for (const auto &msg : all_messages) {
    EXPECT_NE(msg.find("SGPR race on s"), std::string::npos) << msg;
    EXPECT_NE(msg.find("conflicts with pending load"), std::string::npos) << msg;
  }
}

} // namespace

#endif // HAS_DEVICE_KERNELS
