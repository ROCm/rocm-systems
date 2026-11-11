// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scoped_temp.h"

#include "checkpoint_generated.h"
#include "rocjitsu/config/checkpoint.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;

void expect_restore_failure(const std::string &path, std::string_view expected) {
  std::string message;
  unsigned diagnostics = 0;
  auto sink = [&](std::string_view text) {
    message.assign(text);
    ++diagnostics;
  };
  // Call below the C API's exception boundary: reverting a returned failure to
  // a throw must fail this check even though the C API would catch it.
  EXPECT_NO_THROW({
    util::FailureOr<config::LoadedConfig> result =
        config::restore_checkpoint(path, util::DiagnosticEmitter(sink));
    EXPECT_TRUE(result.failed());
  });
  EXPECT_EQ(diagnostics, 1u);
  EXPECT_EQ(message, expected);
}

struct InvalidCheckpoint {
  bool missing_config = false;
  bool missing_vm = false;
  const char *arch = "cdna3";
  uint32_t num_xcds = 1;
  uint32_t num_shader_engines = 1;
  uint32_t num_compute_units = 1;
  bool wavefront = false;
  uint32_t wf_id = 0;
  uint8_t vgpr_lane_count = 64;
  size_t vgpr_bytes = 0;
};

[[nodiscard]] testing::AssertionResult write_checkpoint(test::ScopedTempFile &file,
                                                        const InvalidCheckpoint &input) {
  flatbuffers::FlatBufferBuilder builder;
  auto arch = builder.CreateString(input.arch);
  auto cu = fb::CreateComputeUnitConfig(builder, 1, 104, 256, 64);
  auto se = fb::CreateShaderEngineConfig(builder, input.num_compute_units, cu);
  auto xcd = fb::CreateXcdConfig(builder, input.num_shader_engines, se);
  auto gpu = fb::CreateAmdgpuConfig(builder, input.num_xcds, 0, xcd);
  auto vm = fb::CreateVirtualMachineConfig(builder, arch, gpu);
  auto mode = builder.CreateString("functional");
  auto simulation = fb::CreateSimulationConfig(builder, 0, 1, mode, input.missing_vm ? 0 : vm);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::ComputeUnitState>>> cus;
  if (input.wavefront) {
    auto vgprs = builder.CreateVector(std::vector<uint8_t>(input.vgpr_bytes));
    fb::WavefrontStateBuilder wave(builder);
    wave.add_wf_id(input.wf_id);
    wave.add_vgpr_lane_count(input.vgpr_lane_count);
    wave.add_vgprs(vgprs);
    auto wave_offset = wave.Finish();
    auto waves = builder.CreateVector(&wave_offset, 1);
    auto state = fb::CreateComputeUnitState(builder, 0, waves);
    cus = builder.CreateVector(&state, 1);
  }
  auto checkpoint =
      fb::CreateSimulationCheckpoint(builder, 0, input.missing_config ? 0 : simulation, cus);
  builder.Finish(checkpoint);
  if (testing::AssertionResult result = file.initialize("rocjitsu-restore-errors-"); !result)
    return result;
  return file.write(std::string_view(reinterpret_cast<const char *>(builder.GetBufferPointer()),
                                     builder.GetSize()));
}

TEST(CheckpointRestoreErrorsTest, MissingFileReturnsFailure) {
  test::ScopedTempDirectory directory;
  ASSERT_TRUE(directory.initialize("rocjitsu-restore-errors-"));
  const std::string path = directory.path() + "/absent";
  expect_restore_failure(path, "Cannot open checkpoint file: " + path);
}

TEST(CheckpointRestoreErrorsTest, EmptyFileReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rocjitsu-restore-errors-"));
  expect_restore_failure(file.path(), "Empty or unreadable checkpoint file: " + file.path());
}

TEST(CheckpointRestoreErrorsTest, InvalidFormatReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(file.initialize("rocjitsu-restore-errors-"));
  ASSERT_TRUE(file.write("not a checkpoint"));
  expect_restore_failure(file.path(), "Invalid checkpoint format: " + file.path());
}

TEST(CheckpointRestoreErrorsTest, MissingConfigReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.missing_config = true}));
  expect_restore_failure(file.path(), "Checkpoint missing config section: " + file.path());
}

TEST(CheckpointRestoreErrorsTest, MissingVmReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.missing_vm = true}));
  expect_restore_failure(file.path(), "Checkpoint missing simulation config");
}

TEST(CheckpointRestoreErrorsTest, MissingOrInvalidArchitectureReturnsFailure) {
  for (const char *arch : {"", "unknown"}) {
    SCOPED_TRACE(arch);
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, {.arch = arch}));
    expect_restore_failure(file.path(), "Checkpoint has missing or invalid architecture");
  }
}

TEST(CheckpointRestoreErrorsTest, CuCapacityOverflowReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(
      file, {.num_shader_engines = std::numeric_limits<uint32_t>::max(), .num_compute_units = 2}));
  expect_restore_failure(file.path(), "Checkpoint execution capacity exceeds uint32 range");
}

TEST(CheckpointRestoreErrorsTest, TotalCapacityOverflowReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(
      file, {.num_xcds = std::numeric_limits<uint32_t>::max(), .num_compute_units = 3}));
  expect_restore_failure(file.path(), "Checkpoint execution capacity exceeds uint32 range");
}

TEST(CheckpointRestoreErrorsTest, UnavailableWavefrontSlotReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.wavefront = true, .wf_id = 1}));
  expect_restore_failure(file.path(), "Failed to restore wavefront into its recorded slot");
}

TEST(CheckpointRestoreErrorsTest, InvalidVgprLaneCountReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.wavefront = true, .vgpr_lane_count = 16}));
  expect_restore_failure(file.path(), "Invalid VGPR checkpoint lane count");
}

TEST(CheckpointRestoreErrorsTest, InvalidVgprPayloadReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.wavefront = true, .vgpr_bytes = 1}));
  expect_restore_failure(file.path(), "Invalid VGPR checkpoint payload size");
}

TEST(CheckpointRestoreErrorsTest, UnrecognizedLegacyVgprWidthReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.wavefront = true, .vgpr_lane_count = 0, .vgpr_bytes = 1}));
  expect_restore_failure(file.path(), "Invalid VGPR checkpoint lane count");
}

} // namespace
