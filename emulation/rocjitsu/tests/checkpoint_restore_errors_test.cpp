// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scoped_temp.h"

#include "checkpoint_generated.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/vm/soc.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
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

struct Gfx10WaveResources {
  uint32_t vgprs = 16;
  uint32_t lds_base = 0;
  uint64_t scratch_base = 0;
  uint32_t trapsts = 0;
};

struct InvalidCheckpoint {
  bool missing_config = false;
  bool missing_vm = false;
  const char *arch = "cdna3";
  uint32_t num_xcds = 1;
  uint32_t num_shader_engines = 1;
  uint32_t num_compute_units = 1;
  uint32_t num_wf_slots = 1;
  bool has_lds_state = false;
  bool wavefront = false;
  uint32_t wf_id = 0;
  uint8_t vgpr_lane_count = 64;
  size_t vgpr_bytes = 0;
  bool halted = false;
  std::optional<fb::GraphicsWait> graphics_wait = std::nullopt;
  std::optional<Gfx10WaveResources> gfx10_resources = std::nullopt;
  std::optional<uint32_t> rdna2_process_id = std::nullopt;
  bool rdna2_internal_address_space = false;
  std::optional<std::vector<uint32_t>> rdna2_register_blocks = std::nullopt;
  std::vector<uint32_t> sgprs = {};
  uint8_t kernel_wave_size = 0;
  std::optional<uint32_t> num_vgprs = std::nullopt;
  uint32_t accum_vgpr_base = 0;
  std::optional<uint32_t> num_sgprs = std::nullopt;
};

[[nodiscard]] testing::AssertionResult write_checkpoint(test::ScopedTempFile &file,
                                                        const InvalidCheckpoint &input) {
  flatbuffers::FlatBufferBuilder builder;
  auto arch = builder.CreateString(input.arch);
  auto cu = fb::CreateComputeUnitConfig(builder, input.num_wf_slots, 104, 256, 64);
  auto se = fb::CreateShaderEngineConfig(builder, input.num_compute_units, cu);
  auto xcd = fb::CreateXcdConfig(builder, input.num_shader_engines, se);
  auto gpu = fb::CreateAmdgpuConfig(builder, input.num_xcds, 0, xcd);
  auto vm = fb::CreateVirtualMachineConfig(builder, arch, gpu);
  auto mode = builder.CreateString("functional");
  auto simulation = fb::CreateSimulationConfig(builder, 0, 1, mode, input.missing_vm ? 0 : vm);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fb::ComputeUnitState>>> cus;
  if (input.wavefront) {
    auto vgprs = builder.CreateVector(std::vector<uint8_t>(input.vgpr_bytes));
    const bool cdna3 = std::string_view(input.arch) == "cdna3";
    const fb::LdsWindow empty_window(0, 0);
    const auto sgprs = input.sgprs.empty() ? 0 : builder.CreateVector(input.sgprs);
    const fb::Gfx10Apertures apertures;
    const fb::Gfx10WaitLimits wait_limits;
    const auto &resources = input.gfx10_resources;
    const auto wave_resources =
        resources
            ? fb::CreateGfx10WaveResources(builder, &apertures, false, &wait_limits,
                                           resources->trapsts, resources->scratch_base, 0,
                                           resources->lds_base, 0, 0, 0, 0, 0, resources->vgprs)
            : 0;
    fb::WavefrontStateBuilder wave(builder);
    wave.add_gfx10_resources(wave_resources);
    wave.add_kernel_wave_size(input.kernel_wave_size);
    wave.add_halted(input.halted);
    if (input.graphics_wait)
      wave.add_graphics_wait(*input.graphics_wait);
    if (input.rdna2_process_id)
      wave.add_rdna2_process_id(*input.rdna2_process_id);
    wave.add_rdna2_internal_address_space(input.rdna2_internal_address_space);
    if (cdna3 || input.has_lds_state)
      wave.add_lds_window(&empty_window);
    if (sgprs.o)
      wave.add_sgprs(sgprs);
    if (input.num_sgprs)
      wave.add_num_sgprs(*input.num_sgprs);
    wave.add_wf_id(input.wf_id);
    wave.add_vgpr_lane_count(input.vgpr_lane_count);
    wave.add_vgprs(vgprs);
    if (input.num_vgprs)
      wave.add_num_vgprs(*input.num_vgprs);
    wave.add_accum_vgpr_base(input.accum_vgpr_base);
    auto wave_offset = wave.Finish();
    auto waves = builder.CreateVector(&wave_offset, 1);
    const auto lds =
        cdna3 || input.has_lds_state ? builder.CreateVector(std::vector<uint8_t>{}) : 0;
    const auto blocks =
        input.rdna2_register_blocks ? builder.CreateVector(*input.rdna2_register_blocks) : 0;
    const auto cu_resources = resources ? fb::CreateGfx10CuResources(builder, &apertures) : 0;
    fb::ComputeUnitStateBuilder state(builder);
    state.add_gfx10_resources(cu_resources);
    if (blocks.o)
      state.add_rdna2_register_blocks(blocks);
    state.add_wavefronts(waves);
    state.add_lds(lds);
    auto cu = state.Finish();
    cus = builder.CreateVector(&cu, 1);
  }
  auto checkpoint =
      fb::CreateSimulationCheckpoint(builder, 0, input.missing_config ? 0 : simulation, cus);
  builder.Finish(checkpoint);
  if (testing::AssertionResult result = file.initialize("rocjitsu-restore-errors-"); !result)
    return result;
  return file.write(std::string_view(reinterpret_cast<const char *>(builder.GetBufferPointer()),
                                     builder.GetSize()));
}

TEST(CheckpointRestoreErrorsTest, LegacyVgprCountsDefaultToTheConfiguredAllocation) {
  for (const char *arch : {"cdna3", "cdna5", "rdna4"}) {
    SCOPED_TRACE(arch);
    InvalidCheckpoint record;
    record.arch = arch;
    record.wavefront = true;
    record.has_lds_state = true;
    record.kernel_wave_size = std::string_view(arch) == "cdna3" ? 64 : 32;
    record.vgpr_lane_count = record.kernel_wave_size;
    test::ScopedTempFile empty;
    ASSERT_TRUE(write_checkpoint(empty, {.arch = arch}));
    auto configured = config::restore_checkpoint(empty.path(), {});
    ASSERT_TRUE(configured.succeeded());
    const auto *unit = configured.value().soc()->xcd(0)->shader_engine(0)->compute_unit(0);
    record.vgpr_bytes =
        unit->vgpr_allocation_block_size() * record.vgpr_lane_count * sizeof(uint32_t);
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, record));
    util::StringDiagnostic diagnostic;
    auto restored = config::restore_checkpoint(file.path(), diagnostic.emitter());
    ASSERT_TRUE(restored.succeeded()) << diagnostic.message();
    const auto *cu = restored.value().soc()->xcd(0)->shader_engine(0)->compute_unit(0);
    ASSERT_NE(cu->wf(0), nullptr);
    EXPECT_EQ(cu->wf(0)->num_vgprs(), cu->config().vgprs_per_wf);
    record.num_vgprs = UINT32_MAX;
    {
      test::ScopedTempFile invalid_file;
      ASSERT_TRUE(write_checkpoint(invalid_file, record));
      expect_restore_failure(invalid_file.path(), "Invalid checkpoint VGPR allocation");
    }
  }
}

TEST(CheckpointRestoreErrorsTest, RejectsLegacyActiveWavesWithoutLdsResources) {
  for (const char *arch : {"cdna1", "cdna2", "cdna4", "cdna5", "rdna3", "rdna3_5", "rdna4"}) {
    SCOPED_TRACE(arch);
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, {.arch = arch, .wavefront = true}));
    expect_restore_failure(file.path(), "Cannot restore active wave without saved LDS state");
  }
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

TEST(CheckpointRestoreErrorsTest, InvalidComputeUnitConfigurationReturnsFailure) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.num_wf_slots = 33}));
  expect_restore_failure(file.path(), "num_wf_slots exceeds the ISA maximum of 32");
}

TEST(CheckpointRestoreErrorsTest, RejectsActiveRdna1WaveWithoutItsResourceRecord) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.arch = "rdna1", .wavefront = true}));
  expect_restore_failure(
      file.path(),
      "Cannot restore active RDNA1 wave: checkpoint omitted its VGPR and LDS allocations");
}

TEST(CheckpointRestoreErrorsTest, LegacyGfx10CheckpointCanRestoreWithoutAnActiveWave) {
  for (const char *arch : {"rdna1", "rdna2"}) {
    for (bool halted_record : {false, true}) {
      SCOPED_TRACE(arch);
      InvalidCheckpoint input{.arch = arch, .wavefront = halted_record, .halted = halted_record};
      test::ScopedTempFile file;
      ASSERT_TRUE(write_checkpoint(file, input));
      util::StringDiagnostic diagnostic;
      EXPECT_TRUE(config::restore_checkpoint(file.path(), diagnostic.emitter()).succeeded())
          << diagnostic.message();
    }
  }
}

TEST(CheckpointRestoreErrorsTest, ResourceReaderChecksActiveWaveSchedulerState) {
  InvalidCheckpoint running{.arch = "rdna3_5",
                            .has_lds_state = true,
                            .wavefront = true,
                            .vgpr_lane_count = 32,
                            .vgpr_bytes = 256 * 32 * sizeof(uint32_t),
                            .kernel_wave_size = 32};
  for (const auto grant : {std::optional<fb::GraphicsWait>{}, std::optional{fb::GraphicsWait_None},
                           std::optional{fb::GraphicsWait_LaunchDefaults}}) {
    running.graphics_wait = grant;
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, running));
    util::StringDiagnostic diagnostic;
    ASSERT_TRUE(config::restore_checkpoint(file.path(), diagnostic.emitter()).succeeded())
        << diagnostic.message();
  }
  for (const auto grant : {fb::GraphicsWait_Export, fb::GraphicsWait_Parameters}) {
    running.graphics_wait = grant;
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, running));
    expect_restore_failure(invalid_file.path(),
                           "Cannot restore an active wave requiring scheduler state");
  }
}

TEST(CheckpointRestoreErrorsTest, Rdna2RestoresSharedResourcesAndRegisterBlockPlacement) {
  InvalidCheckpoint input{.arch = "rdna2",
                          .num_wf_slots = 2,
                          .wavefront = true,
                          .vgpr_lane_count = 32,
                          .vgpr_bytes = 256 * 32 * sizeof(uint32_t),
                          .gfx10_resources = Gfx10WaveResources{16, 256, 0x34000, 0x20},
                          .rdna2_process_id = 17,
                          .rdna2_internal_address_space = true,
                          .sgprs = std::vector<uint32_t>(104),
                          .kernel_wave_size = 32,
                          .num_sgprs = 96};
  for (bool explicit_block : {false, true}) {
    if (explicit_block)
      input.rdna2_register_blocks = std::vector<uint32_t>{1};
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, input));
    auto loaded = config::restore_checkpoint(file.path(), {});
    ASSERT_TRUE(loaded.succeeded());
    auto *soc = loaded.value().soc();
    auto *cu = soc->xcd(0)->shader_engine(0)->compute_unit(0);
    const auto *wf = cu->wf(0);
    ASSERT_NE(wf, nullptr);
    EXPECT_EQ(wf->lds_base(), 256u);
    EXPECT_EQ(wf->scratch_base(), 0x34000u);
    EXPECT_EQ(wf->process_id(), 17u);
    EXPECT_EQ(wf->address_space(), soc->internal_address_space());
    EXPECT_EQ(wf->num_sgprs(), 96u);
    EXPECT_EQ(wf->num_vgprs(), 16u);
    EXPECT_EQ(wf->trapsts(), 0x20u);
    EXPECT_EQ(wf->vgpr_alloc().base, explicit_block ? cu->vgpr_allocation_block_size() : 0u);
    EXPECT_EQ(wf->sgpr_alloc().base, explicit_block ? cu->sgpr_allocation_block_size() : 0u);
  }
  input.rdna2_register_blocks->clear();
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, input));
    expect_restore_failure(invalid_file.path(), "Invalid RDNA2 checkpoint register block count");
  }
  input.rdna2_register_blocks->push_back(2);
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, input));
    expect_restore_failure(invalid_file.path(),
                           "Failed to restore wavefront into its recorded slot or register block");
  }
}

TEST(CheckpointRestoreErrorsTest, Rdna2ValidatesLogicalAllocationAndPhysicalPayload) {
  InvalidCheckpoint input{.arch = "rdna2",
                          .wavefront = true,
                          .vgpr_lane_count = 32,
                          .vgpr_bytes = 256 * 32 * sizeof(uint32_t),
                          .sgprs = std::vector<uint32_t>(104),
                          .kernel_wave_size = 32};
  auto reject = [&](std::string_view reason) {
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, input));
    expect_restore_failure(file.path(), reason);
  };
  reject("Cannot restore active RDNA2 wave: checkpoint omitted the owning process");
  input.rdna2_process_id = 0;
  reject("Cannot restore active RDNA2 wave: checkpoint omitted LDS and scratch allocation state");
  input.rdna2_process_id = 1;
  input.gfx10_resources.emplace();
  reject("RDNA2 checkpoint process lacks its address space");
  input.rdna2_process_id = 0;
  reject("Cannot restore active RDNA2 wave: checkpoint omitted its register allocation");
  input.num_sgprs = 16;
  input.gfx10_resources->vgprs = 257;
  reject("Invalid GFX10 wave allocation");
  input.num_sgprs = 105;
  input.gfx10_resources->vgprs = 4;
  reject("Invalid RDNA2 checkpoint register allocation");
  input.num_sgprs = 16;
  input.gfx10_resources->vgprs = 0;
  reject("Invalid GFX10 wave allocation");
  input.gfx10_resources->vgprs = 4;
  input.sgprs.resize(16);
  reject("Invalid RDNA2 checkpoint SGPR payload size");
  input.sgprs.resize(104);
  input.vgpr_lane_count = 64;
  reject("Invalid RDNA2 checkpoint VGPR payload or lane count");
  input.vgpr_lane_count = 32;
  input.vgpr_bytes = 4 * 32 * sizeof(uint32_t);
  reject("Invalid VGPR checkpoint payload size");
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
TEST(CheckpointRestoreErrorsTest, Cdna4RejectsInvalidRegisterBoundsBeforeAllocating) {
  for (uint32_t total : {0u, 513u}) {
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(
        file, {.arch = "cdna4", .has_lds_state = true, .wavefront = true, .num_vgprs = total}));
    expect_restore_failure(file.path(), "Invalid CDNA4 checkpoint VGPR allocation");
  }
  for (uint32_t base : {2u, 260u}) {
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, {.arch = "cdna4",
                                        .has_lds_state = true,
                                        .wavefront = true,
                                        .num_vgprs = 32,
                                        .accum_vgpr_base = base}));
    expect_restore_failure(file.path(), "Invalid CDNA4 checkpoint AccVGPR boundary");
  }
  test::ScopedTempFile missing_total;
  ASSERT_TRUE(write_checkpoint(
      missing_total,
      {.arch = "cdna4", .has_lds_state = true, .wavefront = true, .accum_vgpr_base = 8}));
  expect_restore_failure(missing_total.path(), "Invalid CDNA4 checkpoint AccVGPR boundary");
}
TEST(CheckpointRestoreErrorsTest, Cdna4RejectsTruncatedRoundedSgprPayload) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.arch = "cdna4",
                                      .has_lds_state = true,
                                      .wavefront = true,
                                      .sgprs = std::vector<uint32_t>(40),
                                      .num_sgprs = 40}));
  expect_restore_failure(file.path(), "Invalid CDNA4 checkpoint SGPR allocation");
}
TEST(CheckpointRestoreErrorsTest, Cdna4LegacyRecordUsesConfiguredVgprCount) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.arch = "cdna4",
                                      .has_lds_state = true,
                                      .wavefront = true,
                                      .vgpr_bytes = 512 * 64 * sizeof(uint32_t)}));
  auto restored = config::restore_checkpoint(file.path(), {});
  ASSERT_TRUE(restored.succeeded());
  const auto *cu = restored.value().soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  const auto *wave = cu->wf(0);
  ASSERT_NE(wave, nullptr);
  EXPECT_EQ(wave->num_vgprs(), cu->config().vgprs_per_wf);
  EXPECT_EQ(wave->accum_vgpr_base(), 0u);
  EXPECT_EQ(wave->num_sgprs(), cu->config().sgprs_per_wf);
}
TEST(CheckpointRestoreErrorsTest, Cdna4LegacySgprPayloadCarriesItsRequestedCount) {
  std::vector<uint32_t> values(40);
  values.back() = 0xbeefu;
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.arch = "cdna4",
                                      .has_lds_state = true,
                                      .wavefront = true,
                                      .vgpr_bytes = 512 * 64 * sizeof(uint32_t),
                                      .sgprs = values}));
  auto restored = config::restore_checkpoint(file.path(), {});
  ASSERT_TRUE(restored.succeeded());
  const auto *cu = restored.value().soc()->xcd(0)->shader_engine(0)->compute_unit(0);
  const auto *wave = cu->wf(0);
  ASSERT_NE(wave, nullptr);
  EXPECT_EQ(wave->num_sgprs(), 40u);
  EXPECT_EQ(cu->read_sgpr_storage(wave->sgpr_alloc().base + 39), 0xbeefu);
  EXPECT_EQ(cu->read_sgpr_storage(wave->sgpr_alloc().base + 47), 0u);
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
