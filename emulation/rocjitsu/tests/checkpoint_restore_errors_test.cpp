// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "config_test_util.h"
#include "scoped_temp.h"

#include "checkpoint_generated.h"
#include "rocjitsu/config/checkpoint.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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
  std::optional<uint32_t> rdna2_process_id = std::nullopt;
  bool rdna2_internal_address_space = false;
  std::optional<Gfx10WaveResources> gfx10_resources = std::nullopt;
  bool halted = false;
  std::vector<uint32_t> sgprs = {};
  uint8_t kernel_wave_size = 0;
  std::optional<std::vector<uint32_t>> register_blocks = std::nullopt;
  std::vector<uint32_t> program = {};
  uint32_t status = 0;
  bool has_cp_cursor = false;
  bool has_lds_cursor = false;
  std::optional<fb::GraphicsWait> graphics_wait = std::nullopt;
  uint32_t sleep_cycles = 0;
  bool sleep_wakeup = false;
  uint32_t dispatch_id = 0;
  uint8_t waiting_barrier = 0;
  uint8_t barrier_complete_mask = 0;
  std::optional<uint32_t> dispatch_cursor = std::nullopt;
  std::vector<fb::Rdna4CheckpointGroup> groups = {};
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
    const bool has_lds = input.has_lds_state || std::string_view(input.arch) == "cdna3";
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
    if (input.rdna2_process_id)
      wave.add_rdna2_process_id(*input.rdna2_process_id);
    wave.add_rdna2_internal_address_space(input.rdna2_internal_address_space);
    wave.add_halted(input.halted);
    if (has_lds)
      wave.add_lds_window(&empty_window);
    if (sgprs.o)
      wave.add_sgprs(sgprs);
    if (input.num_sgprs)
      wave.add_num_sgprs(*input.num_sgprs);
    if (!input.program.empty()) {
      wave.add_pc(0x1000);
      wave.add_exec(1);
    }
    if (input.graphics_wait)
      wave.add_graphics_wait(*input.graphics_wait);
    wave.add_status(input.status);
    wave.add_wf_id(input.wf_id);
    wave.add_sleep_cycles(input.sleep_cycles);
    wave.add_sleep_wakeup(input.sleep_wakeup);
    wave.add_dispatch_id(input.dispatch_id);
    wave.add_waiting_barrier(input.waiting_barrier);
    wave.add_barrier_complete_mask(input.barrier_complete_mask);
    wave.add_vgpr_lane_count(input.vgpr_lane_count);
    wave.add_vgprs(vgprs);
    if (input.num_vgprs)
      wave.add_num_vgprs(*input.num_vgprs);
    wave.add_accum_vgpr_base(input.accum_vgpr_base);
    auto wave_offset = wave.Finish();
    auto waves = builder.CreateVector(&wave_offset, 1);
    const auto groups = input.dispatch_cursor ? builder.CreateVectorOfStructs(input.groups) : 0;
    const auto blocks = input.register_blocks ? builder.CreateVector(*input.register_blocks) : 0;
    const auto lds = has_lds ? builder.CreateVector(std::vector<uint8_t>{}) : 0;
    const auto cu_resources = resources ? fb::CreateGfx10CuResources(builder, &apertures) : 0;
    fb::ComputeUnitStateBuilder state(builder);
    state.add_gfx10_resources(cu_resources);

    state.add_wavefronts(waves);
    state.add_lds(lds);
    if (input.has_lds_cursor)
      state.add_lds_cursor(0);
    state.add_rdna2_register_blocks(blocks);
    state.add_rdna4_workgroups(groups);
    auto cu = state.Finish();
    cus = builder.CreateVector(&cu, 1);
  }
  flatbuffers::Offset<fb::GpuMemoryState> memory;
  if (!input.program.empty()) {
    std::vector<uint8_t> bytes(4096);
    std::memcpy(bytes.data(), input.program.data(), input.program.size() * sizeof(uint32_t));
    auto page = fb::CreateMemoryPage(builder, 0x1000, builder.CreateVector(bytes));
    memory = fb::CreateGpuMemoryState(builder, builder.CreateVector(&page, 1));
  }
  flatbuffers::Offset<fb::CommandProcessorState> cp;
  if (input.has_cp_cursor || input.dispatch_cursor)
    cp = fb::CreateCommandProcessorState(
        builder, 0, 0, 0,
        builder.CreateVector(std::vector<uint32_t>{input.dispatch_cursor.value_or(1)}));
  auto checkpoint = fb::CreateSimulationCheckpoint(
      builder, 0, input.missing_config ? 0 : simulation, cus, cp, memory);
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

TEST(CheckpointRestoreErrorsTest, RejectsUnsupportedSplitBarriersAndGraphicsWaits) {
  for (std::string_view arch : {"cdna1", "cdna2", "cdna3", "cdna4", "rdna1", "rdna2", "rdna3",
                                "rdna3_5", "rdna4", "cdna5"}) {
    SCOPED_TRACE(arch);
    InvalidCheckpoint record;
    record.arch = arch.data();
    record.wavefront = true;
    const bool split_barriers = arch == "rdna4" || arch == "cdna5";
    const bool gfx10 = arch == "rdna1" || arch == "rdna2";
    if (!split_barriers && !gfx10) {
      record.dispatch_cursor = 1;
      record.dispatch_id = 8;
      record.groups = {fb::Rdna4CheckpointGroup(8, 0, 1, 0, 0, true)};
    }
    const auto check = [&] {
      test::ScopedTempFile file;
      ASSERT_TRUE(write_checkpoint(file, record));
      util::StringDiagnostic error;
      ASSERT_TRUE(config::restore_checkpoint(file.path(), error.emitter()).failed());
      EXPECT_NE(error.message().find("checkpoint wave barrier"), std::string::npos)
          << error.message();
    };
    record.waiting_barrier = 1;
    check();
    record.waiting_barrier = 0;
    record.barrier_complete_mask = split_barriers ? 4 : 1;
    check();
  }
  for (const auto wait : {fb::GraphicsWait_Export, fb::GraphicsWait_Parameters}) {
    for (std::string_view arch : {"cdna3", "rdna3", "rdna3_5", "rdna4"}) {
      SCOPED_TRACE(arch);
      InvalidCheckpoint record;
      record.arch = arch.data();
      record.has_lds_state = true;
      record.wavefront = true;
      record.vgpr_lane_count = record.kernel_wave_size = arch == "cdna3" ? 64 : 32;
      record.vgpr_bytes = 256 * record.vgpr_lane_count * sizeof(uint32_t);
      record.graphics_wait = wait;
      test::ScopedTempFile file;
      ASSERT_TRUE(write_checkpoint(file, record));
      if (arch == "cdna3") {
        expect_restore_failure(file.path(),
                               "Graphics wait on an incompatible checkpoint architecture");
        continue;
      }
      config::LoadedConfig restored;
      ASSERT_TRUE(test::restore_valid_checkpoint(file.path(), restored));
      const auto *wave = restored.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->wf(0);
      ASSERT_NE(wave, nullptr);
      EXPECT_EQ(wave->state(), wait == fb::GraphicsWait_Export ? amdgpu::WfState::EXPORT_WAIT
                                                               : amdgpu::WfState::PARAM_WAIT);
    }
  }
}

TEST(CheckpointRestoreErrorsTest, Rdna4ValidatesWorkgroupGraphBeforeRestoringWaves) {
  InvalidCheckpoint record;
  record.arch = "rdna4";
  record.wavefront = true;
  record.vgpr_lane_count = 32;
  record.dispatch_id = 8;
  record.dispatch_cursor = 8;
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, record));
    expect_restore_failure(invalid_file.path(),
                           "RDNA4 checkpoint dispatch cursor aliases a live wave");
  }
  record.dispatch_cursor = 1;
  record.waiting_barrier = 1;
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, record));
    expect_restore_failure(invalid_file.path(),
                           "Invalid RDNA4 checkpoint wave barrier or wake state");
  }
  record.waiting_barrier = 0;
  record.groups = {fb::Rdna4CheckpointGroup(7, 0, 1, 0, 0, true)};
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, record));
    expect_restore_failure(invalid_file.path(),
                           "RDNA4 checkpoint workgroup member count differs from saved waves");
  }
  record.groups = {fb::Rdna4CheckpointGroup(8, 0, 1, 1, 0, true)};
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, record));
    expect_restore_failure(invalid_file.path(),
                           "Invalid RDNA4 checkpoint workgroup count or duplicate identity");
  }
}

TEST(CheckpointRestoreErrorsTest,
     LegacyRdna4AcceptsOrdinaryWaveAndDiagnosesMissingSleepMembership) {
  InvalidCheckpoint record;
  record.arch = "rdna4";
  record.wavefront = true;
  record.has_lds_state = true;
  record.vgpr_lane_count = 32;
  record.vgpr_bytes = 256 * 32 * sizeof(uint32_t); // A complete wave32 block.
  test::ScopedTempFile ordinary;
  ASSERT_TRUE(write_checkpoint(ordinary, record));
  config::LoadedConfig restored;
  ASSERT_TRUE(test::restore_valid_checkpoint(ordinary.path(), restored));
  record.sleep_cycles = UINT32_MAX;
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, record));
    expect_restore_failure(invalid_file.path(),
                           "Legacy RDNA4 checkpoint lacks workgroup identity for sleeping wave");
  }
}

TEST(CheckpointRestoreErrorsTest, LegacyCdna5RestoresWaveLocalSleepWithoutAGroupTable) {
  InvalidCheckpoint record;
  record.arch = "cdna5";
  record.wavefront = true;
  record.has_lds_state = true;
  record.vgpr_lane_count = 32;
  record.vgpr_bytes = 256 * 32 * sizeof(uint32_t);
  for (uint32_t delay : {0u, 127u, UINT32_MAX}) {
    record.sleep_cycles = delay;
    record.sleep_wakeup = !delay;
    config::LoadedConfig copy;
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, record));
    ASSERT_TRUE(test::restore_valid_checkpoint(file.path(), copy));
    const auto *wave = copy.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->wf(0);
    ASSERT_NE(wave, nullptr);
    EXPECT_EQ(wave->sleep_cycles(), delay);
    EXPECT_EQ(wave->sleep_wakeup(), !delay);
  }
}

TEST(CheckpointTest, LegacyGfx1151PartialGraphicsGrantSurvivesRestore) {
  using amdgpu::Wavefront;
  for (uint32_t grant : {Wavefront::kStatusExportReadyMask, Wavefront::kStatusParamReadyMask}) {
    SCOPED_TRACE(grant);
    InvalidCheckpoint record;
    record.arch = "rdna3_5";
    record.wavefront = true;
    record.kernel_wave_size = record.vgpr_lane_count = 32;
    record.vgpr_bytes = 256 * 32 * sizeof(uint32_t);
    record.status = grant;
    record.has_lds_state = true;
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, record));
    config::LoadedConfig restored;
    ASSERT_TRUE(test::restore_valid_checkpoint(file.path(), restored));
    const auto *wave = restored.soc()->xcd(0)->shader_engine(0)->compute_unit(0)->wf(0);
    ASSERT_NE(wave, nullptr);
    EXPECT_EQ(wave->status_raw() &
                  (Wavefront::kStatusExportReadyMask | Wavefront::kStatusParamReadyMask),
              grant);
  }
}

TEST(CheckpointTest, RetiredShaderExportRowsSurviveASecondExportAfterRestore) {
  for (const std::string_view target : {"rdna3", "rdna3_5", "rdna4"}) {
    SCOPED_TRACE(target);
    InvalidCheckpoint record;
    record.arch = target.data();
    record.wavefront = true;
    record.kernel_wave_size = record.vgpr_lane_count = 32;
    record.vgpr_bytes = 256 * 32 * sizeof(uint32_t);
    record.has_lds_state = true;
    const bool gfx12 = target == "rdna4";
    const auto first = gfx12 ? rdna4::build_vexport({.en = 1, .tgt = 0, .vsrc0 = 4})
                             : rdna3::build_exp({.en = 1, .tgt = 0, .vsrc0 = 4});
    const auto second = gfx12 ? rdna4::build_vexport({.en = 1, .tgt = 1, .done = 1, .vsrc0 = 5})
                              : rdna3::build_exp({.en = 1, .tgt = 1, .done = 1, .vsrc0 = 5});
    record.program.assign(first.begin(), first.end());
    record.program.push_back(gfx12 ? 0xbfc40000u : 0xbf89fff0u);
    record.program.insert(record.program.end(), second.begin(), second.end());
    record.program.push_back(0xbfb00000);
    using Vm = std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)>;
    const auto restore = [](const std::string &path) {
      rj_vm_t *result = nullptr;
      EXPECT_EQ(rj_vm_restore_checkpoint(path.c_str(), &result), ROCJITSU_STATUS_SUCCESS);
      return Vm(result, &rj_vm_destroy);
    };
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, record));
    auto original = restore(file.path());
    ASSERT_TRUE(original);
    auto *cu = original->soc->xcd(0)->shader_engine(0)->compute_unit(0);
    auto *wave = cu->wf(0);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(5);
    for (unsigned reg : {4u, 5u})
      for (unsigned lane : {0u, 2u})
        cu->write_vgpr(wave->vgpr_alloc().base + reg, lane, 0x11220000 + reg * 256 + lane);
    const auto next = wave->pc + 12;
    for (unsigned i = 0; i < 32 && wave->pc != next; ++i)
      ASSERT_TRUE(cu->step());
    ASSERT_EQ(wave->pc, next);
    ASSERT_EQ(wave->shader_exports().size(), 1u);
    test::ScopedTempFile saved;
    ASSERT_TRUE(saved.initialize("rocjitsu-export-rows-"));
    ASSERT_EQ(rj_vm_save_checkpoint(original.get(), saved.path().c_str(), 1),
              ROCJITSU_STATUS_SUCCESS);
    auto resumed = restore(saved.path());
    ASSERT_TRUE(resumed);
    auto *copy = resumed->soc->xcd(0)->shader_engine(0)->compute_unit(0);
    ASSERT_EQ(copy->wf(0)->shader_exports().size(), 1u);
    for (auto *device : {cu, copy}) {
      auto *live = device->wf(0);
      for (unsigned i = 0; i < 32 && live->shader_exports().size() < 2; ++i)
        ASSERT_TRUE(device->step());
      ASSERT_EQ(live->shader_exports().size(), 2u);
      for (unsigned target = 0; target < 2; ++target) {
        const auto &row = live->shader_exports()[target];
        EXPECT_EQ(row.target, target);
        EXPECT_EQ(row.row, 0u);
        EXPECT_EQ(row.components, 1u);
        EXPECT_EQ(row.done, bool(target));
        EXPECT_EQ(row.valid_lanes, 5u);
        for (unsigned lane : {0u, 2u})
          EXPECT_EQ(row.values[lane][0], 0x11220000u + (4 + target) * 256 + lane);
      }
    }
  }
}

TEST(CheckpointTest, GraphicsGrantMigrationAndHeldFrontendSurvivePublicRestore) {
  using amdgpu::Wavefront;
  using amdgpu::WfState;
  enum class Op { Export, Parameters, WaitEvent };
  enum class Format { Legacy, ResourceOnly, ExplicitLaunch, CursorOnly, Marked };
  constexpr uint32_t nop = 0xbf800000, endpgm = 0xbfb00000, wait_event = 0xbf8b0000;
  constexpr uint32_t unrelated = 1u << 3; // Preserve the low user-priority bit.
  const auto gfx11_export = rdna3::build_exp({.en = 1, .tgt = 0, .done = 1});
  const auto gfx12_export = rdna4::build_vexport({.en = 1, .tgt = 0, .done = 1});
  const auto parameter = rdna3::build_ldsdir(rdna3::kLdsParamLoadLdsdir, {.vdst = 2});
  for (const std::string_view arch : {"rdna3", "rdna3_5", "rdna4"}) {
    const bool gfx12 = arch == "rdna4";
    for (const auto op : {Op::Export, Op::Parameters, Op::WaitEvent}) {
      if ((gfx12 && op == Op::Parameters) || (op == Op::WaitEvent && arch != "rdna3_5"))
        continue;
      const auto mask = op == Op::Parameters ? Wavefront::kStatusParamReadyMask
                                             : Wavefront::kStatusExportReadyMask;
      const auto parked = op == Op::Parameters ? WfState::PARAM_WAIT : WfState::EXPORT_WAIT;
      const auto words = gfx12 ? gfx12_export : gfx11_export;
      for (const auto format : {Format::Legacy, Format::ResourceOnly, Format::ExplicitLaunch,
                                Format::CursorOnly, Format::Marked}) {
        SCOPED_TRACE(testing::Message() << arch << " opcode=" << static_cast<unsigned>(op)
                                        << " checkpoint format=" << static_cast<unsigned>(format));
        InvalidCheckpoint record;
        record.arch = arch.data();
        record.wavefront = true;
        record.kernel_wave_size = record.vgpr_lane_count = 32;
        record.vgpr_bytes = 256 * 32 * sizeof(uint32_t);
        record.status = unrelated;
        record.has_lds_state = true;
        record.has_cp_cursor = format != Format::Legacy;
        record.has_lds_cursor = format == Format::ResourceOnly || format == Format::Marked;
        if (format == Format::ExplicitLaunch)
          record.graphics_wait = fb::GraphicsWait_LaunchDefaults;
        else if (format == Format::Marked)
          record.graphics_wait = fb::GraphicsWait_None;

        record.program = {nop};
        if (op == Op::Parameters)
          record.program.insert(record.program.end(), parameter.begin(), parameter.end());
        else if (op == Op::WaitEvent)
          record.program.push_back(wait_event);
        else
          record.program.insert(record.program.end(), words.begin(), words.end());
        record.program.push_back(endpgm);
        test::ScopedTempFile legacy;
        ASSERT_TRUE(write_checkpoint(legacy, record));
        using Vm = std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)>;
        const auto restore = [](const std::string &path) {
          rj_vm_t *result = nullptr;
          EXPECT_EQ(rj_vm_restore_checkpoint(path.c_str(), &result), ROCJITSU_STATUS_SUCCESS);
          return Vm(result, &rj_vm_destroy);
        };
        const auto wave = [](const Vm &vm) {
          return vm->soc->xcd(0)->shader_engine(0)->compute_unit(0)->wf(0);
        };
        const auto advance = [](const Vm &vm, Wavefront &wf, WfState until) {
          for (unsigned step = 0; step < 64 && wf.state() != until; ++step) {
            int active = 0;
            ASSERT_EQ(rj_vm_step(vm.get(), &active), ROCJITSU_STATUS_SUCCESS);
          }
          EXPECT_EQ(wf.state(), until);
        };
        auto original = restore(legacy.path());
        ASSERT_TRUE(original);
        auto *first = wave(original);
        ASSERT_NE(first, nullptr);
        const bool legacy_ready = format == Format::Legacy || format == Format::ResourceOnly ||
                                  format == Format::ExplicitLaunch;
        const uint32_t standalone =
            Wavefront::kStatusExportReadyMask | (gfx12 ? 0u : Wavefront::kStatusParamReadyMask);
        EXPECT_EQ(first->status_raw(), unrelated | (legacy_ready ? standalone : 0u));
        ASSERT_NO_FATAL_FAILURE(advance(original, *first, legacy_ready ? WfState::HALTED : parked));

        // A current writer must preserve both an ungranted runnable wave and
        // a parked wave. S_WAIT_EVENT has already advanced the saved PC.
        for (bool withhold : {false, true}) {
          auto source = restore(legacy.path());
          ASSERT_TRUE(source);
          auto *src = wave(source);
          ASSERT_NE(src, nullptr);
          src->set_status_raw(withhold ? src->status_raw() & ~mask : src->status_raw() | mask);
          const auto save_and_restore = [&](const Vm &vm) {
            test::ScopedTempFile file;
            const auto initialized = file.initialize("rocjitsu-graphics-checkpoint-");
            EXPECT_TRUE(initialized);
            if (!initialized)
              return Vm(nullptr, &rj_vm_destroy);
            EXPECT_EQ(rj_vm_save_checkpoint(vm.get(), file.path().c_str(), 0),
                      ROCJITSU_STATUS_SUCCESS);
            return restore(file.path());
          };
          auto copy = save_and_restore(source);
          ASSERT_TRUE(copy);
          auto *live = wave(copy);
          ASSERT_NE(live, nullptr);
          EXPECT_EQ(live->status_raw(), src->status_raw());
          ASSERT_NO_FATAL_FAILURE(advance(copy, *live, withhold ? parked : WfState::HALTED));
          if (!withhold)
            continue;
          const auto parked_pc = live->pc;
          auto resumed = save_and_restore(copy);
          ASSERT_TRUE(resumed);
          auto *held = wave(resumed);
          ASSERT_NE(held, nullptr);
          EXPECT_EQ(held->state(), parked);
          int active = 0;
          ASSERT_EQ(rj_vm_step(resumed.get(), &active), ROCJITSU_STATUS_SUCCESS);
          EXPECT_EQ(held->pc, parked_pc);
          held->set_status_raw(held->status_raw() | mask);
          ASSERT_NO_FATAL_FAILURE(advance(resumed, *held, WfState::HALTED));
        }
      }
    }
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

TEST(CheckpointRestoreErrorsTest, Rdna2RestoresSharedResourcesAndRegisterBlockPlacement) {
  InvalidCheckpoint input{.arch = "rdna2",
                          .num_wf_slots = 2,
                          .wavefront = true,
                          .vgpr_lane_count = 32,
                          .vgpr_bytes = 256 * 32 * sizeof(uint32_t),
                          .rdna2_process_id = 17,
                          .rdna2_internal_address_space = true,
                          .gfx10_resources = Gfx10WaveResources{16, 256, 0x34000, 0x20},
                          .sgprs = std::vector<uint32_t>(104),
                          .kernel_wave_size = 32,
                          .num_sgprs = 96};
  for (bool explicit_block : {false, true}) {
    if (explicit_block)
      input.register_blocks = std::vector<uint32_t>{1};
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
  input.register_blocks->clear();
  {
    test::ScopedTempFile invalid_file;
    ASSERT_TRUE(write_checkpoint(invalid_file, input));
    expect_restore_failure(invalid_file.path(), "Invalid RDNA2 checkpoint register block count");
  }
  input.register_blocks->push_back(2);
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
TEST(CheckpointRestoreErrorsTest, LegacyCheckpointCanRestoreWithoutAnActiveRdna2Wave) {
  for (auto input :
       {InvalidCheckpoint{.arch = "rdna1"}, InvalidCheckpoint{.arch = "rdna2"},
        InvalidCheckpoint{.arch = "rdna1", .wavefront = true, .halted = true},
        InvalidCheckpoint{.arch = "rdna2", .wavefront = true, .halted = true},
        InvalidCheckpoint{
            .arch = "cdna3", .wavefront = true, .vgpr_bytes = 512 * 64 * sizeof(uint32_t)}}) {
    test::ScopedTempFile file;
    ASSERT_TRUE(write_checkpoint(file, input));
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(config::restore_checkpoint(file.path(), diagnostic.emitter()).succeeded())
        << input.arch << ": " << diagnostic.message();
  }
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

TEST(CheckpointRestoreErrorsTest, RejectsActiveRdna1WaveWithoutItsResourceRecord) {
  test::ScopedTempFile file;
  ASSERT_TRUE(write_checkpoint(file, {.arch = "rdna1", .wavefront = true}));
  expect_restore_failure(
      file.path(),
      "Cannot restore active RDNA1 wave: checkpoint omitted its VGPR and LDS allocations");
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
