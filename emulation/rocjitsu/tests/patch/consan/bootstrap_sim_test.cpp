// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"
#include "config_test_util.h"
#include "halt_snapshot_plugin.h"
#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/input.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

constexpr uint32_t kCompletedRowStatus = wire::kRowEntered | wire::kRowExited;

constexpr uint64_t kImageBase = 0x1000000;
constexpr uint64_t kOriginalArguments = 0x40000;
constexpr uint64_t kWrapper = 0x50000;
constexpr uint64_t kOutput = 0x80000;
constexpr uint64_t kOpaqueContext = 0x90000;
constexpr uint64_t kReport = 0xa0000;

constexpr std::string_view kTopology = R"({
  "max_ticks":100000,"num_threads":1,"vm":{"arch":"rdna3"},
  "topology":{"root":{"name":"soc","type":"soc","children":[
    {"name":"vram","type":"gpu_memory"},
    {"name":"xcd0","type":"xcd","children":[
      {"name":"l2","type":"l2_cache"},
      {"name":"cp","type":"command_processor"},
      {"name":"se0","type":"shader_engine","children":[
        {"name":"cu[0:2]","type":"compute_unit","config":[
          {"key":"num_wf_slots","value":"10"},
          {"key":"sgprs_per_wf","value":"800"},
          {"key":"vgprs_per_wf","value":"256"},
          {"key":"lds_size_kb","value":"64"}]}]}]}]},
    "links":[
      {"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
      {"src":"xcd0.cp.req_1","dst":"xcd0.se0.cu1.cpl","latency":1,"weight":2},
      {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10},
      {"src":"xcd0.se0.cu1.req","dst":"xcd0.l2.cpl_1","latency":1,"weight":10}]}})";

enum class Bootstrap { None, Enabled, OmitPointerRestore, OmitPayloadLoad, Instrumented };

struct RunResult {
  std::array<uint32_t, 2> output{};
  std::vector<::rocjitsu::test::WavefrontSnapshot> waves;
  uint16_t context_sgpr = 0;
  Manifest manifest;
  std::vector<uint8_t> report;
  std::vector<uint32_t> full_output;
  uint32_t original_sgprs = 0, original_vgprs = 0;
};

class ConSanBootstrapSim : public ::testing::Test {
protected:
  void SetUp() override {
#ifndef CONSAN_FIXTURE_DIR
    GTEST_SKIP() << "gfx1100 assembler/linker fixtures unavailable";
#endif
  }

  void run(std::string_view fixture, Bootstrap mode, RunResult &result,
           test::TargetMode target_mode = {"gfx1100", 32}, uint32_t threads = 64,
           bool target_fixture = false) {
#ifdef CONSAN_FIXTURE_DIR
    const auto original =
        test::read_fixture(target_fixture ? test::target_fixture_path(target_mode, fixture)
                                          : std::filesystem::path(CONSAN_FIXTURE_DIR) /
                                                (std::string(fixture) + ".hsaco"));
    ASSERT_FALSE(original.empty());
    const auto *profile = target_profile(target_mode.target);
    ASSERT_NE(profile, nullptr);
    const auto arch = profile->arch;
    const uint32_t wave_count = (threads + target_mode.wave_size - 1) / target_mode.wave_size;
    InputError error;
    const FailureOr<InputProgram> analyzed = build_input_program(
        original, *target_profile(target_mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(analyzed.succeeded()) << error.message;
    const InputProgram &input = analyzed.value();
    result.original_sgprs = input.original_sgprs;
    result.original_vgprs = input.original_vgprs;
    result.context_sgpr = input.entry_plan.storage.persistent_base;
    std::vector<uint8_t> image = original;
    if (mode == Bootstrap::Instrumented) {
      const auto artifact =
          prepare(original, *target_profile(target_mode.target), "lds_producer_consumer", error);
      ASSERT_TRUE(artifact.succeeded()) << error.message;
      image = artifact.value().image;
      result.manifest = artifact.value().manifest;
    } else if (mode != Bootstrap::None) {
      AmdGpuCodeObject object(image.data(), image.size());
      CodeObjectPatcher patcher(object);
      std::vector<uint8_t> text(patcher.text_bytes().begin(), patcher.text_bytes().end());
      append_nop_padding(
          text,
          padding_for_residue(text.size(), input.kernel.entry_text_offset, kKernelEntryAlignment),
          arch);
      const uint64_t bootstrap_offset = text.size();
      std::vector<uint32_t> words = input.entry_plan.prologue.words;
      if (mode == Bootstrap::OmitPayloadLoad) {
        words[0] = build_s_nop(0, arch);
        words[1] = build_s_nop(0, arch);
      }
      if (mode == Bootstrap::OmitPointerRestore) {
        words[words.size() - 2] = build_s_nop(0, arch);
        words[words.size() - 1] = build_s_nop(0, arch);
      }
      append_words(text, words);
      const std::optional<int16_t> branch =
          compute_sopp_branch_simm16(text.size(), input.kernel.entry_text_offset);
      ASSERT_TRUE(branch.has_value());
      const std::array<uint32_t, 1> jump = {build_s_branch(*branch, arch)};
      append_words(text, jump);
      ASSERT_TRUE(patcher.replace_text(text));
      image = std::move(patcher).emit();

      AmdGpuCodeObject moved(image.data(), image.size());
      ASSERT_TRUE(moved.is_valid());
      const Section &section = *moved.text_sections().front();
      const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
          image, section.sectionOffset(), section.size(), section.sectionHeaderIndex());
      ASSERT_EQ(kernels.size(), 1u);
      CodeObjectPatcher redirect(moved);
      ASSERT_TRUE(redirect.redirect_kernel_entry(kernels[0].descriptor_file_offset,
                                                 input.kernel.entry_text_offset, bootstrap_offset));
      image = std::move(redirect).emit();
    }

    std::string topology(kTopology);
    const std::string vm = "\"arch\":\"" + std::string(config::arch_to_string(arch)) + "\"" +
                           (profile->public_id == ROCJITSU_CODE_TARGET_INVALID
                                ? ""
                                : ",\"target\":\"" + std::string(target_mode.target) + "\"");
    topology.replace(topology.find("\"arch\":\"rdna3\""), std::string("\"arch\":\"rdna3\"").size(),
                     vm);
    config::LoadedConfig loaded;
    ASSERT_TRUE(::rocjitsu::test::load_valid_config_from_string(topology, loaded));
    SoC *soc = loaded.soc();
    amdgpu::GpuMemory *memory = loaded.memory();
    simdojo::SimulationEngine engine(loaded.engine_config);
    engine.topology().set_root(loaded.take_root());
    loaded.wire_links(engine.topology());
    engine.create();
    ::rocjitsu::test::HaltSnapshotPlugin *snapshots = nullptr;
    const std::shared_ptr<ExecutionPluginGroup> plugins =
        ::rocjitsu::test::make_halt_snapshot_group(&snapshots);
    soc->set_plugin_group(plugins);

    // Load the actual selected descriptor, including its wave mode and
    // register allocation, rather than substituting a synthetic dispatch ABI.
    AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    object.load_to_memory(memory, kImageBase);
    const uint64_t descriptor = object.kernel_descriptor_offset("lds_producer_consumer");
    ASSERT_NE(descriptor, 0u);
    std::array<uint8_t, 16> arguments{};
    const uint32_t input_value = 7;
    std::memcpy(arguments.data(), &input_value, sizeof(input_value));
    std::memcpy(arguments.data() + 8, &kOutput, sizeof(kOutput));
    memory->load_image(arguments.data(), arguments.size(), kOriginalArguments);
    if (mode == Bootstrap::Instrumented) {
      const auto transport =
          initialize_transport(result.manifest,
                               {{threads, 1, 1},
                                { threads,
                                  1,
                                  1 }},
                               result.manifest.entry.original_group_bytes, 0, 41, kReport);
      ASSERT_TRUE(transport.succeeded());
      memory->load_image(transport.value().context.data(), transport.value().context.size(),
                         kOpaqueContext);
      result.report = transport.value().report;
      memory->load_image(result.report.data(), result.report.size(), kReport);
    }
    uint64_t dispatched_arguments = kOriginalArguments;
    if (mode != Bootstrap::None) {
      constexpr std::array<KernargExtensionPayloadLayout, 1> payloads = {{
        {.size = 8,
         .alignment = 8 }
      }};
      const std::optional<KernargExtensionLayout> layout =
          make_kernarg_extension_layout(arguments.size(), payloads);
      ASSERT_TRUE(layout.has_value());
      const std::array<KernargExtensionPayloadWrite, 1> writes = {{
        {.data = &kOpaqueContext,
         .size = sizeof(kOpaqueContext) }
      }};
      std::vector<uint8_t> wrapper(layout->wrapper_size);
      ASSERT_TRUE(write_kernarg_extension_wrapper(wrapper, *layout, arguments.data(),
                                                  kOriginalArguments, writes));
      // A safe binding control makes restoration observable: the original
      // pointer still names input=7 while the copied prefix now says input=13.
      if (mode != Bootstrap::Instrumented) {
        const uint32_t poison = 13;
        std::memcpy(wrapper.data(), &poison, sizeof(poison));
      }
      memory->load_image(wrapper.data(), wrapper.size(), kWrapper);
      dispatched_arguments = kWrapper;
    }
    ::rocjitsu::test::AqlQueue queue(memory, soc->xcd(0)->command_processor());
    queue.dispatch(kImageBase + descriptor, threads, static_cast<uint16_t>(threads),
                   dispatched_arguments);
    engine.run();
    amdgpu::CommandProcessor *cp = soc->xcd(0)->command_processor();
    ASSERT_FALSE(cp->queue_faulted_for_test(1, 0));
    ASSERT_EQ(cp->dispatched_count(), 1u);
    ASSERT_EQ(cp->dispatched_workgroups(), 1u);
    ASSERT_EQ(snapshots->snapshots().size(), wave_count);
    result.waves = snapshots->snapshots();
    std::ranges::sort(result.waves, {}, &::rocjitsu::test::WavefrontSnapshot::wf_id);
    soc->flush_all();
    if (mode == Bootstrap::Instrumented)
      memory->read_block(kReport, result.report);
    std::vector<uint8_t> output(std::max(wave_count, 2u) * sizeof(uint32_t));
    memory->read_block(kOutput, output);
    std::memcpy(result.output.data(), output.data(), sizeof(result.output));
    result.full_output.resize(wave_count);
    std::memcpy(result.full_output.data(), output.data(), wave_count * sizeof(uint32_t));
#else
    (void)fixture;
    (void)mode;
    (void)result;
    (void)target_mode;
    (void)threads;
    (void)target_fixture;
#endif
  }
};

TEST_F(ConSanBootstrapSim, ActualWave32DescriptorRunsBothDefinedDisjointWaves) {
  RunResult original;
  run("disjoint", Bootstrap::None, original);
  ASSERT_EQ(original.waves.size(), 2u);
  EXPECT_EQ(original.output, (std::array<uint32_t, 2>{7, 8}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : original.waves) {
    EXPECT_EQ(wave.wf_size, 32u);
    EXPECT_EQ(wave.num_vgprs, 8u);
    EXPECT_EQ(wave.exec, 0xffffffffu);
  }
}

TEST_F(ConSanBootstrapSim, ContextDeliveryAndOriginalArgumentViewSurviveInBothWaves) {
  RunResult original;
  RunResult wrapped;
  run("disjoint", Bootstrap::None, original);
  run("disjoint", Bootstrap::Enabled, wrapped);
  ASSERT_EQ(original.waves.size(), 2u);
  ASSERT_EQ(wrapped.waves.size(), 2u);
  EXPECT_EQ(wrapped.output, original.output);
  EXPECT_EQ(wrapped.output, (std::array<uint32_t, 2>{7, 8}));
  for (size_t wave_index = 0; wave_index < 2; ++wave_index) {
    const ::rocjitsu::test::WavefrontSnapshot &before = original.waves[wave_index];
    const ::rocjitsu::test::WavefrontSnapshot &after = wrapped.waves[wave_index];
    EXPECT_EQ(after.wf_size, 32u);
    EXPECT_EQ(after.sgpr64(wrapped.context_sgpr), kOpaqueContext);
    EXPECT_EQ(after.sgpr64(0), kOriginalArguments);
    EXPECT_EQ(after.exec, before.exec);
    EXPECT_EQ(after.vcc, before.vcc);
    EXPECT_EQ(after.status & 1, before.status & 1);
    EXPECT_EQ(after.mode_raw, before.mode_raw);
    for (uint32_t reg = 0; reg < 8; ++reg) {
      for (uint32_t lane = 0; lane < 32; ++lane)
        EXPECT_EQ(after.vgpr(reg, lane), before.vgpr(reg, lane));
    }
  }
}

TEST_F(ConSanBootstrapSim, RemovingPointerRestoreChangesTheIndependentNumericOracle) {
  RunResult control;
  run("disjoint", Bootstrap::OmitPointerRestore, control);
  ASSERT_EQ(control.waves.size(), 2u);
  EXPECT_EQ(control.output, (std::array<uint32_t, 2>{13, 14}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : control.waves)
    EXPECT_EQ(wave.sgpr64(0), kWrapper);
}

TEST_F(ConSanBootstrapSim, RemovingPayloadLoadStopsContextDelivery) {
  RunResult control;
  run("disjoint", Bootstrap::OmitPayloadLoad, control);
  ASSERT_EQ(control.waves.size(), 2u);
  EXPECT_EQ(control.output, (std::array<uint32_t, 2>{7, 8}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : control.waves)
    EXPECT_NE(wave.sgpr64(control.context_sgpr), kOpaqueContext);
}

TEST_F(ConSanBootstrapSim,
       ProductionProbesPreserveDefinedOutputsAndGuestStateIncludingAddressAlias) {
  for (const std::string_view fixture : {"disjoint", "alias"}) {
    RunResult original;
    RunResult instrumented;
    run(fixture, Bootstrap::None, original);
    run(fixture, Bootstrap::Instrumented, instrumented);
    ASSERT_EQ(original.waves.size(), 2u);
    ASSERT_EQ(instrumented.waves.size(), 2u);
    EXPECT_EQ(instrumented.output, (std::array<uint32_t, 2>{7, 8}));
    EXPECT_EQ(instrumented.output, original.output);
    for (size_t wave = 0; wave < 2; ++wave) {
      const auto &before = original.waves[wave];
      const auto &after = instrumented.waves[wave];
      EXPECT_EQ(after.exec, before.exec);
      EXPECT_EQ(after.vcc, before.vcc);
      EXPECT_EQ(after.status & 1, before.status & 1);
      EXPECT_EQ(after.mode_raw, before.mode_raw);
      EXPECT_EQ(after.sgpr64(0), kOriginalArguments);
      for (uint32_t reg = 0; reg < 16; ++reg)
        EXPECT_EQ(after.sgprs[reg], before.sgprs[reg]);
      for (uint32_t reg = 0; reg < 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane)
          EXPECT_EQ(after.vgpr(reg, lane), before.vgpr(reg, lane));
      }
      const size_t row = kReportHeaderBytes + wave * (kRowHeaderBytes + 8 * kAccessRecordBytes);
      EXPECT_EQ(wire::read_u32(instrumented.report, row + wire::kRowStatus),
                wire::kRowEntered | wire::kRowExited);
      EXPECT_EQ(wire::read_u32(instrumented.report, row + wire::kRowLoss), 0u);
      EXPECT_EQ(wire::read_u32(instrumented.report, row + wire::kRowFinalSequence), 2u);
      for (uint32_t site = 0; site < 2; ++site) {
        const size_t record = row + kRowHeaderBytes + site * 4 * kAccessRecordBytes;
        EXPECT_EQ(wire::read_u32(instrumented.report, record + wire::kRecordCommit),
                  wire::kAccessCommitted);
        EXPECT_EQ(wire::read_u32(instrumented.report, record + wire::kRecordGuestMask), 1u);
        EXPECT_EQ(wire::read_u32(instrumented.report, record + wire::kRecordLdsOffset), wave * 4);
      }
    }
  }
}

TEST_F(ConSanBootstrapSim, ProductionRaceReadReadAndCollisionMasksComeFromGuestParticipants) {
  for (const std::string_view fixture : {"race", "read_read", "collision", "single_lane"}) {
    RunResult result;
    run(fixture, Bootstrap::Instrumented, result);
    ASSERT_EQ(result.waves.size(), 2u);
    const uint32_t sites = static_cast<uint32_t>(result.manifest.sites.size());
    const uint32_t stride = kRowHeaderBytes + sites * 4 * kAccessRecordBytes;
    uint32_t records = 0;
    for (uint32_t wave = 0; wave < 2; ++wave) {
      const size_t row = kReportHeaderBytes + wave * stride;
      EXPECT_EQ(wire::read_u64(result.report, row + wire::kRowSerial), 41u);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWave), wave);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowStatus), kCompletedRowStatus);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowLoss), 0u);
      for (uint32_t site = 0; site < sites; ++site) {
        for (uint32_t bank = 0; bank < 4; ++bank) {
          const size_t record = row + kRowHeaderBytes + (site * 4 + bank) * kAccessRecordBytes;
          if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0u)
            continue;
          ++records;
          EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordCommit),
                    wire::kAccessCommitted);
          EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordSite), site);
          EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordLdsOffset), 0u);
          EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordGuestMask),
                    fixture == "collision" ? 3u : 1u);
          EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordFlags),
                    wire::kExactGuestMask);
          EXPECT_EQ(wave, sites == 2 ? site : 0u);
        }
      }
    }
    EXPECT_EQ(records, sites);
  }
}

TEST_F(ConSanBootstrapSim,
       ProductionReservationsGroupAddressesAndPublishExhaustionWithoutInventingLanes) {
  for (const std::string_view fixture :
       {"four_ranges", "exhausted", "empty", "branched", "immediate"}) {
    RunResult result;
    run(fixture, Bootstrap::Instrumented, result);
    ASSERT_EQ(result.waves.size(), 2u);
    const uint32_t stride = kRowHeaderBytes + static_cast<uint32_t>(result.manifest.sites.size()) *
                                                  4 * kAccessRecordBytes;
    uint32_t count = 0;
    for (uint32_t wave = 0; wave < 2; ++wave) {
      const size_t row = kReportHeaderBytes + wave * stride;
      const bool lost = fixture == "exhausted" && wave == 0;
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowStatus),
                lost ? (kCompletedRowStatus | wire::kRowExhausted) : kCompletedRowStatus);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowLoss), lost ? 1u : 0u);
      for (const SiteFacts &site : result.manifest.sites) {
        for (uint32_t bank = 0; bank < 4; ++bank) {
          const size_t record =
              row + kRowHeaderBytes + (site.first_slot + bank) * kAccessRecordBytes;
          if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0)
            continue;
          ++count;
          if (fixture == "four_ranges" || fixture == "exhausted") {
            EXPECT_EQ(wave, 0u);
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordGuestMask), 1u << bank);
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordLdsOffset), bank * 4);
          } else {
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordGuestMask), 1u);
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordLdsOffset),
                      fixture == "immediate" && wave == 0 ? 4u : 0u);
          }
          if (fixture == "branched") {
            EXPECT_EQ(site.id, 1u);
          }
        }
      }
    }
    EXPECT_EQ(count, fixture == "empty"                                     ? 0u
                     : (fixture == "four_ranges" || fixture == "exhausted") ? 4u
                     : fixture == "branched"                                ? 1u
                                                                            : 2u);
  }
}

class ConSanTargetSim : public ConSanBootstrapSim,
                        public ::testing::WithParamInterface<test::TargetMode> {};

void expect_guest_state(const RunResult &before, const RunResult &after) {
  ASSERT_EQ(before.waves.size(), after.waves.size());
  EXPECT_EQ(before.full_output, after.full_output);
  for (size_t index = 0; index < before.waves.size(); ++index) {
    const auto &a = before.waves[index];
    const auto &b = after.waves[index];
    SCOPED_TRACE(index);
    EXPECT_EQ(a.wf_size, b.wf_size);
    EXPECT_EQ(a.exec, b.exec);
    EXPECT_EQ(a.vcc, b.vcc);
    EXPECT_EQ(a.status & 1, b.status & 1);
    EXPECT_EQ(a.mode_raw, b.mode_raw);
    EXPECT_EQ(a.vgpr_msb_mode, b.vgpr_msb_mode);
    for (uint32_t reg = 0; reg < before.original_sgprs; ++reg)
      EXPECT_EQ(a.sgpr(reg), b.sgpr(reg)) << "SGPR " << reg;
    for (uint32_t reg = 0; reg < before.original_vgprs; ++reg)
      for (uint32_t lane = 0; lane < a.wf_size; ++lane)
        EXPECT_EQ(a.vgpr(reg, lane), b.vgpr(reg, lane)) << "VGPR " << reg << " lane " << lane;
  }
}

struct ObservedAccess {
  uint32_t wave, site, offset;
  uint64_t mask;
};

std::vector<ObservedAccess> published_accesses(const RunResult &result, bool exhausted = false) {
  std::vector<ObservedAccess> accesses;
  const uint32_t stride =
      kRowHeaderBytes + result.manifest.sites.size() * kBanksPerSite * kAccessRecordBytes;
  for (uint32_t wave = 0; wave < result.waves.size(); ++wave) {
    const size_t row = kReportHeaderBytes + wave * stride;
    EXPECT_EQ(wire::read_u64(result.report, row + wire::kRowSerial), 41u);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWave), wave);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowStatus),
              exhausted && wave == 0 ? (kCompletedRowStatus | wire::kRowExhausted)
                                     : kCompletedRowStatus);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowLoss),
              exhausted && wave == 0 ? 1u : 0u);
    for (const auto &site : result.manifest.sites) {
      for (uint32_t bank = 0; bank < kBanksPerSite; ++bank) {
        const size_t record = row + kRowHeaderBytes + (site.first_slot + bank) * kAccessRecordBytes;
        if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0)
          continue;
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordCommit),
                  wire::kAccessCommitted);
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordSite), site.id);
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordFlags),
                  wire::kExactGuestMask);
        accesses.push_back({wave, site.id,
                            *wire::read_u32(result.report, record + wire::kRecordLdsOffset),
                            *wire::read_u64(result.report, record + wire::kRecordGuestMask)});
      }
    }
  }
  return accesses;
}

TEST_P(ConSanTargetSim, AssemblyAliasPreservesAllGuestRegistersAndPublishesBothWaves) {
  const auto target = GetParam();
  for (const auto *fixture : {"native-2", "native-6"}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    run(fixture, Bootstrap::None, before, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, after, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8}));
    const auto records = published_accesses(after);
    ASSERT_EQ(records.size(), 4u);
    for (const auto &record : records) {
      EXPECT_EQ(record.mask, 1u);
      EXPECT_EQ(record.offset, record.wave * 4);
    }
  }
}

TEST_P(ConSanTargetSim, HighLaneMasksAndEmptyExecHaveRealPublicationEvidence) {
  const auto target = GetParam();
  for (const auto *fixture : {"native-14", "native-12", "native-11", "native-4", "native-13"}) {
    SCOPED_TRACE(fixture);
    RunResult after;
    run(fixture, Bootstrap::Instrumented, after, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    const std::string_view name(fixture);
    if (name == "native-14") {
      RunResult before;
      run(fixture, Bootstrap::None, before, target, 2 * target.wave_size, true);
      ASSERT_FALSE(HasFatalFailure());
      const uint64_t high_lane = uint64_t{1} << (target.wave_size - 1);
      EXPECT_EQ(before.waves[1].vcc, high_lane);
      for (size_t wave = 0; wave < before.waves.size(); ++wave) {
        EXPECT_EQ(after.waves[wave].exec, before.waves[wave].exec);
        EXPECT_EQ(after.waves[wave].vcc, before.waves[wave].vcc);
        EXPECT_EQ(after.waves[wave].status & 1, before.waves[wave].status & 1);
        EXPECT_EQ(after.waves[wave].mode_raw, before.waves[wave].mode_raw);
        for (uint32_t reg = 0; reg < before.original_sgprs; ++reg)
          EXPECT_EQ(after.waves[wave].sgpr(reg), before.waves[wave].sgpr(reg));
        // v3 is the deliberately unsynchronized read result; do not assign it
        // a fixed numeric oracle. Every other guest VGPR must be preserved.
        for (uint32_t reg = 0; reg < before.original_vgprs; ++reg) {
          if (reg != 3) {
            for (uint32_t lane = 0; lane < target.wave_size; ++lane)
              EXPECT_EQ(after.waves[wave].vgpr(reg, lane), before.waves[wave].vgpr(reg, lane));
          }
        }
      }
    }
    const auto records = published_accesses(after, name == "native-11");
    ASSERT_EQ(records.size(), name == "native-14"   ? 2u
                              : name == "native-12" ? 0u
                              : name == "native-11" ? 4u
                                                    : 1u);
    for (size_t index = 0; index < records.size(); ++index) {
      const auto &record = records[index];
      if (name == "native-14") {
        EXPECT_EQ(record.wave, index);
        EXPECT_EQ(record.mask, uint64_t{1} << (target.wave_size - 1));
      } else if (name == "native-11") {
        EXPECT_EQ(record.wave, 0u);
        EXPECT_EQ(record.mask, uint64_t{1} << index);
        EXPECT_EQ(record.offset, index * 4);
      } else {
        EXPECT_EQ(record.mask, name == "native-13" ? 1u : 3u);
      }
    }
  }
}

TEST_P(ConSanTargetSim, ThreeWavesAndPartialTailUseDistinctRows) {
  const auto target = GetParam();
  RunResult before, after;
  const uint32_t threads = 3 * target.wave_size + 1;
  run("native-2", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-2", Bootstrap::Instrumented, after, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8, 9, 10}));
  const auto records = published_accesses(after);
  ASSERT_EQ(records.size(), 8u);
  for (const auto &record : records) {
    EXPECT_EQ(record.mask, 1u);
    EXPECT_EQ(record.offset, record.wave * 4);
  }
  EXPECT_EQ(after.waves.back().exec, 1u);
}

// gfx1250 HIP currently emits global prefetch and WAVE_MODE setup; static
// admission tests require a precise rejection. Its native probes run above.
class ConSanHipSim : public ConSanTargetSim {};
TEST_P(ConSanHipSim, CompiledHipRaceDisjointAndReadReadExecuteTheirNativeSites) {
  const auto target = GetParam();
  for (const auto *fixture : {"hip-1", "hip-2", "hip-3"}) {
    SCOPED_TRACE(fixture);
    RunResult after;
    run(fixture, Bootstrap::Instrumented, after, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    const auto records = published_accesses(after);
    ASSERT_EQ(records.size(), std::string_view(fixture) == "hip-2" ? 4u : 2u);
    uint32_t reads = 0, writes = 0;
    for (const auto &record : records) {
      EXPECT_EQ(record.mask, 1u);
      (after.manifest.sites[record.site].access == AccessKind::Read ? reads : writes)++;
    }
    EXPECT_EQ(writes, std::string_view(fixture) == "hip-3"   ? 0u
                      : std::string_view(fixture) == "hip-1" ? 1u
                                                             : 2u);
    EXPECT_GT(reads, 0u);
    if (std::string_view(fixture) == "hip-2") {
      RunResult before;
      run(fixture, Bootstrap::None, before, target, 2 * target.wave_size, true);
      ASSERT_FALSE(HasFatalFailure());
      expect_guest_state(before, after);
      EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8}));
    }
  }
}

const auto target_name = [](const auto &info) {
  return std::string(info.param.target) + "Wave" + std::to_string(info.param.wave_size);
};
INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanTargetSim, ::testing::ValuesIn(test::kTargetModes),
                         target_name);
INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanHipSim,
                         ::testing::ValuesIn(std::span(test::kTargetModes).first(6)), target_name);

} // namespace
} // namespace rocjitsu::consan
