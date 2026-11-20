// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"
#include "config_test_util.h"
#include "halt_snapshot_plugin.h"
#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#ifdef CONSAN_ANALYZER_ENABLED
#include "rocjitsu/code/consan/analysis/lds_race.h"
#endif
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
#include <tuple>

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

enum class Bootstrap {
  None,
  Enabled,
  OmitPointerRestore,
  OmitPayloadLoad,
  Instrumented,
  InstrumentedWithoutPublicationWait
};

struct RunResult {
  std::array<uint32_t, 2> output{};
  std::vector<::rocjitsu::test::WavefrontSnapshot> waves;
  uint16_t context_sgpr = 0;
  Manifest manifest;
  std::vector<uint8_t> report;
  std::vector<uint32_t> full_output;
  uint32_t original_sgprs = 0, original_vgprs = 0;
  uint32_t workgroups = 1, waves_per_workgroup = 0;
};

// Simulator-only near-limit control: seed report state after the entry callback
// and before any guest operation. This does not claim billions of iterations.
class SeedCountersAtGuestEntry : public ExecutionPlugin {
public:
  SeedCountersAtGuestEntry(uint64_t entry_pc, uint32_t stride, uint32_t waves)
      : ExecutionPlugin("seed_counters"), entry_pc_(entry_pc), stride_(stride),
        seeded_(waves, false) {}
  bool requires_serial_hot_hooks() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &,
                                        amdgpu::Wavefront &wave) override {
    if (pc != entry_pc_)
      return;
    const uint32_t index = wave.wave_in_group();
    ASSERT_LT(index, seeded_.size());
    if (seeded_[index])
      return;
    const uint64_t row = kReport + kReportHeaderBytes + uint64_t{index} * stride_;
    std::array<uint8_t, 4> word{};
    ASSERT_EQ(wave.read_gpu_memory(row + wire::kRowStatus, word),
              amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(wire::read_u32(word, 0), wire::kRowEntered);
    wire::write_u32(word, 0, UINT32_MAX - 1);
    for (const uint32_t offset :
         {wire::kRowFinalSequence, wire::kRowFinalPhase, wire::kRowLoss, wire::kRowSyncLoss})
      ASSERT_EQ(wave.write_gpu_memory(row + offset, word), amdgpu::VmAccessOutcome::Complete);
    seeded_[index] = true;
  }
  bool seeded_every_wave() const {
    return std::ranges::all_of(seeded_, [](bool value) { return value; });
  }

private:
  uint64_t entry_pc_;
  uint32_t stride_;
  std::vector<bool> seeded_;
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
           bool target_fixture = false, uint32_t output_words_per_wave = 1,
           bool seed_counters = false, uint32_t workgroups = 1) {
#ifdef CONSAN_FIXTURE_DIR
    const std::vector<uint8_t> original =
        test::read_fixture(target_fixture ? test::target_fixture_path(target_mode, fixture)
                                          : std::filesystem::path(CONSAN_FIXTURE_DIR) /
                                                (std::string(fixture) + ".hsaco"));
    ASSERT_FALSE(original.empty());
    const TargetProfile *profile = target_profile(target_mode.target);
    ASSERT_NE(profile, nullptr);
    const rj_code_arch_t arch = profile->arch;
    const uint32_t waves_per_workgroup =
        (threads + target_mode.wave_size - 1) / target_mode.wave_size;
    const uint32_t wave_count = workgroups * waves_per_workgroup;
    result.workgroups = workgroups;
    result.waves_per_workgroup = waves_per_workgroup;
    InputError error;
    const FailureOr<InputProgram> analyzed = build_input_program(
        original, *target_profile(target_mode.target), "lds_producer_consumer", error);
    ASSERT_TRUE(analyzed.succeeded()) << error.message;
    const InputProgram &input = analyzed.value();
    result.original_sgprs = input.original_sgprs;
    result.original_vgprs = input.original_vgprs;
    std::vector<uint8_t> image = original;
    const bool instrumented =
        mode == Bootstrap::Instrumented || mode == Bootstrap::InstrumentedWithoutPublicationWait;
    if (instrumented) {
      const FailureOr<PreparedArtifact> artifact =
          prepare(original, *target_profile(target_mode.target), "lds_producer_consumer", error);
      ASSERT_TRUE(artifact.succeeded()) << error.message;
      image = artifact.value().image;
      result.manifest = artifact.value().manifest;
      if (mode == Bootstrap::InstrumentedWithoutPublicationWait) {
        ASSERT_EQ(result.manifest.barriers.size(), 2u);
        AmdGpuCodeObject prepared(image.data(), image.size());
        ASSERT_TRUE(prepared.is_valid());
        ASSERT_EQ(prepared.text_sections().size(), 1u);
        const uint64_t wait = prepared.text_sections().front()->sectionOffset() +
                              result.manifest.barriers.back().prepared_wait;
        const FailureOr<uint32_t> nop = build_s_nop(0, arch);
        ASSERT_TRUE(nop.succeeded());
        ASSERT_TRUE(wire::read_u32(image, wait).has_value());
        ASSERT_NE(*wire::read_u32(image, wait), nop.value());
        // Fault injection changes only the relocated publication wait. Keep
        // the completion callback and all other instrumentation intact.
        wire::write_u32(image, wait, nop.value());
        result.manifest.prepared.sha256 = digest(image);
      }
    } else if (mode != Bootstrap::None) {
      const std::optional<DbiEntryProloguePlan> plan = test::bootstrap_plan(input);
      ASSERT_TRUE(plan);
      result.context_sgpr = plan->storage.persistent_base;
      AmdGpuCodeObject object(image.data(), image.size());
      CodeObjectPatcher patcher(object);
      std::vector<uint8_t> text(patcher.text_bytes().begin(), patcher.text_bytes().end());
      ASSERT_TRUE(
          append_nop_padding(text,
                             padding_for_residue(text.size(), input.kernel.entry_text_offset,
                                                 kKernelEntryAlignment),
                             arch)
              .succeeded());
      const uint64_t bootstrap_offset = text.size();
      std::vector<uint32_t> words = plan->prologue.words;
      const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
      ASSERT_TRUE(nop.succeeded());
      if (mode == Bootstrap::OmitPayloadLoad) {
        words[0] = nop.value();
        words[1] = nop.value();
      }
      if (mode == Bootstrap::OmitPointerRestore) {
        words[words.size() - 2] = nop.value();
        words[words.size() - 1] = nop.value();
      }
      append_words(text, words);
      const std::optional<int16_t> branch =
          compute_sopp_branch_simm16(text.size(), input.kernel.entry_text_offset);
      ASSERT_TRUE(branch.has_value());
      const util::FailureOr<uint32_t> branch_word = build_s_branch(*branch, arch);
      ASSERT_TRUE(branch_word.succeeded());
      const std::array<uint32_t, 1> jump = {branch_word.value()};
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
    const std::string vm = "\"arch\":\"" + std::string(config::arch_to_string(arch)) +
                           "\",\"target\":\"" + std::string(target_mode.target) + "\"";
    topology.replace(topology.find("\"arch\":\"rdna3\""), std::string("\"arch\":\"rdna3\"").size(),
                     vm);
    config::LoadedConfig loaded;
    ASSERT_TRUE(::rocjitsu::test::load_valid_config_from_string(topology, loaded));
    SoC *soc = loaded.soc();
    amdgpu::GpuMemory *memory = loaded.memory();
    simdojo::SimulationEngine engine(loaded.engine_config);
    engine.topology().set_root(loaded.take_root());
    util::StringDiagnostic admission_diagnostic;
    ASSERT_TRUE(loaded.wire_links(engine.topology(), admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();
    ASSERT_TRUE(engine.create(admission_diagnostic.emitter()).succeeded())
        << admission_diagnostic.message();
    ::rocjitsu::test::HaltSnapshotPlugin *snapshots = nullptr;
    const std::shared_ptr<ExecutionPluginGroup> plugins =
        ::rocjitsu::test::make_halt_snapshot_group(&snapshots);

    SeedCountersAtGuestEntry *counter_seed = nullptr;

    // Load the actual selected descriptor, including its wave mode and
    // register allocation, rather than substituting a synthetic dispatch ABI.
    AmdGpuCodeObject object(image.data(), image.size());
    ASSERT_TRUE(object.is_valid());
    if (seed_counters) {
      ASSERT_EQ(mode, Bootstrap::Instrumented);
      ASSERT_EQ(workgroups, 1u);
      auto seed = std::make_unique<SeedCountersAtGuestEntry>(
          kImageBase + object.text_sections().front()->vaddr() + input.kernel.entry_text_offset,
          static_cast<uint32_t>(report_row_bytes(result.manifest.sites.size(), true)), wave_count);
      counter_seed = seed.get();
      ASSERT_TRUE(plugins->add(std::move(seed)));
    }
    soc->set_plugin_group(plugins);
    object.load_to_memory(memory, kImageBase);
    const uint64_t descriptor = object.kernel_descriptor_offset("lds_producer_consumer");
    ASSERT_NE(descriptor, 0u);
    std::array<uint8_t, 16> arguments{};
    const uint32_t input_value = 7;
    std::memcpy(arguments.data(), &input_value, sizeof(input_value));
    std::memcpy(arguments.data() + 4, &waves_per_workgroup, sizeof(waves_per_workgroup));
    std::memcpy(arguments.data() + 8, &kOutput, sizeof(kOutput));
    memory->load_image(arguments.data(), arguments.size(), kOriginalArguments);
    if (instrumented) {
      const FailureOr<TransportImage> transport =
          initialize_transport(result.manifest,
                               {{threads * workgroups, 1, 1},
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
      if (!instrumented) {
        const uint32_t poison = 13;
        std::memcpy(wrapper.data(), &poison, sizeof(poison));
      }
      memory->load_image(wrapper.data(), wrapper.size(), kWrapper);
      dispatched_arguments = kWrapper;
    }
    ::rocjitsu::test::AqlQueue queue(memory, soc->xcd(0)->command_processor());
    queue.dispatch(kImageBase + descriptor, threads * workgroups, static_cast<uint16_t>(threads),
                   dispatched_arguments);
    engine.run();
    if (counter_seed) {
      EXPECT_TRUE(counter_seed->seeded_every_wave());
    }
    amdgpu::CommandProcessor *cp = soc->xcd(0)->command_processor();
    ASSERT_FALSE(cp->queue_faulted_for_test(1, 0));
    ASSERT_EQ(cp->dispatched_count(), 1u);
    ASSERT_EQ(cp->dispatched_workgroups(), workgroups);
    ASSERT_EQ(snapshots->snapshots().size(), wave_count);
    result.waves = snapshots->snapshots();
    if (workgroups > 1)
      std::ranges::sort(result.waves, [](const auto &first, const auto &second) {
        return std::tuple{first.sgpr(14), first.vgpr(0, 0)} <
               std::tuple{second.sgpr(14), second.vgpr(0, 0)};
      });
    else
      std::ranges::sort(result.waves, {}, &::rocjitsu::test::WavefrontSnapshot::wf_id);
    soc->flush_all();
    if (instrumented)
      memory->read_block(kReport, result.report);
    std::vector<uint8_t> output(std::max(wave_count * output_words_per_wave, 2u) *
                                sizeof(uint32_t));
    memory->read_block(kOutput, output);
    std::memcpy(result.output.data(), output.data(), sizeof(result.output));
    result.full_output.resize(wave_count * output_words_per_wave);
    std::memcpy(result.full_output.data(), output.data(),
                result.full_output.size() * sizeof(uint32_t));
#else
    (void)fixture;
    (void)mode;
    (void)result;
    (void)target_mode;
    (void)threads;
    (void)target_fixture;
    (void)output_words_per_wave;
    (void)workgroups;
#endif
  }
};

TEST_F(ConSanBootstrapSim, ActualWave32DescriptorRunsBothDefinedDisjointWaves) {
  RunResult original;
  run("disjoint", Bootstrap::None, original);
  ASSERT_FALSE(HasFatalFailure());
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
  ASSERT_FALSE(HasFatalFailure());
  run("disjoint", Bootstrap::Enabled, wrapped);
  ASSERT_FALSE(HasFatalFailure());
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
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(control.waves.size(), 2u);
  EXPECT_EQ(control.output, (std::array<uint32_t, 2>{13, 14}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : control.waves)
    EXPECT_EQ(wave.sgpr64(0), kWrapper);
}

TEST_F(ConSanBootstrapSim, RemovingPayloadLoadStopsContextDelivery) {
  RunResult control;
  run("disjoint", Bootstrap::OmitPayloadLoad, control);
  ASSERT_FALSE(HasFatalFailure());
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
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, instrumented);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(original.waves.size(), 2u);
    ASSERT_EQ(instrumented.waves.size(), 2u);
    EXPECT_EQ(instrumented.output, (std::array<uint32_t, 2>{7, 8}));
    EXPECT_EQ(instrumented.output, original.output);
    for (size_t wave = 0; wave < 2; ++wave) {
      const ::rocjitsu::test::WavefrontSnapshot &before = original.waves[wave];
      const ::rocjitsu::test::WavefrontSnapshot &after = instrumented.waves[wave];
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
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(result.waves.size(), 2u);
    const uint32_t sites = static_cast<uint32_t>(result.manifest.sites.size());
    const uint32_t stride = kRowHeaderBytes + sites * 4 * kAccessRecordBytes;
    uint32_t records = 0;
    for (uint32_t wave = 0; wave < 2; ++wave) {
      const size_t row = kReportHeaderBytes + wave * stride;
      EXPECT_EQ(wire::read_u64(result.report, row + wire::kRowSerial), 41u);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWave),
                wave % result.waves_per_workgroup);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWorkgroupX),
                wave / result.waves_per_workgroup);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowStatus), kCompletedRowStatus);
      EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowLoss), 0u);
      for (uint32_t site = 0; site < sites; ++site) {
        for (uint32_t address_group = 0; address_group < 4; ++address_group) {
          const size_t record =
              row + kRowHeaderBytes + (site * 4 + address_group) * kAccessRecordBytes;
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
    ASSERT_FALSE(HasFatalFailure());
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
        for (uint32_t address_group = 0; address_group < 4; ++address_group) {
          const size_t record =
              row + kRowHeaderBytes + (site.first_slot + address_group) * kAccessRecordBytes;
          if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0)
            continue;
          ++count;
          if (fixture == "four_ranges" || fixture == "exhausted") {
            EXPECT_EQ(wave, 0u);
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordGuestMask),
                      1u << address_group);
            EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordLdsOffset),
                      address_group * 4);
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

TEST_F(ConSanBootstrapSim, Gfx1250PublicationBarrierBlocksAnEarlyConsumer) {
  // Both kernels initialize LDS to zero and synchronize. Only the producer
  // then delays. Without the second barrier the consumer must see the old
  // value; original and instrumented barriers must both expose input=7.
  RunResult without_publication_barrier;
  ASSERT_NO_FATAL_FAILURE(
      run("native-37", Bootstrap::None, without_publication_barrier, {"gfx1250", 32}, 65, true));
  EXPECT_EQ(without_publication_barrier.output, (std::array<uint32_t, 2>{0, 0}));

  for (const Bootstrap mode : {Bootstrap::None, Bootstrap::Instrumented}) {
    SCOPED_TRACE(mode == Bootstrap::None ? "original" : "instrumented");
    RunResult synchronized;
    ASSERT_NO_FATAL_FAILURE(run("native-36", mode, synchronized, {"gfx1250", 32}, 65, true));
    EXPECT_EQ(synchronized.output, (std::array<uint32_t, 2>{7, 0}));
  }

  // Callback execution must not substitute for the wait. If callback growth
  // hides this missing wait, increase the producer delay to restore the oracle.
  RunResult without_wait;
  ASSERT_NO_FATAL_FAILURE(run("native-36", Bootstrap::InstrumentedWithoutPublicationWait,
                              without_wait, {"gfx1250", 32}, 65, true));
  EXPECT_EQ(without_wait.output, (std::array<uint32_t, 2>{0, 0}));
}

class ConSanTargetSim : public ConSanBootstrapSim,
                        public ::testing::WithParamInterface<test::TargetMode> {};

void expect_guest_state(const RunResult &before, const RunResult &after) {
  ASSERT_EQ(before.waves.size(), after.waves.size());
  EXPECT_EQ(before.full_output, after.full_output);
  for (size_t index = 0; index < before.waves.size(); ++index) {
    const ::rocjitsu::test::WavefrontSnapshot &a = before.waves[index];
    const ::rocjitsu::test::WavefrontSnapshot &b = after.waves[index];
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
  uint32_t workgroup = 0;
};

std::vector<ObservedAccess> published_accesses(const RunResult &result, bool exhausted = false) {
  std::vector<ObservedAccess> accesses;
  const uint32_t stride = static_cast<uint32_t>(
      report_row_bytes(result.manifest.sites.size(), !result.manifest.barriers.empty()));
  for (uint32_t wave = 0; wave < result.waves.size(); ++wave) {
    const size_t row = kReportHeaderBytes + wave * stride;
    EXPECT_EQ(wire::read_u64(result.report, row + wire::kRowSerial), 41u);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWave),
              wave % result.waves_per_workgroup);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowWorkgroupX),
              wave / result.waves_per_workgroup);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowStatus),
              exhausted && wave == 0 ? (kCompletedRowStatus | wire::kRowExhausted)
                                     : kCompletedRowStatus);
    EXPECT_EQ(wire::read_u32(result.report, row + wire::kRowLoss),
              exhausted && wave == 0 ? 1u : 0u);
    for (const SiteFacts &site : result.manifest.sites) {
      for (uint32_t address_group = 0; address_group < kAddressGroupsPerSite; ++address_group) {
        const size_t record =
            row + kRowHeaderBytes + (site.first_slot + address_group) * kAccessRecordBytes;
        if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0)
          continue;
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordCommit),
                  wire::kAccessCommitted);
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordSite), site.id);
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordWidth), site.width);
        EXPECT_EQ(wire::read_u32(result.report, record + wire::kRecordFlags),
                  wire::kExactGuestMask);
        accesses.push_back({wave % result.waves_per_workgroup, site.id,
                            *wire::read_u32(result.report, record + wire::kRecordLdsOffset),
                            *wire::read_u64(result.report, record + wire::kRecordGuestMask),
                            wave / result.waves_per_workgroup});
      }
    }
  }
  return accesses;
}

TEST_P(ConSanTargetSim, AssemblyAliasPreservesAllGuestRegistersAndPublishesBothWaves) {
  const test::TargetMode target = GetParam();
  for (const char *fixture : {"native-2", "native-6"}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    run(fixture, Bootstrap::None, before, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, after, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8}));
    const std::vector<ObservedAccess> records = published_accesses(after);
    ASSERT_EQ(records.size(), 4u);
    for (const ObservedAccess &record : records) {
      EXPECT_EQ(record.mask, 1u);
      EXPECT_EQ(record.offset, record.wave * 4);
    }
  }
}

TEST_P(ConSanTargetSim, HighLaneMasksAndEmptyExecHaveRealPublicationEvidence) {
  const test::TargetMode target = GetParam();
  for (const char *fixture : {"native-14", "native-12", "native-11", "native-4", "native-13"}) {
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
    const std::vector<ObservedAccess> records = published_accesses(after, name == "native-11");
    ASSERT_EQ(records.size(), name == "native-14"   ? 2u
                              : name == "native-12" ? 0u
                              : name == "native-11" ? 4u
                                                    : 1u);
    for (size_t index = 0; index < records.size(); ++index) {
      const ObservedAccess &record = records[index];
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
  const test::TargetMode target = GetParam();
  RunResult before, after;
  const uint32_t threads = 3 * target.wave_size + 1;
  run("native-2", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-2", Bootstrap::Instrumented, after, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8, 9, 10}));
  const std::vector<ObservedAccess> records = published_accesses(after);
  ASSERT_EQ(records.size(), 8u);
  for (const ObservedAccess &record : records) {
    EXPECT_EQ(record.mask, 1u);
    EXPECT_EQ(record.offset, record.wave * 4);
  }
  EXPECT_EQ(after.waves.back().exec, 1u);
}

// gfx1250 HIP currently emits global prefetch and WAVE_MODE setup; static
// admission tests require a precise rejection. Its native probes run above.
class ConSanHipSim : public ConSanTargetSim {};
TEST_P(ConSanHipSim, CompiledHipRaceDisjointAndReadReadExecuteTheirNativeSites) {
  const test::TargetMode target = GetParam();
  for (const char *fixture : {"hip-1", "hip-2", "hip-3"}) {
    SCOPED_TRACE(fixture);
    RunResult after;
    run(fixture, Bootstrap::Instrumented, after, target, 2 * target.wave_size, true);
    ASSERT_FALSE(HasFatalFailure());
    const std::vector<ObservedAccess> records = published_accesses(after);
    ASSERT_EQ(records.size(), std::string_view(fixture) == "hip-2" ? 4u : 2u);
    uint32_t reads = 0, writes = 0;
    for (const ObservedAccess &record : records) {
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

TEST_F(ConSanBootstrapSim, Gfx1250BarrierPreservesOutputAndGuestStateAtEmptyExecAndHighLanes) {
  for (const char *fixture :
       {"native-15", "native-16", "native-20", "native-21", "native-24", "native-39"}) {
    for (const uint32_t threads : {64u, 65u}) {
      SCOPED_TRACE(fixture);
      SCOPED_TRACE(threads);
      RunResult before, after;
      run(fixture, Bootstrap::None, before, {"gfx1250", 32}, threads, true);
      ASSERT_FALSE(HasFatalFailure());
      run(fixture, Bootstrap::Instrumented, after, {"gfx1250", 32}, threads, true);
      ASSERT_FALSE(HasFatalFailure());
      EXPECT_EQ(before.output[0], 7u);
      EXPECT_EQ(after.output[0], 7u);
      expect_guest_state(before, after);
      const std::vector<ObservedAccess> records = published_accesses(after);
      const bool reuses_word =
          std::string_view(fixture) == "native-24" || std::string_view(fixture) == "native-39";
      ASSERT_EQ(records.size(), reuses_word ? 3u : 2u);
      EXPECT_EQ(records.front().wave, 0u);
      EXPECT_EQ(records.back().wave, 1u);
      if (reuses_word) {
        EXPECT_EQ(records[1].wave, 0u);
        EXPECT_EQ(records[1].site, 2u);
        EXPECT_EQ(after.manifest.sites[records[1].site].access, AccessKind::Write);
      }
      const uint64_t mask = std::string_view(fixture) == "native-20" ? uint64_t{1} << 31 : 1;
      for (const ObservedAccess &record : records)
        EXPECT_EQ(record.mask, mask);
    }
  }
}

TEST_F(ConSanBootstrapSim, RepeatedBarrierLoopsPreserveStateAndRetainIndependentSites) {
  for (const auto &[fixture, iterations] :
       std::array<std::pair<const char *, uint32_t>, 6>{{{"native-25", 0},
                                                         {"native-26", 1},
                                                         {"native-27", 3},
                                                         {"native-28", 6},
                                                         {"native-29", 20},
                                                         {"native-32", 10}}}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    run(fixture, Bootstrap::None, before, {"gfx1250", 32}, 64, true);
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    EXPECT_EQ(after.output[0], 7u);
    const bool late = std::string_view(fixture) == "native-32";
#ifdef CONSAN_ANALYZER_ENABLED
    CapturedExecution execution;
    execution.serial = 41;
    execution.dimensions = {{64, 1, 1}, {64, 1, 1}};
    execution.group_bytes = after.manifest.entry.prepared_group_bytes;
    execution.report = after.report;
    const LdsRaceExecutionResult analysis =
        analyze_lds_race_execution(after.manifest, execution, 8);
    EXPECT_EQ(analysis.rejected_records, 0u);
    EXPECT_EQ(analysis.valid_observations, 1 + 2 * (late ? 1u : std::min(iterations, 4u)));
    EXPECT_EQ(analysis.cross_wave_pairs, 0u);
    EXPECT_EQ(analysis.ordering_unresolved_pairs, late ? 1u : 0u);
    EXPECT_EQ(analysis.lost_occurrences, late ? 0u : 2 * (iterations - std::min(iterations, 4u)));
    EXPECT_EQ(analysis.synchronization_loss,
              2 * (2 * iterations - std::min(2 * iterations, kSyncCapacity)));
    EXPECT_EQ(analysis.coverage_complete, iterations <= 4);
#endif
    const uint32_t stride = *wire::read_u32(after.report, wire::kReportRowStride);
    for (uint32_t wave = 0; wave < 2; ++wave) {
      const size_t row = kReportHeaderBytes + wave * stride;
      EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowFinalPhase), 2 * iterations);
      EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowFinalSequence),
                (late ? 1u : iterations) + (wave == 0 ? 1u : 0u));
      for (uint32_t bank = 0; bank < (late ? 1u : std::min(iterations, 4u)); ++bank) {
        const size_t record =
            row + kRowHeaderBytes + (wave * kAddressGroupsPerSite + bank) * kAccessRecordBytes;
        EXPECT_EQ(wire::read_u32(after.report, record + wire::kRecordSequence), bank + 1);
        EXPECT_EQ(wire::read_u32(after.report, record + wire::kRecordPhase),
                  2 * (late ? 9 : bank) + wave);
      }
    }
  }
}

TEST_F(ConSanBootstrapSim, ThreeBarrierGroupsRetainEachOccurrenceWithoutChangingGuestState) {
  RunResult before, after;
  run("native-38", Bootstrap::None, before, {"gfx1250", 32}, 64, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-38", Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
  ASSERT_EQ(after.manifest.barriers.size(), 3u);
#ifdef CONSAN_ANALYZER_ENABLED
  const CapturedExecution execution = {.serial = 41,
                                       .dimensions = {{64, 1, 1}, {64, 1, 1}},
                                       .group_bytes = after.manifest.entry.prepared_group_bytes,
                                       .arguments = {},
                                       .report = after.report};
  const LdsRaceExecutionResult analysis = analyze_lds_race_execution(after.manifest, execution, 8);
  EXPECT_EQ(analysis.rejected_records, 0u);
  EXPECT_EQ(analysis.valid_observations, 7u);
  EXPECT_EQ(analysis.cross_wave_pairs, 0u);
  EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(analysis.lost_occurrences, 0u);
  EXPECT_EQ(analysis.synchronization_loss, 0u);
  EXPECT_TRUE(analysis.coverage_complete);
#endif
  const uint32_t stride = *wire::read_u32(after.report, wire::kReportRowStride);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    const size_t row = kReportHeaderBytes + wave * stride;
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowFinalPhase), 9u);
    for (uint32_t index = 0; index < 9; ++index) {
      const size_t record =
          row + sync_offset(after.manifest.sites.size()) + index * kSyncRecordBytes;
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kSyncCommit), wire::kBarrierCompleted);
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kSyncGroup), index % 3);
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kSyncOccurrence), index / 3 + 1);
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kSyncPhase), index + 1);
    }
  }
}

#ifdef CONSAN_ANALYZER_ENABLED
TEST_F(ConSanBootstrapSim, UnequalWaveTripCountsKeepValidOrderingWitnesses) {
  for (const auto &[fixture, consumer_iterations] :
       std::array<std::pair<const char *, uint32_t>, 2>{{{"native-40", 2}, {"native-41", 0}}}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    run(fixture, Bootstrap::None, before, {"gfx1250", 32}, 64, true);
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    const CapturedExecution execution = {.serial = 41,
                                         .dimensions = {{64, 1, 1}, {64, 1, 1}},
                                         .group_bytes = after.manifest.entry.prepared_group_bytes,
                                         .arguments = {},
                                         .report = after.report};
    const LdsRaceExecutionResult analysis =
        analyze_lds_race_execution(after.manifest, execution, 8);
    EXPECT_EQ(analysis.valid_observations, 4 + consumer_iterations);
    EXPECT_EQ(analysis.rejected_records, 0u);
    EXPECT_EQ(analysis.integrity_events, 0u);
    EXPECT_EQ(analysis.ordered_pairs, 3 * consumer_iterations);
    EXPECT_EQ(analysis.cross_wave_pairs, 0u);
    EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
    EXPECT_FALSE(analysis.coverage_complete);
    const uint32_t stride = *wire::read_u32(after.report, wire::kReportRowStride);
    EXPECT_EQ(wire::read_u32(after.report, kReportHeaderBytes + wire::kRowFinalPhase), 6u);
    EXPECT_EQ(wire::read_u32(after.report, kReportHeaderBytes + stride + wire::kRowFinalPhase),
              2 * consumer_iterations);
  }
}

TEST_F(ConSanBootstrapSim, RemovingLoopPublicationOrReuseBarrierRetainsConfirmedEndpoints) {
  for (const char *fixture : {"native-30", "native-31"}) {
    SCOPED_TRACE(fixture);
    RunResult after;
    run(fixture, Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true);
    ASSERT_FALSE(HasFatalFailure());
    CapturedExecution execution;
    execution.serial = 41;
    execution.dimensions = {{64, 1, 1}, {64, 1, 1}};
    execution.group_bytes = after.manifest.entry.prepared_group_bytes;
    execution.report = after.report;
    const LdsRaceExecutionResult analysis =
        analyze_lds_race_execution(after.manifest, execution, 8);
    EXPECT_TRUE(analysis.coverage_complete);
    EXPECT_EQ(analysis.valid_observations, 7u);
    EXPECT_EQ(analysis.cross_wave_pairs, std::string_view(fixture) == "native-30" ? 3u : 2u);
    EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
    EXPECT_EQ(analysis.lost_occurrences, 0u);
  }
}

#endif

TEST_F(ConSanBootstrapSim, SeededNearLimitLoopCountersSaturateWithoutStoppingGuest) {
  RunResult before, after;
  run("native-28", Bootstrap::None, before, {"gfx1250", 32}, 64, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-28", Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true, 1, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
  const uint32_t stride = *wire::read_u32(after.report, wire::kReportRowStride);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    const size_t row = kReportHeaderBytes + wave * stride;
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowFinalSequence), UINT32_MAX);
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowFinalPhase), UINT32_MAX);
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowLoss), UINT32_MAX);
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowCounterExhausted), 1u);
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowSyncLoss), UINT32_MAX);
    EXPECT_EQ(wire::read_u32(after.report, row + wire::kRowStatus), 7u);
    const std::span<const uint8_t> journal =
        std::span(after.report)
            .subspan(row + sync_offset(after.manifest.sites.size()),
                     kSyncCapacity * kSyncRecordBytes);
    EXPECT_TRUE(std::ranges::all_of(journal, [](uint8_t byte) { return byte == 0; }));
  }
}

TEST_F(ConSanBootstrapSim, RepeatedAddressGroupExhaustionIsSeparateFromOccupiedBanks) {
  RunResult before, after;
  run("native-33", Bootstrap::None, before, {"gfx1250", 32}, 64, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-33", Bootstrap::Instrumented, after, {"gfx1250", 32}, 64, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
#ifdef CONSAN_ANALYZER_ENABLED
  CapturedExecution execution;
  execution.serial = 41;
  execution.dimensions = {{64, 1, 1}, {64, 1, 1}};
  execution.group_bytes = after.manifest.entry.prepared_group_bytes;
  execution.report = after.report;
  const LdsRaceExecutionResult result = analyze_lds_race_execution(after.manifest, execution, 8);
  EXPECT_EQ(result.valid_observations, 13u);
  EXPECT_EQ(result.cross_wave_pairs, 0u);
  EXPECT_EQ(result.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(result.rejected_records, 0u);
  // Three wave/site streams each present five addresses on three iterations.
  // The fifth address gives 3*3=9 exhaustion events. After the first iteration,
  // four occupied banks per stream give 3*2*4=24 dropped representatives;
  // the legacy loss total combines both: 9+24=33.
  EXPECT_EQ(result.address_group_exhaustion, 9u);
  EXPECT_EQ(result.dropped_representatives, 24u);
  EXPECT_EQ(result.lost_occurrences, 33u);
  EXPECT_EQ(result.synchronization_loss, 0u);
  EXPECT_EQ(result.integrity_events, 0u);
  EXPECT_FALSE(result.coverage_complete);
#endif
}

TEST_P(ConSanTargetSim, NativeWidthsPreserveEveryWordAndAllGuestState) {
  const test::TargetMode target = GetParam();
  const uint32_t wave_size = target.wave_size;
  for (const char *name : {"u16", "i16", "b64", "b128", "u16-high", "i16-high", "b64-high",
                           "b128-high", "b128-alias"}) {
    const std::string_view kind(name);
    if (std::string_view(target.target) != "gfx1250" && kind != "i16" && kind != "b64-high" &&
        kind != "b128-alias")
      continue;
    for (const uint32_t threads : {2 * wave_size, 2 * wave_size + 1}) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(threads);
      const std::string fixture = "width-" + std::string(name);
      RunResult before, after;
      run(fixture, Bootstrap::None, before, target, threads, true, 4);
      ASSERT_FALSE(HasFatalFailure());
      run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4);
      ASSERT_FALSE(HasFatalFailure());
      expect_guest_state(before, after);
      std::vector<uint32_t> expected;
      const uint32_t waves = (threads + wave_size - 1) / wave_size;
      for (uint32_t wave = 0; wave < waves; ++wave) {
        const bool inactive =
            kind.ends_with("-high") && wave * wave_size + wave_size - 1 >= threads;
        std::array<uint32_t, 4> words{7u + wave, 0x01234567u, 0x76543210u, 0xa5a5a5a5u};
        if (kind.starts_with("i16"))
          words[0] += 0xffff8000u;
        else if (kind.starts_with("u16"))
          words[0] += 0x8000u;
        else
          words[1] = 0x89abcdefu;
        if (kind.starts_with("b128")) {
          words[2] = 0x13579bdfu;
          words[3] = 0xfedcba98u;
        }
        if (inactive)
          words.fill(0);
        expected.insert(expected.end(), words.begin(), words.end());
      }
      EXPECT_EQ(before.full_output, expected);
      EXPECT_EQ(after.full_output, expected);
      const std::vector<ObservedAccess> records = published_accesses(after);
      const bool high = kind.ends_with("-high");
      EXPECT_EQ(records.size(), 2u * (high ? threads / wave_size : waves));
      for (const ObservedAccess &record : records)
        EXPECT_EQ(record.mask, high ? uint64_t{1} << (wave_size - 1) : 1);
    }
  }
}

TEST_F(ConSanBootstrapSim, MixedWidthBarrierPreservesTheIndependentCrossWaveValueAndState) {
  RunResult before, after;
  run("mixed-5", Bootstrap::None, before, {"gfx1250", 32}, 65, true);
  ASSERT_FALSE(HasFatalFailure());
  run("mixed-5", Bootstrap::Instrumented, after, {"gfx1250", 32}, 65, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(before.output[0], 8u);
  EXPECT_EQ(after.output[0], 8u);
  const std::vector<ObservedAccess> records = published_accesses(after);
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].offset, 0u);
  EXPECT_EQ(records[1].offset, 4u);
  EXPECT_EQ(after.manifest.sites[0].width, 8u);
  EXPECT_EQ(after.manifest.sites[1].width, 4u);
}

TEST_F(ConSanBootstrapSim, BoundedWorkgroupsPreserveEveryOutputRegisterAndLaunchIdentity) {
  for (const test::TargetMode target : test::kTargetModes)
    for (const char *fixture : {"group-1", "group-2", "group-5"}) {
      const bool barrier = std::string_view(fixture) != "group-1";
      if (barrier && std::string_view(target.target) != "gfx1250")
        continue;
      for (const uint32_t groups : {1u, 4u, 16u})
        for (const uint32_t threads : {2 * target.wave_size, 2 * target.wave_size + 1}) {
          SCOPED_TRACE(target.target);
          SCOPED_TRACE(target.wave_size);
          SCOPED_TRACE(fixture);
          SCOPED_TRACE(groups);
          SCOPED_TRACE(threads);
          RunResult before, after;
          run(fixture, Bootstrap::None, before, target, threads, true, 1, false, groups);
          ASSERT_FALSE(HasFatalFailure());
          run(fixture, Bootstrap::Instrumented, after, target, threads, true, 1, false, groups);
          ASSERT_FALSE(HasFatalFailure());
          expect_guest_state(before, after);
          const uint32_t waves = (threads + target.wave_size - 1) / target.wave_size;
          const bool ttmp = std::string_view(target.target) == "gfx1201" ||
                            std::string_view(target.target) == "gfx1250";
          EXPECT_EQ(after.manifest.entry.workgroup_id_scalar, ttmp ? 117u : 15u);
          std::vector<uint32_t> expected;
          for (uint32_t group = 0; group < groups; ++group)
            for (uint32_t wave = 0; wave < waves; ++wave) {
              expected.push_back(7 + 16 * group + (barrier ? 0 : wave));
              if (ttmp) {
                EXPECT_EQ(before.waves[group * waves + wave].ttmp(9), group);
                EXPECT_EQ(after.waves[group * waves + wave].ttmp(9), group);
              }
              EXPECT_EQ(after.waves[group * waves + wave].sgpr(14), group);
              EXPECT_EQ(after.waves[group * waves + wave].sgpr(15), 99u);
              EXPECT_EQ(after.waves[group * waves + wave].vgpr(0, 0), wave * target.wave_size);
            }
          EXPECT_EQ(before.full_output, expected);
          EXPECT_EQ(after.full_output, expected);
          const std::vector<ObservedAccess> records = published_accesses(after);
          ASSERT_EQ(records.size(), 2u * groups * waves);
          for (const ObservedAccess &record : records) {
            EXPECT_LT(record.workgroup, groups);
            EXPECT_LT(record.wave, waves);
            EXPECT_EQ(record.mask, 1u);
            EXPECT_EQ(record.offset, record.site == 0 || !barrier ? record.wave * 4 : 0);
          }
          if (barrier)
            for (uint32_t row = 0; row < groups * waves; ++row) {
              const size_t offset = kReportHeaderBytes +
                                    row * report_row_bytes(after.manifest.sites.size(), true) +
                                    sync_offset(after.manifest.sites.size());
              EXPECT_EQ(wire::read_u32(after.report, offset + wire::kSyncParticipants), waves);
              EXPECT_EQ(wire::read_u32(after.report, offset + wire::kSyncCommit),
                        wire::kBarrierCompleted);
            }
        }
    }
}

TEST_F(ConSanBootstrapSim, MissingOriginalWorkgroupIdKeepsSingleGroupFunctional) {
  for (const test::TargetMode target : test::kTargetModes) {
    SCOPED_TRACE(target.target);
    SCOPED_TRACE(target.wave_size);
    RunResult before, after;
    const uint32_t threads = 2 * target.wave_size + 1;
    run("native-no-workgroup-id", Bootstrap::None, before, target, threads, true);
    ASSERT_FALSE(HasFatalFailure());
    run("native-no-workgroup-id", Bootstrap::Instrumented, after, target, threads, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    EXPECT_EQ(after.full_output, (std::vector<uint32_t>{7, 8, 9}));
    EXPECT_EQ(after.manifest.entry.workgroup_id_scalar, UINT32_MAX);
    EXPECT_EQ(published_accesses(after).size(), 6u);
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
