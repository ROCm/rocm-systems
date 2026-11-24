// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aql_queue.h"
#include "config_test_util.h"
#include "halt_snapshot_plugin.h"
#include "native_d16_oracle.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "target_fixture.h"
#include "transpose_oracle.h"

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
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
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
    if (pc != entry_pc_) {
      return;
    }
    const uint32_t index = wave.wave_in_group();
    ASSERT_LT(index, seeded_.size());
    if (seeded_[index]) {
      return;
    }
    const uint64_t row = kReport + kReportHeaderBytes + uint64_t{index} * stride_;
    std::array<uint8_t, 4> word{};
    ASSERT_EQ(wave.read_gpu_memory(row + wire::kRowStatus, word),
              amdgpu::VmAccessOutcome::Complete);
    ASSERT_EQ(wire::read_u32(word, 0), wire::kRowEntered);
    wire::write_u32(word, 0, UINT32_MAX - 1);
    // Entry initialization may remain cached. Seed through the owning CU's
    // write-through path so compiled probes cannot reload stale counter zeros.
    wave.cu().l1_scalar().invalidate_all();
    amdgpu::L1VectorCache &cache = wave.cu().l1_vector();
    for (const uint32_t offset :
         {wire::kRowFinalSequence, wire::kRowFinalPhase, wire::kRowLoss, wire::kRowSyncLoss}) {
      const std::array<uint64_t, 1> address = {row + offset};
      ASSERT_EQ(cache.store(address.data(), 1, sizeof(uint32_t), 1, word.data(), amdgpu::Mtype::RW,
                            false, 1, wave.process_id()),
                amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(wave.read_gpu_memory(row + offset, word), amdgpu::VmAccessOutcome::Complete);
      ASSERT_EQ(wire::read_u32(word, 0), UINT32_MAX - 1);
    }
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
           bool seed_counters = false, uint32_t workgroups = 1,
           std::unique_ptr<ExecutionPlugin> observer = {}) {
#ifdef CONSAN_FIXTURE_DIR
    const std::vector<uint8_t> original =
        test::read_fixture(target_fixture ? test::target_fixture_path(target_mode, fixture)
                                          : std::filesystem::path(CONSAN_FIXTURE_DIR) /
                                                (std::string(fixture) + ".hsaco"));
    ASSERT_FALSE(original.empty());
    const TargetProfile *profile = target_profile(target_mode.target);
    ASSERT_NE(profile, nullptr);
    const rocjitsu::IsaFamily arch = profile->arch;
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
        const BarrierFacts &publication = result.manifest.barriers.back();
        if (arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5) {
          ASSERT_LT(publication.prepared_signal, publication.prepared_wait);
        } else {
          ASSERT_EQ(publication.prepared_signal, publication.prepared_wait);
        }
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
    if (fixture.starts_with("width-flat-")) {
      // This bare AQL harness bypasses the KFD initialization of the shared aperture.
      constexpr uint64_t shared = uint64_t{1} << 48;
      soc->set_apertures(shared, shared + UINT32_MAX, 0, 0);
    }
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
    if (observer) {
      ASSERT_TRUE(plugins->add(std::move(observer)));
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
    if (workgroups > 1) {
      std::ranges::sort(result.waves, [](const auto &first, const auto &second) {
        return std::tuple{first.sgpr(14), first.vgpr(0, 0)} <
               std::tuple{second.sgpr(14), second.vgpr(0, 0)};
      });
    } else {
      std::ranges::sort(result.waves, {}, &::rocjitsu::test::WavefrontSnapshot::wf_id);
    }
    soc->flush_all();
    if (instrumented) {
      memory->read_block(kReport, result.report);
    }
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
    (void)observer;
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
      for (uint32_t lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(after.vgpr(reg, lane), before.vgpr(reg, lane));
      }
    }
  }
}

TEST_F(ConSanBootstrapSim, RemovingPointerRestoreChangesTheIndependentNumericOracle) {
  RunResult control;
  run("disjoint", Bootstrap::OmitPointerRestore, control);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(control.waves.size(), 2u);
  EXPECT_EQ(control.output, (std::array<uint32_t, 2>{13, 14}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : control.waves) {
    EXPECT_EQ(wave.sgpr64(0), kWrapper);
  }
}

TEST_F(ConSanBootstrapSim, RemovingPayloadLoadStopsContextDelivery) {
  RunResult control;
  run("disjoint", Bootstrap::OmitPayloadLoad, control);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(control.waves.size(), 2u);
  EXPECT_EQ(control.output, (std::array<uint32_t, 2>{7, 8}));
  for (const ::rocjitsu::test::WavefrontSnapshot &wave : control.waves) {
    EXPECT_NE(wave.sgpr64(control.context_sgpr), kOpaqueContext);
  }
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
      for (uint32_t reg = 0; reg < 16; ++reg) {
        EXPECT_EQ(after.sgprs[reg], before.sgprs[reg]);
      }
      for (uint32_t reg = 0; reg < 8; ++reg) {
        for (uint32_t lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(after.vgpr(reg, lane), before.vgpr(reg, lane));
        }
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
          if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0u) {
            continue;
          }
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
          if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0) {
            continue;
          }
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

class ConSanTargetSim : public ConSanBootstrapSim,
                        public ::testing::WithParamInterface<test::TargetMode> {
protected:
  void check_atomic_add(const std::string &form, uint32_t width,
                        const std::vector<uint32_t> &expected_words);
  void check_atomic_peers(const std::string &form, uint32_t width);
};

TEST_P(ConSanTargetSim, PublicationBarrierBlocksAnEarlyConsumer) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size + 1;
  // Both kernels initialize LDS to zero and synchronize. Only the producer
  // then delays. Without the second barrier the consumer must see the old
  // value; original and instrumented barriers must both expose input=7.
  RunResult without_publication_barrier;
  ASSERT_NO_FATAL_FAILURE(
      run("native-37", Bootstrap::None, without_publication_barrier, target, threads, true));
  EXPECT_EQ(without_publication_barrier.output, (std::array<uint32_t, 2>{0, 0}));

  for (const Bootstrap mode : {Bootstrap::None, Bootstrap::Instrumented}) {
    SCOPED_TRACE(mode == Bootstrap::None ? "original" : "instrumented");
    RunResult synchronized;
    ASSERT_NO_FATAL_FAILURE(run("native-36", mode, synchronized, target, threads, true));
    EXPECT_EQ(synchronized.output, (std::array<uint32_t, 2>{7, 0}));
  }

  // Callback execution must not substitute for the wait. If callback growth
  // hides this missing wait, increase the producer delay to restore the oracle.
  RunResult without_wait;
  ASSERT_NO_FATAL_FAILURE(run("native-36", Bootstrap::InstrumentedWithoutPublicationWait,
                              without_wait, target, threads, true));
  EXPECT_EQ(without_wait.output, (std::array<uint32_t, 2>{0, 0}));
}

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
    for (uint32_t reg = 0; reg < before.original_sgprs; ++reg) {
      EXPECT_EQ(a.sgpr(reg), b.sgpr(reg)) << "SGPR " << reg;
    }
    for (uint32_t reg = 0; reg < before.original_vgprs; ++reg) {
      for (uint32_t lane = 0; lane < a.wf_size; ++lane) {
        EXPECT_EQ(a.vgpr(reg, lane), b.vgpr(reg, lane)) << "VGPR " << reg << " lane " << lane;
      }
    }
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
        if (wire::read_u32(result.report, record + wire::kRecordCommit) == 0) {
          continue;
        }
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
        for (uint32_t reg = 0; reg < before.original_sgprs; ++reg) {
          EXPECT_EQ(after.waves[wave].sgpr(reg), before.waves[wave].sgpr(reg));
        }
        // v3 is the deliberately unsynchronized read result; do not assign it
        // a fixed numeric oracle. Every other guest VGPR must be preserved.
        for (uint32_t reg = 0; reg < before.original_vgprs; ++reg) {
          if (reg != 3) {
            for (uint32_t lane = 0; lane < target.wave_size; ++lane) {
              EXPECT_EQ(after.waves[wave].vgpr(reg, lane), before.waves[wave].vgpr(reg, lane));
            }
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

TEST_P(ConSanTargetSim, BarrierPreservesOutputAndGuestStateAtEmptyExecAndHighLanes) {
  const test::TargetMode target = GetParam();
  for (const char *fixture :
       {"native-15", "native-16", "native-20", "native-21", "native-24", "native-39"}) {
    for (const uint32_t threads : {2 * target.wave_size, 2 * target.wave_size + 1}) {
      SCOPED_TRACE(fixture);
      SCOPED_TRACE(threads);
      RunResult before, after;
      run(fixture, Bootstrap::None, before, target, threads, true);
      ASSERT_FALSE(HasFatalFailure());
      run(fixture, Bootstrap::Instrumented, after, target, threads, true);
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
      const uint64_t mask =
          std::string_view(fixture) == "native-20" ? uint64_t{1} << (target.wave_size - 1) : 1;
      for (const ObservedAccess &record : records) {
        EXPECT_EQ(record.mask, mask);
      }
    }
  }
}

TEST_P(ConSanTargetSim, RepeatedBarrierLoopsPreserveStateAndRetainIndependentSites) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  for (const auto &[fixture, iterations] :
       std::array<std::pair<const char *, uint32_t>, 6>{{{"native-25", 0},
                                                         {"native-26", 1},
                                                         {"native-27", 3},
                                                         {"native-28", 6},
                                                         {"native-29", 20},
                                                         {"native-32", 10}}}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    run(fixture, Bootstrap::None, before, target, threads, true);
    ASSERT_FALSE(HasFatalFailure());
    run(fixture, Bootstrap::Instrumented, after, target, threads, true);
    ASSERT_FALSE(HasFatalFailure());
    expect_guest_state(before, after);
    EXPECT_EQ(after.output[0], 7u);
    const bool late = std::string_view(fixture) == "native-32";
#ifdef CONSAN_ANALYZER_ENABLED
    CapturedExecution execution;
    execution.serial = 41;
    execution.dimensions = {{threads, 1, 1}, {threads, 1, 1}};
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

TEST_P(ConSanTargetSim, RepeatedBarrierLoopsRetainHighLaneMasks) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  RunResult before, after;
  run("native-42", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-42", Bootstrap::Instrumented, after, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
#ifdef CONSAN_ANALYZER_ENABLED
  const CapturedExecution execution = {.serial = 41,
                                       .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
                                       .group_bytes = after.manifest.entry.prepared_group_bytes,
                                       .arguments = {},
                                       .report = after.report};
  const LdsRaceExecutionResult analysis = analyze_lds_race_execution(after.manifest, execution, 8);
  EXPECT_EQ(analysis.valid_observations, 7u);
  EXPECT_EQ(analysis.rejected_records, 0u);
  EXPECT_EQ(analysis.cross_wave_pairs, 0u);
  EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(analysis.lost_occurrences, 0u);
  EXPECT_EQ(analysis.synchronization_loss, 0u);
  EXPECT_TRUE(analysis.coverage_complete);
#endif
  const uint32_t stride = *wire::read_u32(after.report, wire::kReportRowStride);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    for (uint32_t bank = 0; bank < 3; ++bank) {
      const size_t record = kReportHeaderBytes + wave * stride + kRowHeaderBytes +
                            (wave * kAddressGroupsPerSite + bank) * kAccessRecordBytes;
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kRecordSite), wave);
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kRecordSequence), bank + 1);
      EXPECT_EQ(wire::read_u32(after.report, record + wire::kRecordPhase), 2 * bank + wave);
      EXPECT_EQ(wire::read_u64(after.report, record + wire::kRecordGuestMask),
                uint64_t{1} << (target.wave_size - 4 + bank));
    }
  }
}

TEST_P(ConSanTargetSim, ThreeBarrierGroupsRetainEachOccurrenceWithoutChangingGuestState) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  RunResult before, after;
  run("native-38", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-38", Bootstrap::Instrumented, after, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
  ASSERT_EQ(after.manifest.barriers.size(), 3u);
#ifdef CONSAN_ANALYZER_ENABLED
  const CapturedExecution execution = {.serial = 41,
                                       .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
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

TEST_P(ConSanTargetSim, RemovingLoopPublicationOrReuseBarrierRetainsConfirmedEndpoints) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  for (const char *fixture : {"native-30", "native-31"}) {
    SCOPED_TRACE(fixture);
    RunResult after;
    run(fixture, Bootstrap::Instrumented, after, target, threads, true);
    ASSERT_FALSE(HasFatalFailure());
    CapturedExecution execution;
    execution.serial = 41;
    execution.dimensions = {{threads, 1, 1}, {threads, 1, 1}};
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

TEST_P(ConSanTargetSim, SeededNearLimitLoopCountersSaturateWithoutStoppingGuest) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  RunResult before, after;
  run("native-28", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-28", Bootstrap::Instrumented, after, target, threads, true, 1, true);
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

TEST_P(ConSanTargetSim, RepeatedAddressGroupExhaustionIsSeparateFromOccupiedBanks) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  RunResult before, after;
  run("native-33", Bootstrap::None, before, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  run("native-33", Bootstrap::Instrumented, after, target, threads, true);
  ASSERT_FALSE(HasFatalFailure());
  expect_guest_state(before, after);
  EXPECT_EQ(after.output[0], 7u);
#ifdef CONSAN_ANALYZER_ENABLED
  CapturedExecution execution;
  execution.serial = 41;
  execution.dimensions = {{threads, 1, 1}, {threads, 1, 1}};
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
        kind != "b128-alias") {
      continue;
    }
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
        if (kind.starts_with("i16")) {
          words[0] += 0xffff8000u;
        } else if (kind.starts_with("u16")) {
          words[0] += 0x8000u;
        } else {
          words[1] = 0x89abcdefu;
        }
        if (kind.starts_with("b128")) {
          words[2] = 0x13579bdfu;
          words[3] = 0xfedcba98u;
        }
        if (inactive) {
          words.fill(0);
        }
        expected.insert(expected.end(), words.begin(), words.end());
      }
      EXPECT_EQ(before.full_output, expected);
      EXPECT_EQ(after.full_output, expected);
      const std::vector<ObservedAccess> records = published_accesses(after);
      const bool high = kind.ends_with("-high");
      EXPECT_EQ(records.size(), 2u * (high ? threads / wave_size : waves));
      for (const ObservedAccess &record : records) {
        EXPECT_EQ(record.mask, high ? uint64_t{1} << (wave_size - 1) : 1);
      }
    }
  }
}

// The capture counts probes. Count the original memory operations independently,
// including a second execution if instrumentation accidentally duplicates one.
class GuestIssueTrace final : public ExecutionPlugin {
public:
  using Masks = std::array<std::vector<uint64_t>, 2>;

  GuestIssueTrace(std::string_view mnemonic, Masks &masks)
      : ExecutionPlugin("guest_issue_trace"), mnemonic_(mnemonic), masks_(masks) {}
  bool requires_serial_hot_hooks() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &instruction,
                                        amdgpu::Wavefront &wave) override {
    if (instruction.mnemonic() == mnemonic_ && wave.exec())
      masks_.at(wave.wave_in_group()).push_back(wave.exec());
  }

private:
  std::string_view mnemonic_;
  Masks &masks_;
};

TEST_F(ConSanBootstrapSim, GroupFlatLoadUsesTheOriginalPointerAndGuestBarrier) {
  const test::TargetMode target{"gfx950", 64};
  const uint32_t threads = 2 * target.wave_size;
  const uint64_t high = uint64_t{1} << (target.wave_size - 1);
  for (std::string_view scenario : {"overlap", "adjacent", "ordered", "empty", "oob"}) {
    SCOPED_TRACE(scenario);
    const bool ordered = scenario == "ordered";
    const bool empty = scenario == "empty";
    const bool oob = scenario == "oob";
    const std::string fixture = "width-flat-load-" + std::string(scenario);
    RunResult before, after;
    GuestIssueTrace::Masks before_masks, after_masks;
    ASSERT_NO_FATAL_FAILURE(
        run(fixture, Bootstrap::None, before, target, threads, true, 1, false, 1,
            std::make_unique<GuestIssueTrace>("flat_load_dword", before_masks)));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 1,
                                false, 1,
                                std::make_unique<GuestIssueTrace>("flat_load_dword", after_masks)));
    expect_guest_state(before, after);
    EXPECT_EQ(before.full_output, (std::vector<uint32_t>{7, 8}));
    EXPECT_TRUE(before_masks[0].empty());
    EXPECT_EQ(before_masks[1], empty ? std::vector<uint64_t>{} : std::vector<uint64_t>{high});
    EXPECT_EQ(after_masks, before_masks);
    ASSERT_EQ(after.manifest.sites.size(), 2u);
    EXPECT_EQ(after.manifest.sites[0].access, AccessKind::Write);
    EXPECT_EQ(after.manifest.sites[1].access, AccessKind::Read);
    EXPECT_EQ(after.manifest.sites[1].origin, AccessOrigin::GroupFlat);
    const auto records = published_accesses(after);
    ASSERT_EQ(records.size(), empty ? 1u : 2u);
    for (unsigned wave = 0; wave < records.size(); ++wave) {
      EXPECT_EQ(after.manifest.sites[wave].width, 4u);
      EXPECT_EQ(records[wave].wave, wave);
      EXPECT_EQ(records[wave].site, wave);
      EXPECT_EQ(records[wave].mask, high);
      const uint32_t flat_offset = oob ? 1024 : (scenario == "adjacent" ? 1016 : 1020);
      EXPECT_EQ(records[wave].offset, wave ? flat_offset : 1020u);
    }
#ifdef CONSAN_ANALYZER_ENABLED
    const CapturedExecution execution = {.serial = 41,
                                         .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
                                         .group_bytes = after.manifest.entry.prepared_group_bytes,
                                         .arguments = {},
                                         .report = after.report};
    const auto analysis = analyze_lds_race_execution(after.manifest, execution, 4);
    EXPECT_EQ(analysis.coverage_complete, !oob);
    EXPECT_EQ(analysis.rejected_records, oob ? 1u : 0u);
    EXPECT_EQ(analysis.valid_observations, records.size() - (oob ? 1u : 0u));
    EXPECT_EQ(analysis.cross_wave_pairs, scenario == "overlap" ? 1u : 0u);
    EXPECT_EQ(analysis.ordered_pairs, ordered ? 1u : 0u);
    EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
#endif
  }
}

#ifdef CONSAN_ANALYZER_ENABLED
void expect_atomic_analysis(const RunResult &run, uint32_t threads, size_t observations,
                            unsigned cross_wave_pairs) {
  const CapturedExecution execution = {.serial = 41,
                                       .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
                                       .group_bytes = run.manifest.entry.prepared_group_bytes,
                                       .arguments = {},
                                       .report = run.report};
  const LdsRaceExecutionResult analysis = analyze_lds_race_execution(run.manifest, execution, 8);
  EXPECT_TRUE(analysis.coverage_complete);
  EXPECT_EQ(analysis.valid_observations, observations);
  EXPECT_EQ(analysis.rejected_records, 0u);
  EXPECT_EQ(analysis.cross_wave_pairs, cross_wave_pairs);
  EXPECT_EQ(analysis.collision_groups, 0u);
  EXPECT_EQ(analysis.ordered_pairs, 0u);
  EXPECT_EQ(analysis.ordering_unresolved_pairs, 0u);
}
#endif

void ConSanTargetSim::check_atomic_add(const std::string &form, uint32_t width,
                                       const std::vector<uint32_t> &expected_words) {
  SCOPED_TRACE(form);
  const std::string fixture = "width-atomic-" + form + "-count";
  const std::string mnemonic = "ds_add_" + form;
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  const uint64_t high = uint64_t{1} << (target.wave_size - 1);
  const uint64_t atomic_mask = high | (high >> 1);
  RunResult before, after;
  GuestIssueTrace::Masks before_masks, after_masks;
  ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, width / 4,
                              false, 1, std::make_unique<GuestIssueTrace>(mnemonic, before_masks)));
  ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true,
                              width / 4, false, 1,
                              std::make_unique<GuestIssueTrace>(mnemonic, after_masks)));
  expect_guest_state(before, after);
  EXPECT_EQ(before.full_output, expected_words);
  EXPECT_EQ(after_masks, before_masks);
  EXPECT_EQ(after.manifest.scope, ScopeKind::NativeLdsNoSynchronization);
  ASSERT_EQ(after.manifest.sites.size(), 3u);
  const std::array roles{AccessKind::Write, AccessKind::Atomic, AccessKind::Read};
  const auto records = published_accesses(after);
  ASSERT_EQ(records.size(), 6u);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    EXPECT_EQ(before_masks[wave], (std::vector<uint64_t>{atomic_mask}));
    for (uint32_t site = 0; site < roles.size(); ++site) {
      EXPECT_EQ(after.manifest.sites[site].access, roles[site]);
      EXPECT_EQ(after.manifest.sites[site].width, width);
      const auto &record = records[3 * wave + site];
      EXPECT_EQ(record.wave, wave);
      EXPECT_EQ(record.site, site);
      EXPECT_EQ(record.offset, 1024 - 3 * width + 2 * width * wave);
      EXPECT_EQ(record.mask, site == 1 ? atomic_mask : high);
    }
  }
#ifdef CONSAN_ANALYZER_ENABLED
  expect_atomic_analysis(after, threads, records.size(), 0);
#endif
}

void ConSanTargetSim::check_atomic_peers(const std::string &form, uint32_t width) {
  SCOPED_TRACE(form);
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  const uint64_t high = uint64_t{1} << (target.wave_size - 1);
  const uint32_t atomic_offset = 1024 - 2 * width;
  const uint32_t overlap_offset = atomic_offset + width - 4;
  for (const auto &[peer, role, offset, pairs] :
       {std::tuple{"read", AccessKind::Read, overlap_offset, 1u},
        std::tuple{"write", AccessKind::Write, overlap_offset, 1u},
        std::tuple{"atomic", AccessKind::Atomic, atomic_offset, 0u},
        std::tuple{"adjacent", AccessKind::Write, atomic_offset + width, 0u}}) {
    SCOPED_TRACE(peer);
    const std::string fixture = "width-atomic-" + form + "-" + peer;
    RunResult before, after;
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true));
    // The peer discards its racing load before publishing the output.
    expect_guest_state(before, after);
    EXPECT_EQ(before.full_output, (std::vector<uint32_t>{7, 8}));
    ASSERT_EQ(after.manifest.sites.size(), 2u);
    EXPECT_EQ(after.manifest.sites[0].access, AccessKind::Atomic);
    EXPECT_EQ(after.manifest.sites[1].access, role);
    const auto records = published_accesses(after);
    ASSERT_EQ(records.size(), 2u);
    for (uint32_t wave = 0; wave < 2; ++wave) {
      EXPECT_EQ(after.manifest.sites[wave].width,
                wave == 0 || role == AccessKind::Atomic ? width : 4u);
      EXPECT_EQ(records[wave].wave, wave);
      EXPECT_EQ(records[wave].site, wave);
      EXPECT_EQ(records[wave].mask, high);
      EXPECT_EQ(records[wave].offset, wave == 0 ? atomic_offset : offset);
    }
#ifdef CONSAN_ANALYZER_ENABLED
    expect_atomic_analysis(after, threads, records.size(), pairs);
#endif
  }
}

TEST_P(ConSanTargetSim, AtomicU32AddsOnceAndConflictsWithOverlappingOrdinaryPeers) {
  ASSERT_NO_FATAL_FAILURE(check_atomic_add("u32", 4, {9, 10}));
  ASSERT_NO_FATAL_FAILURE(check_atomic_peers("u32", 4));
}

TEST_P(ConSanTargetSim, AtomicU64PropagatesCarryAndConflictsWithTheUpperDword) {
  ASSERT_NO_FATAL_FAILURE(check_atomic_add("u64", 8, {0, 4, 0, 4}));
  ASSERT_NO_FATAL_FAILURE(check_atomic_peers("u64", 8));
}

TEST_P(ConSanTargetSim, AtomicF32PerformsFloatingPointAddition) {
  ASSERT_NO_FATAL_FAILURE(check_atomic_add("f32", 4, {0x40200000u, 0x40200000u}));
  ASSERT_NO_FATAL_FAILURE(check_atomic_peers("f32", 4));
}

class ConSanAtomicF64Sim : public ConSanTargetSim {};

TEST_P(ConSanAtomicF64Sim, AtomicF64PerformsFloatingPointAddition) {
  ASSERT_NO_FATAL_FAILURE(check_atomic_add("f64", 8, {0, 0x40040000u, 0, 0x40040000u}));
  ASSERT_NO_FATAL_FAILURE(check_atomic_peers("f64", 8));
}

TEST_P(ConSanTargetSim, AtomicCompareStorePreservesOldValueAndUsesThePreGuestAddress) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  const uint64_t high = uint64_t{1} << (target.wave_size - 1);
  const std::string_view gpu = target.target;
  const std::string_view mnemonic =
      gpu == "gfx942" || gpu == "gfx950" ? "ds_cmpst_rtn_b32" : "ds_cmpstore_rtn_b32";
  RunResult before, after;
  GuestIssueTrace::Masks before_masks, after_masks;
  ASSERT_NO_FATAL_FAILURE(run("width-atomic-cas-count", Bootstrap::None, before, target, threads,
                              true, 4, false, 1,
                              std::make_unique<GuestIssueTrace>(mnemonic, before_masks)));
  ASSERT_NO_FATAL_FAILURE(run("width-atomic-cas-count", Bootstrap::Instrumented, after, target,
                              threads, true, 4, false, 1,
                              std::make_unique<GuestIssueTrace>(mnemonic, after_masks)));
  expect_guest_state(before, after);
  // Both waves see the old value 7. Only wave zero matches the comparison.
  // The neighboring inactive lane retains the original address in the aliased VGPR.
  EXPECT_EQ(before.full_output, (std::vector<uint32_t>{7, 0x2468, 1008, 0, 7, 7, 1016, 0}));
  EXPECT_EQ(after_masks, before_masks);
  EXPECT_EQ(after.manifest.scope, ScopeKind::NativeLdsNoSynchronization);
  const auto records = published_accesses(after);
  ASSERT_EQ(records.size(), 6u);
  const std::array roles{AccessKind::Write, AccessKind::Atomic, AccessKind::Read};
  ASSERT_EQ(after.manifest.sites.size(), roles.size());
  for (uint32_t wave = 0; wave < 2; ++wave) {
    EXPECT_EQ(before_masks[wave], (std::vector<uint64_t>{high}));
    for (uint32_t site = 0; site < roles.size(); ++site) {
      EXPECT_EQ(after.manifest.sites[site].access, roles[site]);
      EXPECT_EQ(after.manifest.sites[site].width, 4u);
      const auto &record = records[3 * wave + site];
      EXPECT_EQ(record.wave, wave);
      EXPECT_EQ(record.site, site);
      EXPECT_EQ(record.offset, 1012 + 8 * wave);
      EXPECT_EQ(record.mask, high);
    }
  }
#ifdef CONSAN_ANALYZER_ENABLED
  expect_atomic_analysis(after, threads, records.size(), 0);
#endif
  ASSERT_NO_FATAL_FAILURE(check_atomic_peers("cas", 4));
}

class NativeWidthIssueCounter final : public ExecutionPlugin {
public:
  explicit NativeWidthIssueCounter(std::vector<std::array<unsigned, 2>> &counts)
      : ExecutionPlugin("native_width_issue_counter"), counts_(counts) {}
  bool requires_serial_hot_hooks() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &instruction,
                                        amdgpu::Wavefront &wave) override {
    if (!wave.exec())
      return;
    switch (instruction.amdgpu_operation()) {
    case amdgpu::Operation::DS_STORE_B8:
    case amdgpu::Operation::DS_WRITE_B8:
    case amdgpu::Operation::DS_STORE_2ADDR_B32:
    case amdgpu::Operation::DS_STORE_2ADDR_STRIDE64_B32:
    case amdgpu::Operation::DS_WRITE2_B32:
    case amdgpu::Operation::DS_WRITE2ST64_B32:
    case amdgpu::Operation::DS_STORE_2ADDR_B64:
    case amdgpu::Operation::DS_STORE_2ADDR_STRIDE64_B64:
    case amdgpu::Operation::DS_WRITE2_B64:
    case amdgpu::Operation::DS_WRITE2ST64_B64:
    case amdgpu::Operation::DS_STORE_B96:
    case amdgpu::Operation::DS_WRITE_B96:
      ++counts_.at(wave.wave_in_group())[0];
      break;
    case amdgpu::Operation::DS_LOAD_I8:
    case amdgpu::Operation::DS_READ_I8:
    case amdgpu::Operation::DS_LOAD_U8:
    case amdgpu::Operation::DS_READ_U8:
    case amdgpu::Operation::DS_LOAD_2ADDR_B32:
    case amdgpu::Operation::DS_LOAD_2ADDR_STRIDE64_B32:
    case amdgpu::Operation::DS_READ2_B32:
    case amdgpu::Operation::DS_READ2ST64_B32:
    case amdgpu::Operation::DS_LOAD_2ADDR_B64:
    case amdgpu::Operation::DS_LOAD_2ADDR_STRIDE64_B64:
    case amdgpu::Operation::DS_READ2_B64:
    case amdgpu::Operation::DS_READ2ST64_B64:
    case amdgpu::Operation::DS_LOAD_B96:
    case amdgpu::Operation::DS_READ_B96:
      ++counts_.at(wave.wave_in_group())[1];
      break;
    default:
      break;
    }
  }

private:
  std::vector<std::array<unsigned, 2>> &counts_;
};

TEST_P(ConSanTargetSim, DualInstructionsExecuteOnceAndPreserveAllGuestRegisters) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size + 1;
  for (const unsigned bits : {32u, 64u}) {
    for (const char *spacing : {"normal", "stride64"}) {
      const std::string fixture =
          "width-dual-b" + std::to_string(bits) + "-" + spacing + "-values-high";
      SCOPED_TRACE(fixture);
      RunResult before, after;
      std::vector<std::array<unsigned, 2>> before_counts(3), after_counts(3);
      ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, 4, false,
                                  1, std::make_unique<NativeWidthIssueCounter>(before_counts)));
      ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4,
                                  false, 1,
                                  std::make_unique<NativeWidthIssueCounter>(after_counts)));
      expect_guest_state(before, after);
      EXPECT_EQ(before_counts, (std::vector<std::array<unsigned, 2>>{{0, 1}, {1, 0}, {0, 0}}));
      EXPECT_EQ(after_counts, before_counts);
      const std::vector<uint32_t> expected = {7,
                                              0x89abcdefu,
                                              bits == 32 ? 0x76543210u : 0x13579bdfu,
                                              bits == 32 ? 0xa5a5a5a5u : 0xfedcba98u,
                                              8,
                                              0x89abcdefu,
                                              bits == 32 ? 0x76543210u : 0x13579bdfu,
                                              bits == 32 ? 0xa5a5a5a5u : 0xfedcba98u,
                                              0,
                                              0,
                                              0,
                                              0};
      EXPECT_EQ(before.full_output, expected);
      EXPECT_EQ(after.full_output, expected);
    }
  }
}

#ifdef CONSAN_ANALYZER_ENABLED
TEST_P(ConSanTargetSim, DualStoreCollisionMatchesDifferentOrdinalsInDistinctLaneGroups) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = target.wave_size;
  RunResult after;
  ASSERT_NO_FATAL_FAILURE(run("width-dual-b32-normal-collision", Bootstrap::Instrumented, after,
                              target, threads, true, 4));
  const CapturedExecution execution = {.serial = 41,
                                       .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
                                       .group_bytes = after.manifest.entry.prepared_group_bytes,
                                       .arguments = {},
                                       .report = after.report};
  const LdsRaceExecutionResult analysis = analyze_lds_race_execution(after.manifest, execution, 4);
  EXPECT_TRUE(analysis.coverage_complete);
  EXPECT_EQ(analysis.valid_observations, 4u);
  EXPECT_EQ(analysis.cross_wave_pairs, 0u);
  EXPECT_EQ(analysis.collision_groups, 1u);
  EXPECT_EQ(analysis.collision_lane_pairs, 1u);
  ASSERT_EQ(analysis.collision_examples.size(), 1u);
  const LdsRaceFinding &example = analysis.collision_examples.front();
  EXPECT_EQ(example.first.offset, 28u);
  EXPECT_EQ(example.second.offset, 28u);
  EXPECT_EQ(example.first.mask | example.second.mask, 3u);
  EXPECT_NE(example.first.mask, example.second.mask);
  EXPECT_EQ(example.first.range_id, example.first.mask == 1 ? 1u : 0u);
  EXPECT_EQ(example.second.range_id, example.second.mask == 1 ? 1u : 0u);
}
#endif

TEST_P(ConSanTargetSim, B96PreservesThreeWordsAndRecordsThePreGuestAddressOnce) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size + 1;
  for (const bool high : {false, true}) {
    const char *fixture = high ? "width-b96-alias-high" : "width-b96";
    SCOPED_TRACE(fixture);
    RunResult before, after;
    std::vector<std::array<unsigned, 2>> before_counts(3), after_counts(3);
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, 4, false,
                                1, std::make_unique<NativeWidthIssueCounter>(before_counts)));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4,
                                false, 1, std::make_unique<NativeWidthIssueCounter>(after_counts)));
    expect_guest_state(before, after);
    const unsigned active_waves = high ? 2 : 3;
    std::vector<uint32_t> expected;
    for (unsigned wave = 0; wave < 3; ++wave) {
      const unsigned count = wave < active_waves ? 1 : 0;
      EXPECT_EQ(before_counts[wave], (std::array<unsigned, 2>{count, count}));
      EXPECT_EQ(after_counts[wave], before_counts[wave]);
      std::array<uint32_t, 4> words{7 + wave, 0x89abcdefu, 0x13579bdfu, 0xa5a5a5a5u};
      if (!count)
        words.fill(0);
      expected.insert(expected.end(), words.begin(), words.end());
    }
    EXPECT_EQ(before.full_output, expected);
    EXPECT_EQ(after.full_output, expected);
    const std::vector<ObservedAccess> records = published_accesses(after);
    ASSERT_EQ(after.manifest.sites.size(), 2u);
    EXPECT_EQ(after.manifest.sites[0].access, AccessKind::Write);
    EXPECT_EQ(after.manifest.sites[1].access, AccessKind::Read);
    ASSERT_EQ(records.size(), 2 * active_waves);
    for (unsigned index = 0; index < records.size(); ++index) {
      const ObservedAccess &record = records[index];
      EXPECT_EQ(record.site, index % 2);
      EXPECT_EQ(after.manifest.sites[index % 2].width, 12u);
      EXPECT_EQ(record.wave, index / 2);
      EXPECT_EQ(record.offset, 16 + 16 * (index / 2));
      EXPECT_EQ(record.mask, high ? uint64_t{1} << (target.wave_size - 1) : 1u);
    }
  }
}

TEST_P(ConSanTargetSim, ByteAccessPreservesNeighborBytesAndExtendsOnlyActiveLanes) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size + 1;
  for (const bool high : {false, true}) {
    const char *fixture = high ? "width-i8-alias-high" : "width-u8";
    SCOPED_TRACE(fixture);
    RunResult before, after;
    std::vector<std::array<unsigned, 2>> before_counts(3), after_counts(3);
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, 4, false,
                                1, std::make_unique<NativeWidthIssueCounter>(before_counts)));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4,
                                false, 1, std::make_unique<NativeWidthIssueCounter>(after_counts)));
    expect_guest_state(before, after);
    for (unsigned wave = 0; wave < 3; ++wave) {
      const unsigned count = high && wave == 2 ? 0 : 1;
      EXPECT_EQ(before_counts[wave], (std::array<unsigned, 2>{count, count}));
      EXPECT_EQ(after_counts[wave], before_counts[wave]);
      const uint32_t byte = 0x87u + wave;
      EXPECT_EQ(after.full_output[4 * wave], count ? (high ? 0xffffff00u | byte : byte) : 0);
      EXPECT_EQ(after.full_output[4 * wave + 1], count ? (byte << 24) | 0x00234567u : 0);
    }
  }
}

// Observe the guest issue and routed LDS request independently of the emitted probe.
struct TransposeTrace {
  std::vector<uint64_t> issued_exec;
  std::vector<uint64_t> routed_masks;
};

class TransposeRequests final : public ExecutionPlugin {
public:
  TransposeRequests(TransposeTrace &trace, std::string mnemonic, uint32_t width,
                    std::string_view scenario)
      : ExecutionPlugin("transpose_requests"), trace_(trace), mnemonic_(std::move(mnemonic)),
        width_(width), scenario_(scenario) {}
  bool requires_serial_hot_hooks() const override { return true; }
  bool observes_memory_routing() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst,
                                        amdgpu::Wavefront &wave) override {
    if (inst.mnemonic() == mnemonic_)
      trace_.issued_exec.push_back(wave.exec());
  }
  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                  const Instruction &, amdgpu::Wavefront &wave) override {
    if (access.mnemonic != mnemonic_)
      return;
    EXPECT_EQ(access.bytes_per_lane(), width_);
    EXPECT_EQ(wave.wave_in_group(), 0u);
    ASSERT_EQ(trace_.issued_exec.size(), 1u);
    EXPECT_EQ(access.architectural_exec_lane_mask, trace_.issued_exec.front());
    trace_.routed_masks.push_back(access.request_lane_mask);
    ASSERT_EQ(access.addresses.size(), access.wavefront_size);
    for (uint32_t lane = 0; lane < access.wavefront_size; ++lane) {
      if ((access.request_lane_mask & (uint64_t{1} << lane)) == 0)
        continue;
      const uint32_t offset = scenario_ == "overlap" || scenario_ == "adjacent" ? 32 * (lane % 4)
                              : scenario_ == "short" && lane >= 4               ? 0
                                                                                : 16 * lane;
      EXPECT_EQ(access.addresses[lane], wave.lds_base() + 2048 + offset) << lane;
    }
  }

private:
  TransposeTrace &trace_;
  std::string mnemonic_;
  uint32_t width_;
  std::string_view scenario_;
};

TEST_F(ConSanBootstrapSim, TransposeIssuesOnceAndRoutesPhysicalSourceLanes) {
  for (const test::TargetMode target :
       {test::TargetMode{"gfx950", 64}, test::TargetMode{"gfx1250", 32}}) {
    const bool cdna4 = target.wave_size == 64;
    const uint64_t full = cdna4 ? UINT64_MAX : UINT32_MAX;
    for (const uint32_t bits : {4u, 6u, 8u, 16u}) {
      const uint32_t width = 4 * test::transpose_words(bits, cdna4);
      const std::string payload = "b" + std::to_string(8 * width);
      const std::string mnemonic = cdna4 ? "ds_read_" + payload + "_tr_b" + std::to_string(bits)
                                         : "ds_load_tr" + std::to_string(bits) + "_" + payload;
      for (const std::string_view scenario :
           {"full", "partial", "empty", "overlap", "adjacent", "short"}) {
        if (cdna4 && scenario == "short")
          continue;
        SCOPED_TRACE(target.target);
        SCOPED_TRACE(mnemonic);
        SCOPED_TRACE(scenario);
        const std::string fixture =
            "width-transpose-" + std::to_string(bits) + "-" + std::string(scenario);
        const bool peer = scenario == "overlap" || scenario == "adjacent";
        const uint32_t threads = scenario == "short" ? 4 : (peer ? 2 : 1) * target.wave_size;
        const uint32_t words_per_wave = 4 * std::min(threads, target.wave_size);
        const uint64_t exec = scenario == "short"     ? 15
                              : scenario == "empty"   ? 0
                              : scenario == "partial" ? (uint64_t{1} << (target.wave_size - 1))
                                                      : full;
        const uint64_t requests = cdna4 ? exec : full;
        RunResult before, after;
        for (auto mode : {Bootstrap::None, Bootstrap::Instrumented}) {
          TransposeTrace trace;
          RunResult &result = mode == Bootstrap::None ? before : after;
          ASSERT_NO_FATAL_FAILURE(
              run(fixture, mode, result, target, threads, true, words_per_wave, false, 1,
                  std::make_unique<TransposeRequests>(trace, mnemonic, width, scenario)));
          EXPECT_EQ(trace.issued_exec, (std::vector<uint64_t>{exec}));
          if (requests || !trace.routed_masks.empty()) {
            EXPECT_EQ(trace.routed_masks, (std::vector<uint64_t>{requests}));
          }
        }
        expect_guest_state(before, after);
        if (cdna4 && scenario == "partial")
          continue; // ISA result is undefined; replay checks the diagnostic.
        ASSERT_EQ(after.full_output.size(), 4 * threads);
        for (uint32_t thread = 0; thread < threads; ++thread) {
          const auto expected =
              test::transpose_output(thread % target.wave_size, bits, cdna4, scenario);
          for (uint32_t word = 0; word < 4; ++word)
            EXPECT_EQ(after.full_output[4 * thread + word], expected[word])
                << thread << " word " << word;
        }
        if (scenario == "short") {
          for (uint32_t lane = threads; lane < target.wave_size; ++lane) {
            const auto expected = test::transpose_output(lane, bits, cdna4, scenario);
            for (uint32_t word = 0; word < width / 4; ++word)
              EXPECT_EQ(after.waves[0].vgpr(8 + word, lane), expected[word]) << lane;
          }
        }
      }
    }
  }
}

class D16IssueTrace final : public ExecutionPlugin {
public:
  explicit D16IssueTrace(std::vector<std::vector<std::string>> &instructions)
      : ExecutionPlugin("d16_issue_trace"), instructions_(instructions) {}
  bool requires_serial_hot_hooks() const override { return true; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &instruction,
                                        amdgpu::Wavefront &wave) override {
    const std::string_view name = instruction.mnemonic();
    if (wave.exec() && name.starts_with("ds_") && name.find("_d16") != std::string_view::npos)
      instructions_.at(wave.wave_in_group()).emplace_back(name);
  }

private:
  std::vector<std::vector<std::string>> &instructions_;
};

TEST_P(ConSanTargetSim, D16AccessesSelectSubwordsAndExecuteEachGuestFormOnce) {
  const test::TargetMode target = GetParam();
  const bool legacy = target_profile(target.target)->id < 1000;
  const std::string read = legacy ? "ds_read_" : "ds_load_";
  const std::string write = legacy ? "ds_write_" : "ds_store_";
  for (const std::string_view name :
       {"d16-u8", "d16-i8-high", "d16-u16", "d16-overlap", "d16-adjacent"}) {
    SCOPED_TRACE(name);
    const bool peer = name == "d16-overlap" || name == "d16-adjacent";
    const bool signed_byte = name == "d16-i8-high";
    const bool halfword = name == "d16-u16";
    const uint32_t threads = 2 * target.wave_size + (peer ? 0 : 1);
    const uint32_t waves = peer ? 2 : 3;
    const std::string fixture = "width-" + std::string(name);
    RunResult before, after;
    std::vector<std::vector<std::string>> before_trace(waves), after_trace(waves);
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, 4, false,
                                1, std::make_unique<D16IssueTrace>(before_trace)));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4,
                                false, 1, std::make_unique<D16IssueTrace>(after_trace)));
    expect_guest_state(before, after);
    const std::string load = read + (halfword ? "u16_d16" : signed_byte ? "i8_d16" : "u8_d16");
    const std::vector<std::string> operations =
        peer ? std::vector<std::string>{write + "b8_d16_hi", write + "b16_d16_hi",
                                        read + "u8_d16",     read + "u8_d16_hi",
                                        read + "i8_d16",     read + "i8_d16_hi",
                                        read + "u16_d16",    read + "u16_d16_hi"}
             : std::vector<std::string>{write + (halfword ? "b16_d16_hi" : "b8_d16_hi"), load,
                                        load + "_hi"};
    EXPECT_EQ(before_trace, after_trace);
    ASSERT_EQ(after.full_output.size(), 4 * waves);
    for (uint32_t wave = 0; wave < waves; ++wave) {
      const bool empty = (peer && wave == 1) || (signed_byte && wave == 2);
      if (empty)
        EXPECT_TRUE(before_trace[wave].empty());
      else
        EXPECT_EQ(before_trace[wave], operations);
      if (peer)
        continue;
      const std::array<uint32_t, 4> expected =
          empty ? std::array<uint32_t, 4>{}
                : test::d16_output(target.target, wave, halfword, signed_byte);
      EXPECT_TRUE(
          std::equal(expected.begin(), expected.end(), after.full_output.begin() + 4 * wave));
      // Lane one starts with the same sentinels as the active lane but neither
      // writes the alias address nor executes a D16 load.
      if (wave < 2) {
        EXPECT_EQ(after.waves[wave].vgpr(8, 1), 0xdeadbeefu);
        EXPECT_EQ(after.waves[wave].vgpr(9, 1), 0x01234567u);
      }
    }
  }
}

TEST_P(ConSanTargetSim, B96OnlyConflictsWithTheOverlappingPeerDword) {
  const test::TargetMode target = GetParam();
  const uint32_t threads = 2 * target.wave_size;
  for (const auto &[fixture, peer_offset] : std::array<std::pair<const char *, uint32_t>, 3>{
           {{"width-b96-overlap", 24}, {"width-b96-adjacent", 28}, {"width-b96-disjoint", 32}}}) {
    SCOPED_TRACE(fixture);
    RunResult before, after;
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::None, before, target, threads, true, 4));
    ASSERT_NO_FATAL_FAILURE(run(fixture, Bootstrap::Instrumented, after, target, threads, true, 4));
    expect_guest_state(before, after);
    const std::vector<ObservedAccess> records = published_accesses(after);
    ASSERT_EQ(records.size(), 2u);
    ASSERT_EQ(after.manifest.sites.size(), 2u);
    for (unsigned index = 0; index < 2; ++index) {
      EXPECT_EQ(records[index].wave, index);
      EXPECT_EQ(records[index].mask, 1u);
      EXPECT_EQ(records[index].site, index);
      EXPECT_EQ(records[index].offset, index == 0 ? 16u : peer_offset);
      EXPECT_EQ(after.manifest.sites[index].access, AccessKind::Write);
      EXPECT_EQ(after.manifest.sites[index].width, index == 0 ? 12u : 4u);
    }
#ifdef CONSAN_ANALYZER_ENABLED
    const CapturedExecution execution = {.serial = 41,
                                         .dimensions = {{threads, 1, 1}, {threads, 1, 1}},
                                         .group_bytes = after.manifest.entry.prepared_group_bytes,
                                         .arguments = {},
                                         .report = after.report};
    const LdsRaceExecutionResult analysis =
        analyze_lds_race_execution(after.manifest, execution, 8);
    EXPECT_TRUE(analysis.coverage_complete);
    EXPECT_EQ(analysis.valid_observations, 2u);
    EXPECT_EQ(analysis.cross_wave_pairs, peer_offset == 24 ? 1u : 0u);
    EXPECT_EQ(analysis.rejected_records, 0u);
#endif
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
  for (const test::TargetMode target : test::kTargetModes) {
    for (const char *fixture : {"group-1", "group-2", "group-5"}) {
      const bool barrier = std::string_view(fixture) != "group-1";
      for (const uint32_t groups : {1u, 4u, 16u}) {
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
          for (uint32_t group = 0; group < groups; ++group) {
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
          if (barrier) {
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
INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanAtomicF64Sim,
                         ::testing::Values(test::TargetMode{"gfx942", 64},
                                           test::TargetMode{"gfx950", 64},
                                           test::TargetMode{"gfx1250", 32}),
                         target_name);
INSTANTIATE_TEST_SUITE_P(ConSanTargets, ConSanHipSim,
                         ::testing::ValuesIn(std::span(test::kTargetModes).first(6)), target_name);
} // namespace
} // namespace rocjitsu::consan
