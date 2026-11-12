// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/async_mma_policy.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::race_detector;
using namespace rocjitsu::test::execution_plugin;

struct Report {
  std::map<std::string, std::string> fields;
  std::string trace;
};

std::vector<Report> reports(const std::string &output) {
  std::vector<Report> result;
  for (size_t pos = 0; (pos = output.find("RACE ", pos)) != std::string::npos;) {
    const auto newline = output.find('\n', pos);
    const auto end = output.find("END_RACE\n", newline);
    EXPECT_NE(end, std::string::npos) << output;
    if (end == std::string::npos)
      break;
    Report report;
    std::istringstream header(output.substr(pos + 5, newline - pos - 5));
    for (std::string field; header >> field;) {
      const auto equals = field.find('=');
      EXPECT_NE(equals, std::string::npos) << field;
      report.fields.emplace(field.substr(0, equals), field.substr(equals + 1));
    }
    report.trace = output.substr(newline + 1, end - newline - 1);
    result.push_back(std::move(report));
    pos = end + 9;
  }
  return result;
}

class HazardProbe final : public ExecutionPlugin {
public:
  bool observes_before_execute_instruction() const override { return true; }
  bool observes_async_instruction_issued() const override { return true; }

  explicit HazardProbe(bool interested, std::string name = "hazard_probe")
      : ExecutionPlugin(std::move(name)), interested(interested) {}
  bool observes_register_hazards() const override { return interested; }
  bool supports_async_instructions() const override { return true; }
  bool observes_hot_hooks_for_wavefront(const Wavefront *) const override { return subscribed; }
  void onAmdgpuRegisterHazard(const RegisterHazardObservation &hazard, const Wavefront &) override {
    hazards.push_back(hazard);
    phases.emplace_back(hazard.consumer_pc, "hazard");
  }
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &, Wavefront &,
                                        std::span<const uint32_t>) override {
    phases.emplace_back(pc, "before");
    if (before)
      before(pc);
  }
  void onAmdgpuAsyncInstructionIssued(uint64_t pc, const Instruction &, Wavefront &) override {
    phases.emplace_back(pc, "async");
  }
  void onAmdgpuDispatchExecutionEnd(uint32_t) override { ++completed; }
  void onAmdgpuWavefrontHalted(Wavefront &wave) override {
    if (halt)
      halt(wave);
  }
  void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, std::span<Wavefront *> waves) override {
    if (dispatch)
      dispatch(waves);
  }

  bool interested;
  bool subscribed = true;
  unsigned completed = 0;
  std::vector<RegisterHazardObservation> hazards;
  std::vector<std::pair<uint64_t, std::string>> phases;
  std::function<void(uint64_t)> before;
  std::function<void(std::span<Wavefront *>)> dispatch;
  std::function<void(Wavefront &)> halt;
};

struct DiagnosticHarness {
  PluginFixture fixture;
  StringSink *sink = nullptr;
  RaceDetectorPlugin *race = nullptr;
  HazardProbe *probe = nullptr;

  void initialize(bool warnings = false, bool probe_first = false, std::string_view arch = "cdna4",
                  bool observe_hazards = true, uint32_t async_helpers = 0) {
    ASSERT_NO_FATAL_FAILURE(fixture.initialize(2, arch, arch == "cdna4" ? 64 : 32,
                                               arch == "cdna4" ? 104 : 128, 256, 1, async_helpers,
                                               warnings ? "warn" : "off"));
    PluginSinkConfig sinks;
    sink = &sinks.emplace<StringSink>();
    fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sinks));
    auto checker = std::make_unique<RaceDetectorPlugin>();
    race = checker.get();
    auto observer = std::make_unique<HazardProbe>(observe_hazards);
    probe = observer.get();
    if (probe_first)
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
    ASSERT_TRUE(fixture.plugin_group_->add(std::move(checker)));
    if (!probe_first)
      ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
    fixture.soc->set_plugin_group(fixture.plugin_group_);
    fixture.plugin_group_->onInit();
  }

  void run(const std::vector<uint32_t> &code, uint32_t grid = 64, uint32_t workgroup = 64,
           uint32_t lds_bytes = 0) {
    const auto kernel = fixture.write_kernel(0x1000, code.data(), code.size(), 12, lds_bytes);
    test::AqlQueue queue(fixture.mem, fixture.cp());
    queue.dispatch(kernel, grid, workgroup);
    fixture.run_until_idle();
    ASSERT_EQ(probe->completed, 1u) << sink->str();
    for (uint32_t slot = 0; slot < 2; ++slot)
      if (const auto *wave = fixture.cu()->wf(slot))
        EXPECT_FALSE(wave->has_plugin_state(race->slot_index()));
  }
};

struct Program {
  static constexpr uint64_t kEntry = 0x1000 + sizeof(amdhsa::kernel_descriptor_t);
  std::vector<uint32_t> words;

  uint64_t emit(std::span<const uint32_t> instruction) {
    const uint64_t pc = kEntry + words.size() * sizeof(uint32_t);
    words.insert(words.end(), instruction.begin(), instruction.end());
    return pc;
  }
  void exec(uint8_t mask) {
    emit(cdna4::build_sop1(cdna4::kSMovB64Sop1,
                           {.ssrc0 = static_cast<uint8_t>(128 + mask), .sdst = 126}));
  }
  void setup() {
    exec(4);
    emit(cdna4::build_sop1(cdna4::kSMovB32Sop1, {.ssrc0 = 255, .sdst = 0}));
    emit(std::array{0x2000u});
    emit(cdna4::build_sop1(cdna4::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 1}));
    emit(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 0, .vdst = 0}));
    emit(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 128, .vdst = 1}));
  }
  uint64_t global_load() {
    return emit(cdna4::build_flat(cdna4::kFlatLoadDwordFlat,
                                  {.seg = 2, .addr = 0, .saddr = 0x7f, .vdst = 4}));
  }
  uint64_t vector_read(uint8_t destination = 8) {
    return emit(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 260, .vdst = destination}));
  }
  void end() { emit(cdna4::build_sopp(cdna4::kSEndpgmSopp)); }
};

TEST(RegisterHazardPluginTest, RegisterCheckingUsesOnlyCoreFindings) {
  for (const char *config : std::array<const char *, 3>{nullptr, "", "{}"}) {
    RaceDetectorPlugin plugin(config);
    EXPECT_TRUE(plugin.observes_register_hazards());
    EXPECT_FALSE(plugin.observes_vgpr_reads());
    EXPECT_FALSE(plugin.observes_vgpr_writes());
    EXPECT_FALSE(plugin.observes_scalar_register_reads());
    EXPECT_FALSE(plugin.observes_scalar_register_writes());
    EXPECT_TRUE(plugin.observes_memory_routing());
    EXPECT_TRUE(plugin.observes_tensor_dma_issue());
  }
}

TEST(RegisterHazardPluginTest, Cdna2TrapEntryDrainsObserverStateIndependentOfCoreWarnings) {
  for (const bool warnings : {false, true}) {
    for (const bool handler_installed : {false, true}) {
      SCOPED_TRACE(std::format("warnings={} handler={}", warnings, handler_installed));
      PluginFixture fixture;
      ASSERT_NO_FATAL_FAILURE(
          fixture.initialize(1, "cdna2", 64, 102, 32, 1, 0, warnings ? "warn" : "off"));
      fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      ASSERT_TRUE(fixture.plugin_group_->add(std::make_unique<HazardProbe>(true)));
      fixture.soc->set_plugin_group(fixture.plugin_group_);
      auto *cu = fixture.cu();
      if (handler_installed)
        cu->set_trap_handler_resolver(
            [](const Wavefront &) { return ComputeUnitCore::TrapHandlerConfig{0x2000, 0, false}; });
      auto *wave = cu->dispatch_wf(0, 0x1000, 32, 8);
      ASSERT_NE(wave, nullptr);
      auto &scoreboard = wave->ensure_memory_wait_scoreboard();
      constexpr RegisterRef result{RegClass::SGPR, 2, 1};
      scoreboard.add({scoreboard.issue(WaitCounterKind::Ds), 0x1000, UINT64_MAX, result,
                      WaitCounterKind::Ds, 0xf});
      ASSERT_TRUE(wave->memory_wait_shadow().pending(result));

      // A completed memory exception enters between instructions, without S_TRAP's wait path.
      fixture.mem->write32(0x1000, S_NOP);
      wave->set_mode_raw(wave->mode_raw() | (1u << 20));
      wave->complete_flat_memory_violation();
      ASSERT_TRUE(wave->memory_trap_pending());
      cu->step();

      EXPECT_EQ(wave->in_trap_handler(), handler_installed);
      EXPECT_FALSE(wave->memory_trap_pending());
      EXPECT_EQ(wave->memory_wait_shadow().pending(result), !handler_installed);
      EXPECT_EQ(scoreboard.outstanding(WaitCounterKind::Ds), handler_installed ? 0u : 1u);
      wave->halt();
    }
  }
}

TEST(RegisterHazardPluginTest, IssuerReportsExactFieldsAndWaitControls) {
  enum class Producer { Global, Lds, Sgpr, Ttmp };
  for (const auto producer : {Producer::Global, Producer::Lds, Producer::Sgpr, Producer::Ttmp})
    for (bool write : {false, true})
      for (bool waited : {false, true}) {
        SCOPED_TRACE(std::format("producer={} write={} waited={}", static_cast<int>(producer),
                                 write, waited));
        Program program;
        program.setup();
        const bool vector = producer == Producer::Global || producer == Producer::Lds;
        const uint8_t scalar = producer == Producer::Ttmp ? kTtmpSelectorFirst : 4;
        const uint64_t producer_pc =
            producer == Producer::Global ? program.global_load()
            : producer == Producer::Lds
                ? program.emit(cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 4}))
                : program.emit(cdna4::build_smem(cdna4::kSLoadDwordSmem,
                                                 {.sbase = 0, .sdata = scalar, .imm = 1}));
        if (waited)
          program.emit(cdna4::build_sopp(
              cdna4::kSWaitcntSopp,
              {.simm16 = static_cast<uint16_t>(producer == Producer::Global ? 0x0f70 : 0xc07f)}));
        const uint64_t consumer_pc =
            vector ? program.emit(cdna4::build_vop1(
                         cdna4::kVMovB32Vop1, {.src0 = static_cast<uint16_t>(write ? 128 : 260),
                                               .vdst = static_cast<uint8_t>(write ? 4 : 8)}))
                   : program.emit(cdna4::build_sop1(
                         cdna4::kSMovB32Sop1, {.ssrc0 = static_cast<uint8_t>(write ? 128 : scalar),
                                               .sdst = static_cast<uint8_t>(write ? scalar : 8)}));
        program.end();
        DiagnosticHarness h;
        ASSERT_NO_FATAL_FAILURE(h.initialize());
        if (producer == Producer::Ttmp)
          h.probe->dispatch = [](std::span<Wavefront *> waves) {
            // The load must be permitted to write TTMP before it can create a dependency.
            for (auto *wave : waves)
              wave->set_in_trap_handler(true);
          };
        ASSERT_NO_FATAL_FAILURE(h.run(program.words));
        EXPECT_EQ(h.fixture.cu()->memory_wait_diagnostic_count(), 0u);
        const auto output = reports(h.sink->str());
        ASSERT_EQ(output.size(), waited ? 0u : 1u) << h.sink->str();
        if (waited) {
          EXPECT_TRUE(h.probe->hazards.empty());
          continue;
        }
        const auto &report = output.front();
        EXPECT_EQ(report.fields.at("producer_pc"), std::format("{:#x}", producer_pc));
        EXPECT_EQ(report.fields.at("consumer_pc"), std::format("{:#x}", consumer_pc));
        EXPECT_EQ(report.fields.at("access"), write ? "write" : "read");
        EXPECT_EQ(report.fields.at("type"), vector                       ? "VGPR"
                                            : producer == Producer::Ttmp ? "TTMP"
                                                                         : "SGPR");
        EXPECT_EQ(report.fields.at("lane"), vector ? "2" : "-1");
        EXPECT_EQ(report.fields.at("reg"), producer == Producer::Ttmp ? "0" : "4");
        EXPECT_EQ(report.fields.at("wave"), "0");
        EXPECT_EQ(report.fields.at("wg"), "0,0,0");
        EXPECT_EQ(report.fields.at("checker"), "core");
        EXPECT_NE(report.trace.find(std::format("==>  {:#x}", producer_pc)), std::string::npos);
        EXPECT_NE(report.trace.find(std::format("==>  {:#x}", consumer_pc)), std::string::npos);
        EXPECT_NE(report.trace.find(vector ? "v_mov_b32" : "s_mov_b32"), std::string::npos);
        EXPECT_NE(h.race->getSummary().find("  1 race(s) detected"), std::string::npos);
        ASSERT_EQ(h.probe->hazards.size(), 1u);
        EXPECT_EQ(report.fields.at("counter"), producer == Producer::Global ? "vmcnt" : "lgkmcnt");
        EXPECT_EQ(report.fields.at("required_wait"), "0");
        EXPECT_EQ(report.fields.at("width"), "1");
        EXPECT_EQ(report.fields.at("bytes"), "0xf");
        EXPECT_EQ(report.fields.at("lanes"), vector ? "0x4" : "0xffffffffffffffff");
        const auto first = std::ranges::find(h.probe->phases, consumer_pc,
                                             &std::pair<uint64_t, std::string>::first);
        ASSERT_NE(first, h.probe->phases.end());
        EXPECT_EQ(first->second, "hazard");
        ASSERT_NE(first + 1, h.probe->phases.end());
        EXPECT_EQ((first + 1)->second, "before");
      }
}

TEST(RegisterHazardPluginTest, DebuggerReadsDoNotReportOrRetirePendingLoads) {
  for (bool scalar : {false, true}) {
    SCOPED_TRACE(scalar);
    Program program;
    program.setup();
    const auto producer = scalar ? program.emit(cdna4::build_smem(
                                       cdna4::kSLoadDwordSmem, {.sbase = 0, .sdata = 4, .imm = 1}))
                                 : program.global_load();
    const auto inspect = program.emit(cdna4::build_sopp(cdna4::kSNopSopp));
    const auto consumer =
        scalar ? program.emit(cdna4::build_sop1(cdna4::kSMovB32Sop1, {.ssrc0 = 4, .sdst = 8}))
               : program.vector_read();
    program.end();
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize());
    const uint32_t value = 42;
    h.fixture.mem->load_image(reinterpret_cast<const uint8_t *>(&value), sizeof(value), 0x2000);
    Wavefront *wave = nullptr;
    h.probe->dispatch = [&](std::span<Wavefront *> waves) {
      ASSERT_EQ(waves.size(), 1u);
      wave = waves.front();
    };
    bool inspected = false;
    h.probe->before = [&](uint64_t pc) {
      if (pc != inspect)
        return;
      ASSERT_NE(wave, nullptr);
      auto &state = wave->ensure_memory_wait_scoreboard();
      ASSERT_EQ(state.events().size(), 1u);
      EXPECT_TRUE(h.probe->hazards.empty());
      const auto saved_pc = wave->pc;
      for (uint64_t inspection_pc : {producer, inspect}) {
        wave->pc = inspection_pc;
        EXPECT_EQ(scalar ? wave->debug_read_sgpr(4) : wave->debug_read_vgpr(4, 2), value);
        EXPECT_TRUE(h.probe->hazards.empty());
        EXPECT_EQ(state.events().size(), 1u);
        EXPECT_EQ(state.outstanding(scalar ? WaitCounterKind::Ds : WaitCounterKind::Load), 1u);
      }
      wave->pc = saved_pc;
      inspected = true;
    };
    ASSERT_NO_FATAL_FAILURE(h.run(program.words));
    EXPECT_TRUE(inspected);
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), 1u) << h.sink->str();
    EXPECT_EQ(output.front().fields.at("producer_pc"), std::format("{:#x}", producer));
    EXPECT_EQ(output.front().fields.at("consumer_pc"), std::format("{:#x}", consumer));
  }
}

TEST(RegisterHazardPluginTest, ReadModifyWriteAndLaterUseDoNotRetireTheProducer) {
  for (bool waited : {false, true}) {
    Program program;
    program.setup();
    const auto producer = program.global_load();
    const auto rmw = program.vector_read(4);
    if (waited)
      program.emit(cdna4::build_sopp(cdna4::kSWaitcntSopp, {.simm16 = 0x0f70}));
    const auto later = program.vector_read();
    program.end();
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize());
    ASSERT_NO_FATAL_FAILURE(h.run(program.words));
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), waited ? 2u : 3u) << h.sink->str();
    for (const auto &report : output)
      EXPECT_EQ(report.fields.at("producer_pc"), std::format("{:#x}", producer));
    EXPECT_EQ(output[0].fields.at("consumer_pc"), std::format("{:#x}", rmw));
    EXPECT_EQ(output[1].fields.at("consumer_pc"), std::format("{:#x}", rmw));
    EXPECT_EQ(output[0].fields.at("access"), "read");
    EXPECT_EQ(output[1].fields.at("access"), "write");
    if (!waited) {
      EXPECT_EQ(output[2].fields.at("consumer_pc"), std::format("{:#x}", later));
      EXPECT_EQ(output[2].fields.at("access"), "read");
    }
  }
}

TEST(RegisterHazardPluginTest, DistinctConsumersSurviveSiteSuppressionAndWarningBudget) {
  for (bool warnings : {false, true}) {
    Program program;
    program.setup();
    const auto producer = program.global_load();
    std::vector<uint64_t> consumers;
    for (unsigned i = 0; i < 20; ++i)
      consumers.push_back(program.vector_read());
    program.end();
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize(warnings));
    testing::internal::CaptureStderr();
    h.run(program.words);
    const auto stderr_output = testing::internal::GetCapturedStderr();
    ASSERT_FALSE(testing::Test::HasFatalFailure());
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), consumers.size()) << h.sink->str();
    for (unsigned i = 0; i < output.size(); ++i) {
      EXPECT_EQ(output[i].fields.at("producer_pc"), std::format("{:#x}", producer));
      EXPECT_EQ(output[i].fields.at("consumer_pc"), std::format("{:#x}", consumers[i]));
    }
    EXPECT_EQ(h.fixture.cu()->memory_wait_diagnostic_count(), warnings ? 20u : 0u);
    EXPECT_EQ(stderr_output.find("further diagnostics on this CU are suppressed") !=
                  std::string::npos,
              warnings);
    if (!warnings)
      EXPECT_EQ(stderr_output.find("memory-wait:"), std::string::npos);
    EXPECT_NE(h.race->getSummary().find("  20 race(s) detected"), std::string::npos);
  }
}

TEST(RegisterHazardPluginTest, RepeatedSiteIsSuppressedWithoutRetiringTheProducer) {
  Program program;
  program.setup();
  program.global_load();
  program.emit(cdna4::build_sop1(cdna4::kSMovB32Sop1, {.ssrc0 = 130, .sdst = 2}));
  const auto first = program.vector_read();
  program.emit(cdna4::build_sop2(cdna4::kSSubU32Sop2, {.ssrc0 = 2, .ssrc1 = 129, .sdst = 2}));
  program.emit(cdna4::build_sopc(cdna4::kSCmpLgU32Sopc, {.ssrc0 = 2, .ssrc1 = 128}));
  program.emit(cdna4::build_sopp(cdna4::kSCbranchScc1Sopp, {.simm16 = uint16_t(-4)}));
  const auto second = program.vector_read(9);
  program.end();
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize());
  ASSERT_NO_FATAL_FAILURE(h.run(program.words));
  const auto output = reports(h.sink->str());
  ASSERT_EQ(output.size(), 2u) << h.sink->str();
  EXPECT_EQ(output[0].fields.at("consumer_pc"), std::format("{:#x}", first));
  EXPECT_EQ(output[1].fields.at("consumer_pc"), std::format("{:#x}", second));
  EXPECT_EQ(h.probe->hazards.size(), 3u);
  EXPECT_NE(h.race->getSummary().find("1 repeated register finding(s) suppressed"),
            std::string::npos);
}

TEST(RegisterHazardPluginTest, DistinctProducersAtOneConsumerAreNotSuppressed) {
  Program program;
  program.setup();
  program.exec(1);
  const auto first = program.global_load();
  program.exec(2);
  const auto second = program.global_load();
  program.exec(3);
  const auto consumer = program.vector_read();
  program.end();
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize());
  ASSERT_NO_FATAL_FAILURE(h.run(program.words));
  const auto output = reports(h.sink->str());
  ASSERT_EQ(output.size(), 2u) << h.sink->str();
  EXPECT_EQ(output[0].fields.at("producer_pc"), std::format("{:#x}", first));
  EXPECT_EQ(output[1].fields.at("producer_pc"), std::format("{:#x}", second));
  EXPECT_EQ(output[0].fields.at("lanes"), "0x1");
  EXPECT_EQ(output[1].fields.at("lanes"), "0x2");
  for (const auto &report : output)
    EXPECT_EQ(report.fields.at("consumer_pc"), std::format("{:#x}", consumer));
}

TEST(RegisterHazardPluginTest, ObserverOrderDoesNotChangeFindings) {
  Program program;
  program.setup();
  program.global_load();
  program.vector_read();
  program.end();
  std::array<std::vector<Report>, 2> output;
  for (bool probe_first : {false, true}) {
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize(false, probe_first));
    ASSERT_NO_FATAL_FAILURE(h.run(program.words));
    output[probe_first] = reports(h.sink->str());
  }
  ASSERT_EQ(output[0].size(), 1u);
  ASSERT_EQ(output[1].size(), 1u);
  EXPECT_EQ(output[0][0].fields, output[1][0].fields);
  EXPECT_EQ(output[0][0].trace, output[1][0].trace);
}

TEST(RegisterHazardPluginTest, RaceAloneEnablesCoreCheckingWithWarningsOff) {
  Program program;
  program.setup();
  program.global_load();
  program.vector_read();
  program.end();
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize(false, false, "cdna4", false));
  EXPECT_FALSE(h.probe->observes_register_hazards());
  ASSERT_NO_FATAL_FAILURE(h.run(program.words));
  EXPECT_EQ(reports(h.sink->str()).size(), 1u);
  EXPECT_EQ(h.fixture.cu()->memory_wait_diagnostic_count(), 0u);
}

TEST(RegisterHazardPluginTest, FindingsPrecedeActualAsyncIssueNotification) {
  if (!async_mma_policy::candidate("v_wmma_f32_32x16x128_f4"))
    GTEST_SKIP() << "Migrated matrix handlers require issuer-owned error reporting.";
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(1, "cdna5", 32, 128, 256, 1, 4, "off"));
  fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<HazardProbe>(true);
  auto *probe = observer.get();
  probe->dispatch = [](std::span<Wavefront *> waves) {
    ASSERT_EQ(waves.size(), 1u);
    // Seed a pending WMMA source to isolate issuer/callback ordering from
    // memory execution. No encoding is claimed for this synthetic producer.
    auto &scoreboard = waves.front()->ensure_memory_wait_scoreboard();
    scoreboard.add({scoreboard.issue(WaitCounterKind::Load),
                    0x800,
                    0xffffffff,
                    {RegClass::VGPR, 0, 1},
                    WaitCounterKind::Load,
                    0xf});
  };
  ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
  fixture.soc->set_plugin_group(fixture.plugin_group_);
  fixture.plugin_group_->onInit();

  Program program;
  for (uint8_t dst : {uint8_t{64}, uint8_t{96}})
    program.emit(cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                                    {.vdst = dst,
                                     .src0 = 256,
                                     .src1 = 288,
                                     .src2 = static_cast<uint16_t>(256 + dst),
                                     .opsel_hi = 3}));
  program.emit(cdna5::build_sopp(cdna5::kSEndpgmSopp));
  fixture.run_kernel(program.words.data(), program.words.size(), 32, 32);
  fixture.shutdown();
  ASSERT_EQ(probe->completed, 1u);
  const auto issued =
      std::ranges::find(probe->phases, "async", &std::pair<uint64_t, std::string>::second);
  ASSERT_NE(issued, probe->phases.end());
  const auto first =
      std::ranges::find(probe->phases, issued->first, &std::pair<uint64_t, std::string>::first);
  ASSERT_LT(first, issued);
  EXPECT_EQ(first->second, "hazard");
  const auto finding =
      std::ranges::find(probe->hazards, issued->first, &RegisterHazardObservation::consumer_pc);
  ASSERT_NE(finding, probe->hazards.end());
  ASSERT_TRUE(finding->consumer_instruction);
  EXPECT_EQ(finding->consumer_instruction->pc(), issued->first);
  EXPECT_EQ(fixture.cu()->memory_wait_diagnostic_count(), 0u);
}

TEST(RegisterHazardPluginTest, RaceReportsAndTraceSurviveMmaWithRequestedHelpers) {
  constexpr uint32_t packed_ones = 0x22222222; // Eight FP4 values of 1.0.
  for (uint64_t exec : {0xffffffffULL, 0x80ULL, 0ULL})
    for (bool waited : {false, true}) {
      Program program;
      const uint64_t producer = program.emit(cdna5::build_vglobal(
          cdna5::kGlobalLoadB32Vglobal, {.saddr = 124, .vdst = 0, .vaddr = 120}));
      if (waited)
        program.emit(cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
      // Offload requires full EXEC. The load keeps its original lane mask,
      // so the following MMA can still consume a partially pending source.
      program.emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 193, .sdst = 126}));
      std::array<uint64_t, 3> consumers;
      for (unsigned i = 0; i < 2; ++i) {
        const uint8_t dst = i ? 96 : 64;
        consumers[i] = program.emit(cdna5::build_vop3p(cdna5::kVWmmaF3232x16x128F4Vop3p,
                                                       {.vdst = dst,
                                                        .src0 = 256,
                                                        .src1 = 288,
                                                        .src2 = static_cast<uint16_t>(256 + dst),
                                                        .opsel_hi = 3}));
      }
      consumers[2] =
          program.emit(cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 256, .vdst = 150}));
      program.emit(cdna5::build_sopp(cdna5::kSEndpgmSopp));

      std::vector<std::string> reference_traces;
      for (uint32_t helpers : {0, 4})
        for (bool probe_first : {false, true}) {
          SCOPED_TRACE(std::format("exec={:#x} waited={} helpers={} probe_first={}", exec, waited,
                                   helpers, probe_first));
          DiagnosticHarness h;
          ASSERT_NO_FATAL_FAILURE(h.initialize(false, probe_first, "cdna5", true, helpers));
          h.fixture.mem->load_image(reinterpret_cast<const uint8_t *>(&packed_ones),
                                    sizeof(packed_ones), 0x2000);
          h.probe->dispatch = [exec](std::span<Wavefront *> waves) {
            ASSERT_EQ(waves.size(), 1u);
            auto &wave = *waves.front();
            wave.set_exec(exec);
            const RegisterAccess registers(wave);
            for (uint32_t reg = 0; reg < 256; ++reg)
              for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
                registers.write_vgpr(wave.vgpr_alloc().base + reg, lane,
                                     reg < 48     ? packed_ones
                                     : reg == 120 ? 0x2000
                                                  : 0);
          };
          bool halted = false;
          h.probe->halt = [&](Wavefront &wave) {
            halted = true;
            // Halt follows helper completion; instruction-issue callbacks cannot
            // safely inspect these registers while an async job may write them.
            const RegisterAccess registers(wave);
            for (uint32_t base : {64, 96})
              for (uint32_t reg = base; reg < base + 16; ++reg)
                for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
                  EXPECT_EQ(registers.read_vgpr(wave.vgpr_alloc().base + reg, lane),
                            std::bit_cast<uint32_t>(128.0f))
                      << "reg=" << reg << " lane=" << lane;
          };
          ASSERT_NO_FATAL_FAILURE(h.run(program.words, 32, 32));
          ASSERT_TRUE(halted);
          const auto async_issues = std::ranges::count(h.probe->phases, "async",
                                                       &std::pair<uint64_t, std::string>::second);
          if (helpers && async_mma_policy::candidate("v_wmma_f32_32x16x128_f4"))
            EXPECT_GT(async_issues, 0);
          else
            EXPECT_EQ(async_issues, 0);
          EXPECT_EQ(h.fixture.cu()->memory_wait_diagnostic_count(), 0u);

          const auto output = reports(h.sink->str());
          ASSERT_EQ(output.size(), !waited && exec ? 3u : 0u) << h.sink->str();
          EXPECT_EQ(h.probe->hazards.size(), output.size());
          for (size_t i = 0; i < output.size(); ++i) {
            const auto &report = output[i];
            EXPECT_EQ(report.fields.at("producer_pc"), std::format("{:#x}", producer));
            EXPECT_EQ(report.fields.at("consumer_pc"), std::format("{:#x}", consumers[i]));
            EXPECT_EQ(report.fields.at("type"), "VGPR");
            EXPECT_EQ(report.fields.at("access"), "read");
            EXPECT_EQ(report.fields.at("reg"), "0");
            EXPECT_EQ(report.fields.at("width"), "1");
            EXPECT_EQ(report.fields.at("lanes"), std::format("{:#x}", exec));
            EXPECT_EQ(report.fields.at("bytes"), "0xf");
            EXPECT_EQ(report.fields.at("counter"), "loadcnt");
            EXPECT_EQ(report.fields.at("required_wait"), "0");
            const auto first = std::ranges::find(h.probe->phases, consumers[i],
                                                 &std::pair<uint64_t, std::string>::first);
            ASSERT_NE(first, h.probe->phases.end());
            EXPECT_EQ(first->second, "hazard");
            ASSERT_NE(first + 1, h.probe->phases.end());
            EXPECT_EQ((first + 1)->first, consumers[i]);
            EXPECT_TRUE((first + 1)->second == "before" || (first + 1)->second == "async");
            if (!helpers && !probe_first)
              reference_traces.push_back(report.trace);
            else
              EXPECT_EQ(report.trace, reference_traces.at(i));
          }
          if (!output.empty())
            for (unsigned i = 0; i < 2; ++i) {
              const auto line = std::format("{:#x}  v_wmma_f32_", consumers[i]);
              const auto &trace = output.back().trace;
              const auto pos = trace.find(line);
              ASSERT_NE(pos, std::string::npos) << trace;
              EXPECT_EQ(trace.find(line, pos + line.size()), std::string::npos) << trace;
            }
        }
    }
}

TEST(RegisterHazardPluginTest, TraceEvictionAndCodeMutationPreserveOwnedProducerEvidence) {
  Program program;
  program.setup();
  const auto producer = program.global_load();
  const auto rewrite_at = program.emit(cdna4::build_sopp(cdna4::kSNopSopp));
  for (unsigned i = 0; i < 300; ++i)
    program.emit(cdna4::build_sopp(cdna4::kSNopSopp));
  const auto consumer = program.vector_read();
  program.end();
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize());
  h.probe->before = [&](uint64_t pc) {
    if (pc == rewrite_at) {
      const auto replacement = cdna4::build_sopp(cdna4::kSNopSopp, {.simm16 = 7});
      h.fixture.mem->load_image(reinterpret_cast<const uint8_t *>(replacement.data()),
                                sizeof(replacement), producer);
    }
  };
  ASSERT_NO_FATAL_FAILURE(h.run(program.words));
  const auto output = reports(h.sink->str());
  ASSERT_EQ(output.size(), 1u);
  EXPECT_NE(output[0].trace.find("global_load_dword"), std::string::npos);
  EXPECT_NE(output[0].trace.find("(before trace window)"), std::string::npos);
  EXPECT_NE(output[0].trace.find(std::format("==>  {:#x}", consumer)), std::string::npos);
  ASSERT_EQ(h.probe->hazards.size(), 1u);
  const auto &hazard = h.probe->hazards.front();
  ASSERT_TRUE(hazard.producer_instruction);
  ASSERT_TRUE(hazard.consumer_instruction);
  EXPECT_EQ(hazard.producer_instruction->pc(), producer);
  EXPECT_EQ(hazard.consumer_instruction->pc(), consumer);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  util::StringDiagnostic error;
  const auto decoded =
      decoder->decode_window(hazard.producer_instruction->encoding(), producer, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  EXPECT_EQ(decoded.value()->mnemonic(), "global_load_dword");
}

TEST(RegisterHazardPluginTest, MixedFlatWaitCompletesOnlyItsRoutedLanes) {
  for (bool local : {false, true})
    for (bool waited : {false, true}) {
      SCOPED_TRACE(std::format("local={} waited={}", local, waited));
      Program program;
      // Work-item IDs start in v0. Lanes 0 and 1 use global and shared apertures.
      program.emit(
          cdna4::build_vop2(cdna4::kVLshlrevB32Vop2, {.src0 = 144, .vsrc1 = 0, .vdst = 1}));
      program.emit(cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 255, .vdst = 0}));
      program.emit(std::array{0x2000u});
      program.exec(3);
      program.emit(
          cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.addr = 0, .saddr = 0x7f, .vdst = 4}));
      if (waited)
        program.emit(cdna4::build_sopp(cdna4::kSWaitcntSopp,
                                       {.simm16 = static_cast<uint16_t>(local ? 0xc07f : 0x0f70)}));
      program.exec(local ? 2 : 1);
      program.vector_read();
      program.end();
      DiagnosticHarness h;
      ASSERT_NO_FATAL_FAILURE(h.initialize());
      h.fixture.cu()->set_apertures(uint64_t{1} << 48, (uint64_t{1} << 48) + UINT32_MAX, 0, 0);
      ASSERT_NO_FATAL_FAILURE(h.run(program.words, 64, 64, 16384));
      const auto output = reports(h.sink->str());
      ASSERT_EQ(output.size(), !waited ? 1u : 0u) << h.sink->str();
      if (!waited) {
        EXPECT_EQ(output[0].fields.at("lanes"), local ? "0x2" : "0x1");
        EXPECT_EQ(output[0].fields.at("counter"), local ? "lgkmcnt" : "vmcnt");
      }
    }
}

TEST(RegisterHazardPluginTest, SpecialRegisterResultsRequireTheirWait) {
  for (bool waited : {false, true}) {
    Program program;
    program.emit(
        cdna5::build_sop1(cdna5::kSSendmsgRtnB32Sop1, {.ssrc0 = 128, .sdst = kModernM0Selector}));
    if (waited)
      program.emit(cdna5::build_sopp(cdna5::kSWaitKmcntSopp, {.simm16 = 0}));
    program.emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = kModernM0Selector, .sdst = 8}));
    program.emit(cdna5::build_sopp(cdna5::kSEndpgmSopp));
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize(false, false, "cdna5"));
    ASSERT_NO_FATAL_FAILURE(h.run(program.words, 32, 32));
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), !waited ? 1u : 0u) << h.sink->str();
    if (!waited) {
      EXPECT_EQ(output[0].fields.at("type"), "M0");
      EXPECT_EQ(output[0].fields.at("lane"), "-1");
      EXPECT_EQ(output[0].fields.at("counter"), "kmcnt");
    }
  }
}

TEST(RegisterHazardPluginTest, EmbeddedWaitIsAppliedBeforeCoreConsumerChecking) {
  for (bool waited : {false, true}) {
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize(false, false, "rdna3"));
    h.probe->dispatch = [&](std::span<Wavefront *> waves) {
      ASSERT_EQ(waves.size(), 1u);
      auto &wave = *waves.front();
      // Seed an inline producer to isolate embedded-wait handling.
      const auto words = rdna3::build_ldsdir(rdna3::kLdsDirectLoadLdsdir, {.vdst = 4});
      const InstructionSnapshot producer(0x800, words);
      auto &scoreboard = wave.ensure_memory_wait_scoreboard();
      MemoryWaitScoreboard::Event event{scoreboard.issue(WaitCounterKind::Exp),
                                        0x800,
                                        0xffffffff,
                                        {RegClass::VGPR, 4, 1},
                                        WaitCounterKind::Exp,
                                        0xf};
      event.instruction = producer;
      scoreboard.add(event);
    };
    Program program;
    program.emit(rdna3::build_vinterp(rdna3::kVInterpP10F32Vinterp,
                                      {.vdst = 8,
                                       .wait_exp = static_cast<uint8_t>(waited ? 0 : 7),
                                       .src0 = 260,
                                       .src1 = 261,
                                       .src2 = 262}));
    program.emit(rdna3::build_sopp(rdna3::kSEndpgmSopp));
    ASSERT_NO_FATAL_FAILURE(h.run(program.words, 32, 32));
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), waited ? 0u : 1u) << h.sink->str();
    if (!waited)
      EXPECT_EQ(output[0].fields.at("counter"), "expcnt");
  }
}

TEST(RegisterHazardPluginTest, CrossWaveLdsChecksRequireTheBarrier) {
  for (bool barrier : {false, true}) {
    Program program;
    program.setup();
    program.emit(cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 0, .data0 = 1}));
    program.emit(cdna4::build_sopp(cdna4::kSWaitcntSopp, {.simm16 = 0xc07f}));
    if (barrier)
      program.emit(cdna4::build_sopp(cdna4::kSBarrierSopp));
    program.emit(cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 4}));
    program.emit(cdna4::build_sopp(cdna4::kSWaitcntSopp, {.simm16 = 0xc07f}));
    program.end();
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize());
    ASSERT_NO_FATAL_FAILURE(h.run(program.words, 128, 128, 16384));
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), barrier ? 0u : 1u) << h.sink->str();
    if (!barrier) {
      EXPECT_EQ(output[0].fields.at("type"), "LDS");
      EXPECT_EQ(output[0].fields.at("checker"), "plugin");
      EXPECT_NE(output[0].trace.find("ds_write_b32"), std::string::npos);
      EXPECT_NE(output[0].trace.find("ds_read_b32"), std::string::npos);
    }
  }
}

TEST(RegisterHazardPluginTest, CounterOnlyIssuesParticipateInLdsPartialWaits) {
  for (bool waited : {false, true}) {
    DiagnosticHarness h;
    ASSERT_NO_FATAL_FAILURE(h.initialize());
    auto *wave = h.fixture.cu()->dispatch_wf(0, 0x100, 104, 256);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(1);
    std::array<Wavefront *, 1> waves{wave};
    h.fixture.plugin_group_->onAmdgpuWorkgroupDispatched(1, 0, waves);
    for (bool lds : {true, false}) {
      auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
      state->is_load = true;
      state->lds_dst = lds;
      state->lds_base = 0x80;
      state->num_elems = 1;
      state->elem_size = 4;
      state->exec_mask = state->lane_mask = 1;
      state->wf_size = 64;
      state->dst_reg_base = wave->vgpr_alloc().base + 4;
      state->wait_counter_type = WaitCounterType::VMCNT;
      TestMemoryInstruction instruction(std::move(state));
      h.fixture.plugin_group_->onAmdgpuBeforeExecuteInstruction(wave->pc, instruction, *wave, {});
      h.fixture.plugin_group_->onAmdgpuMemoryAccessRouted({}, instruction, *wave);
      wave->pc += 4;
    }
    auto *state = static_cast<RaceWavefrontState *>(wave->plugin_state(h.race->slot_index()));
    auto &race = *state->race_state;
    const auto &events = race.getWaveMemoryEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(race.getDetector()->events().access(events.back()), LdsAccess::None);
    EXPECT_TRUE(race.getDetector()->events().ldsIntervals(events.back()).empty());
    if (waited) {
      race.applyWaitCounter(WaitCounterType::VMCNT, 1);
    }
    race.getDetector()->validateRead(0x80, race.getWaveId(), 0, 4, MemoryOrderClass::UNORDERED);
    const auto output = reports(h.sink->str());
    ASSERT_EQ(output.size(), waited ? 0u : 1u) << h.sink->str();
    if (!waited)
      EXPECT_EQ(output.front().fields.at("type"), "LDS");
  }
}

TEST(RegisterHazardPluginTest, ReplayFindingsRespectTheExplicitXcntSetting) {
  for (bool enabled : {false, true})
    for (bool waited : {false, true}) {
      DiagnosticHarness h;
      ASSERT_NO_FATAL_FAILURE(h.initialize(false, false, "cdna5"));
      h.fixture.cu()->restore_wait_diagnostics(MemoryWaitDiagnostics::Off,
                                               enabled ? MemoryWaitDiagnostics::Warn
                                                       : MemoryWaitDiagnostics::Off);
      Program program;
      program.emit(cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = 128, .sdst = 0}));
      program.emit(cdna5::build_smem(cdna5::kSLoadB32Smem,
                                     {.sbase = 0, .sdata = 4, .soffset = kModernNullSelector}));
      if (waited)
        program.emit(cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = 0}));
      program.emit(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 129, .sdst = 0}));
      program.emit(cdna5::build_sopp(cdna5::kSEndpgmSopp));
      ASSERT_NO_FATAL_FAILURE(h.run(program.words, 32, 32));
      const auto output = reports(h.sink->str());
      ASSERT_EQ(output.size(), enabled && !waited ? 1u : 0u) << h.sink->str();
      EXPECT_EQ(h.fixture.cu()->xcnt_diagnostic_count(), enabled && !waited ? 1u : 0u);
      if (enabled && !waited) {
        EXPECT_EQ(output.front().fields.at("counter"), "xcnt");
        EXPECT_EQ(output.front().fields.at("access"), "write");
        EXPECT_EQ(output.front().fields.at("type"), "SGPR");
        EXPECT_EQ(output.front().fields.at("reg"), "0");
      }
    }
}

TEST(RegisterHazardPluginTest, MissingEvidenceKeepsFindingAndLogicalWaveContext) {
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize());
  auto *cu = h.fixture.cu();
  ASSERT_NE(cu->dispatch_wf(0, 0, 104, 256), nullptr);
  auto *wave = cu->dispatch_wf(5, 0, 104, 256);
  ASSERT_NE(wave, nullptr);
  ASSERT_NE(wave->wf_id(), 0u);
  wave->set_dispatch_id(17);
  wave->set_wg_coord(2, 3, 4);
  std::array<Wavefront *, 1> waves{wave};
  h.fixture.plugin_group_->onAmdgpuWorkgroupDispatched(17, 5, waves);
  KernelDispatchInfo info{};
  info.dispatch_id = 17;
  info.kernel_name = "racy_kernel";
  info.kernel_symbol = "_Z11racy_kernelv";
  h.fixture.plugin_group_->onAmdgpuDispatchPacketProcessed(info);
  const RegisterHazardObservation hazard{0x100,        0x200,       {RegClass::VGPR, 4, 2}, 0x40,
                                         0x6,          false,       WaitCounterKind::Load,  0,
                                         std::nullopt, std::nullopt};
  h.fixture.plugin_group_->onAmdgpuRegisterHazard(hazard, *wave);
  const auto output = reports(h.sink->str());
  ASSERT_EQ(output.size(), 1u);
  EXPECT_EQ(output[0].fields.at("kernel"), "racy_kernel");
  EXPECT_EQ(output[0].fields.at("symbol"), "_Z11racy_kernelv");
  EXPECT_EQ(output[0].fields.at("dispatch"), "17");
  EXPECT_EQ(output[0].fields.at("wave"), "0");
  EXPECT_EQ(output[0].fields.at("lane"), "6");
  EXPECT_EQ(output[0].fields.at("wg"), "2,3,4");
  EXPECT_EQ(output[0].fields.at("width"), "2");
  EXPECT_EQ(output[0].fields.at("bytes"), "0x6");
  EXPECT_NE(output[0].trace.find("0x100  (encoding unavailable)"), std::string::npos);
  EXPECT_NE(output[0].trace.find("0x200  (encoding unavailable)"), std::string::npos);
}

TEST(RegisterHazardPluginTest, GroupHonorsHazardInterestAndWaveSubscription) {
  DiagnosticHarness h;
  ASSERT_NO_FATAL_FAILURE(h.initialize());
  auto *wave = h.fixture.cu()->dispatch_wf(0, 0, 104, 256);
  ASSERT_NE(wave, nullptr);
  ExecutionPluginGroup group(PluginSinkConfig{});
  EXPECT_FALSE(group.observes_register_hazards());
  std::array<HazardProbe *, 3> probes;
  for (unsigned i = 0; i < probes.size(); ++i) {
    auto probe = std::make_unique<HazardProbe>(i != 0, std::format("probe{}", i));
    probe->subscribed = i != 1;
    probes[i] = probe.get();
    ASSERT_TRUE(group.add(std::move(probe)));
  }
  EXPECT_TRUE(group.observes_register_hazards());
  group.onAmdgpuWavefrontDispatched(*wave);
  const RegisterHazardObservation hazard{
      0x100,        0x200,       {RegClass::SGPR, 4, 1}, ~uint64_t{0},
      0xf,          false,       WaitCounterKind::Ds,    0,
      std::nullopt, std::nullopt};
  group.onAmdgpuRegisterHazard(hazard, *wave);
  EXPECT_TRUE(probes[0]->hazards.empty());
  EXPECT_TRUE(probes[1]->hazards.empty());
  EXPECT_EQ(probes[2]->hazards.size(), 1u);
}

} // namespace
