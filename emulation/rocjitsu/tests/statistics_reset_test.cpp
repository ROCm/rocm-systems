// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/config_common.h"
#include "rocjitsu/stats/statistics.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/soc.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu;

GpuPciDeviceSpec statistics_reset_spec() {
  config::KfdDeviceConfig config;
  config.gfx_target_version = 120500;
  config.vendor_id = 0x1002;
  config.device_id = 0x1250;
  config.local_mem_size = 16ULL * 1024 * 1024;
  config.num_shader_engines = 2;
  config.num_shader_arrays_per_engine = 2;
  config.num_cu_per_sh = 8;
  config.wave_front_size = 32;
  config.max_waves_per_simd = 16;
  config.max_slots_scratch_cu = 32;
  config.lds_size_kb = 320;
  return gpu_pci_spec_from_config(config, {});
}

class StatisticsResetGpu {
public:
  explicit StatisticsResetGpu(std::string name)
      : memory(name + "_memory"), soc(name + "_soc", &memory, ROCJITSU_CODE_ARCH_CDNA5),
        device(std::move(name), statistics_reset_spec(), nullptr, &soc) {}

  void bind(const std::shared_ptr<StatisticsSession> &session, uint32_t gpu_id) {
    session->register_gpu(gpu_id);
    soc.set_statistics(session, gpu_id);
  }

  amdgpu::GpuMemory memory;
  SoC soc;
  GpuPciDevice device;
};

class StatisticsResetTest : public testing::Test {
protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 ("rocjitsu-statistics-reset-test-" +
                  std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  std::shared_ptr<StatisticsSession> create_session(bool filtered = false) {
    StatisticsConfig config{.enabled = true, .directory = directory_.string()};
    if (filtered) {
      config.include = {"excluded.metric"};
      config.scopes.clear();
    }
    auto result = StatisticsSession::create(config, "node", "functional");
    EXPECT_TRUE(result.succeeded());
    return result.succeeded() ? std::move(result).value() : nullptr;
  }
  void finish(const std::shared_ptr<StatisticsSession> &session) {
    session->mark_run_complete();
    session->finish();
    std::ifstream file(session->output_path());
    ASSERT_TRUE(file.is_open());
    output_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    std::istringstream lines(output_);
    std::string first;
    ASSERT_TRUE(static_cast<bool>(std::getline(lines, first)));
    ASSERT_TRUE(plugin_detail::flexbuffer_from_json(first, run_));
    EXPECT_EQ(run()["record"].AsString().str(), "run");
  }
  flexbuffers::Map run() const { return flexbuffers::GetRoot(run_.GetBuffer()).AsMap(); }

  static void expect_reset(const flexbuffers::Map &gpu, const char *kind, uint64_t attempts,
                           uint64_t completed, uint64_t failed) {
    const auto counts = gpu["resets"].AsMap()[kind].AsMap();
    EXPECT_EQ(counts["attempts"].AsString().str(), std::to_string(attempts));
    EXPECT_EQ(counts["completed"].AsString().str(), std::to_string(completed));
    EXPECT_EQ(counts["failed"].AsString().str(), std::to_string(failed));
    EXPECT_FALSE(counts["overflow"].AsBool());
  }

  std::filesystem::path directory_;
  std::string output_;
  flexbuffers::Builder run_;
};

TEST_F(StatisticsResetTest, PciResetPreservesHandlesAndTotalsWithIndependentGpuMetadata) {
  auto session = create_session();
  ASSERT_NE(session, nullptr);
  StatisticsResetGpu first("first");
  StatisticsResetGpu second("second");
  ASSERT_TRUE(first.device.usable());
  ASSERT_TRUE(second.device.usable());
  first.bind(session, 0);
  second.bind(session, 1);
  auto first_scope = session->scope("memory", 0);
  auto second_scope = session->scope("memory", 1);
  ASSERT_NE(first_scope, nullptr);
  ASSERT_NE(second_scope, nullptr);
  auto first_counter = first_scope->counter("requests", "requests", "Observed requests");
  auto second_counter = second_scope->counter("requests", "requests", "Observed requests");
  first_counter->add(7);
  second_counter->add(11);
  ASSERT_NO_THROW(first.device.reset(simdojo::ResetKind::FunctionLevel));
  EXPECT_EQ(session->scope("memory", 0), first_scope);
  EXPECT_EQ(first_scope->counter("requests", "requests", "Observed requests"), first_counter);
  EXPECT_EQ(first_counter->value(), 7u);
  EXPECT_EQ(second_counter->value(), 11u);
  first_counter->add(5);
  ASSERT_NO_THROW(first.device.reset(simdojo::ResetKind::Bus));
  ASSERT_NO_THROW(second.device.reset(simdojo::ResetKind::LostConnection));
  EXPECT_EQ(first_counter->value(), 12u);
  EXPECT_EQ(second_counter->value(), 11u);
  ASSERT_NO_FATAL_FAILURE(finish(session));
  EXPECT_EQ(run()["accumulation_scope"].AsString().str(), "session");
  const auto devices = run()["devices"].AsVector();
  ASSERT_EQ(devices.size(), 2u);
  EXPECT_EQ(devices[0].AsMap()["gpu_id"].AsUInt32(), 0u);
  EXPECT_EQ(devices[1].AsMap()["gpu_id"].AsUInt32(), 1u);
  expect_reset(devices[0].AsMap(), "function_level", 1, 1, 0);
  expect_reset(devices[0].AsMap(), "bus", 1, 1, 0);
  expect_reset(devices[0].AsMap(), "lost_connection", 0, 0, 0);
  expect_reset(devices[1].AsMap(), "function_level", 0, 0, 0);
  expect_reset(devices[1].AsMap(), "bus", 0, 0, 0);
  expect_reset(devices[1].AsMap(), "lost_connection", 1, 1, 0);
  EXPECT_NE(output_.find("\"value\":\"12\""), std::string::npos);
  EXPECT_NE(output_.find("\"value\":\"11\""), std::string::npos);
  EXPECT_EQ(output_.find("\"snapshot_id\":\"2\""), std::string::npos);
  EXPECT_NE(output_.find("\"epoch\":\"0\""), std::string::npos);
}

TEST_F(StatisticsResetTest, GpuMetadataSurvivesAllMetricAndScopeFilters) {
  auto session = create_session(true);
  ASSERT_NE(session, nullptr);
  StatisticsResetGpu first("first");
  StatisticsResetGpu second("second");
  ASSERT_TRUE(first.device.usable());
  ASSERT_TRUE(second.device.usable());
  first.bind(session, 0);
  second.bind(session, 1);
  EXPECT_EQ(session->scope("memory", 0), nullptr);
  ASSERT_NO_THROW(first.device.reset(simdojo::ResetKind::FunctionLevel));
  session->register_gpu(0);
  ASSERT_NO_FATAL_FAILURE(finish(session));
  const auto devices = run()["devices"].AsVector();
  ASSERT_EQ(devices.size(), 2u);
  expect_reset(devices[0].AsMap(), "function_level", 1, 1, 0);
  expect_reset(devices[1].AsMap(), "function_level", 0, 0, 0);
  EXPECT_EQ(output_.find("\"record\":\"metric\""), std::string::npos);
  EXPECT_EQ(output_.find("\"record\":\"scope\""), std::string::npos);
}

TEST_F(StatisticsResetTest, FailedPciResetIsRecordedWithoutClearingPreviousObservations) {
  auto session = create_session();
  ASSERT_NE(session, nullptr);
  StatisticsResetGpu gpu("gpu");
  ASSERT_TRUE(gpu.device.usable());
  gpu.bind(session, 0);
  auto scope = session->scope("memory", 0);
  ASSERT_NE(scope, nullptr);
  auto counter = scope->counter("requests", "requests", "Observed requests");
  counter->add(3);
  const auto gart = gpu.soc.gpu_vm().gart_address_space();
  ASSERT_TRUE(gart);
  ASSERT_TRUE(gpu.soc.gpu_vm().retain_queue(gart));
  EXPECT_THROW(gpu.device.reset(simdojo::ResetKind::Bus), std::runtime_error);
  EXPECT_EQ(counter->value(), 3u);
  ASSERT_TRUE(gpu.soc.gpu_vm().release_queue(gart));
  ASSERT_NO_THROW(gpu.device.reset(simdojo::ResetKind::Bus));
  counter->add(2);
  EXPECT_EQ(counter->value(), 5u);
  ASSERT_NO_FATAL_FAILURE(finish(session));
  const auto devices = run()["devices"].AsVector();
  ASSERT_EQ(devices.size(), 1u);
  expect_reset(devices[0].AsMap(), "bus", 2, 1, 1);
  EXPECT_EQ(devices[0].AsMap()["resets_in_progress"].AsString().str(), "0");
}

TEST_F(StatisticsResetTest, MovingAnObservationTransfersItsSingleOutcome) {
  auto session = create_session(true);
  ASSERT_NE(session, nullptr);
  session->register_gpu(7);
  {
    auto original = session->observe_reset(7, StatisticsReset::Kind::FunctionLevel);
    auto moved = std::move(original);
    moved.complete();
  }
  { auto failed = session->observe_reset(7, StatisticsReset::Kind::LostConnection); }
  ASSERT_NO_FATAL_FAILURE(finish(session));
  const auto devices = run()["devices"].AsVector();
  ASSERT_EQ(devices.size(), 1u);
  EXPECT_EQ(devices[0].AsMap()["gpu_id"].AsUInt32(), 7u);
  expect_reset(devices[0].AsMap(), "function_level", 1, 1, 0);
  expect_reset(devices[0].AsMap(), "lost_connection", 1, 0, 1);
  EXPECT_EQ(devices[0].AsMap()["resets_in_progress"].AsString().str(), "0");
}

TEST_F(StatisticsResetTest, PrematureFinishReportsAnOutstandingResetAsIncomplete) {
  auto session = create_session(true);
  ASSERT_NE(session, nullptr);
  session->register_gpu(0);
  auto reset = session->observe_reset(0, StatisticsReset::Kind::FunctionLevel);
  ASSERT_NO_FATAL_FAILURE(finish(session));
  const auto devices = run()["devices"].AsVector();
  ASSERT_EQ(devices.size(), 1u);
  EXPECT_EQ(devices[0].AsMap()["resets_in_progress"].AsString().str(), "1");
  EXPECT_NE(output_.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
  session.reset();
  reset.complete();
}

} // namespace
