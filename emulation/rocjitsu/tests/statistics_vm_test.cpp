// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/statistics.h"
#include "rocjitsu/vm/api.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>

namespace {

std::string statistics_vm_config(const std::string &directory, bool enabled = true,
                                 uint32_t num_gpus = 2) {
  return std::format(R"({{
    "num_threads":1,"cpu_dispatch_threads":1,
    "statistics":{{"enabled":{},"output":{{"directory":"{}"}}}},
    "vm":{{"arch":"cdna3","gpu":{{"num_gpus":{},"device":{{
      "gpu_id":9420,"gfx_target_version":90042,"unique_id":9420,
      "simd_count":4,"num_shader_engines":1,"num_shader_arrays_per_engine":1,
      "num_cu_per_sh":1,"num_sdma_engines":0,"local_mem_size":1048576}}}}}},
    "topology":{{"root":{{"name":"soc","type":"soc","children":[
      {{"name":"vram","type":"gpu_memory"}},
      {{"name":"xcd0","type":"xcd","children":[
        {{"name":"l2","type":"l2_cache"}},
        {{"name":"cp","type":"command_processor"}},
        {{"name":"se0","type":"shader_engine","children":[
          {{"name":"cu0","type":"compute_unit"}}]}}]}}]}}}}
  }})",
                     enabled, directory, num_gpus);
}

TEST(StatisticsVmTest, PassiveConfigDoesNotOpenOutputAndAllGpusShareSession) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-vm-"));
  const std::string directory = temporary.path() + "/output";
  const std::string json = statistics_vm_config(directory);
  rj_vm_config_t *config = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &config), ROCJITSU_STATUS_SUCCESS);
  EXPECT_FALSE(std::filesystem::exists(directory));
  rj_vm_t *raw = nullptr;
  const rj_status_t status = rj_vm_create_from_config(config, RJ_VM_MODE_DEFAULT, &raw);
  rj_vm_config_destroy(config);
  ASSERT_EQ(status, ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw, rj_vm_destroy);
  ASSERT_EQ(vm->vm->num_socs(), 2u);
  const auto session = vm->vm->statistics();
  ASSERT_NE(session, nullptr);
  EXPECT_EQ(vm->vm->soc(0)->statistics(), session);
  EXPECT_EQ(vm->vm->soc(1)->statistics(), session);
  const std::string output_path = session->output_path();
  EXPECT_TRUE(std::filesystem::exists(output_path));
  ASSERT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
  vm.reset();
  std::ifstream input(output_path);
  const std::string output{std::istreambuf_iterator<char>(input), {}};
  EXPECT_NE(output.find("\"schema\":\"rocjitsu.statistics.v1\""), std::string::npos);
  EXPECT_NE(output.find("\"gpu_id\":0"), std::string::npos);
  EXPECT_NE(output.find("\"gpu_id\":1"), std::string::npos);
  EXPECT_NE(output.find("cache.backing_read_attempts"), std::string::npos);
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":true"), std::string::npos);
}

TEST(StatisticsVmTest, DisabledSessionDoesNotCreateOutput) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-disabled-"));
  const std::string directory = temporary.path() + "/absent";
  const std::string json = statistics_vm_config(directory, false);
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->vm->statistics(), nullptr);
  rj_vm_destroy(raw);
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(StatisticsVmTest, InvalidFormatFailsBeforeOutputActivation) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-invalid-"));
  const std::string directory = temporary.path() + "/absent";
  std::string json = statistics_vm_config(directory);
  const size_t position = json.find("\"output\":{");
  ASSERT_NE(position, std::string::npos);
  json.insert(position + std::string_view("\"output\":{").size(), "\"format\":\"invalid\",");
  rj_vm_t *raw = nullptr;
  EXPECT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(raw, nullptr);
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(StatisticsVmTest, DebugPausedResidentWaveMakesQuiescentRunIncomplete) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-paused-"));
  const std::string json = statistics_vm_config(temporary.path());
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
            ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw, rj_vm_destroy);
  const std::string output_path = vm->vm->statistics()->output_path();
  auto *cu = vm->vm->soc(0)->all_cus().front();
  auto *wave = cu->dispatch_wf(0, 0x1000, 16, 16);
  ASSERT_NE(wave, nullptr);
  wave->set_debug_suspended(true);
  ASSERT_TRUE(cu->has_active_wfs());
  ASSERT_FALSE(cu->has_runnable_wfs());
  ASSERT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
  vm.reset();
  std::ifstream input(output_path);
  const std::string output{std::istreambuf_iterator<char>(input), {}};
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
}

TEST(StatisticsVmTest, UnreadComputeRingsMakeQuiescentRunIncompleteWithoutChangingCheckpoint) {
  using namespace rocjitsu::amdgpu;
  for (const QueuePacketFormat format : {QueuePacketFormat::Aql, QueuePacketFormat::Pm4}) {
    SCOPED_TRACE(format == QueuePacketFormat::Aql ? "AQL" : "PM4");
    rocjitsu::test::ScopedTempDirectory temporary;
    ASSERT_TRUE(temporary.initialize("rocjitsu-stats-unread-ring-"));
    const std::string json = statistics_vm_config(temporary.path(), true, 1);
    rj_vm_t *raw = nullptr;
    ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
              ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw, rj_vm_destroy);
    auto *soc = vm->vm->soc(0);
    auto *cp = soc->xcd(0)->command_processor();
    auto *memory = soc->memory();
    constexpr uint64_t kRing = 0x1000;
    constexpr uint64_t kReadPointer = 0x1100;
    constexpr uint64_t kWritePointer = 0x1108;
    memory->write32(kRing,
                    format == QueuePacketFormat::Pm4 ? 0x80000000 : HSA_PACKET_TYPE_BARRIER_AND);
    memory->write64(kReadPointer, 0);
    memory->write64(kWritePointer, 1);
    const uint64_t registration = cp->register_queue({
        .address_space = soc->internal_address_space(),
        .queue_id = 1,
        .ring_base_va = kRing,
        .ring_size = 256,
        .read_ptr_va = kReadPointer,
        .write_ptr_va = kWritePointer,
        .last_doorbell = std::numeric_limits<uint64_t>::max(),
        .packet_format = format,
        .initial_consumer_cursor = 0,
    });
    ASSERT_NE(registration, 0u);
    cp->set_queue_debug_suspended(1, 0, true);
    if (format == QueuePacketFormat::Pm4)
      ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 1), QueueSubmissionStatus::Accepted);
    else
      cp->notify_queue_doorbell(registration, 0);
    ASSERT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
    ASSERT_EQ(vm->engine->last_exit().reason, simdojo::ExitReason::COMPLETED);
    EXPECT_EQ(memory->read64(kReadPointer), 0u);
    EXPECT_FALSE(cp->has_pending_aql_dispatch_work());
    EXPECT_FALSE(soc->all_cus().front()->has_active_wfs());
    const std::string checkpoint = temporary.path() + "/paused.rjc";
    EXPECT_EQ(rj_vm_save_checkpoint(vm.get(), checkpoint.c_str(), 0), ROCJITSU_STATUS_SUCCESS);

    const std::string output_path = vm->vm->statistics()->output_path();
    vm.reset();
    std::ifstream input(output_path);
    const std::string output{std::istreambuf_iterator<char>(input), {}};
    EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
  }
}

class StatisticsExitComponent final : public simdojo::Component {
public:
  StatisticsExitComponent() : Component("statistics_requested_exit") {}
  void startup() override { engine()->request_exit("statistics test requested exit", 0); }
};

TEST(StatisticsVmTest, SuccessfulRequestedExitRemainsExplicitlyIncomplete) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-requested-exit-"));
  const std::string json = statistics_vm_config(temporary.path());
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
            ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw, rj_vm_destroy);
  const std::string output_path = vm->vm->statistics()->output_path();
  vm->engine->shutdown();
  vm->vm->add_child(std::make_unique<StatisticsExitComponent>());
  ASSERT_TRUE(vm->engine->topology().partition_balanced(1).succeeded());
  ASSERT_TRUE(vm->engine->create().succeeded());
  ASSERT_EQ(rj_vm_run(vm.get(), nullptr), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(vm->engine->last_exit().reason, simdojo::ExitReason::EXIT_REQUEST);
  vm.reset();
  std::ifstream input(output_path);
  const std::string output{std::istreambuf_iterator<char>(input), {}};
  EXPECT_NE(output.find("\"record\":\"session_end\",\"complete\":false"), std::string::npos);
}

TEST(StatisticsVmTest, DisablingSessionUnbindsEveryGpuCacheAndComputeUnit) {
  rocjitsu::test::ScopedTempDirectory temporary;
  ASSERT_TRUE(temporary.initialize("rocjitsu-stats-disable-existing-"));
  const std::string json = statistics_vm_config(temporary.path());
  rj_vm_t *raw = nullptr;
  ASSERT_EQ(rj_vm_create_from_string(json.c_str(), RJ_VM_MODE_DEFAULT, &raw),
            ROCJITSU_STATUS_SUCCESS);
  std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)> vm(raw, rj_vm_destroy);
  const std::shared_ptr<rocjitsu::StatisticsSession> session = vm->vm->statistics();
  std::vector<std::shared_ptr<stats::Scope>> scopes;
  for (uint32_t gpu = 0; gpu < vm->vm->num_socs(); ++gpu) {
    rocjitsu::SoC *soc = vm->vm->soc(gpu);
    for (rocjitsu::amdgpu::Xcd *xcd : soc->xcds())
      scopes.push_back(session->scope(xcd->l2_cache()->full_path(), gpu));
    for (rocjitsu::amdgpu::ComputeUnitCore *cu : soc->all_cus()) {
      scopes.push_back(session->scope(cu->full_path(), gpu));
      scopes.push_back(session->scope(cu->full_path() + ".l1_scalar", gpu));
      scopes.push_back(session->scope(cu->full_path() + ".l1_vector", gpu));
    }
  }
  const auto access = [&] {
    constexpr uint64_t kAddress = 0x1000;
    constexpr uint32_t kValue = 42;
    for (uint32_t gpu = 0; gpu < vm->vm->num_socs(); ++gpu) {
      rocjitsu::SoC *soc = vm->vm->soc(gpu);
      soc->memory()->write32(kAddress, kValue);
      for (rocjitsu::amdgpu::Xcd *xcd : soc->xcds()) {
        uint32_t observed = 0;
        ASSERT_EQ(xcd->l2_cache()->read(kAddress, reinterpret_cast<uint8_t *>(&observed),
                                        sizeof(observed)),
                  rocjitsu::amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(observed, kValue);
      }
      for (rocjitsu::amdgpu::ComputeUnitCore *cu : soc->all_cus()) {
        uint32_t observed = 0;
        ASSERT_EQ(cu->l1_scalar().load(kAddress, 1, &observed),
                  rocjitsu::amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(observed, kValue);
        observed = 0;
        ASSERT_EQ(cu->l1_vector().load(&kAddress, 1, sizeof(observed), 1,
                                       reinterpret_cast<uint8_t *>(&observed),
                                       rocjitsu::amdgpu::Mtype::RW, false, false, 0),
                  rocjitsu::amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(observed, kValue);
        cu->step();
      }
    }
  };
  const auto snapshot = [&] {
    std::ostringstream output;
    for (const std::shared_ptr<stats::Scope> &scope : scopes)
      scope->write_samples(output);
    return output.str();
  };
  ASSERT_NO_FATAL_FAILURE(access());
  const std::string before_disable = snapshot();
  EXPECT_NE(before_disable.find("cache.read_requests"), std::string::npos);
  rocjitsu::StatisticsConfig disabled;
  ASSERT_TRUE(
      vm->vm->initialize_statistics(disabled, "node", simdojo::ExecMode::FUNCTIONAL).succeeded());
  EXPECT_EQ(vm->vm->statistics(), nullptr);
  for (uint32_t gpu = 0; gpu < vm->vm->num_socs(); ++gpu)
    EXPECT_EQ(vm->vm->soc(gpu)->statistics(), nullptr);
  ASSERT_NO_FATAL_FAILURE(access());
  EXPECT_EQ(snapshot(), before_disable);
}

} // namespace
