// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/perfsim/plugin.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/plugins/perfsim/observer_abi_v13.h"
#include "rocjitsu/vm/plugins/plugin_loader.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "scoped_temp.h"
#include "util/dynamic_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef PERFSIM_FAKE_BACKEND_PATH
#error "PERFSIM_FAKE_BACKEND_PATH must be defined"
#endif
#ifndef PERFSIM_PLUGIN_DIR
#error "PERFSIM_PLUGIN_DIR must be defined"
#endif

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::perfsim::observer_abi_v13;
using rocjitsu::plugins::perfsim::PerfsimPlugin;

FfmWaveInfo make_wave_info(EntityId dispatch_id, EntityId cluster_id, EntityId workgroup_id,
                           EntityId wavegroup_id, EntityId wave_id) {
  return FfmWaveInfo{
      FfmWorkgroupInfo{FfmClusterInfo{FfmDispatchInfo{dispatch_id}, cluster_id}, workgroup_id},
      wavegroup_id, wave_id};
}

FfmInstructionInfo
make_instruction_info(EntityId dispatch_id, EntityId cluster_id, EntityId workgroup_id,
                      EntityId wavegroup_id, EntityId wave_id, EntityId instruction_id, uint64_t pc,
                      const uint32_t raw_isa[4], FfmInstructionCounters instruction_counters = {},
                      FfmWaitInfo wait_info = {FFM_WAIT_TYPE_NONE, FFM_WAIT_NAME_NONE}) {
  return {instruction_id,
          make_wave_info(dispatch_id, cluster_id, workgroup_id, wavegroup_id, wave_id),
          {pc, {raw_isa[0], raw_isa[1], raw_isa[2], raw_isa[3]}},
          instruction_counters,
          wait_info};
}

FfmMemoryAccess make_memory_access(EntityId instruction_id, FfmWaveInfo wave_info,
                                   uint64_t exec_mask, uint32_t wave_size,
                                   const uint64_t *addresses, uint32_t data_size_bytes,
                                   FfmResourceType resource_type, bool is_atomic, bool is_read,
                                   bool is_write) {
  const uint32_t bounded_wave_size = std::min(wave_size, FFM_MAX_WAVE_SIZE);
  const uint64_t valid_lane_mask =
      bounded_wave_size >= FFM_MAX_WAVE_SIZE ? UINT64_MAX : (uint64_t{1} << bounded_wave_size) - 1;
  FfmMemoryAccess access{};
  access.instruction_id = instruction_id;
  access.wave_info = wave_info;
  access.exec_mask = exec_mask & valid_lane_mask;
  access.wave_size = bounded_wave_size;
  access.data_size_bytes = data_size_bytes;
  access.resource_type = resource_type;
  access.flags = encode_memory_flags(is_atomic, is_read, is_write);
  if (addresses) {
    for (uint32_t lane = 0; lane < bounded_wave_size; ++lane) {
      if (((access.exec_mask >> lane) & uint64_t{1}) != 0)
        access.addresses[lane] = addresses[lane];
    }
  }
  return access;
}

class SyntheticInstruction final : public Instruction {
public:
  SyntheticInstruction(std::string_view mnemonic, std::span<const uint32_t> words,
                       uint64_t flags = 0)
      : Instruction(mnemonic, nullptr) {
    if (words.size() > raw_.size())
      throw std::invalid_argument("too many synthetic instruction words");
    std::copy(words.begin(), words.end(), raw_.begin());
    raw_encoding_ = raw_.data();
    size_ = static_cast<int>(words.size() * sizeof(uint32_t));
    flags_ = flags;
  }

private:
  std::array<uint32_t, 5> raw_{};
};

struct WaveFixture {
  std::unique_ptr<GpuMemory> memory = std::make_unique<GpuMemory>("perfsim_test_memory");
  std::unique_ptr<L2Cache> l2 = std::make_unique<L2Cache>("perfsim_test_l2");
  std::unique_ptr<ComputeUnitCore> cu;

  explicit WaveFixture(uint32_t slots = 2) {
    ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA5;
    config.target = ROCJITSU_CODE_TARGET_GFX1250;
    config.num_wf_slots = slots;
    config.sgprs_per_wf = 128;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("perfsim_test_cu", config, memory.get(), l2.get());
  }

  Wavefront &wave(uint32_t dispatch_id, uint32_t rocjitsu_workgroup_id,
                  std::array<uint32_t, 3> coordinate, uint32_t wave_in_group, uint32_t queue_id = 3,
                  uint32_t process_id = 4) {
    Wavefront *wf = cu->dispatch_wf(rocjitsu_workgroup_id, 0x1000, 64, 64);
    if (!wf)
      throw std::runtime_error("could not allocate test wavefront");
    wf->set_dispatch_id(dispatch_id);
    wf->set_wg_coord(coordinate[0], coordinate[1], coordinate[2]);
    wf->set_wave_in_group(wave_in_group);
    wf->set_queue_id(queue_id);
    wf->set_process_id(process_id);
    wf->set_cluster_info(0, 1);
    return *wf;
  }
};

KernelDispatchInfo dispatch_info(uint32_t id) {
  KernelDispatchInfo info;
  info.dispatch_id = id;
  info.kernel_name = "test_kernel_" + std::to_string(id);
  info.lds_size_bytes = 4096;
  info.wave_size = 32;
  info.code_target = ROCJITSU_CODE_TARGET_GFX1250;
  info.grid_size_x = 32;
  info.grid_size_y = 2;
  info.grid_size_z = 3;
  info.workgroup_size_x = 32;
  info.workgroup_size_y = 1;
  info.workgroup_size_z = 1;
  info.cluster_size_x = 1;
  info.cluster_size_y = 1;
  info.cluster_size_z = 1;
  info.workgroup_count = 6;
  info.wfs_per_workgroup = 1;
  info.sgprs_per_wf = 64;
  info.vgprs_per_wf = 96;
  return info;
}

std::string json_string(std::string_view value) {
  constexpr std::string_view hex = "0123456789abcdef";
  std::string result{"\""};
  for (const unsigned char byte : value) {
    switch (byte) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (byte < 0x20) {
        result += "\\u00";
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0xF]);
      } else {
        result.push_back(static_cast<char>(byte));
      }
    }
  }
  result.push_back('"');
  return result;
}

std::string plugin_config(std::string_view library_path) {
  return std::string{"{\"library_path\":"} + json_string(library_path) + "}";
}

std::string plugin_config() { return plugin_config(PERFSIM_FAKE_BACKEND_PATH); }

std::string plugin_config_with_staging_budget(uint64_t max_staged_bytes) {
  return std::string{"{\"library_path\":"} + json_string(PERFSIM_FAKE_BACKEND_PATH) +
         ",\"max_staged_bytes\":" + std::to_string(max_staged_bytes) + "}";
}

std::string read_file(const std::string &path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::string> lines(const std::string &contents) {
  std::vector<std::string> result;
  std::istringstream input(contents);
  for (std::string line; std::getline(input, line);)
    result.push_back(std::move(line));
  return result;
}

size_t line_with_prefix(const std::vector<std::string> &trace, std::string_view prefix,
                        size_t start = 0) {
  for (size_t i = start; i < trace.size(); ++i)
    if (std::string_view(trace[i]).starts_with(prefix))
      return i;
  return trace.size();
}

class PerfsimPluginTest : public ::testing::Test {
protected:
  void SetUp() override {
    trace_env_ = std::make_unique<rocjitsu::test::ScopedEnvironmentVariable>(
        "ROCJITSU_PERFSIM_FAKE_TRACE", trace_.path());
    mode_env_ = std::make_unique<rocjitsu::test::ScopedEnvironmentVariable>(
        "ROCJITSU_PERFSIM_FAKE_MODE", "");
  }

  rocjitsu::test::ScopedTempFile trace_{"rocjitsu-perfsim-"};
  std::unique_ptr<rocjitsu::test::ScopedEnvironmentVariable> trace_env_;
  std::unique_ptr<rocjitsu::test::ScopedEnvironmentVariable> mode_env_;
};

class BlockingOverlapSink final : public PluginSink {
public:
  void write(std::string_view) override {
    const int active = active_writes_.fetch_add(1, std::memory_order_relaxed) + 1;
    int observed = max_active_writes_.load(std::memory_order_relaxed);
    while (active > observed &&
           !max_active_writes_.compare_exchange_weak(observed, active, std::memory_order_relaxed)) {
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (!first_entered_) {
      first_entered_ = true;
      cv_.notify_all();
      cv_.wait(lock, [&]() { return release_first_; });
    } else if (active > 1) {
      overlap_observed_ = true;
      cv_.notify_all();
    }
    active_writes_.fetch_sub(1, std::memory_order_relaxed);
  }

  bool wait_for_first(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&]() { return first_entered_; });
  }

  bool wait_for_overlap(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&]() { return overlap_observed_; });
  }

  void release_first() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_first_ = true;
    cv_.notify_all();
  }

  int max_active_writes() const { return max_active_writes_.load(std::memory_order_relaxed); }

private:
  std::atomic<int> active_writes_{0};
  std::atomic<int> max_active_writes_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool first_entered_ = false;
  bool overlap_observed_ = false;
  bool release_first_ = false;
};

class InitSinkWriterPlugin final : public ExecutionPlugin {
public:
  explicit InitSinkWriterPlugin(BlockingOverlapSink &sink_probe)
      : ExecutionPlugin("init_sink_writer"), sink_probe_(sink_probe) {}

  void onInit() override {
    const bool backend_write_entered = sink_probe_.wait_for_first(std::chrono::seconds(5));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      write_started_ = true;
      cv_.notify_all();
    }
    if (backend_write_entered)
      sink().write("peer plugin initialized\n");
  }

  bool wait_for_write_start(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&]() { return write_started_; });
  }

private:
  BlockingOverlapSink &sink_probe_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool write_started_ = false;
};

TEST(PerfsimPluginConfigTest, EscapesBackendPathAsJson) {
  std::string path{"a\"b\\c\n"};
  path.push_back('\x01');
  EXPECT_EQ(plugin_config(path), "{\"library_path\":\"a\\\"b\\\\c\\n\\u0001\"}");
}

TEST(PerfsimPluginConfigTest, RejectsRelativeBackendPath) {
  EXPECT_THROW(PerfsimPlugin(R"({"library_path":"relative/libgpucsim_ffm_plugin.so"})"),
               std::invalid_argument);
}

TEST(PerfsimPluginConfigTest, RejectsInvalidStagingBudgets) {
  const std::string prefix = std::string{"{\"library_path\":"} +
                             json_string(PERFSIM_FAKE_BACKEND_PATH) + ",\"max_staged_bytes\":";
  for (std::string_view value : {"0", "-1", "1.5", "\"4096\""}) {
    SCOPED_TRACE(value);
    const std::string config = prefix + std::string(value) + "}";
    EXPECT_THROW(PerfsimPlugin(config.c_str()), std::invalid_argument);
  }
}

TEST_F(PerfsimPluginTest, RequestsV13AndForwardsOwnedFieldsInOrder) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();
    plugin.onInit();

    KernelDispatchInfo info = dispatch_info(7);
    info.kernel_name = "v13_owned_kernel";
    info.lds_size_bytes = 1025;
    plugin.onAmdgpuDispatchPacketProcessed(info);
    info.kernel_name = "mutated_after_callback";
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 99, {2, 3, 4}, 5);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 2> wmma_words{0xCC400123, 0xABCDEF01};
    SyntheticInstruction wmma("v_wmma_f32_16x16x16_f16", wmma_words, MFMA);
    plugin.onAmdgpuBeforeExecuteInstruction(0x12345678, wmma, wave);

    std::array<uint64_t, 32> addresses{};
    addresses[0] = 0x100000;
    addresses[3] = 0x200000;
    MemoryAccessObservation access;
    access.mnemonic = "global_load_dwordx2";
    access.pc = 0x12345678;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.route = MemoryRoute::GLOBAL;
    access.decoded_space = DecodedMemorySpace::GLOBAL;
    access.is_load = true;
    access.wavefront_size = 32;
    access.element_size_bytes = 4;
    access.elements_per_lane = 2;
    access.active_lane_mask = 0xB;
    access.architectural_exec_lane_mask = 0xB;
    access.valid_lane_mask = 0x9;
    access.request_lane_mask = 0x9;
    access.addresses = addresses;
    plugin.onAmdgpuMemoryAccessRouted(access);
    addresses[0] = 0xDEADBEEF;

    const std::array<uint32_t, 3> tensor_words{0xD0710001, 0x7C000000, 0x18140C00};
    SyntheticInstruction tensor("tensor_load_to_lds", tensor_words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(0x12345680, tensor, wave);
    std::vector<uint64_t> tensor_addresses{0x300000, 0x300004, 0x300004};
    TensorDmaMemoryAccessObservation tensor_access;
    tensor_access.mnemonic = "tensor_load_to_lds";
    tensor_access.pc = 0x12345680;
    tensor_access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    tensor_access.dispatch_id = info.dispatch_id;
    tensor_access.queue_id = wave.queue_id();
    tensor_access.workgroup_id = wave.wg_id();
    tensor_access.wavefront_id = wave.wf_id();
    tensor_access.process_id = wave.process_id();
    tensor_access.element_size_bytes = 4;
    tensor_access.tile_dim0 = 11;
    tensor_access.tile_dim1 = 12;
    tensor_access.data_size = 2;
    tensor_access.tensor_dim0_stride = 52;
    tensor_access.tensor_dim1_stride = 104;
    tensor_access.is_load = true;
    tensor_access.addresses = std::span<const uint64_t>(tensor_addresses);
    plugin.onAmdgpuTensorDmaMemoryAccess(tensor_access);
    tensor_addresses[0] = 0xDEADBEEF;

    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0x1234568C, end, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
    plugin.onShutdown();
  }

  const auto trace = lines(read_file(trace_.path()));
  ASSERT_GE(trace.size(), 10u);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 13"), 1);
  EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                          [](const std::string &line) { return line.starts_with("get_api "); }),
            1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 13"), 1);
  EXPECT_EQ(line_with_prefix(trace, "init_rejected "), trace.size());
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "shutdown"), 1);
  const size_t shutdown = line_with_prefix(trace, "shutdown");
  ASSERT_LT(shutdown, trace.size());
  EXPECT_EQ(line_with_prefix(trace, "unload"), trace.size());

  const size_t begin =
      line_with_prefix(trace, "begin 7 96 128 1025 32 1 32 2 3 32 1 1 v13_owned_kernel");
  const size_t first_instruction = line_with_prefix(trace, "instruction 7 4003002 0 1 5 0 ");
  const size_t memory = line_with_prefix(trace, "memory 7 4003002 0 1 5 0 9 32 8 5 0 1 0 1048576");
  const size_t tensor_instruction =
      line_with_prefix(trace, "instruction 7 4003002 0 1 5 1 ", first_instruction + 1);
  const size_t tensor =
      line_with_prefix(trace, "tdm 7 4003002 0 1 5 1 3 4 1 0 11 12 2 52 104 3145728 3145732 "
                              "3145732");
  const size_t end_instruction =
      line_with_prefix(trace, "instruction 7 4003002 0 1 5 2 ", tensor_instruction + 1);
  const size_t end = line_with_prefix(trace, "end 7 ");
  ASSERT_LT(begin, trace.size());
  ASSERT_LT(first_instruction, trace.size());
  ASSERT_LT(memory, trace.size());
  ASSERT_LT(tensor_instruction, trace.size());
  ASSERT_LT(tensor, trace.size());
  ASSERT_LT(end_instruction, trace.size());
  ASSERT_LT(end, trace.size());
  EXPECT_LT(begin, first_instruction);
  EXPECT_LT(first_instruction, memory);
  EXPECT_LT(memory, tensor_instruction);
  EXPECT_LT(tensor_instruction, tensor);
  EXPECT_LT(tensor, end_instruction);
  EXPECT_LT(end_instruction, end);
}

TEST_F(PerfsimPluginTest, FallsBackToV8AndForwardsLegacyPayloadPrefixes) {
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "v8_only", 1);
  WaveFixture fixture;
  const std::string config = plugin_config();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    KernelDispatchInfo info = dispatch_info(8);
    info.kernel_name = "ignored_by_v8";
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 99, {2, 3, 4}, 5);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 3> tensor_words{0xD0710001, 0x7C000000, 0x18140C00};
    SyntheticInstruction tensor("tensor_load_to_lds", tensor_words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(0x12345680, tensor, wave);
    const std::array<uint64_t, 2> tensor_addresses{0x300000, 0x300004};
    TensorDmaMemoryAccessObservation tensor_access;
    tensor_access.mnemonic = "tensor_load_to_lds";
    tensor_access.pc = 0x12345680;
    tensor_access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    tensor_access.dispatch_id = info.dispatch_id;
    tensor_access.queue_id = wave.queue_id();
    tensor_access.workgroup_id = wave.wg_id();
    tensor_access.wavefront_id = wave.wf_id();
    tensor_access.process_id = wave.process_id();
    tensor_access.element_size_bytes = 4;
    tensor_access.tile_dim0 = 11;
    tensor_access.tile_dim1 = 12;
    tensor_access.data_size = 2;
    tensor_access.tensor_dim0_stride = 52;
    tensor_access.tensor_dim1_stride = 104;
    tensor_access.is_load = true;
    tensor_access.addresses = std::span<const uint64_t>(tensor_addresses);
    plugin.onAmdgpuTensorDmaMemoryAccess(tensor_access);

    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0x1234568C, end, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }

  const auto trace = lines(read_file(trace_.path()));
  for (uint32_t version = 13; version >= 8; --version)
    EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api " + std::to_string(version)), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 8"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "shutdown"), 1);

  const size_t begin = line_with_prefix(trace, "begin 8 ");
  ASSERT_LT(begin, trace.size());
  EXPECT_EQ(trace[begin], "begin 8 96 128 4096 32 1 32 2 3 32 1 1");
  EXPECT_NE(line_with_prefix(trace, "tdm 8 4003002 0 1 5 0 2 4 1 0 3145728 3145732"), trace.size());
}

TEST_F(PerfsimPluginTest, AcceptsBackendSelectedCompatibleVersion) {
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "old_version", 1);
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 13"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 12"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "shutdown"), 1);
}

TEST_F(PerfsimPluginTest, FallsBackToV12WithDispatchNameAndHostLogger) {
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "v12_only", 1);
  const std::string config = std::string{"{\"plugins\":{\"perfsim\":"} + plugin_config() + "}}";
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_EQ(PluginLoader::load_from_config(config, group, PERFSIM_PLUGIN_DIR), 1);
  group.onInit();

  KernelDispatchInfo info = dispatch_info(12);
  info.kernel_name = "v12_fallback_kernel";
  group.onAmdgpuDispatchPacketProcessed(info);
  group.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  group.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  group.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 13"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 12"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 11"), 0);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 12"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "host_log present"), 1);
  EXPECT_NE(line_with_prefix(trace, "begin 12 96 128 4096 32 1 32 2 3 32 1 1 "
                                    "v12_fallback_kernel"),
            trace.size());
  EXPECT_NE(sink.str().find("[perfsim:warn] fake backend initialized\n"), std::string::npos);
}

TEST_F(PerfsimPluginTest, FallsBackToV11WithDispatchNameAndNoHostLogger) {
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "v11_only", 1);
  const std::string config = std::string{"{\"plugins\":{\"perfsim\":"} + plugin_config() + "}}";
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_EQ(PluginLoader::load_from_config(config, group, PERFSIM_PLUGIN_DIR), 1);
  group.onInit();

  KernelDispatchInfo info = dispatch_info(11);
  info.kernel_name = "v11_fallback_kernel";
  group.onAmdgpuDispatchPacketProcessed(info);
  group.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  group.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  group.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 13"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 12"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 11"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "get_api 10"), 0);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 11"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "host_log absent"), 1);
  EXPECT_NE(line_with_prefix(trace, "begin 11 96 128 4096 32 1 32 2 3 32 1 1 "
                                    "v11_fallback_kernel"),
            trace.size());
  EXPECT_EQ(sink.str().find("fake backend initialized"), std::string::npos);
}

TEST_F(PerfsimPluginTest, ReportsNegotiatedVersionForUnrepresentableInstruction) {
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "v8_only", 1);
  WaveFixture fixture;
  const std::string config = plugin_config();
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    const KernelDispatchInfo info = dispatch_info(9);
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 5> words{};
    SyntheticInstruction oversized("oversized", words);
    plugin.onAmdgpuBeforeExecuteInstruction(0x2000, oversized, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("instruction encoding is not representable by FFM v8"),
            std::string::npos);
  EXPECT_EQ(diagnostic.find("FFM v13"), std::string::npos);
}

TEST_F(PerfsimPluginTest, RejectsMalformedTensorDmaMetadataIndependently) {
  struct MalformedCase {
    std::string_view name;
    uint32_t element_size_bytes;
    uint32_t data_size;
    uint32_t tile_dim0;
    int64_t tensor_dim0_stride;
    int64_t tensor_dim1_stride;
  };
  constexpr std::array cases{
      MalformedCase{"non_power_of_two_element_size", 3, 2, 1, 0, 0},
      MalformedCase{"element_size_code_mismatch", 4, 1, 1, 0, 0},
      MalformedCase{"zero_tile_dim0", 4, 2, 0, 0, 0},
      MalformedCase{"negative_tensor_dim0_stride", 4, 2, 1, -1, 0},
      MalformedCase{"negative_tensor_dim1_stride", 4, 2, 1, 0, -1},
  };
  const std::string config = plugin_config();
  for (size_t index = 0; index < cases.size(); ++index) {
    const MalformedCase &test_case = cases[index];
    SCOPED_TRACE(test_case.name);
    WaveFixture fixture;
    testing::internal::CaptureStderr();
    {
      PerfsimPlugin plugin(config.c_str());
      plugin.onInit();

      const uint32_t dispatch_id = 18 + static_cast<uint32_t>(index);
      const KernelDispatchInfo info = dispatch_info(dispatch_id);
      plugin.onAmdgpuDispatchPacketProcessed(info);
      plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
      Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
      plugin.onAmdgpuWavefrontDispatched(wave);

      const std::array<uint32_t, 3> words{0xD0710001, 0x7C000000, 0x18140C00};
      SyntheticInstruction tensor("tensor_load_to_lds", words, MEMORY_OP);
      plugin.onAmdgpuBeforeExecuteInstruction(0x1280, tensor, wave);
      const std::array<uint64_t, 1> addresses{0x300000};
      TensorDmaMemoryAccessObservation access;
      access.mnemonic = "tensor_load_to_lds";
      access.pc = 0x1280;
      access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
      access.dispatch_id = info.dispatch_id;
      access.queue_id = wave.queue_id();
      access.workgroup_id = wave.wg_id();
      access.wavefront_id = wave.wf_id();
      access.process_id = wave.process_id();
      access.element_size_bytes = test_case.element_size_bytes;
      access.is_load = true;
      access.addresses = std::span<const uint64_t>(addresses);
      access.tile_dim0 = test_case.tile_dim0;
      access.tile_dim1 = 1;
      access.data_size = test_case.data_size;
      access.tensor_dim0_stride = test_case.tensor_dim0_stride;
      access.tensor_dim1_stride = test_case.tensor_dim1_stride;
      plugin.onAmdgpuTensorDmaMemoryAccess(access);

      const std::array<uint32_t, 1> end_words{0xBF810000};
      SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
      plugin.onAmdgpuBeforeExecuteInstruction(0x128C, end, wave);
      plugin.onAmdgpuWavefrontHalted(wave);
      plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
      plugin.onShutdown();
    }
    const std::string diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("tensor-DMA observation is malformed"), std::string::npos);

    const std::string dispatch_prefix = std::to_string(18 + index) + " ";
    const auto trace = lines(read_file(trace_.path()));
    EXPECT_EQ(line_with_prefix(trace, "begin " + dispatch_prefix), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "instruction " + dispatch_prefix), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "tdm " + dispatch_prefix), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "end " + dispatch_prefix), trace.size());
  }
}

TEST_F(PerfsimPluginTest, PreservesMixedFlatAndDualLdsRecordOrder) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(11);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 10, {0, 0, 0}, 0);
  wave.set_lds_base(0x1000);
  plugin.onAmdgpuWavefrontDispatched(wave);

  const std::array<uint32_t, 1> flat_words{0xDEAD0001};
  SyntheticInstruction flat("flat_load_dword", flat_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2000, flat, wave);

  std::array<uint64_t, 32> flat_addresses{};
  flat_addresses[0] = 100;
  flat_addresses[1] = 200;
  flat_addresses[2] = 300;
  MemoryAccessObservation flat_access;
  flat_access.mnemonic = "flat_load_dword";
  flat_access.pc = 0x2000;
  flat_access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
  flat_access.dispatch_id = info.dispatch_id;
  flat_access.queue_id = wave.queue_id();
  flat_access.workgroup_id = wave.wg_id();
  flat_access.wavefront_id = wave.wf_id();
  flat_access.process_id = wave.process_id();
  flat_access.route = MemoryRoute::GLOBAL;
  flat_access.decoded_space = DecodedMemorySpace::FLAT;
  flat_access.is_load = true;
  flat_access.atomic_op = AtomicOp::NONE;
  flat_access.wavefront_size = 32;
  flat_access.element_size_bytes = 4;
  flat_access.elements_per_lane = 1;
  flat_access.active_lane_mask = 0x7;
  flat_access.architectural_exec_lane_mask = 0x7;
  flat_access.valid_lane_mask = 0x7;
  flat_access.request_lane_mask = 0x7;
  flat_access.scratch_lane_mask = 0x5;
  flat_access.addresses = flat_addresses;
  plugin.onAmdgpuMemoryAccessRouted(flat_access);

  const std::array<uint32_t, 1> ds_words{0xD8000000};
  SyntheticInstruction ds("ds_read2_b32", ds_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2004, ds, wave);
  std::array<uint64_t, 32> primary{};
  std::array<uint64_t, 32> secondary{};
  primary[0] = 0x1000 + 400;
  primary[1] = 0x1000 + 404;
  secondary[0] = 0x1000 + 800;
  secondary[1] = 0x1000 + 804;
  MemoryAccessObservation ds_access = flat_access;
  ds_access.mnemonic = "ds_read2_b32";
  ds_access.pc = 0x2004;
  ds_access.route = MemoryRoute::LOCAL;
  ds_access.decoded_space = DecodedMemorySpace::LOCAL;
  ds_access.atomic_op = AtomicOp::NONE;
  ds_access.active_lane_mask = 0x3;
  ds_access.architectural_exec_lane_mask = 0x3;
  ds_access.valid_lane_mask = 0x3;
  ds_access.request_lane_mask = 0x3;
  ds_access.scratch_lane_mask = 0;
  ds_access.addresses = primary;
  ds_access.secondary_addresses = secondary;
  plugin.onAmdgpuMemoryAccessRouted(ds_access);

  const std::array<uint32_t, 1> flat_lds_words{0xDEAD0002};
  SyntheticInstruction flat_lds("flat_load_dword", flat_lds_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2008, flat_lds, wave);
  std::array<uint64_t, 32> effective_lds{};
  std::array<uint64_t, 32> aperture_addresses{};
  effective_lds[0] = 0x1000 + 40;
  effective_lds[1] = 0x1000 + 44;
  effective_lds[2] = 0x1000 + 48;
  effective_lds[3] = 0x1000 + 52;
  aperture_addresses[0] = 900;
  aperture_addresses[1] = 904;
  aperture_addresses[2] = 908;
  aperture_addresses[3] = 912;
  MemoryAccessObservation flat_lds_access = ds_access;
  flat_lds_access.mnemonic = "flat_load_dword";
  flat_lds_access.pc = 0x2008;
  flat_lds_access.decoded_space = DecodedMemorySpace::FLAT;
  flat_lds_access.normalized_to_local = true;
  flat_lds_access.active_lane_mask = 0xF;
  flat_lds_access.architectural_exec_lane_mask = 0xF;
  flat_lds_access.valid_lane_mask = 0xF;
  flat_lds_access.request_lane_mask = 0x3;
  flat_lds_access.flat_local_lane_mask = 0x3;
  flat_lds_access.addresses = effective_lds;
  flat_lds_access.pre_routing_addresses = aperture_addresses;
  flat_lds_access.secondary_addresses = {};
  plugin.onAmdgpuMemoryAccessRouted(flat_lds_access);

  const std::array<uint32_t, 1> append_words{0xD8340000};
  SyntheticInstruction append("ds_append", append_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x200C, append, wave);
  std::array<uint64_t, 32> append_addresses{};
  append_addresses.fill(0x1000 + 1200);
  MemoryAccessObservation append_access = ds_access;
  append_access.mnemonic = "ds_append";
  append_access.pc = 0x200C;
  append_access.atomic_op = AtomicOp::APPEND;
  append_access.active_lane_mask = 0xF;
  append_access.architectural_exec_lane_mask = 0xF;
  append_access.valid_lane_mask = 0xF;
  append_access.request_lane_mask = 0xF;
  append_access.addresses = append_addresses;
  append_access.secondary_addresses = {};
  plugin.onAmdgpuMemoryAccessRouted(append_access);

  const std::array<uint32_t, 1> consume_words{0xD8350000};
  SyntheticInstruction consume("ds_consume", consume_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2010, consume, wave);
  MemoryAccessObservation consume_access = append_access;
  consume_access.mnemonic = "ds_consume";
  consume_access.pc = 0x2010;
  consume_access.atomic_op = AtomicOp::CONSUME;
  plugin.onAmdgpuMemoryAccessRouted(consume_access);

  const std::array<const char *, 7> other_ffm_silent_memory{"ds_load_tr4_b64",
                                                            "ds_load_tr6_b96",
                                                            "ds_load_tr8_b64",
                                                            "ds_atomic_async_barrier_arrive_b64",
                                                            "ds_atomic_barrier_arrive_rtn_b64",
                                                            "global_load_block",
                                                            "global_store_block"};
  for (size_t i = 0; i < other_ffm_silent_memory.size(); ++i) {
    const uint64_t pc = 0x2014 + i * 4;
    SyntheticInstruction silent(other_ffm_silent_memory[i], ds_words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(pc, silent, wave);
    MemoryAccessObservation silent_access = append_access;
    silent_access.mnemonic = other_ffm_silent_memory[i];
    silent_access.pc = pc;
    silent_access.atomic_op = i < 3 || i >= 5 ? AtomicOp::NONE : AtomicOp::BARRIER_ARRIVE;
    if (i >= 5) {
      silent_access.decoded_space = DecodedMemorySpace::GLOBAL;
      silent_access.route = MemoryRoute::GLOBAL;
    }
    plugin.onAmdgpuMemoryAccessRouted(silent_access);
  }

  SyntheticInstruction tr16("ds_load_tr16_b128", ds_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2030, tr16, wave);
  MemoryAccessObservation tr16_access = ds_access;
  tr16_access.mnemonic = "ds_load_tr16_b128";
  tr16_access.pc = 0x2030;
  tr16_access.elements_per_lane = 4;
  tr16_access.active_lane_mask = 0xFFFF'FFFF;
  tr16_access.valid_lane_mask = 0xFFFF'FFFF;
  tr16_access.request_lane_mask = 0xFFFF'FFFF;
  tr16_access.architectural_exec_lane_mask = 0x3;
  tr16_access.secondary_addresses = {};
  plugin.onAmdgpuMemoryAccessRouted(tr16_access);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2034, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const size_t scratch = line_with_prefix(trace, "memory 11 0 0 0 0 0 5 32 4 4 0 1 0 100 0 300");
  const size_t global = line_with_prefix(trace, "memory 11 0 0 0 0 0 2 32 4 5 0 1 0 0 200 0");
  const size_t first_lds = line_with_prefix(trace, "memory 11 0 0 0 0 1 3 32 4 3 0 1 0 400 404");
  const size_t second_lds = line_with_prefix(trace, "memory 11 0 0 0 0 1 3 32 4 3 0 1 0 800 804");
  const size_t aperture_lds =
      line_with_prefix(trace, "memory 11 0 0 0 0 2 3 32 4 3 0 1 0 900 904 0 0");
  const size_t tr16_lds = line_with_prefix(trace, "memory 11 0 0 0 0 12 3 32 16 3 0 1 0 400 404");
  ASSERT_LT(scratch, trace.size());
  ASSERT_LT(global, trace.size());
  ASSERT_LT(first_lds, trace.size());
  ASSERT_LT(second_lds, trace.size());
  ASSERT_LT(aperture_lds, trace.size());
  ASSERT_LT(tr16_lds, trace.size());
  EXPECT_LT(scratch, global);
  EXPECT_LT(global, first_lds);
  EXPECT_LT(first_lds, second_lds);
  EXPECT_LT(second_lds, aperture_lds);
  EXPECT_LT(aperture_lds, tr16_lds);
  for (uint32_t instruction_id = 3; instruction_id <= 11; ++instruction_id) {
    EXPECT_EQ(line_with_prefix(trace, "memory 11 0 0 0 0 " + std::to_string(instruction_id) + " "),
              trace.size());
  }
}

TEST_F(PerfsimPluginTest, SplitsMixedFlatResourcesInFirstRequestLaneOrder) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(13);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  const std::array<uint32_t, 1> words{0xDEAD0004};
  SyntheticInstruction first_global("flat_load_dword", words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2200, first_global, wave);
  std::array<uint64_t, 32> first_addresses{};
  first_addresses[0] = 100;
  first_addresses[1] = 500;
  first_addresses[2] = 300;
  first_addresses[3] = 900;
  MemoryAccessObservation first;
  first.mnemonic = "flat_load_dword";
  first.pc = 0x2200;
  first.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
  first.dispatch_id = info.dispatch_id;
  first.queue_id = wave.queue_id();
  first.workgroup_id = wave.wg_id();
  first.wavefront_id = wave.wf_id();
  first.process_id = wave.process_id();
  first.route = MemoryRoute::GLOBAL;
  first.decoded_space = DecodedMemorySpace::FLAT;
  first.is_load = true;
  first.wavefront_size = 32;
  first.element_size_bytes = 4;
  first.elements_per_lane = 1;
  first.active_lane_mask = 0xF;
  first.architectural_exec_lane_mask = 0xF;
  first.valid_lane_mask = 0xF;
  first.request_lane_mask = 0xF;
  first.flat_local_lane_mask = 0x2;
  first.flat_dds_lane_mask = 0x8;
  first.scratch_lane_mask = 0x4;
  first.addresses = first_addresses;
  plugin.onAmdgpuMemoryAccessRouted(first);

  SyntheticInstruction first_local("flat_load_dword", words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2204, first_local, wave);
  std::array<uint64_t, 32> rewritten_addresses{};
  std::array<uint64_t, 32> original_addresses{};
  rewritten_addresses[0] = 10;
  rewritten_addresses[1] = 20;
  rewritten_addresses[2] = 30;
  original_addresses[0] = 600;
  original_addresses[1] = 700;
  original_addresses[2] = 800;
  MemoryAccessObservation second = first;
  second.pc = 0x2204;
  second.route = MemoryRoute::LOCAL;
  second.normalized_to_local = true;
  second.active_lane_mask = 0x7;
  second.architectural_exec_lane_mask = 0x7;
  second.valid_lane_mask = 0x7;
  second.request_lane_mask = 0x7;
  second.flat_local_lane_mask = 0x1;
  second.flat_dds_lane_mask = 0;
  second.scratch_lane_mask = 0x2;
  second.addresses = rewritten_addresses;
  second.pre_routing_addresses = original_addresses;
  plugin.onAmdgpuMemoryAccessRouted(second);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2208, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const std::array<std::string_view, 6> expected{"memory 13 0 0 0 0 0 1 32 4 5 0 1 0 100 0 0 0",
                                                 "memory 13 0 0 0 0 0 10 32 4 3 0 1 0 0 500 0 900",
                                                 "memory 13 0 0 0 0 0 4 32 4 4 0 1 0 0 0 300",
                                                 "memory 13 0 0 0 0 1 1 32 4 3 0 1 0 600 0 0",
                                                 "memory 13 0 0 0 0 1 2 32 4 4 0 1 0 0 700 0",
                                                 "memory 13 0 0 0 0 1 4 32 4 5 0 1 0 0 0 800"};
  size_t position = 0;
  for (std::string_view prefix : expected) {
    position = line_with_prefix(trace, prefix, position);
    ASSERT_LT(position, trace.size()) << prefix;
    ++position;
  }
}

TEST_F(PerfsimPluginTest, ReportsFlatDdsLoadsAsLdsWithOriginalAddresses) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(15);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  const std::array<uint32_t, 1> words{0xDEAD0015};
  SyntheticInstruction load("flat_load_dword", words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2280, load, wave);

  constexpr uint64_t kDdsAddress = 0x1234'0000'8000'0040;
  std::array<uint64_t, 32> effective_addresses{};
  std::array<uint64_t, 32> original_addresses{};
  effective_addresses[0] = 0x8000'0040;
  original_addresses[0] = kDdsAddress;
  MemoryAccessObservation access;
  access.mnemonic = "flat_load_dword";
  access.pc = 0x2280;
  access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
  access.dispatch_id = info.dispatch_id;
  access.queue_id = wave.queue_id();
  access.workgroup_id = wave.wg_id();
  access.wavefront_id = wave.wf_id();
  access.process_id = wave.process_id();
  access.route = MemoryRoute::LOCAL;
  access.decoded_space = DecodedMemorySpace::FLAT;
  access.normalized_to_local = true;
  access.is_load = true;
  access.wavefront_size = 32;
  access.element_size_bytes = 4;
  access.elements_per_lane = 1;
  access.active_lane_mask = 1;
  access.architectural_exec_lane_mask = 1;
  access.valid_lane_mask = 1;
  access.request_lane_mask = 1;
  access.flat_dds_lane_mask = 1;
  access.addresses = effective_addresses;
  access.pre_routing_addresses = original_addresses;
  plugin.onAmdgpuMemoryAccessRouted(access);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2284, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const std::string memory_prefix =
      "memory 15 0 0 0 0 0 1 32 4 3 0 1 0 " + std::to_string(kDdsAddress);
  EXPECT_NE(line_with_prefix(trace, memory_prefix), trace.size());
}

TEST_F(PerfsimPluginTest, RejectsWholeDispatchForFlatDdsStoresAndAtomics) {
  struct FailureCase {
    uint32_t dispatch_id;
    const char *mnemonic;
    bool is_load;
    AtomicOp atomic_op;
  };
  constexpr std::array<FailureCase, 2> cases{{
      {16, "flat_store_dword", false, AtomicOp::NONE},
      {17, "flat_atomic_add", true, AtomicOp::ADD},
  }};

  testing::internal::CaptureStderr();
  for (const auto &test : cases) {
    WaveFixture fixture;
    const std::string config = plugin_config();
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    const KernelDispatchInfo info = dispatch_info(test.dispatch_id);
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 1> words{0xDEAD0016};
    SyntheticInstruction instruction(test.mnemonic, words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(0x22C0, instruction, wave);
    std::array<uint64_t, 32> effective_addresses{};
    std::array<uint64_t, 32> original_addresses{};
    effective_addresses[0] = 0x8000'0040;
    original_addresses[0] = 0x1234'0000'8000'0040;
    MemoryAccessObservation access;
    access.mnemonic = test.mnemonic;
    access.pc = 0x22C0;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.route = MemoryRoute::LOCAL;
    access.decoded_space = DecodedMemorySpace::FLAT;
    access.normalized_to_local = true;
    access.is_load = test.is_load;
    access.atomic_op = test.atomic_op;
    access.wavefront_size = 32;
    access.element_size_bytes = 4;
    access.elements_per_lane = 1;
    access.active_lane_mask = 1;
    access.architectural_exec_lane_mask = 1;
    access.valid_lane_mask = 1;
    access.request_lane_mask = 1;
    access.flat_dds_lane_mask = 1;
    access.addresses = effective_addresses;
    access.pre_routing_addresses = original_addresses;
    plugin.onAmdgpuMemoryAccessRouted(access);

    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0x22C4, end, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), cases.size());
  EXPECT_NE(diagnostic.find("FLAT DDS store or atomic"), std::string::npos);

  const auto trace = lines(read_file(trace_.path()));
  for (const auto &test : cases) {
    const std::string dispatch = std::to_string(test.dispatch_id);
    EXPECT_EQ(line_with_prefix(trace, "begin " + dispatch + " "), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "instruction " + dispatch + " "), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "memory " + dispatch + " "), trace.size());
    EXPECT_EQ(line_with_prefix(trace, "end " + dispatch + " "), trace.size());
  }
}

TEST_F(PerfsimPluginTest, MatchesFfmRegularMemoryCallbackSizes) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(14);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  wave.set_lds_base(0x1000);
  plugin.onAmdgpuWavefrontDispatched(wave);

  struct SizeCase {
    const char *mnemonic;
    DecodedMemorySpace space;
    uint32_t element_size;
    uint32_t elements;
    uint32_t expected_size;
  };
  constexpr std::array<SizeCase, 9> cases{{
      {"ds_load_u8", DecodedMemorySpace::LOCAL, 1, 1, 4},
      {"ds_load_u16", DecodedMemorySpace::LOCAL, 2, 1, 4},
      {"global_load_u8", DecodedMemorySpace::GLOBAL, 1, 1, 4},
      {"global_load_u16", DecodedMemorySpace::GLOBAL, 2, 1, 4},
      {"buffer_load_u8", DecodedMemorySpace::GLOBAL, 1, 1, 4},
      {"buffer_load_u16", DecodedMemorySpace::GLOBAL, 2, 1, 4},
      {"tbuffer_load_format_d16_xyz", DecodedMemorySpace::GLOBAL, 2, 3, 4},
      {"flat_load_u8", DecodedMemorySpace::FLAT, 1, 1, 1},
      {"flat_load_u16", DecodedMemorySpace::FLAT, 2, 1, 2},
  }};

  const std::array<uint32_t, 1> words{0xDEAD0014};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto &test = cases[i];
    const uint64_t pc = 0x2300 + i * 4;
    SyntheticInstruction instruction(test.mnemonic, words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(pc, instruction, wave);

    std::array<uint64_t, 32> addresses{};
    addresses[0] = test.space == DecodedMemorySpace::LOCAL ? 0x1100 : 0x2000;
    MemoryAccessObservation access;
    access.mnemonic = test.mnemonic;
    access.pc = pc;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.route =
        test.space == DecodedMemorySpace::LOCAL ? MemoryRoute::LOCAL : MemoryRoute::GLOBAL;
    access.decoded_space = test.space;
    access.is_load = true;
    access.wavefront_size = 32;
    access.element_size_bytes = test.element_size;
    access.elements_per_lane = test.elements;
    access.active_lane_mask = 1;
    access.architectural_exec_lane_mask = 1;
    access.valid_lane_mask = 1;
    access.request_lane_mask = 1;
    access.addresses = addresses;
    plugin.onAmdgpuMemoryAccessRouted(access);
  }

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2400, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  for (size_t i = 0; i < cases.size(); ++i) {
    const std::string prefix = "memory 14 0 0 0 0 " + std::to_string(i) + " 1 32 " +
                               std::to_string(cases[i].expected_size) + " ";
    EXPECT_NE(line_with_prefix(trace, prefix), trace.size()) << cases[i].mnemonic;
  }
}

TEST_F(PerfsimPluginTest, ForwardsNominalWidthsForPartialOobVbufferAccesses) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(15);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  struct PartialOobCase {
    const char *mnemonic;
    uint32_t elements;
    uint32_t expected_size;
    std::array<uint64_t, 4> element_lane_masks;
  };
  constexpr std::array<PartialOobCase, 3> cases{{
      {"buffer_load_b64", 2, 8, {0x3, 0x1, 0, 0}},
      {"buffer_load_b96", 3, 12, {0x3, 0x1, 0x1, 0}},
      {"buffer_load_b128", 4, 16, {0x3, 0x3, 0x1, 0x1}},
  }};

  const std::array<uint32_t, 1> words{0xDEAD0015};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto &test = cases[i];
    const uint64_t pc = 0x2500 + i * 4;
    SyntheticInstruction instruction(test.mnemonic, words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(pc, instruction, wave);

    std::array<uint64_t, 32> addresses{};
    addresses[0] = 0x2000 + i * 0x100;
    addresses[1] = 0x3000 + i * 0x100;
    MemoryAccessObservation access;
    access.mnemonic = test.mnemonic;
    access.pc = pc;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.route = MemoryRoute::GLOBAL;
    access.decoded_space = DecodedMemorySpace::GLOBAL;
    access.is_load = true;
    access.wavefront_size = 32;
    access.element_size_bytes = 4;
    access.elements_per_lane = test.elements;
    access.active_lane_mask = 0x3;
    access.architectural_exec_lane_mask = 0x3;
    access.valid_lane_mask = 0x3;
    access.request_lane_mask = 0x3;
    access.addresses = addresses;
    access.element_lane_masks =
        std::span<const uint64_t>(test.element_lane_masks).first(test.elements);
    plugin.onAmdgpuMemoryAccessRouted(access);
  }

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x2600, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                          [](const std::string &line) {
                            return std::string_view(line).starts_with("memory 15 ");
                          }),
            cases.size());
  for (size_t i = 0; i < cases.size(); ++i) {
    const std::string prefix = "memory 15 0 0 0 0 " + std::to_string(i) + " 3 32 " +
                               std::to_string(cases[i].expected_size) + " 5 0 1 0 " +
                               std::to_string(0x2000 + i * 0x100) + " " +
                               std::to_string(0x3000 + i * 0x100);
    EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                            [&](const std::string &line) {
                              return std::string_view(line).starts_with(prefix);
                            }),
              1)
        << cases[i].mnemonic;
  }
}

TEST_F(PerfsimPluginTest, RejectsWholeDispatchWhenFlatAtomicResolvesToScratch) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    const KernelDispatchInfo info = dispatch_info(12);
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 1> atomic_words{0xDEAD0003};
    SyntheticInstruction atomic("flat_atomic_add", atomic_words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(0x2100, atomic, wave);
    std::array<uint64_t, 32> addresses{};
    addresses[0] = 0xABC000;
    MemoryAccessObservation access;
    access.mnemonic = "flat_atomic_add";
    access.pc = 0x2100;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.route = MemoryRoute::GLOBAL;
    access.decoded_space = DecodedMemorySpace::FLAT;
    access.is_load = true;
    access.atomic_op = AtomicOp::ADD;
    access.wavefront_size = 32;
    access.element_size_bytes = 4;
    access.elements_per_lane = 1;
    access.active_lane_mask = 1;
    access.architectural_exec_lane_mask = 1;
    access.valid_lane_mask = 1;
    access.request_lane_mask = 1;
    access.scratch_lane_mask = 1;
    access.addresses = addresses;
    plugin.onAmdgpuMemoryAccessRouted(access);

    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0x2104, end, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("FLAT atomic resolving to scratch"), std::string::npos);

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(trace, "begin 12 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "instruction 12 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "memory 12 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "end 12 "), trace.size());
}

TEST_F(PerfsimPluginTest, ReplaysInterleavedDispatchesAsOneGloballyOrderedEpoch) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  for (uint32_t id : {21u, 22u}) {
    plugin.onAmdgpuDispatchPacketProcessed(dispatch_info(id));
    plugin.onAmdgpuDispatchExecutionBegin(id);
  }
  Wavefront &first = fixture.wave(21, 1, {1, 0, 0}, 0);
  Wavefront &second = fixture.wave(22, 2, {2, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(first);
  plugin.onAmdgpuWavefrontDispatched(second);

  const std::array<uint32_t, 1> words{0xBF810000};
  SyntheticInstruction end("s_endpgm", words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x3000, end, first);
  plugin.onAmdgpuBeforeExecuteInstruction(0x4000, end, second);
  plugin.onAmdgpuWavefrontHalted(first);
  plugin.onAmdgpuWavefrontHalted(second);
  plugin.onAmdgpuDispatchExecutionEnd(21);
  const auto partial_trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(partial_trace, "begin "), partial_trace.size());
  plugin.onAmdgpuDispatchExecutionEnd(22);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const size_t begin_first = line_with_prefix(trace, "begin 21 ");
  const size_t begin_second = line_with_prefix(trace, "begin 22 ");
  const size_t instruction_first = line_with_prefix(trace, "instruction 21 ");
  const size_t instruction_second = line_with_prefix(trace, "instruction 22 ");
  const size_t end_first = line_with_prefix(trace, "end 21 ");
  const size_t end_second = line_with_prefix(trace, "end 22 ");
  ASSERT_LT(begin_first, trace.size());
  ASSERT_LT(begin_second, trace.size());
  ASSERT_LT(instruction_first, trace.size());
  ASSERT_LT(instruction_second, trace.size());
  ASSERT_LT(end_first, trace.size());
  ASSERT_LT(end_second, trace.size());
  EXPECT_LT(begin_first, begin_second);
  EXPECT_LT(begin_second, instruction_first);
  EXPECT_LT(instruction_first, instruction_second);
  EXPECT_LT(instruction_second, end_first);
  EXPECT_LT(end_first, end_second);
}

TEST_F(PerfsimPluginTest, ReplaysLargeEpochInOrder) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(24);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  constexpr uint32_t kAddCount = 4096;
  constexpr uint64_t kFirstPc = 0x8000;
  const std::array<uint32_t, 1> add_words{0x7E000200};
  SyntheticInstruction add("v_add_f32", add_words);
  for (uint32_t i = 0; i < kAddCount; ++i)
    plugin.onAmdgpuBeforeExecuteInstruction(kFirstPc + i * 4, add, wave);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(kFirstPc + kAddCount * 4, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const auto count_prefix = [&](std::string_view prefix) {
    return std::count_if(trace.begin(), trace.end(), [&](const std::string &line) {
      return std::string_view(line).starts_with(prefix);
    });
  };
  EXPECT_EQ(count_prefix("instruction 24 "), kAddCount + 1);
  const size_t first = line_with_prefix(trace, "instruction 24 0 0 0 0 0 32768 ");
  const size_t last = line_with_prefix(trace, "instruction 24 0 0 0 0 4096 49152 ");
  ASSERT_LT(first, trace.size());
  ASSERT_LT(last, trace.size());
  EXPECT_LT(first, last);
  EXPECT_LT(last, line_with_prefix(trace, "end 24 "));
}

TEST_F(PerfsimPluginTest, CompactsInterleavedChunksWithoutReorderingSurvivors) {
  WaveFixture fixture;
  const std::string config = plugin_config_with_staging_budget(8 * 1024 * 1024);
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  for (uint32_t id : {33u, 34u}) {
    plugin.onAmdgpuDispatchPacketProcessed(dispatch_info(id));
    plugin.onAmdgpuDispatchExecutionBegin(id);
  }
  Wavefront &rejected_wave = fixture.wave(33, 0, {0, 0, 0}, 0);
  Wavefront &supported_wave = fixture.wave(34, 1, {1, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(rejected_wave);
  plugin.onAmdgpuWavefrontDispatched(supported_wave);

  constexpr uint32_t kAddCount = 2050;
  constexpr uint64_t kRejectedFirstPc = 0x10000;
  constexpr uint64_t kSupportedFirstPc = 0x20000;
  const std::array<uint32_t, 1> add_words{0x7E000200};
  SyntheticInstruction add("v_add_f32", add_words);
  for (uint32_t i = 0; i < kAddCount; ++i) {
    plugin.onAmdgpuBeforeExecuteInstruction(kRejectedFirstPc + i * 4, add, rejected_wave);
    plugin.onAmdgpuBeforeExecuteInstruction(kSupportedFirstPc + i * 4, add, supported_wave);
  }

  testing::internal::CaptureStderr();
  SyntheticInstruction scratch("scratch_load_dword", add_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x30000, scratch, rejected_wave);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  const uint64_t supported_end_pc = kSupportedFirstPc + kAddCount * 4;
  plugin.onAmdgpuBeforeExecuteInstruction(supported_end_pc, end, supported_wave);
  plugin.onAmdgpuWavefrontHalted(supported_wave);
  plugin.onAmdgpuDispatchExecutionEnd(34);
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("dedicated SCRATCH instructions are not FFM-compatible"),
            std::string::npos);
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), 1);

  // The two dispatches cross the 4096-event chunk boundary before dispatch 33
  // is rejected. Its removal compacts dispatch 34's records across chunks.
  const auto trace = lines(read_file(trace_.path()));
  const auto count_prefix = [&](std::string_view prefix) {
    return std::count_if(trace.begin(), trace.end(), [&](const std::string &line) {
      return std::string_view(line).starts_with(prefix);
    });
  };
  EXPECT_EQ(count_prefix("instruction 34 "), kAddCount + 1);
  EXPECT_EQ(count_prefix("begin 33 "), 0);
  EXPECT_EQ(count_prefix("instruction 33 "), 0);
  EXPECT_EQ(count_prefix("end 33 "), 0);
  const size_t begin = line_with_prefix(trace, "begin 34 ");
  const size_t first = line_with_prefix(trace, "instruction 34 1 0 0 0 0 " +
                                                   std::to_string(kSupportedFirstPc) + " ");
  const size_t last =
      line_with_prefix(trace, "instruction 34 1 0 0 0 " + std::to_string(kAddCount) + " " +
                                  std::to_string(supported_end_pc) + " ");
  const size_t dispatch_end = line_with_prefix(trace, "end 34 ");
  ASSERT_LT(begin, trace.size());
  ASSERT_LT(first, trace.size());
  ASSERT_LT(last, trace.size());
  ASSERT_LT(dispatch_end, trace.size());
  EXPECT_LT(begin, first);
  EXPECT_LT(first, last);
  EXPECT_LT(last, dispatch_end);

  plugin.onAmdgpuWavefrontHalted(rejected_wave);
  plugin.onAmdgpuDispatchExecutionEnd(33);
  plugin.onShutdown();
}

TEST_F(PerfsimPluginTest, RejectsLongLivedDispatchAtStagingBudgetAndReusesBudget) {
  WaveFixture fixture;
  const std::string config = plugin_config_with_staging_budget(4096);
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    const KernelDispatchInfo rejected_info = dispatch_info(25);
    plugin.onAmdgpuDispatchPacketProcessed(rejected_info);
    plugin.onAmdgpuDispatchExecutionBegin(rejected_info.dispatch_id);
    Wavefront &rejected_wave = fixture.wave(rejected_info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(rejected_wave);

    const std::array<uint32_t, 1> add_words{0x7E000200};
    SyntheticInstruction add("v_add_f32", add_words);
    for (uint32_t i = 0; i < 512; ++i)
      plugin.onAmdgpuBeforeExecuteInstruction(0x9000 + i * 4, add, rejected_wave);

    plugin.onAmdgpuWavefrontHalted(rejected_wave);
    plugin.onAmdgpuDispatchExecutionEnd(rejected_info.dispatch_id);

    const KernelDispatchInfo supported_info = dispatch_info(26);
    plugin.onAmdgpuDispatchPacketProcessed(supported_info);
    plugin.onAmdgpuDispatchExecutionBegin(supported_info.dispatch_id);
    Wavefront &supported_wave = fixture.wave(supported_info.dispatch_id, 1, {1, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(supported_wave);
    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0xA000, end, supported_wave);
    plugin.onAmdgpuWavefrontHalted(supported_wave);
    plugin.onAmdgpuDispatchExecutionEnd(supported_info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  const size_t budget_diagnostic = diagnostic.find("staging budget of 4096 bytes exceeded");
  ASSERT_NE(budget_diagnostic, std::string::npos);
  EXPECT_EQ(diagnostic.find("staging budget", budget_diagnostic + 1), std::string::npos);

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(trace, "begin 25 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "instruction 25 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "end 25 "), trace.size());
  EXPECT_NE(line_with_prefix(trace, "begin 26 "), trace.size());
  EXPECT_NE(line_with_prefix(trace, "instruction 26 "), trace.size());
  EXPECT_NE(line_with_prefix(trace, "end 26 "), trace.size());
}

TEST_F(PerfsimPluginTest, RejectedDispatchDoesNotBlockOverlappingSupportedDispatch) {
  WaveFixture fixture;
  const std::string config = plugin_config_with_staging_budget(4096);
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  for (uint32_t id : {27u, 28u}) {
    plugin.onAmdgpuDispatchPacketProcessed(dispatch_info(id));
    plugin.onAmdgpuDispatchExecutionBegin(id);
  }
  Wavefront &rejected_wave = fixture.wave(27, 0, {0, 0, 0}, 0);
  Wavefront &supported_wave = fixture.wave(28, 1, {1, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(rejected_wave);
  plugin.onAmdgpuWavefrontDispatched(supported_wave);

  const std::array<uint32_t, 1> add_words{0x7E000200};
  SyntheticInstruction add("v_add_f32", add_words);
  testing::internal::CaptureStderr();
  for (uint32_t i = 0; i < 128; ++i)
    plugin.onAmdgpuBeforeExecuteInstruction(0xB000 + i * 4, add, rejected_wave);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0xC000, end, supported_wave);
  plugin.onAmdgpuWavefrontHalted(supported_wave);
  plugin.onAmdgpuDispatchExecutionEnd(28);
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("staging budget of 4096 bytes exceeded"), std::string::npos);
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), 1);

  // Dispatch 27 is still executing, but once rejected it must retain no
  // events and must not hold dispatch 28's completed epoch in memory.
  const auto partial_trace = lines(read_file(trace_.path()));
  const size_t begin = line_with_prefix(partial_trace, "begin 28 ");
  const size_t instruction = line_with_prefix(partial_trace, "instruction 28 ");
  const size_t dispatch_end = line_with_prefix(partial_trace, "end 28 ");
  ASSERT_LT(begin, partial_trace.size());
  ASSERT_LT(instruction, partial_trace.size());
  ASSERT_LT(dispatch_end, partial_trace.size());
  EXPECT_LT(begin, instruction);
  EXPECT_LT(instruction, dispatch_end);
  EXPECT_EQ(line_with_prefix(partial_trace, "begin 27 "), partial_trace.size());
  EXPECT_EQ(line_with_prefix(partial_trace, "instruction 27 "), partial_trace.size());
  EXPECT_EQ(line_with_prefix(partial_trace, "end 27 "), partial_trace.size());

  plugin.onAmdgpuWavefrontHalted(rejected_wave);
  plugin.onAmdgpuDispatchExecutionEnd(27);
  plugin.onShutdown();
}

TEST_F(PerfsimPluginTest, IgnoresCallbacksAfterEarlyDispatchRejection) {
  WaveFixture fixture;
  const std::string config = plugin_config_with_staging_budget(4096);
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    KernelDispatchInfo rejected_info = dispatch_info(29);
    rejected_info.code_target = ROCJITSU_CODE_TARGET_GFX950;
    plugin.onAmdgpuDispatchPacketProcessed(rejected_info);
    plugin.onAmdgpuDispatchExecutionBegin(rejected_info.dispatch_id);
    Wavefront &rejected_wave = fixture.wave(rejected_info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(rejected_wave);

    const std::array<uint32_t, 1> add_words{0x7E000200};
    SyntheticInstruction add("v_add_f32", add_words);
    for (uint32_t i = 0; i < 512; ++i)
      plugin.onAmdgpuBeforeExecuteInstruction(0xD000 + i * 4, add, rejected_wave);

    const KernelDispatchInfo supported_info = dispatch_info(30);
    plugin.onAmdgpuDispatchPacketProcessed(supported_info);
    plugin.onAmdgpuDispatchExecutionBegin(supported_info.dispatch_id);
    Wavefront &supported_wave = fixture.wave(supported_info.dispatch_id, 1, {1, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(supported_wave);
    const std::array<uint32_t, 1> end_words{0xBF810000};
    SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0xE000, end, supported_wave);
    plugin.onAmdgpuWavefrontHalted(supported_wave);
    plugin.onAmdgpuDispatchExecutionEnd(supported_info.dispatch_id);

    const auto partial_trace = lines(read_file(trace_.path()));
    EXPECT_NE(line_with_prefix(partial_trace, "begin 30 "), partial_trace.size());
    EXPECT_NE(line_with_prefix(partial_trace, "instruction 30 "), partial_trace.size());
    EXPECT_NE(line_with_prefix(partial_trace, "end 30 "), partial_trace.size());
    EXPECT_EQ(line_with_prefix(partial_trace, "begin 29 "), partial_trace.size());
    EXPECT_EQ(line_with_prefix(partial_trace, "instruction 29 "), partial_trace.size());
    EXPECT_EQ(line_with_prefix(partial_trace, "end 29 "), partial_trace.size());

    plugin.onAmdgpuWavefrontHalted(rejected_wave);
    plugin.onAmdgpuDispatchExecutionEnd(rejected_info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  const size_t target_diagnostic = diagnostic.find("code target is not gfx1250");
  ASSERT_NE(target_diagnostic, std::string::npos);
  EXPECT_EQ(diagnostic.find("code target is not gfx1250", target_diagnostic + 1),
            std::string::npos);
  EXPECT_EQ(diagnostic.find("staging budget"), std::string::npos);
}

TEST_F(PerfsimPluginTest, RejectsTensorDmaBeforeMaterializingAddressesOverBudget) {
  WaveFixture fixture;
  const std::string config = plugin_config_with_staging_budget(4096);
  bool addresses_materialized = false;
  struct AddressSource {
    bool *materialized = nullptr;
  } source{&addresses_materialized};
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();

    const KernelDispatchInfo info = dispatch_info(32);
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);

    const std::array<uint32_t, 3> tensor_words{0xD0710001, 0x7C000000, 0x18140C00};
    SyntheticInstruction tensor("tensor_load_to_lds", tensor_words, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(0xF000, tensor, wave);
    TensorDmaMemoryAccessObservation access;
    access.mnemonic = "tensor_load_to_lds";
    access.pc = 0xF000;
    access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    access.dispatch_id = info.dispatch_id;
    access.queue_id = wave.queue_id();
    access.workgroup_id = wave.wg_id();
    access.wavefront_id = wave.wf_id();
    access.process_id = wave.process_id();
    access.element_size_bytes = 4;
    access.tile_dim0 = 32;
    access.tile_dim1 = 32;
    access.data_size = 2;
    access.tensor_dim0_stride = 128;
    access.tensor_dim1_stride = 4096;
    access.is_load = true;
    access.addresses = TensorDmaAddressView::deferred(
        1024, &source, [](const void *context, std::span<uint64_t> destination) {
          *static_cast<const AddressSource *>(context)->materialized = true;
          std::fill(destination.begin(), destination.end(), 0x100000);
          return true;
        });
    plugin.onAmdgpuTensorDmaMemoryAccess(access);

    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(addresses_materialized);
  EXPECT_NE(diagnostic.find("staging budget of 4096 bytes exceeded"), std::string::npos);
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), 1);
  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(trace, "begin 32 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "instruction 32 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "tdm 32 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "end 32 "), trace.size());
}

TEST_F(PerfsimPluginTest, PreservesCollidingWrappedFfmWaveIdentities) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(23);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &first = fixture.wave(info.dispatch_id, 1, {1000, 0, 0}, 0);
  Wavefront &second = fixture.wave(info.dispatch_id, 2, {0, 1, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(first);
  plugin.onAmdgpuWavefrontDispatched(second);

  const std::array<uint32_t, 1> words{0xBF810000};
  SyntheticInstruction end("s_endpgm", words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x4100, end, first);
  plugin.onAmdgpuBeforeExecuteInstruction(0x4200, end, second);
  plugin.onAmdgpuWavefrontHalted(first);
  plugin.onAmdgpuWavefrontHalted(second);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const auto count_prefix = [&](std::string_view prefix) {
    return std::count_if(trace.begin(), trace.end(), [&](const std::string &line) {
      return std::string_view(line).starts_with(prefix);
    });
  };
  EXPECT_EQ(count_prefix("instruction 23 1000 0 0 0 0 "), 2);
  EXPECT_NE(line_with_prefix(trace, "end 23 "), trace.size());
}

TEST_F(PerfsimPluginTest, RejectsWholeUnsupportedDispatchWithoutBackendCallbacks) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  testing::internal::CaptureStderr();
  {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();
    KernelDispatchInfo info = dispatch_info(31);
    info.code_target = ROCJITSU_CODE_TARGET_GFX950;
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);
    const std::array<uint32_t, 1> words{0xBF810000};
    SyntheticInstruction end("s_endpgm", words, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(0x5000, end, wave);
    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
    plugin.onShutdown();
  }
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_NE(diagnostic.find("skipped dispatch 31"), std::string::npos);
  EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), '\n'), 1);

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(trace, "begin 31 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "instruction 31 "), trace.size());
  EXPECT_EQ(line_with_prefix(trace, "end 31 "), trace.size());
}

TEST_F(PerfsimPluginTest, ReusesOnlyConsecutiveRepeatedWaitOrdinals) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(35);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  const std::array<uint32_t, 1> wait_words{0xBF8C0000};
  SyntheticInstruction wait("s_wait_loadcnt", wait_words);
  const std::array<uint32_t, 4> first_fetch{wait_words[0], 0x11111111, 0x22222222, 0x33333333};
  const std::array<uint32_t, 4> retry_fetch{wait_words[0], 0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC};
  plugin.onAmdgpuBeforeExecuteInstruction(0x6000, wait, wave, first_fetch);
  plugin.onAmdgpuBeforeExecuteInstruction(0x6000, wait, wave, retry_fetch);

  const std::array<uint32_t, 1> add_words{0x7E000200};
  SyntheticInstruction add("v_add_f32", add_words);
  plugin.onAmdgpuBeforeExecuteInstruction(0x6004, add, wave);
  plugin.onAmdgpuBeforeExecuteInstruction(0x6004, add, wave);
  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x6008, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  const auto count_prefix = [&](std::string_view prefix) {
    return std::count_if(trace.begin(), trace.end(), [&](const std::string &line) {
      return std::string_view(line).starts_with(prefix);
    });
  };
  EXPECT_EQ(count_prefix("instruction 35 0 0 0 0 0 24576 "), 2);
  EXPECT_NE(line_with_prefix(trace,
                             "instruction 35 0 0 0 0 0 24576 bf8c0000 11111111 22222222 33333333 "),
            trace.size());
  EXPECT_NE(line_with_prefix(trace,
                             "instruction 35 0 0 0 0 0 24576 bf8c0000 aaaaaaaa bbbbbbbb cccccccc "),
            trace.size());
  EXPECT_EQ(count_prefix("instruction 35 0 0 0 0 1 24580 "), 1);
  EXPECT_EQ(count_prefix("instruction 35 0 0 0 0 2 24580 "), 1);
}

TEST_F(PerfsimPluginTest, MatchesFfmSwmmacSmemAndClusterInstructionCounters) {
  WaveFixture fixture;
  const std::string config = plugin_config();
  PerfsimPlugin plugin(config.c_str());
  plugin.onInit();

  const KernelDispatchInfo info = dispatch_info(36);
  plugin.onAmdgpuDispatchPacketProcessed(info);
  plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
  Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
  plugin.onAmdgpuWavefrontDispatched(wave);

  const std::array<uint32_t, 1> swmmac_words{0xDEAD3601};
  SyntheticInstruction swmmac("v_swmmac_f32_16x16x16_f16", swmmac_words);
  plugin.onAmdgpuBeforeExecuteInstruction(0x7000, swmmac, wave);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  const auto smem_words = cdna5::build_smem(cdna5::kSDcacheInvSmem);
  DecodeResult smem_result = decoder->decode(smem_words.data());
  ASSERT_FALSE(smem_result.failed());
  std::unique_ptr<Instruction> smem_cache = std::move(smem_result).value();
  ASSERT_EQ(smem_cache->mnemonic(), "s_dcache_inv");
  ASSERT_FALSE(smem_cache->is_memory_op());
  plugin.onAmdgpuBeforeExecuteInstruction(0x7004, *smem_cache, wave);

  const std::array<uint32_t, 1> cluster_words{0xDEAD3603};
  SyntheticInstruction cluster_load("cluster_load_b128", cluster_words, MEMORY_OP);
  plugin.onAmdgpuBeforeExecuteInstruction(0x7008, cluster_load, wave);

  const std::array<uint32_t, 1> end_words{0xBF810000};
  SyntheticInstruction end("s_endpgm", end_words, PROGRAM_TERMINATOR);
  plugin.onAmdgpuBeforeExecuteInstruction(0x700C, end, wave);
  plugin.onAmdgpuWavefrontHalted(wave);
  plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);
  plugin.onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_NE(line_with_prefix(trace,
                             "instruction 36 0 0 0 0 0 28672 dead3601 0 0 0 1 0 0 0 0 0 0 0 1 0 0"),
            trace.size());
  const size_t smem_instruction = line_with_prefix(trace, "instruction 36 0 0 0 0 1 28676 ");
  ASSERT_LT(smem_instruction, trace.size());
  EXPECT_TRUE(std::string_view(trace[smem_instruction]).ends_with(" 0 0 1 0 0 0 0 0 0 0 0"));
  EXPECT_NE(line_with_prefix(trace,
                             "instruction 36 0 0 0 0 2 28680 dead3603 0 0 0 0 0 0 0 0 1 1 0 0 0 0"),
            trace.size());
}

TEST_F(PerfsimPluginTest, ConstructorFailuresReleaseSingletonClaimAndRetainBackend) {
  const std::string config = plugin_config();
  for (const char *mode :
       {"reject_all", "too_old_version", "bad_version", "missing_on_init",
        "missing_on_dispatch_begin", "missing_on_dispatch_end", "missing_on_instruction",
        "missing_on_memory_access", "missing_on_tdm_memory_access", "missing_on_shutdown"}) {
    setenv("ROCJITSU_PERFSIM_FAKE_MODE", mode, 1);
    EXPECT_THROW(PerfsimPlugin(config.c_str()), std::runtime_error) << mode;
  }

  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "", 1);
  auto first = std::make_unique<PerfsimPlugin>(config.c_str());
  EXPECT_THROW(PerfsimPlugin(config.c_str()), std::runtime_error);
  first.reset();
  EXPECT_NO_THROW(PerfsimPlugin(config.c_str()));
  EXPECT_THROW(PerfsimPlugin(R"({"library_path":"/does/not/exist.so"})"), std::runtime_error);
  EXPECT_THROW(PerfsimPlugin("{}"), std::invalid_argument);
  EXPECT_NO_THROW(PerfsimPlugin(config.c_str()));
  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(line_with_prefix(trace, "unload"), trace.size());
}

TEST_F(PerfsimPluginTest, SequentialInstancesShutdownOnceWithoutUnloadingBackend) {
  const std::string config = plugin_config();
  for (int i = 0; i < 2; ++i) {
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();
    plugin.onShutdown();
  }

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 13"), 2);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "shutdown"), 2);
  EXPECT_EQ(line_with_prefix(trace, "unload"), trace.size());
}

TEST_F(PerfsimPluginTest, LoadsAsRocjitsuPluginWithStagingBudgetSchema) {
  const std::string config =
      std::string{"{\"plugins\":{\"perfsim\":"} + plugin_config_with_staging_budget(4096) + "}}";
  ExecutionPluginGroup group(PluginSinkConfig{});
  testing::internal::CaptureStderr();
  const size_t loaded = PluginLoader::load_from_config(config, group, PERFSIM_PLUGIN_DIR);
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  ASSERT_EQ(loaded, 1);
  EXPECT_EQ(diagnostic.find("not in schema"), std::string::npos);
  ASSERT_EQ(group.num_plugins(), 1u);
  group.onInit();
  group.onShutdown();
  EXPECT_NE(read_file(trace_.path()).find("init 13\nshutdown\n"), std::string::npos);
}

TEST_F(PerfsimPluginTest, RoutesV13HostLogThroughConfiguredSink) {
  const std::string config = std::string{"{\"plugins\":{\"perfsim\":"} + plugin_config() + "}}";
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "host_log", 1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_EQ(PluginLoader::load_from_config(config, group, PERFSIM_PLUGIN_DIR), 1);
  group.onInit();
  group.onShutdown();

  EXPECT_NE(sink.str().find("[perfsim:warn] fake backend initialized\n"), std::string::npos);
  EXPECT_NE(sink.str().find("[perfsim:error] fake backend shutting down\n"), std::string::npos);
}

TEST_F(PerfsimPluginTest, SerializesBackendWorkerAndPeerPluginAtSharedSink) {
  const std::string config = std::string{"{\"plugins\":{\"perfsim\":"} + plugin_config() + "}}";
  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "async_host_log", 1);
  PluginSinkConfig sink_config;
  BlockingOverlapSink &sink = sink_config.emplace<BlockingOverlapSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_EQ(PluginLoader::load_from_config(config, group, PERFSIM_PLUGIN_DIR), 1);
  auto peer = std::make_unique<InitSinkWriterPlugin>(sink);
  InitSinkWriterPlugin *peer_ptr = peer.get();
  ASSERT_TRUE(group.add(std::move(peer)));

  std::thread init_thread([&]() { group.onInit(); });
  const bool first_entered = sink.wait_for_first(std::chrono::seconds(5));
  const bool peer_started = peer_ptr->wait_for_write_start(std::chrono::seconds(5));
  const bool overlap_observed =
      peer_started && sink.wait_for_overlap(std::chrono::milliseconds(100));
  sink.release_first();
  init_thread.join();
  group.onShutdown();

  EXPECT_TRUE(first_entered);
  EXPECT_TRUE(peer_started);
  EXPECT_FALSE(overlap_observed);
  EXPECT_EQ(sink.max_active_writes(), 1);
}

TEST_F(PerfsimPluginTest, RequiredLoaderRetriesAfterBackendFactoryFailure) {
  const std::string config =
      std::string{"{\"require_all_plugins\":true,\"plugins\":{\"perfsim\":"} + plugin_config() +
      "}}";

  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "reject_all", 1);
  EXPECT_THROW(PluginLoader::configure_plugin_group(config, PERFSIM_PLUGIN_DIR),
               std::runtime_error);

  setenv("ROCJITSU_PERFSIM_FAKE_MODE", "", 1);
  auto group = PluginLoader::configure_plugin_group(config, PERFSIM_PLUGIN_DIR);
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->num_plugins(), 1u);
  group->onInit();
  group->onShutdown();

  const auto trace = lines(read_file(trace_.path()));
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "init 13"), 1);
  EXPECT_EQ(std::count(trace.begin(), trace.end(), "shutdown"), 1);
  EXPECT_EQ(line_with_prefix(trace, "unload"), trace.size());
}

TEST_F(PerfsimPluginTest, RealBackendMatchesDirectFfmForCanonicalStream) {
  const char *backend_path_env = std::getenv("ROCJITSU_PERFSIM_REAL_BACKEND");
  if (!backend_path_env || backend_path_env[0] == '\0')
    GTEST_SKIP() << "set ROCJITSU_PERFSIM_REAL_BACKEND to libgpucsim_ffm_plugin.so";
  const std::string backend_path{backend_path_env};

  static_assert(FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION == 13);
  rocjitsu::test::ScopedTempFile direct_report{"rocjitsu-perfsim-direct-"};
  rocjitsu::test::ScopedTempFile bridge_report{"rocjitsu-perfsim-bridge-"};
  rocjitsu::test::ScopedEnvironmentVariable report_path{"GPUCSIM_REPORT_PATH",
                                                        direct_report.path()};
  rocjitsu::test::ScopedEnvironmentVariable target{"GPUCSIM_TARGET", "gfx1250"};
  rocjitsu::test::ScopedEnvironmentVariable wmma_only{"GPUCSIM_WMMA_ONLY", "0"};
  rocjitsu::test::ScopedEnvironmentVariable verbose{"GPUCSIM_VERBOSE", "0"};
  rocjitsu::test::ScopedEnvironmentVariable probes{"GPUCSIM_PROBE", "0"};
  rocjitsu::test::ScopedEnvironmentVariable smem_model{"GPUCSIM_SMEM_SQC_MODEL", "0"};
  rocjitsu::test::ScopedEnvironmentVariable wait_histogram{"GPUCSIM_WAIT_HISTOGRAM", "0"};
  rocjitsu::test::ScopedEnvironmentVariable ds_audit{"GPUCSIM_DS_AUDIT", ""};

  // The direct half retains its mapping for the process lifetime, matching the
  // adapter. Some Perfsim builds perform LLVM-global cleanup when finally unloaded.
  const util::LibraryHandle direct_backend = util::open_library(backend_path.c_str());
  ASSERT_NE(direct_backend, nullptr) << util::last_library_error();
  const auto get_api =
      util::lookup_symbol<FfmObserverPluginGetApiFn>(direct_backend, "ffm_observer_plugin_get_api");
  ASSERT_NE(get_api, nullptr) << util::last_library_error();
  constexpr uint32_t kOldestCompatibleApiVersion = 8;
  FfmObserverPluginApi *raw_api = nullptr;
  uint32_t requested_version = FFM_OBSERVER_PLUGIN_CURRENT_API_VERSION;
  for (;; --requested_version) {
    raw_api = invoke_foreign_abi(get_api, requested_version);
    if (raw_api || requested_version == kOldestCompatibleApiVersion)
      break;
  }
  ASSERT_NE(raw_api, nullptr);
  uint32_t negotiated_version = 0;
  std::memcpy(&negotiated_version, raw_api, sizeof(negotiated_version));
  ASSERT_GE(negotiated_version, kOldestCompatibleApiVersion);
  ASSERT_LE(negotiated_version, requested_version);
  FfmObserverPluginApiPrefix api{};
  std::memcpy(&api, raw_api, sizeof(api));
  ASSERT_NE(api.on_init, nullptr);
  ASSERT_NE(api.on_dispatch_begin, nullptr);
  ASSERT_NE(api.on_instruction, nullptr);
  ASSERT_NE(api.on_memory_access, nullptr);
  ASSERT_NE(api.on_tdm_memory_access, nullptr);
  ASSERT_NE(api.on_dispatch_end, nullptr);
  ASSERT_NE(api.on_shutdown, nullptr);

  constexpr uint32_t kDispatchId = 37;
  constexpr EntityId kClusterId = 0;
  constexpr uint64_t kNopPc = 0x8000;
  constexpr uint64_t kGlobalPc = 0x8004;
  constexpr uint64_t kTensorPc = 0x8010;
  constexpr uint64_t kWaitPc = 0x801C;
  constexpr uint64_t kEndPc = 0x8020;
  constexpr auto kNopEncoding = cdna5::build_sopp(cdna5::kSNopSopp);
  constexpr auto kGlobalEncoding = cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal);
  constexpr std::array<uint32_t, 3> kTensorEncoding{0xD0710001, 0x7C000000, 0x18140C00};
  constexpr auto kWaitEncoding = cdna5::build_sopp(cdna5::kSWaitLoadcntSopp);
  constexpr auto kEndEncoding = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  constexpr std::array<uint32_t, 4> kNopFetch{kNopEncoding[0], kGlobalEncoding[0],
                                              kGlobalEncoding[1], kGlobalEncoding[2]};
  constexpr std::array<uint32_t, 4> kGlobalFetch{kGlobalEncoding[0], kGlobalEncoding[1],
                                                 kGlobalEncoding[2], kTensorEncoding[0]};
  constexpr std::array<uint32_t, 4> kTensorFetch{kTensorEncoding[0], kTensorEncoding[1],
                                                 kTensorEncoding[2], kWaitEncoding[0]};
  constexpr std::array<uint32_t, 4> kWaitFetch{kWaitEncoding[0], kEndEncoding[0], 0, 0};
  constexpr std::array<uint32_t, 4> kEndFetch{kEndEncoding[0], 0, 0, 0};
  constexpr uint64_t kGlobalAddress = 0x10'0000;
  constexpr std::array<uint64_t, 2> kTensorAddresses{0x20'0000, 0x20'0004};

  const KernelDispatchInfo info = dispatch_info(kDispatchId);
  FfmDispatchMetadata metadata{};
  metadata.dispatch_info.dispatch_id = info.dispatch_id;
  metadata.vgpr_count = info.vgprs_per_wf;
  metadata.sgpr_count = 128;
  metadata.lds_size_bytes = info.lds_size_bytes;
  metadata.wave_size = info.wave_size;
  metadata.num_waves_per_wg = info.wfs_per_workgroup;
  metadata.grid_size[0] = info.grid_size_x;
  metadata.grid_size[1] = info.grid_size_y;
  metadata.grid_size[2] = info.grid_size_z;
  metadata.workgroup_size[0] = info.workgroup_size_x;
  metadata.workgroup_size[1] = info.workgroup_size_y;
  metadata.workgroup_size[2] = info.workgroup_size_z;
  metadata.dispatch_name = info.kernel_name.c_str();

  const auto direct_log = negotiated_version >= 12 ? +[](FfmLogLevel, const char *) {} : nullptr;
  const FfmHostApi direct_host{negotiated_version, direct_log};
  invoke_foreign_abi(api.on_init, &direct_host);
  invoke_foreign_abi(api.on_dispatch_begin, &metadata);
  const FfmWaveInfo direct_wave = make_wave_info(kDispatchId, kClusterId, 0, 0, 0);
  FfmInstructionCounters salu_counter{};
  salu_counter.salu_count = 1;
  FfmInstructionCounters global_counter{};
  global_counter.tex_count = 1;
  global_counter.global_scratch_load = 1;
  FfmInstructionCounters tensor_counter{};
  tensor_counter.tex_count = 1;
  FfmInstructionInfo direct_nop = make_instruction_info(kDispatchId, kClusterId, 0, 0, 0, 0, kNopPc,
                                                        kNopFetch.data(), salu_counter);
  FfmInstructionInfo direct_global = make_instruction_info(
      kDispatchId, kClusterId, 0, 0, 0, 1, kGlobalPc, kGlobalFetch.data(), global_counter);
  FfmInstructionInfo direct_tensor = make_instruction_info(
      kDispatchId, kClusterId, 0, 0, 0, 2, kTensorPc, kTensorFetch.data(), tensor_counter);
  FfmInstructionInfo direct_wait =
      make_instruction_info(kDispatchId, kClusterId, 0, 0, 0, 3, kWaitPc, kWaitFetch.data(),
                            salu_counter, {FFM_WAIT_TYPE_LOAD, FFM_WAIT_NAME_S_WAIT_LOADCNT});
  FfmInstructionInfo direct_end = make_instruction_info(kDispatchId, kClusterId, 0, 0, 0, 4, kEndPc,
                                                        kEndFetch.data(), salu_counter);
  invoke_foreign_abi(api.on_instruction, &direct_nop);
  invoke_foreign_abi(api.on_instruction, &direct_global);
  std::array<uint64_t, 32> global_addresses{};
  global_addresses[0] = kGlobalAddress;
  FfmMemoryAccess direct_memory = make_memory_access(1, direct_wave, 1, 32, global_addresses.data(),
                                                     4, FFM_RESOURCE_GLOBAL, false, true, false);
  invoke_foreign_abi(api.on_memory_access, &direct_memory);
  invoke_foreign_abi(api.on_instruction, &direct_tensor);
  FfmTdmMemoryAccess direct_tdm{};
  direct_tdm.instruction_id = 2;
  direct_tdm.wave_info = direct_wave;
  direct_tdm.num_addresses = static_cast<uint32_t>(kTensorAddresses.size());
  direct_tdm.addresses = kTensorAddresses.data();
  direct_tdm.data_size_bytes = 4;
  direct_tdm.flags = encode_tdm_flags(true, false);
  direct_tdm.tile_dim0 = 2;
  direct_tdm.tile_dim1 = 1;
  direct_tdm.data_size = 2;
  direct_tdm.tensor_dim0_stride = 8;
  direct_tdm.tensor_dim1_stride = 16;
  invoke_foreign_abi(api.on_tdm_memory_access, &direct_tdm);
  invoke_foreign_abi(api.on_instruction, &direct_wait);
  invoke_foreign_abi(api.on_instruction, &direct_end);
  invoke_foreign_abi(api.on_dispatch_end, &metadata);

  constexpr std::array<uint32_t, 2> kAdditionalDispatchIds{38, 39};
  for (uint32_t dispatch_id : kAdditionalDispatchIds) {
    const KernelDispatchInfo additional_info = dispatch_info(dispatch_id);
    FfmDispatchMetadata additional_metadata = metadata;
    additional_metadata.dispatch_info.dispatch_id = dispatch_id;
    additional_metadata.dispatch_name = additional_info.kernel_name.c_str();
    invoke_foreign_abi(api.on_dispatch_begin, &additional_metadata);
    FfmInstructionInfo additional_nop = make_instruction_info(
        dispatch_id, kClusterId, 0, 0, 0, 0, kNopPc, kNopFetch.data(), salu_counter);
    FfmInstructionInfo additional_end = make_instruction_info(
        dispatch_id, kClusterId, 0, 0, 0, 1, kEndPc, kEndFetch.data(), salu_counter);
    invoke_foreign_abi(api.on_instruction, &additional_nop);
    invoke_foreign_abi(api.on_instruction, &additional_end);
    invoke_foreign_abi(api.on_dispatch_end, &additional_metadata);
  }
  invoke_foreign_abi(api.on_shutdown);

  const std::string direct_json = read_file(direct_report.path());
  ASSERT_FALSE(direct_json.empty());
  for (uint32_t dispatch_id : {kDispatchId, kAdditionalDispatchIds[0], kAdditionalDispatchIds[1]})
    ASSERT_NE(direct_json.find("\"dispatch_id\": " + std::to_string(dispatch_id)),
              std::string::npos);
  ASSERT_EQ(setenv("GPUCSIM_REPORT_PATH", bridge_report.path().c_str(), 1), 0);

  WaveFixture fixture(3);
  {
    const std::string config = plugin_config(backend_path);
    PerfsimPlugin plugin(config.c_str());
    plugin.onInit();
    plugin.onAmdgpuDispatchPacketProcessed(info);
    plugin.onAmdgpuDispatchExecutionBegin(info.dispatch_id);
    Wavefront &wave = fixture.wave(info.dispatch_id, 0, {0, 0, 0}, 0);
    plugin.onAmdgpuWavefrontDispatched(wave);

    SyntheticInstruction nop("s_nop", kNopEncoding);
    plugin.onAmdgpuBeforeExecuteInstruction(kNopPc, nop, wave, kNopFetch);

    SyntheticInstruction global("global_load_b32", kGlobalEncoding, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(kGlobalPc, global, wave, kGlobalFetch);
    MemoryAccessObservation memory_access;
    memory_access.mnemonic = "global_load_b32";
    memory_access.pc = kGlobalPc;
    memory_access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    memory_access.dispatch_id = info.dispatch_id;
    memory_access.queue_id = wave.queue_id();
    memory_access.workgroup_id = wave.wg_id();
    memory_access.wavefront_id = wave.wf_id();
    memory_access.process_id = wave.process_id();
    memory_access.route = MemoryRoute::GLOBAL;
    memory_access.decoded_space = DecodedMemorySpace::GLOBAL;
    memory_access.is_load = true;
    memory_access.wavefront_size = 32;
    memory_access.element_size_bytes = 4;
    memory_access.elements_per_lane = 1;
    memory_access.active_lane_mask = 1;
    memory_access.architectural_exec_lane_mask = 1;
    memory_access.valid_lane_mask = 1;
    memory_access.request_lane_mask = 1;
    memory_access.addresses = global_addresses;
    plugin.onAmdgpuMemoryAccessRouted(memory_access);

    SyntheticInstruction tensor("tensor_load_to_lds", kTensorEncoding, MEMORY_OP);
    plugin.onAmdgpuBeforeExecuteInstruction(kTensorPc, tensor, wave, kTensorFetch);
    TensorDmaMemoryAccessObservation tensor_access;
    tensor_access.mnemonic = "tensor_load_to_lds";
    tensor_access.pc = kTensorPc;
    tensor_access.compute_unit_id = static_cast<uint32_t>(wave.cu().id());
    tensor_access.dispatch_id = info.dispatch_id;
    tensor_access.queue_id = wave.queue_id();
    tensor_access.workgroup_id = wave.wg_id();
    tensor_access.wavefront_id = wave.wf_id();
    tensor_access.process_id = wave.process_id();
    tensor_access.element_size_bytes = 4;
    tensor_access.tile_dim0 = 2;
    tensor_access.tile_dim1 = 1;
    tensor_access.data_size = 2;
    tensor_access.tensor_dim0_stride = 8;
    tensor_access.tensor_dim1_stride = 16;
    tensor_access.is_load = true;
    tensor_access.addresses = std::span<const uint64_t>(kTensorAddresses);
    plugin.onAmdgpuTensorDmaMemoryAccess(tensor_access);

    SyntheticInstruction wait("s_wait_loadcnt", kWaitEncoding);
    plugin.onAmdgpuBeforeExecuteInstruction(kWaitPc, wait, wave, kWaitFetch);
    SyntheticInstruction end("s_endpgm", kEndEncoding, PROGRAM_TERMINATOR);
    plugin.onAmdgpuBeforeExecuteInstruction(kEndPc, end, wave, kEndFetch);

    plugin.onAmdgpuWavefrontHalted(wave);
    plugin.onAmdgpuDispatchExecutionEnd(info.dispatch_id);

    for (uint32_t dispatch_id : kAdditionalDispatchIds) {
      const KernelDispatchInfo additional_info = dispatch_info(dispatch_id);
      plugin.onAmdgpuDispatchPacketProcessed(additional_info);
      plugin.onAmdgpuDispatchExecutionBegin(dispatch_id);
      Wavefront &additional_wave = fixture.wave(dispatch_id, 0, {0, 0, 0}, 0);
      SyntheticInstruction additional_nop("s_nop", kNopEncoding);
      plugin.onAmdgpuWavefrontDispatched(additional_wave);
      plugin.onAmdgpuBeforeExecuteInstruction(kNopPc, additional_nop, additional_wave, kNopFetch);
      SyntheticInstruction additional_end("s_endpgm", kEndEncoding, PROGRAM_TERMINATOR);
      plugin.onAmdgpuBeforeExecuteInstruction(kEndPc, additional_end, additional_wave, kEndFetch);
      plugin.onAmdgpuWavefrontHalted(additional_wave);
      plugin.onAmdgpuDispatchExecutionEnd(dispatch_id);
    }
    plugin.onShutdown();
  }

  const std::string bridge_json = read_file(bridge_report.path());
  ASSERT_FALSE(bridge_json.empty());
  for (uint32_t dispatch_id : {kDispatchId, kAdditionalDispatchIds[0], kAdditionalDispatchIds[1]})
    ASSERT_NE(bridge_json.find("\"dispatch_id\": " + std::to_string(dispatch_id)),
              std::string::npos);
  EXPECT_EQ(bridge_json, direct_json);
}

} // namespace
