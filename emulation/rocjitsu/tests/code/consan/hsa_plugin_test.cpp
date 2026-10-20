// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace.h"
#include "rocjitsu/code/consan/capture.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/hooks/consan/api.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>

namespace rocjitsu::consan {
namespace {

class ConSanHsaPlugin : public ::testing::Test {
protected:
  void SetUp() override {
    current_ = this;
    std::ifstream input(CONSAN_ORIGINAL_FIXTURE, std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
    std::ifstream facts_file(std::filesystem::path(CONSAN_ARTIFACT_FIXTURE) / "facts.bin",
                             std::ios::binary);
    const std::vector<uint8_t> encoded_facts{std::istreambuf_iterator<char>(facts_file),
                                             std::istreambuf_iterator<char>()};
    const auto facts = decode_manifest(encoded_facts);
    ASSERT_TRUE(facts.succeeded());
    facts_ = facts.value();
    const char *root = std::getenv("CONSAN_TEST_ARTIFACT_ROOT");
    const std::filesystem::path parent = root ? root : std::filesystem::temp_directory_path();
    std::filesystem::create_directories(parent);
    std::string pattern =
        (parent / (std::string("fake-hsa-") +
                   ::testing::UnitTest::GetInstance()->current_test_info()->name() + "-XXXXXX"))
            .string();
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back(0);
    char *created = ::mkdtemp(temporary.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
    preserve_ = root != nullptr;
    for (const char *name : {"CONSAN_ARTIFACT_DIR", "CONSAN_CAPTURE_PATH"}) {
      const char *value = std::getenv(name);
      environment_[name] = value ? std::optional<std::string>(value) : std::nullopt;
    }
    ASSERT_EQ(::setenv("CONSAN_ARTIFACT_DIR", CONSAN_ARTIFACT_FIXTURE, 1), 0);
    capture_ = directory_ / "capture.bin";
    ASSERT_EQ(::setenv("CONSAN_CAPTURE_PATH", capture_.c_str(), 1), 0);
    library_ = ::dlopen(CONSAN_PLUGIN_FIXTURE, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library_, nullptr) << ::dlerror();
    load_ = reinterpret_cast<bool (*)(HsaApiTable *, uint64_t, uint64_t, const char *const *)>(
        ::dlsym(library_, "OnLoad"));
    unload_ = reinterpret_cast<void (*)()>(::dlsym(library_, "OnUnload"));
    finish_ = reinterpret_cast<rj_consan_finish_fn>(::dlsym(library_, "rj_consan_finish"));
    drain_ = reinterpret_cast<rj_consan_drain_fn>(::dlsym(library_, "rj_consan_drain"));
    statistics_ =
        reinterpret_cast<rj_consan_statistics_fn>(::dlsym(library_, "rj_consan_get_statistics"));
    ASSERT_TRUE(load_ && unload_ && finish_ && drain_ && statistics_);
    initialize_api();
    ASSERT_TRUE(load_(&table_, 0, 0, nullptr));
    installed_ = true;
    const uint32_t input_value = 7;
    std::memcpy(arguments_.data(), &input_value, 4);
  }

  void TearDown() override {
    // These are fake signals, not a GPU. Release an interrupted unit test's
    // worker before destroying its fake API table; do not mask test assertions.
    {
      std::lock_guard lock(fake_mutex_);
      for (auto &[handle, value] : signal_values_) {
        (void)handle;
        value = 0;
      }
      fake_condition_.notify_all();
    }
    if (installed_) {
      (void)finish_();
      unload_();
      EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
    }
    if (library_)
      ::dlclose(library_);
    for (const auto &[name, value] : environment_) {
      if (value)
        (void)::setenv(name.c_str(), value->c_str(), 1);
      else
        (void)::unsetenv(name.c_str());
    }
    if (!preserve_ && !directory_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(directory_, ignored);
    }
    for (const auto &[memory, bytes] : allocations_) {
      (void)bytes;
      std::free(memory);
    }
    current_ = nullptr;
  }

  void initialize_api() {
    core_.version.minor_id = sizeof(core_);
    amd_.version.minor_id = sizeof(amd_);
    table_.core_ = &core_;
    table_.amd_ext_ = &amd_;
    core_.hsa_shut_down_fn = [] { return HSA_STATUS_SUCCESS; };
    core_.hsa_code_object_reader_create_from_memory_fn = create_reader;
    core_.hsa_code_object_reader_destroy_fn = destroy_reader;
    core_.hsa_executable_load_agent_code_object_fn = load_code;
    core_.hsa_executable_freeze_fn = [](hsa_executable_t, const char *) {
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_executable_destroy_fn = [](hsa_executable_t) { return HSA_STATUS_SUCCESS; };
    core_.hsa_executable_get_symbol_by_name_fn =
        [](hsa_executable_t, const char *, const hsa_agent_t *, hsa_executable_symbol_t *symbol) {
          symbol->handle = 10;
          return HSA_STATUS_SUCCESS;
        };
    core_.hsa_executable_symbol_get_info_fn = symbol_info;
    core_.hsa_queue_destroy_fn = [](hsa_queue_t *) { return HSA_STATUS_SUCCESS; };
    core_.hsa_signal_create_fn = create_signal;
    core_.hsa_signal_destroy_fn = destroy_signal;
    core_.hsa_signal_load_scacquire_fn = load_signal;
    core_.hsa_signal_store_screlease_fn = store_signal;
    core_.hsa_signal_subtract_screlease_fn = subtract_signal;
    core_.hsa_signal_wait_scacquire_fn = wait_signal;
    core_.hsa_iterate_agents_fn = [](hsa_status_t (*callback)(hsa_agent_t, void *), void *data) {
      return callback(hsa_agent_t{2}, data);
    };
    core_.hsa_agent_get_info_fn = [](hsa_agent_t agent, hsa_agent_info_t attribute, void *value) {
      if (attribute == HSA_AGENT_INFO_DEVICE)
        *static_cast<hsa_device_type_t *>(value) =
            agent.handle == 1 ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU;
      else if (attribute == HSA_AGENT_INFO_ISA)
        *static_cast<hsa_isa_t *>(value) = {3};
      else
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_isa_get_info_alt_fn = [](hsa_isa_t, hsa_isa_info_t attribute, void *value) {
      constexpr char name[] = "amdgcn-amd-amdhsa--gfx1100";
      if (attribute == HSA_ISA_INFO_NAME_LENGTH)
        *static_cast<uint32_t *>(value) = sizeof(name);
      else if (attribute == HSA_ISA_INFO_NAME)
        std::memcpy(value, name, sizeof(name));
      else
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_agent_iterate_memory_pools_fn =
        [](hsa_agent_t, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
          return callback(hsa_amd_memory_pool_t{4}, data);
        };
    amd_.hsa_amd_memory_pool_get_info_fn = [](hsa_amd_memory_pool_t,
                                              hsa_amd_memory_pool_info_t attribute, void *value) {
      if (attribute == HSA_AMD_MEMORY_POOL_INFO_SEGMENT)
        *static_cast<uint32_t *>(value) = 0;
      else if (attribute == HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS)
        *static_cast<uint32_t *>(value) = 3;
      else if (attribute == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED)
        *static_cast<bool *>(value) = true;
      else
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_memory_pool_allocate_fn = allocate;
    amd_.hsa_amd_memory_pool_free_fn = free_memory;
    amd_.hsa_amd_agents_allow_access_fn = [](uint32_t, const hsa_agent_t *, const uint32_t *,
                                             const void *) { return HSA_STATUS_SUCCESS; };
    amd_.hsa_amd_pointer_info_fn = pointer_info;
    amd_.hsa_amd_queue_intercept_create_fn = [](hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                                void (*)(hsa_status_t, hsa_queue_t *, void *),
                                                void *, uint32_t, uint32_t, hsa_queue_t **queue) {
      *queue = &current_->queue_;
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_queue_intercept_register_fn =
        [](hsa_queue_t *, hsa_amd_queue_intercept_handler handler, void *context) {
          current_->handler_ = handler;
          current_->handler_context_ = context;
          return HSA_STATUS_SUCCESS;
        };
    amd_.hsa_amd_profiling_set_profiler_enabled_fn = [](hsa_queue_t *, int enabled) {
      return enabled == 0 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    };
  }

  bool bind() {
    return core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                              &reader_) == HSA_STATUS_SUCCESS &&
           core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr) ==
               HSA_STATUS_SUCCESS &&
           core_.hsa_executable_freeze_fn({7}, nullptr) == HSA_STATUS_SUCCESS;
  }

  bool create_queue_and_signal() {
    hsa_queue_t *queue = nullptr;
    if (core_.hsa_queue_create_fn({1}, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                  &queue) != HSA_STATUS_SUCCESS ||
        queue != &queue_)
      return false;
    return core_.hsa_signal_create_fn(1, 0, nullptr, &signal_) == HSA_STATUS_SUCCESS;
  }

  void submit(uint32_t grid = 64, const void *arguments = nullptr, uint16_t workgroup = 64) {
    hsa_kernel_dispatch_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH | (1u << HSA_PACKET_HEADER_BARRIER);
    packet.setup = 1;
    packet.grid_size_x = grid;
    packet.grid_size_y = packet.grid_size_z = 1;
    packet.workgroup_size_x = workgroup;
    packet.workgroup_size_y = packet.workgroup_size_z = 1;
    packet.kernel_object = 0x123400;
    packet.group_segment_size = 8;
    packet.kernarg_address = const_cast<void *>(arguments ? arguments : arguments_.data());
    packet.completion_signal = signal_;
    ASSERT_NE(handler_, nullptr);
    handler_(&packet, 1, 0, handler_context_, packet_writer);
  }

  void next_signal() {
    ASSERT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_SUCCESS);
    signal_ = {};
    ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &signal_), HSA_STATUS_SUCCESS);
  }

  size_t report_capacity() {
    std::lock_guard lock(fake_mutex_);
    return allocations_.at(report_);
  }

  void expect_report_extent(const CapturedExecution &execution, uint32_t threads) {
    const LaunchDimensions dimensions{{threads, 1, 1}, {threads, 1, 1}};
    EXPECT_EQ(execution.dimensions.grid, dimensions.grid);
    EXPECT_EQ(execution.dimensions.workgroup, dimensions.workgroup);
    const auto layout =
        plan_launch(facts_, dimensions, execution.group_bytes, execution.private_bytes);
    ASSERT_TRUE(layout.succeeded());
    EXPECT_EQ(execution.report.size(), layout.value().report_bytes);
    EXPECT_EQ(wire::read_u64(execution.context, wire::kContextReportBytes),
              layout.value().report_bytes);
    EXPECT_EQ(wire::read_u32(execution.report, wire::kReportRows), layout.value().wave_count);
    EXPECT_EQ(wire::read_u32(execution.report, wire::kReportRowStride), layout.value().row_stride);
    for (size_t dimension = 0; dimension < 3; ++dimension) {
      EXPECT_EQ(wire::read_u32(execution.context, wire::kContextGrid + dimension * 4),
                dimensions.grid[dimension]);
      EXPECT_EQ(wire::read_u32(execution.context, wire::kContextWorkgroup + dimension * 4),
                dimensions.workgroup[dimension]);
    }
    for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave) {
      const size_t row = kReportHeaderBytes + size_t{wave} * layout.value().row_stride;
      EXPECT_EQ(wire::read_u64(execution.report, row + wire::kRowSerial), execution.serial);
      EXPECT_EQ(wire::read_u32(execution.report, row + wire::kRowWave), wave);
    }
  }

  void change_geometry(uint16_t first_threads, uint16_t second_threads) {
    ASSERT_TRUE(bind());
    ASSERT_TRUE(create_queue_and_signal());
    auto_complete_ = false;
    submit(first_threads, nullptr, first_threads);
    ASSERT_EQ(published_packets_, 1u);
    ASSERT_GE(report_capacity(), report_bytes_);
    EXPECT_EQ(statistics_().copied, 0u);
    const uintptr_t first_address = reinterpret_cast<uintptr_t>(report_);
    const size_t first_capacity = report_capacity();
    const std::vector<uint8_t> first_report(report_, report_ + report_bytes_);
    store_signal(published_.completion_signal, 0);
    ASSERT_EQ(drain_(), 0);
    EXPECT_EQ(statistics_().copied, 1u);
    std::memset(report_, 0xcd, report_bytes_);
    next_signal();
    const uint32_t second_argument = 23;
    std::memcpy(arguments_.data(), &second_argument, sizeof(second_argument));
    row_status_ = 0x12345678;
    auto_complete_ = true;
    submit(second_threads, nullptr, second_threads);
    ASSERT_EQ(published_packets_, 2u);
    ASSERT_EQ(drain_(), 0);
    ASSERT_GE(report_capacity(), report_bytes_);
    if (second_threads < first_threads) {
      EXPECT_EQ(reinterpret_cast<uintptr_t>(report_), first_address);
      EXPECT_EQ(report_capacity(), first_capacity);
      EXPECT_GT(report_capacity(), report_bytes_);
      EXPECT_EQ(statistics_().reused, 1u);
    } else {
      EXPECT_NE(reinterpret_cast<uintptr_t>(report_), first_address);
      EXPECT_GT(report_capacity(), first_capacity);
      EXPECT_EQ(statistics_().reused, 0u);
    }
    const std::vector<uint8_t> second_report(report_, report_ + report_bytes_);
    std::memset(report_, 0xef, report_bytes_);
    ASSERT_EQ(finish_(), 0);
    const auto captured = decode_capture(read_capture());
    ASSERT_TRUE(captured.succeeded());
    ASSERT_EQ(captured.value().executions.size(), 2u);
    const auto &first = captured.value().executions[0];
    const auto &second = captured.value().executions[1];
    expect_report_extent(first, first_threads);
    expect_report_extent(second, second_threads);
    EXPECT_EQ(first.report, first_report);
    EXPECT_EQ(second.report, second_report);
    EXPECT_EQ(wire::read_u32(first.arguments, 0), 7u);
    EXPECT_EQ(wire::read_u32(second.arguments, 0), second_argument);
    EXPECT_EQ(first.serial, 1u);
    EXPECT_EQ(second.serial, 2u);
    EXPECT_EQ(first.storage_slot, second.storage_slot);
    EXPECT_LT(first.storage_generation, second.storage_generation);
    EXPECT_EQ(statistics_().submitted, 2u);
    EXPECT_EQ(statistics_().forwarded, 2u);
    EXPECT_EQ(statistics_().copied, 2u);
    EXPECT_EQ(statistics_().written, 2u);
    EXPECT_EQ(forwarding_calls_, 2u);
    EXPECT_TRUE(allocations_.empty());
  }

  static hsa_status_t create_reader(const void *bytes, size_t size,
                                    hsa_code_object_reader_t *reader) {
    if (current_->throw_reader_) {
      mark_exception();
      throw std::bad_alloc();
    }
    reader->handle = ++current_->next_reader_;
    current_->readers_[reader->handle] = {static_cast<const uint8_t *>(bytes), size};
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t destroy_reader(hsa_code_object_reader_t reader) {
    current_->readers_.erase(reader.handle);
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t load_code(hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t reader,
                                const char *, hsa_loaded_code_object_t *) {
    const auto bytes = current_->readers_.at(reader.handle);
    current_->loaded_image_.assign(bytes.begin(), bytes.end());
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t symbol_info(hsa_executable_symbol_t, hsa_executable_symbol_info_t attribute,
                                  void *value) {
    if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT)
      *static_cast<uint64_t *>(value) = 0x123400;
    else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE)
      *static_cast<uint32_t *>(value) = 32;
    else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE)
      *static_cast<uint32_t *>(value) = 8;
    else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE)
      *static_cast<uint32_t *>(value) = 0;
    else
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t allocate(hsa_amd_memory_pool_t, size_t bytes, uint32_t, void **memory) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->last_allocation_bytes_ = bytes;
    if (++current_->allocation_calls_ == current_->fail_allocation_)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    *memory = std::aligned_alloc(64, (bytes + 63) & ~size_t{63});
    if (!*memory)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    current_->allocations_[*memory] = bytes;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t free_memory(void *memory) {
    std::lock_guard lock(current_->fake_mutex_);
    // Normal launches may replace report_; only the armed, single-launch
    // finish test needs to inspect it here.
    if (current_->throw_report_free_ && memory == current_->report_) {
      mark_exception();
      throw std::bad_alloc();
    }
    if (current_->allocations_.erase(memory) != 1)
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    std::free(memory);
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t pointer_info(const void *pointer, hsa_amd_pointer_info_t *info,
                                   void *(*)(size_t), uint32_t *, hsa_agent_t **) {
    if (current_->throw_pointer_info_) {
      mark_exception();
      throw 42;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(current_->arguments_.data());
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address < base || address >= base + current_->arguments_.size())
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    info->size = sizeof(*info);
    info->type = HSA_EXT_POINTER_TYPE_HSA;
    info->agentBaseAddress = info->hostBaseAddress = current_->arguments_.data();
    info->sizeInBytes = current_->arguments_.size();
    info->global_flags = 3;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t create_signal(hsa_signal_value_t value, uint32_t, const hsa_agent_t *,
                                    hsa_signal_t *signal) {
    std::lock_guard lock(current_->fake_mutex_);
    if (current_->fail_proxy_ && current_->signal_.handle != 0)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    signal->handle = ++current_->next_signal_;
    current_->signal_values_[signal->handle] = value;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t destroy_signal(hsa_signal_t signal) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->signal_values_.erase(signal.handle);
    return HSA_STATUS_SUCCESS;
  }
  static hsa_signal_value_t load_signal(hsa_signal_t signal) {
    std::lock_guard lock(current_->fake_mutex_);
    return current_->signal_values_.at(signal.handle);
  }
  static void store_signal(hsa_signal_t signal, hsa_signal_value_t value) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->signal_values_.at(signal.handle) = value;
    current_->fake_condition_.notify_all();
  }
  static void subtract_signal(hsa_signal_t signal, hsa_signal_value_t value) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->signal_values_.at(signal.handle) -= value;
    ++current_->forwarding_calls_;
    current_->fake_condition_.notify_all();
  }
  static hsa_signal_value_t wait_signal(hsa_signal_t signal, hsa_signal_condition_t,
                                        hsa_signal_value_t, uint64_t, hsa_wait_state_t) {
    if (current_->throw_wait_) {
      mark_exception();
      throw std::bad_alloc();
    }
    std::unique_lock lock(current_->fake_mutex_);
    current_->fake_condition_.wait_for(lock, std::chrono::milliseconds(1), [&] {
      return current_->signal_values_.at(signal.handle) < 1;
    });
    return current_->signal_values_.at(signal.handle);
  }
  static void packet_writer(const void *packets, uint64_t count) {
    ASSERT_EQ(count, 1u);
    auto &state = *current_;
    std::memcpy(&state.published_, packets, sizeof(state.published_));
    ++state.published_packets_;
    const auto wrapper =
        std::span(static_cast<const uint8_t *>(state.published_.kernarg_address), 32);
    EXPECT_EQ(*wire::read_u64(wrapper, 16), reinterpret_cast<uint64_t>(state.arguments_.data()));
    EXPECT_EQ(*wire::read_u32(wrapper, 0), *wire::read_u32(state.arguments_, 0));
    const auto context =
        std::span(reinterpret_cast<const uint8_t *>(*wire::read_u64(wrapper, 24)), kContextBytes);
    state.report_ =
        reinterpret_cast<uint8_t *>(*wire::read_u64(context, wire::kContextReportPointer));
    state.report_bytes_ = *wire::read_u64(context, wire::kContextReportBytes);
    // This deliberately malformed row proves that the real runtime saves raw
    // bytes without diagnosing them. No ISA executes in this fake-API unit test.
    wire::write_u32(std::span(state.report_, state.report_bytes_),
                    kReportHeaderBytes + wire::kRowStatus, state.row_status_);
    if (state.auto_complete_)
      store_signal(state.published_.completion_signal, 0);
  }

  std::vector<uint8_t> read_capture() {
    std::ifstream file(capture_, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  }

  static void mark_exception() {
    constexpr char message[] = "consan injected exception\n";
    if (::write(STDERR_FILENO, message, sizeof(message) - 1) !=
        static_cast<ssize_t>(sizeof(message) - 1))
      std::_Exit(90);
  }

  inline static ConSanHsaPlugin *current_ = nullptr;
  CoreApiTable core_{};
  AmdExtTable amd_{};
  HsaApiTable table_{};
  void *library_ = nullptr;
  bool (*load_)(HsaApiTable *, uint64_t, uint64_t, const char *const *) = nullptr;
  void (*unload_)() = nullptr;
  rj_consan_finish_fn finish_ = nullptr;
  rj_consan_drain_fn drain_ = nullptr;
  rj_consan_statistics_fn statistics_ = nullptr;
  hsa_amd_queue_intercept_handler handler_ = nullptr;
  void *handler_context_ = nullptr;
  hsa_queue_t queue_{};
  hsa_code_object_reader_t reader_{};
  hsa_signal_t signal_{};
  hsa_kernel_dispatch_packet_t published_{};
  std::mutex fake_mutex_;
  std::condition_variable fake_condition_;
  std::map<uint64_t, hsa_signal_value_t> signal_values_;
  std::map<uint64_t, std::span<const uint8_t>> readers_;
  std::map<void *, size_t> allocations_;
  std::map<std::string, std::optional<std::string>> environment_;
  std::filesystem::path directory_, capture_;
  std::vector<uint8_t> original_, loaded_image_;
  Manifest facts_;
  alignas(64) std::array<uint8_t, 16> arguments_{};
  uint8_t *report_ = nullptr;
  size_t report_bytes_ = 0;
  size_t last_allocation_bytes_ = 0;
  uint64_t next_signal_ = 100, next_reader_ = 200;
  uint32_t allocation_calls_ = 0, fail_allocation_ = 0, published_packets_ = 0,
           forwarding_calls_ = 0;
  uint32_t row_status_ = UINT32_MAX;
  bool installed_ = false, preserve_ = false, fail_proxy_ = false, auto_complete_ = true;
  bool throw_reader_ = false, throw_report_free_ = false, throw_pointer_info_ = false,
       throw_wait_ = false;
};

class ConSanHsaPluginDeathTest : public ConSanHsaPlugin {
protected:
  void SetUp() override {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ConSanHsaPlugin::SetUp();
  }

  template <typename Action> void expect_termination(Action &&action) {
    // Distinguish the injected exception's terminate path from a returned
    // status or an exception escaping the loaded DSO into its caller.
    ASSERT_EXIT(
        {
          std::set_terminate([] { std::_Exit(86); });
          ::alarm(5);
          try {
            action();
          } catch (...) {
            std::_Exit(89);
          }
          std::_Exit(87);
        },
        ::testing::ExitedWithCode(86), "consan injected exception");
  }
};

TEST_F(ConSanHsaPluginDeathTest, HsaHookExceptionTerminatesInsteadOfReturningAStatus) {
  expect_termination([&] {
    throw_reader_ = true;
    hsa_code_object_reader_t reader{};
    (void)core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                             &reader);
  });
}

TEST_F(ConSanHsaPluginDeathTest, PacketCallbackExceptionTerminatesInsteadOfReturningToHsa) {
  expect_termination([&] {
    ASSERT_TRUE(bind());
    ASSERT_TRUE(create_queue_and_signal());
    throw_pointer_info_ = true;
    submit();
  });
}

TEST_F(ConSanHsaPluginDeathTest, ObserverExceptionTerminatesInsteadOfLeavingAStoppedWorker) {
  expect_termination([&] {
    throw_wait_ = true;
    ASSERT_TRUE(bind());
    ASSERT_TRUE(create_queue_and_signal());
    submit();
    for (;;)
      ::pause();
  });
}

TEST_F(ConSanHsaPluginDeathTest, FinishExceptionTerminatesInsteadOfReturningInvalidFacts) {
  expect_termination([&] {
    ASSERT_TRUE(bind());
    ASSERT_TRUE(create_queue_and_signal());
    submit();
    ASSERT_EQ(drain_(), 0);
    // Only the report is freed by finish after joining both workers. Observer
    // wrapper/context cleanup must not trigger this exception on another path.
    {
      std::lock_guard lock(fake_mutex_);
      throw_report_free_ = true;
    }
    (void)finish_();
  });
}

TEST_F(ConSanHsaPlugin, NormalReaderLoadUsesPreparedBytesAndProjectsTheOriginalArgumentAbi) {
  ASSERT_TRUE(bind());
  EXPECT_NE(loaded_image_, original_);
  std::ifstream prepared(std::filesystem::path(CONSAN_ARTIFACT_FIXTURE) / "prepared.hsaco",
                         std::ios::binary);
  const std::vector<uint8_t> expected{std::istreambuf_iterator<char>(prepared),
                                      std::istreambuf_iterator<char>()};
  EXPECT_EQ(loaded_image_, expected);
  uint32_t kernarg = 0;
  EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernarg, 16u);
  EXPECT_EQ(finish_(), 0);
  EXPECT_TRUE(decode_capture(read_capture()).succeeded());
}

TEST_F(ConSanHsaPlugin, CompletionGatesTheRealCopyPathAndRepeatedLaunchActuallyReusesItsSlot) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit();
  ASSERT_EQ(published_packets_, 1u);
  EXPECT_NE(published_.completion_signal.handle, signal_.handle);
  EXPECT_EQ((published_.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 3,
            HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ((published_.header >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) & 3,
            HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_NE(published_.header & (1u << HSA_PACKET_HEADER_BARRIER), 0u);
  EXPECT_EQ(statistics_().forwarded, 0u);
  EXPECT_EQ(statistics_().copied, 0u);
  EXPECT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_ERROR_INVALID_SIGNAL);
  const auto first_report = report_;
  store_signal(published_.completion_signal, 0);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(statistics_().copied, 1u);
  std::memset(first_report, 0xcd, report_bytes_);
  ASSERT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_SUCCESS);
  signal_ = {};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &signal_), HSA_STATUS_SUCCESS);
  const uint32_t second = 23;
  std::memcpy(arguments_.data(), &second, 4);
  auto_complete_ = true;
  submit();
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(report_, first_report);
  EXPECT_EQ(statistics_().reused, 1u);
  ASSERT_EQ(finish_(), 0);
  const auto captured = decode_capture(read_capture());
  ASSERT_TRUE(captured.succeeded());
  ASSERT_EQ(captured.value().executions.size(), 2u);
  EXPECT_EQ(wire::read_u32(captured.value().executions[0].arguments, 0), 7u);
  EXPECT_EQ(wire::read_u32(captured.value().executions[1].arguments, 0), 23u);
  EXPECT_EQ(captured.value().executions[0].storage_slot,
            captured.value().executions[1].storage_slot);
  EXPECT_EQ(captured.value().executions[1].storage_generation, 2u);
  expect_report_extent(captured.value().executions[0], 64);
  expect_report_extent(captured.value().executions[1], 64);
  EXPECT_EQ(
      wire::read_u32(captured.value().executions[0].report, kReportHeaderBytes + wire::kRowStatus),
      UINT32_MAX);
  EXPECT_EQ(forwarding_calls_, 2u);
}

TEST_F(ConSanHsaPlugin, GrowingGeometryHasEnoughCapacityBeforePublicationAndCopiesTheFullReport) {
  change_geometry(64, 128);
}

TEST_F(ConSanHsaPlugin, ShrinkingGeometryReusesCapacityButInitializesAndCapturesOnlyItsExtent) {
  change_geometry(128, 64);
}

TEST_F(ConSanHsaPlugin, FailedReportGrowthKeepsTheIdleAllocationAndPublishesNoPartialPacket) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit();
  ASSERT_EQ(drain_(), 0);
  const uintptr_t first_address = reinterpret_cast<uintptr_t>(report_);
  const size_t first_capacity = report_capacity();
  next_signal();
  fail_allocation_ = allocation_calls_ + 1;
  const auto layout = plan_launch(facts_, {{128, 1, 1}, {128, 1, 1}}, 8, 0);
  ASSERT_TRUE(layout.succeeded());
  submit(128, nullptr, 128);
  EXPECT_EQ(last_allocation_bytes_, layout.value().report_bytes);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(report_), first_address);
  EXPECT_EQ(report_capacity(), first_capacity);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(statistics_().submitted, 1u);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(statistics_().copied, 1u);
  EXPECT_EQ(statistics_().written, 1u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
  EXPECT_TRUE(allocations_.empty());
}

TEST_F(ConSanHsaPlugin, FailedReportAllocationPublishesNoPartialPacket) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_allocation_ = 1;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
  EXPECT_TRUE(allocations_.empty());
}

TEST_F(ConSanHsaPlugin, FailedWrapperAllocationReleasesUnsubmittedResources) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_allocation_ = 2;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_TRUE(allocations_.empty());
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, FailedContextAllocationReleasesTheWrapperAndReportLease) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_allocation_ = 3;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_TRUE(allocations_.empty());
}

TEST_F(ConSanHsaPlugin, FailedProxyAllocationNeverPublishesTheKernel) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_proxy_ = true;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_TRUE(allocations_.empty());
}

TEST_F(ConSanHsaPlugin, InvalidGeometryFailsBeforeAllocation) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit(128);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, ShortOriginalArgumentAllocationFailsBeforePublication) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit(64, arguments_.data() + 12);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, ConcurrentFinishClosesAndPublishesOneCompleteCapture) {
  ASSERT_TRUE(bind());
  std::promise<void> release;
  auto ready = release.get_future().share();
  auto first = std::async(std::launch::async, [&] {
    ready.wait();
    return finish_();
  });
  auto second = std::async(std::launch::async, [&] {
    ready.wait();
    return finish_();
  });
  release.set_value();
  EXPECT_EQ(first.get(), 0);
  EXPECT_EQ(second.get(), 0);
  EXPECT_TRUE(decode_capture(read_capture()).succeeded());
}

TEST_F(ConSanHsaPlugin, ShutdownRequiresExplicitFinishAndProfilingCannotBeEnabled) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  EXPECT_NE(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
  EXPECT_NE(amd_.hsa_amd_profiling_set_profiler_enabled_fn(&queue_, 1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(finish_(), 0);
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
}

TEST_F(ConSanHsaPlugin, ExistingAmdTablePrefixIsBoundedAndRestored) {
  (void)finish_();
  unload_();
  installed_ = false;
  initialize_api();

  // The previous tools ABI admitted this 688-byte prefix. Put its end at an
  // inaccessible page so copying the newer shared header's full table faults.
  constexpr size_t kRuntimeTableBytes = 688;
  static_assert(kRuntimeTableBytes < sizeof(AmdExtTable));
  const long page_size = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, static_cast<long>(sizeof(AmdExtTable)));
  const size_t mapping_bytes = 2 * static_cast<size_t>(page_size);
  void *mapping =
      ::mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  const auto unmap = [mapping_bytes](void *memory) { (void)::munmap(memory, mapping_bytes); };
  std::unique_ptr<void, decltype(unmap)> storage(mapping, unmap);
  auto *end = static_cast<uint8_t *>(mapping) + page_size;
  ASSERT_EQ(::mprotect(end, page_size, PROT_NONE), 0);
  auto *prefix = reinterpret_cast<AmdExtTable *>(end - kRuntimeTableBytes);
  std::memcpy(prefix, &amd_, kRuntimeTableBytes);
  table_.amd_ext_ = prefix;

  prefix->version.minor_id = kRuntimeTableBytes - 1;
  EXPECT_FALSE(load_(&table_, 0, 0, nullptr));
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
  prefix->version.minor_id = kRuntimeTableBytes;
  const bool accepted = load_(&table_, 0, 0, nullptr);
  EXPECT_TRUE(accepted);
  if (accepted) {
    installed_ = true;
    EXPECT_TRUE(bind());
    EXPECT_TRUE(create_queue_and_signal());
    EXPECT_EQ(finish_(), 0);
    unload_();
    installed_ = false;
    EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
    EXPECT_EQ(prefix->hsa_amd_profiling_set_profiler_enabled_fn,
              amd_.hsa_amd_profiling_set_profiler_enabled_fn);
    EXPECT_EQ(prefix->hsa_amd_queue_intercept_register_fn,
              amd_.hsa_amd_queue_intercept_register_fn);
  }
  table_.amd_ext_ = &amd_;
}

TEST_F(ConSanHsaPlugin, UnloadRestoresOurHooksWithoutOverwritingLaterCoreOrExtensionCallbacks) {
  ASSERT_TRUE(bind());
  ASSERT_EQ(finish_(), 0);
  const auto later_reader =
      +[](hsa_file_t, hsa_code_object_reader_t *) { return HSA_STATUS_ERROR; };
  const auto later_profiling = +[](hsa_queue_t *, int) { return HSA_STATUS_ERROR; };
  core_.hsa_code_object_reader_create_from_file_fn = later_reader;
  amd_.hsa_amd_profiling_set_profiler_enabled_fn = later_profiling;
  unload_();
  installed_ = false;
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_file_fn, later_reader);
  EXPECT_EQ(amd_.hsa_amd_profiling_set_profiler_enabled_fn, later_profiling);
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
  EXPECT_EQ(core_.hsa_signal_create_fn, create_signal);
  const auto handler =
      +[](const void *, uint64_t, uint64_t, void *, hsa_amd_queue_intercept_packet_writer) {};
  ASSERT_NE(amd_.hsa_amd_queue_intercept_register_fn, nullptr);
  EXPECT_EQ(amd_.hsa_amd_queue_intercept_register_fn(&queue_, handler, this), HSA_STATUS_SUCCESS);
  EXPECT_EQ(handler_, handler);
  EXPECT_EQ(handler_context_, this);
}

} // namespace
} // namespace rocjitsu::consan
