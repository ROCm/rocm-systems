// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace.h"
#include "rocjitsu/code/consan/capture.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/hooks/consan/api.h"
#include "rocjitsu/hooks/consan/testing.h"

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
#include <optional>
#include <sys/mman.h>
#include <unistd.h>

namespace rocjitsu::consan {
namespace {
class ConSanHsaPlugin : public ::testing::Test {
protected:
  virtual const char *original_fixture() const { return CONSAN_ORIGINAL_FIXTURE; }
  virtual const char *artifact_fixture() const { return CONSAN_ARTIFACT_FIXTURE; }
  virtual bool application_profile() const { return false; }

  void SetUp() override {
    current_ = this;
    std::ifstream input(original_fixture(), std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
    std::ifstream facts_file(std::filesystem::path(artifact_fixture()) / "facts.bin",
                             std::ios::binary);
    const std::vector<uint8_t> encoded_facts{std::istreambuf_iterator<char>(facts_file),
                                             std::istreambuf_iterator<char>()};
    const FailureOr<ImageFacts> facts = decode_image_facts(encoded_facts);
    ASSERT_TRUE(facts.succeeded());
    image_facts_ = facts.value();
    ASSERT_FALSE(image_facts_.entries.empty());
    const ImageEntryFacts *first_selected = nullptr;
    for (const ImageEntryFacts &entry : image_facts_.entries) {
      argument_extent_ = std::max<size_t>(argument_extent_, entry.binding.original_kernarg_bytes);
      if (entry.selected && first_selected == nullptr) {
        first_selected = &entry;
      }
    }
    ASSERT_NE(first_selected, nullptr);
    ASSERT_LE(argument_extent_, arguments_.size());
    facts_ = *first_selected->selected;
    const char *root = std::getenv("CONSAN_TEST_ARTIFACT_ROOT");
    const std::filesystem::path parent = root ? root : std::filesystem::temp_directory_path();
    std::filesystem::create_directories(parent);
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // Parameterized labels must not create subdirectories.
    std::ranges::replace(test_name, '/', '-');
    std::string pattern = (parent / ("fake-hsa-" + test_name + "-XXXXXX")).string();
    std::vector<char> temporary(pattern.begin(), pattern.end());
    temporary.push_back(0);
    char *created = ::mkdtemp(temporary.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
    preserve_ = root != nullptr;
    for (const char *name : {"RJ_CONSAN_ARTIFACT_DIR", "RJ_CONSAN_CAPTURE_PATH",
                             "RJ_CONSAN_SELECTION_PROFILE", "RJ_CONSAN_APPLICATION_PROFILE"}) {
      const char *value = std::getenv(name);
      environment_[name] = value ? std::optional<std::string>(value) : std::nullopt;
    }
    ASSERT_EQ(::setenv("RJ_CONSAN_ARTIFACT_DIR", artifact_fixture(), 1), 0);
    capture_ = directory_ / "capture.bin";
    ASSERT_EQ(::setenv("RJ_CONSAN_CAPTURE_PATH", capture_.c_str(), 1), 0);
    std::string profile;
    constexpr char kDigits[] = "0123456789abcdef";
    for (const uint8_t value : image_facts_.selection_profile) {
      profile.push_back(kDigits[value >> 4]);
      profile.push_back(kDigits[value & 15]);
    }
    ASSERT_EQ(::setenv("RJ_CONSAN_SELECTION_PROFILE", profile.c_str(), 1), 0);
    if (application_profile())
      ASSERT_EQ(::setenv("RJ_CONSAN_APPLICATION_PROFILE", "1", 1), 0);
    else
      ASSERT_EQ(::unsetenv("RJ_CONSAN_APPLICATION_PROFILE"), 0);
    const char *native_images = std::getenv("RJ_CONSAN_NATIVE_IMAGES");
    environment_["RJ_CONSAN_NATIVE_IMAGES"] =
        native_images ? std::optional<std::string>(native_images) : std::nullopt;
    ASSERT_EQ(::unsetenv("RJ_CONSAN_NATIVE_IMAGES"), 0);
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
    const rj_consan_statistics initial = statistics_();
    EXPECT_EQ(initial.submitted, 0u);
    EXPECT_EQ(initial.forwarded, 0u);
    EXPECT_EQ(initial.copied, 0u);
    EXPECT_EQ(initial.written, 0u);
    EXPECT_EQ(initial.reused, 0u);
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
      EXPECT_EQ(core_.hsa_init_fn, original_initialization_);
    }
    if (library_) {
      ::dlclose(library_);
    }
    for (const auto &[name, value] : environment_) {
      if (value) {
        (void)::setenv(name.c_str(), value->c_str(), 1);
      } else {
        (void)::unsetenv(name.c_str());
      }
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
    core_.hsa_init_fn = [] {
      ++current_->initialization_calls_;
      if (current_->initialization_status_ != HSA_STATUS_SUCCESS)
        return current_->initialization_status_;
      ++current_->hsa_references_;
      return HSA_STATUS_SUCCESS;
    };
    original_initialization_ = core_.hsa_init_fn;
    core_.hsa_shut_down_fn = [] {
      ++current_->shutdown_calls_;
      if (current_->shutdown_status_ != HSA_STATUS_SUCCESS)
        return current_->shutdown_status_;
      if (current_->hsa_references_ == 0)
        return HSA_STATUS_ERROR_NOT_INITIALIZED;
      if (current_->resume_shutdown_.valid()) {
        current_->shutdown_entered_.set_value();
        current_->resume_shutdown_.wait();
      }
      --current_->hsa_references_;
      if (current_->unload_during_shutdown_ && current_->hsa_references_ == 0) {
        const decltype(CoreApiTable::hsa_executable_destroy_fn) destroy =
            current_->core_.hsa_executable_destroy_fn;
        current_->unload_();
        current_->installed_ = false;
        if (current_->close_during_shutdown_) {
          if (::dlclose(current_->library_) != 0)
            return HSA_STATUS_ERROR;
          current_->library_ = nullptr;
        }
        return destroy({7});
      }
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_code_object_reader_create_from_memory_fn = create_reader;
    core_.hsa_code_object_reader_destroy_fn = destroy_reader;
    core_.hsa_executable_load_agent_code_object_fn = load_code;
    core_.hsa_executable_freeze_fn = [](hsa_executable_t, const char *) {
      ++current_->freeze_calls_;
      return current_->freeze_status_;
    };
    core_.hsa_executable_destroy_fn = [](hsa_executable_t) {
      if (current_->check_partial_load_storage_) {
        // The loader's reader has already gone away. Its borrowed prepared
        // bytes must remain valid until this underlying destroy returns.
        EXPECT_FALSE(current_->partial_load_bytes_.empty());
        EXPECT_EQ(current_->partial_load_bytes_.size(), current_->loaded_image_.size());
        EXPECT_TRUE(std::equal(current_->partial_load_bytes_.begin(),
                               current_->partial_load_bytes_.end(), current_->loaded_image_.begin(),
                               current_->loaded_image_.end()));
        current_->checked_partial_load_storage_ = true;
        current_->partial_load_bytes_ = {};
      }
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_executable_get_symbol_by_name_fn = [](hsa_executable_t, const char *name,
                                                    const hsa_agent_t *,
                                                    hsa_executable_symbol_t *symbol) {
      for (size_t index = 0; index < current_->image_facts_.entries.size(); ++index) {
        if (current_->image_facts_.entries[index].binding.name + ".kd" == name) {
          symbol->handle = 10 + index;
          return HSA_STATUS_SUCCESS;
        }
      }
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    };
    core_.hsa_executable_symbol_get_info_fn = symbol_info;
    core_.hsa_queue_destroy_fn = [](hsa_queue_t *) {
      ++current_->queue_destroy_calls_;
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_signal_create_fn = create_signal;
    core_.hsa_signal_destroy_fn = destroy_signal;
    core_.hsa_signal_load_scacquire_fn = load_signal;
    core_.hsa_signal_store_screlease_fn = store_signal;
    core_.hsa_signal_subtract_screlease_fn = subtract_signal;
    core_.hsa_signal_wait_scacquire_fn = wait_signal;
    amd_.hsa_amd_signal_create_fn = create_amd_signal;
    core_.hsa_iterate_agents_fn = [](hsa_status_t (*callback)(hsa_agent_t, void *), void *data) {
      return callback(hsa_agent_t{2}, data);
    };
    core_.hsa_agent_get_info_fn = [](hsa_agent_t agent, hsa_agent_info_t attribute, void *value) {
      if (attribute == HSA_AGENT_INFO_DEVICE) {
        *static_cast<hsa_device_type_t *>(value) =
            agent.handle == 1 ? HSA_DEVICE_TYPE_GPU : HSA_DEVICE_TYPE_CPU;
      } else if (attribute == HSA_AGENT_INFO_ISA) {
        *static_cast<hsa_isa_t *>(value) = {3};
      } else if (attribute == HSA_AGENT_INFO_WORKGROUP_MAX_SIZE) {
        *static_cast<uint32_t *>(value) = 1024;
      } else if (attribute == HSA_AGENT_INFO_WORKGROUP_MAX_DIM) {
        const std::array<uint16_t, 3> dimensions = {1024, 1024, 1024};
        std::memcpy(value, dimensions.data(), sizeof(dimensions));
      } else {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      return HSA_STATUS_SUCCESS;
    };
    core_.hsa_isa_get_info_alt_fn = [](hsa_isa_t, hsa_isa_info_t attribute, void *value) {
      const std::string name =
          "amdgcn-amd-amdhsa--gfx" + std::to_string(current_->image_facts_.target);
      if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
        *static_cast<uint32_t *>(value) = static_cast<uint32_t>(name.size() + 1);
      } else if (attribute == HSA_ISA_INFO_NAME) {
        std::memcpy(value, name.c_str(), name.size() + 1);
      } else {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_agent_iterate_memory_pools_fn =
        [](hsa_agent_t, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
          return callback(hsa_amd_memory_pool_t{4}, data);
        };
    amd_.hsa_amd_memory_pool_get_info_fn = [](hsa_amd_memory_pool_t,
                                              hsa_amd_memory_pool_info_t attribute, void *value) {
      if (attribute == HSA_AMD_MEMORY_POOL_INFO_SEGMENT) {
        *static_cast<uint32_t *>(value) = 0;
      } else if (attribute == HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS) {
        *static_cast<uint32_t *>(value) = 3;
      } else if (attribute == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED) {
        *static_cast<bool *>(value) = true;
      } else {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      }
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_memory_pool_allocate_fn = allocate;
    amd_.hsa_amd_memory_pool_free_fn = free_memory;
    amd_.hsa_amd_agents_allow_access_fn = [](uint32_t, const hsa_agent_t *, const uint32_t *,
                                             const void *) { return HSA_STATUS_SUCCESS; };
    amd_.hsa_amd_pointer_info_fn = pointer_info;
    amd_.hsa_amd_queue_create_fn = [](hsa_agent_t, hsa_amd_queue_create_desc_t *descs,
                                      uint32_t count) {
      ++current_->native_queue_calls_;
      if (descs == nullptr || count == 0)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      for (uint32_t index = 0; index < count; ++index)
        descs[index].queue = &current_->plain_queue_;
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_queue_intercept_create_fn = [](hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                                void (*)(hsa_status_t, hsa_queue_t *, void *),
                                                void *, uint32_t, uint32_t, hsa_queue_t **queue) {
      if (current_->queue_create_status_ != HSA_STATUS_SUCCESS) {
        *queue = nullptr;
        return current_->queue_create_status_;
      }
      *queue = current_->queue_create_result_ ? current_->queue_create_result_ : &current_->queue_;
      return HSA_STATUS_SUCCESS;
    };
    amd_.hsa_amd_queue_intercept_register_fn =
        [](hsa_queue_t *, hsa_amd_queue_intercept_handler handler, void *context) {
          if (current_->block_queue_registration_) {
            current_->queue_registration_entered_.set_value();
            current_->resume_queue_registration_.wait();
          }
          if (current_->queue_register_status_ != HSA_STATUS_SUCCESS)
            return current_->queue_register_status_;
          current_->handler_ = handler;
          current_->handler_context_ = context;
          return HSA_STATUS_SUCCESS;
        };
    amd_.hsa_amd_profiling_set_profiler_enabled_fn = [](hsa_queue_t *, int enabled) {
      ++current_->profiling_calls_;
      current_->profiling_enabled_ = enabled;
      return current_->profiling_status_;
    };
  }

  bool bind() {
    return core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                              &reader_) == HSA_STATUS_SUCCESS &&
           core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr) ==
               HSA_STATUS_SUCCESS &&
           core_.hsa_executable_freeze_fn({7}, nullptr) == HSA_STATUS_SUCCESS;
  }

  bool create_queue_and_signal(bool amd_signal = false) {
    hsa_queue_t *queue = nullptr;
    if (core_.hsa_queue_create_fn({1}, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                  &queue) != HSA_STATUS_SUCCESS ||
        queue != &queue_) {
      return false;
    }
    const hsa_status_t status = amd_signal
                                    ? amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &signal_)
                                    : core_.hsa_signal_create_fn(1, 0, nullptr, &signal_);
    return status == HSA_STATUS_SUCCESS;
  }

  hsa_kernel_dispatch_packet_t dispatch_packet(hsa_signal_t completion, uint32_t grid = 64,
                                               const void *arguments = nullptr,
                                               uint16_t workgroup = 64) {
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
    packet.completion_signal = completion;
    return packet;
  }

  void submit(uint32_t grid = 64, const void *arguments = nullptr, uint16_t workgroup = 64) {
    const hsa_kernel_dispatch_packet_t packet =
        dispatch_packet(signal_, grid, arguments, workgroup);
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
    const FailureOr<LaunchLayout> layout =
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
    const FailureOr<Capture> captured = decode_capture(read_capture());
    ASSERT_TRUE(captured.succeeded());
    ASSERT_EQ(captured.value().executions.size(), 2u);
    const CapturedExecution &first = captured.value().executions[0];
    const CapturedExecution &second = captured.value().executions[1];
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
    const std::span<const uint8_t> bytes = current_->readers_.at(reader.handle);
    ++current_->load_calls_;
    current_->loaded_image_.assign(bytes.begin(), bytes.end());
    if (current_->load_status_ != HSA_STATUS_SUCCESS) {
      current_->partial_load_bytes_ = bytes;
    }
    return current_->load_status_;
  }
  static hsa_status_t symbol_info(hsa_executable_symbol_t symbol,
                                  hsa_executable_symbol_info_t attribute, void *value) {
    if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT &&
        current_->resume_symbol_query_.valid()) {
      current_->symbol_query_entered_.set_value();
      current_->resume_symbol_query_.wait();
    }
    if (symbol.handle < 10 || symbol.handle - 10 >= current_->image_facts_.entries.size()) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    const size_t index = symbol.handle - 10;
    const ImageEntryFacts &entry = current_->image_facts_.entries[index];
    if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
      *static_cast<uint64_t *>(value) = 0x123400 + index * 0x100;
    } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE) {
      uint32_t bytes = entry.binding.original_kernarg_bytes;
      if (entry.selected) {
        const std::optional<KernargExtensionLayout> layout =
            make_kernarg_extension_layout(bytes, kContextPointerPayloads);
        if (!layout) {
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        bytes = layout->wrapper_size;
      }
      *static_cast<uint32_t *>(value) = bytes;
    } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT) {
      *static_cast<uint32_t *>(value) = 16;
    } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
      *static_cast<uint32_t *>(value) = entry.binding.prepared_group_bytes;
    } else if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
      *static_cast<uint32_t *>(value) = entry.binding.prepared_private_bytes;
    } else {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t allocate(hsa_amd_memory_pool_t, size_t bytes, uint32_t, void **memory) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->last_allocation_bytes_ = bytes;
    if (++current_->allocation_calls_ == current_->fail_allocation_) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    *memory = std::aligned_alloc(64, (bytes + 63) & ~size_t{63});
    if (!*memory) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    current_->allocations_[*memory] = bytes;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t free_memory(void *memory) {
    std::lock_guard lock(current_->fake_mutex_);
    // Normal launches may replace report_; only the armed, single-launch
    // finish test needs to inspect it here.
    if (current_->allocations_.erase(memory) != 1) {
      return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    }
    std::free(memory);
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t pointer_info(const void *pointer, hsa_amd_pointer_info_t *info,
                                   void *(*)(size_t), uint32_t *, hsa_agent_t **) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(current_->arguments_.data());
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address < base || address >= base + current_->arguments_.size()) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    info->size = sizeof(*info);
    info->type = HSA_EXT_POINTER_TYPE_HSA;
    info->agentBaseAddress = info->hostBaseAddress = current_->arguments_.data();
    info->sizeInBytes = current_->argument_extent_;
    info->global_flags = current_->pointer_global_flags_;
    switch (current_->pointer_case_) {
    case 1:
      info->type = HSA_EXT_POINTER_TYPE_UNKNOWN;
      break;
    case 2:
      info->hostBaseAddress = nullptr;
      break;
    case 3:
      info->agentBaseAddress = current_->arguments_.data() + 1;
      break;
    case 4:
      info->size = offsetof(hsa_amd_pointer_info_t, global_flags);
      break;
    case 5:
      info->sizeInBytes = current_->facts_.entry.original_kernarg_bytes - 1;
      break;
    case 6:
      info->global_flags = 0;
      break;
    }
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t create_signal(hsa_signal_value_t value, uint32_t, const hsa_agent_t *,
                                    hsa_signal_t *signal) {
    std::lock_guard lock(current_->fake_mutex_);
    if (current_->fail_proxy_ && current_->signal_.handle != 0) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    signal->handle = ++current_->next_signal_;
    current_->signal_values_[signal->handle] = value;
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t destroy_signal(hsa_signal_t signal) {
    std::lock_guard lock(current_->fake_mutex_);
    current_->signal_values_.erase(signal.handle);
    return HSA_STATUS_SUCCESS;
  }
  static hsa_status_t create_amd_signal(hsa_signal_value_t value, uint32_t consumers,
                                        const hsa_agent_t *agents, uint64_t attributes,
                                        hsa_signal_t *signal) {
    current_->amd_signal_value_ = value;
    current_->amd_signal_consumers_ = consumers;
    current_->amd_signal_agents_ = agents;
    current_->amd_signal_attributes_ = attributes;
    if (current_->amd_signal_status_ != HSA_STATUS_SUCCESS) {
      return current_->amd_signal_status_;
    }
    return create_signal(value, consumers, agents, signal);
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
    std::unique_lock lock(current_->fake_mutex_);
    current_->fake_condition_.wait_for(lock, std::chrono::milliseconds(1), [&] {
      return current_->signal_values_.at(signal.handle) < 1;
    });
    const hsa_signal_value_t observed = current_->signal_values_.at(signal.handle);
    if (current_->delayed_native_signal_ != 0 &&
        signal.handle != current_->delayed_native_signal_ && !current_->following_wait_reported_) {
      current_->following_wait_reported_ = true;
      current_->following_wait_entered_.set_value();
    }
    if (signal.handle == current_->delayed_native_signal_ && observed == 0 &&
        current_->resume_native_wait_.valid()) {
      lock.unlock();
      current_->native_wait_entered_.set_value();
      current_->resume_native_wait_.wait();
    }
    return observed;
  }
  static void barrier_writer(const void *packets, uint64_t count) {
    ASSERT_EQ(count, 1u);
    std::memcpy(&current_->published_, packets, sizeof(current_->published_));
    ++current_->published_packets_;
    hsa_barrier_and_packet_t barrier{};
    std::memcpy(&barrier, packets, sizeof(barrier));
    current_->published_barriers_.push_back(barrier);
  }

  static void completing_barrier_writer(const void *packets, uint64_t count) {
    barrier_writer(packets, count);
    // A real downstream writer may call another tool. Re-enter our statistics
    // accessor to require publication outside the ConSan runtime mutex.
    EXPECT_EQ(current_->statistics_().submitted, 0u);
    const hsa_barrier_and_packet_t &barrier = current_->published_barriers_.back();
    for (const hsa_signal_t dependency : barrier.dep_signal) {
      if (dependency.handle != 0) {
        EXPECT_EQ(load_signal(dependency), 0);
      }
    }
    if (barrier.completion_signal.handle != 0) {
      store_signal(barrier.completion_signal, load_signal(barrier.completion_signal) - 1);
    }
  }

  static void packet_writer(const void *packets, uint64_t count) {
    ASSERT_EQ(count, 1u);
    ConSanHsaPlugin &state = *current_;
    std::memcpy(&state.published_, packets, sizeof(state.published_));
    ++state.published_packets_;
    ASSERT_GE(state.published_.kernel_object, 0x123400u);
    const size_t index = (state.published_.kernel_object - 0x123400) / 0x100;
    ASSERT_LT(index, state.image_facts_.entries.size());
    const ImageEntryFacts &entry = state.image_facts_.entries[index];
    if (!entry.selected) {
      EXPECT_EQ(state.published_.kernarg_address, state.arguments_.data());
      EXPECT_EQ(state.published_.completion_signal.handle, state.signal_.handle);
      // Native publication can reenter the API without the runtime lock held.
      EXPECT_EQ(state.statistics_().native_submitted, 1u);
      if (state.auto_complete_) {
        store_signal(state.published_.completion_signal, 0);
      }
      return;
    }
    inspect_selected(state.published_, entry);
    if (state.auto_complete_)
      store_signal(state.published_.completion_signal, 0);
  }

  static void inspect_selected(const hsa_kernel_dispatch_packet_t &packet,
                               const ImageEntryFacts &entry) {
    ConSanHsaPlugin &state = *current_;
    const std::optional<KernargExtensionLayout> layout = make_kernarg_extension_layout(
        entry.binding.original_kernarg_bytes, kContextPointerPayloads);
    ASSERT_TRUE(layout.has_value());
    const auto wrapper =
        std::span(static_cast<const uint8_t *>(packet.kernarg_address), layout->wrapper_size);
    EXPECT_EQ(*wire::read_u64(wrapper, layout->original_kernarg_pointer_offset),
              reinterpret_cast<uint64_t>(state.arguments_.data()));
    EXPECT_TRUE(std::equal(wrapper.begin(), wrapper.begin() + entry.binding.original_kernarg_bytes,
                           state.arguments_.begin()));
    const auto context = std::span(
        reinterpret_cast<const uint8_t *>(*wire::read_u64(wrapper, layout->payload_offsets[0])),
        kContextBytes);
    state.report_ =
        reinterpret_cast<uint8_t *>(*wire::read_u64(context, wire::kContextReportPointer));
    state.report_bytes_ = *wire::read_u64(context, wire::kContextReportBytes);
    // This deliberately malformed row proves that the real runtime saves raw
    // bytes without diagnosing them. No ISA executes in this fake-API unit test.
    wire::write_u32(std::span(state.report_, state.report_bytes_),
                    kReportHeaderBytes + wire::kRowStatus, state.row_status_);
  }

  std::vector<uint8_t> read_capture() {
    std::ifstream file(capture_, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  }

  inline static ConSanHsaPlugin *current_ = nullptr;
  CoreApiTable core_{};
  decltype(CoreApiTable::hsa_init_fn) original_initialization_ = nullptr;
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
  hsa_queue_t plain_queue_{};
  hsa_queue_t *queue_create_result_ = nullptr;
  uint32_t profiling_calls_ = 0;
  int profiling_enabled_ = 0;
  hsa_status_t profiling_status_ = HSA_STATUS_SUCCESS;
  uint32_t hsa_references_ = 1;
  uint32_t initialization_calls_ = 0, shutdown_calls_ = 0;
  hsa_status_t initialization_status_ = HSA_STATUS_SUCCESS;
  hsa_status_t shutdown_status_ = HSA_STATUS_SUCCESS;
  std::promise<void> shutdown_entered_;
  std::shared_future<void> resume_shutdown_;
  uint32_t native_queue_calls_ = 0, queue_destroy_calls_ = 0;
  hsa_status_t queue_create_status_ = HSA_STATUS_SUCCESS;
  hsa_status_t queue_register_status_ = HSA_STATUS_SUCCESS;
  bool block_queue_registration_ = false;
  std::promise<void> queue_registration_entered_;
  std::shared_future<void> resume_queue_registration_;
  hsa_code_object_reader_t reader_{};
  hsa_signal_t signal_{};
  hsa_kernel_dispatch_packet_t published_{};
  std::vector<hsa_barrier_and_packet_t> published_barriers_;
  std::mutex fake_mutex_;
  std::condition_variable fake_condition_;
  std::promise<void> symbol_query_entered_;
  std::shared_future<void> resume_symbol_query_;
  std::promise<void> native_wait_entered_;
  std::shared_future<void> resume_native_wait_;
  uint64_t delayed_native_signal_ = 0;
  std::promise<void> following_wait_entered_;
  bool following_wait_reported_ = false;
  std::map<uint64_t, hsa_signal_value_t> signal_values_;
  std::map<uint64_t, std::span<const uint8_t>> readers_;
  std::map<void *, size_t> allocations_;
  std::map<std::string, std::optional<std::string>> environment_;
  std::filesystem::path directory_, capture_;
  std::vector<uint8_t> original_, loaded_image_;
  Manifest facts_;
  ImageFacts image_facts_;
  size_t argument_extent_ = 16;
  alignas(64) std::array<uint8_t, 32> arguments_{};
  uint8_t *report_ = nullptr;
  size_t report_bytes_ = 0;
  size_t last_allocation_bytes_ = 0;
  uint64_t next_signal_ = 100, next_reader_ = 200;
  uint32_t allocation_calls_ = 0, fail_allocation_ = 0, published_packets_ = 0,
           forwarding_calls_ = 0;
  uint32_t row_status_ = UINT32_MAX;
  unsigned pointer_case_ = 0;
  uint32_t pointer_global_flags_ =
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  hsa_status_t amd_signal_status_ = HSA_STATUS_SUCCESS;
  hsa_status_t freeze_status_ = HSA_STATUS_SUCCESS;
  uint32_t freeze_calls_ = 0;
  hsa_status_t load_status_ = HSA_STATUS_SUCCESS;
  uint32_t load_calls_ = 0;
  std::span<const uint8_t> partial_load_bytes_;
  bool check_partial_load_storage_ = false;
  bool checked_partial_load_storage_ = false;
  hsa_signal_value_t amd_signal_value_ = 0;
  uint32_t amd_signal_consumers_ = 0;
  const hsa_agent_t *amd_signal_agents_ = nullptr;
  uint64_t amd_signal_attributes_ = 0;
  bool installed_ = false, preserve_ = false, fail_proxy_ = false, auto_complete_ = true;
  bool unload_during_shutdown_ = false, close_during_shutdown_ = false;
};

class ConSanApplicationHsaPlugin : public ConSanHsaPlugin {
protected:
  bool application_profile() const override { return true; }
  void configure_native_image(bool configure_capture = true) {
    EXPECT_NE(finish_(), 0);
    unload_();
    installed_ = false;
    const std::filesystem::path path = directory_ / "native.hsaco";
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(original_.data()), original_.size());
    file.close();
    ASSERT_TRUE(file);
    ASSERT_EQ(::setenv("RJ_CONSAN_NATIVE_IMAGES", path.c_str(), 1), 0);
    ASSERT_EQ(::unsetenv("RJ_CONSAN_ARTIFACT_DIR"), 0);
    ASSERT_EQ(::unsetenv("RJ_CONSAN_SELECTION_PROFILE"), 0);
    if (!configure_capture) {
      ASSERT_EQ(::unsetenv("RJ_CONSAN_CAPTURE_PATH"), 0);
    }
    const std::string target = "gfx" + std::to_string(image_facts_.target);
    const FailureOr<NativeImageInventory> inventory = inspect_native_image(original_, target);
    ASSERT_TRUE(inventory.succeeded());
    image_facts_.entries = inventory.value().entries;
    initialize_api();
    ASSERT_TRUE(load_(&table_, 0, 0, nullptr));
    installed_ = true;
  }
  hsa_amd_queue_create_desc_t native_descriptor() {
    hsa_amd_queue_create_desc_t desc = {};
    desc.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
    desc.queue_size_bytes = 1024 * 1024;
    desc.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
    desc.engine.compute.private_segment_size = UINT32_MAX;
    return desc;
  }
  hsa_amd_ext_kernel_dispatch_packet_t extended_packet() {
    hsa_amd_ext_kernel_dispatch_packet_t packet = {};
    packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC | (1u << HSA_PACKET_HEADER_BARRIER) |
                    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    packet.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
    packet.setup = 3;
    packet.workgroup_size_x = 64;
    packet.workgroup_size_y = 2;
    packet.workgroup_size_z = 1;
    packet.cluster_count_x = 512;
    packet.cluster_count_y = packet.cluster_count_z = 1;
    packet.cluster_size_x = packet.cluster_size_y = packet.cluster_size_z = 1;
    packet.perf_hint.hint_val = 127;
    packet.group_segment_size = image_facts_.entries.front().binding.original_group_bytes;
    packet.kernel_object = 0x123400;
    packet.kernarg_address = arguments_.data();
    return packet;
  }
  static void application_writer(const void *packets, uint64_t count) {
    ASSERT_EQ(count, 1u);
    ConSanApplicationHsaPlugin &state = *static_cast<ConSanApplicationHsaPlugin *>(current_);
    std::array<uint8_t, 64> bytes = {};
    std::memcpy(bytes.data(), packets, bytes.size());
    state.application_packets_.push_back(bytes);
    hsa_amd_ext_kernel_dispatch_packet_t packet = {};
    std::memcpy(&packet, packets, sizeof(packet));
    state.application_completions_.push_back(packet.completion_signal);
    if (packet.kernel_object >= 0x123400) {
      const size_t index = (packet.kernel_object - 0x123400) / 0x100;
      if (index < state.image_facts_.entries.size() && state.image_facts_.entries[index].selected) {
        hsa_kernel_dispatch_packet_t normalized = {};
        normalized.kernel_object = packet.kernel_object;
        normalized.kernarg_address = packet.kernarg_address;
        inspect_selected(normalized, state.image_facts_.entries[index]);
        state.application_reports_.push_back(std::span(state.report_, state.report_bytes_));
      }
    }
    if (state.auto_complete_ && packet.completion_signal.handle != 0)
      store_signal(packet.completion_signal, 0);
  }
  void submit_application(const hsa_amd_ext_kernel_dispatch_packet_t &packet) {
    ASSERT_NE(handler_, nullptr);
    handler_(&packet, 1, 0, handler_context_, application_writer);
  }
  static void application_batch_writer(const void *packets, uint64_t count) {
    ASSERT_EQ(count, 1u);
    hsa_kernel_dispatch_packet_t packet = {};
    std::memcpy(&packet, packets, sizeof(packet));
    const uint16_t type = packet.header & 0xff;
    if (type == HSA_PACKET_TYPE_BARRIER_AND || type == HSA_PACKET_TYPE_BARRIER_OR)
      completing_barrier_writer(packets, count);
    else
      application_writer(packets, count);
  }
  hsa_status_t create_application_queue(bool native, hsa_queue_t **queue) {
    if (!native)
      return core_.hsa_queue_create_fn({1}, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                       queue);
    hsa_amd_queue_create_desc_t desc = native_descriptor();
    const hsa_status_t status = amd_.hsa_amd_queue_create_fn({1}, &desc, 1);
    *queue = desc.queue;
    return status;
  }
  void expect_more_than_sixteen_queues(bool native) {
    configure_native_image();
    ASSERT_TRUE(installed_);
    std::array<hsa_queue_t, 33> queues = {};
    size_t created_count = 0;
    for (size_t index = 0; index < queues.size() - 1; ++index) {
      queue_create_result_ = &queues[index];
      hsa_queue_t *created = nullptr;
      const hsa_status_t status = create_application_queue(native, &created);
      EXPECT_EQ(status, HSA_STATUS_SUCCESS) << index;
      EXPECT_EQ(created, &queues[index]);
      if (status != HSA_STATUS_SUCCESS || created != &queues[index])
        break;
      ++created_count;
    }
    if (created_count == queues.size() - 1) {
      queue_create_result_ = &queues.back();
      queue_create_status_ = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      hsa_queue_t *created = nullptr;
      EXPECT_EQ(create_application_queue(native, &created), HSA_STATUS_ERROR_OUT_OF_RESOURCES);
      EXPECT_EQ(created, nullptr);
      queue_create_status_ = HSA_STATUS_SUCCESS;
      const hsa_status_t retried = create_application_queue(native, &created);
      EXPECT_EQ(retried, HSA_STATUS_SUCCESS);
      EXPECT_EQ(created, &queues.back());
      if (retried == HSA_STATUS_SUCCESS && created == &queues.back())
        ++created_count;
      EXPECT_TRUE(bind());
      if (!HasFailure()) {
        submit_application(extended_packet());
        EXPECT_EQ(application_packets_.size(), 1u);
        EXPECT_EQ(drain_(), 0);
        EXPECT_EQ(statistics_().native_completed, 1u);
      }
    }
    for (size_t index = 0; index < created_count; ++index)
      EXPECT_EQ(core_.hsa_queue_destroy_fn(&queues[index]), HSA_STATUS_SUCCESS);
    if (created_count == queues.size()) {
      EXPECT_EQ(finish_(), 0);
    }
  }
  bool wait_for_native_completed(uint64_t count) {
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (statistics_().native_completed < count) {
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }
  static void late_application_teardown() {
    ConSanApplicationHsaPlugin &state = *static_cast<ConSanApplicationHsaPlugin *>(current_);
    if (state.core_.hsa_executable_destroy_fn({7}) != HSA_STATUS_SUCCESS)
      std::_Exit(82);
    state.unload_();
    constexpr char message[] = "consan late teardown completed\n";
    if (::write(STDERR_FILENO, message, sizeof(message) - 1) != sizeof(message) - 1)
      std::_Exit(86);
    std::_Exit(0);
  }
  std::vector<std::array<uint8_t, 64>> application_packets_;
  std::vector<hsa_signal_t> application_completions_;
  std::vector<std::span<uint8_t>> application_reports_;
};

#if defined(CONSAN_MULTI_TRACE_ARTIFACT_FIXTURE)
class ConSanTraceHsaPlugin : public ConSanApplicationHsaPlugin {
protected:
  const char *original_fixture() const override { return CONSAN_MULTI_ORIGINAL_FIXTURE; }
  const char *artifact_fixture() const override { return CONSAN_MULTI_TRACE_ARTIFACT_FIXTURE; }
  hsa_amd_ext_kernel_dispatch_packet_t trace_packet(size_t index = 0) {
    hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
    packet.setup = 1;
    packet.workgroup_size_x = 32;
    packet.workgroup_size_y = 1;
    packet.cluster_count_x = 16;
    packet.kernel_object += index * 0x100;
    packet.group_segment_size = image_facts_.entries[index].binding.original_group_bytes;
    return packet;
  }
  bool wait_for_copied(uint64_t count) {
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (statistics_().copied < count) {
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }
};

TEST_F(ConSanTraceHsaPlugin, RejectedDispatchNotifiesFollowingBarrierWaiters) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_allocation_ = 1;
  submit_application(trace_packet());
  for (const bool batch : {false, true}) {
    std::array<hsa_barrier_and_packet_t, 2> barriers = {};
    barriers[0].header = HSA_PACKET_TYPE_BARRIER_AND;
    barriers[1].header = HSA_PACKET_TYPE_BARRIER_OR;
    for (hsa_barrier_and_packet_t &barrier : barriers) {
      ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &barrier.completion_signal),
                HSA_STATUS_SUCCESS);
      barrier.dep_signal[0] = signal_;
    }
    if (batch)
      handler_(barriers.data(), barriers.size(), 0, handler_context_, application_writer);
    else
      for (const hsa_barrier_and_packet_t &barrier : barriers)
        handler_(&barrier, 1, 0, handler_context_, application_writer);
    for (const hsa_barrier_and_packet_t &barrier : barriers)
      EXPECT_EQ(load_signal(barrier.completion_signal), -1);
  }
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanTraceHsaPlugin, RejectedBarrierRetainsPendingSelectedWorkUntilCompletion) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(trace_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  const hsa_signal_t proxy = application_completions_[0];
  fail_allocation_ = allocation_calls_ + 1;
  submit_application(trace_packet(1));
  hsa_barrier_and_packet_t barrier = {};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = signal_;
  barrier.dep_signal[0] = proxy;
  handler_(&barrier, 1, 0, handler_context_, application_writer);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(load_signal(proxy), 1);
  EXPECT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(statistics_().submitted, 1u);
  EXPECT_EQ(statistics_().copied, 0u);
  std::future<int> finishing = std::async(std::launch::async, [&] { return finish_(); });
  EXPECT_EQ(finishing.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(proxy, 0);
  EXPECT_NE(finishing.get(), 0);
  EXPECT_TRUE(allocations_.empty());
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanTraceHsaPlugin, RejectedBarriersLeaveUntrackedAndNullCompletionsAlone) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t untracked = {};
  ASSERT_EQ(create_signal(1, 0, nullptr, &untracked), HSA_STATUS_SUCCESS);
  fail_allocation_ = 1;
  submit_application(trace_packet());
  for (const hsa_signal_t completion : {untracked, hsa_signal_t{0}}) {
    hsa_barrier_and_packet_t barrier = {};
    barrier.header = HSA_PACKET_TYPE_BARRIER_OR;
    barrier.completion_signal = completion;
    handler_(&barrier, 1, 0, handler_context_, application_writer);
  }
  EXPECT_EQ(load_signal(untracked), 1);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanTraceHsaPlugin, RejectsSelectedNonunitClusterBeforeAllocatingOrSubmitting) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = trace_packet();
  packet.cluster_size_x = 2;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanTraceHsaPlugin, NativeNonunitClusterRetainsPacketAndCompletionOwnership) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  hsa_amd_ext_kernel_dispatch_packet_t packet = trace_packet(2);
  packet.cluster_size_x = 2;
  packet.dep_signal = signal_;
  submit_application(packet);
  ASSERT_EQ(application_packets_.size(), 1u);
  const hsa_signal_t proxy = application_completions_[0];
  EXPECT_NE(proxy.handle, 0u);
  packet.completion_signal = proxy;
  EXPECT_EQ(std::memcmp(application_packets_[0].data(), &packet, sizeof(packet)), 0);
  EXPECT_EQ(load_signal(signal_), 1);
  store_signal(proxy, 0);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanTraceHsaPlugin, NullCompletionPreservesExtendedDependenciesAndArguments) {
  pointer_global_flags_ = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  hsa_amd_ext_kernel_dispatch_packet_t packet = trace_packet();
  packet.dep_signal = signal_;
  submit_application(packet);
  ASSERT_EQ(application_packets_.size(), 1u);
  ASSERT_EQ(application_reports_.size(), 1u);
  const std::vector<uint8_t> expected_report(application_reports_[0].begin(),
                                             application_reports_[0].end());
  hsa_amd_ext_kernel_dispatch_packet_t published = {};
  std::memcpy(&published, application_packets_[0].data(), sizeof(published));
  EXPECT_NE(published.kernarg_address, packet.kernarg_address);
  EXPECT_NE(published.completion_signal.handle, 0u);
  EXPECT_NE(published.completion_signal.handle, signal_.handle);
  packet.kernarg_address = published.kernarg_address;
  packet.completion_signal = published.completion_signal;
  constexpr uint16_t fences = (3u << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                              (3u << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  packet.header = (packet.header & ~fences) |
                  (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                  (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  std::array<uint8_t, 64> expected = {};
  std::memcpy(expected.data(), &packet, sizeof(packet));
  EXPECT_EQ(application_packets_[0], expected);
  store_signal(published.completion_signal, 0);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(forwarding_calls_, 0u);
  EXPECT_EQ(statistics_().forwarded, 0u);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(finish_(), 0);
  const FailureOr<Capture> capture = decode_capture(read_capture());
  ASSERT_TRUE(capture.succeeded());
  ASSERT_EQ(capture.value().executions.size(), 1u);
  EXPECT_EQ(capture.value().header.execution_conditions, kApplicationExecutionConditions);
  EXPECT_EQ(capture.value().executions[0].completion, kCompletionVisible);
  EXPECT_EQ(capture.value().executions[0].dimensions.grid[0], 512u);
  EXPECT_EQ(capture.value().executions[0].report, expected_report);
  EXPECT_TRUE(published_barriers_.empty());
}

TEST_F(ConSanTraceHsaPlugin, ReverseCompletionCopiesBeforeSerialOrderedWritingAndSlotReuse) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(trace_packet());
  arguments_[0] = 23;
  submit_application(trace_packet(1));
  ASSERT_EQ(application_reports_.size(), 2u);
  const std::vector<uint8_t> first(application_reports_[0].begin(), application_reports_[0].end());
  const std::vector<uint8_t> second(application_reports_[1].begin(), application_reports_[1].end());
  store_signal(application_completions_[1], 0);
  ASSERT_TRUE(wait_for_copied(1));
  EXPECT_EQ(statistics_().written, 0u);
  std::fill(application_reports_[1].begin(), application_reports_[1].end(), 0xa5);
  arguments_[0] = 41;
  submit_application(trace_packet());
  ASSERT_EQ(application_reports_.size(), 3u);
  EXPECT_EQ(application_reports_[1].data(), application_reports_[2].data());
  const std::vector<uint8_t> third(application_reports_[2].begin(), application_reports_[2].end());
  store_signal(application_completions_[2], 0);
  ASSERT_TRUE(wait_for_copied(2));
  std::future<int> draining = std::async(std::launch::async, [&] { return drain_(); });
  EXPECT_EQ(draining.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  ASSERT_EQ(draining.get(), 0);
  ASSERT_EQ(finish_(), 0);
  const FailureOr<Capture> capture = decode_capture(read_capture());
  ASSERT_TRUE(capture.succeeded());
  ASSERT_EQ(capture.value().executions.size(), 3u);
  const std::array<uint8_t, 3> inputs = {7, 23, 41};
  const std::array<size_t, 3> entries = {0, 1, 0};
  const std::array<std::vector<uint8_t>, 3> reports = {first, second, third};
  for (size_t index = 0; index < 3; ++index) {
    const CapturedExecution &saved = capture.value().executions[index];
    EXPECT_EQ(saved.serial, index + 1);
    EXPECT_EQ(saved.entry_key, image_facts_.entries[entries[index]].key);
    EXPECT_EQ(saved.arguments[0], inputs[index]);
    EXPECT_EQ(saved.report, reports[index]);
  }
  EXPECT_EQ(statistics_().copied, 3u);
  EXPECT_EQ(statistics_().written, 3u);
  EXPECT_EQ(statistics_().forwarded, 0u);
  EXPECT_EQ(statistics_().reused, 1u);
}
TEST_F(ConSanTraceHsaPlugin, MoreThanSixteenPendingDispatchesRetainIndependentStorage) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  for (unsigned index = 0; index < 17; ++index)
    submit_application(trace_packet());
  ASSERT_EQ(application_completions_.size(), 17u);
  EXPECT_EQ(statistics_().submitted, 17u);
  EXPECT_EQ(statistics_().copied, 0u);
  for (const hsa_signal_t proxy : application_completions_)
    store_signal(proxy, 0);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(finish_(), 0);
  const FailureOr<Capture> captured = decode_capture(read_capture());
  ASSERT_TRUE(captured.succeeded());
  EXPECT_EQ(captured.value().executions.size(), 17u);
}

TEST_F(ConSanTraceHsaPlugin, QueueDestroyWaitsForSelectedObservationAndIndependentCopy) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(trace_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  std::future<hsa_status_t> destroying =
      std::async(std::launch::async, [&] { return core_.hsa_queue_destroy_fn(&queue_); });
  EXPECT_EQ(destroying.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  EXPECT_EQ(destroying.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(statistics_().copied, 1u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanTraceHsaPlugin, ExecutableDestroyRetainsFactsThroughSelectedCopyAndWriting) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(trace_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  std::future<hsa_status_t> destroying =
      std::async(std::launch::async, [&] { return core_.hsa_executable_destroy_fn({7}); });
  EXPECT_EQ(destroying.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  EXPECT_EQ(destroying.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(finish_(), 0);
  EXPECT_TRUE(decode_capture(read_capture()).succeeded());
}

class ConSanTracePointerAdmission : public ConSanTraceHsaPlugin,
                                    public ::testing::WithParamInterface<unsigned> {};

TEST_P(ConSanTracePointerAdmission, RejectUnprovenHostReadBeforePublishing) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  pointer_global_flags_ = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  pointer_case_ = GetParam();
  submit_application(trace_packet());
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

INSTANTIATE_TEST_SUITE_P(InvalidOriginalArguments, ConSanTracePointerAdmission,
                         ::testing::Range(1u, 7u));
#endif

TEST_F(ConSanApplicationHsaPlugin, LdsSelectedDispatchCannotSilentlySucceedInApplicationProfile) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
  packet.setup = 1;
  packet.workgroup_size_y = 1;
  packet.cluster_count_x = 1;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, NativeQueueBeforeImageLoadInstallsThePacketInterceptor) {
  hsa_amd_queue_create_desc_t desc = native_descriptor();
  EXPECT_EQ(amd_.hsa_amd_queue_create_fn({1}, &desc, 1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(native_queue_calls_, 1u);
  EXPECT_EQ(desc.queue, &queue_);
  EXPECT_NE(handler_, nullptr);
  EXPECT_EQ(queue_destroy_calls_, 1u);
}

TEST_F(ConSanApplicationHsaPlugin, CoreQueueCreationAdmitsMoreThanSixteenDistinctQueues) {
  expect_more_than_sixteen_queues(false);
}

TEST_F(ConSanApplicationHsaPlugin, AmdQueueCreationAdmitsMoreThanSixteenDistinctQueues) {
  expect_more_than_sixteen_queues(true);
}

TEST_F(ConSanApplicationHsaPlugin, NativeQueueRegistrationFailureReturnsNoUnobservedQueue) {
  queue_register_status_ = HSA_STATUS_ERROR;
  hsa_amd_queue_create_desc_t desc = native_descriptor();
  EXPECT_EQ(amd_.hsa_amd_queue_create_fn({1}, &desc, 1), HSA_STATUS_ERROR);
  EXPECT_EQ(desc.queue, nullptr);
  EXPECT_EQ(queue_destroy_calls_, 2u);
}

TEST_F(ConSanApplicationHsaPlugin, UnsupportedNativePlacementCannotSilentlyChangeToSystemMemory) {
  hsa_amd_queue_create_desc_t desc = native_descriptor();
  desc.flags = 1;
  EXPECT_EQ(amd_.hsa_amd_queue_create_fn({1}, &desc, 1), HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);
  EXPECT_EQ(desc.queue, nullptr);
  EXPECT_EQ(handler_, nullptr);
  EXPECT_EQ(queue_destroy_calls_, 1u);
}

TEST_F(ConSanApplicationHsaPlugin, FinishDuringQueueRegistrationWaitsForUnpublishedQueueCleanup) {
  block_queue_registration_ = true;
  std::promise<void> resume;
  resume_queue_registration_ = resume.get_future().share();
  hsa_amd_queue_create_desc_t desc = native_descriptor();
  std::future<hsa_status_t> creating =
      std::async(std::launch::async, [&] { return amd_.hsa_amd_queue_create_fn({1}, &desc, 1); });
  queue_registration_entered_.get_future().wait();
  std::future<int> finishing = std::async(std::launch::async, [&] { return finish_(); });
  EXPECT_EQ(finishing.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  resume.set_value();
  EXPECT_NE(creating.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(desc.queue, nullptr);
  EXPECT_NE(finishing.get(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
  EXPECT_EQ(queue_destroy_calls_, 2u);
}

TEST_F(ConSanApplicationHsaPlugin, KnownNativeImageLoadsWithoutAPreparedArtifact) {
  configure_native_image();
  ASSERT_TRUE(installed_);
  EXPECT_TRUE(bind());
  EXPECT_EQ(load_calls_, 1u);
  EXPECT_EQ(loaded_image_, original_);
  EXPECT_EQ(core_.hsa_executable_destroy_fn({7}), HSA_STATUS_SUCCESS);
  EXPECT_NE(finish_(), 0); // Loading alone is not a selected capture or execution proof.
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, NativeOnlyInstallAndExecutionDoNotRequireACapturePath) {
  configure_native_image(false);
  ASSERT_TRUE(installed_);
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit_application(extended_packet());
  ASSERT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(finish_(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, SelectedArtifactInstallStillRequiresACapturePath) {
  (void)finish_();
  unload_();
  installed_ = false;
  ASSERT_EQ(::unsetenv("RJ_CONSAN_CAPTURE_PATH"), 0);
  initialize_api();
  EXPECT_FALSE(load_(&table_, 0, 0, nullptr));
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
  EXPECT_EQ(core_.hsa_init_fn, original_initialization_);
}

TEST_F(ConSanApplicationHsaPlugin, ChangedNativeImageBytesNeverReachTheUnderlyingLoader) {
  configure_native_image();
  ASSERT_TRUE(installed_);
  original_.back() ^= 1;
  EXPECT_FALSE(bind());
  EXPECT_EQ(load_calls_, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, ExtendedNativePacketsKeepBytesAndUseIndependentCompletion) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
  packet.dep_signal = signal_;
  submit_application(packet);
  ASSERT_EQ(application_packets_.size(), 1u);
  ASSERT_EQ(application_completions_.size(), 1u);
  hsa_amd_ext_kernel_dispatch_packet_t first = packet;
  first.completion_signal = application_completions_[0];
  std::array<uint8_t, 64> first_bytes = {};
  std::memcpy(first_bytes.data(), &first, sizeof(first));
  EXPECT_EQ(application_packets_[0], first_bytes);
  packet.dep_signal = {};
  packet.cluster_count_x = 3;
  submit_application(packet);
  ASSERT_EQ(application_packets_.size(), 2u);
  ASSERT_EQ(application_completions_.size(), 2u);
  const hsa_signal_t completion = packet.completion_signal;
  packet.completion_signal = application_completions_[1];
  std::array<uint8_t, 64> original = {};
  std::memcpy(original.data(), &packet, sizeof(packet));
  EXPECT_EQ(application_packets_[1], original);
  packet.completion_signal = completion;
  EXPECT_EQ(statistics_().native_submitted, 2u);
  EXPECT_EQ(statistics_().native_completed, 0u);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_TRUE(published_barriers_.empty());
  EXPECT_NE(application_completions_[0].handle, signal_.handle);
  EXPECT_NE(application_completions_[0].handle, application_completions_[1].handle);
  for (const hsa_signal_t proxy : application_completions_)
    store_signal(proxy, 0);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 2u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(statistics_().written, 0u);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, RejectedNativeAdmissionNotifiesFollowingBarriers) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  fail_proxy_ = true;
  submit_application(extended_packet());
  fail_proxy_ = false;
  for (const bool batch : {false, true}) {
    std::array<hsa_barrier_and_packet_t, 2> barriers = {};
    barriers[0].header = HSA_PACKET_TYPE_BARRIER_AND;
    barriers[1].header = HSA_PACKET_TYPE_BARRIER_OR;
    for (hsa_barrier_and_packet_t &barrier : barriers) {
      ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &barrier.completion_signal),
                HSA_STATUS_SUCCESS);
      barrier.dep_signal[0] = signal_;
    }
    if (batch)
      handler_(barriers.data(), barriers.size(), 0, handler_context_, application_writer);
    else
      for (const hsa_barrier_and_packet_t &barrier : barriers)
        handler_(&barrier, 1, 0, handler_context_, application_writer);
    for (const hsa_barrier_and_packet_t &barrier : barriers)
      EXPECT_EQ(load_signal(barrier.completion_signal), -1);
  }
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, RejectedBarrierRetainsPendingNativeUntilActualCompletion) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  const hsa_signal_t proxy = application_completions_[0];
  fail_proxy_ = true;
  submit_application(extended_packet());
  fail_proxy_ = false;
  hsa_barrier_and_packet_t barrier = {};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = signal_;
  barrier.dep_signal[0] = proxy;
  handler_(&barrier, 1, 0, handler_context_, application_writer);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(load_signal(proxy), 1);
  EXPECT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(statistics_().native_submitted, 1u);
  EXPECT_EQ(statistics_().native_completed, 0u);
  std::future<int> finishing = std::async(std::launch::async, [&] { return finish_(); });
  EXPECT_EQ(finishing.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(proxy, 0);
  EXPECT_NE(finishing.get(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, RejectedNativeBarriersPreserveUntrackedAndNullCompletions) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t untracked = {};
  ASSERT_EQ(create_signal(1, 0, nullptr, &untracked), HSA_STATUS_SUCCESS);
  fail_proxy_ = true;
  submit_application(extended_packet());
  fail_proxy_ = false;
  for (const hsa_signal_t completion : {untracked, hsa_signal_t{0}}) {
    hsa_barrier_and_packet_t barrier = {};
    barrier.header = HSA_PACKET_TYPE_BARRIER_OR;
    barrier.completion_signal = completion;
    handler_(&barrier, 1, 0, handler_context_, application_writer);
  }
  EXPECT_EQ(load_signal(untracked), 1);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, CoalescedNativePacketsRetainOrderAndIndependentCompletions) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  std::array<hsa_amd_ext_kernel_dispatch_packet_t, 2> packets = {extended_packet(),
                                                                 extended_packet()};
  packets[1].cluster_count_x = 3;
  handler_(packets.data(), packets.size(), 0, handler_context_, application_writer);
  ASSERT_EQ(application_packets_.size(), 2u);
  EXPECT_NE(application_completions_[0].handle, application_completions_[1].handle);
  for (size_t index = 0; index < packets.size(); ++index) {
    packets[index].completion_signal = application_completions_[index];
    std::array<uint8_t, 64> expected = {};
    std::memcpy(expected.data(), &packets[index], sizeof(packets[index]));
    EXPECT_EQ(application_packets_[index], expected);
    store_signal(application_completions_[index], 0);
  }
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 2u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, FullExecutionBatchRetainsItsTrailingBarrierCompletion) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  const size_t executions = CaptureLimits{}.executions;
  std::vector<hsa_amd_ext_kernel_dispatch_packet_t> packets(executions + 1, extended_packet());
  hsa_barrier_and_packet_t barrier = {};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = signal_;
  std::memcpy(&packets.back(), &barrier, sizeof(barrier));
  handler_(packets.data(), packets.size(), 0, handler_context_, application_batch_writer);
  EXPECT_EQ(application_packets_.size(), executions);
  EXPECT_EQ(published_barriers_.size(), 1u);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_submitted, executions);
  EXPECT_EQ(statistics_().native_completed, executions);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, ExecutionBudgetRejectionNotifiesBarrierAndRetainsPendingLaunch) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  const size_t executions = CaptureLimits{}.executions;
  std::vector<hsa_amd_ext_kernel_dispatch_packet_t> initial(executions - 1, extended_packet());
  handler_(initial.data(), initial.size(), 0, handler_context_, application_writer);
  ASSERT_EQ(drain_(), 0);
  auto_complete_ = false;
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), executions);
  const hsa_signal_t proxy = application_completions_.back();
  hsa_barrier_and_packet_t barrier = {};
  barrier.header = HSA_PACKET_TYPE_BARRIER_OR;
  barrier.completion_signal = signal_;
  barrier.dep_signal[0] = proxy;
  std::array<hsa_amd_ext_kernel_dispatch_packet_t, 2> rejected = {extended_packet(), {}};
  std::memcpy(&rejected[1], &barrier, sizeof(barrier));
  handler_(rejected.data(), rejected.size(), 0, handler_context_, application_batch_writer);
  EXPECT_EQ(application_packets_.size(), executions);
  EXPECT_TRUE(published_barriers_.empty());
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(load_signal(proxy), 1);
  EXPECT_EQ(statistics_().native_completed, executions - 1);
  store_signal(proxy, 0);
  EXPECT_NE(finish_(), 0);
  EXPECT_EQ(statistics_().native_completed, executions);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, UnknownExtendedKernelNeverReachesTheWriter) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
  packet.kernel_object += 8;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, ExtendedGridOverflowNeverReachesTheWriter) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
  packet.cluster_count_x = UINT32_MAX;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, LaterCompletionCannotSatisfyAnEarlierDrain) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(extended_packet());
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 2u);
  store_signal(application_completions_[1], 0);
  ASSERT_TRUE(wait_for_native_completed(1));
  std::future<int> draining = std::async(std::launch::async, [&] { return drain_(); });
  EXPECT_EQ(draining.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  EXPECT_EQ(draining.get(), 0);
  EXPECT_EQ(statistics_().native_completed, 2u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, ExecutableDestroyWaitsForItsNativeLaunch) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  std::future<hsa_status_t> destroying =
      std::async(std::launch::async, [&] { return core_.hsa_executable_destroy_fn({7}); });
  EXPECT_EQ(destroying.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  EXPECT_EQ(destroying.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, IndependentQueueDestroyDoesNotWaitForAnotherQueuesLaunch) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit_application(extended_packet());
  const hsa_amd_queue_intercept_handler first_handler = handler_;
  void *first_context = handler_context_;
  hsa_queue_t second = {};
  queue_create_result_ = &second;
  hsa_queue_t *created = nullptr;
  ASSERT_EQ(
      core_.hsa_queue_create_fn({1}, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &created),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(created, &second);
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 2u);
  store_signal(application_completions_[1], 0);
  ASSERT_TRUE(wait_for_native_completed(1));
  EXPECT_EQ(core_.hsa_queue_destroy_fn(&second), HSA_STATUS_SUCCESS);
  EXPECT_EQ(statistics_().native_completed, 1u);
  handler_ = first_handler;
  handler_context_ = first_context;
  std::future<hsa_status_t> destroying =
      std::async(std::launch::async, [&] { return core_.hsa_queue_destroy_fn(&queue_); });
  EXPECT_EQ(destroying.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_[0], 0);
  EXPECT_EQ(destroying.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, OriginalNonzeroCompletionIsRejectedWithoutSignalMutation) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
  packet.completion_signal = signal_;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, ProfilingUsesTheOriginalQueueApiAndShutdownDrainsNativeWork) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  profiling_status_ = HSA_STATUS_ERROR;
  EXPECT_EQ(amd_.hsa_amd_profiling_set_profiler_enabled_fn(&queue_, 1), HSA_STATUS_ERROR);
  EXPECT_EQ(profiling_enabled_, 1);
  profiling_status_ = HSA_STATUS_SUCCESS;
  EXPECT_EQ(amd_.hsa_amd_profiling_set_profiler_enabled_fn(&queue_, 1), HSA_STATUS_SUCCESS);
  EXPECT_EQ(profiling_calls_, 2u);
  submit_application(extended_packet());
  ASSERT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanApplicationHsaPlugin, IntermediateHsaShutdownPreservesRemainingClientsNativeDispatch) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  ASSERT_NE(core_.hsa_init_fn, nullptr);
  ASSERT_EQ(core_.hsa_init_fn(), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_references_, 2u);
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 1u);
  submit_application(extended_packet());
  ASSERT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(drain_(), 0);
  auto_complete_ = false;
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 2u);
  std::future<hsa_status_t> shutting_down =
      std::async(std::launch::async, [&] { return core_.hsa_shut_down_fn(); });
  EXPECT_EQ(shutting_down.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_.back(), 0);
  EXPECT_EQ(shutting_down.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 0u);
  EXPECT_EQ(shutdown_calls_, 2u);
  EXPECT_EQ(statistics_().native_completed, 2u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, FailedHsaInitializationDoesNotDelayFinalNativeShutdown) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  ASSERT_NE(core_.hsa_init_fn, nullptr);
  initialization_status_ = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  EXPECT_EQ(core_.hsa_init_fn(), HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  EXPECT_EQ(hsa_references_, 1u);
  auto_complete_ = false;
  submit_application(extended_packet());
  ASSERT_EQ(application_completions_.size(), 1u);
  std::future<hsa_status_t> shutting_down =
      std::async(std::launch::async, [&] { return core_.hsa_shut_down_fn(); });
  EXPECT_EQ(shutting_down.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  store_signal(application_completions_.front(), 0);
  EXPECT_EQ(shutting_down.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 0u);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, FailedIntermediateHsaShutdownPreservesRemainingReference) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  ASSERT_NE(core_.hsa_init_fn, nullptr);
  ASSERT_EQ(core_.hsa_init_fn(), HSA_STATUS_SUCCESS);
  shutdown_status_ = HSA_STATUS_ERROR;
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_ERROR);
  EXPECT_EQ(hsa_references_, 2u);
  shutdown_status_ = HSA_STATUS_SUCCESS;
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 1u);
  submit_application(extended_packet());
  ASSERT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(core_.hsa_shut_down_fn(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 0u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanApplicationHsaPlugin, InitializationQueuedBehindFinalShutdownUsesRestoredCoreApi) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit_application(extended_packet());
  ASSERT_EQ(drain_(), 0);
  unload_during_shutdown_ = true;
  std::promise<void> resume;
  resume_shutdown_ = resume.get_future().share();
  std::future<void> shutdown_entered = shutdown_entered_.get_future();
  const decltype(CoreApiTable::hsa_init_fn) initialize = core_.hsa_init_fn;
  const decltype(CoreApiTable::hsa_shut_down_fn) shutdown = core_.hsa_shut_down_fn;
  ASSERT_NE(initialize, original_initialization_);
  std::future<hsa_status_t> shutting_down =
      std::async(std::launch::async, [&] { return shutdown(); });
  const std::future_status entered = shutdown_entered.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(entered, std::future_status::ready);
  if (entered != std::future_status::ready) {
    resume.set_value();
    (void)shutting_down.get();
    return;
  }
  std::promise<void> initialization_started;
  std::future<void> started = initialization_started.get_future();
  std::future<hsa_status_t> initializing = std::async(std::launch::async, [&] {
    initialization_started.set_value();
    return initialize();
  });
  EXPECT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_EQ(initializing.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  resume.set_value();
  EXPECT_EQ(shutting_down.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(initializing.get(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 1u);
  EXPECT_EQ(initialization_calls_, 1u);
  EXPECT_EQ(shutdown_calls_, 1u);
  EXPECT_FALSE(installed_);
  EXPECT_EQ(core_.hsa_init_fn, original_initialization_);
  resume_shutdown_ = {};
  EXPECT_EQ(shutdown(), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_references_, 0u);
  EXPECT_EQ(shutdown(), HSA_STATUS_ERROR_NOT_INITIALIZED);
}

TEST_F(ConSanApplicationHsaPlugin, RuntimeOutlivesEarlierRegisteredApplicationExitHandlers) {
  EXPECT_EXIT(
      {
        (void)finish_();
        unload_();
        installed_ = false;
        if (::dlclose(library_) != 0)
          std::_Exit(81);
        library_ = nullptr;
        // HIP registers its teardown before loading the HSA tool. Its late
        // executable destruction must still have a live hook owner at exit.
        if (std::atexit(late_application_teardown) != 0)
          std::_Exit(83);
        library_ = ::dlopen(CONSAN_PLUGIN_FIXTURE, RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr)
          std::_Exit(84);
        load_ = reinterpret_cast<bool (*)(HsaApiTable *, uint64_t, uint64_t, const char *const *)>(
            ::dlsym(library_, "OnLoad"));
        unload_ = reinterpret_cast<void (*)()>(::dlsym(library_, "OnUnload"));
        finish_ = reinterpret_cast<rj_consan_finish_fn>(::dlsym(library_, "rj_consan_finish"));
        initialize_api();
        if (!load_ || !unload_ || !finish_ || !load_(&table_, 0, 0, nullptr))
          std::_Exit(85);
        (void)finish_();
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "consan late teardown completed");
}

TEST_F(ConSanApplicationHsaPlugin, ShutdownRetainsItsRuntimeAndCodeThroughRocrToolClose) {
  EXPECT_EXIT(
      {
        (void)finish_();
        unload_during_shutdown_ = close_during_shutdown_ = true;
        if (core_.hsa_shut_down_fn() != HSA_STATUS_SUCCESS)
          std::_Exit(87);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

#if defined(CONSAN_APPLICATION_BARRIER_ARTIFACT) && defined(CONSAN_APPLICATION_LOOP_ARTIFACT)
class ConSanApplicationSelectedLdsHsaPlugin : public ConSanApplicationHsaPlugin,
                                              public ::testing::WithParamInterface<ScopeKind> {
protected:
  const char *original_fixture() const override {
    static const std::string barrier =
        (std::filesystem::path(CONSAN_ORIGINAL_FIXTURE).parent_path() /
         "targets/gfx1100/wave32/native-15.hsaco")
            .string();
    static const std::string loop = (std::filesystem::path(CONSAN_ORIGINAL_FIXTURE).parent_path() /
                                     "targets/gfx1100/wave32/native-27.hsaco")
                                        .string();
    switch (GetParam()) {
    case ScopeKind::NativeLdsWorkgroupBarrier:
      return barrier.c_str();
    case ScopeKind::NativeLdsWorkgroupBarrierLoop:
      return loop.c_str();
    default:
      return CONSAN_ORIGINAL_FIXTURE;
    }
  }

  const char *artifact_fixture() const override {
    switch (GetParam()) {
    case ScopeKind::NativeLdsWorkgroupBarrier:
      return CONSAN_APPLICATION_BARRIER_ARTIFACT;
    case ScopeKind::NativeLdsWorkgroupBarrierLoop:
      return CONSAN_APPLICATION_LOOP_ARTIFACT;
    default:
      return CONSAN_ARTIFACT_FIXTURE;
    }
  }
};

TEST_P(ConSanApplicationSelectedLdsHsaPlugin, SelectedLdsCannotUseTheNativeApplicationProfile) {
  ASSERT_EQ(facts_.scope, GetParam());
  ASSERT_EQ(image_facts_.entries.size(), 1u);
  ASSERT_TRUE(image_facts_.entries.front().selected.has_value());
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_kernel_dispatch_packet_t packet = dispatch_packet({}, 128);
  packet.group_segment_size = facts_.entry.original_group_bytes;
  packet.private_segment_size = facts_.entry.original_private_bytes;
  const LaunchDimensions dimensions = {.grid = {128, 1, 1}, .workgroup = {64, 1, 1}};
  const FailureOr<LaunchLayout> layout =
      plan_launch(facts_, dimensions, packet.group_segment_size, packet.private_segment_size);
  ASSERT_TRUE(layout.succeeded());
  ASSERT_EQ(layout.value().workgroup_count, 2u);
  ASSERT_EQ(layout.value().wave_count, 4u);
  ASSERT_NE(handler_, nullptr);
  handler_(&packet, 1, 0, handler_context_, application_writer);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(published_packets_, 0u);
  const rj_consan_statistics statistics = statistics_();
  EXPECT_EQ(statistics.submitted, 0u);
  EXPECT_EQ(statistics.native_submitted, 0u);
  EXPECT_EQ(statistics.native_completed, 0u);
  EXPECT_EQ(statistics.copied, 0u);
  EXPECT_EQ(statistics.written, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

INSTANTIATE_TEST_SUITE_P(SelectedLdsScopes, ConSanApplicationSelectedLdsHsaPlugin,
                         ::testing::Values(ScopeKind::NativeLdsNoSynchronization,
                                           ScopeKind::NativeLdsWorkgroupBarrier,
                                           ScopeKind::NativeLdsWorkgroupBarrierLoop),
                         [](const ::testing::TestParamInfo<ScopeKind> &info) {
                           switch (info.param) {
                           case ScopeKind::NativeLdsWorkgroupBarrier:
                             return "Barrier";
                           case ScopeKind::NativeLdsWorkgroupBarrierLoop:
                             return "BarrierLoop";
                           default:
                             return "NoSynchronization";
                           }
                         });
#endif

#if defined(CONSAN_NATIVE_SHAPE_ARTIFACT_FIXTURE)
class ConSanLegacyNativeShapeHsaPlugin : public ConSanHsaPlugin {
protected:
  const char *original_fixture() const override { return CONSAN_NATIVE_SHAPE_ORIGINAL_FIXTURE; }
  const char *artifact_fixture() const override { return CONSAN_NATIVE_SHAPE_ARTIFACT_FIXTURE; }

  void submit_native(uint16_t threads) {
    hsa_kernel_dispatch_packet_t packet = dispatch_packet(signal_, threads, nullptr, threads);
    packet.header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    handler_(&packet, 1, 0, handler_context_, packet_writer);
  }
};

TEST_F(ConSanLegacyNativeShapeHsaPlugin, RejectsWorkgroupDifferentFromOriginalRequiredShape) {
  ASSERT_FALSE(image_facts_.entries[0].selected.has_value());
  const std::array<uint32_t, 3> required = {64, 1, 1};
  ASSERT_EQ(image_facts_.entries[0].binding.required_workgroup_size, required);
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit_native(32);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanLegacyNativeShapeHsaPlugin, RetainsTheOriginalRequiredShapeAndNativePacket) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit_native(64);
  ASSERT_EQ(published_packets_, 1u);
  EXPECT_EQ(published_.workgroup_size_x, 64u);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_submitted, 1u);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
}
#endif

#if defined(CONSAN_ALIGNMENT_ORIGINAL_FIXTURE) &&                                                  \
    defined(CONSAN_SELECTED_ALIGNMENT_ARTIFACT_FIXTURE) &&                                         \
    defined(CONSAN_NATIVE_ALIGNMENT_ARTIFACT_FIXTURE)
class ConSanKernargAlignmentHsaPlugin : public ConSanHsaPlugin,
                                        public ::testing::WithParamInterface<bool> {
protected:
  const char *original_fixture() const override { return CONSAN_ALIGNMENT_ORIGINAL_FIXTURE; }
  const char *artifact_fixture() const override {
    return GetParam() ? CONSAN_SELECTED_ALIGNMENT_ARTIFACT_FIXTURE
                      : CONSAN_NATIVE_ALIGNMENT_ARTIFACT_FIXTURE;
  }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(ConSanHsaPlugin::SetUp());
    const ImageEntryFacts &entry = image_facts_.entries.front();
    ASSERT_EQ(entry.binding.name, "mk_a");
    ASSERT_EQ(entry.selected.has_value(), GetParam());
    ASSERT_EQ(entry.binding.kernarg_alignment, 32u);
    ASSERT_EQ(entry.binding.original_kernarg_bytes, 16u);
    ASSERT_EQ(argument_extent_, arguments_.size());
    hsa_amd_pointer_info_t info = {};
    info.size = sizeof(info);
    ASSERT_EQ(pointer_info(arguments_.data() + 16, &info, nullptr, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(info.hostBaseAddress, arguments_.data());
    ASSERT_EQ(info.sizeInBytes, 32u);
    ASSERT_LE(size_t{16} + entry.binding.original_kernarg_bytes, info.sizeInBytes);
    ASSERT_TRUE(bind());
    uint32_t alignment = 0;
    ASSERT_EQ(core_.hsa_executable_symbol_get_info_fn(
                  {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &alignment),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(alignment, 16u);
    ASSERT_TRUE(create_queue_and_signal());
  }

  hsa_kernel_dispatch_packet_t alignment_packet(const void *arguments) {
    hsa_kernel_dispatch_packet_t packet = dispatch_packet(signal_, 64, arguments, 64);
    packet.group_segment_size = image_facts_.entries.front().binding.original_group_bytes;
    packet.private_segment_size = image_facts_.entries.front().binding.original_private_bytes;
    packet.header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    return packet;
  }
};

TEST_P(ConSanKernargAlignmentHsaPlugin, RejectsHsaAlignedArgumentsBelowMetadataAlignment) {
  const uint8_t *arguments = arguments_.data() + 16;
  ASSERT_EQ(reinterpret_cast<uintptr_t>(arguments) % 16, 0u);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(arguments) % 32, 16u);
  const hsa_kernel_dispatch_packet_t packet = alignment_packet(arguments);
  handler_(&packet, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_P(ConSanKernargAlignmentHsaPlugin, AcceptsArgumentsMeetingMetadataAndHsaAlignment) {
  ASSERT_EQ(reinterpret_cast<uintptr_t>(arguments_.data()) % 32, 0u);
  const hsa_kernel_dispatch_packet_t packet = alignment_packet(arguments_.data());
  handler_(&packet, 1, 0, handler_context_, packet_writer);
  ASSERT_EQ(published_packets_, 1u);
  if (GetParam())
    EXPECT_NE(published_.completion_signal.handle, signal_.handle);
  else
    EXPECT_EQ(std::memcmp(&published_, &packet, sizeof(packet)), 0);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  ASSERT_EQ(finish_(), 0);
  const uint64_t selected = GetParam() ? 1 : 0;
  const uint64_t native = GetParam() ? 0 : 1;
  const rj_consan_statistics statistics = statistics_();
  EXPECT_EQ(statistics.submitted, selected);
  EXPECT_EQ(statistics.forwarded, selected);
  EXPECT_EQ(statistics.copied, selected);
  EXPECT_EQ(statistics.written, selected);
  EXPECT_EQ(statistics.native_submitted, native);
  EXPECT_EQ(statistics.native_completed, native);
  const FailureOr<Capture> captured = decode_capture(read_capture());
  ASSERT_TRUE(captured.succeeded());
  EXPECT_EQ(captured.value().executions.size(), selected);
}

INSTANTIATE_TEST_SUITE_P(SelectedAndNative, ConSanKernargAlignmentHsaPlugin,
                         ::testing::Values(true, false),
                         [](const ::testing::TestParamInfo<bool> &info) {
                           return info.param ? "Selected" : "Native";
                         });
#endif

#if defined(CONSAN_MULTI_ORIGINAL_FIXTURE) && defined(CONSAN_MULTI_ARTIFACT_FIXTURE)
class ConSanNativeAbiHsaPlugin : public ConSanApplicationHsaPlugin {
protected:
  const char *original_fixture() const override {
    static const std::string path =
        (std::filesystem::path(CONSAN_MULTI_ORIGINAL_FIXTURE).parent_path() /
         "multi-entry-abi.hsaco")
            .string();
    return path.c_str();
  }
  const char *artifact_fixture() const override { return CONSAN_MULTI_ARTIFACT_FIXTURE; }
  hsa_amd_ext_kernel_dispatch_packet_t private_packet() {
    hsa_amd_ext_kernel_dispatch_packet_t packet = extended_packet();
    packet.workgroup_size_x = 32;
    packet.kernel_object = 0x123600;
    packet.group_segment_size = 64;
    packet.private_segment_size = 68;
    return packet;
  }
};

TEST_F(ConSanNativeAbiHsaPlugin, PrivateSegmentsAndRequiredTwoDimensionalWorkgroupsRemainNative) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  const hsa_amd_ext_kernel_dispatch_packet_t packet = private_packet();
  submit_application(packet);
  ASSERT_EQ(application_packets_.size(), 1u);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanNativeAbiHsaPlugin, OriginalMetadataAlignmentStillAppliesWhenRocrReportsLess) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = private_packet();
  packet.kernarg_address = arguments_.data() + 16;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanNativeAbiHsaPlugin, EqualFlatSizeCannotReplaceTheRequiredWorkgroupShape) {
  configure_native_image();
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_amd_ext_kernel_dispatch_packet_t packet = private_packet();
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  submit_application(packet);
  EXPECT_TRUE(application_packets_.empty());
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_NE(finish_(), 0);
}

class ConSanMultiEntryHsaPlugin : public ConSanHsaPlugin {
protected:
  const char *original_fixture() const override { return CONSAN_MULTI_ORIGINAL_FIXTURE; }
  const char *artifact_fixture() const override { return CONSAN_MULTI_ARTIFACT_FIXTURE; }

  hsa_kernel_dispatch_packet_t entry_packet(size_t index, hsa_signal_t completion) {
    hsa_kernel_dispatch_packet_t packet = dispatch_packet(completion);
    packet.kernel_object = 0x123400 + index * 0x100;
    packet.group_segment_size = image_facts_.entries[index].binding.original_group_bytes;
    packet.header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    return packet;
  }
};

TEST_F(ConSanMultiEntryHsaPlugin, SelectedEntriesAndNativeSiblingPreserveBindingsAcrossRepeat) {
  ASSERT_EQ(image_facts_.entries.size(), 3u);
  ASSERT_TRUE(image_facts_.entries[0].selected.has_value());
  ASSERT_TRUE(image_facts_.entries[1].selected.has_value());
  ASSERT_FALSE(image_facts_.entries[2].selected.has_value());
  ASSERT_TRUE(bind());
  for (size_t index = 0; index < image_facts_.entries.size(); ++index) {
    const EntryFacts &binding = image_facts_.entries[index].binding;
    uint32_t size = 0, group = 0, alignment = 0, private_bytes = 1;
    const hsa_executable_symbol_t symbol = {10 + index};
    EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                  symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &size),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                  symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &alignment),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                  symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                  symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(size, binding.original_kernarg_bytes);
    EXPECT_EQ(group, binding.original_group_bytes);
    EXPECT_EQ(private_bytes, binding.original_private_bytes);
    EXPECT_EQ(alignment, 16u);
  }
  ASSERT_TRUE(create_queue_and_signal());
  const std::array<size_t, 4> order = {0, 1, 2, 0};
  for (size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
    const size_t index = order[ordinal];
    std::fill(arguments_.begin(), arguments_.end(), static_cast<uint8_t>(7 + ordinal));
    const hsa_kernel_dispatch_packet_t original = entry_packet(index, signal_);
    handler_(&original, 1, 0, handler_context_, packet_writer);
    ASSERT_EQ(drain_(), 0);
    EXPECT_EQ(load_signal(signal_), 0);
    if (index == 2) {
      EXPECT_EQ(std::memcmp(&published_, &original, sizeof(original)), 0);
    }
    if (ordinal + 1 < order.size()) {
      next_signal();
    }
  }
  EXPECT_EQ(statistics_().submitted, 3u);
  EXPECT_EQ(statistics_().forwarded, 3u);
  EXPECT_EQ(statistics_().copied, 3u);
  EXPECT_EQ(statistics_().written, 3u);
  EXPECT_EQ(statistics_().reused, 2u);
  EXPECT_EQ(statistics_().native_submitted, 1u);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(forwarding_calls_, 3u);
  EXPECT_EQ(published_packets_, 4u);
  ASSERT_EQ(finish_(), 0);
  const FailureOr<Capture> captured = decode_capture(read_capture());
  ASSERT_TRUE(captured.succeeded());
  ASSERT_EQ(captured.value().executions.size(), 3u);
  const std::array<size_t, 3> selected = {0, 1, 0};
  for (size_t index = 0; index < selected.size(); ++index) {
    const CapturedExecution &execution = captured.value().executions[index];
    EXPECT_EQ(execution.entry_key, image_facts_.entries[selected[index]].key);
    EXPECT_EQ(execution.arguments.size(), selected[index] == 1 ? 32u : 16u);
    EXPECT_EQ(execution.storage_generation, index + 1);
  }
}

TEST_F(ConSanMultiEntryHsaPlugin, CompletedNativeAllowsNextEntryBeforeObserverBookkeeping) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  std::promise<void> resume;
  resume_native_wait_ = resume.get_future().share();
  std::future<void> entered = native_wait_entered_.get_future();
  std::future<void> following = following_wait_entered_.get_future();
  delayed_native_signal_ = signal_.handle;
  const hsa_kernel_dispatch_packet_t native = entry_packet(2, signal_);
  handler_(&native, 1, 0, handler_context_, packet_writer);
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().native_completed, 0u);
  hsa_signal_t next = {};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &next), HSA_STATUS_SUCCESS);
  auto_complete_ = false;
  const hsa_kernel_dispatch_packet_t selected = entry_packet(0, next);
  handler_(&selected, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(published_packets_, 2u);
  EXPECT_EQ(load_signal(next), 1);
  EXPECT_EQ(statistics_().submitted, 1u);
  // Retain the native signal and all resources until the observer returns.
  EXPECT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_ERROR_INVALID_SIGNAL);
  resume.set_value();
  EXPECT_EQ(following.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  // Retiring the earlier native launch must not reopen admission or permit
  // destruction while the following selected dispatch is still on the GPU.
  EXPECT_EQ(core_.hsa_queue_destroy_fn(&queue_), HSA_STATUS_ERROR_INVALID_QUEUE);
  EXPECT_EQ(core_.hsa_executable_destroy_fn({7}), HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  store_signal(published_.completion_signal, 0);
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_EQ(load_signal(next), 0);
  EXPECT_EQ(finish_(), 0);
  EXPECT_TRUE(std::filesystem::exists(capture_));
}

TEST_F(ConSanMultiEntryHsaPlugin, PendingNativeCompletionKeepsResourcesAndBlocksOverlap) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  const hsa_kernel_dispatch_packet_t native = entry_packet(2, signal_);
  handler_(&native, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_EQ(statistics_().native_submitted, 1u);
  EXPECT_EQ(statistics_().native_completed, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_ERROR_INVALID_SIGNAL);
  EXPECT_EQ(core_.hsa_queue_destroy_fn(&queue_), HSA_STATUS_ERROR_INVALID_QUEUE);
  EXPECT_EQ(core_.hsa_executable_destroy_fn({7}), HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  hsa_signal_t next{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &next), HSA_STATUS_SUCCESS);
  const hsa_kernel_dispatch_packet_t selected = entry_packet(0, next);
  handler_(&selected, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_EQ(load_signal(next), -1);
  EXPECT_EQ(load_signal(signal_), 1);
  store_signal(signal_, 0);
  EXPECT_NE(drain_(), 0);
  EXPECT_EQ(statistics_().native_completed, 1u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanMultiEntryHsaPlugin, LastEntryQueryFailurePublishesNoEarlierBindingOrCapture) {
  (void)finish_();
  unload_();
  installed_ = false;
  core_.hsa_executable_symbol_get_info_fn =
      [](hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute, void *value) {
        if (symbol.handle == 12) {
          return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        return symbol_info(symbol, attribute, value);
      };
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  EXPECT_FALSE(bind());
  uint32_t kernarg = 0;
  EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(kernarg, image_facts_.entries[0].binding.original_kernarg_bytes);
  EXPECT_EQ(handler_, nullptr);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}
TEST_F(ConSanMultiEntryHsaPlugin, DuplicateLoadedKernelObjectsCannotPublishPartialBindings) {
  (void)finish_();
  unload_();
  installed_ = false;
  core_.hsa_executable_symbol_get_info_fn =
      [](hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute, void *value) {
        if (symbol.handle == 11 && attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
          *static_cast<uint64_t *>(value) = 0x123400;
          return HSA_STATUS_SUCCESS;
        }
        return symbol_info(symbol, attribute, value);
      };
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  EXPECT_FALSE(bind());
  EXPECT_EQ(handler_, nullptr);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}
#endif

class ConSanHsaPluginDeathTest : public ConSanHsaPlugin {
protected:
  void SetUp() override {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    ConSanHsaPlugin::SetUp();
  }

  void expect_interrupted_freeze(bool finish_during_query) {
    // Keep a regression in worker teardown inside a subprocess: the original
    // race destroys joinable threads and terminates during OnUnload.
    ASSERT_EXIT(
        {
          std::set_terminate([] { std::_Exit(86); });
          ::alarm(5);
          if (core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                                 &reader_) != HSA_STATUS_SUCCESS ||
              core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr) !=
                  HSA_STATUS_SUCCESS) {
            std::_Exit(81);
          }
          std::promise<void> resume;
          resume_symbol_query_ = resume.get_future().share();
          std::future<void> entered = symbol_query_entered_.get_future();
          std::future<hsa_status_t> frozen = std::async(
              std::launch::async, [&] { return core_.hsa_executable_freeze_fn({7}, nullptr); });
          if (entered.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            std::_Exit(82);
          }
          const bool interrupted = finish_during_query
                                       ? finish_() != 0
                                       : core_.hsa_executable_destroy_fn({7}) == HSA_STATUS_SUCCESS;
          resume.set_value();
          const hsa_status_t status = frozen.get();
          if (!finish_during_query) {
            (void)finish_();
          }
          const bool no_capture =
              std::filesystem::is_empty(directory_) && statistics_().submitted == 0;
          unload_();
          if (!interrupted || status != HSA_STATUS_ERROR_INVALID_ARGUMENT || !no_capture) {
            std::_Exit(83);
          }
          std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), "");
  }
};

TEST_F(ConSanHsaPluginDeathTest, FinishDuringFreezePreventsWorkerStartupAndUnloadsNormally) {
  expect_interrupted_freeze(true);
}

TEST_F(ConSanHsaPluginDeathTest, DestroyDuringFreezeInvalidatesTheAdmittedModule) {
  expect_interrupted_freeze(false);
}

TEST_F(ConSanHsaPlugin, LegacyProfileStillRejectsCoarseOriginalArguments) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  pointer_global_flags_ = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, RuntimeConfigurationRequiresPathsAndIndependentProfile) {
  (void)finish_();
  unload_();
  installed_ = false;
  for (const char *name :
       {"RJ_CONSAN_ARTIFACT_DIR", "RJ_CONSAN_CAPTURE_PATH", "RJ_CONSAN_SELECTION_PROFILE"}) {
    SCOPED_TRACE(name);
    const std::string saved = std::getenv(name);
    for (const char *value : {static_cast<const char *>(nullptr), ""}) {
      if (value) {
        ASSERT_EQ(::setenv(name, value, 1), 0);
      } else {
        ASSERT_EQ(::unsetenv(name), 0);
      }
      installed_ = load_(&table_, 0, 0, nullptr);
      ASSERT_FALSE(installed_);
      EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
    }
    ASSERT_EQ(::setenv(name, saved.c_str(), 1), 0);
  }
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  ASSERT_TRUE(bind());
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, MalformedExpectedProfileLeavesOriginalTablesUntouched) {
  (void)finish_();
  unload_();
  installed_ = false;
  const std::string saved = std::getenv("RJ_CONSAN_SELECTION_PROFILE");
  for (const std::string &invalid :
       {std::string("0"), std::string(63, '0'), std::string(65, '0'), std::string(64, 'z')}) {
    ASSERT_EQ(::setenv("RJ_CONSAN_SELECTION_PROFILE", invalid.c_str(), 1), 0);
    EXPECT_FALSE(load_(&table_, 0, 0, nullptr));
    EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
  }
  ASSERT_EQ(::setenv("RJ_CONSAN_SELECTION_PROFILE", saved.c_str(), 1), 0);
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
}

TEST_F(ConSanHsaPlugin, DifferentValidExpectedProfileRejectsBeforePreparedLoad) {
  (void)finish_();
  unload_();
  installed_ = false;
  const std::string wrong(64, '0');
  ASSERT_EQ(::setenv("RJ_CONSAN_SELECTION_PROFILE", wrong.c_str(), 1), 0);
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  EXPECT_FALSE(bind());
  EXPECT_TRUE(loaded_image_.empty());
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, UnknownKernelObjectCannotBorrowTheAdmittedEntryFacts) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_kernel_dispatch_packet_t packet = dispatch_packet(signal_);
  packet.kernel_object += 64;
  handler_(&packet, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(statistics_().native_submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, WrongLoadedResourceFailsFreezeWithoutPublishingBindingsOrCapture) {
  core_.hsa_executable_symbol_get_info_fn =
      [](hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute, void *value) {
        if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
          *static_cast<uint32_t *>(value) = 1024;
          return HSA_STATUS_SUCCESS;
        }
        return symbol_info(symbol, attribute, value);
      };
  // Replace the unhooked callback saved at install, not the active hook.
  (void)finish_();
  unload_();
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  EXPECT_FALSE(bind());
  EXPECT_EQ(handler_, nullptr);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, PartialUnderlyingLoadFailureRetainsPreparedStorageUntilExecutableDestroy) {
  load_status_ = HSA_STATUS_ERROR;
  check_partial_load_storage_ = true;
  ASSERT_EQ(core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                               &reader_),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr),
            HSA_STATUS_ERROR);
  EXPECT_EQ(load_calls_, 1u);
  EXPECT_EQ(readers_.size(), 1u);
  EXPECT_FALSE(partial_load_bytes_.empty());
  std::ifstream prepared(std::filesystem::path(artifact_fixture()) / "prepared.hsaco",
                         std::ios::binary);
  const std::vector<uint8_t> expected = {std::istreambuf_iterator<char>(prepared),
                                         std::istreambuf_iterator<char>()};
  ASSERT_FALSE(expected.empty());
  EXPECT_EQ(loaded_image_, expected);
  EXPECT_NE(loaded_image_, original_);
  load_status_ = HSA_STATUS_SUCCESS;
  EXPECT_EQ(core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(load_calls_, 1u);
  EXPECT_EQ(core_.hsa_executable_freeze_fn({7}, nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(freeze_calls_, 0u);
  EXPECT_FALSE(create_queue_and_signal());
  EXPECT_EQ(handler_, nullptr);
  uint32_t kernarg = 0;
  EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(kernarg, facts_.entry.original_kernarg_bytes);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
  EXPECT_FALSE(checked_partial_load_storage_);
  EXPECT_EQ(core_.hsa_executable_destroy_fn({7}), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(checked_partial_load_storage_);
  EXPECT_TRUE(partial_load_bytes_.empty());
}

TEST_F(ConSanHsaPlugin, UnderlyingFreezeFailureCannotBeRetriedIntoSuccessfulCapture) {
  freeze_status_ = HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  ASSERT_EQ(core_.hsa_code_object_reader_create_from_memory_fn(original_.data(), original_.size(),
                                                               &reader_),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(core_.hsa_executable_load_agent_code_object_fn({7}, {1}, reader_, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(core_.hsa_executable_freeze_fn({7}, nullptr), HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(freeze_calls_, 1u);
  freeze_status_ = HSA_STATUS_SUCCESS;
  EXPECT_EQ(core_.hsa_executable_freeze_fn({7}, nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(freeze_calls_, 1u);
  EXPECT_FALSE(create_queue_and_signal());
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, LaterFreezeFailureRevokesEarlierBindingAndPoisonsCapture) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  freeze_status_ = HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  EXPECT_EQ(core_.hsa_executable_freeze_fn({7}, nullptr), HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  uint32_t kernarg = 0;
  EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(kernarg, facts_.entry.original_kernarg_bytes);
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
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
  uint32_t alignment = 0;
  EXPECT_EQ(core_.hsa_executable_symbol_get_info_fn(
                {10}, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &alignment),
            HSA_STATUS_SUCCESS);
  // ROCR's code-object-v5 API alignment is 16 even when metadata alignment is 8.
  EXPECT_EQ(alignment, 16u);
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
  uint8_t *const first_report = report_;
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
  const FailureOr<Capture> captured = decode_capture(read_capture());
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
  const FailureOr<LaunchLayout> layout = plan_launch(facts_, {{128, 1, 1}, {128, 1, 1}}, 8, 0);
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
  submit(129);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, CompleteWorkgroupsPublishAllRowsAndCaptureTheirFullExtent) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  ASSERT_EQ(facts_.target, 1100u);
  const LaunchDimensions dimensions = {.grid = {128, 1, 1}, .workgroup = {64, 1, 1}};
  const FailureOr<LaunchLayout> layout = plan_launch(facts_, dimensions, 8, 0);
  ASSERT_TRUE(layout.succeeded());
  ASSERT_EQ(layout.value().workgroup_count, 2u);
  ASSERT_EQ(layout.value().waves_per_workgroup, 2u);
  ASSERT_EQ(layout.value().wave_count, 4u);
  auto_complete_ = false;
  submit(128);
  ASSERT_EQ(published_packets_, 1u);
  EXPECT_EQ(published_.grid_size_x, 128u);
  EXPECT_EQ(published_.workgroup_size_x, 64u);
  EXPECT_EQ(report_capacity(), layout.value().report_bytes);
  EXPECT_EQ(report_bytes_, layout.value().report_bytes);
  EXPECT_EQ(statistics_().submitted, 1u);
  EXPECT_EQ(statistics_().forwarded, 0u);
  EXPECT_EQ(statistics_().copied, 0u);
  store_signal(published_.completion_signal, 0);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  ASSERT_EQ(finish_(), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(statistics_().copied, 1u);
  EXPECT_EQ(statistics_().written, 1u);
  const FailureOr<Capture> captured = decode_capture(read_capture());
  ASSERT_TRUE(captured.succeeded());
  ASSERT_EQ(captured.value().executions.size(), 1u);
  const CapturedExecution &execution = captured.value().executions[0];
  EXPECT_EQ(execution.dimensions.grid, dimensions.grid);
  EXPECT_EQ(execution.dimensions.workgroup, dimensions.workgroup);
  ASSERT_EQ(execution.report.size(), layout.value().report_bytes);
  EXPECT_EQ(wire::read_u64(execution.context, wire::kContextReportBytes),
            layout.value().report_bytes);
  EXPECT_EQ(wire::read_u32(execution.report, wire::kReportRows), 4u);
  for (uint32_t row = 0; row < 4; ++row) {
    const size_t offset = kReportHeaderBytes + row * layout.value().row_stride;
    EXPECT_EQ(wire::read_u32(execution.report, offset + wire::kRowWave), row % 2);
    EXPECT_EQ(wire::read_u32(execution.report, offset + wire::kRowWorkgroupX), row / 2);
  }
}

TEST_F(ConSanHsaPlugin, AmdSignalsCompleteAdmittedDispatches) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal(true));
  submit();
  ASSERT_EQ(published_packets_, 1u);
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(core_.hsa_signal_destroy_fn(signal_), HSA_STATUS_SUCCESS);
  signal_ = {};
  ASSERT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &signal_), HSA_STATUS_SUCCESS);
  submit();
  ASSERT_EQ(drain_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().forwarded, 2u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, AmdSignalCreationPreservesArgumentsAndFailedOutputsRemainUntracked) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  const hsa_agent_t consumer{2};
  hsa_signal_t extra{};
  ASSERT_EQ(amd_.hsa_amd_signal_create_fn(7, 1, &consumer, HSA_AMD_SIGNAL_IPC, &extra),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(amd_signal_value_, 7);
  EXPECT_EQ(amd_signal_consumers_, 1u);
  EXPECT_EQ(amd_signal_agents_, &consumer);
  EXPECT_EQ(amd_signal_attributes_, uint64_t{HSA_AMD_SIGNAL_IPC});
  ASSERT_EQ(core_.hsa_signal_destroy_fn(extra), HSA_STATUS_SUCCESS);

  // A failing create may leave the output untouched. It must not register a
  // pre-existing handle obtained outside either intercepted creation API.
  ASSERT_EQ(create_signal(1, 0, nullptr, &extra), HSA_STATUS_SUCCESS);
  amd_signal_status_ = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  EXPECT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &extra), amd_signal_status_);
  submit(129);
  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = extra;
  handler_(&barrier, 1, 0, handler_context_, barrier_writer);
  EXPECT_EQ(load_signal(extra), 1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedAmdDispatchNotifiesItsCompletion) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal(true));
  submit(129);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedBatchesNotifyAmdDispatchAndBarrierCompletions) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal(true));
  for (const uint16_t type : {HSA_PACKET_TYPE_BARRIER_AND, HSA_PACKET_TYPE_BARRIER_OR}) {
    hsa_signal_t second{}, completed{};
    ASSERT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &second), HSA_STATUS_SUCCESS);
    ASSERT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &completed), HSA_STATUS_SUCCESS);
    hsa_barrier_and_packet_t barrier{};
    barrier.header = type;
    barrier.dep_signal[0] = second;
    barrier.completion_signal = completed;
    std::array packets{dispatch_packet(signal_), hsa_kernel_dispatch_packet_t{},
                       dispatch_packet(second)};
    std::memcpy(&packets[1], &barrier, sizeof(barrier));
    handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
    EXPECT_EQ(load_signal(signal_), -1);
    EXPECT_EQ(load_signal(second), -1);
    EXPECT_EQ(load_signal(completed), -1);
  }
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, AmdBarriersAfterADispatchRejectionReceiveFailureNotifications) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal(true));
  submit(129);
  for (const bool batch : {false, true}) {
    std::array<hsa_barrier_and_packet_t, 2> packets{};
    packets[0].header = HSA_PACKET_TYPE_BARRIER_AND;
    packets[1].header = HSA_PACKET_TYPE_BARRIER_OR;
    for (hsa_barrier_and_packet_t &packet : packets) {
      ASSERT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &packet.completion_signal),
                HSA_STATUS_SUCCESS);
      packet.dep_signal[0] = signal_;
    }
    if (batch) {
      handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
    } else {
      for (const hsa_barrier_and_packet_t &packet : packets) {
        handler_(&packet, 1, 0, handler_context_, completing_barrier_writer);
      }
    }
    for (const hsa_barrier_and_packet_t &packet : packets) {
      EXPECT_EQ(load_signal(packet.completion_signal), -1);
    }
  }
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedAmdBarrierCannotOverwritePendingDispatchCompletion) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal(true));
  auto_complete_ = false;
  submit();
  ASSERT_EQ(published_packets_, 1u);
  const hsa_signal_t proxy = published_.completion_signal;
  hsa_signal_t rejected{};
  ASSERT_EQ(amd_.hsa_amd_signal_create_fn(1, 0, nullptr, 0, &rejected), HSA_STATUS_SUCCESS);
  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = signal_;
  std::array packets{dispatch_packet(rejected), hsa_kernel_dispatch_packet_t{}};
  std::memcpy(&packets[1], &barrier, sizeof(barrier));
  handler_(packets.data(), packets.size(), 0, handler_context_, barrier_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(load_signal(rejected), -1);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_TRUE(published_barriers_.empty());
  store_signal(proxy, 0);
  EXPECT_NE(finish_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
}

TEST_F(ConSanHsaPlugin, UntrackedBarriersAfterARejectionRetainTheirSignalsWithoutPublication) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit(129);
  for (const uint16_t type : {HSA_PACKET_TYPE_BARRIER_AND, HSA_PACKET_TYPE_BARRIER_OR}) {
    hsa_barrier_and_packet_t barrier{};
    barrier.header = type;
    ASSERT_EQ(create_signal(1, 0, nullptr, &barrier.completion_signal), HSA_STATUS_SUCCESS);
    handler_(&barrier, 1, 0, handler_context_, completing_barrier_writer);
    EXPECT_EQ(core_.hsa_signal_wait_scacquire_fn(barrier.completion_signal, HSA_SIGNAL_CONDITION_LT,
                                                 1, 1000000, HSA_WAIT_STATE_BLOCKED),
              1);
  }
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, MissingAmdSignalCreationKeepsCoreSignalSupportAvailable) {
  (void)finish_();
  unload_();
  installed_ = false;
  initialize_api();
  amd_.hsa_amd_signal_create_fn = nullptr;
  installed_ = load_(&table_, 0, 0, nullptr);
  ASSERT_TRUE(installed_);
  EXPECT_EQ(amd_.hsa_amd_signal_create_fn, nullptr);
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit();
  EXPECT_EQ(drain_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedBatchNotifiesEveryDiscardedDispatch) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t second{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &second), HSA_STATUS_SUCCESS);
  const std::array packets{dispatch_packet(signal_), dispatch_packet(second)};
  handler_(packets.data(), packets.size(), 0, handler_context_, packet_writer);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(load_signal(second), -1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, RejectedBatchPreservesSignalsForAlreadyPendingWork) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit();
  ASSERT_EQ(published_packets_, 1u);
  hsa_signal_t second{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &second), HSA_STATUS_SUCCESS);
  const std::array packets{dispatch_packet(signal_), dispatch_packet(second)};
  handler_(packets.data(), packets.size(), 0, handler_context_, packet_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(load_signal(second), -1);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_EQ(statistics_().submitted, 1u);
  EXPECT_EQ(statistics_().forwarded, 0u);
  store_signal(published_.completion_signal, 0);
  EXPECT_NE(finish_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(forwarding_calls_, 1u);
}

TEST_F(ConSanHsaPlugin, RejectedBatchNotifiesBarrierWaitersOfFailure) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t second{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &second), HSA_STATUS_SUCCESS);
  for (const uint16_t type : {HSA_PACKET_TYPE_BARRIER_AND, HSA_PACKET_TYPE_BARRIER_OR}) {
    hsa_signal_t completed{};
    ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &completed), HSA_STATUS_SUCCESS);
    hsa_barrier_and_packet_t barrier{};
    barrier.header = type;
    barrier.dep_signal[0] = second;
    barrier.completion_signal = completed;
    std::array packets{dispatch_packet(signal_), hsa_kernel_dispatch_packet_t{},
                       dispatch_packet(second)};
    std::memcpy(&packets[1], &barrier, sizeof(barrier));
    handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
    std::unique_lock lock(fake_mutex_);
    EXPECT_TRUE(fake_condition_.wait_for(lock, std::chrono::milliseconds(100),
                                         [&] { return signal_values_.at(completed.handle) < 1; }));
    EXPECT_EQ(signal_values_.at(completed.handle), -1);
  }
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_EQ(load_signal(second), -1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, BarrierOnlyBatchPreservesOrderAndDoesNotFailCapture) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t second{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &second), HSA_STATUS_SUCCESS);
  std::array<hsa_barrier_and_packet_t, 2> packets{};
  packets[0].header = HSA_PACKET_TYPE_BARRIER_AND;
  packets[0].completion_signal = signal_;
  packets[1].header = HSA_PACKET_TYPE_BARRIER_OR;
  packets[1].dep_signal[0] = signal_;
  packets[1].completion_signal = second;
  handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
  ASSERT_EQ(published_barriers_.size(), packets.size());
  EXPECT_EQ(std::memcmp(published_barriers_.data(), packets.data(), sizeof(packets)), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(load_signal(second), 0);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedBatchPreservesPendingWorkWhenNotifyingBarrierFailure) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit();
  ASSERT_EQ(published_packets_, 1u);
  const hsa_signal_t proxy = published_.completion_signal;
  hsa_signal_t completed{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &completed), HSA_STATUS_SUCCESS);
  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.dep_signal[0] = signal_;
  barrier.completion_signal = completed;
  std::array packets{dispatch_packet(signal_), hsa_kernel_dispatch_packet_t{}};
  std::memcpy(&packets[1], &barrier, sizeof(barrier));
  handler_(packets.data(), packets.size(), 0, handler_context_, barrier_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(load_signal(completed), -1);
  EXPECT_TRUE(published_barriers_.empty());
  EXPECT_EQ(published_packets_, 1u);
  store_signal(proxy, 0);
  EXPECT_NE(finish_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(statistics_().forwarded, 1u);
  EXPECT_EQ(forwarding_calls_, 1u);
}

TEST_F(ConSanHsaPlugin, RejectedNonDispatchPacketDoesNotModifyATrackedSignal) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_AGENT_DISPATCH;
  packet.completion_signal = signal_;
  handler_(&packet, 1, 0, handler_context_, packet_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, RejectedBarrierCannotOverwritePendingDispatchCompletion) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  auto_complete_ = false;
  submit();
  const hsa_signal_t proxy = published_.completion_signal;
  hsa_signal_t rejected{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &rejected), HSA_STATUS_SUCCESS);
  hsa_barrier_and_packet_t barrier{};
  barrier.header = HSA_PACKET_TYPE_BARRIER_AND;
  barrier.completion_signal = signal_;
  std::array packets{dispatch_packet(rejected), hsa_kernel_dispatch_packet_t{}};
  std::memcpy(&packets[1], &barrier, sizeof(barrier));
  handler_(packets.data(), packets.size(), 0, handler_context_, barrier_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(load_signal(rejected), -1);
  EXPECT_EQ(published_packets_, 1u);
  EXPECT_TRUE(published_barriers_.empty());
  store_signal(proxy, 0);
  EXPECT_NE(finish_(), 0);
  EXPECT_EQ(load_signal(signal_), 0);
  EXPECT_EQ(forwarding_calls_, 1u);
}

TEST_F(ConSanHsaPlugin, RejectedUnsupportedPacketNotifiesItsNeighboringBarrierOfFailure) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_signal_t completed{};
  ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &completed), HSA_STATUS_SUCCESS);
  hsa_agent_dispatch_packet_t unsupported{};
  unsupported.header = HSA_PACKET_TYPE_AGENT_DISPATCH;
  unsupported.completion_signal = signal_;
  std::array<hsa_barrier_and_packet_t, 2> packets{};
  std::memcpy(&packets[0], &unsupported, sizeof(unsupported));
  packets[1].header = HSA_PACKET_TYPE_BARRIER_AND;
  packets[1].completion_signal = completed;
  handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(load_signal(completed), -1);
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, BarrierBatchWithoutCompletionSignalsStillPassesThrough) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  std::array<hsa_barrier_and_packet_t, 2> packets{};
  packets[0].header = HSA_PACKET_TYPE_BARRIER_AND;
  packets[1].header = HSA_PACKET_TYPE_BARRIER_OR;
  handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
  ASSERT_EQ(published_barriers_.size(), packets.size());
  EXPECT_EQ(std::memcmp(published_barriers_.data(), packets.data(), sizeof(packets)), 0);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, IndividualBarriersStillPassThroughWithoutDispatchAdmission) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  hsa_barrier_and_packet_t packet{};
  packet.completion_signal = signal_;
  for (const uint16_t type : {HSA_PACKET_TYPE_BARRIER_AND, HSA_PACKET_TYPE_BARRIER_OR}) {
    packet.header = type;
    handler_(&packet, 1, 0, handler_context_, barrier_writer);
    EXPECT_EQ(published_.header, type);
    EXPECT_EQ(published_.completion_signal.handle, signal_.handle);
  }
  EXPECT_EQ(published_packets_, 2u);
  EXPECT_EQ(load_signal(signal_), 1);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, BarriersAfterADispatchRejectionReceiveFailureNotifications) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  submit(129);
  ASSERT_EQ(load_signal(signal_), -1);
  for (const bool batch : {false, true}) {
    std::array<hsa_barrier_and_packet_t, 2> packets{};
    packets[0].header = HSA_PACKET_TYPE_BARRIER_AND;
    packets[1].header = HSA_PACKET_TYPE_BARRIER_OR;
    for (hsa_barrier_and_packet_t &packet : packets) {
      ASSERT_EQ(core_.hsa_signal_create_fn(1, 0, nullptr, &packet.completion_signal),
                HSA_STATUS_SUCCESS);
      packet.dep_signal[0] = signal_;
    }
    if (batch) {
      handler_(packets.data(), packets.size(), 0, handler_context_, completing_barrier_writer);
    } else {
      for (const hsa_barrier_and_packet_t &packet : packets) {
        handler_(&packet, 1, 0, handler_context_, completing_barrier_writer);
      }
    }
    for (const hsa_barrier_and_packet_t &packet : packets) {
      EXPECT_EQ(load_signal(packet.completion_signal), -1);
    }
  }
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(statistics_().submitted, 0u);
  EXPECT_NE(finish_(), 0);
}

TEST_F(ConSanHsaPlugin, ShortOriginalArgumentAllocationFailsBeforePublication) {
  ASSERT_TRUE(bind());
  ASSERT_TRUE(create_queue_and_signal());
  argument_extent_ = 12;
  submit();
  EXPECT_EQ(published_packets_, 0u);
  EXPECT_EQ(allocation_calls_, 0u);
  EXPECT_EQ(load_signal(signal_), -1);
  EXPECT_NE(finish_(), 0);
  EXPECT_FALSE(std::filesystem::exists(capture_));
}

TEST_F(ConSanHsaPlugin, ConcurrentFinishClosesAndPublishesOneCompleteCapture) {
  ASSERT_TRUE(bind());
  std::promise<void> release;
  std::shared_future<void> ready = release.get_future().share();
  std::future<int> first = std::async(std::launch::async, [&] {
    ready.wait();
    return finish_();
  });
  std::future<int> second = std::async(std::launch::async, [&] {
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
    hsa_signal_t completion{};
    EXPECT_EQ(prefix->hsa_amd_signal_create_fn(1, 0, nullptr, 0, &completion), HSA_STATUS_SUCCESS);
    EXPECT_EQ(core_.hsa_signal_destroy_fn(completion), HSA_STATUS_SUCCESS);
    EXPECT_EQ(finish_(), 0);
    unload_();
    installed_ = false;
    EXPECT_EQ(core_.hsa_code_object_reader_create_from_memory_fn, create_reader);
    EXPECT_EQ(prefix->hsa_amd_profiling_set_profiler_enabled_fn,
              amd_.hsa_amd_profiling_set_profiler_enabled_fn);
    EXPECT_EQ(prefix->hsa_amd_queue_intercept_register_fn,
              amd_.hsa_amd_queue_intercept_register_fn);
    EXPECT_EQ(prefix->hsa_amd_signal_create_fn, create_amd_signal);
  }
  table_.amd_ext_ = &amd_;
}

TEST_F(ConSanHsaPlugin, UnloadRestoresOurHooksWithoutOverwritingLaterCoreOrExtensionCallbacks) {
  ASSERT_TRUE(bind());
  ASSERT_EQ(finish_(), 0);
  const auto later_reader =
      +[](hsa_file_t, hsa_code_object_reader_t *) { return HSA_STATUS_ERROR; };
  const auto later_profiling = +[](hsa_queue_t *, int) { return HSA_STATUS_ERROR; };
  const auto later_signal = +[](hsa_signal_value_t, uint32_t, const hsa_agent_t *, uint64_t,
                                hsa_signal_t *) { return HSA_STATUS_ERROR; };
  core_.hsa_code_object_reader_create_from_file_fn = later_reader;
  amd_.hsa_amd_profiling_set_profiler_enabled_fn = later_profiling;
  amd_.hsa_amd_signal_create_fn = later_signal;
  unload_();
  installed_ = false;
  EXPECT_EQ(core_.hsa_code_object_reader_create_from_file_fn, later_reader);
  EXPECT_EQ(amd_.hsa_amd_profiling_set_profiler_enabled_fn, later_profiling);
  EXPECT_EQ(amd_.hsa_amd_signal_create_fn, later_signal);
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
