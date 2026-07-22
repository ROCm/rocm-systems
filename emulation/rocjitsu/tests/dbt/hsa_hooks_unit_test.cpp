// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_replay_provenance.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "waitcheck_fixture.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

extern "C" bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char *const *failed_tool_names);
extern "C" void OnUnload();
using ConSanTransformOverride = rocjitsu::ConSanResult (*)(std::span<const uint8_t>,
                                                           const rocjitsu::ConSanOptions &);

namespace {

using ExpectedQueueInterceptPacketWriter = void (*)(const void *, uint64_t);
using ExpectedQueueInterceptHandler = void (*)(const void *, uint64_t, uint64_t, void *,
                                               ExpectedQueueInterceptPacketWriter);
using ExpectedQueueInterceptCreate = hsa_status_t(HSA_API *)(
    hsa_agent_t, uint32_t, hsa_queue_type32_t, void (*)(hsa_status_t, hsa_queue_t *, void *),
    void *, uint32_t, uint32_t, hsa_queue_t **);
using ExpectedQueueInterceptRegister = hsa_status_t(HSA_API *)(hsa_queue_t *,
                                                               ExpectedQueueInterceptHandler,
                                                               void *);

static_assert(
    std::is_same_v<hsa_amd_queue_intercept_packet_writer_t, ExpectedQueueInterceptPacketWriter>);
static_assert(std::is_same_v<hsa_amd_queue_intercept_handler_t, ExpectedQueueInterceptHandler>);
static_assert(std::is_same_v<hsa_amd_queue_intercept_create_fn_t, ExpectedQueueInterceptCreate>);
static_assert(
    std::is_same_v<hsa_amd_queue_intercept_register_fn_t, ExpectedQueueInterceptRegister>);
static_assert(std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn),
                             ExpectedQueueInterceptCreate>);
static_assert(std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn),
                             ExpectedQueueInterceptRegister>);

TEST(HsaHooksUnitTest, QueueInterceptionEntriesUsePublicAbiSignatures) {
  EXPECT_TRUE((std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_create_fn),
                              ExpectedQueueInterceptCreate>));
  EXPECT_TRUE((std::is_same_v<decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn),
                              ExpectedQueueInterceptRegister>));
}

constexpr hsa_agent_t kGuestAgent{1};
constexpr hsa_agent_t kHostAgent{2};
// An agent that is neither the guest nor the guest's execution host. Its queues
// are tracked for doorbell forwarding but never rewritten, and host_lds_bytes
// (derived from the guest target arch) does not apply to them.
constexpr hsa_agent_t kUnrelatedAgent{3};
constexpr hsa_isa_t kGuestIsa{950};
constexpr hsa_isa_t kHostIsa{1201};
constexpr hsa_amd_memory_pool_t kGuestPool{10};
constexpr hsa_amd_memory_pool_t kHostPool{20};
constexpr hsa_amd_memory_pool_t kHostKernargPool{21};
constexpr uint32_t kGuestNodeId = 100;
constexpr uint32_t kHostNodeId = 200;
constexpr uint32_t kVirtualLdsWrapperStateOffsetForTest = 8;
constexpr uint32_t kVirtualLdsWrapperSizeForTest = 32;
constexpr uint16_t kVirtualLdsWrapperFlagsForTest =
    rocjitsu::kVirtualLdsFlagRuntimeStateBlock | rocjitsu::kVirtualLdsFlagWorkgroupIdX;

std::mutex g_pool_mutex;
std::condition_variable g_pool_cv;
bool g_block_guest_pool_iteration = false;
bool g_guest_pool_iteration_entered = false;
bool g_release_guest_pool_iteration = false;
bool g_fail_guest_pool_iteration_once = false;
std::mutex g_agent_mutex;
std::condition_variable g_agent_cv;
bool g_block_agent_iteration = false;
bool g_agent_iteration_entered = false;
bool g_release_agent_iteration = false;
int g_fake_shutdown_calls = 0;
hsa_amd_memory_pool_t g_last_allocate_pool{};
hsa_agent_t g_last_agent_memory_pool_agent{};
hsa_amd_memory_pool_t g_last_agent_memory_pool{};
int g_agent_memory_pool_get_info_calls = 0;
int g_fake_allocation_storage = 0;
hsa_agent_t g_pointer_info_accessible[2] = {};
std::vector<uint64_t> g_last_batch_src_agents;
std::vector<uint64_t> g_last_batch_dst_agents;
std::vector<uint64_t> g_last_memory_lock_agents;
std::vector<uint64_t> g_last_memory_lock_to_pool_agents;
std::vector<uint64_t> g_last_vmem_access_agents;
hsa_amd_memory_pool_t g_last_memory_lock_to_pool_pool{};
int g_code_object_reader_create_calls = 0;
bool g_fail_replacement_reader_create = false;
bool g_fail_core_memory_allocate = false;
int g_core_memory_allocate_calls = 0;
int g_core_memory_free_calls = 0;
std::vector<size_t> g_core_memory_allocation_sizes;
std::vector<void *> g_core_memory_allocations;
std::vector<rocjitsu::ConSanMoiReportHeader> g_core_memory_headers_at_free;
std::vector<uint32_t> g_sc_markers_at_free;
std::vector<std::vector<uint8_t>> g_code_object_reader_inputs;
std::vector<uint64_t> g_destroyed_code_object_readers;
std::vector<uint64_t> g_loaded_code_object_readers;
rocjitsu::ConSanResult g_transform_override_result;
std::vector<rocjitsu::ConSanFlavor> g_transform_override_flavors;
std::vector<rocjitsu::ConSanMoiEngine> g_transform_override_engines;
std::vector<bool> g_transform_override_abort_unmatched_waits;
std::vector<bool> g_transform_override_track_barriers;
std::vector<bool> g_transform_override_track_atomics;
std::vector<uint32_t> g_transform_override_runtime_sample_strides;
std::vector<uint64_t> g_transform_override_report_sizes;
std::vector<std::optional<uint64_t>> g_transform_override_sc_report_addresses;
std::vector<std::optional<rocjitsu::ConSanMoiReportLayoutOverride>>
    g_transform_override_report_layouts;
std::vector<std::vector<uint8_t>> g_fake_allocations;
std::vector<hsa_amd_memory_pool_t> g_fake_allocation_pools;
std::vector<size_t> g_fake_allocation_sizes;
std::vector<void *> g_fake_freed_allocations;
std::array<hsa_kernel_dispatch_packet_t, 4> g_fake_queue_packets{};
hsa_queue_t g_fake_queue{};
hsa_agent_t g_last_queue_create_agent{};
hsa_queue_t *g_last_destroyed_queue = nullptr;
int g_fake_signal_store_relaxed_calls = 0;
int g_fake_signal_store_screlease_calls = 0;
hsa_signal_t g_last_signal_store_signal{};
hsa_signal_value_t g_last_signal_store_value = 0;
uint64_t g_next_fake_signal_handle = 10000;
std::vector<hsa_signal_t> g_fake_created_signals;
std::vector<hsa_signal_t> g_fake_destroyed_signals;
struct FakeSignalValue {
  uint64_t handle = 0;
  hsa_signal_value_t value = 0;
};
std::vector<FakeSignalValue> g_fake_signal_values;
hsa_queue_t *g_last_intercept_registered_queue = nullptr;
hsa_amd_queue_intercept_handler_t g_fake_intercept_handler = nullptr;
void *g_fake_intercept_user_data = nullptr;
std::vector<hsa_kernel_dispatch_packet_t> g_last_intercept_written_packets;
uint64_t g_fake_symbol_kernel_object = 0;
uint32_t g_fake_symbol_group_segment_size = 0;
uint32_t g_fake_symbol_private_segment_size = 0;
std::string g_fake_symbol_name = "oversized_kernel.kd";
// Records the arguments the hook forwards to the original agent-code-object
// loader, so a test can assert a non-guest load reaches the loader unchanged.
int g_fake_load_agent_calls = 0;
hsa_agent_t g_last_load_agent{};
hsa_code_object_reader_t g_last_load_reader{};
constexpr hsa_executable_t kFakeExecutable{123};
constexpr hsa_executable_symbol_t kFakeKernelSymbol{500};

const char *isa_name(hsa_isa_t isa) {
  if (isa.handle == kGuestIsa.handle)
    return "amdgcn-amd-amdhsa--gfx950";
  if (isa.handle == kHostIsa.handle)
    return "amdgcn-amd-amdhsa--gfx1201";
  return "";
}

hsa_status_t HSA_API fake_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void *),
                                         void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  {
    std::unique_lock lock(g_agent_mutex);
    if (g_block_agent_iteration) {
      g_agent_iteration_entered = true;
      g_agent_cv.notify_all();
      g_agent_cv.wait(lock, [] { return g_release_agent_iteration; });
    }
  }

  hsa_status_t status = callback(kGuestAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kHostAgent, data);
}

hsa_status_t HSA_API fake_iterate_agents_host_first(hsa_status_t (*callback)(hsa_agent_t, void *),
                                                    void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = callback(kHostAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kGuestAgent, data);
}

hsa_status_t HSA_API fake_shut_down() {
  ++g_fake_shutdown_calls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute,
                                         void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_AGENT_INFO_DEVICE) {
    *static_cast<hsa_device_type_t *>(value) = HSA_DEVICE_TYPE_GPU;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AGENT_INFO_ISA) {
    *static_cast<hsa_isa_t *>(value) = agent.handle == kGuestAgent.handle ? kGuestIsa : kHostIsa;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID)) {
    *static_cast<uint32_t *>(value) =
        agent.handle == kGuestAgent.handle ? kGuestNodeId : kHostNodeId;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_agent_iterate_isas(hsa_agent_t agent,
                                             hsa_status_t (*callback)(hsa_isa_t, void *),
                                             void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent.handle == kGuestAgent.handle)
    return callback(kGuestIsa, data);
  if (agent.handle == kHostAgent.handle)
    return callback(kHostIsa, data);
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t HSA_API fake_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  const char *name = isa_name(isa);
  if (name[0] == '\0')
    return HSA_STATUS_ERROR_INVALID_ISA;

  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) = static_cast<uint32_t>(std::strlen(name));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::strcpy(static_cast<char *>(value), name);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                       void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
                                       uint32_t, uint32_t, hsa_queue_t **queue) {
  if (queue == nullptr || size == 0 || size > g_fake_queue_packets.size())
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  g_last_queue_create_agent = agent;
  g_fake_queue_packets = {};
  g_fake_queue = {};
  g_fake_queue.type = type;
  g_fake_queue.features = HSA_QUEUE_FEATURE_KERNEL_DISPATCH;
  g_fake_queue.base_address = g_fake_queue_packets.data();
  g_fake_queue.doorbell_signal = hsa_signal_t{77};
  g_fake_queue.size = size;
  g_fake_queue.id = 1234;
  *queue = &g_fake_queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_queue_destroy(hsa_queue_t *queue) {
  g_last_destroyed_queue = queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_queue_intercept_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  return fake_queue_create(agent, size, type, callback, data, private_segment_size,
                           group_segment_size, queue);
}

hsa_status_t HSA_API fake_amd_queue_intercept_register(hsa_queue_t *queue,
                                                       hsa_amd_queue_intercept_handler_t callback,
                                                       void *user_data) {
  g_last_intercept_registered_queue = queue;
  g_fake_intercept_handler = callback;
  g_fake_intercept_user_data = user_data;
  return HSA_STATUS_SUCCESS;
}

void fake_intercept_packet_writer(const void *pkts, uint64_t pkt_count) {
  g_last_intercept_written_packets.clear();
  if (pkts == nullptr || pkt_count == 0)
    return;

  const auto *packets = static_cast<const hsa_kernel_dispatch_packet_t *>(pkts);
  g_last_intercept_written_packets.assign(packets, packets + pkt_count);
}

void HSA_API fake_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  ++g_fake_signal_store_relaxed_calls;
  g_last_signal_store_signal = signal;
  g_last_signal_store_value = value;
}

void HSA_API fake_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
  ++g_fake_signal_store_screlease_calls;
  g_last_signal_store_signal = signal;
  g_last_signal_store_value = value;
}

void set_fake_signal_value(hsa_signal_t signal, hsa_signal_value_t value) {
  for (FakeSignalValue &entry : g_fake_signal_values) {
    if (entry.handle == signal.handle) {
      entry.value = value;
      return;
    }
  }
  g_fake_signal_values.push_back(FakeSignalValue{.handle = signal.handle, .value = value});
}

hsa_status_t HSA_API fake_signal_create(hsa_signal_value_t initial_value, uint32_t,
                                        const hsa_agent_t *, hsa_signal_t *signal) {
  if (signal == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *signal = hsa_signal_t{g_next_fake_signal_handle++};
  g_fake_created_signals.push_back(*signal);
  set_fake_signal_value(*signal, initial_value);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_signal_destroy(hsa_signal_t signal) {
  g_fake_destroyed_signals.push_back(signal);
  return HSA_STATUS_SUCCESS;
}

hsa_signal_value_t HSA_API fake_signal_load_scacquire(hsa_signal_t signal) {
  for (const FakeSignalValue &entry : g_fake_signal_values) {
    if (entry.handle == signal.handle)
      return entry.value;
  }
  return 1;
}

hsa_status_t HSA_API
fake_code_object_reader_create_from_file(hsa_file_t, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_create_from_memory(
    const void *bytes, size_t size, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  ++g_code_object_reader_create_calls;
  if (g_code_object_reader_create_calls > 1 && g_fail_replacement_reader_create)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  const auto *begin = static_cast<const uint8_t *>(bytes);
  g_code_object_reader_inputs.emplace_back(begin, begin == nullptr ? begin : begin + size);
  code_object_reader->handle = 100u + static_cast<uint64_t>(g_code_object_reader_create_calls);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_destroy(hsa_code_object_reader_t reader) {
  g_destroyed_code_object_readers.push_back(reader.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_system_get_extension_table(uint16_t, uint16_t, uint16_t, void *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_system_get_major_extension_table(uint16_t, uint16_t, size_t, void *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_queue_intercept_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
                                                 void (*)(hsa_status_t, hsa_queue_t *, void *),
                                                 void *, uint32_t, uint32_t, hsa_queue_t **) {
  return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
}

hsa_status_t HSA_API fake_queue_intercept_register(hsa_queue_t *, hsa_amd_queue_intercept_handler_t,
                                                   void *) {
  return HSA_STATUS_ERROR_INVALID_QUEUE;
}

hsa_status_t HSA_API fake_agent_iterate_regions(hsa_agent_t agent,
                                                hsa_status_t (*callback)(hsa_region_t, void *),
                                                void *data) {
  if (agent.handle != kHostAgent.handle || callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  return callback(hsa_region_t{30}, data);
}

hsa_status_t HSA_API fake_region_get_info(hsa_region_t region, hsa_region_info_t attribute,
                                          void *value) {
  if (region.handle != 30 || value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
  case HSA_REGION_INFO_SEGMENT:
    *static_cast<hsa_region_segment_t *>(value) = HSA_REGION_SEGMENT_GLOBAL;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:
    *static_cast<bool *>(value) = true;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_ALLOC_MAX_SIZE:
    *static_cast<size_t *>(value) = rocjitsu::kConSanMoiAutoReportProcessCeilingBytes;
    return HSA_STATUS_SUCCESS;
  case HSA_REGION_INFO_GLOBAL_FLAGS:
    *static_cast<uint32_t *>(value) = HSA_REGION_GLOBAL_FLAG_FINE_GRAINED;
    return HSA_STATUS_SUCCESS;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t HSA_API fake_core_memory_allocate(hsa_region_t region, size_t size, void **ptr) {
  ++g_core_memory_allocate_calls;
  if (region.handle != 30 || ptr == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (g_fail_core_memory_allocate)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  void *allocation = std::malloc(size);
  if (allocation == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  *ptr = allocation;
  g_core_memory_allocation_sizes.push_back(size);
  g_core_memory_allocations.push_back(allocation);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_core_memory_free(void *ptr) {
  if (ptr == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  ++g_core_memory_free_calls;
  const auto it = std::ranges::find(g_core_memory_allocations, ptr);
  if (it == g_core_memory_allocations.end())
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  const size_t index = static_cast<size_t>(it - g_core_memory_allocations.begin());
  if (g_core_memory_allocation_sizes[index] >= sizeof(rocjitsu::ConSanMoiReportHeader))
    g_core_memory_headers_at_free.push_back(
        *static_cast<const rocjitsu::ConSanMoiReportHeader *>(ptr));
  else
    g_sc_markers_at_free.push_back(*static_cast<const uint32_t *>(ptr));
  g_core_memory_allocation_sizes.erase(g_core_memory_allocation_sizes.begin() + index);
  g_core_memory_allocations.erase(it);
  std::free(ptr);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_memory_assign_agent(void *, hsa_agent_t agent, hsa_access_permission_t) {
  return agent.handle == kHostAgent.handle ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_AGENT;
}

rocjitsu::ConSanResult transform_override(std::span<const uint8_t> bytes,
                                          const rocjitsu::ConSanOptions &options) {
  g_transform_override_flavors.push_back(options.flavor);
  g_transform_override_engines.push_back(options.moi_engine);
  g_transform_override_abort_unmatched_waits.push_back(options.abort_unmatched_barrier_wait);
  g_transform_override_track_barriers.push_back(options.moi_track_barriers);
  g_transform_override_track_atomics.push_back(options.moi_track_atomics);
  g_transform_override_runtime_sample_strides.push_back(options.moi_runtime_sample_stride);
  g_transform_override_sc_report_addresses.push_back(options.report_buffer_address);
  g_transform_override_report_sizes.push_back(options.moi_report_buffer_size);
  g_transform_override_report_layouts.push_back(options.moi_report_layout);
  rocjitsu::ConSanResult result = g_transform_override_result;
  result.visited_code_object = true;
  result.input_size = bytes.size();
  result.flavor = options.flavor;
  result.moi_engine = options.moi_engine;
  return result;
}

hsa_status_t HSA_API fake_executable_load_agent_code_object(
    hsa_executable_t, hsa_agent_t agent, hsa_code_object_reader_t reader, const char *,
    hsa_loaded_code_object_t *loaded_code_object) {
  ++g_fake_load_agent_calls;
  g_last_load_agent = agent;
  g_last_load_reader = reader;
  g_loaded_code_object_readers.push_back(reader.handle);
  if (loaded_code_object != nullptr)
    loaded_code_object->handle = 77;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_get_symbol_by_name(hsa_executable_t, const char *symbol_name,
                                                        const hsa_agent_t *,
                                                        hsa_executable_symbol_t *symbol) {
  if (symbol_name == nullptr || symbol == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (symbol_name != g_fake_symbol_name)
    return HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
  *symbol = kFakeKernelSymbol;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, void *),
    void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return callback(executable, agent, kFakeKernelSymbol, data);
}

hsa_status_t HSA_API fake_executable_symbol_get_info(hsa_executable_symbol_t symbol,
                                                     hsa_executable_symbol_info_t attribute,
                                                     void *value) {
  if (symbol.handle != kFakeKernelSymbol.handle || value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH) {
    const uint32_t name_length = static_cast<uint32_t>(g_fake_symbol_name.size());
    std::memcpy(value, &name_length, sizeof(name_length));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_NAME) {
    std::memcpy(value, g_fake_symbol_name.data(), g_fake_symbol_name.size());
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
    std::memcpy(value, &g_fake_symbol_kernel_object, sizeof(g_fake_symbol_kernel_object));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) {
    std::memcpy(value, &g_fake_symbol_group_segment_size, sizeof(g_fake_symbol_group_segment_size));
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) {
    std::memcpy(value, &g_fake_symbol_private_segment_size,
                sizeof(g_fake_symbol_private_segment_size));
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                                   uint32_t, void **ptr) {
  g_last_allocate_pool = memory_pool;
  if (ptr != nullptr) {
    g_fake_allocations.emplace_back(size == 0 ? 1 : size);
    g_fake_allocation_pools.push_back(memory_pool);
    g_fake_allocation_sizes.push_back(size);
    *ptr = g_fake_allocations.back().data();
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_pool_free(void *ptr) {
  g_fake_freed_allocations.push_back(ptr);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agents_allow_access(uint32_t, const hsa_agent_t *, const uint32_t *,
                                                  const void *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_async_batch_copy(const hsa_amd_memory_copy_op_t *copy_ops,
                                                      uint32_t num_copy_ops, uint32_t,
                                                      const hsa_signal_t *) {
  g_last_batch_src_agents.clear();
  g_last_batch_dst_agents.clear();
  if (copy_ops == nullptr && num_copy_ops != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  for (uint32_t op_idx = 0; op_idx < num_copy_ops; ++op_idx) {
    const hsa_amd_memory_copy_op_t &op = copy_ops[op_idx];
    switch (static_cast<hsa_amd_memory_copy_op_type_t>(op.type)) {
    case HSA_AMD_MEMORY_COPY_OP_LINEAR:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRC:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_DST:
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST:
      if (op.num_entries == 0) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent.handle);
        continue;
      }
      if (op.dst_agent_list == nullptr)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      for (uint16_t entry_idx = 0; entry_idx < op.num_entries; ++entry_idx) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent_list[entry_idx].handle);
      }
      continue;
    case HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST:
      if (op.num_entries == 0 || op.dst_agent_list == nullptr)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      for (uint16_t entry_idx = 0; entry_idx < op.num_entries; ++entry_idx) {
        g_last_batch_src_agents.push_back(op.src_agent.handle);
        g_last_batch_dst_agents.push_back(op.dst_agent_list[entry_idx].handle);
      }
      continue;
    }
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_lock(void *, size_t, hsa_agent_t *agents, int num_agent,
                                          void **) {
  g_last_memory_lock_agents.clear();
  if (agents == nullptr && num_agent != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < num_agent; ++i)
    g_last_memory_lock_agents.push_back(agents[i].handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_memory_lock_to_pool(void *, size_t, hsa_agent_t *agents,
                                                  int num_agent, hsa_amd_memory_pool_t pool,
                                                  uint32_t, void **) {
  g_last_memory_lock_to_pool_pool = pool;
  g_last_memory_lock_to_pool_agents.clear();
  if (agents == nullptr && num_agent != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (int i = 0; i < num_agent; ++i)
    g_last_memory_lock_to_pool_agents.push_back(agents[i].handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_vmem_set_access(void *, size_t,
                                              const hsa_amd_memory_access_desc_t *desc,
                                              size_t desc_cnt) {
  g_last_vmem_access_agents.clear();
  if (desc == nullptr && desc_cnt != 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < desc_cnt; ++i)
    g_last_vmem_access_agents.push_back(desc[i].agent_handle.handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_pointer_info(const void *, hsa_amd_pointer_info_t *info,
                                           void *(*)(size_t), uint32_t *num_agents_accessible,
                                           hsa_agent_t **accessible) {
  if (info != nullptr) {
    info->size = sizeof(hsa_amd_pointer_info_t);
    info->agentOwner = kHostAgent;
  }
  if (num_agents_accessible != nullptr && accessible != nullptr) {
    g_pointer_info_accessible[0] = kHostAgent;
    g_pointer_info_accessible[1] = kGuestAgent;
    *num_agents_accessible = 2;
    *accessible = g_pointer_info_accessible;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent.handle == kGuestAgent.handle) {
    if (g_fail_guest_pool_iteration_once) {
      g_fail_guest_pool_iteration_once = false;
      return HSA_STATUS_ERROR;
    }
    std::unique_lock lock(g_pool_mutex);
    if (g_block_guest_pool_iteration) {
      g_guest_pool_iteration_entered = true;
      g_pool_cv.notify_all();
      g_pool_cv.wait(lock, [] { return g_release_guest_pool_iteration; });
    }
    lock.unlock();
    return callback(kGuestPool, data);
  }
  if (agent.handle == kHostAgent.handle) {
    hsa_status_t status = callback(kHostPool, data);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    return callback(kHostKernargPool, data);
  }
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t HSA_API fake_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool,
                                                   hsa_amd_memory_pool_info_t attribute,
                                                   void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_AMD_MEMORY_POOL_INFO_SEGMENT) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS) {
    *static_cast<uint32_t *>(value) = pool.handle == kHostKernargPool.handle ? 1u : 2u;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED) {
    *static_cast<bool *>(value) = true;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_LOCATION) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t HSA_API fake_amd_agent_memory_pool_get_info(hsa_agent_t agent,
                                                         hsa_amd_memory_pool_t memory_pool,
                                                         hsa_amd_agent_memory_pool_info_t attribute,
                                                         void *value) {
  ++g_agent_memory_pool_get_info_calls;
  g_last_agent_memory_pool_agent = agent;
  g_last_agent_memory_pool = memory_pool;

  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (memory_pool.handle == 0)
    return HSA_STATUS_SUCCESS;
  if (attribute == HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

struct FakeApiTable {
  CoreApiTable core{};
  AmdExtTable amd{};
  HsaApiTable table{};

  FakeApiTable() {
    core.version.minor_id = sizeof(CoreApiTable);
    amd.version.minor_id = sizeof(AmdExtTable);
    table.version.minor_id = sizeof(HsaApiTable);
    table.core_ = &core;
    table.amd_ext_ = &amd;

    core.hsa_shut_down_fn = fake_shut_down;
    core.hsa_iterate_agents_fn = fake_iterate_agents;
    core.hsa_agent_get_info_fn = fake_agent_get_info;
    core.hsa_agent_iterate_isas_fn = fake_agent_iterate_isas;
    core.hsa_isa_get_info_alt_fn = fake_isa_get_info_alt;
    core.hsa_queue_create_fn = fake_queue_create;
    core.hsa_queue_destroy_fn = fake_queue_destroy;
    core.hsa_signal_create_fn = fake_signal_create;
    core.hsa_signal_destroy_fn = fake_signal_destroy;
    core.hsa_signal_load_scacquire_fn = fake_signal_load_scacquire;
    core.hsa_signal_store_relaxed_fn = fake_signal_store_relaxed;
    core.hsa_signal_store_screlease_fn = fake_signal_store_screlease;
    core.hsa_code_object_reader_create_from_file_fn = fake_code_object_reader_create_from_file;
    core.hsa_code_object_reader_create_from_memory_fn = fake_code_object_reader_create_from_memory;
    core.hsa_code_object_reader_destroy_fn = fake_code_object_reader_destroy;
    core.hsa_system_get_extension_table_fn = fake_system_get_extension_table;
    core.hsa_system_get_major_extension_table_fn = fake_system_get_major_extension_table;
    core.hsa_executable_load_agent_code_object_fn = fake_executable_load_agent_code_object;
    core.hsa_executable_get_symbol_by_name_fn = fake_executable_get_symbol_by_name;
    core.hsa_executable_symbol_get_info_fn = fake_executable_symbol_get_info;
    core.hsa_agent_iterate_regions_fn = fake_agent_iterate_regions;
    core.hsa_region_get_info_fn = fake_region_get_info;
    core.hsa_memory_allocate_fn = fake_core_memory_allocate;
    core.hsa_memory_free_fn = fake_core_memory_free;
    core.hsa_memory_assign_agent_fn = fake_memory_assign_agent;
    core.hsa_executable_iterate_agent_symbols_fn = fake_executable_iterate_agent_symbols;
    amd.hsa_amd_agent_iterate_memory_pools_fn = fake_amd_agent_iterate_memory_pools;
    amd.hsa_amd_memory_pool_get_info_fn = fake_amd_memory_pool_get_info;
    amd.hsa_amd_agent_memory_pool_get_info_fn = fake_amd_agent_memory_pool_get_info;
    amd.hsa_amd_memory_pool_allocate_fn = fake_amd_memory_pool_allocate;
    amd.hsa_amd_memory_async_batch_copy_fn = fake_amd_memory_async_batch_copy;
    amd.hsa_amd_memory_lock_fn = fake_amd_memory_lock;
    amd.hsa_amd_memory_lock_to_pool_fn = fake_amd_memory_lock_to_pool;
    amd.hsa_amd_pointer_info_fn = fake_amd_pointer_info;
    amd.hsa_amd_vmem_set_access_fn = fake_amd_vmem_set_access;
    amd.hsa_amd_queue_intercept_create_fn = fake_queue_intercept_create;
    amd.hsa_amd_queue_intercept_register_fn = fake_queue_intercept_register;
    amd.hsa_amd_memory_pool_free_fn = fake_amd_memory_pool_free;
    amd.hsa_amd_agents_allow_access_fn = fake_amd_agents_allow_access;
  }
};

void write_runtime_config_path() {
  std::filesystem::path runtime_dir =
      std::filesystem::temp_directory_path() /
      ("rocjitsu-hsa-hooks-unit-" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::create_directories(runtime_dir);
  setenv("ROCJITSU_RUNTIME_DIR", runtime_dir.c_str(), 1);

  std::ofstream config_path(rocjitsu::rpc_default_config_file_path());
  config_path << RJ_HOOK_UNIT_CONFIG_PATH << '\n';
}

class InstalledHook {
public:
  explicit InstalledHook(FakeApiTable &api) {
    OnUnload();
    write_runtime_config_path();
    installed_ = OnLoad(&api.table, 0, 0, nullptr);
  }
  ~InstalledHook() { OnUnload(); }

  [[nodiscard]] bool installed() const { return installed_; }

private:
  bool installed_ = false;
};

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name); old != nullptr)
      old_ = old;
    if (value != nullptr)
      setenv(name, value, 1);
    else
      unsetenv(name);
  }
  ~ScopedEnvVar() {
    if (old_)
      setenv(name_.c_str(), old_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

private:
  std::string name_;
  std::optional<std::string> old_;
};

class InstalledDbiHook {
public:
  explicit InstalledDbiHook(FakeApiTable &api) {
    write_runtime_config_path();
    const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
    const std::filesystem::path library =
        executable.parent_path().parent_path() /
        "lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so";
    library_ = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) {
      error_ = dlerror();
      return;
    }
    on_load_ = reinterpret_cast<OnLoadFn>(dlsym(library_, "OnLoad"));
    on_unload_ = reinterpret_cast<OnUnloadFn>(dlsym(library_, "OnUnload"));
    set_override_ = reinterpret_cast<SetOverrideFn>(
        dlsym(library_, "rj_dbi_test_set_consan_transform_override"));
    if (on_load_ == nullptr || on_unload_ == nullptr || set_override_ == nullptr) {
      error_ = dlerror();
      return;
    }
    on_unload_();
    set_override_(transform_override);
    installed_ = on_load_(&api.table, 0, 0, nullptr);
    if (!installed_)
      error_ = "DBI OnLoad returned false";
  }
  ~InstalledDbiHook() {
    if (on_unload_ != nullptr)
      on_unload_();
    if (set_override_ != nullptr)
      set_override_(nullptr);
    if (library_ != nullptr)
      dlclose(library_);
  }

  [[nodiscard]] bool installed() const { return installed_; }
  [[nodiscard]] const std::string &error() const { return error_; }

private:
  using OnLoadFn = bool (*)(HsaApiTable *, uint64_t, uint64_t, const char *const *);
  using OnUnloadFn = void (*)();
  using SetOverrideFn = void (*)(ConSanTransformOverride);
  void *library_ = nullptr;
  OnLoadFn on_load_ = nullptr;
  OnUnloadFn on_unload_ = nullptr;
  SetOverrideFn set_override_ = nullptr;
  bool installed_ = false;
  std::string error_;
};

struct ConSanHookProfile {
  const char *name;
  const char *mode;
  rocjitsu::ConSanFlavor expected_flavor;
  rocjitsu::ConSanMoiEngine expected_engine;
};

constexpr std::array kConSanHookProfiles = {
    ConSanHookProfile{"supercollider", "supercollider", rocjitsu::ConSanFlavor::SuperCollider,
                      rocjitsu::ConSanMoiEngine::RecordReplay},
    ConSanHookProfile{"record_replay", "record-replay", rocjitsu::ConSanFlavor::Moi,
                      rocjitsu::ConSanMoiEngine::RecordReplay},
    ConSanHookProfile{"inline_shadow", "inline-shadow", rocjitsu::ConSanFlavor::Moi,
                      rocjitsu::ConSanMoiEngine::InlineShadow},
    ConSanHookProfile{"sampled", "sampled", rocjitsu::ConSanFlavor::Moi,
                      rocjitsu::ConSanMoiEngine::Sampled},
};

void reset_code_object_observations() {
  g_code_object_reader_create_calls = 0;
  g_fail_replacement_reader_create = false;
  g_code_object_reader_inputs.clear();
  g_destroyed_code_object_readers.clear();
  g_loaded_code_object_readers.clear();
  g_transform_override_flavors.clear();
  g_transform_override_engines.clear();
  g_transform_override_abort_unmatched_waits.clear();
  g_transform_override_track_barriers.clear();
  g_transform_override_track_atomics.clear();
  g_transform_override_runtime_sample_strides.clear();
  g_transform_override_sc_report_addresses.clear();
  g_transform_override_report_sizes.clear();
  g_transform_override_report_layouts.clear();
  g_transform_override_result = {};
}

void reset_core_memory_observations() {
  ASSERT_TRUE(g_core_memory_allocations.empty());
  g_fail_core_memory_allocate = false;
  g_core_memory_allocate_calls = 0;
  g_core_memory_free_calls = 0;
  g_core_memory_allocation_sizes.clear();
  g_core_memory_headers_at_free.clear();
  g_sc_markers_at_free.clear();
}

void configure_consan_profile(const ConSanHookProfile &profile, bool fail_closed) {
  setenv("RJ_CONSAN_MODE", profile.mode, 1);
  unsetenv("RJ_CONSAN_POLICY");
  unsetenv("RJ_CONSAN_FLAVOR");
  unsetenv("RJ_CONSAN_MOI_ENGINE");
  unsetenv("RJ_CONSAN_MOI_BACKEND");
  setenv("RJ_CONSAN_FAIL_CLOSED", fail_closed ? "1" : "0", 1);
  unsetenv("RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT");
  unsetenv("RJ_CONSAN_MOI_TRACK_BARRIERS");
  unsetenv("RJ_CONSAN_MOI_TRACK_ATOMICS");
  unsetenv("RJ_CONSAN_SC_REPORT_MODE");
  unsetenv("RJ_CONSAN_REPORT_BUFFER");
  if (profile.expected_flavor == rocjitsu::ConSanFlavor::Moi) {
    setenv("RJ_CONSAN_MOI_REPORT_BUFFER", "4096", 1);
    setenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", "65536", 1);
    setenv("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", "0", 1);
  } else {
    unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER");
    unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE");
    unsetenv("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE");
  }
}

TEST(HsaHooksUnitTest, ConSanLoadedWithoutConfigurationDefaultsToMoiRecordReplay) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", nullptr);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", nullptr);
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", nullptr);
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", nullptr);
  ScopedEnvVar legacy_engine("RJ_CONSAN_MOI_BACKEND", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_MOI_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", "0");

  reset_code_object_observations();
  rocjitsu::ConSanResult unchanged;
  unchanged.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;
  g_transform_override_result = unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_flavors.size(), 1u);
  EXPECT_EQ(g_transform_override_flavors.front(), rocjitsu::ConSanFlavor::Moi);
  ASSERT_EQ(g_transform_override_engines.size(), 1u);
  EXPECT_EQ(g_transform_override_engines.front(), rocjitsu::ConSanMoiEngine::RecordReplay);
}

TEST(HsaHooksUnitTest, ConSanLegacySelectionRemainsActive) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", nullptr);
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", "moi");
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", "record_replay");
  ScopedEnvVar auto_report_size("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", "0");

  reset_code_object_observations();
  rocjitsu::ConSanResult unchanged;
  unchanged.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;
  g_transform_override_result = unchanged;
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_flavors.size(), 1u);
  EXPECT_EQ(g_transform_override_flavors.front(), rocjitsu::ConSanFlavor::Moi);
  ASSERT_EQ(g_transform_override_engines.size(), 1u);
  EXPECT_EQ(g_transform_override_engines.front(), rocjitsu::ConSanMoiEngine::RecordReplay);
}

TEST(HsaHooksUnitTest, ConSanRejectsInvalidMode) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "magic");
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", nullptr);
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", nullptr);
  ScopedEnvVar legacy_engine("RJ_CONSAN_MOI_BACKEND", nullptr);

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanRejectsInvalidPolicy) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "record-replay");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "fatal-races");
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", nullptr);
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", nullptr);
  ScopedEnvVar legacy_engine("RJ_CONSAN_MOI_BACKEND", nullptr);

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanRejectsModeCombinedWithLegacySelection) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "record-replay");
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", "moi");
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", nullptr);
  ScopedEnvVar legacy_engine("RJ_CONSAN_MOI_BACKEND", nullptr);

  reset_code_object_observations();
  FakeApiTable api;
  const auto original_load = api.core.hsa_executable_load_agent_code_object_fn;
  InstalledDbiHook hook(api);
  EXPECT_FALSE(hook.installed());
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn, original_load);
}

TEST(HsaHooksUnitTest, ConSanStrictPolicyRequiresCompleteInstrumentationButNotCleanDiagnostics) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
  ScopedEnvVar require_records("RJ_CONSAN_MOI_REQUIRE_RECORDS", nullptr);
  ScopedEnvVar forbid_diagnostics("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", nullptr);
  ScopedEnvVar forbid_overflow("RJ_CONSAN_MOI_FORBID_OVERFLOW", nullptr);

  ASSERT_EXIT(
      {
        FakeApiTable api;
        InstalledDbiHook hook(api);
        if (!hook.installed())
          std::_Exit(1);
      },
      testing::ExitedWithCode(86),
      "installed ConSan hook.*policy=strict.*fail_closed=true require_patch=true.*"
      "moi_require_records=true.*moi_forbid_diagnostics=false.*moi_forbid_overflow=true");
}

TEST(HsaHooksUnitTest, ConSanStrictPolicyTerminatesAtRejectedCodeObjectLoad) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[0], false);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", "strict");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", nullptr);
  g_transform_override_result.outcome = rocjitsu::ConSanTransformOutcome::Unsupported;

  ASSERT_EXIT(([] {
                FakeApiTable api;
                InstalledDbiHook hook(api);
                if (!hook.installed())
                  std::_Exit(1);
                std::array<uint8_t, 8> original{};
                original[0] = 0x7f;
                original[1] = 'E';
                original[2] = 'L';
                original[3] = 'F';
                hsa_code_object_reader_t reader{};
                if (api.core.hsa_code_object_reader_create_from_memory_fn(
                        original.data(), original.size(), &reader) != HSA_STATUS_SUCCESS)
                  std::_Exit(2);
                (void)api.core.hsa_executable_load_agent_code_object_fn(
                    hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
                std::_Exit(3);
              }()),
              testing::ExitedWithCode(92),
              "ConSan load rejection.*reason=non-installable-transform-outcome.*"
              "policy=strict action=terminate exit_code=92");
}

void expect_transform_profile(const ConSanHookProfile &profile) {
  ASSERT_EQ(g_transform_override_flavors.size(), 1u);
  ASSERT_EQ(g_transform_override_engines.size(), 1u);
  ASSERT_EQ(g_transform_override_abort_unmatched_waits.size(), 1u);
  EXPECT_EQ(g_transform_override_flavors.front(), profile.expected_flavor);
  EXPECT_EQ(g_transform_override_engines.front(), profile.expected_engine);
  EXPECT_FALSE(g_transform_override_abort_unmatched_waits.front());
  ASSERT_EQ(g_transform_override_track_barriers.size(), 1u);
  ASSERT_EQ(g_transform_override_track_atomics.size(), 1u);
  ASSERT_EQ(g_transform_override_runtime_sample_strides.size(), 1u);
  const bool expected_sync_defaults = profile.expected_flavor == rocjitsu::ConSanFlavor::Moi;
  EXPECT_EQ(g_transform_override_track_barriers.front(), expected_sync_defaults);
  EXPECT_EQ(g_transform_override_track_atomics.front(), expected_sync_defaults);
  const uint32_t expected_runtime_sample_stride =
      profile.expected_engine == rocjitsu::ConSanMoiEngine::Sampled ? 16384u : 1u;
  EXPECT_EQ(g_transform_override_runtime_sample_strides.front(), expected_runtime_sample_stride);
}

void run_hook_load_case(const ConSanHookProfile &profile, bool fail_closed,
                        rocjitsu::ConSanResult transform_result, hsa_status_t expected_load_status,
                        uint64_t expected_loaded_reader,
                        std::span<const uint8_t> expected_replacement = {},
                        bool fail_replacement_reader_create = false) {
  reset_code_object_observations();
  g_fail_replacement_reader_create = fail_replacement_reader_create;
  configure_consan_profile(profile, fail_closed);
  g_transform_override_result = std::move(transform_result);
  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << profile.name << ": " << hook.error();

  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t original_reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &original_reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(original_reader.handle, 101u);

  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, original_reader, nullptr, nullptr);
  EXPECT_EQ(status, expected_load_status) << profile.name;
  expect_transform_profile(profile);
  if (expected_loaded_reader == 0) {
    EXPECT_TRUE(g_loaded_code_object_readers.empty()) << profile.name;
  } else {
    ASSERT_EQ(g_loaded_code_object_readers.size(), 1u) << profile.name;
    EXPECT_EQ(g_loaded_code_object_readers.front(), expected_loaded_reader) << profile.name;
  }
  if (!expected_replacement.empty()) {
    ASSERT_EQ(g_code_object_reader_inputs.size(), 2u) << profile.name;
    EXPECT_EQ(g_code_object_reader_inputs.back(),
              std::vector<uint8_t>(expected_replacement.begin(), expected_replacement.end()))
        << profile.name;
    EXPECT_EQ(g_destroyed_code_object_readers, std::vector<uint64_t>{102u}) << profile.name;
  }
  if (fail_replacement_reader_create) {
    EXPECT_EQ(g_code_object_reader_create_calls, 2) << profile.name;
    ASSERT_EQ(g_code_object_reader_inputs.size(), 1u) << profile.name;
    EXPECT_TRUE(g_destroyed_code_object_readers.empty()) << profile.name;
  }
}

void reset_pool_blocker(bool enabled) {
  std::lock_guard lock(g_pool_mutex);
  g_block_guest_pool_iteration = enabled;
  g_guest_pool_iteration_entered = false;
  g_release_guest_pool_iteration = false;
  g_fail_guest_pool_iteration_once = false;
}

void release_pool_blocker() {
  {
    std::lock_guard lock(g_pool_mutex);
    g_release_guest_pool_iteration = true;
  }
  g_pool_cv.notify_all();
}

void reset_agent_blocker(bool enabled) {
  std::lock_guard lock(g_agent_mutex);
  g_block_agent_iteration = enabled;
  g_agent_iteration_entered = false;
  g_release_agent_iteration = false;
}

void release_agent_blocker() {
  {
    std::lock_guard lock(g_agent_mutex);
    g_release_agent_iteration = true;
  }
  g_agent_cv.notify_all();
}

void expect_batch_copy_forwarding(const hsa_amd_memory_copy_op_t &op,
                                  const std::vector<uint64_t> &expected_src_agents,
                                  const std::vector<uint64_t> &expected_dst_agents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_batch_src_agents.clear();
  g_last_batch_dst_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_async_batch_copy_fn, fake_amd_memory_async_batch_copy);

  EXPECT_EQ(api.amd.hsa_amd_memory_async_batch_copy_fn(&op, 1, 0, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_batch_src_agents, expected_src_agents);
  EXPECT_EQ(g_last_batch_dst_agents, expected_dst_agents);
}

rocjitsu::ConSanMoiSampledSyncDecodeResult
sampled_atomic(rocjitsu::ConSanMoiSampledSyncRole role, rocjitsu::ConSanMoiSampledSyncScope scope,
               rocjitsu::ConSanMoiSampledSyncOutcome outcome, uint64_t address = 0x1000,
               uint32_t byte_count = 4, uint32_t epoch = 7) {
  return {
      rocjitsu::ConSanMoiSampledSyncClassification::Valid,
      {
          .address = address,
          .byte_count = byte_count,
          .kind = rocjitsu::ConSanMoiSampledSyncKind::Atomic,
          .role = role,
          .scope = scope,
          .outcome = outcome,
          .epoch_before = epoch,
          .epoch_after = epoch,
      },
  };
}

TEST(HsaHooksUnitTest, ConSanLoaderHonorsAllTypedOutcomesAcrossAllProfiles) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", nullptr);
  ScopedEnvVar policy("RJ_CONSAN_POLICY", nullptr);
  ScopedEnvVar flavor("RJ_CONSAN_FLAVOR", nullptr);
  ScopedEnvVar engine("RJ_CONSAN_MOI_ENGINE", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_MOI_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", nullptr);

  for (const ConSanHookProfile &profile : kConSanHookProfiles) {
    SCOPED_TRACE(profile.name);

    rocjitsu::ConSanResult unchanged;
    unchanged.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;
    run_hook_load_case(profile, false, unchanged, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, unchanged, HSA_STATUS_SUCCESS, 101u);

    rocjitsu::ConSanResult unsupported;
    unsupported.outcome = rocjitsu::ConSanTransformOutcome::Unsupported;
    run_hook_load_case(profile, false, unsupported, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, unsupported, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);

    rocjitsu::ConSanResult invalid;
    invalid.outcome = rocjitsu::ConSanTransformOutcome::Invalid;
    run_hook_load_case(profile, false, invalid, HSA_STATUS_SUCCESS, 101u);
    run_hook_load_case(profile, true, invalid, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);

    const std::array<uint8_t, 7> replacement = {'p', 'a', 't', 'c', 'h', 'e', 'd'};
    rocjitsu::ConSanResult modified;
    modified.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
    modified.modified = true;
    modified.final_validation_passed = true;
    modified.elf_bytes.assign(replacement.begin(), replacement.end());
    run_hook_load_case(profile, false, modified, HSA_STATUS_SUCCESS, 102u, replacement);
    run_hook_load_case(profile, true, modified, HSA_STATUS_SUCCESS, 102u, replacement);
    run_hook_load_case(profile, false, modified, HSA_STATUS_SUCCESS, 101u, {}, true);
    run_hook_load_case(profile, true, modified, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u, {}, true);

    rocjitsu::ConSanResult corrupt = modified;
    corrupt.final_validation_passed = false;
    run_hook_load_case(profile, false, corrupt, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
    run_hook_load_case(profile, true, corrupt, HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  }
}

TEST(HsaHooksUnitTest, ConSanWaitcheckReportsHazardBeforeTransformRegardlessOfWaitcheckEnv) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  ScopedEnvVar waitcheck_enable("ROCJITSU_WAITCHECK", "0");
  ScopedEnvVar waitcheck_mode("ROCJITSU_WAITCHECK_MODE", "dispatch");
  ScopedEnvVar waitcheck_fail("ROCJITSU_WAITCHECK_FAIL", "0");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  std::vector<uint32_t> clean_kernel;
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::s_wait_loadcnt(0));
  rocjitsu::waitcheck_test::append_inst(clean_kernel, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  std::vector<uint32_t> hazardous_kernel;
  rocjitsu::waitcheck_test::append_inst(hazardous_kernel,
                                        rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous_kernel,
                                        rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_multi_kernel_code_object(
          {{"clean_kernel", clean_kernel}, {"hazardous_kernel", hazardous_kernel}});
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("ConSan preflight reported reader=101 target=gfx1201 "
                                        "reason=wait-hazard diagnostics=1");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanWaitcheckReportsAnalysisFailureBeforeTransform) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_invalid_instruction_code_object();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("ConSan preflight reported reader=101 target=gfx1201 "
                                        "reason=analysis-failed");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanWaitcheckPassesBeforeTransformForCleanCodeObject) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[1], false);
  g_transform_override_result.outcome = rocjitsu::ConSanTransformOutcome::Unchanged;

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();

  const std::vector<uint8_t> original =
      rocjitsu::waitcheck_test::make_gfx1201_correct_wait_code_object();
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);

  testing::internal::CaptureStderr();
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      hsa_executable_t{7}, kHostAgent, reader, nullptr, nullptr);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  expect_transform_profile(kConSanHookProfiles[1]);
  EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{101u});
  const size_t waitcheck_pos = log.find("waitcheck preflight reader=101 target=gfx1201 "
                                        "outcome=passed");
  const size_t consan_pos = log.find("ConSan patch begin reader=101");
  EXPECT_NE(waitcheck_pos, std::string::npos) << log;
  EXPECT_NE(consan_pos, std::string::npos) << log;
  EXPECT_LT(waitcheck_pos, consan_pos) << log;
}

TEST(HsaHooksUnitTest, ConSanRequirePatchRejectsPrologueOnlyMoiMutation) {
  ScopedEnvVar require_patch("RJ_CONSAN_REQUIRE_PATCH", "1");
  ScopedEnvVar require_records("RJ_CONSAN_MOI_REQUIRE_RECORDS", nullptr);

  rocjitsu::ConSanResult prologue_only;
  prologue_only.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
  prologue_only.modified = true;
  prologue_only.final_validation_passed = true;
  prologue_only.elf_bytes = {0x7f, 'E', 'L', 'F', 'p', 'r', 'o', 'l'};
  prologue_only.site_dispositions.push_back(
      {.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic,
       .disposition = rocjitsu::ConSanSiteDisposition::Supported,
       .reason = rocjitsu::ConSanSiteDispositionReason::None,
       .container_name = "supported_atomic",
       .mnemonic = "global_atomic_add"});
  rocjitsu::ConSanPatchInfo prologue_patch;
  prologue_patch.phase = rocjitsu::ConSanPatchPhase::Instrumentation;
  prologue_patch.kind = rocjitsu::ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  prologue_only.patches.push_back(prologue_patch);

  for (size_t i = 1; i < kConSanHookProfiles.size(); ++i) {
    SCOPED_TRACE(kConSanHookProfiles[i].name);
    run_hook_load_case(kConSanHookProfiles[i], false, prologue_only,
                       HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  }

  rocjitsu::ConSanResult resource_plan_only = prologue_only;
  resource_plan_only.site_dispositions.clear();
  rocjitsu::ConSanCandidateResourcePlan resource_plan;
  resource_plan.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic;
  resource_plan.source = rocjitsu::ConSanRegisterAllocationSource::Unsupported;
  resource_plan.reason = rocjitsu::ConSanRegisterPlanReason::NoLegalWindow;
  resource_plan_only.resource_plans.push_back(resource_plan);
  for (size_t i = 1; i < kConSanHookProfiles.size(); ++i) {
    SCOPED_TRACE(kConSanHookProfiles[i].name);
    run_hook_load_case(kConSanHookProfiles[i], false, resource_plan_only,
                       HSA_STATUS_ERROR_INVALID_CODE_OBJECT, 0u);
  }

  rocjitsu::ConSanResult site_patched = prologue_only;
  rocjitsu::ConSanPatchInfo site_patch;
  site_patch.phase = rocjitsu::ConSanPatchPhase::Instrumentation;
  site_patch.kind = rocjitsu::ConSanPatchKind::TrampolineMoiAtomicRecord;
  site_patched.patches.push_back(site_patch);
  run_hook_load_case(kConSanHookProfiles[1], false, site_patched, HSA_STATUS_SUCCESS, 102u,
                     site_patched.elf_bytes);
}

TEST(HsaHooksUnitTest, ConSanSynchronizationDefaultsRemainExplicitlyOverridable) {
  reset_code_object_observations();
  configure_consan_profile(kConSanHookProfiles[2], false);
  ScopedEnvVar track_barriers("RJ_CONSAN_MOI_TRACK_BARRIERS", "0");
  ScopedEnvVar track_atomics("RJ_CONSAN_MOI_TRACK_ATOMICS", "0");
  ScopedEnvVar abort_unmatched("RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT", "1");

  FakeApiTable api;
  InstalledDbiHook hook(api);
  ASSERT_TRUE(hook.installed()) << hook.error();
  constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(), original.size(),
                                                                  &reader),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(g_transform_override_track_barriers.size(), 1u);
  ASSERT_EQ(g_transform_override_track_atomics.size(), 1u);
  ASSERT_EQ(g_transform_override_abort_unmatched_waits.size(), 1u);
  EXPECT_FALSE(g_transform_override_track_barriers.front());
  EXPECT_FALSE(g_transform_override_track_atomics.front());
  EXPECT_TRUE(g_transform_override_abort_unmatched_waits.front());
}

rocjitsu::ConSanResult diagnostic_coverage_transform_result() {
  rocjitsu::ConSanResult result;
  result.flavor = rocjitsu::ConSanFlavor::Moi;
  result.moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
  result.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
  result.modified = true;
  result.final_validation_passed = true;
  result.elf_bytes = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};

  auto append_site =
      [&](rocjitsu::ConSanResourceSiteKind kind, rocjitsu::ConSanSiteDisposition disposition,
          rocjitsu::ConSanSiteDispositionReason reason, rocjitsu::ConSanSiteLoweringOutcome outcome,
          rocjitsu::ConSanSiteLoweringReason lowering_reason,
          rocjitsu::ConSanRegisterPlanReason resource_reason, std::string container, bool in_kernel,
          uint64_t text_offset, std::string mnemonic) {
        rocjitsu::ConSanSiteDispositionRecord site;
        site.site_kind = kind;
        site.disposition = disposition;
        site.reason = reason;
        site.container_name = std::move(container);
        site.in_kernel = in_kernel;
        site.text_offset = text_offset;
        site.mnemonic = std::move(mnemonic);
        site.lowering_outcome = outcome;
        site.lowering_reason = lowering_reason;
        site.resource_reason = resource_reason;
        result.site_dispositions.push_back(std::move(site));
      };
  append_site(
      rocjitsu::ConSanResourceSiteKind::Access, rocjitsu::ConSanSiteDisposition::Unsupported,
      rocjitsu::ConSanSiteDispositionReason::UnsupportedMnemonic,
      rocjitsu::ConSanSiteLoweringOutcome::Unsupported,
      rocjitsu::ConSanSiteLoweringReason::SemanticUnsupported,
      rocjitsu::ConSanRegisterPlanReason::None, "unsupported_kernel", true, 0x10, "ds_load_b96");
  append_site(rocjitsu::ConSanResourceSiteKind::Barrier, rocjitsu::ConSanSiteDisposition::Supported,
              rocjitsu::ConSanSiteDispositionReason::None,
              rocjitsu::ConSanSiteLoweringOutcome::ResourceFailed,
              rocjitsu::ConSanSiteLoweringReason::UnsupportedResourcePlan,
              rocjitsu::ConSanRegisterPlanReason::DynamicStack, "barrier_helper", false, 0x20,
              "s_barrier_wait");
  append_site(rocjitsu::ConSanResourceSiteKind::Atomic, rocjitsu::ConSanSiteDisposition::Supported,
              rocjitsu::ConSanSiteDispositionReason::None,
              rocjitsu::ConSanSiteLoweringOutcome::PlacementOrLoweringFailed,
              rocjitsu::ConSanSiteLoweringReason::InstrumentationPatchMissing,
              rocjitsu::ConSanRegisterPlanReason::None, "atomic_kernel", true, 0x30,
              "global_atomic_add");
  append_site(rocjitsu::ConSanResourceSiteKind::Fence, rocjitsu::ConSanSiteDisposition::Supported,
              rocjitsu::ConSanSiteDispositionReason::None,
              rocjitsu::ConSanSiteLoweringOutcome::Patched,
              rocjitsu::ConSanSiteLoweringReason::None, rocjitsu::ConSanRegisterPlanReason::None,
              "fence_kernel", true, 0x40, "fence");
  return result;
}

TEST(HsaHooksUnitTest, ConSanCoverageSiteDiagnosticsRetainStableReasonsAndSourceLocations) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "3");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  const rocjitsu::ConSanResult result = diagnostic_coverage_transform_result();

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 102u, result.elf_bytes);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=access disposition=unsupported "
                     "reason=unsupported_mnemonic outcome=unsupported "
                     "lowering_reason=semantic_unsupported resource_reason=none "
                     "container=unsupported_kernel scope=kernel text=0x10 "
                     "mnemonic=ds_load_b96"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=barrier disposition=supported "
                     "reason=none outcome=resource_failed "
                     "lowering_reason=unsupported_resource_plan resource_reason=dynamic_stack "
                     "container=barrier_helper scope=function text=0x20 "
                     "mnemonic=s_barrier_wait"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=atomic disposition=supported "
                     "reason=none outcome=placement_or_lowering_failed "
                     "lowering_reason=instrumentation_patch_missing resource_reason=none "
                     "container=atomic_kernel scope=kernel text=0x30 "
                     "mnemonic=global_atomic_add"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("ConSan coverage_site reader=101 kind=fence disposition=supported "
                     "reason=none outcome=patched lowering_reason=none resource_reason=none "
                     "container=fence_kernel scope=kernel text=0x40 mnemonic=fence"),
            std::string::npos)
      << log;
}

TEST(HsaHooksUnitTest, ConSanCoverageDoesNotResurrectNotApplicableResourcePlan) {
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");
  const ConSanHookProfile &profile = kConSanHookProfiles[1];
  rocjitsu::ConSanResult result;
  result.flavor = rocjitsu::ConSanFlavor::Moi;
  result.moi_engine = rocjitsu::ConSanMoiEngine::RecordReplay;
  result.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
  result.modified = true;
  result.final_validation_passed = true;
  result.elf_bytes = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};
  result.site_dispositions.push_back(
      {.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic,
       .disposition = rocjitsu::ConSanSiteDisposition::NotApplicable,
       .reason = rocjitsu::ConSanSiteDispositionReason::NoAtomicAcquireConsumer,
       .container_name = "isolated_release",
       .in_kernel = true,
       .text_offset = 0x30,
       .mnemonic = "ds_add_u32",
       .lowering_outcome = rocjitsu::ConSanSiteLoweringOutcome::NotApplicable,
       .lowering_reason = rocjitsu::ConSanSiteLoweringReason::SemanticNotApplicable});
  rocjitsu::ConSanCandidateResourcePlan atomic_plan;
  atomic_plan.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic;
  atomic_plan.text_offset = 0x30;
  result.resource_plans.push_back(std::move(atomic_plan));

  testing::internal::CaptureStderr();
  run_hook_load_case(profile, false, result, HSA_STATUS_SUCCESS, 102u, result.elf_bytes);
  const std::string log = testing::internal::GetCapturedStderr();

  EXPECT_NE(log.find("atomic_discovered=0 atomic_supported=0 atomic_selected=0 "
                     "atomic_patched=0 atomic_unsupported=0 atomic_resource_failed=0 "
                     "atomic_placement_or_lowering_failed=0 atomic_expert_limit_omitted=0"),
            std::string::npos)
      << log;
  EXPECT_EQ(log.find("coverage_site reader=101 kind=atomic"), std::string::npos) << log;
}

rocjitsu::ConSanResult auto_report_atomic_transform_result() {
  rocjitsu::ConSanResult result;
  result.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
  result.modified = true;
  result.final_validation_passed = true;
  result.elf_bytes = {0x7f, 'E', 'L', 'F', 'p', 'a', 't', 'c', 'h'};
  rocjitsu::ConSanCandidateResourcePlan atomic_plan;
  atomic_plan.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic;
  result.resource_plans.push_back(atomic_plan);
  result.kernels.emplace_back();
  result.kernels.back().name = "auto_report_atomic";
  result.kernels.back().atomic_sites.emplace_back();
  result.site_dispositions.push_back({.site_kind = rocjitsu::ConSanResourceSiteKind::Atomic,
                                      .disposition = rocjitsu::ConSanSiteDisposition::Supported,
                                      .reason = rocjitsu::ConSanSiteDispositionReason::None,
                                      .container_name = "auto_report_atomic",
                                      .mnemonic = "global_atomic_add"});
  return result;
}

rocjitsu::ConSanResult auto_sc_transform_result() {
  rocjitsu::ConSanResult result;
  result.outcome = rocjitsu::ConSanTransformOutcome::ModifiedValid;
  result.modified = true;
  result.final_validation_passed = true;
  result.elf_bytes = {0x7f, 'E', 'L', 'F', 's', 'c'};
  rocjitsu::ConSanPatchInfo patch;
  patch.phase = rocjitsu::ConSanPatchPhase::Instrumentation;
  patch.kind = rocjitsu::ConSanPatchKind::LocalCaveLdsStoreCheckTrap;
  result.patches.push_back(patch);
  return result;
}

TEST(HsaHooksUnitTest, ConSanScAutoReportUsesMarkerAndCleansUpWithoutTrapFallback) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);
  ScopedEnvVar log_level("RJ_CONSAN_LOG", "1");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(g_core_memory_allocations.size(), 1u);
    ASSERT_EQ(g_core_memory_allocation_sizes.back(), sizeof(uint32_t));
    *static_cast<uint32_t *>(g_core_memory_allocations.front()) = 1;
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 2u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses[0]);
    EXPECT_TRUE(g_transform_override_sc_report_addresses[1]);
  }
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 1);
  EXPECT_EQ(g_sc_markers_at_free, std::vector<uint32_t>{1});
}

TEST(HsaHooksUnitTest, ConSanScTrapIsExplicitAndAllocationFailureDoesNotFallBack) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "0");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", "trap");
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 0);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses.front());
  }

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar explicit_report_buffer("RJ_CONSAN_REPORT_BUFFER", "4096");
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 0);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_EQ(g_transform_override_sc_report_addresses.front(), 4096u);
  }

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_sc_transform_result();
  {
    ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(g_core_memory_allocate_calls, 1);
    ASSERT_EQ(g_transform_override_sc_report_addresses.size(), 1u);
    EXPECT_FALSE(g_transform_override_sc_report_addresses.front());
    EXPECT_EQ(g_loaded_code_object_readers, std::vector<uint64_t>{reader.handle});
  }
  g_fail_core_memory_allocate = false;
  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 0);
}

TEST(HsaHooksUnitTest, ConSanScAutoReportAllocationFailureRejectsWhenFailClosed) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "supercollider");
  ScopedEnvVar report_mode("RJ_CONSAN_SC_REPORT_MODE", nullptr);
  ScopedEnvVar report_buffer("RJ_CONSAN_REPORT_BUFFER", nullptr);
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_sc_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_TRUE(g_loaded_code_object_readers.empty());
  }
  g_fail_core_memory_allocate = false;
}

TEST(HsaHooksUnitTest, ConSanAutoReportUsesExactLayoutAcrossTwoLiveCodeObjectsAndCleansUp) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "record-replay");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_MOI_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar dynamic_records("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS", "0");
  ScopedEnvVar max_patches("RJ_CONSAN_MAX_PATCHES", nullptr);

  reset_code_object_observations();
  reset_core_memory_observations();
  g_transform_override_result = auto_report_atomic_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();

    constexpr std::array<uint8_t, 8> first = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    constexpr std::array<uint8_t, 8> second = {0x7f, 'E', 'L', 'F', 5, 6, 7, 8};
    hsa_code_object_reader_t first_reader{};
    hsa_code_object_reader_t second_reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(first.data(), first.size(),
                                                                    &first_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(second.data(), second.size(),
                                                                    &second_reader),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                first_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                second_reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);

    ASSERT_EQ(g_core_memory_allocation_sizes.size(), 2u);
    EXPECT_EQ(g_core_memory_allocations.size(), 2u);
    EXPECT_EQ(g_core_memory_free_calls, 0);
    for (size_t size : g_core_memory_allocation_sizes) {
      EXPECT_GE(size, sizeof(rocjitsu::ConSanMoiReportHeader));
      EXPECT_LE(size, static_cast<size_t>(rocjitsu::kConSanMoiAutoReportBufferCeilingBytes));
    }
    ASSERT_EQ(g_transform_override_report_sizes.size(), 4u);
    ASSERT_EQ(g_transform_override_report_layouts.size(), 4u);
    for (size_t inventory_index : {0u, 2u}) {
      EXPECT_EQ(g_transform_override_report_sizes[inventory_index], 0u);
      EXPECT_FALSE(g_transform_override_report_layouts[inventory_index]);
      const size_t patch_index = inventory_index + 1u;
      EXPECT_EQ(g_transform_override_report_sizes[patch_index],
                g_core_memory_allocation_sizes[inventory_index / 2u]);
      ASSERT_TRUE(g_transform_override_report_layouts[patch_index]);
      EXPECT_EQ(g_transform_override_report_layouts[patch_index]->required_bytes,
                g_core_memory_allocation_sizes[inventory_index / 2u]);
    }
  }

  EXPECT_TRUE(g_core_memory_allocations.empty());
  EXPECT_EQ(g_core_memory_free_calls, 2);
  ASSERT_EQ(g_core_memory_headers_at_free.size(), 2u);
  for (const auto &header : g_core_memory_headers_at_free) {
    EXPECT_TRUE(rocjitsu::consan_moi_report_header_is_current(header));
    EXPECT_EQ(header.atomic_record_capacity, rocjitsu::kConSanMoiRecordReplayDynamicEventHeadroom);
    EXPECT_GT(header.diagnostic_capacity, 0u);
  }
}

TEST(HsaHooksUnitTest, ConSanAutoReportAllocationFailureFailsClosedWithoutLeakingBudget) {
  ScopedEnvVar mode("RJ_CONSAN_MODE", "record-replay");
  ScopedEnvVar fail_closed("RJ_CONSAN_FAIL_CLOSED", "1");
  ScopedEnvVar report_buffer("RJ_CONSAN_MOI_REPORT_BUFFER", nullptr);
  ScopedEnvVar report_size("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar auto_report_size("RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE", nullptr);
  ScopedEnvVar dynamic_records("RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS", "0");

  reset_code_object_observations();
  reset_core_memory_observations();
  g_fail_core_memory_allocate = true;
  g_transform_override_result = auto_report_atomic_transform_result();
  {
    FakeApiTable api;
    InstalledDbiHook hook(api);
    ASSERT_TRUE(hook.installed()) << hook.error();
    constexpr std::array<uint8_t, 8> original = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
    hsa_code_object_reader_t reader{};
    ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(original.data(),
                                                                    original.size(), &reader),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(hsa_executable_t{7}, kHostAgent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_OUT_OF_RESOURCES);
    EXPECT_EQ(g_core_memory_allocate_calls, 1);
    EXPECT_TRUE(g_core_memory_allocations.empty());
    ASSERT_EQ(g_transform_override_report_sizes.size(), 1u);
    EXPECT_EQ(g_transform_override_report_sizes.front(), 0u);
  }
  g_fail_core_memory_allocate = false;
  EXPECT_EQ(g_core_memory_free_calls, 0);
  EXPECT_TRUE(g_core_memory_allocations.empty());
}

TEST(HsaHooksUnitTest, SampledAtomicPairAcceptsCompleteReleaseToAcquireEvidence) {
  using Outcome = rocjitsu::ConSanMoiSampledSyncOutcome;
  using Role = rocjitsu::ConSanMoiSampledSyncRole;
  using Scope = rocjitsu::ConSanMoiSampledSyncScope;

  const auto release = sampled_atomic(Role::Release, Scope::Workgroup, Outcome::NotApplicable);
  const auto acquire = sampled_atomic(Role::Acquire, Scope::System, Outcome::NotApplicable);
  EXPECT_TRUE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(release, acquire));

  const auto successful_cas =
      sampled_atomic(Role::RmwAcquireRelease, Scope::Agent, Outcome::CasSuccess);
  const auto failed_acquire =
      sampled_atomic(Role::RmwAcquire, Scope::Workgroup, Outcome::CasFailure);
  EXPECT_TRUE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(successful_cas,
                                                                             failed_acquire));
}

TEST(HsaHooksUnitTest, SampledAtomicPairRejectsIncompleteDirectionRangeScopeEpochAndOutcome) {
  using Outcome = rocjitsu::ConSanMoiSampledSyncOutcome;
  using Role = rocjitsu::ConSanMoiSampledSyncRole;
  using Scope = rocjitsu::ConSanMoiSampledSyncScope;

  const auto release = sampled_atomic(Role::RmwRelease, Scope::Agent, Outcome::RmwReturnsOld);
  const auto acquire = sampled_atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld);
  EXPECT_TRUE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(acquire, release));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      release, sampled_atomic(Role::Release, Scope::System, Outcome::NotApplicable)));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      acquire, sampled_atomic(Role::Acquire, Scope::System, Outcome::NotApplicable)));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      release, sampled_atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1004)));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      release, sampled_atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1000, 8)));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      sampled_atomic(Role::RmwRelease, Scope::Wavefront, Outcome::RmwReturnsOld), acquire));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      release,
      sampled_atomic(Role::RmwAcquire, Scope::Agent, Outcome::RmwReturnsOld, 0x1000, 4, 8)));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      sampled_atomic(Role::RmwRelease, Scope::Agent, Outcome::CasFailure), acquire));
}

TEST(HsaHooksUnitTest, SampledAtomicPairFailsClosedOnMissingMalformedAndCollidingHalves) {
  using Classification = rocjitsu::ConSanMoiSampledSyncClassification;
  using Outcome = rocjitsu::ConSanMoiSampledSyncOutcome;
  using Role = rocjitsu::ConSanMoiSampledSyncRole;
  using Scope = rocjitsu::ConSanMoiSampledSyncScope;

  const auto release = sampled_atomic(Role::Release, Scope::Agent, Outcome::NotApplicable);
  const auto acquire = sampled_atomic(Role::Acquire, Scope::Agent, Outcome::NotApplicable);
  auto malformed = acquire;
  malformed.classification = Classification::Malformed;
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(release, malformed));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(
      rocjitsu::ConSanMoiSampledSyncDecodeResult{}, acquire));

  EXPECT_TRUE(rocjitsu::consan_moi_sampled_sync_report_is_complete(0, 0, 0));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_sync_report_is_complete(1, 0, 0));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_sync_report_is_complete(0, 1, 0));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_sync_report_is_complete(0, 0, 1));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_sync_report_is_complete(0, 0, 0, 1, 0));
  EXPECT_FALSE(rocjitsu::consan_moi_sampled_sync_report_is_complete(0, 0, 0, 0, 1));
}

TEST(HsaHooksUnitTest, InlineReleaseRenderingRequiresOneStableReleaseAndSnapshot) {
  using State = rocjitsu::ConSanMoiInlineReleaseSnapshotState;
  rocjitsu::ConSanMoiInlineReleaseSnapshotWords words;
  words.version_before = words.slot.version = words.version_after = 4;
  words.slot.owner_id = 2;
  words.slot.epoch_plus_one = 7;
  words.slot.workgroup_key = 0x30;
  words.slot.atomic_address = 0x4000;
  words.slot.dispatch_id = 0x500000006ull;
  words.snapshot.entry_count = 1;
  words.snapshot.entries[0] = {9, 3};
  EXPECT_EQ(rocjitsu::classify_consan_moi_inline_release_snapshot(words).state, State::Stable);

  words.version_after = 6;
  EXPECT_EQ(rocjitsu::classify_consan_moi_inline_release_snapshot(words).state,
            State::ChangedDuringRead);
  words.version_after = 4;
  words.snapshot.flags = rocjitsu::consan_moi_inline_causal_snapshot_flag(
      rocjitsu::ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow);
  EXPECT_EQ(rocjitsu::classify_consan_moi_inline_release_snapshot(words).state,
            State::CapacityOverflow);
  words.snapshot.flags = rocjitsu::consan_moi_inline_causal_snapshot_flag(
      rocjitsu::ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete);
  EXPECT_EQ(rocjitsu::classify_consan_moi_inline_release_snapshot(words).state,
            State::SourceIncomplete);
}

TEST(HsaHooksUnitTest, InlineTokenRenderingAdmitsOnlyStableDirectOrInheritedState) {
  using Kind = rocjitsu::ConSanMoiInlineTokenEvidenceKind;
  using State = rocjitsu::ConSanMoiInlineAcquiredTokenState;
  rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 2,
      .consumer_owner_id = 4,
      .producer_owner_id = 2,
      .producer_epoch_plus_one = 7,
      .workgroup_key = 0x30,
      .kind = static_cast<uint32_t>(Kind::Direct),
      .dispatch_id = 0x500000006ull,
      .source_release_address = 0x4000,
      .source_release_version = 4,
  };
  EXPECT_EQ(rocjitsu::consan_moi_inline_classify_acquired_token({2, token, 2}).state,
            State::Stable);
  token.kind = static_cast<uint32_t>(Kind::Inherited);
  EXPECT_EQ(rocjitsu::consan_moi_inline_classify_acquired_token({2, token, 2}).state,
            State::Stable);
  EXPECT_EQ(rocjitsu::consan_moi_inline_classify_acquired_token({2, token, 4}).state,
            State::Changed);
  token.kind = 0xffffffffu;
  EXPECT_EQ(rocjitsu::consan_moi_inline_classify_acquired_token({2, token, 2}).state,
            State::Malformed);
}

TEST(HsaHooksUnitTest, RecordReplayProvenanceUsesActualConflictingWorkgroupCell) {
  using AccessKind = rocjitsu::ConSanMoiShadowAccessKind;
  const auto access = [](uint32_t event_index, uint32_t workgroup_x, uint32_t owner, uint64_t lanes,
                         uint32_t instruction, uint32_t byte_offset, uint32_t byte_count,
                         uint32_t start_cell,
                         uint32_t cell_count) -> rocjitsu::ConSanMoiAccessRecord {
    return {
        .generation = 7,
        .workgroup_x = workgroup_x,
        .workgroup_y = 0,
        .workgroup_z = 0,
        .wave_id = owner,
        .lane_mask = lanes,
        .instruction_offset = instruction,
        .access_kind = static_cast<uint32_t>(AccessKind::Write),
        .lds_byte_offset = byte_offset,
        .lds_byte_count = byte_count,
        .start_cell = start_cell,
        .cell_count = cell_count,
        .epoch = 1,
        .event_index = event_index,
    };
  };
  const std::array records = {
      access(1, 0, 1, 0x1, 0x10, 8, 8, 2, 2),
      // Same-owner replacement changes only cell 2. It must be the provenance
      // selected for the later two-cell conflict, not the older wider access.
      // Its high instruction bits are deliberately absent from the packed
      // identity which the companion must match.
      access(2, 0, 1, 0x2, rocjitsu::consan_moi_exact_shadow::max_instruction_offset + 1u + 0x11u,
             8, 4, 2, 1),
      // The same cell in another workgroup must remain isolated.
      access(3, 1, 9, 0xff, 0x99, 8, 4, 2, 1),
      access(4, 0, 3, 0xc, 0x20, 8, 8, 2, 2),
  };
  rocjitsu::ConSanMoiDiagnosticRecord diagnostic{
      .kind = static_cast<uint32_t>(rocjitsu::ConSanMoiDiagnosticKind::AccessConflict),
      .backend = static_cast<uint32_t>(rocjitsu::ConSanMoiEngine::RecordReplay),
      .generation = 7,
      .epoch = 1,
      .first_owner_id = 1,
      .second_owner_id = 3,
      .first_instruction_offset = 0x11,
      .second_instruction_offset = 0x20,
      .first_access_kind = static_cast<uint32_t>(AccessKind::Write),
      .second_access_kind = static_cast<uint32_t>(AccessKind::Write),
  };
  // The packed-only engine intentionally cannot recover prior lane/range
  // evidence and must leave it unknown.
  EXPECT_EQ(diagnostic.first_lane_mask, 0u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 0u);
  const rocjitsu::ConSanMoiReplayProvenanceRepair repair =
      rocjitsu::repair_consan_moi_record_replay_provenance(records, {&diagnostic, 1});
  EXPECT_EQ(repair.repaired_diagnostic_count, 1u);
  EXPECT_EQ(repair.unresolved_diagnostic_count, 0u);
  EXPECT_EQ(diagnostic.first_instruction_offset, 0x11u);
  EXPECT_EQ(diagnostic.first_lane_mask, 0x2u);
  EXPECT_EQ(diagnostic.first_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 4u);
  EXPECT_EQ(diagnostic.second_lane_mask, 0xcu);
  EXPECT_EQ(diagnostic.second_lds_byte_offset, 8u);
  EXPECT_EQ(diagnostic.second_lds_byte_count, 8u);
}

TEST(HsaHooksUnitTest, RecordReplayProvenanceMismatchRemainsUnknown) {
  using AccessKind = rocjitsu::ConSanMoiShadowAccessKind;
  const std::array records = {
      rocjitsu::ConSanMoiAccessRecord{
          .generation = 3,
          .workgroup_x = 0,
          .workgroup_y = 0,
          .workgroup_z = 0,
          .wave_id = 1,
          .lane_mask = 0x5,
          .instruction_offset = 0x10,
          .access_kind = static_cast<uint32_t>(AccessKind::Write),
          .lds_byte_offset = 0,
          .lds_byte_count = 4,
          .start_cell = 0,
          .cell_count = 1,
          .epoch = 1,
          .event_index = 1,
      },
      rocjitsu::ConSanMoiAccessRecord{
          .generation = 3,
          .workgroup_x = 0,
          .workgroup_y = 0,
          .workgroup_z = 0,
          .wave_id = 2,
          .lane_mask = 0xa,
          .instruction_offset = 0x20,
          .access_kind = static_cast<uint32_t>(AccessKind::Write),
          .lds_byte_offset = 0,
          .lds_byte_count = 4,
          .start_cell = 0,
          .cell_count = 1,
          .epoch = 1,
          .event_index = 2,
      },
  };
  rocjitsu::ConSanMoiDiagnosticRecord diagnostic{
      .kind = static_cast<uint32_t>(rocjitsu::ConSanMoiDiagnosticKind::AccessConflict),
      .backend = static_cast<uint32_t>(rocjitsu::ConSanMoiEngine::RecordReplay),
      .generation = 3,
      .epoch = 1,
      .first_owner_id = 1,
      .second_owner_id = 2,
      .first_instruction_offset = 0xdead,
      .second_instruction_offset = 0x20,
      .first_access_kind = static_cast<uint32_t>(AccessKind::Write),
      .second_access_kind = static_cast<uint32_t>(AccessKind::Write),
  };

  const rocjitsu::ConSanMoiReplayProvenanceRepair repair =
      rocjitsu::repair_consan_moi_record_replay_provenance(records, {&diagnostic, 1});

  EXPECT_EQ(repair.repaired_diagnostic_count, 0u);
  EXPECT_EQ(repair.unresolved_diagnostic_count, 1u);
  EXPECT_EQ(diagnostic.first_lane_mask, 0u);
  EXPECT_EQ(diagnostic.first_lds_byte_count, 0u);
}

void reset_queue_fakes() {
  g_fake_queue_packets = {};
  g_fake_queue = {};
  g_last_queue_create_agent = {};
  g_last_destroyed_queue = nullptr;
  g_fake_allocations.clear();
  g_fake_allocation_pools.clear();
  g_fake_allocation_sizes.clear();
  g_fake_freed_allocations.clear();
  g_fake_signal_store_relaxed_calls = 0;
  g_fake_signal_store_screlease_calls = 0;
  g_last_signal_store_signal = {};
  g_last_signal_store_value = 0;
  g_next_fake_signal_handle = 10000;
  g_fake_created_signals.clear();
  g_fake_destroyed_signals.clear();
  g_fake_signal_values.clear();
  g_last_intercept_registered_queue = nullptr;
  g_fake_intercept_handler = nullptr;
  g_fake_intercept_user_data = nullptr;
  g_last_intercept_written_packets.clear();
  g_fake_symbol_kernel_object = 0;
  g_fake_symbol_group_segment_size = 0;
  g_fake_symbol_private_segment_size = 0;
  g_fake_symbol_name = "oversized_kernel.kd";
  g_fake_load_agent_calls = 0;
  g_last_load_agent = {};
  g_last_load_reader = {};
}

void write_bytes(std::vector<uint8_t> &image, size_t offset, const void *src, size_t size) {
  if (image.size() < offset + size)
    image.resize(offset + size);
  std::memcpy(image.data() + offset, src, size);
}

template <typename T>
void write_struct(std::vector<uint8_t> &image, size_t offset, const T &value) {
  write_bytes(image, offset, &value, sizeof(T));
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment <= 1)
    return value;
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

rocjitsu::Elf64_Ehdr make_amdgpu_elf_header(uint32_t mach) {
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(header.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  header.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  header.e_ident[rocjitsu::EI_DATA] = 1;
  header.e_ident[rocjitsu::EI_VERSION] = 1;
  header.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_AMDGPU_HSA;
  header.e_ident[rocjitsu::EI_ABIVERSION] = rocjitsu::ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_type = rocjitsu::ET_DYN;
  header.e_machine = rocjitsu::EM_AMDGPU;
  header.e_version = 1;
  header.e_flags = mach;
  header.e_ehsize = sizeof(rocjitsu::Elf64_Ehdr);
  return header;
}

struct VirtualLdsMetadataForTest {
  std::string kernel_name;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t virtual_descriptor_vaddr = 0;
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  uint32_t kernarg_size = 0;
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

std::vector<uint8_t>
make_translated_metadata_elf(uint32_t mach, const std::vector<VirtualLdsMetadataForTest> &records) {
  std::vector<rocjitsu::SidecarVariantMetadata> sidecars;
  std::vector<rocjitsu::KernargExtensionMetadata> kernarg_extensions;
  std::vector<rocjitsu::VirtualLdsKernelMetadata> virtual_lds;
  for (const VirtualLdsMetadataForTest &record : records) {
    sidecars.push_back({
        .kernel_name = record.kernel_name,
        .variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .normal_descriptor_vaddr = record.normal_descriptor_vaddr,
        .variant_descriptor_vaddr = record.virtual_descriptor_vaddr,
    });
    kernarg_extensions.push_back({
        .kernel_name = record.kernel_name,
        .variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .original_kernarg_size = record.kernarg_size,
        .payloads = {{
            .size = rocjitsu::kVirtualLdsRuntimeStateBytes,
            .alignment = alignof(uint64_t),
            .name = std::string(rocjitsu::kVirtualLdsRuntimeStatePayloadName),
        }},
    });
    const rocjitsu::KernargExtensionPayloadLayout payload{
        .size = rocjitsu::kVirtualLdsRuntimeStateBytes,
        .alignment = alignof(uint64_t),
    };
    const auto layout =
        rocjitsu::make_kernarg_extension_layout(record.kernarg_size, std::span{&payload, 1});
    EXPECT_TRUE(layout.has_value());
    if (layout) {
      EXPECT_EQ(layout->payload_offsets.front(), record.backing_pointer_kernarg_offset);
    }
    virtual_lds.push_back({
        .kernel_name = record.kernel_name,
        .sidecar_variant_name = std::string(rocjitsu::kVirtualLdsSidecarVariantName),
        .static_lds_bytes = record.static_lds_bytes,
        .normal_private_segment_size = record.normal_private_segment_size,
        .virtual_private_segment_size = record.virtual_private_segment_size,
        .virtual_lds_base_sgpr = record.virtual_lds_base_sgpr,
        .flags = record.flags,
    });
  }

  const std::array metadata = {
      rocjitsu::serialize_sidecar_metadata(sidecars),
      rocjitsu::serialize_kernarg_extension_metadata(kernarg_extensions),
      rocjitsu::serialize_virtual_lds_metadata(virtual_lds),
  };
  const std::array<std::string_view, 3> metadata_names = {
      rocjitsu::kSidecarMetadataSectionName,
      rocjitsu::kKernargExtensionMetadataSectionName,
      rocjitsu::kVirtualLdsMetadataSectionName,
  };

  std::string shstr(1, '\0');
  std::array<uint32_t, 3> metadata_name_offsets{};
  for (size_t i = 0; i < metadata_names.size(); ++i) {
    metadata_name_offsets[i] = static_cast<uint32_t>(shstr.size());
    shstr.append(metadata_names[i]);
    shstr.push_back('\0');
  }
  const uint32_t shstrtab_name = static_cast<uint32_t>(shstr.size());
  shstr.append(".shstrtab");
  shstr.push_back('\0');

  auto header = make_amdgpu_elf_header(mach);
  std::vector<uint8_t> image(sizeof(header));
  std::array<size_t, 3> metadata_offsets{};
  for (size_t i = 0; i < metadata.size(); ++i) {
    metadata_offsets[i] = image.size();
    write_bytes(image, metadata_offsets[i], metadata[i].data(), metadata[i].size());
  }
  const size_t shstrtab_offset = image.size();
  write_bytes(image, shstrtab_offset, shstr.data(), shstr.size());

  const size_t section_header_offset = align_up(image.size(), alignof(rocjitsu::Elf64_Shdr));
  image.resize(section_header_offset);

  std::array<rocjitsu::Elf64_Shdr, 5> sections{};
  for (size_t i = 0; i < metadata.size(); ++i) {
    sections[i + 1].sh_name = metadata_name_offsets[i];
    sections[i + 1].sh_type = rocjitsu::SHT_PROGBITS;
    sections[i + 1].sh_offset = metadata_offsets[i];
    sections[i + 1].sh_size = metadata[i].size();
  }
  sections[4].sh_name = shstrtab_name;
  sections[4].sh_type = rocjitsu::SHT_STRTAB;
  sections[4].sh_offset = shstrtab_offset;
  sections[4].sh_size = shstr.size();
  write_bytes(image, section_header_offset, sections.data(), sizeof(sections));

  header.e_shoff = section_header_offset;
  header.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 4;
  write_struct(image, 0, header);
  return image;
}

struct VirtualLdsRegistrationForTest {};

VirtualLdsRegistrationForTest register_virtual_lds_kernel_for_test(
    FakeApiTable &api, const rocr::llvm::amdhsa::kernel_descriptor_t &normal_descriptor,
    const rocr::llvm::amdhsa::kernel_descriptor_t &virtual_descriptor, uint32_t static_lds_bytes,
    uint32_t kernarg_size = 0,
    uint32_t backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
    uint16_t flags = kVirtualLdsWrapperFlagsForTest, bool resolve_symbol_by_name = true,
    bool request_loaded_code_object = true) {
  const auto normal_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  const auto virtual_object = reinterpret_cast<uintptr_t>(&virtual_descriptor);
  const int64_t descriptor_delta =
      static_cast<int64_t>(virtual_object) - static_cast<int64_t>(normal_object);
  constexpr uint64_t kNormalDescriptorVaddr = 0x100000000ull;
  const uint64_t virtual_descriptor_vaddr =
      static_cast<uint64_t>(static_cast<int64_t>(kNormalDescriptorVaddr) + descriptor_delta);

  g_fake_symbol_kernel_object = normal_object;
  g_fake_symbol_group_segment_size = normal_descriptor.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = normal_descriptor.private_segment_fixed_size;

  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = kNormalDescriptorVaddr,
      .virtual_descriptor_vaddr = virtual_descriptor_vaddr,
      .static_lds_bytes = static_lds_bytes,
      .normal_private_segment_size = normal_descriptor.private_segment_fixed_size,
      .virtual_private_segment_size = virtual_descriptor.private_segment_fixed_size,
      .kernarg_size = kernarg_size,
      .backing_pointer_kernarg_offset = backing_pointer_kernarg_offset,
      .virtual_lds_base_sgpr = 8,
      .flags = flags,
  }};
  // Load a target-matching object with the DBT metadata section prebuilt. This
  // exercises the same hook registry path as translated code objects without
  // depending on the production translator in this unit-test helper.
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  hsa_code_object_reader_t reader{};
  EXPECT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(
                kFakeExecutable, kGuestAgent, reader, nullptr,
                request_loaded_code_object ? &loaded : nullptr),
            HSA_STATUS_SUCCESS);

  if (!resolve_symbol_by_name)
    return {};

  hsa_executable_symbol_t symbol{};
  EXPECT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);

  uint64_t kernel_object = 0;
  EXPECT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernel_object, normal_object);

  return {};
}

struct IteratedSymbolForTest {
  hsa_agent_t agent{};
  hsa_executable_symbol_t symbol{};
};

hsa_status_t HSA_API capture_iterated_symbol_for_test(hsa_executable_t, hsa_agent_t agent,
                                                      hsa_executable_symbol_t symbol, void *data) {
  auto *captured = static_cast<IteratedSymbolForTest *>(data);
  captured->agent = agent;
  captured->symbol = symbol;
  return HSA_STATUS_SUCCESS;
}

TEST(HsaHooksUnitTest, IterateAgentsDropsGuestOwnSlotWhenGuestAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);

  std::vector<uint64_t> seen;
  hsa_status_t status = api.core.hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
        return HSA_STATUS_SUCCESS;
      },
      &seen);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

TEST(HsaHooksUnitTest, IterateAgentsDropsGuestOwnSlotWhenHostAppearsFirst) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_host_first;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents_host_first);

  std::vector<uint64_t> seen;
  hsa_status_t status = api.core.hsa_iterate_agents_fn(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
        return HSA_STATUS_SUCCESS;
      },
      &seen);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsScalarSourceAndDestinationAgents) {
  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR;
  op.src_agent = kGuestAgent;
  op.dst_agent = kGuestAgent;
  op.size = 64;

  expect_batch_copy_forwarding(op, {kHostAgent.handle}, {kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsMultiLinearScalarSourceAndDestinationList) {
  int src0 = 0;
  int src1 = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *src_list[] = {&src0, &src1};
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};
  size_t sizes[] = {64, 128};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR;
  op.num_entries = 2;
  op.src_list = src_list;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size_list = sizes;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsBroadcastScalarSourceAndDestinationList) {
  int src = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST;
  op.num_entries = 2;
  op.src = &src;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size = 64;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, BatchCopyMapsMultiIndirectScalarSourceAndDestinationList) {
  int src0 = 0;
  int src1 = 0;
  int dst0 = 0;
  int dst1 = 0;
  void *src_list[] = {&src0, &src1};
  void *dst_list[] = {&dst0, &dst1};
  hsa_agent_t dst_agents[] = {kGuestAgent, kHostAgent};
  size_t sizes[] = {64, 128};

  hsa_amd_memory_copy_op_t op{};
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_SRCDST;
  op.num_entries = 2;
  op.src_list = src_list;
  op.src_agent = kGuestAgent;
  op.dst_agent_list = dst_agents;
  op.dst_list = dst_list;
  op.size_list = sizes;

  expect_batch_copy_forwarding(op, {kHostAgent.handle, kHostAgent.handle},
                               {kHostAgent.handle, kHostAgent.handle});
}

TEST(HsaHooksUnitTest, PointerInfoReportsGuestIdentityOnce) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_pointer_info_fn, fake_amd_pointer_info);

  hsa_amd_pointer_info_t info{};
  uint32_t accessible_count = 0;
  hsa_agent_t *accessible = nullptr;
  hsa_status_t status = api.amd.hsa_amd_pointer_info_fn(&g_fake_allocation_storage, &info, nullptr,
                                                        &accessible_count, &accessible);

  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.agentOwner.handle, kGuestAgent.handle);
  ASSERT_NE(accessible, nullptr);
  ASSERT_EQ(accessible_count, 1u);
  EXPECT_EQ(accessible[0].handle, kGuestAgent.handle);
}

TEST(HsaHooksUnitTest, AgentMemoryPoolGetInfoRejectsNullPoolBeforeForwarding) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_agent_memory_pool_get_info_calls = 0;
  g_last_agent_memory_pool_agent = {};
  g_last_agent_memory_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_agent_memory_pool_get_info_fn, fake_amd_agent_memory_pool_get_info);

  uint32_t access = 0;
  hsa_status_t status = api.amd.hsa_amd_agent_memory_pool_get_info_fn(
      kGuestAgent, hsa_amd_memory_pool_t{}, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

  EXPECT_EQ(status, HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  EXPECT_EQ(g_agent_memory_pool_get_info_calls, 0);
}

TEST(HsaHooksUnitTest, PoolMapperRetriesAfterTransientPoolIterationFailure) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  g_fail_guest_pool_iteration_once = true;
  void *ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kGuestPool.handle);

  ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
}

TEST(HsaHooksUnitTest, MemoryLockDeduplicatesAgentsAfterGuestMapping) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_memory_lock_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_lock_fn, fake_amd_memory_lock);

  hsa_agent_t agents[] = {kGuestAgent, kHostAgent};
  int storage = 0;
  void *agent_ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_lock_fn(&storage, sizeof(storage), agents, 2, &agent_ptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_memory_lock_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, MemoryLockToPoolMapsPoolAndDeduplicatesAgents) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_memory_lock_to_pool_agents.clear();
  g_last_memory_lock_to_pool_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_lock_to_pool_fn, fake_amd_memory_lock_to_pool);

  hsa_agent_t agents[] = {kGuestAgent, kHostAgent};
  int storage = 0;
  void *agent_ptr = nullptr;
  EXPECT_EQ(api.amd.hsa_amd_memory_lock_to_pool_fn(&storage, sizeof(storage), agents, 2, kGuestPool,
                                                   0, &agent_ptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_memory_lock_to_pool_pool.handle, kHostPool.handle);
  EXPECT_EQ(g_last_memory_lock_to_pool_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, VmemSetAccessDeduplicatesDescriptorsAfterGuestMapping) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_last_vmem_access_agents.clear();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_vmem_set_access_fn, fake_amd_vmem_set_access);

  hsa_amd_memory_access_desc_t desc[] = {
      {.permissions = HSA_ACCESS_PERMISSION_RW, .agent_handle = kGuestAgent},
      {.permissions = HSA_ACCESS_PERMISSION_RW, .agent_handle = kHostAgent},
  };
  int storage = 0;
  EXPECT_EQ(api.amd.hsa_amd_vmem_set_access_fn(&storage, sizeof(storage), desc, 2),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_vmem_access_agents, std::vector<uint64_t>{kHostAgent.handle});
}

TEST(HsaHooksUnitTest, PoolAllocateWaitsForAgentDiscoveryPublication) {
  reset_pool_blocker(false);
  reset_agent_blocker(true);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_iterate_agents_fn, fake_iterate_agents);
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  std::vector<uint64_t> seen;
  hsa_status_t iterate_status = HSA_STATUS_ERROR;
  std::thread iterate_thread([&] {
    iterate_status = api.core.hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void *data) -> hsa_status_t {
          static_cast<std::vector<uint64_t> *>(data)->push_back(agent.handle);
          return HSA_STATUS_SUCCESS;
        },
        &seen);
  });

  bool mapper_entered_agent_iteration = false;
  {
    std::unique_lock lock(g_agent_mutex);
    mapper_entered_agent_iteration = g_agent_cv.wait_for(lock, std::chrono::seconds(1),
                                                         [] { return g_agent_iteration_entered; });
  }
  if (!mapper_entered_agent_iteration) {
    release_agent_blocker();
    iterate_thread.join();
    ADD_FAILURE() << "agent mapper did not enter discovery iteration";
    return;
  }

  std::atomic_bool allocate_done = false;
  hsa_status_t allocate_status = HSA_STATUS_ERROR;
  std::thread allocate_thread([&] {
    void *ptr = nullptr;
    allocate_status = api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr);
    allocate_done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(allocate_done.load());

  release_agent_blocker();
  iterate_thread.join();
  allocate_thread.join();

  EXPECT_EQ(iterate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(seen, std::vector<uint64_t>{kGuestAgent.handle});
  EXPECT_EQ(allocate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
  reset_agent_blocker(false);
}

TEST(HsaHooksUnitTest, UninstallDoesNotWaitForPoolMapperDiscoveryLock) {
  reset_pool_blocker(true);
  reset_agent_blocker(false);
  g_last_allocate_pool = {};
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.amd.hsa_amd_memory_pool_allocate_fn, fake_amd_memory_pool_allocate);

  hsa_status_t allocate_status = HSA_STATUS_ERROR;
  std::thread mapper_thread([&] {
    void *ptr = nullptr;
    allocate_status = api.amd.hsa_amd_memory_pool_allocate_fn(kGuestPool, 4096, 0, &ptr);
  });

  bool mapper_entered_pool_iteration = false;
  {
    std::unique_lock lock(g_pool_mutex);
    mapper_entered_pool_iteration = g_pool_cv.wait_for(
        lock, std::chrono::seconds(1), [] { return g_guest_pool_iteration_entered; });
  }
  if (!mapper_entered_pool_iteration) {
    release_pool_blocker();
    mapper_thread.join();
    ADD_FAILURE() << "mapper thread did not enter guest pool discovery";
    return;
  }

  bool uninstall_done = false;
  std::thread uninstall_thread([&] {
    OnUnload();
    std::lock_guard lock(g_pool_mutex);
    uninstall_done = true;
    g_pool_cv.notify_all();
  });

  bool completed_without_pool_release = false;
  {
    std::unique_lock lock(g_pool_mutex);
    completed_without_pool_release =
        g_pool_cv.wait_for(lock, std::chrono::seconds(1), [&] { return uninstall_done; });
  }

  release_pool_blocker();
  uninstall_thread.join();
  mapper_thread.join();

  EXPECT_TRUE(completed_without_pool_release);
  EXPECT_EQ(allocate_status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_allocate_pool.handle, kHostPool.handle);
  reset_pool_blocker(false);
}

TEST(HsaHooksUnitTest, GuestShutdownKeepsHookInstalledForProcessLifetime) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  g_fake_shutdown_calls = 0;
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  auto *patched_shutdown = api.core.hsa_shut_down_fn;
  ASSERT_NE(patched_shutdown, fake_shut_down);

  EXPECT_EQ(patched_shutdown(), HSA_STATUS_SUCCESS);

  EXPECT_EQ(g_fake_shutdown_calls, 0);
  EXPECT_EQ(api.core.hsa_shut_down_fn, patched_shutdown);
  EXPECT_NE(api.core.hsa_shut_down_fn, fake_shut_down);
}

TEST(HsaHooksUnitTest, VirtualLdsSymbolInfoReportsNormalDescriptorUntilPacketFallback) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  kernel_descriptor_t normal_descriptor{};
  normal_descriptor.group_segment_fixed_size = 108288;
  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  g_fake_symbol_group_segment_size = normal_descriptor.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 16;

  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = 0x1000,
      .virtual_descriptor_vaddr = 0x2000,
      .static_lds_bytes = normal_descriptor.group_segment_fixed_size,
      .normal_private_segment_size = 16,
      .virtual_private_segment_size = 16,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(loaded.handle, 77u);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(symbol.handle, kFakeKernelSymbol.handle);

  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));

  uint32_t group_segment_size = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(group_segment_size, normal_descriptor.group_segment_fixed_size);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryKeepsFittingDispatchOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  struct Descriptors {
    kernel_descriptor_t normal{};
    kernel_descriptor_t virtual_sidecar{};
  } descriptors;
  descriptors.normal.group_segment_fixed_size = 32 * 1024;
  descriptors.virtual_sidecar.private_segment_fixed_size = 96;
  ASSERT_GT(reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar),
            reinterpret_cast<uintptr_t>(&descriptors.normal));

  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  g_fake_symbol_group_segment_size = descriptors.normal.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 12;

  const uint64_t normal_descriptor_vaddr = 0x4000;
  const uint64_t sidecar_descriptor_delta =
      reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar) -
      reinterpret_cast<uintptr_t>(&descriptors.normal);
  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = normal_descriptor_vaddr,
      .virtual_descriptor_vaddr = normal_descriptor_vaddr + sidecar_descriptor_delta,
      .static_lds_bytes = descriptors.normal.group_segment_fixed_size,
      .normal_private_segment_size = 12,
      .virtual_private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // This uses the same load-time `.rocjitsu.lds` registry path as real DBT
  // code objects. Even when a virtual sidecar exists, a packet whose total LDS
  // request still fits CDNA3 must remain on the normal descriptor.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));
  EXPECT_EQ(packet.group_segment_size, 64u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryResolvesKernelObjectFromIteratedSymbol) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size,
      /*kernarg_size=*/0, kVirtualLdsWrapperStateOffsetForTest, kVirtualLdsWrapperFlagsForTest,
      /*resolve_symbol_by_name=*/false);
  (void)registration;

  IteratedSymbolForTest iterated{};
  ASSERT_EQ(api.core.hsa_executable_iterate_agent_symbols_fn(
                kFakeExecutable, kGuestAgent, capture_iterated_symbol_for_test, &iterated),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(iterated.agent.handle, kGuestAgent.handle);
  ASSERT_EQ(iterated.symbol.handle, kFakeKernelSymbol.handle);

  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                iterated.symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  // The packet carries 80 bytes of dynamic private memory above the normal
  // descriptor's fixed 40 bytes. The sidecar must retain those 80 bytes.
  packet.private_segment_size = 120;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // This path intentionally never calls hsa_executable_get_symbol_by_name().
  // The iterate wrapper must record the symbol name before the client asks for
  // KERNEL_OBJECT, otherwise the packet scanner cannot associate the normal
  // descriptor with the virtual-LDS metadata loaded from `.rocjitsu.lds`.
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(71024u * 4u));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 176u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRegistryRejectsSidecarDescriptorAsPacketInput) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();

  struct Descriptors {
    kernel_descriptor_t normal{};
    kernel_descriptor_t virtual_sidecar{};
  } descriptors;
  descriptors.normal.group_segment_fixed_size = 108288;
  descriptors.virtual_sidecar.group_segment_fixed_size = 0;
  descriptors.virtual_sidecar.private_segment_fixed_size = 96;
  ASSERT_GT(reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar),
            reinterpret_cast<uintptr_t>(&descriptors.normal));

  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptors.normal);
  g_fake_symbol_group_segment_size = descriptors.normal.group_segment_fixed_size;
  g_fake_symbol_private_segment_size = 12;

  const uint64_t normal_descriptor_vaddr = 0x5000;
  const uint64_t sidecar_descriptor_delta =
      reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar) -
      reinterpret_cast<uintptr_t>(&descriptors.normal);
  const std::vector<VirtualLdsMetadataForTest> metadata = {{
      .kernel_name = "oversized_kernel",
      .normal_descriptor_vaddr = normal_descriptor_vaddr,
      .virtual_descriptor_vaddr = normal_descriptor_vaddr + sidecar_descriptor_delta,
      .static_lds_bytes = descriptors.normal.group_segment_fixed_size,
      .normal_private_segment_size = 12,
      .virtual_private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size,
      .kernarg_size = 0,
      .backing_pointer_kernarg_offset = kVirtualLdsWrapperStateOffsetForTest,
      .virtual_lds_base_sgpr = 8,
      .flags = kVirtualLdsWrapperFlagsForTest,
  }};
  const auto code_object =
      make_translated_metadata_elf(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, metadata);

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(api.core.hsa_code_object_reader_create_from_memory_fn(code_object.data(),
                                                                  code_object.size(), &reader),
            HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  ASSERT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGuestAgent, reader,
                                                              nullptr, &loaded),
            HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernel_object, reinterpret_cast<uintptr_t>(&descriptors.normal));

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar);
  packet.private_segment_size = descriptors.virtual_sidecar.private_segment_fixed_size;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // The sidecar descriptor is write-only from rocjitsu's perspective: it may be
  // installed into a packet after the fallback threshold check, but it must not
  // be accepted as a lookup key for taking the fallback again.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptors.virtual_sidecar));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 96u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, QueueDoorbellSignalStoreIsForwardedAfterTrackedQueueScan) {
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_queue_create_fn, fake_queue_create);
  ASSERT_NE(api.core.hsa_signal_store_relaxed_fn, fake_signal_store_relaxed);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, kHostAgent.handle);

  g_fake_queue_packets[0].header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  EXPECT_EQ(g_fake_signal_store_relaxed_calls, 1);
  EXPECT_EQ(g_last_signal_store_signal.handle, queue->doorbell_signal.handle);
  EXPECT_EQ(g_last_signal_store_value, 0);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_last_destroyed_queue, queue);
}

TEST(HsaHooksUnitTest, QueueDoorbellRaisesPacketPrivateSizeFromDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.private_segment_fixed_size = 40;
  g_fake_symbol_kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  g_fake_symbol_private_segment_size = descriptor.private_segment_fixed_size;

  // Production kernel objects are GPU virtual addresses and cannot safely be
  // dereferenced by the packet hook. Exercise the real symbol-query path that
  // caches the runtime-reported private size before dispatch.
  hsa_executable_symbol_t symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &symbol),
            HSA_STATUS_SUCCESS);
  uint64_t kernel_object = 0;
  ASSERT_EQ(api.core.hsa_executable_symbol_get_info_fn(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object),
            HSA_STATUS_SUCCESS);

  // Frameworks can rediscover the same symbol through another lookup path
  // after querying its object. This must not erase the cached private size.
  hsa_executable_symbol_t repeated_symbol{};
  ASSERT_EQ(api.core.hsa_executable_get_symbol_by_name_fn(
                kFakeExecutable, g_fake_symbol_name.c_str(), &kGuestAgent, &repeated_symbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(repeated_symbol.handle, symbol.handle);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = kernel_object;
  packet.private_segment_size = 0;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Semantic DBT rules may add flat-scratch spills to a descriptor whose source
  // private segment was zero. ROCR can still hand us the original packet value,
  // so the queue scanner must raise the dispatch metadata before hardware sees
  // the packet.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(packet.private_segment_size, 40u);
  EXPECT_EQ(packet.group_segment_size, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteWorksWithoutLoadedCodeObjectOutput) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size, 0,
      kVirtualLdsWrapperStateOffsetForTest, kVirtualLdsWrapperFlagsForTest, true, false);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  // HSA packets report the total group-segment allocation. Static LDS is kept
  // separately in rocjitsu metadata only so symbol-time virtual descriptors,
  // which advertise zero hardware LDS, can still allocate the minimum backing
  // store when a packet arrives with a zero group size.
  packet.group_segment_size = kRequestedLds;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);

  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  // The normal descriptor has no fixed private allocation, so the packet's 12
  // bytes are entirely dynamic and must be added above the sidecar's 96 bytes.
  EXPECT_EQ(packet.private_segment_size, 108u);
  EXPECT_EQ(packet.reserved2, 0u);
  EXPECT_EQ(packet.kernarg_address, g_fake_allocations[1].data());

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data() + kVirtualLdsWrapperStateOffsetForTest,
              sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsScannerKeepsDestroyedSignalSlotUntilReuse) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  constexpr hsa_signal_t kApplicationSignal{4243};
  set_fake_signal_value(kApplicationSignal, 1);

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.completion_signal = kApplicationSignal;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  void *first_wrapper = packet.kernarg_address;

  set_fake_signal_value(kApplicationSignal, 0);
  EXPECT_EQ(api.core.hsa_signal_destroy_fn(kApplicationSignal), HSA_STATUS_SUCCESS);

  // The scanner still needs the slot record to recognize this as the already
  // rewritten packet. Ringing the same packet again must neither free its memory
  // while the stale packet points at it nor perform a second rewrite.
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  EXPECT_EQ(packet.kernarg_address, first_wrapper);
  EXPECT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_TRUE(g_fake_freed_allocations.empty());

  // Packet id 2 reuses slot 0. At this point the old packet is no longer visible,
  // so its explicitly-completed buffers can be retired without loading the
  // destroyed signal, before the replacement dispatch receives new buffers.
  packet = {};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 2);

  EXPECT_EQ(g_fake_allocation_sizes.size(), 4u);
  EXPECT_EQ(g_fake_freed_allocations.size(), 2u);
  EXPECT_NE(packet.kernarg_address, first_wrapper);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteCopiesOriginalKernargIntoWrapperPrefix) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;

  constexpr uint32_t kSourceKernargSize = 16;
  constexpr uint32_t kOriginalPointerOffset = 16;
  constexpr uint32_t kRuntimeStateOffset = 24;
  constexpr uint32_t kWrapperSize = 48;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size,
      kSourceKernargSize, kRuntimeStateOffset);
  (void)registration;

  const std::array<uint8_t, kSourceKernargSize> original_kernarg = {
      0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
      0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x0F,
  };

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.kernarg_address = const_cast<uint8_t *>(original_kernarg.data());
  packet.private_segment_size = 12;
  packet.group_segment_size = normal_descriptor.group_segment_fixed_size;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kWrapperSize);
  ASSERT_EQ(packet.kernarg_address, g_fake_allocations[1].data());
  const auto *wrapper = g_fake_allocations[1].data();
  EXPECT_EQ(std::memcmp(wrapper, original_kernarg.data(), original_kernarg.size()), 0);

  uint64_t copied_original_pointer = 0;
  std::memcpy(&copied_original_pointer, wrapper + kOriginalPointerOffset,
              sizeof(copied_original_pointer));
  EXPECT_EQ(copied_original_pointer, reinterpret_cast<uintptr_t>(original_kernarg.data()));

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, wrapper + kRuntimeStateOffset, sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, normal_descriptor.group_segment_fixed_size);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsBelowThresholdDispatchOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 0;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 16384;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 16;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Virtual LDS metadata may be present for a dynamic-LDS overflow fallback, but
  // a launch that fits in host hardware must keep the normal descriptor and
  // hardware LDS allocation untouched.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 16384u);
  EXPECT_EQ(packet.private_segment_size, 40u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsExactHardwareLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 64 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 0;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // Exactly 64 KiB still fits CDNA3 hardware LDS, so the virtual descriptor is
  // a fallback candidate only and must not be selected for this launch.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsRewriteKeepsStaticPlusDynamicLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 32 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  // HSA packets carry total LDS, not a dynamic-only tail. A 32 KiB static
  // descriptor plus 32 KiB dynamic request is a 64 KiB packet and still fits, so
  // the virtual sidecar is only a fallback candidate and must not replace it.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(packet.group_segment_size, 64u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);
  EXPECT_EQ(packet.reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptRewritePublishesWrapperKernarg) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(g_last_queue_create_agent.handle, kHostAgent.handle);
  EXPECT_EQ(g_last_intercept_registered_queue, queue);
  ASSERT_NE(g_fake_intercept_handler, nullptr);
  EXPECT_EQ(g_fake_intercept_user_data, queue);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  // The intercept adapter must preserve 80 dynamic bytes above normal fixed.
  packet.private_segment_size = 120;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  // HSA packets report total LDS, so dynamic LDS has already been added to the
  // packet value before rocjitsu scans or intercepts the dispatch.
  packet.group_segment_size = kRequestedLds;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], kVirtualLdsWrapperSizeForTest);

  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(written.group_segment_size, 0u);
  EXPECT_EQ(written.private_segment_size, 176u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(written.kernarg_address, g_fake_allocations[1].data());
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data() + kVirtualLdsWrapperStateOffsetForTest,
              sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptReleasesCompletedRetiredBuffers) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  packet.completion_signal = {};
  g_fake_queue_packets[1] = packet;
  g_fake_intercept_handler(&packet, 1, 1, g_fake_intercept_user_data, fake_intercept_packet_writer);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  ASSERT_EQ(g_fake_created_signals.size(), 1u);
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  EXPECT_EQ(g_last_intercept_written_packets[0].completion_signal.handle,
            g_fake_created_signals[0].handle);
  void *first_backing = g_fake_allocations[0].data();
  void *first_wrapper = g_fake_allocations[1].data();
  const hsa_signal_t first_signal = g_fake_created_signals[0];

  // Intercept callbacks retain virtual-LDS buffers after writing packets to
  // ROCR. Real framework packets are often fire-and-forget, so rocjitsu adds a
  // private completion signal and uses it as a fence. When a later callback
  // observes that signal at zero, the old backing/wrapper allocations can be
  // returned before allocating the next oversized dispatch.
  set_fake_signal_value(first_signal, 0);
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.reserved2 = 0;
  packet.completion_signal = {};
  g_fake_queue_packets[2] = packet;
  g_fake_intercept_handler(&packet, 1, 2, g_fake_intercept_user_data, fake_intercept_packet_writer);

  ASSERT_EQ(g_fake_allocation_sizes.size(), 4u);
  ASSERT_EQ(g_fake_created_signals.size(), 2u);
  ASSERT_EQ(g_fake_freed_allocations.size(), 2u);
  EXPECT_EQ(g_fake_freed_allocations[0], first_wrapper);
  EXPECT_EQ(g_fake_freed_allocations[1], first_backing);
  ASSERT_EQ(g_fake_destroyed_signals.size(), 1u);
  EXPECT_EQ(g_fake_destroyed_signals[0].handle, first_signal.handle);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptReleasesBorrowedSignalBeforeDestroy) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_signal_destroy_fn, fake_signal_destroy);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  constexpr hsa_signal_t kApplicationSignal{4242};
  set_fake_signal_value(kApplicationSignal, 1);

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 71024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.completion_signal = kApplicationSignal;

  g_fake_intercept_handler(&packet, 1, 1, g_fake_intercept_user_data, fake_intercept_packet_writer);
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  ASSERT_TRUE(g_fake_freed_allocations.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  EXPECT_EQ(g_last_intercept_written_packets[0].completion_signal.handle,
            kApplicationSignal.handle);

  // Once the application observes zero, it owns the right to destroy its
  // signal immediately. The destroy wrapper must release the already-completed
  // intercept buffers before the original runtime invalidates the handle; a
  // later dispatch must never need to load from that signal again.
  set_fake_signal_value(kApplicationSignal, 0);
  EXPECT_EQ(api.core.hsa_signal_destroy_fn(kApplicationSignal), HSA_STATUS_SUCCESS);

  ASSERT_EQ(g_fake_freed_allocations.size(), 2u);
  ASSERT_EQ(g_fake_destroyed_signals.size(), 1u);
  EXPECT_EQ(g_fake_destroyed_signals[0].handle, kApplicationSignal.handle);

  // Queue destruction must not release the same backing and kernarg twice after
  // signal destruction removed them from the retired list.
  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_freed_allocations.size(), 2u);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptKeepsBelowThresholdPacketOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 0;
  normal_descriptor.private_segment_fixed_size = 40;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 16384;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 16;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(written.group_segment_size, 16384u);
  EXPECT_EQ(written.private_segment_size, 40u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptKeepsStaticPlusDynamicLimitOnNormalDescriptor) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 32 * 1024;
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 64 * 1024;
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  // Queue-intercept dispatches use the same fallback rule as direct queue
  // scanning: if total LDS still fits CDNA3 hardware, leave the packet on the
  // normal descriptor and let real LDS handle the DS traffic.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&normal_descriptor));
  EXPECT_EQ(written.group_segment_size, 64u * 1024u);
  EXPECT_EQ(written.private_segment_size, 12u);
  EXPECT_EQ(written.reserved2, 0u);
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, DoorbellForwardsOversizedPacketOnUnrelatedAgentQueueUnchanged) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // A queue on an agent that is neither the guest nor the guest's execution host.
  // host_lds_bytes is derived from the configured host target (gfx1201 -> 64 KiB),
  // so it does not describe this agent. Its dispatches must be forwarded unchanged
  // even when the group segment exceeds that host limit -- the fail-close for
  // oversized-no-metadata launches applies only to the guest's own queue.
  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kUnrelatedAgent, 2, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                         nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.group_segment_fixed_size = 96 * 1024;

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 96 * 1024; // exceeds the guest-derived 64 KiB limit
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(packet.group_segment_size, 96u * 1024u);
  EXPECT_EQ(packet.private_segment_size, 12u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, InterceptForwardsOversizedPacketOnUnrelatedAgentQueueUnchanged) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
  api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kUnrelatedAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                         nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);
  ASSERT_NE(g_fake_intercept_handler, nullptr);

  kernel_descriptor_t descriptor{};
  descriptor.group_segment_fixed_size = 96 * 1024;

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 96 * 1024; // exceeds the guest-derived 64 KiB limit
  packet.workgroup_size_x = 256;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  // The unrelated-agent intercept queue forwards the oversized packet untouched:
  // no allocation, no rewrite, no abort.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());
  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&descriptor));
  EXPECT_EQ(written.group_segment_size, 96u * 1024u);
  EXPECT_EQ(written.private_segment_size, 12u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, LoadOnUnrelatedAgentForwardsDifferentTargetImageUnchanged) {
  // A code object for an unrelated GPU (a different, non-guest target ISA) loaded
  // on a non-guest agent must be forwarded verbatim to the original loader. The
  // hook must NOT reinterpret it as the guest ISA, translate it, or run
  // rocjitsu metadata validation on it. Even though the reader's bytes ARE
  // registered (memory-backed), the load is gated out by the non-guest early
  // return before any reader lookup or target detection.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // A minimal, valid AMDGPU ELF for gfx942 (neither the guest gfx950 nor the
  // configured host gfx1201). Content is irrelevant: the load must pass through
  // untouched.
  std::vector<uint8_t> image(sizeof(rocjitsu::Elf64_Ehdr), 0);
  const auto ehdr = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  write_struct(image, 0, ehdr);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      kFakeExecutable, kUnrelatedAgent, reader, nullptr, &loaded);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);

  // The original loader saw the unrelated agent and reader verbatim (no remap to
  // the guest execution host, no translated reader substituted).
  ASSERT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_EQ(g_last_load_agent.handle, kUnrelatedAgent.handle);
  EXPECT_EQ(g_last_load_reader.handle, reader.handle);
}

TEST(HsaHooksUnitTest, LoadOnUnrelatedAgentForwardsMalformedMetadataImageUnchanged) {
  // A non-guest load must reach the original loader even if its bytes are not a
  // parseable code object / carry malformed rocjitsu metadata: the hook only
  // validates the guest's own loads. Garbage bytes that would fail
  // parse_virtual_lds_hook_metadata must still pass through unchanged.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  const std::vector<uint8_t> image(64, 0xAB); // not a valid ELF
  hsa_code_object_reader_t reader{};
  ASSERT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);

  hsa_loaded_code_object_t loaded{};
  const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
      kFakeExecutable, kUnrelatedAgent, reader, nullptr, &loaded);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);

  ASSERT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_EQ(g_last_load_agent.handle, kUnrelatedAgent.handle);
  EXPECT_EQ(g_last_load_reader.handle, reader.handle);
}

TEST(HsaHooksUnitDeathTest, InterceptGuestUnregisteredOversizedDispatchAborts) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // fork-based death test: the guest queue's intercept handler runs synchronously
  // (no scanner jthread on the intercept path), so aborting inside it is safe to
  // observe. A dispatch on the GUEST queue whose kernel object has no virtual-LDS
  // metadata and whose group segment exceeds the host LDS limit has no sidecar to
  // fall back to, so it must fail closed rather than submit a host-faulting launch.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        reset_pool_blocker(false);
        reset_queue_fakes();
        FakeApiTable api;
        api.amd.hsa_amd_queue_intercept_create_fn = fake_amd_queue_intercept_create;
        api.amd.hsa_amd_queue_intercept_register_fn = fake_amd_queue_intercept_register;
        InstalledHook hook(api);

        hsa_queue_t *queue = nullptr;
        api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                     &queue);

        kernel_descriptor_t descriptor{};
        descriptor.group_segment_fixed_size = 96 * 1024;

        hsa_kernel_dispatch_packet_t packet{};
        packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
        packet.kernel_object = reinterpret_cast<uintptr_t>(&descriptor);
        packet.group_segment_size = 96 * 1024; // exceeds the guest 64 KiB limit
        packet.workgroup_size_x = 256;
        packet.workgroup_size_y = 1;
        packet.workgroup_size_z = 1;
        packet.grid_size_x = 256;
        packet.grid_size_y = 1;
        packet.grid_size_z = 1;

        constexpr uint64_t kPacketIndex = 2;
        g_fake_queue_packets[kPacketIndex] = packet;
        g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                                 fake_intercept_packet_writer);
      },
      "no virtual-LDS variant but its group segment exceeds the host LDS limit");
}

TEST(HsaHooksUnitTest, UntrackedSignalStoreScreleaseIsForwarded) {
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_signal_store_screlease_fn, fake_signal_store_screlease);

  api.core.hsa_signal_store_screlease_fn(hsa_signal_t{999}, 42);

  EXPECT_EQ(g_fake_signal_store_screlease_calls, 1);
  EXPECT_EQ(g_last_signal_store_signal.handle, 999u);
  EXPECT_EQ(g_last_signal_store_value, 42);
}

} // namespace
