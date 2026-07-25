// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/kmd/linux/rpc.h"
#include "scoped_temp.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

extern "C" bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char *const *failed_tool_names);
extern "C" void OnUnload();
extern "C" size_t rj_test_retained_translation_count(uint64_t executable_handle);
extern "C" void rj_test_retain_translation(uint64_t executable_handle);
extern "C" void rj_test_record_deferred(uint64_t executable_handle, uint64_t agent_handle,
                                        const void *bytes, size_t size);
extern "C" size_t rj_test_deferred_count(uint64_t executable_handle);
extern "C" uint64_t rj_test_create_hidden_child(uint64_t parent_executable_handle,
                                                uint64_t agent_handle, const void *bytes,
                                                size_t size);
extern "C" unsigned rj_test_single_flight_producer_runs(unsigned thread_count, bool succeed);
extern "C" uint64_t rj_test_lazy_rewrite_kernel_object(uint64_t proxy_ko, uint64_t child_ko);

namespace {

constexpr hsa_agent_t kGuestAgent{1};
constexpr hsa_agent_t kHostAgent{2};
// An agent that is neither the guest nor the guest's execution host. Its queues
// are tracked for doorbell forwarding but never rewritten, and host_lds_bytes
// (derived from the guest target arch) does not apply to them.
constexpr hsa_agent_t kUnrelatedAgent{3};
// A gfx1250 agent used to exercise auto-A0 detection. Its ASIC revision is
// controlled per test via g_fake_gfx1250_asic_revision.
constexpr hsa_agent_t kGfx1250Agent{4};
constexpr hsa_isa_t kGuestIsa{950};
constexpr hsa_isa_t kHostIsa{1201};
constexpr hsa_isa_t kGfx1250Isa{1250};

// ASIC revision reported for kGfx1250Agent by fake_agent_get_info. 0 => A0.
uint32_t g_fake_gfx1250_asic_revision = 0;
// When true, fake_agent_get_info fails the ASIC_REVISION query for kGfx1250Agent,
// simulating a gfx1250 device whose stepping cannot be read (unknown revision).
bool g_fake_gfx1250_revision_query_fails = false;
// Controls how the ISA query fails for kGfx1250Agent, to exercise the
// kUnknownTarget path (ISA unreadable => might be A0 => fail closed).
enum class IsaQueryFailure {
  kNone,       ///< ISA query succeeds normally.
  kIterate,    ///< hsa_agent_iterate_isas returns an error.
  kNameLength, ///< HSA_ISA_INFO_NAME_LENGTH lookup errors inside the callback.
  kName,       ///< HSA_ISA_INFO_NAME lookup errors inside the callback.
};
IsaQueryFailure g_fake_gfx1250_isa_query_failure = IsaQueryFailure::kNone;
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
uint64_t g_next_memory_reader_handle = 2;
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
// When set, fake_executable_load_agent_code_object returns this error instead of
// success, to exercise the auto-A0 retention rollback on a failed lower load.
hsa_status_t g_fake_load_agent_status = HSA_STATUS_SUCCESS;
// Maps a memory-reader handle to the exact bytes it was created over, so a test
// can recover the image the loader saw for a substituted (snapshot) reader and
// assert byte-equality with the source. Populated by the fake memory-reader
// constructor; the load fake copies the matching bytes into g_last_load_bytes.
std::unordered_map<uint64_t, std::vector<uint8_t>> g_memory_reader_bytes;
std::vector<uint8_t> g_last_load_bytes;
// Records forwarding through the two agent-less/deprecated load entries, so a
// test can prove they forwarded (no A0 target) vs. were refused (fail-closed).
int g_fake_load_program_calls = 0;
int g_fake_load_deprecated_calls = 0;
constexpr hsa_executable_t kFakeExecutable{123};
constexpr hsa_executable_symbol_t kFakeKernelSymbol{500};

const char *isa_name(hsa_isa_t isa) {
  if (isa.handle == kGuestIsa.handle)
    return "amdgcn-amd-amdhsa--gfx950";
  if (isa.handle == kHostIsa.handle)
    return "amdgcn-amd-amdhsa--gfx1201";
  if (isa.handle == kGfx1250Isa.handle)
    return "amdgcn-amd-amdhsa--gfx1250";
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

// Enumeration that includes the gfx1250 agent, used to exercise the agent-less
// load paths' "an A0 (or possible-A0) target exists" fail-closed branch.
hsa_status_t HSA_API fake_iterate_agents_with_gfx1250(hsa_status_t (*callback)(hsa_agent_t, void *),
                                                      void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = callback(kHostAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kGfx1250Agent, data);
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
    if (agent.handle == kGfx1250Agent.handle)
      *static_cast<hsa_isa_t *>(value) = kGfx1250Isa;
    else
      *static_cast<hsa_isa_t *>(value) = agent.handle == kGuestAgent.handle ? kGuestIsa : kHostIsa;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID)) {
    *static_cast<uint32_t *>(value) =
        agent.handle == kGuestAgent.handle ? kGuestNodeId : kHostNodeId;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION)) {
    if (agent.handle != kGfx1250Agent.handle)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    if (g_fake_gfx1250_revision_query_fails)
      return HSA_STATUS_ERROR; // simulate an unreadable stepping (unknown revision)
    *static_cast<uint32_t *>(value) = g_fake_gfx1250_asic_revision;
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
  if (agent.handle == kGfx1250Agent.handle) {
    if (g_fake_gfx1250_isa_query_failure == IsaQueryFailure::kIterate)
      return HSA_STATUS_ERROR; // iteration itself errors -> unknown target
    return callback(kGfx1250Isa, data);
  }
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t HSA_API fake_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // Simulate a name-query failure for the gfx1250 ISA to exercise kUnknownTarget.
  if (isa.handle == kGfx1250Isa.handle) {
    if (attribute == HSA_ISA_INFO_NAME_LENGTH &&
        g_fake_gfx1250_isa_query_failure == IsaQueryFailure::kNameLength)
      return HSA_STATUS_ERROR;
    if (attribute == HSA_ISA_INFO_NAME &&
        g_fake_gfx1250_isa_query_failure == IsaQueryFailure::kName)
      return HSA_STATUS_ERROR;
  }

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
    const void *code_object, size_t size, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = g_next_memory_reader_handle++;
  // Snapshot the exact bytes this reader was created over so a test can recover
  // the image the loader will see for a substituted (snapshot) reader.
  if (code_object != nullptr)
    g_memory_reader_bytes[code_object_reader->handle].assign(
        static_cast<const uint8_t *>(code_object),
        static_cast<const uint8_t *>(code_object) + size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_destroy(hsa_code_object_reader_t) {
  return HSA_STATUS_SUCCESS;
}

// Records what the fake vendor reader constructor last saw, so a test can assert
// the hook forwarded the fd/offset/size to the real constructor.
struct FakeVendorReaderCall {
  int count = 0;
  hsa_file_t file = -1;
  size_t offset = 0;
  size_t size = 0;
};
FakeVendorReaderCall g_fake_vendor_reader_call;

hsa_status_t
fake_vendor_reader_create_from_file_with_offset_size(hsa_file_t file, size_t offset, size_t size,
                                                     hsa_code_object_reader_t *code_object_reader) {
  ++g_fake_vendor_reader_call.count;
  g_fake_vendor_reader_call.file = file;
  g_fake_vendor_reader_call.offset = offset;
  g_fake_vendor_reader_call.size = size;
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = g_next_memory_reader_handle++;
  return HSA_STATUS_SUCCESS;
}

// A fake AMD loader vendor table with only the reader constructor populated.
hsa_ven_amd_loader_1_03_pfn_t g_fake_vendor_table = {
    nullptr, nullptr, nullptr,
    nullptr, nullptr, fake_vendor_reader_create_from_file_with_offset_size,
    nullptr};

hsa_status_t HSA_API fake_system_get_major_extension_table(uint16_t extension,
                                                           uint16_t version_major,
                                                           size_t table_length, void *table) {
  if (extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::memcpy(table, &g_fake_vendor_table,
              std::min(table_length, sizeof(hsa_ven_amd_loader_1_03_pfn_t)));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_destroy(hsa_executable_t) { return HSA_STATUS_SUCCESS; }

// Hidden-child-executable fakes (Step 3). A test drives create/load/freeze via
// rj_test_create_hidden_child and asserts the sequence and rollback.
uint64_t g_next_child_executable_handle = 900;
int g_fake_create_alt_calls = 0;
int g_fake_freeze_calls = 0;
hsa_profile_t g_fake_parent_profile = HSA_PROFILE_FULL;
hsa_default_float_rounding_mode_t g_fake_parent_rounding = HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
// Records the profile/rounding create_alt was called with, so a test can assert
// the child was created to match the parent's queried shape.
hsa_profile_t g_last_create_alt_profile = HSA_PROFILE_BASE;
hsa_default_float_rounding_mode_t g_last_create_alt_rounding = HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO;
// When set, the corresponding fake fails so a test can exercise child rollback.
hsa_status_t g_fake_create_alt_status = HSA_STATUS_SUCCESS;
hsa_status_t g_fake_freeze_status = HSA_STATUS_SUCCESS;
hsa_status_t g_fake_get_info_status = HSA_STATUS_SUCCESS;

hsa_status_t HSA_API fake_executable_get_info(hsa_executable_t, hsa_executable_info_t attribute,
                                              void *value) {
  if (g_fake_get_info_status != HSA_STATUS_SUCCESS)
    return g_fake_get_info_status;
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
  case HSA_EXECUTABLE_INFO_PROFILE:
    *static_cast<hsa_profile_t *>(value) = g_fake_parent_profile;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
    *static_cast<hsa_default_float_rounding_mode_t *>(value) = g_fake_parent_rounding;
    return HSA_STATUS_SUCCESS;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

hsa_status_t HSA_API fake_executable_create_alt(hsa_profile_t profile,
                                                hsa_default_float_rounding_mode_t rounding,
                                                const char *, hsa_executable_t *executable) {
  ++g_fake_create_alt_calls;
  g_last_create_alt_profile = profile;
  g_last_create_alt_rounding = rounding;
  if (g_fake_create_alt_status != HSA_STATUS_SUCCESS)
    return g_fake_create_alt_status;
  if (executable == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  executable->handle = g_next_child_executable_handle++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_freeze(hsa_executable_t, const char *) {
  ++g_fake_freeze_calls;
  return g_fake_freeze_status;
}

hsa_status_t HSA_API fake_executable_load_agent_code_object(
    hsa_executable_t, hsa_agent_t agent, hsa_code_object_reader_t reader, const char *,
    hsa_loaded_code_object_t *loaded_code_object) {
  ++g_fake_load_agent_calls;
  g_last_load_agent = agent;
  g_last_load_reader = reader;
  // Recover the bytes this reader was created over (for memory/snapshot readers)
  // so a test can assert the loader saw exactly the source image.
  if (auto it = g_memory_reader_bytes.find(reader.handle); it != g_memory_reader_bytes.end())
    g_last_load_bytes = it->second;
  else
    g_last_load_bytes.clear();
  if (g_fake_load_agent_status != HSA_STATUS_SUCCESS)
    return g_fake_load_agent_status; // simulate a lower-loader failure
  if (loaded_code_object != nullptr)
    loaded_code_object->handle = 77;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_load_program_code_object(hsa_executable_t,
                                                              hsa_code_object_reader_t,
                                                              const char *,
                                                              hsa_loaded_code_object_t *loaded) {
  ++g_fake_load_program_calls;
  if (loaded != nullptr)
    loaded->handle = 88;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_load_code_object(hsa_executable_t, hsa_agent_t,
                                                      hsa_code_object_t, const char *) {
  ++g_fake_load_deprecated_calls;
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
    core.hsa_system_get_major_extension_table_fn = fake_system_get_major_extension_table;
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
    core.hsa_executable_load_agent_code_object_fn = fake_executable_load_agent_code_object;
    core.hsa_executable_load_program_code_object_fn = fake_executable_load_program_code_object;
    core.hsa_executable_load_code_object_fn = fake_executable_load_code_object;
    core.hsa_executable_destroy_fn = fake_executable_destroy;
    core.hsa_executable_create_alt_fn = fake_executable_create_alt;
    core.hsa_executable_freeze_fn = fake_executable_freeze;
    core.hsa_executable_get_info_fn = fake_executable_get_info;
    core.hsa_executable_get_symbol_by_name_fn = fake_executable_get_symbol_by_name;
    core.hsa_executable_iterate_agent_symbols_fn = fake_executable_iterate_agent_symbols;
    core.hsa_executable_symbol_get_info_fn = fake_executable_symbol_get_info;
    amd.hsa_amd_agent_iterate_memory_pools_fn = fake_amd_agent_iterate_memory_pools;
    amd.hsa_amd_memory_pool_get_info_fn = fake_amd_memory_pool_get_info;
    amd.hsa_amd_agent_memory_pool_get_info_fn = fake_amd_agent_memory_pool_get_info;
    amd.hsa_amd_memory_pool_allocate_fn = fake_amd_memory_pool_allocate;
    amd.hsa_amd_memory_async_batch_copy_fn = fake_amd_memory_async_batch_copy;
    amd.hsa_amd_memory_lock_fn = fake_amd_memory_lock;
    amd.hsa_amd_memory_lock_to_pool_fn = fake_amd_memory_lock_to_pool;
    amd.hsa_amd_pointer_info_fn = fake_amd_pointer_info;
    amd.hsa_amd_vmem_set_access_fn = fake_amd_vmem_set_access;
    amd.hsa_amd_memory_pool_free_fn = fake_amd_memory_pool_free;
    amd.hsa_amd_agents_allow_access_fn = fake_amd_agents_allow_access;
  }
};

void write_runtime_config_path(const std::string &runtime_dir) {
  setenv("ROCJITSU_RUNTIME_DIR", runtime_dir.c_str(), 1);

  std::ofstream config_path(rocjitsu::rpc_default_config_file_path());
  config_path << RJ_HOOK_UNIT_CONFIG_PATH << '\n';
}

class InstalledHook {
public:
  explicit InstalledHook(FakeApiTable &api) : runtime_dir_("rocjitsu-hsa-hooks-unit-") {
    OnUnload();
    write_runtime_config_path(runtime_dir_.path());
    installed_ = OnLoad(&api.table, 0, 0, nullptr);
  }
  ~InstalledHook() { OnUnload(); }

  [[nodiscard]] bool installed() const { return installed_; }

private:
  rocjitsu::test::ScopedTempDirectory runtime_dir_;
  bool installed_ = false;
};

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

void reset_queue_fakes() {
  g_next_memory_reader_handle = 2;
  g_memory_reader_bytes.clear();
  g_last_load_bytes.clear();
  g_fake_gfx1250_isa_query_failure = IsaQueryFailure::kNone;
  g_fake_load_agent_status = HSA_STATUS_SUCCESS;
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

// Build a minimal, AmdGpuCodeObject-valid gfx1250 ELF carrying a single named
// PROGBITS section with @p section_bytes. Used to construct objects that carry an
// arbitrary (e.g. retired/forged) metadata section without depending on any
// producer that would legitimately emit it.
std::vector<uint8_t> make_gfx1250_elf_with_section(std::string_view section_name,
                                                   std::span<const uint8_t> section_bytes) {
  std::string shstr(1, '\0');
  const uint32_t section_name_offset = static_cast<uint32_t>(shstr.size());
  shstr.append(section_name);
  shstr.push_back('\0');
  const uint32_t shstrtab_name = static_cast<uint32_t>(shstr.size());
  shstr.append(".shstrtab");
  shstr.push_back('\0');

  auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  std::vector<uint8_t> image(sizeof(header));
  const size_t payload_offset = image.size();
  write_bytes(image, payload_offset, section_bytes.data(), section_bytes.size());
  const size_t shstrtab_offset = image.size();
  write_bytes(image, shstrtab_offset, shstr.data(), shstr.size());

  const size_t section_header_offset = align_up(image.size(), alignof(rocjitsu::Elf64_Shdr));
  image.resize(section_header_offset);

  std::array<rocjitsu::Elf64_Shdr, 3> sections{}; // [0]=null, [1]=payload, [2]=shstrtab
  sections[1].sh_name = section_name_offset;
  sections[1].sh_type = rocjitsu::SHT_PROGBITS;
  sections[1].sh_offset = payload_offset;
  sections[1].sh_size = section_bytes.size();
  sections[2].sh_name = shstrtab_name;
  sections[2].sh_type = rocjitsu::SHT_STRTAB;
  sections[2].sh_offset = shstrtab_offset;
  sections[2].sh_size = shstr.size();
  write_bytes(image, section_header_offset, sections.data(), sizeof(sections));

  header.e_shoff = section_header_offset;
  header.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  header.e_shnum = sections.size();
  header.e_shstrndx = 2;
  write_struct(image, 0, header);
  return image;
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

// Point every config-file tier at a fresh empty directory so parse_config()
// sees no config and selects the default auto-A0 mode.
void clear_runtime_config_path() {
  std::filesystem::path runtime_dir =
      std::filesystem::temp_directory_path() /
      ("rocjitsu-hsa-hooks-unit-noconfig-" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(runtime_dir);
  std::filesystem::create_directories(runtime_dir);
  setenv("ROCJITSU_RUNTIME_DIR", runtime_dir.c_str(), 1);
  unsetenv("ROCJITSU_INVOCATION_DIR");
}

// With no config file present, OnLoad installs in the default auto-A0 mode
// (translate gfx1250 B0 -> A0), whereas the config-driven simulation path
// requires a file. Previously a missing config was a hard failure.
TEST(HsaHooksUnitTest, InstallsInAutoA0ModeWhenNoConfigPresent) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  OnUnload();
  clear_runtime_config_path();

  FakeApiTable api;
  auto *original_load = api.core.hsa_executable_load_agent_code_object_fn;
  EXPECT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  // The load-side wrappers are installed, so the table is patched.
  EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn, original_load);
  OnUnload();
}

// Auto-A0 installs a narrow manifest: load/capture entries are patched, but the
// simulation-only remap surface (queue/signal) is left as the runtime's own
// entries. Notably agent_get_info is NOT patched: A0/B0 are silicon steppings of
// the same gfx1250 ISA, so there is no agent identity to present -- the hook only
// *detects* A0 (via the saved ASIC-revision query) to arm translation.
TEST(HsaHooksUnitTest, AutoA0InstallsNarrowManifest) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));

  // Patched in auto-A0.
  EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn,
            fake_executable_load_agent_code_object);
  EXPECT_NE(api.core.hsa_system_get_major_extension_table_fn,
            fake_system_get_major_extension_table);
  // NOT patched in auto-A0. agent_get_info is saved-only (used for detection, not
  // presentation). shut_down is a pure passthrough. The simulation-only
  // queue/signal remap surface is also untouched.
  EXPECT_EQ(api.core.hsa_agent_get_info_fn, fake_agent_get_info);
  EXPECT_EQ(api.core.hsa_shut_down_fn, fake_shut_down);
  EXPECT_EQ(api.core.hsa_queue_create_fn, fake_queue_create);
  EXPECT_EQ(api.core.hsa_signal_store_relaxed_fn, fake_signal_store_relaxed);
  OnUnload();
}

// Simulation installs the full manifest, including the queue/signal surface that
// auto-A0 omits.
TEST(HsaHooksUnitTest, SimulationInstallsFullManifest) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  write_runtime_config_path(); // simulation mode

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  EXPECT_NE(api.core.hsa_queue_create_fn, fake_queue_create);
  EXPECT_NE(api.core.hsa_signal_store_relaxed_fn, fake_signal_store_relaxed);
  // agent_get_info is saved-only in every mode (never wrapped): the hook reads
  // the saved original for detection but does not present a synthetic identity.
  EXPECT_EQ(api.core.hsa_agent_get_info_fn, fake_agent_get_info);
}

// Translated storage retained for an executable is released when that
// executable is destroyed, not before.
TEST(HsaHooksUnitTest, ReleasesTranslatedStorageOnExecutableDestroy) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  ASSERT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
  rj_test_retain_translation(kFakeExecutable.handle);
  rj_test_retain_translation(kFakeExecutable.handle);
  EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 2u);

  // Destroying the executable through the patched wrapper releases its storage.
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
}

// Retained translated storage is dropped when the hook is uninstalled.
TEST(HsaHooksUnitTest, ReleasesTranslatedStorageOnUnload) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  {
    InstalledHook hook(api);
    ASSERT_TRUE(hook.installed());
    rj_test_retain_translation(kFakeExecutable.handle);
    EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 1u);
  }
  // InstalledHook's destructor called OnUnload, which clears the registry.
  EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
}

// Reset the hidden-child-executable fakes to their success defaults.
void reset_child_executable_fakes() {
  g_fake_create_alt_calls = 0;
  g_fake_freeze_calls = 0;
  g_fake_create_alt_status = HSA_STATUS_SUCCESS;
  g_fake_freeze_status = HSA_STATUS_SUCCESS;
  g_fake_get_info_status = HSA_STATUS_SUCCESS;
  g_fake_parent_profile = HSA_PROFILE_FULL;
  g_fake_parent_rounding = HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
}

// A deferred object recorded for an executable is counted and then dropped when
// the executable is destroyed through the patched wrapper (Step 3 registry).
TEST(HsaHooksUnitTest, DeferredObjectRecordedAndDroppedOnDestroy) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  ASSERT_EQ(rj_test_deferred_count(kFakeExecutable.handle), 0u);
  const std::vector<uint8_t> source{0x7f, 'E', 'L', 'F'};
  rj_test_record_deferred(kFakeExecutable.handle, kGuestAgent.handle, source.data(), source.size());
  rj_test_record_deferred(kFakeExecutable.handle, kGuestAgent.handle, source.data(), source.size());
  EXPECT_EQ(rj_test_deferred_count(kFakeExecutable.handle), 2u);

  // Destroying the executable through the patched wrapper drops its deferred
  // records (and, in later steps, tears down their hidden children first).
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_deferred_count(kFakeExecutable.handle), 0u);
}

// Deferred records are cleared when the hook is uninstalled.
TEST(HsaHooksUnitTest, ClearsDeferredObjectsOnUnload) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  {
    InstalledHook hook(api);
    ASSERT_TRUE(hook.installed());
    const std::vector<uint8_t> source{0x7f, 'E', 'L', 'F'};
    rj_test_record_deferred(kFakeExecutable.handle, kGuestAgent.handle, source.data(),
                            source.size());
    EXPECT_EQ(rj_test_deferred_count(kFakeExecutable.handle), 1u);
  }
  EXPECT_EQ(rj_test_deferred_count(kFakeExecutable.handle), 0u);
}

// create_hidden_child_executable queries the parent's profile/rounding, creates a
// child with those, loads the translated bytes, and freezes it -- returning the
// child handle.
TEST(HsaHooksUnitTest, CreatesHiddenChildExecutableMatchingParentShape) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  reset_child_executable_fakes();
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // Parent reports BASE profile + near rounding; the child must be created to match.
  g_fake_parent_profile = HSA_PROFILE_BASE;
  g_fake_parent_rounding = HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;

  const std::vector<uint8_t> translated{0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01};
  const uint64_t child = rj_test_create_hidden_child(kFakeExecutable.handle, kGuestAgent.handle,
                                                     translated.data(), translated.size());
  EXPECT_NE(child, 0u) << "hidden child executable was not created";
  EXPECT_EQ(g_fake_create_alt_calls, 1);
  EXPECT_EQ(g_fake_freeze_calls, 1);
  EXPECT_EQ(g_last_create_alt_profile, HSA_PROFILE_BASE);
  EXPECT_EQ(g_last_create_alt_rounding, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR);
  // The translated bytes reached the loader on the same agent.
  EXPECT_EQ(g_last_load_bytes, translated);
  EXPECT_EQ(g_last_load_agent.handle, kGuestAgent.handle);
}

// A freeze failure rolls the child back: create_hidden_child_executable returns
// 0 and the partially-created child is destroyed (no leak of a live executable).
TEST(HsaHooksUnitTest, HiddenChildRollsBackOnFreezeFailure) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  reset_child_executable_fakes();
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  g_fake_freeze_status = HSA_STATUS_ERROR; // freeze fails after create + load succeed

  const std::vector<uint8_t> translated{0x7f, 'E', 'L', 'F'};
  const uint64_t child = rj_test_create_hidden_child(kFakeExecutable.handle, kGuestAgent.handle,
                                                     translated.data(), translated.size());
  EXPECT_EQ(child, 0u) << "child create should fail closed when freeze fails";
  EXPECT_EQ(g_fake_create_alt_calls, 1);
  EXPECT_EQ(g_fake_freeze_calls, 1);
}

// Single-flight (§14.4): under many threads racing the first dispatch of one
// deferred object, the translation producer runs EXACTLY once and every thread
// observes the same successful result. Bit 16 of the return encodes "all threads
// agreed"; the low bits are the producer run count.
TEST(HsaHooksUnitTest, SingleFlightRunsProducerExactlyOnceOnSuccess) {
  const unsigned result =
      rj_test_single_flight_producer_runs(/*thread_count=*/16, /*succeed=*/true);
  EXPECT_EQ(result & 0xFFFFu, 1u) << "translation producer ran more than once under contention";
  EXPECT_NE(result & 0x10000u, 0u) << "not all threads observed the same ready result";
}

// A failed translation is permanent and single-flight: the producer runs once,
// every racing thread observes failure, and a later attempt does not re-run it.
TEST(HsaHooksUnitTest, SingleFlightFailureIsPermanentAndOnce) {
  const unsigned result =
      rj_test_single_flight_producer_runs(/*thread_count=*/16, /*succeed=*/false);
  EXPECT_EQ(result & 0xFFFFu, 1u) << "failed producer ran more than once (should be permanent)";
  EXPECT_NE(result & 0x10000u, 0u) << "not all threads observed the same failed result";
}

// The dispatch-seam rewrite mechanic (§14.3): a kernel-dispatch packet carrying a
// tracked, translated proxy kernel_object has it swapped for the child's
// kernel_object in place. (Full translate/load producer is proven at full stack.)
TEST(HsaHooksUnitTest, LazyDispatchRewritesProxyKernelObject) {
  constexpr uint64_t kProxyKo = 0x5000;
  constexpr uint64_t kChildKo = 0x9000;
  const uint64_t rewritten = rj_test_lazy_rewrite_kernel_object(kProxyKo, kChildKo);
  EXPECT_EQ(rewritten, kChildKo) << "proxy kernel_object was not rewritten to the translated child";
}

// The rewrite is a no-op when the translated kernel_object equals the proxy's
// (the identity guard), so a packet is never needlessly re-published.
TEST(HsaHooksUnitTest, LazyDispatchIdentityMapLeavesPacketUnchanged) {
  const uint64_t unchanged = rj_test_lazy_rewrite_kernel_object(0x7777, 0x7777);
  EXPECT_EQ(unchanged, 0x7777u) << "identity map should leave the kernel_object unchanged";
}

// Register a code object image and return its reader handle.
hsa_code_object_reader_t register_code_object(FakeApiTable &api,
                                              const std::vector<uint8_t> &image) {
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(
      api.core.hsa_code_object_reader_create_from_memory_fn(image.data(), image.size(), &reader),
      HSA_STATUS_SUCCESS);
  return reader;
}

// In auto-A0 mode, a load on the A0 gfx1250 agent whose bytes were never
// captured is refused (fail-closed): the hook will not forward an untranslated
// B0 object to the loader.
TEST(HsaHooksUnitTest, AutoA0LoadWithoutCapturedBytesFailsClosed) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  // A reader that was never registered through create_from_memory has no bytes.
  hsa_code_object_reader_t unknown_reader{0xBEEF};
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              unknown_reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER);
  EXPECT_EQ(g_fake_load_agent_calls, 0);
  OnUnload();
}

// In auto-A0 mode a non-gfx1250 object on the A0 agent is not ours to translate,
// but it is still loaded from an authoritative hook-owned snapshot rather than
// the caller's mutable reader: the loader sees a *different* reader whose bytes
// equal the source image.
TEST(HsaHooksUnitTest, AutoA0ForwardsNonGfx1250Object) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  hsa_code_object_reader_t reader = register_code_object(api, image);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  // Loaded from the snapshot: a hook-owned reader (not the caller's) whose bytes
  // are byte-identical to the source image.
  EXPECT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_NE(g_last_load_reader.handle, reader.handle);
  EXPECT_EQ(g_last_load_bytes, image);
  OnUnload();
}

// In auto-A0 mode a load onto a gfx1250 agent whose ASIC revision cannot be read
// is refused (fail-closed): the stepping is unknown, the agent might be A0, and
// forwarding would risk running raw B0 text on A0 silicon. The object is not
// handed to the loader.
TEST(HsaHooksUnitTest, AutoA0LoadWithUnknownRevisionFailsClosed) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_revision_query_fails = true; // gfx1250, but stepping unreadable
  g_fake_load_agent_calls = 0;

  // Valid gfx1250 bytes so the refusal is attributable to the unknown stepping,
  // not to missing/invalid source bytes.
  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  hsa_code_object_reader_t reader = register_code_object(api, image);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_AGENT);
  EXPECT_EQ(g_fake_load_agent_calls, 0);

  g_fake_gfx1250_revision_query_fails = false;
  OnUnload();
}

// In auto-A0 mode, an agent whose ISA query *fails* (iteration or name lookup
// errors) is not a confirmed non-gfx1250 agent -- it might be A0. Such a load
// must fail closed, never forward the object unchanged. Covers all three ISA
// query failure points.
TEST(HsaHooksUnitTest, AutoA0LoadWithUnreadableIsaFailsClosed) {
  struct Case {
    const char *name;
    IsaQueryFailure failure;
  };
  const Case cases[] = {
      {"iterate", IsaQueryFailure::kIterate},
      {"name-length", IsaQueryFailure::kNameLength},
      {"name", IsaQueryFailure::kName},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    reset_pool_blocker(false);
    reset_agent_blocker(false);
    reset_queue_fakes();
    OnUnload();
    clear_runtime_config_path(); // auto-A0 mode

    FakeApiTable api;
    ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
    g_fake_gfx1250_isa_query_failure = c.failure; // ISA unreadable -> unknown target
    g_fake_load_agent_calls = 0;

    // Valid gfx1250 bytes: the refusal is attributable to the unreadable ISA, not
    // to missing/invalid source bytes.
    const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
    std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                               reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
    hsa_code_object_reader_t reader = register_code_object(api, image);

    EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                                reader, nullptr, nullptr),
              HSA_STATUS_ERROR_INVALID_AGENT);
    EXPECT_EQ(g_fake_load_agent_calls, 0);
    OnUnload();
  }
}

// In auto-A0 mode a confirmed non-A0 gfx1250 agent (nonzero revision, e.g. real
// B0) is not a translation target: its loads forward to the loader unchanged.
TEST(HsaHooksUnitTest, AutoA0ForwardsConfirmedNonA0Gfx1250Agent) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 7; // gfx1250 but not A0
  g_fake_load_agent_calls = 0;

  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  hsa_code_object_reader_t reader = register_code_object(api, image);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  // Forwarded verbatim: the loader saw the original reader, not a translated one.
  EXPECT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_EQ(g_last_load_reader.handle, reader.handle);

  g_fake_gfx1250_asic_revision = 0;
  OnUnload();
}

// TOCTOU: on the A0 agent the hook must load the bytes it inspected, not whatever
// the application mutates the buffer to after reader creation. Register a memory
// reader, corrupt the caller's buffer before the load, and assert the loader
// still sees the ORIGINAL image (the hook's immutable snapshot).
TEST(HsaHooksUnitTest, AutoA0SnapshotIsImmutableToPostCaptureMutation) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  // A non-gfx1250 object takes the forward-via-snapshot path on the A0 agent.
  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  const std::vector<uint8_t> original = image;
  hsa_code_object_reader_t reader = register_code_object(api, image);

  // Application mutates its buffer after reader creation, before load.
  std::fill(image.begin(), image.end(), uint8_t{0xCC});

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_NE(g_last_load_reader.handle, reader.handle);
  EXPECT_EQ(g_last_load_bytes, original); // the snapshot, not the mutated buffer
  OnUnload();
}

// The authoritative snapshot for a forwarded auto-A0 load is retained for the
// executable's lifetime (ROCR aliases it until executable destroy) and released
// when the executable is destroyed.
TEST(HsaHooksUnitTest, AutoA0SnapshotRetainedUntilExecutableDestroy) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;

  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  hsa_code_object_reader_t reader = register_code_object(api, image);

  ASSERT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  // The snapshot backing ROCR's alias is retained at executable scope.
  EXPECT_GE(rj_test_retained_translation_count(kFakeExecutable.handle), 1u);
  EXPECT_EQ(api.core.hsa_executable_destroy_fn(kFakeExecutable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
  OnUnload();
}

// The snapshot retention is reserved before the lower loader commits, then rolled
// back if that load fails -- so a failed forwarded load leaves no retained
// storage behind (no leak) and reports the failure.
TEST(HsaHooksUnitTest, AutoA0SnapshotRetentionRolledBackOnLoadFailure) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_status = HSA_STATUS_ERROR_INVALID_ISA; // lower load fails

  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  hsa_code_object_reader_t reader = register_code_object(api, image);

  ASSERT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_ISA);
  // Rolled back: nothing retained for the executable after a failed load.
  EXPECT_EQ(rj_test_retained_translation_count(kFakeExecutable.handle), 0u);
  OnUnload();
}

// Regression for the removed provenance-note bypass: a gfx1250 object on the A0
// agent carrying the retired ".rocjitsu.dbt_provenance" section with a matching
// legacy B0->A0 payload must NOT be forwarded unchanged. Such a note was a
// forgeable "already translated, load as-is" credential; the object must now go
// through translation like any other gfx1250 object (and here, being a
// non-dispatchable header-only ELF, fail closed) -- never reach the loader raw.
TEST(HsaHooksUnitTest, AutoA0DoesNotHonorRetiredProvenanceSection) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  // The retired section name and a byte-for-byte valid legacy payload: magic
  // "RJPROV1\0", wire version 1, input revision 2 (Gfx1250B0), output revision 1
  // (Gfx1250A0), reserved 0 -- exactly what the deleted classifier matched.
  static constexpr std::string_view kRetiredSectionName = ".rocjitsu.dbt_provenance";
  const std::array<uint8_t, 24> legacy_note = {'R', 'J', 'P', 'R', 'O', 'V', '1', '\0', // magic
                                               1,   0,   0,   0,  // version = 1
                                               2,   0,   0,   0,  // input_revision = Gfx1250B0
                                               1,   0,   0,   0,  // output_revision = Gfx1250A0
                                               0,   0,   0,   0}; // reserved
  const auto image = make_gfx1250_elf_with_section(kRetiredSectionName, legacy_note);
  hsa_code_object_reader_t reader = register_code_object(api, image);

  // No bypass: the object is translated like any gfx1250 input. This header-only
  // ELF is not dispatchable, so translation fails closed -- and crucially the
  // loader is never handed the original (untranslated) object.
  EXPECT_NE(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_load_agent_calls, 0);
  OnUnload();
}

// Agent-less program-code-object loads have no unique translation target. In
// auto-A0 mode, when an A0 (or possible-A0) agent exists in the system, the load
// is refused (fail-closed) rather than forwarded to the loader with a null agent.
TEST(HsaHooksUnitTest, AutoA0ProgramLoadRefusedWhenA0Present) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  api.core.hsa_iterate_agents_fn = fake_iterate_agents_with_gfx1250; // A0 present
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0; // the gfx1250 agent is A0
  g_fake_load_program_calls = 0;

  hsa_code_object_reader_t reader{0x1234};
  EXPECT_EQ(api.core.hsa_executable_load_program_code_object_fn(kFakeExecutable, reader, nullptr,
                                                                nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(g_fake_load_program_calls, 0);
  OnUnload();
}

// When no A0 target exists (only host/guest agents), an agent-less program load
// is unambiguous and forwards to the loader unchanged.
TEST(HsaHooksUnitTest, AutoA0ProgramLoadForwardsWhenNoA0) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api; // default fake_iterate_agents enumerates host+guest, no gfx1250
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_load_program_calls = 0;

  hsa_code_object_reader_t reader{0x1234};
  EXPECT_EQ(api.core.hsa_executable_load_program_code_object_fn(kFakeExecutable, reader, nullptr,
                                                                nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_load_program_calls, 1);
  OnUnload();
}

// The deprecated direct load passes a raw code-object handle with no readable
// extent. In auto-A0 mode a load onto an A0 gfx1250 agent is refused (the sized
// ABI cannot be safely translated); a non-A0/non-gfx1250 agent forwards.
TEST(HsaHooksUnitTest, AutoA0DeprecatedLoadRefusedOnA0AgentForwardsOtherwise) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_load_deprecated_calls = 0;

  // A0 gfx1250 agent: refused, not forwarded.
  g_fake_gfx1250_asic_revision = 0;
  hsa_code_object_t code_object{0x9999};
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kFakeExecutable, kGfx1250Agent, code_object,
                                                        nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(g_fake_load_deprecated_calls, 0);

  // A non-gfx1250 agent is not our target: forwarded unchanged.
  EXPECT_EQ(api.core.hsa_executable_load_code_object_fn(kFakeExecutable, kHostAgent, code_object,
                                                        nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_load_deprecated_calls, 1);
  OnUnload();
}

// The two fail-closed load entries are patched in auto-A0 (they are part of the
// narrow manifest); simulation patches them too via the full manifest.
TEST(HsaHooksUnitTest, AutoA0PatchesBypassLoadEntries) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  EXPECT_NE(api.core.hsa_executable_load_program_code_object_fn,
            fake_executable_load_program_code_object);
  EXPECT_NE(api.core.hsa_executable_load_code_object_fn, fake_executable_load_code_object);
  OnUnload();
}

// Consolidated fail-closed coverage matrix for the auto-A0 load routing policy.
//
// The security-relevant property is binary per case: an object is either handed
// to the underlying loader (kForward) or refused before the loader is reached
// (kRefuse). Refusing means no untranslated code reaches the agent. The three
// public load entries are keyed on different dimensions, so each has its own
// table; together they assert every (entry x agent-class [x bytes]) combination
// lands on the intended side of the accept/reject line. The exact error codes
// are pinned by the dedicated per-case tests above; this matrix locks the policy
// against a future edit silently flipping one cell.
TEST(HsaHooksUnitTest, AutoA0FailClosedCoverageMatrix) {
  enum class Agent { kA0, kUnknownRevision, kConfirmedNonA0, kNonGfx1250 };
  enum class Bytes { kValidGfx1250, kNonGfx1250, kMissing };
  // kRefuse: loader never reached. kForwardOriginal: loader sees the caller's own
  // reader verbatim -- only on non-A0/non-gfx1250 targets the hook does not police.
  // kForwardSnapshot: loader sees a *different*, hook-owned reader whose bytes
  // equal the source -- the auto-A0 authoritative-snapshot substitution for a
  // forwarded object on an A0/possible-A0 target. kTranslated: loader sees a
  // different reader holding the *translated* ELF -- the only way gfx1250 B0 bytes
  // may reach an A0 agent.
  enum class Outcome { kRefuse, kForwardOriginal, kForwardSnapshot, kTranslated };

  // Apply an agent class to the fakes and return the agent handle to load onto.
  auto setup_agent = [](Agent a) -> hsa_agent_t {
    g_fake_gfx1250_revision_query_fails = false;
    g_fake_gfx1250_asic_revision = 0;
    switch (a) {
    case Agent::kA0:
      return kGfx1250Agent; // gfx1250, revision 0
    case Agent::kUnknownRevision:
      g_fake_gfx1250_revision_query_fails = true; // gfx1250, unreadable stepping
      return kGfx1250Agent;
    case Agent::kConfirmedNonA0:
      g_fake_gfx1250_asic_revision = 7; // gfx1250, nonzero (e.g. real B0)
      return kGfx1250Agent;
    case Agent::kNonGfx1250:
      return kHostAgent;
    }
    return kHostAgent;
  };

  // --- Agent code-object load (the reader seam): agent-class x bytes. ---
  struct AgentLoadCase {
    const char *name;
    Agent agent;
    Bytes bytes;
    Outcome expect;
  };
  // A0: a gfx1250 object is routed into translation -- a header-only ELF (no
  //     loadable code sections) is rejected by the code-object parser, so the
  //     load fails closed rather than forwarding untranslated B0; a non-gfx1250
  //     object is not ours to translate but still loads from the authoritative
  //     snapshot (kForwardSnapshot, not the caller's mutable reader); missing
  //     bytes refuse. (A *successful* translated load is exercised by the
  //     dedicated per-case tests with real fixtures.)
  // UnknownRevision: refuse everything (stepping might be A0), before byte
  //     inspection -- no untranslated object reaches a possible-A0 agent.
  // ConfirmedNonA0 / NonGfx1250: never a translation target -> always forward the
  //     original untouched (including gfx1250 bytes: e.g. B0-on-B0 stays native).
  const AgentLoadCase agent_cases[] = {
      {"A0+gfx1250", Agent::kA0, Bytes::kValidGfx1250, Outcome::kRefuse},
      {"A0+nonGfx1250", Agent::kA0, Bytes::kNonGfx1250, Outcome::kForwardSnapshot},
      {"A0+missing", Agent::kA0, Bytes::kMissing, Outcome::kRefuse},
      {"unknown+gfx1250", Agent::kUnknownRevision, Bytes::kValidGfx1250, Outcome::kRefuse},
      {"unknown+nonGfx1250", Agent::kUnknownRevision, Bytes::kNonGfx1250, Outcome::kRefuse},
      {"unknown+missing", Agent::kUnknownRevision, Bytes::kMissing, Outcome::kRefuse},
      {"nonA0+gfx1250", Agent::kConfirmedNonA0, Bytes::kValidGfx1250, Outcome::kForwardOriginal},
      {"nonA0+nonGfx1250", Agent::kConfirmedNonA0, Bytes::kNonGfx1250, Outcome::kForwardOriginal},
      {"nonA0+missing", Agent::kConfirmedNonA0, Bytes::kMissing, Outcome::kForwardOriginal},
      {"nonGfx1250+gfx1250", Agent::kNonGfx1250, Bytes::kValidGfx1250, Outcome::kForwardOriginal},
      {"nonGfx1250+nonGfx1250", Agent::kNonGfx1250, Bytes::kNonGfx1250, Outcome::kForwardOriginal},
      {"nonGfx1250+missing", Agent::kNonGfx1250, Bytes::kMissing, Outcome::kForwardOriginal},
  };

  for (const AgentLoadCase &c : agent_cases) {
    SCOPED_TRACE(c.name);
    reset_pool_blocker(false);
    reset_agent_blocker(false);
    reset_queue_fakes();
    OnUnload();
    clear_runtime_config_path(); // auto-A0 mode

    FakeApiTable api;
    ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
    const hsa_agent_t agent = setup_agent(c.agent);
    g_fake_load_agent_calls = 0;
    g_last_load_reader = {};

    std::vector<uint8_t> image;
    hsa_code_object_reader_t reader{0xBEEF}; // unregistered handle for kMissing
    if (c.bytes != Bytes::kMissing) {
      const uint32_t mach = c.bytes == Bytes::kValidGfx1250
                                ? rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250
                                : rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942;
      const auto header = make_amdgpu_elf_header(mach);
      image.assign(reinterpret_cast<const uint8_t *>(&header),
                   reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
      reader = register_code_object(api, image);
    }

    const hsa_status_t status = api.core.hsa_executable_load_agent_code_object_fn(
        kFakeExecutable, agent, reader, nullptr, nullptr);
    switch (c.expect) {
    case Outcome::kRefuse:
      EXPECT_NE(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_agent_calls, 0);
      break;
    case Outcome::kForwardOriginal:
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_agent_calls, 1);
      EXPECT_EQ(g_last_load_reader.handle, reader.handle); // caller's own reader
      break;
    case Outcome::kForwardSnapshot:
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_agent_calls, 1);
      EXPECT_NE(g_last_load_reader.handle, reader.handle); // hook-owned snapshot
      EXPECT_EQ(g_last_load_bytes, image);                 // byte-identical to source
      break;
    case Outcome::kTranslated:
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_agent_calls, 1);
      EXPECT_NE(g_last_load_reader.handle, reader.handle); // a translated reader
      break;
    }
    OnUnload();
  }
  g_fake_gfx1250_revision_query_fails = false;
  g_fake_gfx1250_asic_revision = 0;

  // --- Deprecated direct load (sized handle, no readable extent): agent-class. ---
  // Refuse A0 / unknown-revision (cannot bound-read to translate); forward the
  // confirmed non-A0 and non-gfx1250 agents.
  struct DeprecatedCase {
    const char *name;
    Agent agent;
    Outcome expect;
  };
  const DeprecatedCase deprecated_cases[] = {
      {"A0", Agent::kA0, Outcome::kRefuse},
      {"unknown", Agent::kUnknownRevision, Outcome::kRefuse},
      {"nonA0", Agent::kConfirmedNonA0, Outcome::kForwardOriginal},
      {"nonGfx1250", Agent::kNonGfx1250, Outcome::kForwardOriginal},
  };
  for (const DeprecatedCase &c : deprecated_cases) {
    SCOPED_TRACE(std::string("deprecated:") + c.name);
    reset_pool_blocker(false);
    reset_agent_blocker(false);
    reset_queue_fakes();
    OnUnload();
    clear_runtime_config_path(); // auto-A0 mode

    FakeApiTable api;
    ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
    const hsa_agent_t agent = setup_agent(c.agent);
    g_fake_load_deprecated_calls = 0;

    hsa_code_object_t code_object{0x9999};
    const hsa_status_t status =
        api.core.hsa_executable_load_code_object_fn(kFakeExecutable, agent, code_object, nullptr);
    if (c.expect == Outcome::kForwardOriginal) {
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_deprecated_calls, 1);
    } else {
      EXPECT_NE(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_deprecated_calls, 0);
    }
    OnUnload();
  }
  g_fake_gfx1250_revision_query_fails = false;
  g_fake_gfx1250_asic_revision = 0;

  // --- Agent-less program load: keyed on whether an A0 target exists at all. ---
  // (No per-call agent and no byte inspection: the whole system enumeration is
  // the dimension.) An A0 (or possible-A0) present -> refuse; none -> forward.
  struct ProgramCase {
    const char *name;
    hsa_status_t (*iterate)(hsa_status_t (*)(hsa_agent_t, void *), void *);
    bool a0_present; // when true, the gfx1250 agent it enumerates is A0
    Outcome expect;
  };
  const ProgramCase program_cases[] = {
      {"a0-present", fake_iterate_agents_with_gfx1250, true, Outcome::kRefuse},
      {"no-a0", fake_iterate_agents, false, Outcome::kForwardOriginal},
  };
  for (const ProgramCase &c : program_cases) {
    SCOPED_TRACE(std::string("program:") + c.name);
    reset_pool_blocker(false);
    reset_agent_blocker(false);
    reset_queue_fakes();
    OnUnload();
    clear_runtime_config_path(); // auto-A0 mode

    FakeApiTable api;
    api.core.hsa_iterate_agents_fn = c.iterate;
    ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
    g_fake_gfx1250_asic_revision = c.a0_present ? 0 : 7; // enumerated gfx1250 is A0 or not
    g_fake_load_program_calls = 0;

    hsa_code_object_reader_t reader{0x1234};
    const hsa_status_t status = api.core.hsa_executable_load_program_code_object_fn(
        kFakeExecutable, reader, nullptr, nullptr);
    if (c.expect == Outcome::kForwardOriginal) {
      EXPECT_EQ(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_program_calls, 1);
    } else {
      EXPECT_NE(status, HSA_STATUS_SUCCESS);
      EXPECT_EQ(g_fake_load_program_calls, 0);
    }
    OnUnload();
  }
  g_fake_gfx1250_asic_revision = 0;
}

// Query the AMD loader vendor table through the (patched) core entry.
hsa_ven_amd_loader_1_03_pfn_t query_vendor_table(FakeApiTable &api) {
  hsa_ven_amd_loader_1_03_pfn_t table{};
  EXPECT_EQ(api.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                             sizeof(table), &table),
            HSA_STATUS_SUCCESS);
  return table;
}

// Write bytes to a fresh temp file and return an open read fd (owned by caller).
struct TempFile {
  std::filesystem::path path;
  int fd = -1;
  ~TempFile() {
    if (fd >= 0)
      ::close(fd);
    if (!path.empty())
      std::filesystem::remove(path);
  }
};
TempFile make_temp_file(const std::vector<uint8_t> &bytes) {
  TempFile tf;
  tf.path = std::filesystem::temp_directory_path() /
            ("rocjitsu-vendor-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
             std::to_string(reinterpret_cast<uintptr_t>(&tf)));
  {
    std::ofstream out(tf.path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  tf.fd = ::open(tf.path.c_str(), O_RDONLY);
  return tf;
}

// The hook wraps the vendor reader constructor: querying the AMD loader table
// yields the hook's wrapper, not the underlying fake, and calling it forwards
// the fd/offset/size to the real constructor.
TEST(HsaHooksUnitTest, WrapsVendorReaderAndForwardsToOriginal) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  const auto table = query_vendor_table(api);
  ASSERT_NE(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size, nullptr);
  EXPECT_NE(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
            fake_vendor_reader_create_from_file_with_offset_size);

  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  TempFile tf = make_temp_file(image);
  ASSERT_GE(tf.fd, 0);

  g_fake_vendor_reader_call = {};
  hsa_code_object_reader_t reader{};
  EXPECT_EQ(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
                tf.fd, 0, image.size(), &reader),
            HSA_STATUS_SUCCESS);
  // The real (fake) constructor was called with the same fd/offset/size.
  EXPECT_EQ(g_fake_vendor_reader_call.count, 1);
  EXPECT_EQ(g_fake_vendor_reader_call.file, tf.fd);
  EXPECT_EQ(g_fake_vendor_reader_call.offset, 0u);
  EXPECT_EQ(g_fake_vendor_reader_call.size, image.size());
  OnUnload();
}

// Bytes captured through the vendor reader are available to the load path: an
// auto-A0 load of a non-gfx1250 object created via the vendor reader forwards
// verbatim (it has captured bytes, so it does not fail closed for missing bytes).
TEST(HsaHooksUnitTest, VendorReaderCaptureFeedsLoadPath) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  const auto table = query_vendor_table(api);
  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  TempFile tf = make_temp_file(image);
  ASSERT_GE(tf.fd, 0);

  hsa_code_object_reader_t reader{};
  ASSERT_EQ(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
                tf.fd, 0, image.size(), &reader),
            HSA_STATUS_SUCCESS);

  // Non-gfx1250 object on the A0 agent: forwarded (captured bytes were found, so
  // the load did not fail closed for a missing source image), but loaded from the
  // vendor-captured snapshot -- a hook-owned reader whose bytes equal the file.
  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(g_fake_load_agent_calls, 1);
  EXPECT_NE(g_last_load_reader.handle, reader.handle);
  EXPECT_EQ(g_last_load_bytes, image);
  OnUnload();
}

// A partial vendor table too small to contain the reader-create field must not
// be patched, and every byte the caller provided must be left untouched.
TEST(HsaHooksUnitTest, DoesNotPatchVendorTableTooSmallForReaderField) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  write_runtime_config_path();

  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  // Request only the first three pointers -- smaller than the reader field.
  const size_t small_len = 3 * sizeof(void (*)());
  ASSERT_LT(small_len,
            offsetof(hsa_ven_amd_loader_1_03_pfn_t,
                     hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size));
  alignas(
      hsa_ven_amd_loader_1_03_pfn_t) unsigned char buffer[sizeof(hsa_ven_amd_loader_1_03_pfn_t)];
  std::memset(buffer, 0xAB, sizeof(buffer));
  EXPECT_EQ(api.core.hsa_system_get_major_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1, small_len,
                                                             buffer),
            HSA_STATUS_SUCCESS);
  // Bytes beyond the requested length are untouched (the hook did not write the
  // reader field into a too-small table).
  for (size_t i = small_len; i < sizeof(buffer); ++i)
    EXPECT_EQ(buffer[i], 0xAB) << "byte " << i << " was modified";
  OnUnload();
}

// Capture is bounds-checked: an offset past end-of-file yields no captured bytes
// (the load still forwards to the loader; it is not a valid gfx1250 object).
TEST(HsaHooksUnitTest, VendorReaderOutOfBoundsOffsetCapturesNothing) {
  reset_pool_blocker(false);
  reset_agent_blocker(false);
  reset_queue_fakes();
  OnUnload();
  clear_runtime_config_path(); // auto-A0 mode

  FakeApiTable api;
  ASSERT_TRUE(OnLoad(&api.table, 0, 0, nullptr));
  g_fake_gfx1250_asic_revision = 0;
  g_fake_load_agent_calls = 0;

  const auto table = query_vendor_table(api);
  const auto header = make_amdgpu_elf_header(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942);
  std::vector<uint8_t> image(reinterpret_cast<const uint8_t *>(&header),
                             reinterpret_cast<const uint8_t *>(&header) + sizeof(header));
  TempFile tf = make_temp_file(image);
  ASSERT_GE(tf.fd, 0);

  hsa_code_object_reader_t reader{};
  // Offset past EOF: the real constructor still succeeds (fake), but capture is
  // refused, so the auto-A0 load has no bytes for this reader and fails closed.
  ASSERT_EQ(table.hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
                tf.fd, image.size() + 4096, image.size(), &reader),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(api.core.hsa_executable_load_agent_code_object_fn(kFakeExecutable, kGfx1250Agent,
                                                              reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER);
  EXPECT_EQ(g_fake_load_agent_calls, 0);
  OnUnload();
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

TEST(HsaHooksUnitTest, MultiProducerDoorbellRewritesEarlierPublishedPacket) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // Fallback (non-intercept) multi-producer path: a producer publishes two ready
  // packets and rings once with the FINAL packet id. Packet 0 needs virtual-LDS
  // rewriting, packet 1 does not. The doorbell must rewrite the whole published
  // range [next_packet_id, id], not just the named packet -- otherwise packet 0
  // reaches the command processor as an oversized (host-faulting) launch and the
  // frontier advances past it so the scanner skips it too.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000; // exceeds host LDS -> sidecar
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  // Packet 0: the oversized virtual-LDS dispatch that must be rewritten.
  auto &oversized = g_fake_queue_packets[0];
  oversized.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  oversized.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  oversized.group_segment_size = normal_descriptor.group_segment_fixed_size;
  oversized.workgroup_size_x = 64;
  oversized.workgroup_size_y = 1;
  oversized.workgroup_size_z = 1;
  oversized.grid_size_x = 64;
  oversized.grid_size_y = 1;
  oversized.grid_size_z = 1;

  // Packet 1: an ordinary below-threshold dispatch (no rewrite needed).
  auto &ordinary = g_fake_queue_packets[1];
  ordinary.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  ordinary.kernel_object = 0; // no registered virtual-LDS metadata
  ordinary.group_segment_size = 0;
  ordinary.workgroup_size_x = 64;
  ordinary.grid_size_x = 64;

  // Ring once with the FINAL packet id (1).
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 1);

  // Packet 0 was rewritten to the virtual descriptor (range covered), not left
  // on its oversized normal descriptor.
  EXPECT_EQ(oversized.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(oversized.group_segment_size, 0u);
  EXPECT_FALSE(g_fake_allocation_sizes.empty());

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, MultiProducerHoleCloseAdvancesFrontierAcrossReadySuffix) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  // Regression for the stranded-frontier bug: on a size-4 multi-producer queue an
  // out-of-order ring publishes a ready suffix ABOVE an unready hole, the hole
  // later closes, and a subsequent batch must not skip a packet.
  //
  //   1. next_packet_id = 0. Slot 0 (packet 0) is INVALID (hole); packets 1..3 are
  //      ready. A producer rings with id 3. The range [0,4) rewrites 1..3, but the
  //      cursor cannot pass the hole at 0 -- it must REMEMBER 1..3 as ready.
  //   2. Packet 0 closes (becomes ready) and rings with id 0. The cursor must now
  //      catch up across the remembered 1..3 -> next_packet_id = 4, NOT 1.
  //   3. Packets 4 and 5 are published and a producer rings once with id 5. Only
  //      when the cursor is at 4 does 5 - 4 = 1 < size take the range path and
  //      rewrite packet 4. If the cursor were stranded at 1, 5 - 1 = 4 == size
  //      would fall to the single-packet path and packet 4 (needing a virtual-LDS
  //      rewrite) would reach the command processor as an oversized launch.
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000; // exceeds host LDS -> sidecar
  virtual_descriptor.private_segment_fixed_size = 96;
  auto registration = register_virtual_lds_kernel_for_test(
      api, normal_descriptor, virtual_descriptor, normal_descriptor.group_segment_fixed_size);
  (void)registration;

  const auto make_oversized = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
    packet.group_segment_size = normal_descriptor.group_segment_fixed_size;
    packet.workgroup_size_x = 64;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = 64;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
  };
  const auto make_ordinary = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    packet.kernel_object = 0; // no registered virtual-LDS metadata
    packet.workgroup_size_x = 64;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = 64;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
  };
  const auto make_invalid = [&](uint32_t slot) {
    auto &packet = g_fake_queue_packets[slot];
    packet = {};
    packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  };

  // Phase 1: slot 0 (packet 0) is an unready hole; packets 1..3 are ready ordinary
  // dispatches. Ring with the final published id (3).
  make_invalid(0);
  make_ordinary(1);
  make_ordinary(2);
  make_ordinary(3);
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 3);
  // The hole blocks the cursor, so no virtual-LDS rewrite happened yet.
  EXPECT_TRUE(g_fake_allocation_sizes.empty());

  // Phase 2: the hole closes (packet 0 becomes a ready ordinary dispatch) and its
  // producer rings with id 0. The cursor must catch up across the remembered ready
  // suffix 1..3 and land at 4.
  make_ordinary(0);
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);
  EXPECT_TRUE(g_fake_allocation_sizes.empty());

  // Phase 3: packets 4 (slot 0) and 5 (slot 1) are published; packet 4 is the
  // oversized virtual-LDS dispatch. Ring once with the final id (5).
  make_oversized(0); // packet 4 -> slot 4 % 4 == 0
  make_ordinary(1);  // packet 5 -> slot 5 % 4 == 1
  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 5);

  // Packet 4 was rewritten to the virtual descriptor -- it was NOT skipped.
  EXPECT_EQ(g_fake_queue_packets[0].kernel_object,
            reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(g_fake_queue_packets[0].group_segment_size, 0u);
  EXPECT_FALSE(g_fake_allocation_sizes.empty());

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
