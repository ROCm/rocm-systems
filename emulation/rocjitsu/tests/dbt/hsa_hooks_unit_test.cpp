// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "hsa/hsa_api_trace_minimal.h"
#include "rocjitsu/code/dbt/virtual_lds_metadata.h"
#include "rocjitsu/kmd/linux/rpc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

extern "C" bool OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char *const *failed_tool_names);
extern "C" void OnUnload();

namespace {

constexpr hsa_agent_t kGuestAgent{1};
constexpr hsa_agent_t kHostAgent{2};
constexpr hsa_isa_t kGuestIsa{950};
constexpr hsa_isa_t kHostIsa{1201};
constexpr hsa_amd_memory_pool_t kGuestPool{10};
constexpr hsa_amd_memory_pool_t kHostPool{20};
constexpr hsa_amd_memory_pool_t kHostKernargPool{21};
constexpr uint32_t kGuestNodeId = 100;
constexpr uint32_t kHostNodeId = 200;

std::mutex g_pool_mutex;
std::condition_variable g_pool_cv;
bool g_block_guest_pool_iteration = false;
bool g_guest_pool_iteration_entered = false;
bool g_release_guest_pool_iteration = false;
int g_fake_shutdown_calls = 0;
hsa_amd_memory_pool_t g_last_allocate_pool{};
std::vector<std::vector<uint8_t>> g_fake_allocations;
std::vector<hsa_amd_memory_pool_t> g_fake_allocation_pools;
std::vector<size_t> g_fake_allocation_sizes;
std::array<hsa_kernel_dispatch_packet_t, 4> g_fake_queue_packets{};
hsa_queue_t g_fake_queue{};
hsa_agent_t g_last_queue_create_agent{};
hsa_queue_t *g_last_destroyed_queue = nullptr;
int g_fake_signal_store_relaxed_calls = 0;
int g_fake_signal_store_screlease_calls = 0;
hsa_signal_t g_last_signal_store_signal{};
hsa_signal_value_t g_last_signal_store_value = 0;
hsa_queue_t *g_last_intercept_registered_queue = nullptr;
hsa_amd_queue_intercept_handler_t g_fake_intercept_handler = nullptr;
void *g_fake_intercept_user_data = nullptr;
std::vector<hsa_kernel_dispatch_packet_t> g_last_intercept_written_packets;

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

  hsa_status_t status = callback(kGuestAgent, data);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  return callback(kHostAgent, data);
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

hsa_status_t HSA_API
fake_code_object_reader_create_from_file(hsa_file_t, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_create_from_memory(
    const void *, size_t, hsa_code_object_reader_t *code_object_reader) {
  if (code_object_reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  code_object_reader->handle = 2;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_code_object_reader_destroy(hsa_code_object_reader_t) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_executable_load_agent_code_object(hsa_executable_t, hsa_agent_t,
                                                            hsa_code_object_reader_t, const char *,
                                                            hsa_loaded_code_object_t *) {
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

hsa_status_t HSA_API fake_amd_agents_allow_access(uint32_t, const hsa_agent_t *, const uint32_t *,
                                                  const void *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API fake_amd_agent_iterate_memory_pools(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_amd_memory_pool_t, void *), void *data) {
  if (callback == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent.handle == kGuestAgent.handle) {
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
    core.hsa_signal_store_relaxed_fn = fake_signal_store_relaxed;
    core.hsa_signal_store_screlease_fn = fake_signal_store_screlease;
    core.hsa_code_object_reader_create_from_file_fn = fake_code_object_reader_create_from_file;
    core.hsa_code_object_reader_create_from_memory_fn = fake_code_object_reader_create_from_memory;
    core.hsa_code_object_reader_destroy_fn = fake_code_object_reader_destroy;
    core.hsa_executable_load_agent_code_object_fn = fake_executable_load_agent_code_object;
    amd.hsa_amd_agent_iterate_memory_pools_fn = fake_amd_agent_iterate_memory_pools;
    amd.hsa_amd_memory_pool_get_info_fn = fake_amd_memory_pool_get_info;
    amd.hsa_amd_memory_pool_allocate_fn = fake_amd_memory_pool_allocate;
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

void reset_pool_blocker(bool enabled) {
  std::lock_guard lock(g_pool_mutex);
  g_block_guest_pool_iteration = enabled;
  g_guest_pool_iteration_entered = false;
  g_release_guest_pool_iteration = false;
}

void release_pool_blocker() {
  {
    std::lock_guard lock(g_pool_mutex);
    g_release_guest_pool_iteration = true;
  }
  g_pool_cv.notify_all();
}

void reset_queue_fakes() {
  g_fake_queue_packets = {};
  g_fake_queue = {};
  g_last_queue_create_agent = {};
  g_last_destroyed_queue = nullptr;
  g_fake_allocations.clear();
  g_fake_allocation_pools.clear();
  g_fake_allocation_sizes.clear();
  g_fake_signal_store_relaxed_calls = 0;
  g_fake_signal_store_screlease_calls = 0;
  g_last_signal_store_signal = {};
  g_last_signal_store_value = 0;
  g_last_intercept_registered_queue = nullptr;
  g_fake_intercept_handler = nullptr;
  g_fake_intercept_user_data = nullptr;
  g_last_intercept_written_packets.clear();
}

TEST(HsaHooksUnitTest, IterateAgentsDropsGuestOwnSlotWhenGuestAppearsFirst) {
  reset_pool_blocker(false);
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

TEST(HsaHooksUnitTest, UninstallDoesNotWaitForPoolMapperDiscoveryLock) {
  reset_pool_blocker(true);
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

TEST(HsaHooksUnitTest, QueueDoorbellSignalStoreIsForwardedAfterTrackedQueueScan) {
  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());
  ASSERT_NE(api.core.hsa_queue_create_fn, fake_queue_create);
  ASSERT_NE(api.core.hsa_signal_store_relaxed_fn, fake_signal_store_relaxed);

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
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

TEST(HsaHooksUnitTest, VirtualLdsRewriteAllocatesPerWorkgroupBackingState) {
  using rocr::llvm::amdhsa::kernel_descriptor_t;

  reset_pool_blocker(false);
  reset_queue_fakes();
  FakeApiTable api;
  InstalledHook hook(api);
  ASSERT_TRUE(hook.installed());

  hsa_queue_t *queue = nullptr;
  ASSERT_EQ(api.core.hsa_queue_create_fn(kGuestAgent, 4, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                         0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  kernel_descriptor_t normal_descriptor{};
  kernel_descriptor_t virtual_descriptor{};
  normal_descriptor.group_segment_fixed_size = 70000;
  virtual_descriptor.private_segment_fixed_size = 96;
  struct DescriptorDispatchRecord {
    int64_t virtual_descriptor_delta = 0;
    uint32_t kernarg_size = 0;
    uint32_t offset_and_flags = 0;
  } dispatch_record{};
  dispatch_record.virtual_descriptor_delta = reinterpret_cast<intptr_t>(&virtual_descriptor) -
                                             reinterpret_cast<intptr_t>(&normal_descriptor);
  dispatch_record.offset_and_flags =
      offsetof(hsa_kernel_dispatch_packet_t, reserved2) |
      (static_cast<uint32_t>(rocjitsu::kVirtualLdsFlagRuntimeStateBlock |
                             rocjitsu::kVirtualLdsFlagBackingPointerInDispatchPacket |
                             rocjitsu::kVirtualLdsFlagWorkgroupIdX)
       << 24);
  std::memcpy(normal_descriptor.reserved0, "RJLD", 4);
  std::memcpy(normal_descriptor.reserved1, &dispatch_record, sizeof(dispatch_record));

  auto &packet = g_fake_queue_packets[0];
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 1024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  api.core.hsa_signal_store_relaxed_fn(queue->doorbell_signal, 0);

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], 24u);

  EXPECT_EQ(packet.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(packet.group_segment_size, 0u);
  EXPECT_EQ(packet.private_segment_size, 96u);
  ASSERT_NE(packet.reserved2, 0u);
  EXPECT_EQ(packet.reserved2, reinterpret_cast<uintptr_t>(g_fake_allocations[1].data()));

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data(), sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
}

TEST(HsaHooksUnitTest, VirtualLdsInterceptRewriteMirrorsDispatchPacketStateToQueueSlot) {
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
  virtual_descriptor.private_segment_fixed_size = 96;
  struct DescriptorDispatchRecord {
    int64_t virtual_descriptor_delta = 0;
    uint32_t kernarg_size = 0;
    uint32_t offset_and_flags = 0;
  } dispatch_record{};
  dispatch_record.virtual_descriptor_delta = reinterpret_cast<intptr_t>(&virtual_descriptor) -
                                             reinterpret_cast<intptr_t>(&normal_descriptor);
  dispatch_record.offset_and_flags =
      offsetof(hsa_kernel_dispatch_packet_t, reserved2) |
      (static_cast<uint32_t>(rocjitsu::kVirtualLdsFlagRuntimeStateBlock |
                             rocjitsu::kVirtualLdsFlagBackingPointerInDispatchPacket |
                             rocjitsu::kVirtualLdsFlagWorkgroupIdX)
       << 24);
  std::memcpy(normal_descriptor.reserved0, "RJLD", 4);
  std::memcpy(normal_descriptor.reserved1, &dispatch_record, sizeof(dispatch_record));

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  packet.kernel_object = reinterpret_cast<uintptr_t>(&normal_descriptor);
  packet.private_segment_size = 12;
  packet.group_segment_size = 1024;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 256;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  constexpr uint64_t kPacketIndex = 2;
  g_fake_queue_packets[kPacketIndex] = packet;
  g_fake_intercept_handler(&packet, 1, kPacketIndex, g_fake_intercept_user_data,
                           fake_intercept_packet_writer);

  constexpr uint32_t kStaticLds = 70000;
  constexpr uint32_t kDynamicLds = 1024;
  constexpr uint32_t kRequestedLds = kStaticLds + kDynamicLds;
  constexpr uint32_t kGroupsX = 4;
  ASSERT_EQ(g_fake_allocation_sizes.size(), 2u);
  EXPECT_EQ(g_fake_allocation_pools[0].handle, kHostPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[0], static_cast<size_t>(kRequestedLds * kGroupsX));
  EXPECT_EQ(g_fake_allocation_pools[1].handle, kHostKernargPool.handle);
  EXPECT_EQ(g_fake_allocation_sizes[1], 24u);

  ASSERT_EQ(g_last_intercept_written_packets.size(), 1u);
  const hsa_kernel_dispatch_packet_t &written = g_last_intercept_written_packets[0];
  EXPECT_EQ(written.kernel_object, reinterpret_cast<uintptr_t>(&virtual_descriptor));
  EXPECT_EQ(written.group_segment_size, 0u);
  EXPECT_EQ(written.private_segment_size, 96u);
  ASSERT_NE(written.reserved2, 0u);
  EXPECT_EQ(written.reserved2, reinterpret_cast<uintptr_t>(g_fake_allocations[1].data()));
  EXPECT_EQ(g_fake_queue_packets[kPacketIndex].reserved2, written.reserved2);

  struct RuntimeState {
    uint64_t backing_base = 0;
    uint32_t stride_x = 0;
    uint32_t stride_y = 0;
    uint32_t stride_z = 0;
    uint32_t reserved = 0;
  } state{};
  static_assert(sizeof(RuntimeState) == 24);
  std::memcpy(&state, g_fake_allocations[1].data(), sizeof(state));
  EXPECT_EQ(state.backing_base, reinterpret_cast<uintptr_t>(g_fake_allocations[0].data()));
  EXPECT_EQ(state.stride_x, kRequestedLds);
  EXPECT_EQ(state.stride_y, 0u);
  EXPECT_EQ(state.stride_z, 0u);
  EXPECT_EQ(state.reserved, 0u);

  EXPECT_EQ(api.core.hsa_queue_destroy_fn(queue), HSA_STATUS_SUCCESS);
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
