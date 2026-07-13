// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
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
#include "rocjitsu/kmd/linux/rpc.h"

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
constexpr uint32_t kGuestNodeId = 100;
constexpr uint32_t kHostNodeId = 200;

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

hsa_status_t HSA_API fake_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t,
                                                   uint32_t, void **ptr) {
  g_last_allocate_pool = memory_pool;
  if (ptr != nullptr)
    *ptr = &g_fake_allocation_storage;
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
  if (agent.handle == kHostAgent.handle)
    return callback(kHostPool, data);
  return HSA_STATUS_ERROR_INVALID_AGENT;
}

hsa_status_t HSA_API fake_amd_memory_pool_get_info(hsa_amd_memory_pool_t,
                                                   hsa_amd_memory_pool_info_t attribute,
                                                   void *value) {
  if (value == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  if (attribute == HSA_AMD_MEMORY_POOL_INFO_SEGMENT) {
    *static_cast<uint32_t *>(value) = 0;
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS) {
    *static_cast<uint32_t *>(value) = 0;
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
    core.hsa_code_object_reader_create_from_file_fn = fake_code_object_reader_create_from_file;
    core.hsa_code_object_reader_create_from_memory_fn = fake_code_object_reader_create_from_memory;
    core.hsa_code_object_reader_destroy_fn = fake_code_object_reader_destroy;
    core.hsa_executable_load_agent_code_object_fn = fake_executable_load_agent_code_object;
    amd.hsa_amd_agent_iterate_memory_pools_fn = fake_amd_agent_iterate_memory_pools;
    amd.hsa_amd_memory_pool_get_info_fn = fake_amd_memory_pool_get_info;
    amd.hsa_amd_agent_memory_pool_get_info_fn = fake_amd_agent_memory_pool_get_info;
    amd.hsa_amd_memory_pool_allocate_fn = fake_amd_memory_pool_allocate;
    amd.hsa_amd_memory_async_batch_copy_fn = fake_amd_memory_async_batch_copy;
    amd.hsa_amd_memory_lock_fn = fake_amd_memory_lock;
    amd.hsa_amd_memory_lock_to_pool_fn = fake_amd_memory_lock_to_pool;
    amd.hsa_amd_pointer_info_fn = fake_amd_pointer_info;
    amd.hsa_amd_vmem_set_access_fn = fake_amd_vmem_set_access;
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

} // namespace
