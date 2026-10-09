// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define _GNU_SOURCE
#include <dlfcn.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
#define API(name) __typeof__(&name) name;
  API(hsa_init)
  API(hsa_shut_down)
  API(hsa_iterate_agents)
  API(hsa_agent_get_info)
  API(hsa_amd_agent_iterate_memory_pools)
  API(hsa_amd_memory_pool_get_info)
  API(hsa_amd_memory_pool_allocate)
  API(hsa_amd_memory_pool_free)
  API(hsa_amd_agents_allow_access)
  API(hsa_code_object_reader_create_from_memory)
  API(hsa_code_object_reader_destroy)
  API(hsa_executable_create)
  API(hsa_executable_load_agent_code_object)
  API(hsa_executable_freeze)
  API(hsa_executable_get_symbol_by_name)
  API(hsa_executable_symbol_get_info)
  API(hsa_executable_destroy)
  API(hsa_queue_create)
  API(hsa_queue_destroy)
  API(hsa_queue_add_write_index_relaxed)
  API(hsa_signal_create)
  API(hsa_signal_destroy)
  API(hsa_signal_store_relaxed)
  API(hsa_signal_store_screlease)
  API(hsa_signal_wait_scacquire)
#undef API
} api_t;
static api_t api;
static hsa_agent_t cpu, gpu;
static hsa_amd_memory_pool_t kernarg_pool;

static void check(hsa_status_t status, const char *operation) {
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "%s failed: 0x%x\n", operation, status);
    exit(2);
  }
}

static void load_api(void *library) {
#define LOAD(name) do { \
  api.name = (__typeof__(api.name))dlsym(library, #name); \
  if (!api.name) { fprintf(stderr, "missing %s: %s\n", #name, dlerror()); exit(2); } \
} while (0)
  LOAD(hsa_init);
  LOAD(hsa_shut_down);
  LOAD(hsa_iterate_agents);
  LOAD(hsa_agent_get_info);
  LOAD(hsa_amd_agent_iterate_memory_pools);
  LOAD(hsa_amd_memory_pool_get_info);
  LOAD(hsa_amd_memory_pool_allocate);
  LOAD(hsa_amd_memory_pool_free);
  LOAD(hsa_amd_agents_allow_access);
  LOAD(hsa_code_object_reader_create_from_memory);
  LOAD(hsa_code_object_reader_destroy);
  LOAD(hsa_executable_create);
  LOAD(hsa_executable_load_agent_code_object);
  LOAD(hsa_executable_freeze);
  LOAD(hsa_executable_get_symbol_by_name);
  LOAD(hsa_executable_symbol_get_info);
  LOAD(hsa_executable_destroy);
  LOAD(hsa_queue_create);
  LOAD(hsa_queue_destroy);
  LOAD(hsa_queue_add_write_index_relaxed);
  LOAD(hsa_signal_create);
  LOAD(hsa_signal_destroy);
  LOAD(hsa_signal_store_relaxed);
  LOAD(hsa_signal_store_screlease);
  LOAD(hsa_signal_wait_scacquire);
#undef LOAD
}

static hsa_status_t choose_agents(hsa_agent_t agent, void *data) {
  (void)data;
  hsa_device_type_t type;
  hsa_status_t status = api.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (type == HSA_DEVICE_TYPE_CPU && !cpu.handle) cpu = agent;
  if (type == HSA_DEVICE_TYPE_GPU && !gpu.handle) gpu = agent;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t choose_pool(hsa_amd_memory_pool_t pool, void *data) {
  (void)data;
  hsa_amd_segment_t segment;
  uint32_t flags;
  bool allowed;
  hsa_status_t status = api.hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  status = api.hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = api.hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (allowed &&
      (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) ==
          (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
           HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) kernarg_pool = pool;
  return HSA_STATUS_SUCCESS;
}

static void *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  if (fseek(f, 0, SEEK_END)) { perror("fseek"); exit(2); }
  long length = ftell(f);
  if (length <= 0 || length > 1048576) { fprintf(stderr, "bad code object size\n"); exit(2); }
  rewind(f);
  void *bytes = malloc((size_t)length);
  if (!bytes || fread(bytes, 1, (size_t)length, f) != (size_t)length) {
    fprintf(stderr, "code object read failed\n"); exit(2);
  }
  fclose(f);
  *size = (size_t)length;
  return bytes;
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s LIBRARY HSACO\n", argv[0]); return 2; }
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
  load_api(library);
  check(api.hsa_init(), "hsa_init");
  check(api.hsa_iterate_agents(choose_agents, NULL), "iterate agents");
  if (!cpu.handle || !gpu.handle) { fprintf(stderr, "missing CPU or GPU agent\n"); return 2; }
  check(api.hsa_amd_agent_iterate_memory_pools(cpu, choose_pool, NULL), "iterate pools");
  if (!kernarg_pool.handle) { fprintf(stderr, "no allocatable fine kernarg pool\n"); return 2; }
  unsigned *output = NULL;
  void *args = NULL;
  check(api.hsa_amd_memory_pool_allocate(kernarg_pool, 4096, 0, (void **)&output), "allocate output");
  check(api.hsa_amd_memory_pool_allocate(kernarg_pool, 4096, 0, &args), "allocate kernarg");
  check(api.hsa_amd_agents_allow_access(1, &gpu, NULL, output), "allow output");
  check(api.hsa_amd_agents_allow_access(1, &gpu, NULL, args), "allow kernarg");
  output[0] = 0;
  memset(args, 0, 4096);
  memcpy(args, &output, sizeof(output));
  size_t code_size;
  void *code = read_file(argv[2], &code_size);
  hsa_code_object_reader_t reader;
  check(api.hsa_code_object_reader_create_from_memory(code, code_size, &reader), "reader");
  hsa_profile_t profile;
  check(api.hsa_agent_get_info(gpu, HSA_AGENT_INFO_PROFILE, &profile), "GPU profile");
  hsa_executable_t executable;
  check(api.hsa_executable_create(profile, HSA_EXECUTABLE_STATE_UNFROZEN, NULL, &executable), "executable create");
  hsa_loaded_code_object_t loaded;
  check(api.hsa_executable_load_agent_code_object(executable, gpu, reader, NULL, &loaded), "load code object");
  check(api.hsa_executable_freeze(executable, NULL), "freeze executable");
  hsa_executable_symbol_t symbol;
  check(api.hsa_executable_get_symbol_by_name(executable, "rocddi_increment.kd", &gpu, &symbol), "kernel lookup");
  uint64_t kernel_object;
  uint32_t kernarg_size, group_size, private_size;
  check(api.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object), "kernel object");
  check(api.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size), "kernarg size");
  check(api.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size), "group size");
  check(api.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size), "private size");
  if (kernarg_size > 4096) { fprintf(stderr, "kernarg too large\n"); return 2; }
  hsa_queue_t *queue = NULL;
  check(api.hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue), "queue create");
  hsa_signal_t completion;
  check(api.hsa_signal_create(1, 0, NULL, &completion), "signal create");
  for (unsigned iteration = 1; iteration <= 16; iteration++) {
    api.hsa_signal_store_relaxed(completion, 1);
    uint64_t index = api.hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_kernel_dispatch_packet_t *packet = &((hsa_kernel_dispatch_packet_t *)queue->base_address)[index & (queue->size - 1)];
    memset(packet, 0, sizeof(*packet));
    packet->setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet->workgroup_size_x = packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_x = packet->grid_size_y = packet->grid_size_z = 1;
    packet->private_segment_size = private_size;
    packet->group_segment_size = group_size;
    packet->kernel_object = kernel_object;
    packet->kernarg_address = args;
    packet->completion_signal = completion;
    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    api.hsa_signal_store_screlease(queue->doorbell_signal, (hsa_signal_value_t)index);
    hsa_signal_value_t value = api.hsa_signal_wait_scacquire(completion,
        HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    if (value != 0 || output[0] != iteration) {
      fprintf(stderr, "dispatch %u: completion=%" PRId64 " output=%u\n", iteration, value, output[0]);
      return 2;
    }
  }
  printf("gpu_increment=%u kernarg_size=%u\n", output[0], kernarg_size);
  check(api.hsa_signal_destroy(completion), "signal destroy");
  check(api.hsa_queue_destroy(queue), "queue destroy");
  check(api.hsa_executable_destroy(executable), "executable destroy");
  check(api.hsa_code_object_reader_destroy(reader), "reader destroy");
  check(api.hsa_amd_memory_pool_free(args), "free kernarg");
  check(api.hsa_amd_memory_pool_free(output), "free output");
  check(api.hsa_shut_down(), "hsa shutdown");
  free(code);
  dlclose(library);
  return 0;
}
