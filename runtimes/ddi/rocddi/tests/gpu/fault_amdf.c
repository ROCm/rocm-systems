// SPDX-License-Identifier: MIT

#include "fault_ioctl.h"
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "AMDF fault check failed at line %d: %s\n", __LINE__,    \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define INIT(record, tag)                                                      \
  do {                                                                         \
    memset(&(record), 0, sizeof(record));                                      \
    (record).type = (tag);                                                     \
    (record).structure_size = (uint32_t)sizeof(record);                        \
  } while (0)
#define API_ERROR(code) amdf_make_api_status(AMDF_STATUS_CODE_##code)

int main(int argc, char **argv) {
  CHECK(argc == 2 &&
        (!strcmp(argv[1], "create") || !strcmp(argv[1], "doorbell")));
  void (*arm)(int) = dlsym(RTLD_DEFAULT, "rocddi_fault_arm");
  void (*stats)(struct fault_stats *) =
      dlsym(RTLD_DEFAULT, "rocddi_fault_stats");
  CHECK(arm && stats);
  const amdf_api_t *api = NULL;
  CHECK(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                       &api) == AMDF_STATUS_OK);
  const void *extension = NULL;
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension) ==
        AMDF_STATUS_OK);
  const amdf_gpu_api_t *gpu = extension;
  amdf_instance_create_info_t instance_desc;
  INIT(instance_desc, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instance_desc.native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  amdf_instance_t *instance = NULL;
  CHECK(api->instance_create(&instance_desc, &instance) == AMDF_STATUS_OK);
  uint32_t count = 0;
  CHECK(api->endpoint_enumerate(instance, 0, NULL, &count) == AMDF_STATUS_OK &&
        count);
  amdf_endpoint_summary_t *summaries = calloc(count, sizeof(*summaries));
  CHECK(summaries);
  CHECK(api->endpoint_enumerate(instance, count, summaries, &count) ==
        AMDF_STATUS_OK);
  amdf_endpoint_t *endpoint = NULL;
  for (uint32_t i = 0; i < count && !endpoint; ++i)
    if (summaries[i].engine_kind == AMDF_ENGINE_KIND_GPU)
      CHECK(api->endpoint_open(instance, &summaries[i].id, &endpoint) ==
            AMDF_STATUS_OK);
  free(summaries);
  CHECK(endpoint);
  amdf_gpu_endpoint_info_t gpu_info;
  INIT(gpu_info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
  CHECK(gpu->endpoint_query_info(endpoint, &gpu_info) == AMDF_STATUS_OK);
  amdf_endpoint_info_t endpoint_info;
  INIT(endpoint_info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CHECK(api->endpoint_query_info(endpoint, &endpoint_info) == AMDF_STATUS_OK);
  amdf_gpu_device_create_info_t device_desc;
  INIT(device_desc, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  amdf_device_t *device = NULL;
  CHECK(gpu->device_create(endpoint, &device_desc, &device) == AMDF_STATUS_OK);
  uint32_t family_ordinal = UINT32_MAX;
  amdf_queue_family_info_t family;
  for (uint32_t i = 0; i < endpoint_info.queue_family_count; ++i) {
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CHECK(api->endpoint_query_queue_family_info(endpoint, i, &family) ==
          AMDF_STATUS_OK);
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
        (family.roles & AMDF_QUEUE_ROLE_COMPUTE) &&
        (family.user_queue_capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE)) {
      family_ordinal = i;
      break;
    }
  }
  CHECK(family_ordinal != UINT32_MAX);
  uint32_t scope_count = 0;
  amdf_memory_scope_t *scope = NULL;
  CHECK(api->instance_enumerate_memory_scopes(instance, 1, &scope,
                                              &scope_count) == AMDF_STATUS_OK);
  CHECK(scope_count == 1 && scope);
  amdf_memory_scope_info_t scope_info;
  INIT(scope_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CHECK(api->memory_scope_query_info(scope, &scope_info) == AMDF_STATUS_OK);
  amdf_memory_device_access_t access = {0};
  access.device = device;
  access.requirements.access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  access.requirements.flags =
      AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  uint32_t profile_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t caps;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(caps, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = api->memory_scope_query_device_profile(
        scope, i, 1, &access, &profile, &caps);
    if (status == API_ERROR(UNSUPPORTED))
      continue;
    CHECK(status == AMDF_STATUS_OK);
    if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) {
      profile_ordinal = i;
      break;
    }
  }
  CHECK(profile_ordinal != UINT32_MAX);
  CHECK(gpu_info.compute.wavefront_size && gpu_info.topology.xcc_count &&
        gpu_info.topology.shader_engine_count_per_xcc);
  uint32_t waves = gpu_info.topology.xcc_count *
                   gpu_info.topology.shader_engine_count_per_xcc;
  uint32_t private_bytes = 256;
  uint64_t wave_bytes =
      (uint64_t)gpu_info.compute.wavefront_size * private_bytes;
  wave_bytes = (wave_bytes + 255) & ~UINT64_C(255);
  uint64_t scratch_bytes = wave_bytes * waves;
  amdf_memory_create_info_t memory_desc;
  INIT(memory_desc, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  memory_desc.memory_profile_ordinal = profile_ordinal;
  memory_desc.byte_length = scratch_bytes;
  memory_desc.minimum_alignment = 256;
  memory_desc.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  memory_desc.access_count = 1;
  memory_desc.accesses = &access;
  amdf_memory_t *scratch = NULL;
  CHECK(api->memory_create(scope, &memory_desc, &scratch) == AMDF_STATUS_OK);
  amdf_gpu_user_queue_create_info_t queue_desc;
  INIT(queue_desc, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  queue_desc.queue_family_ordinal = family_ordinal;
  queue_desc.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  queue_desc.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  queue_desc.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  queue_desc.ring_byte_length = family.minimum_ring_byte_length;
  queue_desc.scratch.memory = scratch;
  queue_desc.scratch.access_ordinal = 0;
  queue_desc.scratch.byte_length = scratch_bytes;
  queue_desc.scratch.maximum_private_segment_byte_length = private_bytes;
  queue_desc.scratch.maximum_wave_count = waves;
  arm(!strcmp(argv[1], "create") ? ROCDDI_FAULT_CREATE_EFAULT
                                 : ROCDDI_FAULT_DOORBELL_ROLLBACK);
  amdf_user_queue_t *queue = NULL;
  amdf_status_t status = gpu->user_queue_create(device, &queue_desc, &queue);
  struct fault_stats got;
  stats(&got);
  amdf_status_t scratch_destroy = api->memory_destroy(scratch);
  printf("AMDF %s status=0x%llx scratch_destroy=0x%llx creates=%u "
         "create_faults=%u destroy_faults=%u alloc=%u free=%u\n",
         argv[1], (unsigned long long)status,
         (unsigned long long)scratch_destroy, got.creates, got.create_faults,
         got.destroy_faults, got.allocations, got.frees);
  CHECK(status != AMDF_STATUS_OK && !queue);
  CHECK(got.creates == 1);
  CHECK(scratch_destroy == API_ERROR(BUSY));
  if (!strcmp(argv[1], "create"))
    CHECK(got.create_faults == 1 && got.destroy_faults == 0);
  else
    CHECK(got.destroy_faults == 1);
  return 0;
}
