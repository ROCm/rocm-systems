// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/* AMDF v12 negotiation and inert instance checks. No endpoint is activated. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <amdf/amdf.h>
#include <amdf/gpu.h>

#define CHECK(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "AMDF ABI check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; \
  } \
} while (0)
#define INIT(record, tag) do { \
  memset(&(record), 0, sizeof(record)); \
  (record).type = (tag); \
  (record).structure_size = (uint32_t)sizeof(record); \
} while (0)
#define API_ERROR(code) amdf_make_api_status(AMDF_STATUS_CODE_##code)

struct allocation_counts {
  uint32_t allocations;
  uint32_t frees;
};

static void *AMDF_CALL tracked_allocate(void *user_data, uint64_t byte_length,
                                        uint64_t minimum_alignment) {
  struct allocation_counts *counts = (struct allocation_counts *)user_data;
  ++counts->allocations;
  const size_t alignment = (size_t)minimum_alignment;
  const size_t length = (size_t)byte_length;
  if (length > SIZE_MAX - (alignment - 1))
    return NULL;
  return aligned_alloc(alignment, (length + alignment - 1) & ~(alignment - 1));
}

static void AMDF_CALL tracked_free(void *user_data, void *allocation) {
  struct allocation_counts *counts = (struct allocation_counts *)user_data;
  ++counts->frees;
  free(allocation);
}

static int check_short_final_reservation(const amdf_api_t *api,
                                         amdf_instance_t *instance) {
  amdf_memory_reservation_capabilities_t caps;
  INIT(caps, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES);
  CHECK(api->instance_query_host_memory_reservation_capabilities(instance,
                                                                 &caps) ==
        AMDF_STATUS_OK);
  CHECK(caps.mapping_granularity != 0 &&
        caps.mapping_granularity <= UINT64_MAX / 5);
  const uint64_t page = caps.mapping_granularity;
  amdf_memory_reservation_access_t access = {0};
  access.address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
  access.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  access.mapping_granularity = page * 4;
  amdf_memory_reservation_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO);
  create.byte_length = page * 5;
  create.access_count = 1;
  create.accesses = &access;
  amdf_memory_reservation_t *reservation = NULL;
  CHECK(api->memory_reservation_create(instance, &create, &reservation) ==
        AMDF_STATUS_OK);
  amdf_memory_reservation_access_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO);
  CHECK(api->memory_reservation_query_access_info(reservation, 0, &info) ==
        AMDF_STATUS_OK);
  CHECK(info.byte_length == page * 5 && info.mapping_granularity == page * 4);
  amdf_memory_mapping_update_t update = {0};
  update.byte_length = page;
  CHECK(api->memory_reservation_update(reservation, 0, 1, &update) ==
        API_ERROR(INVALID_ARGUMENT));
  update.target_byte_offset = page * 4;
  CHECK(api->memory_reservation_update(reservation, 0, 1, &update) ==
        AMDF_STATUS_OK);
  CHECK(api->memory_reservation_destroy(reservation) == AMDF_STATUS_OK);
  return 0;
}

int main(void) {
  const amdf_api_t *api = NULL;
  CHECK(amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api) == AMDF_STATUS_OK);
  CHECK(api != NULL && api->abi_version == AMDF_ABI_VERSION_12);
  CHECK(api->structure_size >= sizeof(*api));
  amdf_api_t original;
  memcpy(&original, api, sizeof(original));
  const amdf_api_t *again = NULL;
  CHECK(amdf_query_api(1, UINT32_MAX, &again) == AMDF_STATUS_OK);
  CHECK(again == api);
  CHECK(amdf_query_api(2, 2, &again) == API_ERROR(VERSION_MISMATCH));
  CHECK(again == api);
  CHECK(amdf_query_api(3, 3, &again) == API_ERROR(VERSION_MISMATCH));
  CHECK(again == api);
  CHECK(amdf_query_api(5, 5, &again) == API_ERROR(VERSION_MISMATCH));
  CHECK(amdf_query_api(11, 11, &again) == API_ERROR(VERSION_MISMATCH));
  CHECK(amdf_query_api(12, 12, &again) == AMDF_STATUS_OK);
  CHECK(again == api);
  CHECK(amdf_query_api(2, 1, &again) == API_ERROR(INVALID_ARGUMENT));
  CHECK(again == api);
  CHECK(amdf_query_api(2, 2, NULL) == API_ERROR(INVALID_ARGUMENT));
  /* Every negotiated v12 slot exists, including operations whose native
   * backend reports Unsupported. New slots may only append to this table. */
  CHECK(api->instance_create != NULL);
  CHECK(api->instance_destroy != NULL);
  CHECK(api->endpoint_enumerate != NULL);
  CHECK(api->endpoint_open != NULL);
  CHECK(api->endpoint_query_info != NULL);
  CHECK(api->endpoint_close != NULL);
  CHECK(api->query_extension != NULL);
  CHECK(api->endpoint_query_queue_family_info != NULL);
  CHECK(api->device_destroy != NULL);
  CHECK(api->instance_enumerate_memory_scopes != NULL);
  CHECK(api->device_enumerate_memory_scopes != NULL);
  CHECK(api->memory_scope_query_info != NULL);
  CHECK(api->memory_scope_query_device_profile != NULL);
  CHECK(api->memory_create != NULL);
  CHECK(api->memory_import != NULL);
  CHECK(api->memory_query_info != NULL);
  CHECK(api->memory_query_access_info != NULL);
  CHECK(api->memory_export != NULL);
  CHECK(api->external_memory_release != NULL);
  CHECK(api->memory_query_pair_info != NULL);
  CHECK(api->memory_map != NULL);
  CHECK(api->host_mapping_query_info != NULL);
  CHECK(api->host_mapping_cache_control != NULL);
  CHECK(api->host_mapping_destroy != NULL);
  CHECK(api->memory_destroy != NULL);
  CHECK(api->kernel_queue_query_info != NULL);
  CHECK(api->kernel_queue_query_status != NULL);
  CHECK(api->kernel_queue_wait != NULL);
  CHECK(api->kernel_queue_destroy != NULL);
  CHECK(api->user_queue_query_info != NULL);
  CHECK(api->user_queue_map != NULL);
  CHECK(api->user_queue_mapping_query_info != NULL);
  CHECK(api->user_queue_mapping_destroy != NULL);
  CHECK(api->user_queue_query_status != NULL);
  CHECK(api->user_queue_wait_consumed != NULL);
  CHECK(api->user_queue_destroy != NULL);
  CHECK(api->memory_query_address != NULL);
  CHECK(api->memory_scope_query_pair_info != NULL);
  CHECK(api->kernel_queue_refresh_status != NULL);
  CHECK(api->kernel_queue_request_notification != NULL);
  CHECK(api->device_query_memory_reservation_capabilities != NULL);
  CHECK(api->instance_query_host_memory_reservation_capabilities != NULL);
  CHECK(api->memory_reservation_create != NULL);
  CHECK(api->memory_reservation_map != NULL);
  CHECK(api->memory_reservation_query_access_info != NULL);
  CHECK(api->memory_reservation_update != NULL);
  CHECK(api->memory_reservation_detach != NULL);
  CHECK(api->memory_reservation_destroy != NULL);
  CHECK(api->kernel_queue_update_memory != NULL);
  const void *extension = NULL;
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension) == AMDF_STATUS_OK);
  CHECK(extension != NULL);
  const amdf_gpu_api_t *gpu = (const amdf_gpu_api_t *)extension;
  CHECK(gpu->structure_size == offsetof(amdf_gpu_api_t, device_query_queue_clock_info)
        && gpu->extension_version == 1);
  CHECK(gpu->endpoint_query_info && gpu->device_create && gpu->device_query_info);
  CHECK(gpu->user_queue_create);
  CHECK(gpu->kernel_queue_create && gpu->kernel_queue_submit);
  amdf_gpu_api_t original_gpu;
  memcpy(&original_gpu, gpu, sizeof(original_gpu));
  CHECK(api->query_extension(UINT32_MAX, 1, 1, &extension) == API_ERROR(UNSUPPORTED));
  CHECK(extension == gpu);
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 2, 2, &extension) == AMDF_STATUS_OK);
  CHECK(((const amdf_gpu_api_t *)extension)->extension_version == 2);
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 3, 3, &extension) == AMDF_STATUS_OK);
  CHECK(((const amdf_gpu_api_t *)extension)->extension_version == 3);
  CHECK(api->query_extension(AMDF_EXTENSION_OBSERVATION, 1, 1, &extension) == AMDF_STATUS_OK);
  CHECK(((const amdf_observation_api_t *)extension)->query_host_clock != NULL);

  amdf_instance_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  amdf_instance_t *instance = NULL;
  for (uint32_t lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
       lifetime <= AMDF_NATIVE_LIFETIME_INSTANCE; ++lifetime) {
    create.native_lifetime = lifetime;
    CHECK(api->instance_create(&create, &instance) == AMDF_STATUS_OK);
    CHECK(instance != NULL);
    uint32_t count = UINT32_MAX;
    CHECK(api->instance_enumerate_memory_scopes(instance, 0, NULL, &count) == API_ERROR(BUFFER_TOO_SMALL));
    CHECK(count != 0);
    amdf_memory_scope_t *scope = NULL;
    CHECK(api->instance_enumerate_memory_scopes(instance, 1, &scope, &count) == AMDF_STATUS_OK);
    amdf_memory_scope_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
    CHECK(api->memory_scope_query_info(scope, &info) == AMDF_STATUS_OK);
    CHECK(info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM);
    CHECK(check_short_final_reservation(api, instance) == 0);
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    amdf_memory_scope_info_t before;
    memcpy(&before, &info, sizeof(before));
    CHECK(api->memory_scope_query_info(scope, &info) == API_ERROR(INVALID_ARGUMENT));
    CHECK(memcmp(&before, &info, sizeof(info)) == 0);
    CHECK(api->instance_destroy(instance) == AMDF_STATUS_OK);
    instance = NULL;
  }
  create.reserved = 1;
  amdf_instance_t *sentinel = (amdf_instance_t *)&create;
  instance = sentinel;
  CHECK(api->instance_create(&create, &instance) == API_ERROR(INVALID_ARGUMENT));
  CHECK(instance == sentinel);
  create.reserved = 0;
  create.native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  struct allocation_counts counts = {0};
  create.host_allocator.user_data = &counts;
  create.host_allocator.allocate = tracked_allocate;
  create.host_allocator.free = tracked_free;
  CHECK(api->instance_create(&create, &instance) == AMDF_STATUS_OK);
  amdf_memory_scope_t *scope = NULL;
  uint32_t scope_count = 0;
  CHECK(api->instance_enumerate_memory_scopes(instance, 1, &scope,
                                               &scope_count) == AMDF_STATUS_OK);
  amdf_memory_create_info_t memory_create;
  INIT(memory_create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  memory_create.byte_length = 4097;
  amdf_memory_t *memory = NULL;
  amdf_memory_release_info_t release_info = {UINT64_MAX, UINT64_MAX};
  CHECK(api->memory_create(scope, &memory_create, &memory,
                           &release_info) == AMDF_STATUS_OK);
  CHECK(release_info.status == AMDF_STATUS_OK &&
        release_info.abandoned_byte_length == 0);
  CHECK(api->memory_destroy(memory, &release_info) == AMDF_STATUS_OK);
  CHECK(release_info.status == AMDF_STATUS_OK &&
        release_info.abandoned_byte_length == 0);
  memory_create.byte_length = 0;
  memory = (amdf_memory_t *)&memory_create;
  release_info.status = UINT64_MAX;
  release_info.abandoned_byte_length = UINT64_MAX;
  CHECK(api->memory_create(scope, &memory_create, &memory, &release_info) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(memory == (amdf_memory_t *)&memory_create);
  CHECK(release_info.status == AMDF_STATUS_OK &&
        release_info.abandoned_byte_length == 0);
  release_info.status = UINT64_MAX;
  release_info.abandoned_byte_length = UINT64_MAX;
  CHECK(api->memory_destroy(NULL, &release_info) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(release_info.status == AMDF_STATUS_OK &&
        release_info.abandoned_byte_length == 0);
  CHECK(api->instance_destroy(instance) == AMDF_STATUS_OK);
  CHECK(counts.allocations > 1 && counts.allocations == counts.frees);
  instance = sentinel;
  create.host_allocator.allocate = NULL;
  CHECK(api->instance_create(&create, &instance) == API_ERROR(INVALID_ARGUMENT));
  CHECK(instance == sentinel);
  CHECK(memcmp(&original, api, sizeof(original)) == 0);
  CHECK(memcmp(&original_gpu, gpu, sizeof(original_gpu)) == 0);
  puts("AMDF v12 negotiation, allocator, release record, host reservation, "
       "and inert lifetimes passed");
  return 0;
}
