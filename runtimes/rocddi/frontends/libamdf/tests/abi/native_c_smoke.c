/* AMDF v3 negotiation and inert instance checks. No endpoint is activated. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
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

int main(void) {
  const amdf_api_t *api = NULL;
  CHECK(amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api) == AMDF_STATUS_OK);
  CHECK(api != NULL && api->abi_version == AMDF_ABI_VERSION_3);
  CHECK(api->structure_size >= sizeof(*api));
  amdf_api_t original;
  memcpy(&original, api, sizeof(original));
  const amdf_api_t *again = NULL;
  CHECK(amdf_query_api(1, UINT32_MAX, &again) == AMDF_STATUS_OK);
  CHECK(again == api);
  CHECK(amdf_query_api(2, 2, &again) == API_ERROR(VERSION_MISMATCH));
  CHECK(again == api);
  CHECK(amdf_query_api(3, 3, &again) == AMDF_STATUS_OK);
  CHECK(again == api);
  CHECK(amdf_query_api(2, 1, &again) == API_ERROR(INVALID_ARGUMENT));
  CHECK(again == api);
  CHECK(amdf_query_api(2, 2, NULL) == API_ERROR(INVALID_ARGUMENT));
  /* Every v3 slot exists, including operations whose implementation reports
   * Unsupported for this provider. New slots may only append to this table. */
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
  const void *extension = NULL;
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension) == AMDF_STATUS_OK);
  CHECK(extension != NULL);
  const amdf_gpu_api_t *gpu = (const amdf_gpu_api_t *)extension;
  CHECK(gpu->structure_size >= sizeof(*gpu) && gpu->extension_version == 1);
  CHECK(gpu->endpoint_query_info && gpu->device_create && gpu->device_query_info);
  CHECK(gpu->user_queue_create);
  CHECK(gpu->kernel_queue_create && gpu->kernel_queue_submit);
  amdf_gpu_api_t original_gpu;
  memcpy(&original_gpu, gpu, sizeof(original_gpu));
  CHECK(api->query_extension(UINT32_MAX, 1, 1, &extension) == API_ERROR(UNSUPPORTED));
  CHECK(extension == gpu);
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 2, 2, &extension) == API_ERROR(VERSION_MISMATCH));
  CHECK(extension == gpu);

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
  CHECK(memcmp(&original, api, sizeof(original)) == 0);
  CHECK(memcmp(&original_gpu, gpu, sizeof(original_gpu)) == 0);
  puts("AMDF negotiation, immutable slots, output preservation, and both inert lifetimes passed");
  return 0;
}
