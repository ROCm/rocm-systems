/* Opt-in empty native queue qualification. No packet, index, or doorbell is
 * written. AMDF_REQUIRE_GPU=1 requires every advertised native user queue. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <amdf/amdf.h>
#include <amdf/gpu.h>

#define CHECK(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "AMDF queue check failed at line %d: %s\n", __LINE__, #condition); \
    return 1; \
  } \
} while (0)
#define INIT(record, tag) do { \
  memset(&(record), 0, sizeof(record)); \
  (record).type = (tag); (record).structure_size = (uint32_t)sizeof(record); \
} while (0)
#define API_ERROR(code) amdf_make_api_status(AMDF_STATUS_CODE_##code)

static int check_family(const amdf_api_t *api, const amdf_gpu_api_t *gpu,
                        amdf_device_t *device, const amdf_queue_family_info_t *family) {
  amdf_gpu_user_queue_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  create.queue_family_ordinal = family->ordinal;
  create.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  create.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  create.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  create.ring_byte_length = family->minimum_ring_byte_length;
  amdf_user_queue_t *queues[2] = {NULL, NULL};
  amdf_user_queue_mapping_t *mappings[2] = {NULL, NULL};
  amdf_user_queue_info_t info[2];
  amdf_user_queue_mapping_info_t mapped[2];
  for (unsigned i = 0; i != 2; ++i) {
    amdf_status_t queue_status = gpu->user_queue_create(device, &create, &queues[i]);
    if (queue_status != AMDF_STATUS_OK)
      fprintf(stderr, "family ordinal=%u command=%u iteration=%u status=0x%016llx\n",
              family->ordinal, family->command_type, i,
              (unsigned long long)queue_status);
    CHECK(queue_status == AMDF_STATUS_OK);
    INIT(info[i], AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
    CHECK(api->user_queue_query_info(queues[i], &info[i]) == AMDF_STATUS_OK);
    CHECK(info[i].command_type == family->command_type);
    CHECK(info[i].format_version == family->format_version);
    CHECK(info[i].ring_byte_length == create.ring_byte_length);
    CHECK(amdf_queue_id_is_valid(&info[i].queue_id));
    CHECK(api->user_queue_map(queues[i], NULL, &mappings[i]) == AMDF_STATUS_OK);
    INIT(mapped[i], AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
    CHECK(api->user_queue_mapping_query_info(mappings[i], &mapped[i]) == AMDF_STATUS_OK);
    CHECK(amdf_queue_id_is_equal(&info[i].queue_id, &mapped[i].queue_id));
    CHECK(mapped[i].ring_address != 0 && mapped[i].doorbell_address != 0);
    CHECK(mapped[i].read_index_address != 0 && mapped[i].write_index_address != 0);
    CHECK(mapped[i].index_bits == 64 && mapped[i].doorbell_bits == 64);
    CHECK(mapped[i].read_index_address % 8 == 0 && mapped[i].write_index_address % 8 == 0);
    CHECK(mapped[i].doorbell_address % 8 == 0);
    CHECK(api->user_queue_destroy(queues[i]) == API_ERROR(BUSY));
    CHECK(api->device_destroy(device) == API_ERROR(BUSY));
    amdf_user_queue_status_t status;
    INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
    CHECK(api->user_queue_query_status(queues[i], &status) == AMDF_STATUS_OK);
    CHECK(status.state == AMDF_QUEUE_STATE_ACTIVE && status.terminal_status == AMDF_STATUS_OK);
    CHECK(status.producer_index == 0 && status.consumed_index == 0);
    CHECK(api->user_queue_wait_consumed(queues[i], 0, 0, 0) == AMDF_STATUS_OK);
    amdf_user_queue_info_t invalid, saved;
    INIT(invalid, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
    memcpy(&saved, &invalid, sizeof(saved));
    CHECK(api->user_queue_query_info(queues[i], &invalid) == API_ERROR(INVALID_ARGUMENT));
    CHECK(memcmp(&saved, &invalid, sizeof(saved)) == 0);
  }
  CHECK(!amdf_queue_id_is_equal(&info[0].queue_id, &info[1].queue_id));
  CHECK(mapped[0].ring_address != mapped[1].ring_address);
  CHECK(mapped[0].read_index_address != mapped[1].read_index_address);
  for (unsigned i = 0; i != 2; ++i) {
    CHECK(api->user_queue_mapping_destroy(mappings[i]) == AMDF_STATUS_OK);
    CHECK(api->user_queue_destroy(queues[i]) == AMDF_STATUS_OK);
  }
  return 0;
}

static int check_device_producer(const amdf_api_t *api, const amdf_gpu_api_t *gpu,
                                 amdf_device_t *device,
                                 const amdf_gpu_device_info_t *device_info,
                                 const amdf_queue_family_info_t *family) {
  amdf_gpu_user_queue_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  create.queue_family_ordinal = family->ordinal;
  create.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  create.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  create.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  create.ring_byte_length = family->minimum_ring_byte_length;
  amdf_user_queue_t *queue = NULL;
  CHECK(gpu->user_queue_create(device, &create, &queue) == AMDF_STATUS_OK);
  amdf_user_queue_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
  CHECK(api->user_queue_query_info(queue, &info) == AMDF_STATUS_OK);
  CHECK(info.capabilities & AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER);
  CHECK(info.command_type == family->command_type);
  CHECK(info.format_features == family->format_features);
  amdf_user_queue_mapping_t *mapping = NULL;
  CHECK(api->user_queue_map(queue, device, &mapping) == AMDF_STATUS_OK);
  amdf_user_queue_mapping_info_t mapped;
  INIT(mapped, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CHECK(api->user_queue_mapping_query_info(mapping, &mapped) == AMDF_STATUS_OK);
  CHECK(amdf_device_id_is_equal(&mapped.producer_device_id, &device_info->id));
  CHECK(amdf_queue_id_is_equal(&mapped.queue_id, &info.queue_id));
  CHECK(mapped.producer_reset_epoch == device_info->reset_epoch);
  CHECK(mapped.queue_reset_epoch == info.reset_epoch);
  CHECK(mapped.command_type == family->command_type);
  CHECK(mapped.format_version == family->format_version);
  CHECK(mapped.format_features == family->format_features);
  CHECK(mapped.ring_byte_length == create.ring_byte_length);
  CHECK(mapped.ring_address != 0 && mapped.read_index_address != 0);
  CHECK(mapped.write_index_address != 0 && mapped.doorbell_address != 0);
  CHECK((mapped.read_index_address | mapped.write_index_address |
         mapped.doorbell_address) % 8 == 0);
  CHECK(mapped.index_bits == 64 && mapped.doorbell_bits == 64);
  CHECK(api->user_queue_destroy(queue) == API_ERROR(BUSY));
  CHECK(api->user_queue_mapping_destroy(mapping) == AMDF_STATUS_OK);
  CHECK(api->user_queue_destroy(queue) == AMDF_STATUS_OK);
  return 0;
}

int main(int argc, char **argv) {
  const amdf_api_t *api = NULL;
  CHECK(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                       &api) == AMDF_STATUS_OK);
  if (getenv("AMDF_REQUIRE_GPU") == NULL) {
    puts("AMDF native GPU queues disabled; set AMDF_REQUIRE_GPU=1 to require qualification");
    return 0;
  }
  const void *extension = NULL;
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension) == AMDF_STATUS_OK);
  const amdf_gpu_api_t *gpu = (const amdf_gpu_api_t *)extension;
  amdf_instance_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  CHECK(argc == 1 || (argc == 2 && strcmp(argv[1], "instance") == 0));
  create.native_lifetime = argc == 2 ? AMDF_NATIVE_LIFETIME_INSTANCE
                                     : AMDF_NATIVE_LIFETIME_PROCESS;
  amdf_instance_t *instance = NULL;
  CHECK(api->instance_create(&create, &instance) == AMDF_STATUS_OK);
  uint32_t count = 0;
  CHECK(api->endpoint_enumerate(instance, 0, NULL, &count) == AMDF_STATUS_OK);
  CHECK(count != 0);
  amdf_endpoint_summary_t *summaries = (amdf_endpoint_summary_t *)calloc(count, sizeof(*summaries));
  CHECK(summaries != NULL);
  CHECK(api->endpoint_enumerate(instance, count, summaries, &count) == AMDF_STATUS_OK);
  unsigned families = 0;
  unsigned device_producer_families = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_GPU) continue;
    amdf_endpoint_t *endpoint = NULL;
    CHECK(api->endpoint_open(instance, &summaries[i].id, &endpoint) == AMDF_STATUS_OK);
    amdf_endpoint_info_t endpoint_info;
    INIT(endpoint_info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
    CHECK(api->endpoint_query_info(endpoint, &endpoint_info) == AMDF_STATUS_OK);
    amdf_gpu_device_create_info_t device_create;
    INIT(device_create, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
    amdf_device_t *device = NULL;
    CHECK(gpu->device_create(endpoint, &device_create, &device) == AMDF_STATUS_OK);
    amdf_gpu_device_info_t device_info;
    INIT(device_info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO);
    CHECK(gpu->device_query_info(device, &device_info) == AMDF_STATUS_OK);
    CHECK(api->endpoint_close(endpoint) == API_ERROR(BUSY));
    for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count; ++ordinal) {
      amdf_queue_family_info_t family;
      INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
      CHECK(api->endpoint_query_queue_family_info(endpoint, ordinal, &family) == AMDF_STATUS_OK);
      if (!(family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ||
          !(family.user_queue_capabilities & AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) ||
          !(family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE)) continue;
      if (family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
          family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) continue;
      CHECK(check_family(api, gpu, device, &family) == 0);
      if (family.user_queue_capabilities & AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) {
        CHECK(check_device_producer(api, gpu, device, &device_info, &family) == 0);
        ++device_producer_families;
      }
      ++families;
    }
    CHECK(api->device_destroy(device) == AMDF_STATUS_OK);
    if (device_info.features & AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION) {
      device = NULL;
      CHECK(gpu->device_create(endpoint, &device_create, &device) == AMDF_STATUS_OK);
      CHECK(api->device_destroy(device) == AMDF_STATUS_OK);
    }
    CHECK(api->endpoint_close(endpoint) == AMDF_STATUS_OK);
  }
  free(summaries);
  CHECK(families != 0);
  CHECK(api->instance_destroy(instance) == AMDF_STATUS_OK);
  printf("AMDF fresh empty queues passed for %u GPU families; %u device-producer mappings\n",
         families, device_producer_families);
  return 0;
}
