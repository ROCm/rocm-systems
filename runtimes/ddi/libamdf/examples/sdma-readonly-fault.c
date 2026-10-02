/* KFD MEMORY-event qualification through a valid SDMA write to read-only GPU
 * memory. This process intentionally leaves faulted resources to OS teardown. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INIT(record, tag) do { \
  memset(&(record), 0, sizeof(record)); \
  (record).type = (tag); \
  (record).structure_size = (uint32_t)sizeof(record); \
} while (0)
#define REQUIRE(expression) do { \
  if (!(expression)) { \
    fprintf(stderr, "SDMA fault: line %d: %s\n", __LINE__, #expression); \
    return 1; \
  } \
} while (0)
#define CALL(expression) do { \
  amdf_status_t call_status = (expression); \
  if (call_status != AMDF_STATUS_OK) { \
    fprintf(stderr, "SDMA fault: line %d: %s: status=0x%016" PRIx64 "\n", \
            __LINE__, #expression, call_status); \
    return 1; \
  } \
} while (0)

enum { BUFFER_BYTES = 4096, SLOT_WORDS = 32 };
static const uint64_t timeout_ns = UINT64_C(5000000000);
static const amdf_queue_format_features_t required_format_features =
    AMDF_GPU_SDMA_FORMAT_FEATURE_GCR |
    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;

struct buffer {
  amdf_memory_t *memory;
  amdf_host_mapping_t *mapping;
  amdf_host_mapping_info_t host;
  uint64_t address;
};

struct workload {
  const amdf_api_t *api;
  const amdf_gpu_api_t *gpu;
  amdf_instance_t *instance;
  amdf_endpoint_t *endpoint;
  amdf_device_t *device;
  amdf_memory_scope_t *scope;
  amdf_user_queue_t *queue;
  amdf_user_queue_mapping_t *mapping;
  amdf_user_queue_mapping_info_t transport;
  struct buffer source;
  struct buffer target;
  uint32_t family;
  amdf_queue_format_features_t format_features;
};

static uint64_t now_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    perror("clock_gettime");
    exit(1);
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
         (uint64_t)value.tv_nsec;
}

static void relax_cpu(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("pause");
#endif
}

static int select_gfx1201(struct workload *w) {
  uint32_t count = 0;
  CALL(w->api->endpoint_enumerate(w->instance, 0, NULL, &count));
  amdf_endpoint_summary_t *endpoints =
      (amdf_endpoint_summary_t *)calloc(count, sizeof(*endpoints));
  REQUIRE(endpoints != NULL);
  amdf_status_t status =
      w->api->endpoint_enumerate(w->instance, count, endpoints, &count);
  if (status != AMDF_STATUS_OK) {
    free(endpoints);
    REQUIRE(status == AMDF_STATUS_OK);
  }
  for (uint32_t index = 0; index < count; ++index) {
    if (endpoints[index].engine_kind != AMDF_ENGINE_KIND_GPU) continue;
    status = w->api->endpoint_open(
        w->instance, &endpoints[index].id, &w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    amdf_gpu_endpoint_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
    status = w->gpu->endpoint_query_info(w->endpoint, &info);
    if (status != AMDF_STATUS_OK) break;
    if (info.gfx_ip.major == 12 && info.gfx_ip.minor == 0 &&
        info.gfx_ip.stepping == 1)
      break;
    status = w->api->endpoint_close(w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    w->endpoint = NULL;
  }
  free(endpoints);
  REQUIRE(status == AMDF_STATUS_OK && w->endpoint != NULL);
  return 0;
}

static int select_sdma_family(struct workload *w) {
  amdf_endpoint_info_t endpoint;
  INIT(endpoint, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &endpoint));
  for (w->family = 0; w->family < endpoint.queue_family_count; ++w->family) {
    amdf_queue_family_info_t family;
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(
        w->endpoint, w->family, &family));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
        family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 &&
        family.format_features == required_format_features &&
        (family.roles & AMDF_QUEUE_ROLE_TRANSFER) != 0 &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0) {
      w->format_features = family.format_features;
      return 0;
    }
  }
  REQUIRE(w->family < endpoint.queue_family_count);
  return 1;
}

static int create_buffer(struct workload *w, struct buffer *buffer,
                         amdf_memory_access_t permissions) {
  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access = permissions;
  access.requirements.flags =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;

  amdf_memory_scope_info_t scope;
  INIT(scope, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->scope, &scope));
  uint32_t ordinal;
  for (ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0)
      break;
  }
  REQUIRE(ordinal < scope.memory_profile_count);

  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = ordinal;
  create.byte_length = BUFFER_BYTES;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.access_count = 1;
  create.accesses = &access;
  CALL(w->api->memory_create(w->scope, &create, &buffer->memory));

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = BUFFER_BYTES;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(buffer->memory, &map, &buffer->mapping));
  INIT(buffer->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(buffer->mapping, &buffer->host));
  REQUIRE(buffer->host.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
  CALL(w->api->memory_query_address(
      buffer->memory, 0, AMDF_MEMORY_ADDRESS_GPU, &buffer->address));
  return 0;
}

static int create_queue(struct workload *w) {
  amdf_gpu_user_queue_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  create.queue_family_ordinal = w->family;
  create.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  create.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  create.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  create.ring_byte_length = 1024;
  CALL(w->gpu->user_queue_create(w->device, &create, &w->queue));
  amdf_user_queue_info_t queue_info;
  INIT(queue_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
  CALL(w->api->user_queue_query_info(w->queue, &queue_info));
  REQUIRE(queue_info.format_features == w->format_features);
  CALL(w->api->user_queue_map(w->queue, NULL, &w->mapping));
  INIT(w->transport, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(w->mapping, &w->transport));
  REQUIRE(w->transport.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  REQUIRE(w->transport.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1);
  REQUIRE(w->transport.format_features == w->format_features);
  REQUIRE(w->transport.ring_byte_length == 1024);
  REQUIRE(w->transport.index_bits == 64 && w->transport.doorbell_bits == 64);
  return 0;
}

static int prepare(struct workload *w, amdf_native_lifetime_t lifetime) {
  CALL(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                      &w->api));
  const void *extension = NULL;
  CALL(w->api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension));
  w->gpu = (const amdf_gpu_api_t *)extension;

  amdf_instance_create_info_t instance;
  INIT(instance, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instance.native_lifetime = lifetime;
  CALL(w->api->instance_create(&instance, &w->instance));
  uint32_t count = 0;
  CALL(w->api->instance_enumerate_memory_scopes(
      w->instance, 1, &w->scope, &count));
  REQUIRE(count == 1);
  REQUIRE(select_gfx1201(w) == 0);
  REQUIRE(select_sdma_family(w) == 0);

  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  REQUIRE(create_buffer(w, &w->source, AMDF_MEMORY_ACCESS_READ) == 0);
  REQUIRE(create_buffer(w, &w->target, AMDF_MEMORY_ACCESS_READ) == 0);
  REQUIRE(create_queue(w) == 0);
  return 0;
}

static void encode_fault(uint32_t words[SLOT_WORDS], uint64_t source,
                         uint64_t target) {
  const uint32_t commands[SLOT_WORDS] = {
    0x111, 0, 0xc3c00000, 0, 0,
    1, 3, 0, (uint32_t)source, (uint32_t)(source >> 32),
    (uint32_t)target, (uint32_t)(target >> 32)
  };
  memcpy(words, commands, sizeof(commands));
}

static int run_fault(struct workload *w) {
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell =
      (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  uint32_t *source = (uint32_t *)w->source.host.pointer;
  uint32_t *target = (uint32_t *)w->target.host.pointer;
  source[0] = UINT32_C(0x12345678);
  target[0] = UINT32_C(0xa5a5a5a5);

  uint32_t commands[SLOT_WORDS];
  encode_fault(commands, w->source.address, w->target.address);
  memcpy((void *)(uintptr_t)w->transport.ring_address,
         commands, sizeof(commands));
  __atomic_store_n(write_index, sizeof(commands), __ATOMIC_RELEASE);
  __atomic_store_n(doorbell, sizeof(commands), __ATOMIC_RELEASE);

  const amdf_status_t lost =
      amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  const uint64_t begin = now_ns();
  amdf_user_queue_status_t status;
  for (;;) {
    INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
    amdf_status_t observed =
        w->api->user_queue_query_status(w->queue, &status);
    if (observed == lost) continue;
    REQUIRE(observed == AMDF_STATUS_OK);
    if (status.state == AMDF_QUEUE_STATE_DEVICE_LOST) break;
    REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
    if (now_ns() - begin >= timeout_ns) {
      fputs("SDMA fault did not reach AMDF DEVICE_LOST\n", stderr);
      return 1;
    }
    relax_cpu();
  }
  REQUIRE(status.terminal_status == lost);

  for (unsigned observation = 0; observation < 8; ++observation) {
    INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
    CALL(w->api->user_queue_query_status(w->queue, &status));
    REQUIRE(status.state == AMDF_QUEUE_STATE_DEVICE_LOST);
    REQUIRE(status.terminal_status == lost);
  }

  amdf_gpu_device_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO);
  CALL(w->gpu->device_query_info(w->device, &device));
  REQUIRE(device.reset_epoch == 2);
  puts("Verified a read-only SDMA fault produces sticky AMDF device loss");
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 2 || (argc == 2 && strcmp(argv[1], "instance") != 0)) {
    fprintf(stderr, "Usage: %s [instance]\n", argv[0]);
    return 2;
  }
  const amdf_native_lifetime_t lifetime = argc == 2 ?
      AMDF_NATIVE_LIFETIME_INSTANCE : AMDF_NATIVE_LIFETIME_PROCESS;
  struct workload workload;
  memset(&workload, 0, sizeof(workload));
  int result = prepare(&workload, lifetime);
  if (result == 0) result = run_fault(&workload);
  /* The fault evicts this process device. Its objects intentionally remain
   * alive until process exit; no completion or safe retirement was observed. */
  return result;
}
