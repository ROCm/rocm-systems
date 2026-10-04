/* GFX1201 SDMA qualification for private and host-visible LOCAL memory.
 * Both paths exercise SYSTEM -> LOCAL -> SYSTEM. The public path also checks
 * the CPU view and publishes CPU writes from VRAM to a second SDMA copy. */
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
    fprintf(stderr, "SDMA LOCAL: line %d: %s\n", __LINE__, #expression); \
    return 1; \
  } \
} while (0)
#define CALL(expression) do { \
  amdf_status_t call_status = (expression); \
  if (call_status != AMDF_STATUS_OK) { \
    fprintf(stderr, "SDMA LOCAL: line %d: %s: status=0x%016" PRIx64 "\n", \
            __LINE__, #expression, call_status); \
    return 1; \
  } \
} while (0)

enum { BUFFER_BYTES = 65536, SLOT_WORDS = 32, ITERATIONS = 64 };
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
  amdf_memory_scope_t *system_scope;
  amdf_memory_scope_t *local_scope;
  amdf_user_queue_t *queue;
  amdf_user_queue_mapping_t *mapping;
  amdf_user_queue_mapping_info_t transport;
  struct buffer source;
  struct buffer local;
  struct buffer target;
  struct buffer completion;
  uint32_t family;
  amdf_queue_format_features_t format_features;
  int in_flight;
  int public_local;
  int instance_lifetime;
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

static int select_scopes_and_family(struct workload *w) {
  uint32_t count = 0;
  CALL(w->api->instance_enumerate_memory_scopes(
      w->instance, 1, &w->system_scope, &count));
  REQUIRE(count == 1);
  count = 0;
  CALL(w->api->device_enumerate_memory_scopes(
      w->device, 1, &w->local_scope, &count));
  REQUIRE(count == 1);

  amdf_memory_scope_info_t local;
  INIT(local, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->local_scope, &local));
  REQUIRE(local.kind == AMDF_MEMORY_SCOPE_KIND_LOCAL);
  REQUIRE(local.memory_profile_count != 0);

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
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0 &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0 &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
      w->format_features = family.format_features;
      return 0;
    }
  }
  REQUIRE(w->family < endpoint.queue_family_count);
  return 1;
}

static int create_system_buffer(struct workload *w, struct buffer *buffer,
                                amdf_memory_access_t permissions,
                                uint64_t bytes) {
  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access = permissions;
  access.requirements.flags =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;

  amdf_memory_scope_info_t scope;
  INIT(scope, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->system_scope, &scope));
  uint32_t ordinal;
  for (ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->system_scope, ordinal, 1, &access, &profile, &capabilities);
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
  create.byte_length = bytes;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.access_count = 1;
  create.accesses = &access;
  CALL(w->api->memory_create(w->system_scope, &create, &buffer->memory));

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(buffer->memory, &map, &buffer->mapping));
  INIT(buffer->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(buffer->mapping, &buffer->host));
  REQUIRE(buffer->host.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
  CALL(w->api->memory_query_address(
      buffer->memory, 0, AMDF_MEMORY_ADDRESS_GPU, &buffer->address));
  return 0;
}

static int create_local_buffer(struct workload *w) {
  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;

  amdf_memory_scope_info_t scope;
  INIT(scope, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->local_scope, &scope));
  uint32_t ordinal;
  for (ordinal = 0; ordinal < scope.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->local_scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
        ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0) ==
            w->public_local &&
        (profile.guaranteed_flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) != 0 &&
        ((profile.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) ==
            w->public_local)
      break;
  }
  REQUIRE(ordinal < scope.memory_profile_count);

  if (w->public_local) {
    amdf_memory_profile_pair_query_t query;
    INIT(query, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY);
    query.memory_profile_ordinal = ordinal;
    query.access_count = 1;
    query.accesses = &access;
    query.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                           AMDF_MEMORY_FLAG_HOST_VISIBLE;
    for (int device_produces = 0; device_produces != 2; ++device_produces) {
      amdf_memory_profile_site_t *host =
          device_produces ? &query.consumer : &query.producer;
      amdf_memory_profile_site_t *device =
          device_produces ? &query.producer : &query.consumer;
      host->kind = AMDF_MEMORY_SITE_KIND_HOST;
      host->value.host_access = AMDF_MEMORY_MAP_FLAG_READ |
                                AMDF_MEMORY_MAP_FLAG_WRITE;
      device->kind = AMDF_MEMORY_SITE_KIND_DEVICE;
      device->value.device.access_ordinal = 0;
      device->value.device.queue_family_ordinal = w->family;
      amdf_memory_pair_info_t pair;
      INIT(pair, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO);
      CALL(w->api->memory_scope_query_pair_info(w->local_scope,
                                                 &query, &pair));
      REQUIRE((pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE)
              != 0);
      REQUIRE(pair.release.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      REQUIRE(pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    }
  }

  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = ordinal;
  create.byte_length = BUFFER_BYTES;
  create.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL |
      (w->public_local ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0);
  create.access_count = 1;
  create.accesses = &access;
  CALL(w->api->memory_create(w->local_scope, &create, &w->local.memory));

  amdf_memory_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CALL(w->api->memory_query_info(w->local.memory, &info));
  REQUIRE(info.memory_class == AMDF_MEMORY_CLASS_LOCAL);
  REQUIRE((info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) != 0);
  REQUIRE(((info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) ==
          w->public_local);

  amdf_memory_access_info_t actual;
  INIT(actual, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO);
  CALL(w->api->memory_query_access_info(w->local.memory, 0, &actual));
  REQUIRE(actual.access == access.requirements.access);
  REQUIRE((actual.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS) != 0);
  CALL(w->api->memory_query_address(
      w->local.memory, 0, AMDF_MEMORY_ADDRESS_GPU, &w->local.address));

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = BUFFER_BYTES;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ;
  if (w->public_local) {
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    CALL(w->api->memory_map(w->local.memory, &map, &w->local.mapping));
    INIT(w->local.host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
    CALL(w->api->host_mapping_query_info(w->local.mapping, &w->local.host));
    REQUIRE(w->local.host.pointer != NULL);
    REQUIRE(w->local.host.byte_length == BUFFER_BYTES);
    REQUIRE(w->local.host.cacheability ==
            AMDF_HOST_CACHEABILITY_WRITE_COMBINED);
    REQUIRE(w->local.host.flush.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    REQUIRE(w->local.host.invalidate.kind ==
            AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  } else {
    amdf_host_mapping_t *sentinel = (amdf_host_mapping_t *)&map;
    amdf_host_mapping_t *mapping = sentinel;
    REQUIRE(w->api->memory_map(w->local.memory, &map, &mapping) ==
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    REQUIRE(mapping == sentinel);
  }
  return 0;
}

static int check_system_pair(struct workload *w, struct buffer *buffer,
                             int device_produces) {
  amdf_memory_site_t host;
  amdf_memory_site_t device;
  INIT(host, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  INIT(device, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  host.kind = AMDF_MEMORY_SITE_KIND_HOST;
  host.value.host_mapping = buffer->mapping;
  device.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  device.value.device.memory = buffer->memory;
  device.value.device.queue_family_ordinal = w->family;
  amdf_memory_pair_info_t pair;
  INIT(pair, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO);
  CALL(w->api->memory_query_pair_info(
      device_produces ? &device : &host,
      device_produces ? &host : &device, &pair));
  REQUIRE((pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE) != 0);
  const amdf_cache_transition_t *host_transition =
      device_produces ? &pair.acquire : &pair.release;
  const amdf_cache_transition_t *gpu_transition =
      device_produces ? &pair.release : &pair.acquire;
  REQUIRE(host_transition->kind == AMDF_CACHE_TRANSITION_KIND_NONE);
  REQUIRE(gpu_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(gpu_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  return 0;
}

static int check_public_local_pair(struct workload *w,
                                   int device_produces) {
  amdf_memory_site_t host;
  amdf_memory_site_t device;
  INIT(host, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  INIT(device, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  host.kind = AMDF_MEMORY_SITE_KIND_HOST;
  host.value.host_mapping = w->local.mapping;
  device.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  device.value.device.memory = w->local.memory;
  device.value.device.queue_family_ordinal = w->family;
  amdf_memory_pair_info_t pair;
  INIT(pair, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO);
  CALL(w->api->memory_query_pair_info(
      device_produces ? &device : &host,
      device_produces ? &host : &device, &pair));
  REQUIRE((pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE) != 0);
  const amdf_cache_transition_t *host_transition =
      device_produces ? &pair.acquire : &pair.release;
  const amdf_cache_transition_t *gpu_transition =
      device_produces ? &pair.release : &pair.acquire;
  REQUIRE(host_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(host_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
  REQUIRE(host_transition->host_operation ==
          (device_produces ? AMDF_HOST_CACHE_OPERATION_INVALIDATE
                           : AMDF_HOST_CACHE_OPERATION_FLUSH));
  REQUIRE(host_transition->host_fence_after == AMDF_HOST_CACHE_FENCE_X86_MFENCE);
  REQUIRE(gpu_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(gpu_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  REQUIRE(gpu_transition->operation ==
          (device_produces ? AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
                           : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
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

static int prepare(struct workload *w) {
  CALL(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                      &w->api));
  const void *extension = NULL;
  CALL(w->api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension));
  w->gpu = (const amdf_gpu_api_t *)extension;

  amdf_instance_create_info_t instance;
  INIT(instance, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instance.native_lifetime = w->instance_lifetime;
  CALL(w->api->instance_create(&instance, &w->instance));
  REQUIRE(select_gfx1201(w) == 0);

  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  REQUIRE(select_scopes_and_family(w) == 0);
  REQUIRE(create_system_buffer(
      w, &w->source, AMDF_MEMORY_ACCESS_READ, BUFFER_BYTES) == 0);
  REQUIRE(create_local_buffer(w) == 0);
  REQUIRE(create_system_buffer(
      w, &w->target, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      BUFFER_BYTES) == 0);
  REQUIRE(create_system_buffer(
      w, &w->completion, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      4096) == 0);
  REQUIRE(check_system_pair(w, &w->source, 0) == 0);
  REQUIRE(check_system_pair(w, &w->target, 1) == 0);
  REQUIRE(check_system_pair(w, &w->completion, 1) == 0);
  if (w->public_local) {
    REQUIRE(check_public_local_pair(w, 0) == 0);
    REQUIRE(check_public_local_pair(w, 1) == 0);
  }
  REQUIRE(create_queue(w) == 0);
  return 0;
}

static void encode(uint32_t words[SLOT_WORDS], uint64_t source,
                   uint64_t local, uint64_t target, uint32_t length,
                   uint64_t fence, uint32_t token) {
  const uint32_t commands[SLOT_WORDS] = {
    0x111, 0, 0xc3c00000, 0, 0,
    1, length - 1, 0, (uint32_t)source, (uint32_t)(source >> 32),
    (uint32_t)local, (uint32_t)(local >> 32),
    1, length - 1, 0, (uint32_t)local, (uint32_t)(local >> 32),
    (uint32_t)target, (uint32_t)(target >> 32),
    0x111, 0, 0x80400000, 0, 0,
    0x00130005, (uint32_t)fence, (uint32_t)(fence >> 32), token
  };
  memcpy(words, commands, sizeof(commands));
}

static int run(struct workload *w) {
  static const uint32_t lengths[] = {
    1, 4, 63, 64, 65, 257, 4095, 4096, 4097, 65535
  };
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell =
      (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  unsigned char *source = (unsigned char *)w->source.host.pointer;
  unsigned char *target = (unsigned char *)w->target.host.pointer;
  uint64_t producer = __atomic_load_n(write_index, __ATOMIC_ACQUIRE);
  REQUIRE(__atomic_load_n(read_index, __ATOMIC_ACQUIRE) == producer);

  for (unsigned iteration = 0; iteration < ITERATIONS; ++iteration) {
    const uint32_t length =
        lengths[iteration % (sizeof(lengths) / sizeof(lengths[0]))];
    for (uint32_t index = 0; index < BUFFER_BYTES; ++index)
      source[index] = (unsigned char)(index * 17 + iteration * 31);
    memset(target, 0xa5, BUFFER_BYTES);
    __atomic_store_n(completion, 0, __ATOMIC_RELAXED);

    uint32_t commands[SLOT_WORDS];
    encode(commands, w->source.address, w->local.address, w->target.address,
           length, w->completion.address, iteration + 1);
    unsigned char *slot =
        (unsigned char *)(uintptr_t)w->transport.ring_address +
        (producer & (w->transport.ring_byte_length - 1));
    const uint64_t begin = now_ns();
    memcpy(slot, commands, sizeof(commands));
    producer += sizeof(commands);
    w->in_flight = 1;
    __atomic_store_n(write_index, producer, __ATOMIC_RELEASE);
    __atomic_store_n(doorbell, producer, __ATOMIC_RELEASE);

    unsigned polls = 0;
    while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != iteration + 1 ||
           __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < producer) {
      if ((++polls & 255u) == 0 && now_ns() - begin >= timeout_ns) {
        fprintf(stderr,
                "SDMA LOCAL timeout: iteration=%u token=%u consumed=%" PRIu64
                " producer=%" PRIu64 "\n",
                iteration,
                __atomic_load_n(completion, __ATOMIC_ACQUIRE),
                __atomic_load_n(read_index, __ATOMIC_ACQUIRE), producer);
        return 1;
      }
#if defined(__x86_64__)
      __asm__ __volatile__("pause");
#endif
    }
    w->in_flight = 0;
    REQUIRE(memcmp(source, target, length) == 0);
    for (uint32_t index = length; index < BUFFER_BYTES; ++index)
      REQUIRE(target[index] == 0xa5);

    if (w->public_local) {
      unsigned char *local = (unsigned char *)w->local.host.pointer;
      CALL(w->api->host_mapping_cache_control(
          w->local.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, length));
      REQUIRE(memcmp(source, local, length) == 0);
      for (uint32_t index = 0; index < length; ++index)
        local[index] = (unsigned char)(index * 13 + iteration * 7 + 3);
      CALL(w->api->host_mapping_cache_control(
          w->local.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, length));
      memset(target, 0xa5, BUFFER_BYTES);
      __atomic_store_n(completion, 0, __ATOMIC_RELAXED);
      const uint32_t token = ITERATIONS + iteration + 1;
      encode(commands, w->local.address, w->target.address,
             w->local.address, length, w->completion.address, token);
      slot = (unsigned char *)(uintptr_t)w->transport.ring_address +
             (producer & (w->transport.ring_byte_length - 1));
      memcpy(slot, commands, sizeof(commands));
      producer += sizeof(commands);
      w->in_flight = 1;
      __atomic_store_n(write_index, producer, __ATOMIC_RELEASE);
      __atomic_store_n(doorbell, producer, __ATOMIC_RELEASE);
      const uint64_t second_begin = now_ns();
      polls = 0;
      while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != token ||
             __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < producer) {
        if ((++polls & 255u) == 0 && now_ns() - second_begin >= timeout_ns) {
          fprintf(stderr,
                  "SDMA public LOCAL timeout: iteration=%u token=%u consumed=%"
                  PRIu64 " producer=%" PRIu64 "\n",
                  iteration, __atomic_load_n(completion, __ATOMIC_ACQUIRE),
                  __atomic_load_n(read_index, __ATOMIC_ACQUIRE), producer);
          return 1;
        }
#if defined(__x86_64__)
        __asm__ __volatile__("pause");
#endif
      }
      w->in_flight = 0;
      for (uint32_t index = 0; index < length; ++index)
        REQUIRE(target[index] ==
                (unsigned char)(index * 13 + iteration * 7 + 3));
      for (uint32_t index = length; index < BUFFER_BYTES; ++index)
        REQUIRE(target[index] == 0xa5);
    }
  }

  CALL(w->api->user_queue_wait_consumed(w->queue, producer, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
  REQUIRE(status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == producer);
  REQUIRE(status.consumed_index == producer);
  printf("Verified %u SDMA %s LOCAL copies%s and %" PRIu64
         " ring wraps\n", ITERATIONS,
         w->public_local ? "public" : "private",
         w->public_local ? " with CPU access" : "",
         (uint64_t)ITERATIONS * (w->public_local ? 2 : 1) *
             sizeof(uint32_t[SLOT_WORDS]) / w->transport.ring_byte_length);
  return 0;
}

static int release_buffer(struct workload *w, struct buffer *buffer) {
  if (buffer->mapping != NULL) {
    CALL(w->api->host_mapping_destroy(buffer->mapping));
    buffer->mapping = NULL;
  }
  if (buffer->memory != NULL) {
    CALL(w->api->memory_destroy(buffer->memory));
    buffer->memory = NULL;
  }
  return 0;
}

static int cleanup(struct workload *w) {
  if (w->in_flight) {
    fputs("Retaining GPU resources after incomplete LOCAL work.\n", stderr);
    return 1;
  }
  if (w->mapping != NULL) {
    CALL(w->api->user_queue_mapping_destroy(w->mapping));
    w->mapping = NULL;
  }
  if (w->queue != NULL) {
    CALL(w->api->user_queue_destroy(w->queue));
    w->queue = NULL;
  }
  REQUIRE(release_buffer(w, &w->completion) == 0);
  REQUIRE(release_buffer(w, &w->target) == 0);
  REQUIRE(release_buffer(w, &w->local) == 0);
  REQUIRE(release_buffer(w, &w->source) == 0);
  if (w->device != NULL) {
    CALL(w->api->device_destroy(w->device));
    w->device = NULL;
  }
  if (w->endpoint != NULL) {
    CALL(w->api->endpoint_close(w->endpoint));
    w->endpoint = NULL;
  }
  if (w->instance != NULL) {
    CALL(w->api->instance_destroy(w->instance));
    w->instance = NULL;
  }
  return 0;
}

int main(int argc, char **argv) {
  struct workload workload;
  memset(&workload, 0, sizeof(workload));
  workload.instance_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  for (int index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "public") == 0) workload.public_local = 1;
    else if (strcmp(argv[index], "private") == 0) workload.public_local = 0;
    else if (strcmp(argv[index], "instance") == 0)
      workload.instance_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
    else if (strcmp(argv[index], "process") == 0)
      workload.instance_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
    else {
      fprintf(stderr, "usage: %s [private|public] [process|instance]\n", argv[0]);
      return 2;
    }
  }
  int result = prepare(&workload);
  if (result == 0) result = run(&workload);
  int cleanup_result = cleanup(&workload);
  return result || cleanup_result;
}
