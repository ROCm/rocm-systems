/* GFX1201 SDMA qualification for caller-owned host pages. PROCESS uses KFD
 * registration; INSTANCE uses a DRM GEM USERPTR mapping in its acquired VM.
 * Both keep ownership of every page with this caller. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define INIT(record, tag)                                                      \
  do {                                                                         \
    memset(&(record), 0, sizeof(record));                                      \
    (record).type = (tag);                                                     \
    (record).structure_size = (uint32_t)sizeof(record);                        \
  } while (0)
#define REQUIRE(expression)                                                    \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "SDMA REGISTER: line %d: %s\n", __LINE__, #expression);  \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define CALL(expression)                                                       \
  do {                                                                         \
    amdf_status_t call_status = (expression);                                  \
    if (call_status != AMDF_STATUS_OK) {                                       \
      fprintf(stderr, "SDMA REGISTER: line %d: %s: status=0x%016" PRIx64 "\n", \
              __LINE__, #expression, call_status);                             \
      return 1;                                                                \
    }                                                                          \
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
  void *allocation;
  size_t allocation_length;
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
  struct buffer completion;
  uint64_t page;
  uint32_t family;
  amdf_queue_format_features_t format_features;
  int in_flight;
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
    if (endpoints[index].engine_kind != AMDF_ENGINE_KIND_GPU)
      continue;
    status =
        w->api->endpoint_open(w->instance, &endpoints[index].id, &w->endpoint);
    if (status != AMDF_STATUS_OK)
      break;
    amdf_gpu_endpoint_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
    status = w->gpu->endpoint_query_info(w->endpoint, &info);
    if (status != AMDF_STATUS_OK)
      break;
    if (info.gfx_ip.major == 12 && info.gfx_ip.minor == 0 &&
        info.gfx_ip.stepping == 1)
      break;
    status = w->api->endpoint_close(w->endpoint);
    if (status != AMDF_STATUS_OK)
      break;
    w->endpoint = NULL;
  }
  free(endpoints);
  REQUIRE(status == AMDF_STATUS_OK && w->endpoint != NULL);
  return 0;
}

static int select_scope_and_family(struct workload *w) {
  uint32_t count = 0;
  CALL(w->api->instance_enumerate_memory_scopes(w->instance, 1, &w->scope,
                                                &count));
  REQUIRE(count == 1 && w->scope != NULL);
  amdf_endpoint_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &info));
  for (w->family = 0; w->family < info.queue_family_count; ++w->family) {
    amdf_queue_family_info_t family;
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(w->endpoint, w->family,
                                                  &family));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
        family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 &&
        family.format_features == required_format_features) {
      w->format_features = family.format_features;
      break;
    }
  }
  REQUIRE(w->family < info.queue_family_count);
  return 0;
}

static uint64_t common_alignment(uintptr_t host, uint64_t device) {
  const uint64_t common_bits = (uint64_t)host | device;
  return common_bits & (~common_bits + 1);
}

static int check_pair(struct workload *w, struct buffer *buffer,
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
  CALL(w->api->memory_query_pair_info(device_produces ? &device : &host,
                                      device_produces ? &host : &device,
                                      &pair));
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

static int register_buffer(struct workload *w, struct buffer *buffer,
                           amdf_memory_access_t permissions, uint64_t bytes,
                           size_t offset) {
  REQUIRE(offset < w->page);
  const uint64_t cover = (offset + bytes + w->page - 1) & ~(w->page - 1);
  buffer->allocation_length = (size_t)cover;
  REQUIRE(posix_memalign(&buffer->allocation, (size_t)w->page,
                         buffer->allocation_length) == 0);
  memset(buffer->allocation, 0, buffer->allocation_length);
  unsigned char *registered = (unsigned char *)buffer->allocation + offset;

  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access = permissions;
  access.requirements.flags =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  amdf_memory_scope_info_t scope_info;
  INIT(scope_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->scope, &scope_info));
  uint32_t ordinal;
  amdf_memory_profile_t profile;
  amdf_memory_access_capabilities_t capabilities;
  for (ordinal = 0; ordinal < scope_info.memory_profile_count; ++ordinal) {
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    const amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
      continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER)
      break;
  }
  REQUIRE(ordinal < scope_info.memory_profile_count);
  REQUIRE(profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM);
  REQUIRE(profile.registration.registered_host_pointer_alignment == 1);
  REQUIRE(profile.registration.minimum_alignment == 1);
  REQUIRE(capabilities.device_address.minimum_alignment == 1);
  REQUIRE((capabilities.guaranteed_flags & (AMDF_MEMORY_FLAG_HOST_COHERENT |
                                            AMDF_MEMORY_FLAG_DEVICE_ADDRESS)) ==
          (AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS));

  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = ordinal;
  create.access_count = 1;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.byte_length = bytes;
  create.minimum_alignment = 1;
  create.registered_host_pointer = registered;
  create.registered_host_cacheability =
      profile.registration.registered_host_cacheability;
  create.accesses = &access;
  CALL(w->api->memory_create(w->scope, &create, &buffer->memory));

  amdf_memory_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CALL(w->api->memory_query_info(buffer->memory, &info));
  REQUIRE(info.memory_class == AMDF_MEMORY_CLASS_SYSTEM);
  REQUIRE(info.flags == AMDF_MEMORY_FLAG_HOST_VISIBLE);
  REQUIRE(info.source_byte_offset == offset);
  REQUIRE(info.byte_length == bytes);
  REQUIRE(info.native_allocation_byte_length == cover);
  REQUIRE(info.native_allocation_granularity == w->page);
  REQUIRE(info.physical_backing_id.words[0] == 0);
  REQUIRE(info.physical_backing_id.words[1] == 0);

  amdf_memory_access_info_t actual;
  INIT(actual, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO);
  CALL(w->api->memory_query_access_info(buffer->memory, 0, &actual));
  REQUIRE(actual.access == permissions);
  REQUIRE((actual.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0);
  REQUIRE((actual.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS) != 0);
  CALL(w->api->memory_query_address(buffer->memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                    &buffer->address));
  REQUIRE(info.alignment ==
          common_alignment((uintptr_t)registered, buffer->address));

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(buffer->memory, &map, &buffer->mapping));
  INIT(buffer->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(buffer->mapping, &buffer->host));
  REQUIRE(buffer->host.pointer == registered);
  REQUIRE(buffer->host.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
  REQUIRE(buffer->host.flush.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
  REQUIRE(buffer->host.invalidate.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
  if (permissions & AMDF_MEMORY_ACCESS_READ)
    REQUIRE(check_pair(w, buffer, 0) == 0);
  if (permissions & AMDF_MEMORY_ACCESS_WRITE)
    REQUIRE(check_pair(w, buffer, 1) == 0);
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
  const long page = sysconf(_SC_PAGESIZE);
  REQUIRE(page > 0 && ((uint64_t)page & ((uint64_t)page - 1)) == 0);
  w->page = (uint64_t)page;
  CALL(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                      &w->api));
  const void *extension = NULL;
  CALL(w->api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension));
  w->gpu = (const amdf_gpu_api_t *)extension;
  amdf_instance_create_info_t instance;
  INIT(instance, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instance.native_lifetime = lifetime;
  CALL(w->api->instance_create(&instance, &w->instance));
  REQUIRE(select_gfx1201(w) == 0);
  REQUIRE(select_scope_and_family(w) == 0);
  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  amdf_gpu_device_info_t device_info;
  INIT(device_info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO);
  CALL(w->gpu->device_query_info(w->device, &device_info));
  REQUIRE((device_info.features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) !=
          0);
  REQUIRE(register_buffer(w, &w->source, AMDF_MEMORY_ACCESS_READ, BUFFER_BYTES,
                          37) == 0);
  REQUIRE(register_buffer(w, &w->target,
                          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                          BUFFER_BYTES, 123) == 0);
  REQUIRE(register_buffer(w, &w->completion,
                          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                          4096, 12) == 0);
  REQUIRE(create_queue(w) == 0);
  return 0;
}

static void encode(uint32_t words[SLOT_WORDS], uint64_t source, uint64_t target,
                   uint32_t length, uint64_t fence, uint32_t token) {
  const uint32_t commands[SLOT_WORDS] = {0x111,
                                         0,
                                         0xc3c00000,
                                         0,
                                         0,
                                         1,
                                         length - 1,
                                         0,
                                         (uint32_t)source,
                                         (uint32_t)(source >> 32),
                                         (uint32_t)target,
                                         (uint32_t)(target >> 32),
                                         0x111,
                                         0,
                                         0x80400000,
                                         0,
                                         0,
                                         0x00130005,
                                         (uint32_t)fence,
                                         (uint32_t)(fence >> 32),
                                         token};
  memcpy(words, commands, sizeof(commands));
}

static int run(struct workload *w) {
  static const uint32_t lengths[] = {1,   4,    63,   64,   65,
                                     257, 4095, 4096, 4097, 65535};
  uint64_t *read_index = (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell = (uint64_t *)(uintptr_t)w->transport.doorbell_address;
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
    __atomic_thread_fence(__ATOMIC_SEQ_CST);

    uint32_t commands[SLOT_WORDS];
    encode(commands, w->source.address, w->target.address, length,
           w->completion.address, iteration + 1);
    unsigned char *slot =
        (unsigned char *)(uintptr_t)w->transport.ring_address +
        (producer & (w->transport.ring_byte_length - 1));
    memcpy(slot, commands, sizeof(commands));
    producer += sizeof(commands);
    w->in_flight = 1;
    __atomic_store_n(write_index, producer, __ATOMIC_RELEASE);
    __atomic_store_n(doorbell, producer, __ATOMIC_RELEASE);

    const uint64_t begin = now_ns();
    unsigned polls = 0;
    while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != iteration + 1 ||
           __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < producer) {
      if ((++polls & 255u) == 0 && now_ns() - begin >= timeout_ns) {
        fprintf(stderr,
                "SDMA REGISTER timeout: iteration=%u token=%u "
                "consumed=%" PRIu64 " producer=%" PRIu64 "\n",
                iteration, __atomic_load_n(completion, __ATOMIC_ACQUIRE),
                __atomic_load_n(read_index, __ATOMIC_ACQUIRE), producer);
        return 1;
      }
#if defined(__x86_64__)
      __asm__ __volatile__("pause");
#endif
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    w->in_flight = 0;
    REQUIRE(memcmp(source, target, length) == 0);
    for (uint32_t index = length; index < BUFFER_BYTES; ++index)
      REQUIRE(target[index] == 0xa5);
  }

  CALL(w->api->user_queue_wait_consumed(w->queue, producer, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
  REQUIRE(status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == producer);
  REQUIRE(status.consumed_index == producer);
  printf("Verified %u SDMA registered-host copies and %" PRIu64 " ring wraps\n",
         ITERATIONS,
         (uint64_t)ITERATIONS * sizeof(uint32_t[SLOT_WORDS]) /
             w->transport.ring_byte_length);
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
  if (buffer->allocation != NULL) {
    unsigned char *bytes = (unsigned char *)buffer->allocation;
    bytes[0] = 0x5a;
    REQUIRE(bytes[0] == 0x5a);
    free(buffer->allocation);
    buffer->allocation = NULL;
  }
  return 0;
}

static int cleanup(struct workload *w) {
  if (w->in_flight) {
    fputs("Retaining registered pages after incomplete GPU work.\n", stderr);
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
  amdf_native_lifetime_t lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  if (argc == 2 && strcmp(argv[1], "instance") == 0) {
    lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
  } else if (argc != 1) {
    fprintf(stderr, "usage: %s [instance]\n", argv[0]);
    return 2;
  }
  struct workload workload;
  memset(&workload, 0, sizeof(workload));
  int result = prepare(&workload, lifetime);
  if (result == 0)
    result = run(&workload);
  const int cleanup_result = cleanup(&workload);
  return result || cleanup_result;
}
