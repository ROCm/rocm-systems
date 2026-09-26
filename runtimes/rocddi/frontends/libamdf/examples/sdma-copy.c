/* A complete AMDF consumer for the qualified GFX1201/SDMA7 SYSTEM-memory path.
 * Packet construction and completion policy belong to this caller. The provider
 * supplies resources, queue transport, and the directional cache requirements.
 * See README.md for packet provenance and the exact qualification boundary. */
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
  (record).type = (tag); (record).structure_size = (uint32_t)sizeof(record); \
} while (0)
#define REQUIRE(expression) do { \
  if (!(expression)) { \
    fprintf(stderr, "SDMA copy: line %d: %s\n", __LINE__, #expression); \
    return 1; \
  } \
} while (0)
#define CALL(expression) do { \
  amdf_status_t call_status = (expression); \
  if (call_status != AMDF_STATUS_OK) { \
    fprintf(stderr, "SDMA copy: line %d: %s: status=0x%016" PRIx64 "\n", \
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
  amdf_user_queue_t *queue;
  amdf_user_queue_mapping_t *mapping;
  amdf_user_queue_mapping_info_t transport;
  amdf_memory_scope_t *scope;
  struct buffer source, target, completion;
  uint32_t family;
  amdf_queue_format_features_t format_features;
  /* Publication transfers access to the GPU. On timeout this stays set, so
   * cleanup retains everything the device might still touch until process exit. */
  int in_flight;
};

static uint64_t now_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    perror("clock_gettime");
    exit(1);
  }
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static int create_buffer(struct workload *w, struct buffer *b,
                         amdf_memory_access_t permissions, uint64_t bytes) {
  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access = permissions;
  access.requirements.flags = AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  amdf_memory_scope_info_t scope_info;
  INIT(scope_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->scope, &scope_info));
  uint32_t ordinal;
  for (ordinal = 0; ordinal < scope_info.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) break;
  }
  REQUIRE(ordinal < scope_info.memory_profile_count);
  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = ordinal;
  create.byte_length = bytes;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.access_count = 1;
  create.accesses = &access;
  CALL(w->api->memory_create(w->scope, &create, &b->memory));
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(b->memory, &map, &b->mapping));
  INIT(b->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(b->mapping, &b->host));
  REQUIRE(b->host.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
  CALL(w->api->memory_query_address(b->memory, 0, AMDF_MEMORY_ADDRESS_GPU, &b->address));
  return 0;
}

static int check_pair(struct workload *w, struct buffer *b, int device_produces) {
  amdf_memory_site_t host, device;
  INIT(host, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  INIT(device, AMDF_STRUCTURE_TYPE_MEMORY_SITE);
  host.kind = AMDF_MEMORY_SITE_KIND_HOST;
  host.value.host_mapping = b->mapping;
  device.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  device.value.device.memory = b->memory;
  device.value.device.queue_family_ordinal = w->family;
  amdf_memory_pair_info_t pair;
  INIT(pair, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO);
  CALL(w->api->memory_query_pair_info(device_produces ? &device : &host,
                                     device_produces ? &host : &device, &pair));
  REQUIRE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE);
  const amdf_cache_transition_t *host_transition = device_produces ? &pair.acquire : &pair.release;
  const amdf_cache_transition_t *gpu_transition = device_produces ? &pair.release : &pair.acquire;
  REQUIRE(host_transition->kind == AMDF_CACHE_TRANSITION_KIND_NONE);
  REQUIRE(gpu_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(gpu_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  REQUIRE(gpu_transition->operation == (device_produces ? AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
                                                      : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
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
  instance.native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  CALL(w->api->instance_create(&instance, &w->instance));
  uint32_t count = 0;
  CALL(w->api->instance_enumerate_memory_scopes(w->instance, 1, &w->scope, &count));
  REQUIRE(count == 1);
  CALL(w->api->endpoint_enumerate(w->instance, 0, NULL, &count));
  amdf_endpoint_summary_t *endpoints = (amdf_endpoint_summary_t *)calloc(count, sizeof(*endpoints));
  REQUIRE(endpoints != NULL);
  amdf_status_t status = w->api->endpoint_enumerate(w->instance, count, endpoints, &count);
  if (status != AMDF_STATUS_OK) { free(endpoints); REQUIRE(status == AMDF_STATUS_OK); }
  for (uint32_t i = 0; i < count; ++i) {
    if (endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU) continue;
    status = w->api->endpoint_open(w->instance, &endpoints[i].id, &w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    amdf_gpu_endpoint_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
    status = w->gpu->endpoint_query_info(w->endpoint, &info);
    if (status != AMDF_STATUS_OK) break;
    if (info.gfx_ip.major == 12 && info.gfx_ip.minor == 0 && info.gfx_ip.stepping == 1) break;
    status = w->api->endpoint_close(w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    w->endpoint = NULL;
  }
  free(endpoints);
  REQUIRE(status == AMDF_STATUS_OK && w->endpoint != NULL);
  amdf_endpoint_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &info));
  for (w->family = 0; w->family < info.queue_family_count; ++w->family) {
    amdf_queue_family_info_t family;
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(w->endpoint, w->family, &family));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
        family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 &&
        family.format_features == required_format_features &&
        (family.roles & AMDF_QUEUE_ROLE_CACHE_CONTROL) &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) &&
        (family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL)) {
      w->format_features = family.format_features;
      break;
    }
  }
  REQUIRE(w->family < info.queue_family_count);
  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  REQUIRE(create_buffer(w, &w->source, AMDF_MEMORY_ACCESS_READ, BUFFER_BYTES) == 0);
  REQUIRE(create_buffer(w, &w->target, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, BUFFER_BYTES) == 0);
  /* A separate allocation prevents CPU polling from sharing a cache line with
   * the DMA payload. The fence is a completion protocol, not a host atomic API. */
  REQUIRE(create_buffer(w, &w->completion, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096) == 0);
  REQUIRE(check_pair(w, &w->source, 0) == 0);
  REQUIRE(check_pair(w, &w->target, 0) == 0);
  REQUIRE(check_pair(w, &w->target, 1) == 0);
  REQUIRE(check_pair(w, &w->completion, 0) == 0);
  REQUIRE(check_pair(w, &w->completion, 1) == 0);
  amdf_gpu_user_queue_create_info_t queue;
  INIT(queue, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  queue.queue_family_ordinal = w->family;
  queue.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  queue.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  queue.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  queue.ring_byte_length = 1024;
  CALL(w->gpu->user_queue_create(w->device, &queue, &w->queue));
  amdf_user_queue_info_t queue_info;
  INIT(queue_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
  CALL(w->api->user_queue_query_info(w->queue, &queue_info));
  REQUIRE(queue_info.format_features == w->format_features);
  CALL(w->api->user_queue_map(w->queue, NULL, &w->mapping));
  INIT(w->transport, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(w->mapping, &w->transport));
  REQUIRE(w->transport.format_features == w->format_features);
  REQUIRE(w->transport.index_bits == 64 && w->transport.doorbell_bits == 64);
  REQUIRE(w->transport.ring_byte_length == queue.ring_byte_length);
  REQUIRE(w->transport.ring_address && w->transport.read_index_address &&
          w->transport.write_index_address && w->transport.doorbell_address);
  REQUIRE((w->transport.read_index_address | w->transport.write_index_address |
           w->transport.doorbell_address) % 8 == 0);
  REQUIRE(w->completion.address % 4 == 0);
  printf("GFX1201 SYSTEM/SDMA family=%u format=%u ring=%" PRIu64 " bytes; host WB/coherent\n",
         w->family, w->transport.format_version, w->transport.ring_byte_length);
  return 0;
}

static void encode(uint32_t words[SLOT_WORDS], uint64_t source, uint64_t target,
                   uint32_t length, uint64_t fence, uint32_t token) {
  /* ROCr's GFX12 BlitSdmaV5: USER_GCR acquire, COPY_LINEAR (CPV=0),
   * USER_GCR release, then an uncached system-memory fence. Trailing zero NOPs
   * make each slot 128 bytes, so no packet crosses the power-of-two ring end. */
  const uint32_t commands[SLOT_WORDS] = {
    0x111, 0, 0xc3c00000, 0, 0,
    1, length - 1, 0, (uint32_t)source, (uint32_t)(source >> 32),
    (uint32_t)target, (uint32_t)(target >> 32),
    0x111, 0, 0x80400000, 0, 0,
    0x00130005, (uint32_t)fence, (uint32_t)(fence >> 32), token
  };
  memcpy(words, commands, sizeof(commands));
}

static int run(struct workload *w) {
  static const uint32_t lengths[] = {1, 4, 63, 64, 65, 257, 4095, 4096, 4097, 65535};
  uint64_t elapsed[ITERATIONS];
  uint64_t producer = 0;
  uint64_t *read_index = (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index = (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell = (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  unsigned char *source = (unsigned char *)w->source.host.pointer;
  unsigned char *target = (unsigned char *)w->target.host.pointer;
  REQUIRE(__atomic_load_n(read_index, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(__atomic_load_n(write_index, __ATOMIC_ACQUIRE) == 0);
  for (unsigned iteration = 0; iteration < ITERATIONS; ++iteration) {
    uint32_t length = lengths[iteration % (sizeof(lengths) / sizeof(lengths[0]))];
    for (uint32_t i = 0; i < BUFFER_BYTES; ++i) source[i] = (unsigned char)(i * 17 + iteration * 31);
    memset(target, 0xa5, BUFFER_BYTES);
    __atomic_store_n(completion, 0, __ATOMIC_RELAXED);
    uint32_t commands[SLOT_WORDS];
    encode(commands, w->source.address, w->target.address, length,
           w->completion.address, iteration + 1);
    unsigned char *slot = (unsigned char *)(uintptr_t)w->transport.ring_address +
                         (producer & (w->transport.ring_byte_length - 1));
    uint64_t begin = now_ns();
    memcpy(slot, commands, sizeof(commands));
    producer += sizeof(commands);
    w->in_flight = 1;
    __atomic_store_n(write_index, producer, __ATOMIC_RELEASE);
    __atomic_store_n(doorbell, producer, __ATOMIC_RELEASE);
    /* Completion grants host access to the payload; consumption grants reuse
     * of ring bytes. Both must be observed, and neither substitutes for the other. */
    unsigned polls = 0;
    while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != iteration + 1 ||
           __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < producer) {
      if ((++polls & 255u) == 0 && now_ns() - begin >= timeout_ns) {
        fprintf(stderr, "Timeout: iteration=%u token=%u consumed=%" PRIu64 " producer=%" PRIu64 "\n",
                iteration, __atomic_load_n(completion, __ATOMIC_ACQUIRE),
                __atomic_load_n(read_index, __ATOMIC_ACQUIRE), producer);
        return 1;
      }
#if defined(__x86_64__)
      __asm__ __volatile__("pause");
#endif
    }
    elapsed[iteration] = now_ns() - begin;
    w->in_flight = 0;
    REQUIRE(memcmp(source, target, length) == 0);
    for (uint32_t i = length; i < BUFFER_BYTES; ++i) REQUIRE(target[i] == 0xa5);
  }
  CALL(w->api->user_queue_wait_consumed(w->queue, producer, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE && status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == producer && status.consumed_index == producer);
  /* Sort only after all GPU work: these are end-to-end observations including
   * host publication and polling, not isolated engine latency or a jitter bound. */
  for (unsigned i = 1; i < ITERATIONS; ++i) {
    uint64_t value = elapsed[i];
    unsigned j = i;
    while (j && elapsed[j - 1] > value) { elapsed[j] = elapsed[j - 1]; --j; }
    elapsed[j] = value;
  }
  printf("Verified %u SDMA copies (1..65535 bytes), %" PRIu64 " ring wraps; completion and consumption agree\n",
         ITERATIONS, producer / w->transport.ring_byte_length);
  printf("Publication-to-completion+consumption: min=%" PRIu64 " ns median=%" PRIu64 " ns max=%" PRIu64 " ns\n",
         elapsed[0], elapsed[ITERATIONS / 2], elapsed[ITERATIONS - 1]);
  return 0;
}

static int release_buffer(struct workload *w, struct buffer *b) {
  if (b->mapping) { CALL(w->api->host_mapping_destroy(b->mapping)); b->mapping = NULL; }
  if (b->memory) { CALL(w->api->memory_destroy(b->memory)); b->memory = NULL; }
  return 0;
}
static int cleanup(struct workload *w) {
  if (w->in_flight) {
    /* A timeout says nothing about outstanding DMA references. Process teardown
     * owns recovery here; freeing the buffers or unloading the provider would
     * make the failure path unsafe. No device reset is attempted by this example. */
    fputs("Retaining GPU resources until process teardown after incomplete work.\n", stderr);
    return 1;
  }
  if (w->mapping) { CALL(w->api->user_queue_mapping_destroy(w->mapping)); w->mapping = NULL; }
  if (w->queue) { CALL(w->api->user_queue_destroy(w->queue)); w->queue = NULL; }
  REQUIRE(release_buffer(w, &w->completion) == 0);
  REQUIRE(release_buffer(w, &w->target) == 0);
  REQUIRE(release_buffer(w, &w->source) == 0);
  if (w->device) { CALL(w->api->device_destroy(w->device)); w->device = NULL; }
  if (w->endpoint) { CALL(w->api->endpoint_close(w->endpoint)); w->endpoint = NULL; }
  if (w->instance) { CALL(w->api->instance_destroy(w->instance)); w->instance = NULL; }
  return 0;
}

int main(void) {
  struct workload workload;
  memset(&workload, 0, sizeof(workload));
  int result = prepare(&workload);
  if (result == 0) result = run(&workload);
  int cleanup_result = cleanup(&workload);
  return result || cleanup_result;
}
