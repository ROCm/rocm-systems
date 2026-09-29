/* GFX1201 AMDF qualification for same-device GPU-produced SDMA and AQL queues.
 * A host-produced AQL kernel publishes packets into a device-mapped target
 * queue. Completion, consumption, ring reuse, and cleanup are checked. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INIT(record, tag)                                                      \
  do {                                                                         \
    memset(&(record), 0, sizeof(record));                                      \
    (record).type = (tag);                                                     \
    (record).structure_size = (uint32_t)sizeof(record);                        \
  } while (0)
#define REQUIRE(expression)                                                    \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "Device producer: line %d: %s\n", __LINE__,              \
              #expression);                                                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define CALL(expression)                                                       \
  do {                                                                         \
    amdf_status_t call_status = (expression);                                  \
    if (call_status != AMDF_STATUS_OK) {                                       \
      fprintf(stderr,                                                          \
              "Device producer: line %d: %s: status=0x%016" PRIx64 "\n",       \
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
   * cleanup retains everything the device might still touch until process exit.
   */
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

static int create_buffer(struct workload *w, struct buffer *b,
                         amdf_memory_access_t permissions, uint64_t bytes) {
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
  for (ordinal = 0; ordinal < scope_info.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
      continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP))
      break;
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
  CALL(w->api->memory_query_address(b->memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                    &b->address));
  return 0;
}

static int check_pair(struct workload *w, struct buffer *b,
                      int device_produces) {
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
                                      device_produces ? &host : &device,
                                      &pair));
  REQUIRE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE);
  const amdf_cache_transition_t *host_transition =
      device_produces ? &pair.acquire : &pair.release;
  const amdf_cache_transition_t *gpu_transition =
      device_produces ? &pair.release : &pair.acquire;
  REQUIRE(host_transition->kind == AMDF_CACHE_TRANSITION_KIND_NONE);
  REQUIRE(gpu_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(gpu_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  REQUIRE(gpu_transition->operation ==
          (device_produces ? AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
                           : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
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
  CALL(w->api->instance_enumerate_memory_scopes(w->instance, 1, &w->scope,
                                                &count));
  REQUIRE(count == 1);
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
  for (uint32_t i = 0; i < count; ++i) {
    if (endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU)
      continue;
    status = w->api->endpoint_open(w->instance, &endpoints[i].id, &w->endpoint);
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
  REQUIRE(create_buffer(w, &w->source, AMDF_MEMORY_ACCESS_READ, BUFFER_BYTES) ==
          0);
  REQUIRE(create_buffer(w, &w->target,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        BUFFER_BYTES) == 0);
  /* A separate allocation prevents CPU polling from sharing a cache line with
   * the DMA payload. The fence is a completion protocol, not a host atomic API.
   */
  REQUIRE(create_buffer(w, &w->completion,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        4096) == 0);
  REQUIRE(check_pair(w, &w->source, 0) == 0);
  REQUIRE(check_pair(w, &w->target, 0) == 0);
  REQUIRE(check_pair(w, &w->target, 1) == 0);
  REQUIRE(check_pair(w, &w->completion, 0) == 0);
  REQUIRE(check_pair(w, &w->completion, 1) == 0);
  return 0;
}

static void encode(uint32_t words[SLOT_WORDS], uint64_t source, uint64_t target,
                   uint32_t length, uint64_t fence, uint32_t token) {
  /* ROCr's GFX12 BlitSdmaV5: USER_GCR acquire, COPY_LINEAR (CPV=0),
   * USER_GCR release, then an uncached system-memory fence. Trailing zero NOPs
   * make each slot 128 bytes, so no packet crosses the power-of-two ring end.
   */
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

static int release_buffer(struct workload *w, struct buffer *b) {
  if (b->mapping) {
    CALL(w->api->host_mapping_destroy(b->mapping));
    b->mapping = NULL;
  }
  if (b->memory) {
    CALL(w->api->memory_destroy(b->memory));
    b->memory = NULL;
  }
  return 0;
}
static int cleanup(struct workload *w) {
  if (w->in_flight) {
    /* A timeout says nothing about outstanding DMA references. Process teardown
     * owns recovery here; freeing the buffers or unloading the provider would
     * make the failure path unsafe. No device reset is attempted by this
     * example. */
    fputs("Retaining GPU resources until process teardown after incomplete "
          "work.\n",
          stderr);
    return 1;
  }
  if (w->mapping) {
    CALL(w->api->user_queue_mapping_destroy(w->mapping));
    w->mapping = NULL;
  }
  if (w->queue) {
    CALL(w->api->user_queue_destroy(w->queue));
    w->queue = NULL;
  }
  REQUIRE(release_buffer(w, &w->completion) == 0);
  REQUIRE(release_buffer(w, &w->target) == 0);
  REQUIRE(release_buffer(w, &w->source) == 0);
  if (w->device) {
    CALL(w->api->device_destroy(w->device));
    w->device = NULL;
  }
  if (w->endpoint) {
    CALL(w->api->endpoint_close(w->endpoint));
    w->endpoint = NULL;
  }
  if (w->instance) {
    CALL(w->api->instance_destroy(w->instance));
    w->instance = NULL;
  }
  return 0;
}

#include "device-producer-publisher-gfx1201.inc"
#include "device-producer-target-gfx1201.inc"

enum target_kind { TARGET_SDMA, TARGET_AQL };
struct aql_dispatch_packet {
  uint32_t header_setup;
  uint16_t workgroup_size_x, workgroup_size_y, workgroup_size_z, reserved0;
  uint32_t grid_size_x, grid_size_y, grid_size_z;
  uint32_t private_segment_size, group_segment_size;
  uint64_t kernel_object, kernarg_address, reserved2, completion_signal;
};
_Static_assert(sizeof(struct aql_dispatch_packet) == 64, "AQL packet size");
struct publisher_arguments {
  uint64_t commands, ring, write_index, doorbell, completion;
  uint32_t words, header_last;
  uint64_t next_index, doorbell_value;
};
_Static_assert(sizeof(struct publisher_arguments) == 64,
               "publisher kernarg size");
struct extra {
  amdf_user_queue_mapping_t *target_gpu_mapping;
  amdf_user_queue_mapping_info_t target_gpu;
  amdf_user_queue_t *publisher_queue;
  amdf_user_queue_mapping_t *publisher_mapping;
  amdf_user_queue_mapping_info_t publisher;
  struct buffer commands, code, arguments, published;
  struct buffer target_code, target_arguments, target_completion;
  int in_flight;
};

static int select_aql_family(struct workload *w, uint32_t *ordinal) {
  amdf_endpoint_info_t endpoint;
  INIT(endpoint, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &endpoint));
  for (uint32_t candidate = 0; candidate < endpoint.queue_family_count;
       ++candidate) {
    amdf_queue_family_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(w->endpoint, candidate,
                                                  &info));
    if (info.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
        info.format_version == 1 && info.format_features == 0 &&
        (info.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) &&
        (info.user_queue_capabilities &
         AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) &&
        (info.user_queue_capabilities &
         AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER)) {
      *ordinal = candidate;
      return 0;
    }
  }
  fputs("No qualified AQL producer family\n", stderr);
  return 1;
}

static int install_code(struct buffer *buffer, const unsigned char *descriptor,
                        const unsigned char *text, size_t text_length) {
  REQUIRE(256 + text_length <= 4096);
  unsigned char *image = (unsigned char *)buffer->host.pointer;
  memset(image, 0, 4096);
  memcpy(image, descriptor, 64);
  const int64_t entry_offset = 256;
  memcpy(image + 16, &entry_offset, sizeof(entry_offset));
  memcpy(image + 256, text, text_length);
  return 0;
}

static int prepare_extra(struct workload *w, struct extra *e,
                         enum target_kind kind) {
  uint32_t aql_family = UINT32_MAX;
  REQUIRE(select_aql_family(w, &aql_family) == 0);
  amdf_gpu_user_queue_create_info_t target;
  INIT(target, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  target.queue_family_ordinal = kind == TARGET_SDMA ? w->family : aql_family;
  target.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  target.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  target.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                                 AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  target.ring_byte_length = 1024;
  CALL(w->gpu->user_queue_create(w->device, &target, &w->queue));
  amdf_user_queue_info_t target_info;
  INIT(target_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
  CALL(w->api->user_queue_query_info(w->queue, &target_info));
  REQUIRE((target_info.capabilities & target.required_capabilities) ==
          target.required_capabilities);
  CALL(w->api->user_queue_map(w->queue, NULL, &w->mapping));
  INIT(w->transport, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(w->mapping, &w->transport));
  CALL(w->api->user_queue_map(w->queue, w->device, &e->target_gpu_mapping));
  INIT(e->target_gpu, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(e->target_gpu_mapping,
                                             &e->target_gpu));
  REQUIRE(e->target_gpu.ring_address && e->target_gpu.read_index_address &&
          e->target_gpu.write_index_address && e->target_gpu.doorbell_address);
  REQUIRE(e->target_gpu.ring_byte_length == target.ring_byte_length);
  REQUIRE(e->target_gpu.index_bits == 64 && e->target_gpu.doorbell_bits == 64);

  REQUIRE(create_buffer(w, &e->commands, AMDF_MEMORY_ACCESS_READ, 4096) == 0);
  REQUIRE(create_buffer(w, &e->code,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                        4096) == 0);
  REQUIRE(create_buffer(w, &e->arguments, AMDF_MEMORY_ACCESS_READ, 4096) == 0);
  REQUIRE(create_buffer(w, &e->published,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        4096) == 0);
  REQUIRE(install_code(&e->code, producer_kernel_descriptor,
                       producer_kernel_text,
                       sizeof(producer_kernel_text)) == 0);
  if (kind == TARGET_AQL) {
    REQUIRE(create_buffer(w, &e->target_code,
                          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                          4096) == 0);
    REQUIRE(create_buffer(w, &e->target_arguments, AMDF_MEMORY_ACCESS_READ,
                          4096) == 0);
    REQUIRE(create_buffer(w, &e->target_completion,
                          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                          4096) == 0);
    REQUIRE(install_code(&e->target_code, target_kernel_descriptor,
                         target_kernel_text, sizeof(target_kernel_text)) == 0);
    *(uint64_t *)e->target_arguments.host.pointer =
        e->target_completion.address;
  }

  amdf_gpu_user_queue_create_info_t publisher;
  INIT(publisher, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  publisher.queue_family_ordinal = aql_family;
  publisher.priority = AMDF_QUEUE_PRIORITY_NORMAL;
  publisher.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
  publisher.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  publisher.ring_byte_length = 1024;
  CALL(w->gpu->user_queue_create(w->device, &publisher, &e->publisher_queue));
  CALL(w->api->user_queue_map(e->publisher_queue, NULL, &e->publisher_mapping));
  INIT(e->publisher, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(e->publisher_mapping,
                                             &e->publisher));
  REQUIRE(e->publisher.ring_address && e->publisher.read_index_address &&
          e->publisher.write_index_address && e->publisher.doorbell_address);
  return 0;
}

static void make_dispatch(struct aql_dispatch_packet *packet, uint64_t kernel,
                          uint64_t arguments) {
  memset(packet, 0, sizeof(*packet));
  const uint16_t header = (uint16_t)(2u | (2u << 9) | (2u << 11));
  packet->header_setup = (uint32_t)header | (1u << 16);
  packet->workgroup_size_x = 1;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->grid_size_x = 1;
  packet->grid_size_y = 1;
  packet->grid_size_z = 1;
  packet->kernel_object = kernel;
  packet->kernarg_address = arguments;
}

static int run_extra(struct workload *w, struct extra *e,
                     enum target_kind kind) {
  uint32_t *source = (uint32_t *)w->source.host.pointer;
  uint32_t *target = (uint32_t *)w->target.host.pointer;
  uint32_t *published = (uint32_t *)e->published.host.pointer;
  uint32_t *done = kind == TARGET_SDMA
                       ? (uint32_t *)w->completion.host.pointer
                       : (uint32_t *)e->target_completion.host.pointer;
  uint64_t *publisher_read =
      (uint64_t *)(uintptr_t)e->publisher.read_index_address;
  uint64_t *target_read =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *publisher_write =
      (uint64_t *)(uintptr_t)e->publisher.write_index_address;
  uint64_t *publisher_doorbell =
      (uint64_t *)(uintptr_t)e->publisher.doorbell_address;
  const uint64_t target_step = kind == TARGET_SDMA ? SLOT_WORDS * 4 : 1;
  REQUIRE(__atomic_load_n(publisher_read, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(__atomic_load_n(target_read, __ATOMIC_ACQUIRE) == 0);
  for (uint32_t iteration = 0; iteration < ITERATIONS; ++iteration) {
    __atomic_store_n(published, 0, __ATOMIC_RELAXED);
    __atomic_store_n(done, 0, __ATOMIC_RELAXED);
    uint32_t target_token = kind == TARGET_SDMA ? iteration + 1 : 2;
    uint32_t words = 0;
    if (kind == TARGET_SDMA) {
      const uint32_t length = 4096 + (iteration * 73) % 512;
      for (uint32_t index = 0; index < BUFFER_BYTES / 4; ++index) {
        source[index] = index * 17 + iteration * 31;
        target[index] = UINT32_C(0xa5a5a5a5);
      }
      uint32_t commands[SLOT_WORDS];
      encode(commands, w->source.address, w->target.address, length,
             w->completion.address, target_token);
      memcpy(e->commands.host.pointer, commands, sizeof(commands));
      words = SLOT_WORDS;
    } else {
      struct aql_dispatch_packet target_packet;
      make_dispatch(&target_packet, e->target_code.address,
                    e->target_arguments.address);
      memcpy(e->commands.host.pointer, &target_packet, sizeof(target_packet));
      words = sizeof(target_packet) / 4;
    }
    const uint64_t target_index = iteration * target_step;
    const uint64_t next_target_index = target_index + target_step;
    struct publisher_arguments *args =
        (struct publisher_arguments *)e->arguments.host.pointer;
    *args = (struct publisher_arguments){
        .commands = e->commands.address,
        .ring = e->target_gpu.ring_address +
                ((target_index * (kind == TARGET_AQL ? 64 : 1)) &
                 (e->target_gpu.ring_byte_length - 1)),
        .write_index = e->target_gpu.write_index_address,
        .doorbell = e->target_gpu.doorbell_address,
        .completion = e->published.address,
        .words = words,
        .header_last = kind == TARGET_AQL,
        .next_index = next_target_index,
        .doorbell_value = target_index,
    };
    if (kind == TARGET_SDMA)
      args->doorbell_value = next_target_index;

    struct aql_dispatch_packet publisher_packet;
    make_dispatch(&publisher_packet, e->code.address, e->arguments.address);
    uint32_t *slot =
        (uint32_t *)(uintptr_t)(e->publisher.ring_address +
                                ((iteration * 64) &
                                 (e->publisher.ring_byte_length - 1)));
    memcpy((unsigned char *)slot + 4,
           (const unsigned char *)&publisher_packet + 4, 60);
    __atomic_store_n(slot, publisher_packet.header_setup, __ATOMIC_RELEASE);
    e->in_flight = 1;
    w->in_flight = 1;
    __atomic_store_n(publisher_write, iteration + 1, __ATOMIC_RELEASE);
#if defined(__x86_64__)
    __asm__ __volatile__("sfence" ::: "memory");
#endif
    __atomic_store_n(publisher_doorbell, iteration, __ATOMIC_RELAXED);

    const uint64_t begin = now_ns();
    while (__atomic_load_n(published, __ATOMIC_ACQUIRE) != 1 ||
           __atomic_load_n(done, __ATOMIC_ACQUIRE) != target_token ||
           __atomic_load_n(publisher_read, __ATOMIC_ACQUIRE) < iteration + 1 ||
           __atomic_load_n(target_read, __ATOMIC_ACQUIRE) < next_target_index) {
      if (now_ns() - begin >= timeout_ns) {
        fprintf(stderr,
                "Device-producer timeout: mode=%s iteration=%u publisher=%u "
                "target=%u "
                "publisher_read=%" PRIu64 " target_read=%" PRIu64 "\n",
                kind == TARGET_SDMA ? "sdma" : "aql", iteration,
                __atomic_load_n(published, __ATOMIC_ACQUIRE),
                __atomic_load_n(done, __ATOMIC_ACQUIRE),
                __atomic_load_n(publisher_read, __ATOMIC_ACQUIRE),
                __atomic_load_n(target_read, __ATOMIC_ACQUIRE));
        return 1;
      }
#if defined(__x86_64__)
      __asm__ __volatile__("pause");
#endif
    }
    e->in_flight = 0;
    w->in_flight = 0;
    if (kind == TARGET_SDMA) {
      const uint32_t length = 4096 + (iteration * 73) % 512;
      REQUIRE(memcmp(source, target, length) == 0);
      for (uint32_t index = length; index < BUFFER_BYTES; ++index)
        REQUIRE(((unsigned char *)target)[index] == 0xa5);
    }
  }
  CALL(w->api->user_queue_wait_consumed(w->queue, ITERATIONS * target_step, 0,
                                        0));
  CALL(w->api->user_queue_wait_consumed(e->publisher_queue, ITERATIONS, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE &&
          status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == ITERATIONS * target_step &&
          status.consumed_index == ITERATIONS * target_step);
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(e->publisher_queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE &&
          status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == ITERATIONS &&
          status.consumed_index == ITERATIONS);
  printf("Verified %u GPU-produced %s packets, publisher/target completion, "
         "and %" PRIu64 " target ring wraps\n",
         ITERATIONS, kind == TARGET_SDMA ? "SDMA" : "AQL",
         ITERATIONS * target_step * (kind == TARGET_AQL ? 64 : 1) /
             w->transport.ring_byte_length);
  return 0;
}

static int cleanup_extra(struct workload *w, struct extra *e) {
  if (e->publisher_mapping)
    CALL(w->api->user_queue_mapping_destroy(e->publisher_mapping));
  if (e->publisher_queue)
    CALL(w->api->user_queue_destroy(e->publisher_queue));
  if (e->target_gpu_mapping)
    CALL(w->api->user_queue_mapping_destroy(e->target_gpu_mapping));
  REQUIRE(release_buffer(w, &e->target_completion) == 0);
  REQUIRE(release_buffer(w, &e->target_arguments) == 0);
  REQUIRE(release_buffer(w, &e->target_code) == 0);
  REQUIRE(release_buffer(w, &e->published) == 0);
  REQUIRE(release_buffer(w, &e->arguments) == 0);
  REQUIRE(release_buffer(w, &e->code) == 0);
  REQUIRE(release_buffer(w, &e->commands) == 0);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[1], "sdma") && strcmp(argv[1], "aql")) ||
      (strcmp(argv[2], "process") && strcmp(argv[2], "instance"))) {
    fprintf(stderr, "Usage: %s sdma|aql process|instance\n", argv[0]);
    return 2;
  }
  enum target_kind kind =
      strcmp(argv[1], "sdma") == 0 ? TARGET_SDMA : TARGET_AQL;
  amdf_native_lifetime_t lifetime = strcmp(argv[2], "process") == 0
                                        ? AMDF_NATIVE_LIFETIME_PROCESS
                                        : AMDF_NATIVE_LIFETIME_INSTANCE;
  struct workload w = {0};
  struct extra e = {0};
  int result = prepare(&w, lifetime);
  if (!result)
    result = prepare_extra(&w, &e, kind);
  if (!result)
    result = run_extra(&w, &e, kind);
  if (w.in_flight || e.in_flight) {
    fputs("Retaining GPU resources until process teardown after incomplete "
          "work.\n",
          stderr);
    return 1;
  }
  if (cleanup_extra(&w, &e))
    return 1;
  int cleanup_result = cleanup(&w);
  return result || cleanup_result;
}
