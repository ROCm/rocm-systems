/* Direct AQL execution qualification for the GFX1201 Linux KFD provider.
 * The generated kernel uses no runtime libraries and executes fixed private
 * storage from caller-supplied scratch. The consumer owns packet construction,
 * publication, completion, and resource lifetime. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <inttypes.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <aql-copy-add-gfx1201.inc>

#define INIT(record, tag) do { \
  memset(&(record), 0, sizeof(record)); \
  (record).type = (tag); \
  (record).structure_size = (uint32_t)sizeof(record); \
} while (0)
#define REQUIRE(expression) do { \
  if (!(expression)) { \
    fprintf(stderr, "AQL dispatch: line %d: %s\n", __LINE__, #expression); \
    return 1; \
  } \
} while (0)
#define CALL(expression) do { \
  amdf_status_t call_status = (expression); \
  if (call_status != AMDF_STATUS_OK) { \
    fprintf(stderr, "AQL dispatch: line %d: %s: status=0x%016" PRIx64 "\n", \
            __LINE__, #expression, call_status); \
    return 1; \
  } \
} while (0)

enum {
  BUFFER_WORDS = 1024,
  CODE_BYTES = 4096,
  CODE_OFFSET = 256,
  ITERATIONS = 128,
  PACKET_BYTES = 64,
  PRODUCER_THREADS = 4,
  SCRATCH_ALIGNMENT = 256
};
static const uint64_t timeout_ns = UINT64_C(5000000000);

struct aql_dispatch_packet {
  uint32_t header_setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved2;
  uint64_t completion_signal;
};

struct kernel_arguments {
  uint64_t source;
  uint64_t target;
  uint64_t completion;
  uint64_t gate;
  uint64_t started;
  uint32_t count;
  uint32_t addend;
  uint32_t token;
  uint32_t reserved[3];
};

struct aql_barrier_packet {
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  uint64_t dependent_signals[5];
  uint64_t reserved2;
  uint64_t completion_signal;
};

#if defined(__cplusplus)
static_assert(sizeof(aql_dispatch_packet) == PACKET_BYTES);
static_assert(sizeof(kernel_arguments) == 64);
static_assert(offsetof(kernel_arguments, token) == 48);
static_assert(sizeof(aql_barrier_packet) == PACKET_BYTES);
#else
_Static_assert(sizeof(struct aql_dispatch_packet) == PACKET_BYTES,
               "AQL dispatch packet must be 64 bytes");
_Static_assert(sizeof(struct kernel_arguments) == 64,
               "kernel argument slot must be 64 bytes");
_Static_assert(offsetof(struct kernel_arguments, token) == 48,
               "kernel argument layout changed");
_Static_assert(sizeof(struct aql_barrier_packet) == PACKET_BYTES,
               "AQL barrier packet must be 64 bytes");
#endif

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
  amdf_memory_scope_t *local_scope;
  int local_mode;
  amdf_user_queue_t *queue;
  amdf_user_queue_mapping_t *mapping;
  amdf_user_queue_mapping_info_t transport;
  amdf_gpu_endpoint_info_t gpu_endpoint;
  struct buffer code;
  struct buffer arguments;
  struct buffer source;
  struct buffer target;
  struct buffer completion;
  struct buffer scratch;
  uint32_t family;
  uint32_t private_segment_size;
  uint32_t scratch_private_limit;
  uint32_t scratch_wave_count;
  uint64_t scratch_byte_length;
  uint64_t reservations[ITERATIONS];
  uint32_t multi_ready;
  uint32_t multi_start;
  uint32_t first_reserved;
  uint32_t publish_start;
  uint32_t later_published;
  uint32_t hole_release;
  int in_flight;
};

struct producer_context {
  struct workload *workload;
  uint32_t producer;
  int result;
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

static uint32_t load_u32_le(const unsigned char *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int create_buffer(struct workload *w, struct buffer *buffer,
                         amdf_memory_access_t access_flags, uint64_t bytes,
                         uint64_t minimum_alignment) {
  amdf_memory_device_access_t access;
  memset(&access, 0, sizeof(access));
  access.device = w->device;
  access.requirements.access = access_flags;
  const int local = w->local_mode &&
      (buffer == &w->source || buffer == &w->target);
  amdf_memory_scope_t *scope = local ? w->local_scope : w->scope;
  access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
      (local ? 0 : AMDF_MEMORY_FLAG_HOST_COHERENT);
  access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;

  amdf_memory_scope_info_t scope_info;
  INIT(scope_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(scope, &scope_info));
  uint32_t ordinal;
  for (ordinal = 0; ordinal < scope_info.memory_profile_count; ++ordinal) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    REQUIRE(status == AMDF_STATUS_OK);
    if (profile.memory_class ==
            (local ? AMDF_MEMORY_CLASS_LOCAL : AMDF_MEMORY_CLASS_SYSTEM) &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
        (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0) break;
  }
  REQUIRE(ordinal < scope_info.memory_profile_count);
  if (local) {
    amdf_memory_profile_pair_query_t query;
    INIT(query, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY);
    query.memory_profile_ordinal = ordinal;
    query.access_count = 1;
    query.accesses = &access;
    query.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                           AMDF_MEMORY_FLAG_HOST_VISIBLE;
    for (int device_produces = 0; device_produces != 2; ++device_produces) {
      if (device_produces && !(access_flags & AMDF_MEMORY_ACCESS_WRITE))
        continue;
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
      CALL(w->api->memory_scope_query_pair_info(scope, &query, &pair));
      REQUIRE(pair.release.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      REQUIRE(pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    }
  }

  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = ordinal;
  create.byte_length = bytes;
  create.minimum_alignment = minimum_alignment;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
      (local ? AMDF_MEMORY_FLAG_DEVICE_LOCAL : 0);
  create.access_count = 1;
  create.accesses = &access;
  CALL(w->api->memory_create(scope, &create, &buffer->memory));

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(buffer->memory, &map, &buffer->mapping));
  INIT(buffer->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(buffer->mapping, &buffer->host));
  REQUIRE(buffer->host.cacheability ==
          (local ? AMDF_HOST_CACHEABILITY_WRITE_COMBINED
                 : AMDF_HOST_CACHEABILITY_WRITE_BACK));
  CALL(w->api->memory_query_address(
      buffer->memory, 0, AMDF_MEMORY_ADDRESS_GPU, &buffer->address));
  return 0;
}

static int publish_local(struct workload *w, uint64_t bytes) {
  if (!w->local_mode) return 0;
  CALL(w->api->host_mapping_cache_control(
      w->source.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, bytes));
  CALL(w->api->host_mapping_cache_control(
      w->target.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, bytes));
  return 0;
}

static int observe_local(struct workload *w, uint64_t bytes) {
  if (!w->local_mode) return 0;
  CALL(w->api->host_mapping_cache_control(
      w->target.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, bytes));
  return 0;
}

static int check_pair(struct workload *w, struct buffer *buffer,
                      int device_produces) {
  amdf_memory_site_t host, device;
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
  const int local = w->local_mode &&
      (buffer == &w->source || buffer == &w->target);
  REQUIRE(host_transition->kind ==
          (local ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                 : AMDF_CACHE_TRANSITION_KIND_NONE));
  if (local) {
    REQUIRE(host_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    REQUIRE(host_transition->host_operation ==
            (device_produces ? AMDF_HOST_CACHE_OPERATION_INVALIDATE
                             : AMDF_HOST_CACHE_OPERATION_FLUSH));
    REQUIRE(host_transition->host_fence_after == AMDF_HOST_CACHE_FENCE_X86_MFENCE);
  }
  REQUIRE(gpu_transition->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  REQUIRE(gpu_transition->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  REQUIRE(gpu_transition->operation ==
          (device_produces ? AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM
                           : AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
  return 0;
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
    INIT(w->gpu_endpoint, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
    status = w->gpu->endpoint_query_info(w->endpoint, &w->gpu_endpoint);
    if (status != AMDF_STATUS_OK) break;
    if (w->gpu_endpoint.gfx_ip.major == 12 &&
        w->gpu_endpoint.gfx_ip.minor == 0 &&
        w->gpu_endpoint.gfx_ip.stepping == 1) break;
    status = w->api->endpoint_close(w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    w->endpoint = NULL;
  }
  free(endpoints);
  REQUIRE(status == AMDF_STATUS_OK && w->endpoint != NULL);
  return 0;
}

static int create_queue(struct workload *w, amdf_queue_producer_mode_t mode,
                        amdf_queue_priority_t priority) {
  REQUIRE(w->queue == NULL && w->mapping == NULL);
  amdf_gpu_user_queue_create_info_t queue;
  INIT(queue, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO);
  queue.queue_family_ordinal = w->family;
  queue.priority = priority;
  queue.producer_mode = mode;
  queue.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
  queue.ring_byte_length = 1024;
  queue.scratch.memory = w->scratch.memory;
  queue.scratch.access_ordinal = 0;
  queue.scratch.byte_length = w->scratch_byte_length;
  queue.scratch.maximum_private_segment_byte_length = w->scratch_private_limit;
  queue.scratch.maximum_wave_count = w->scratch_wave_count;
  CALL(w->gpu->user_queue_create(w->device, &queue, &w->queue));

  amdf_status_t scratch_destroy = w->api->memory_destroy(w->scratch.memory);
  if (scratch_destroy == AMDF_STATUS_OK) w->scratch.memory = NULL;
  REQUIRE(scratch_destroy == amdf_make_api_status(AMDF_STATUS_CODE_BUSY));

  amdf_user_queue_info_t queue_info;
  INIT(queue_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO);
  CALL(w->api->user_queue_query_info(w->queue, &queue_info));
  REQUIRE(queue_info.producer_mode == mode);
  REQUIRE(queue_info.priority == priority);
  REQUIRE(queue_info.format_features == 0);
  CALL(w->api->user_queue_map(w->queue, NULL, &w->mapping));
  INIT(w->transport, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(w->mapping, &w->transport));
  REQUIRE(w->transport.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  REQUIRE(w->transport.format_version == 1);
  REQUIRE(w->transport.format_features == 0);
  REQUIRE(w->transport.ring_byte_length == 1024);
  REQUIRE(w->transport.index_bits == 64 && w->transport.doorbell_bits == 64);
  REQUIRE(w->transport.ring_address != 0 &&
          w->transport.read_index_address != 0 &&
          w->transport.write_index_address != 0 &&
          w->transport.doorbell_address != 0);
  return 0;
}

static int destroy_queue(struct workload *w) {
  REQUIRE(!w->in_flight);
  if (w->mapping != NULL) {
    CALL(w->api->user_queue_mapping_destroy(w->mapping));
    w->mapping = NULL;
  }
  if (w->queue != NULL) {
    CALL(w->api->user_queue_destroy(w->queue));
    w->queue = NULL;
  }
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
  REQUIRE(w->gpu_endpoint.compute.wavefront_size == 32);
  REQUIRE(w->gpu_endpoint.compute.compute_unit_count != 0);
  REQUIRE(w->gpu_endpoint.compute.maximum_scratch_wave_count_per_compute_unit !=
          0);
  REQUIRE(w->gpu_endpoint.topology.xcc_count == 1);
  REQUIRE(w->gpu_endpoint.topology.shader_engine_count_per_xcc != 0);

  w->private_segment_size = load_u32_le(aql_kernel_descriptor + 4);
  REQUIRE(w->private_segment_size != 0);
  REQUIRE(w->private_segment_size <= UINT32_MAX - 7);
  w->scratch_private_limit = (w->private_segment_size + 7u) & ~UINT32_C(7);
  const uint64_t shader_engines =
      (uint64_t)w->gpu_endpoint.topology.xcc_count *
      w->gpu_endpoint.topology.shader_engine_count_per_xcc;
  REQUIRE(shader_engines != 0 && shader_engines <= UINT32_MAX);
  const uint64_t maximum_scratch_waves =
      (uint64_t)w->gpu_endpoint.compute.compute_unit_count *
      w->gpu_endpoint.compute.maximum_scratch_wave_count_per_compute_unit;
  REQUIRE(shader_engines <= maximum_scratch_waves);
  w->scratch_wave_count = (uint32_t)shader_engines;
  const uint64_t unaligned_wave_bytes =
      (uint64_t)w->gpu_endpoint.compute.wavefront_size *
      w->scratch_private_limit;
  REQUIRE(unaligned_wave_bytes <= UINT64_MAX - (SCRATCH_ALIGNMENT - 1));
  const uint64_t wave_bytes =
      (unaligned_wave_bytes + SCRATCH_ALIGNMENT - 1) &
      ~(uint64_t)(SCRATCH_ALIGNMENT - 1);
  REQUIRE(wave_bytes != 0 && w->scratch_wave_count <= UINT64_MAX / wave_bytes);
  w->scratch_byte_length = wave_bytes * w->scratch_wave_count;

  amdf_endpoint_info_t endpoint_info;
  INIT(endpoint_info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &endpoint_info));
  for (w->family = 0; w->family < endpoint_info.queue_family_count;
       ++w->family) {
    amdf_queue_family_info_t family;
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(
        w->endpoint, w->family, &family));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
        family.format_version == 1 &&
        family.format_features == 0 &&
        (family.roles & AMDF_QUEUE_ROLE_COMPUTE) != 0 &&
        (family.roles & AMDF_QUEUE_ROLE_CACHE_CONTROL) != 0 &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE) != 0 &&
        (family.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI) != 0 &&
        (family.priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_LOW) !=
            0 &&
        (family.priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL) !=
            0 &&
        (family.priority_capabilities & AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH) !=
            0 &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0 &&
        (family.cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0 &&
        (family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL) != 0)
      break;
  }
  REQUIRE(w->family < endpoint_info.queue_family_count);

  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  if (w->local_mode) {
    count = 0;
    CALL(w->api->device_enumerate_memory_scopes(
        w->device, 1, &w->local_scope, &count));
    REQUIRE(count == 1);
  }
  REQUIRE(create_buffer(
      w, &w->code, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      CODE_BYTES, 0) == 0);
  REQUIRE(create_buffer(w, &w->arguments, AMDF_MEMORY_ACCESS_READ,
                        ITERATIONS * sizeof(struct kernel_arguments), 0) == 0);
  REQUIRE(create_buffer(w, &w->source, AMDF_MEMORY_ACCESS_READ,
                        BUFFER_WORDS * sizeof(uint32_t), 0) == 0);
  REQUIRE(create_buffer(w, &w->target,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        BUFFER_WORDS * sizeof(uint32_t), 0) == 0);
  REQUIRE(create_buffer(w, &w->completion,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        4096, 0) == 0);
  REQUIRE(create_buffer(w, &w->scratch,
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                        w->scratch_byte_length, SCRATCH_ALIGNMENT) == 0);
  REQUIRE((w->scratch.address & (SCRATCH_ALIGNMENT - 1)) == 0);
  CALL(w->api->host_mapping_destroy(w->scratch.mapping));
  w->scratch.mapping = NULL;

  REQUIRE(check_pair(w, &w->code, 0) == 0);
  REQUIRE(check_pair(w, &w->arguments, 0) == 0);
  REQUIRE(check_pair(w, &w->source, 0) == 0);
  REQUIRE(check_pair(w, &w->target, 1) == 0);
  REQUIRE(check_pair(w, &w->completion, 1) == 0);

  REQUIRE(CODE_OFFSET + sizeof(aql_kernel_text) <= CODE_BYTES);
  unsigned char *image = (unsigned char *)w->code.host.pointer;
  memset(image, 0, CODE_BYTES);
  memcpy(image, aql_kernel_descriptor, sizeof(aql_kernel_descriptor));
  const int64_t entry_offset = CODE_OFFSET;
  memcpy(image + 16, &entry_offset, sizeof(entry_offset));
  memcpy(image + CODE_OFFSET, aql_kernel_text, sizeof(aql_kernel_text));

  struct kernel_arguments *arguments =
      (struct kernel_arguments *)w->arguments.host.pointer;
  memset(arguments, 0, sizeof(*arguments));
  arguments->source = w->source.address;
  arguments->target = w->target.address;
  arguments->completion = w->completion.address;

  return create_queue(w, AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                      AMDF_QUEUE_PRIORITY_NORMAL);
}

static void initialize_packet(struct workload *w,
                              struct aql_dispatch_packet *packet,
                              uint64_t kernarg_address) {
  memset(packet, 0, sizeof(*packet));
  const uint16_t header = (uint16_t)(2u | (2u << 9) | (2u << 11));
  const uint16_t setup = 1;
  packet->header_setup = (uint32_t)header | ((uint32_t)setup << 16);
  packet->workgroup_size_x = 1;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->grid_size_x = 1;
  packet->grid_size_y = 1;
  packet->grid_size_z = 1;
  packet->private_segment_size = w->private_segment_size;
  packet->kernel_object = w->code.address;
  packet->kernarg_address = kernarg_address;
}

static void publish_dispatch_packet(struct workload *w, uint64_t packet_id,
                                    uint64_t kernarg_address) {
  const uint64_t slot_count = w->transport.ring_byte_length / PACKET_BYTES;
  struct aql_dispatch_packet packet;
  initialize_packet(w, &packet, kernarg_address);
  struct aql_dispatch_packet *slot =
      (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address +
      (packet_id & (slot_count - 1));
  memcpy((unsigned char *)slot + sizeof(uint32_t),
         (const unsigned char *)&packet + sizeof(uint32_t),
         sizeof(packet) - sizeof(uint32_t));
  __atomic_store_n(&slot->header_setup, packet.header_setup, __ATOMIC_RELEASE);
}

static void publish_barrier_packet(struct workload *w, uint64_t packet_id) {
  const uint64_t slot_count = w->transport.ring_byte_length / PACKET_BYTES;
  struct aql_barrier_packet packet;
  memset(&packet, 0, sizeof(packet));
  packet.header = (uint16_t)(3u | (1u << 8) | (2u << 9) | (2u << 11));
  struct aql_barrier_packet *slot =
      (struct aql_barrier_packet *)(uintptr_t)w->transport.ring_address +
      (packet_id & (slot_count - 1));
  memcpy((unsigned char *)slot + sizeof(uint16_t),
         (const unsigned char *)&packet + sizeof(uint16_t),
         sizeof(packet) - sizeof(uint16_t));
  __atomic_store_n(&slot->header, packet.header, __ATOMIC_RELEASE);
}

static void publish_doorbell(uint64_t *doorbell, uint64_t packet_id) {
  __atomic_thread_fence(__ATOMIC_RELEASE);
#if defined(__x86_64__)
  __asm__ __volatile__("sfence" ::: "memory");
#endif
  __atomic_store_n(doorbell, packet_id, __ATOMIC_RELAXED);
}

static int run_single(struct workload *w, const char *priority_name) {
  uint64_t producer = 0;
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell =
      (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  uint32_t *source = (uint32_t *)w->source.host.pointer;
  uint32_t *target = (uint32_t *)w->target.host.pointer;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  struct kernel_arguments *arguments =
      (struct kernel_arguments *)w->arguments.host.pointer;
  REQUIRE(__atomic_load_n(read_index, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(__atomic_load_n(write_index, __ATOMIC_ACQUIRE) == 0);

  for (uint32_t iteration = 0; iteration < ITERATIONS; ++iteration) {
    const uint32_t count = 1 + (iteration * 73) % BUFFER_WORDS;
    const uint32_t addend = UINT32_C(0x1020304) + iteration;
    const uint32_t token = iteration + 1;
    for (uint32_t index = 0; index < BUFFER_WORDS; ++index) {
      source[index] = index * 17 + iteration * 31;
      target[index] = UINT32_C(0xa5a5a5a5);
    }
    REQUIRE(publish_local(w, BUFFER_WORDS * sizeof(uint32_t)) == 0);
    __atomic_store_n(completion, 0, __ATOMIC_RELAXED);
    arguments->count = count;
    arguments->addend = addend;
    arguments->token = token;

    struct aql_dispatch_packet packet;
    initialize_packet(w, &packet, w->arguments.address);

    const uint64_t slot_count =
        w->transport.ring_byte_length / PACKET_BYTES;
    struct aql_dispatch_packet *slot =
        (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address +
        (producer & (slot_count - 1));
    memcpy((unsigned char *)slot + sizeof(uint32_t),
           (const unsigned char *)&packet + sizeof(uint32_t),
           sizeof(packet) - sizeof(uint32_t));
    __atomic_store_n(&slot->header_setup, packet.header_setup, __ATOMIC_RELEASE);

    const uint64_t submitted = producer;
    ++producer;
    w->in_flight = 1;
    __atomic_store_n(write_index, producer, __ATOMIC_RELEASE);
    publish_doorbell(doorbell, submitted);

    const uint64_t begin = now_ns();
    unsigned polls = 0;
    while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != token ||
           __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < producer) {
      if ((++polls & 255u) == 0 && now_ns() - begin >= timeout_ns) {
        fprintf(stderr,
                "AQL timeout: iteration=%u token=%u consumed=%" PRIu64
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
    REQUIRE(observe_local(w, BUFFER_WORDS * sizeof(uint32_t)) == 0);
    for (uint32_t index = 0; index < count; ++index)
      REQUIRE(target[index] == source[index] + addend);
    for (uint32_t index = count; index < BUFFER_WORDS; ++index)
      REQUIRE(target[index] == UINT32_C(0xa5a5a5a5));
  }

  CALL(w->api->user_queue_wait_consumed(w->queue, producer, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
  REQUIRE(status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == producer);
  REQUIRE(status.consumed_index == producer);
  printf("Verified %u %s-priority scratch-backed GFX1201 AQL dispatches, "
         "%" PRIu64 " ring wraps, %u waves, and %" PRIu64
         " scratch bytes\n",
         ITERATIONS, priority_name,
         producer / (w->transport.ring_byte_length / PACKET_BYTES),
         w->scratch_wave_count, w->scratch_byte_length);
  return 0;
}

static void relax_cpu(void) {
#if defined(__x86_64__)
  __asm__ __volatile__("pause");
#endif
}

static int wait_for_at_least(uint32_t *value, uint32_t target,
                             const char *description) {
  const uint64_t begin = now_ns();
  while (__atomic_load_n(value, __ATOMIC_ACQUIRE) < target) {
    if (now_ns() - begin >= timeout_ns) {
      fprintf(stderr, "AQL multi-producer timeout waiting for %s\n",
              description);
      return 1;
    }
    relax_cpu();
  }
  return 0;
}

static int run_ordering_pair(struct workload *w, uint64_t producer,
                             int use_barrier) {
  const uint32_t argument_base = use_barrier ? 2 : 0;
  const uint32_t gate_index = use_barrier ? 6 : 4;
  const uint32_t started_index = gate_index + 1;
  const uint32_t first_token = UINT32_C(0x50000000) + argument_base;
  const uint32_t second_token = first_token + 1;
  uint32_t *source = (uint32_t *)w->source.host.pointer;
  uint32_t *target = (uint32_t *)w->target.host.pointer;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  struct kernel_arguments *arguments =
      (struct kernel_arguments *)w->arguments.host.pointer;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;

  for (uint32_t offset = 0; offset < 2; ++offset) {
    const uint32_t index = argument_base + offset;
    memset(&arguments[index], 0, sizeof(arguments[index]));
    source[index] = index * 31 + 7;
    target[index] = UINT32_C(0xa5a5a5a5);
    __atomic_store_n(&completion[index], 0, __ATOMIC_RELAXED);
    arguments[index].source = w->source.address + index * sizeof(uint32_t);
    arguments[index].target = w->target.address + index * sizeof(uint32_t);
    arguments[index].completion =
        w->completion.address + index * sizeof(uint32_t);
    arguments[index].count = 1;
    arguments[index].addend = UINT32_C(0x3040506) + index;
    arguments[index].token = first_token + offset;
  }
  __atomic_store_n(&completion[gate_index], 0, __ATOMIC_RELAXED);
  __atomic_store_n(&completion[started_index], 0, __ATOMIC_RELAXED);
  arguments[argument_base].gate =
      w->completion.address + gate_index * sizeof(uint32_t);
  arguments[argument_base].started =
      w->completion.address + started_index * sizeof(uint32_t);
  REQUIRE(publish_local(w, (argument_base + 2) * sizeof(uint32_t)) == 0);

  publish_dispatch_packet(
      w, producer,
      w->arguments.address + argument_base * sizeof(struct kernel_arguments));
  uint64_t second_packet = producer + 1;
  if (use_barrier) {
    publish_barrier_packet(w, second_packet);
    ++second_packet;
  }
  publish_dispatch_packet(
      w, second_packet,
      w->arguments.address +
          (argument_base + 1) * sizeof(struct kernel_arguments));

  const uint64_t published = second_packet + 1;
  w->in_flight = 1;
  __atomic_store_n(write_index, published, __ATOMIC_RELEASE);
  publish_doorbell((uint64_t *)(uintptr_t)w->transport.doorbell_address,
                   second_packet);

  int result = wait_for_at_least(&completion[started_index], first_token,
                                 "gated dispatch start");
  if (result == 0 && use_barrier) {
    const uint64_t observation_begin = now_ns();
    while (now_ns() - observation_begin < UINT64_C(10000000)) {
      if (__atomic_load_n(&completion[argument_base + 1], __ATOMIC_ACQUIRE) !=
          0) {
        fputs("AQL barrier allowed the following dispatch to complete early\n",
              stderr);
        result = 1;
        break;
      }
      relax_cpu();
    }
  } else if (result == 0 &&
             wait_for_at_least(&completion[argument_base + 1], second_token,
                               "unbarriered following dispatch") != 0) {
    result = 1;
  }
  if (__atomic_load_n(&completion[argument_base], __ATOMIC_ACQUIRE) != 0) {
    fputs("AQL gated dispatch completed before host release\n", stderr);
    result = 1;
  }

  __atomic_store_n(&completion[gate_index], 1, __ATOMIC_RELEASE);
  int first_finished =
      wait_for_at_least(&completion[argument_base], first_token,
                        "released gated dispatch completion") == 0;
  int second_finished =
      wait_for_at_least(&completion[argument_base + 1], second_token,
                        "following dispatch completion") == 0;
  amdf_status_t consumed = w->api->user_queue_wait_consumed(
      w->queue, published, timeout_ns, UINT64_C(1000000));
  if (!first_finished || !second_finished || consumed != AMDF_STATUS_OK) {
    fprintf(stderr, "AQL ordering retirement failed: status=0x%016" PRIx64
                    "\n",
            consumed);
    return 1;
  }
  w->in_flight = 0;
  REQUIRE(observe_local(w, (argument_base + 2) * sizeof(uint32_t)) == 0);

  for (uint32_t offset = 0; offset < 2; ++offset) {
    const uint32_t index = argument_base + offset;
    if (target[index] != source[index] + arguments[index].addend) {
      fprintf(stderr, "AQL ordering result mismatch at %u\n", index);
      result = 1;
    }
  }
  return result;
}

static int run_barrier_ordering(struct workload *w) {
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  REQUIRE(__atomic_load_n(read_index, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(__atomic_load_n(write_index, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(run_ordering_pair(w, 0, 0) == 0);
  REQUIRE(run_ordering_pair(w, 2, 1) == 0);

  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
  REQUIRE(status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == 5);
  REQUIRE(status.consumed_index == 5);
  puts("Verified AQL barrier ordering against a concurrent-dispatch control");
  return 0;
}

static void *producer_thread(void *argument) {
  struct producer_context *context = (struct producer_context *)argument;
  struct workload *w = context->workload;
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *doorbell =
      (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  const uint64_t slot_count = w->transport.ring_byte_length / PACKET_BYTES;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;

  __atomic_add_fetch(&w->multi_ready, 1, __ATOMIC_RELEASE);
  uint32_t start;
  do {
    start = __atomic_load_n(&w->multi_start, __ATOMIC_ACQUIRE);
    relax_cpu();
  } while (start == 0);
  if (start != 1) return NULL;

  for (uint32_t local = 0; local < ITERATIONS / PRODUCER_THREADS; ++local) {
    const uint32_t dispatch = context->producer + local * PRODUCER_THREADS;
    const uint64_t reservation =
        __atomic_fetch_add(write_index, 1, __ATOMIC_ACQ_REL);
    w->reservations[dispatch] = reservation;

    if (local == 0) {
      __atomic_add_fetch(&w->first_reserved, 1, __ATOMIC_RELEASE);
      uint32_t publish;
      do {
        publish = __atomic_load_n(&w->publish_start, __ATOMIC_ACQUIRE);
        relax_cpu();
      } while (publish == 0);
      if (publish != 1) return NULL;
      if (reservation == 0) {
        uint32_t release;
        do {
          release = __atomic_load_n(&w->hole_release, __ATOMIC_ACQUIRE);
          relax_cpu();
        } while (release == 0);
        if (release != 1) return NULL;
      }
    }

    const uint64_t begin = now_ns();
    while (reservation - __atomic_load_n(read_index, __ATOMIC_ACQUIRE) >=
           slot_count) {
      if (now_ns() - begin >= timeout_ns) {
        uint64_t consumed = __atomic_load_n(read_index, __ATOMIC_ACQUIRE);
        uint64_t producer = __atomic_load_n(write_index, __ATOMIC_ACQUIRE);
        struct aql_dispatch_packet *ring =
            (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address;
        uint32_t header = __atomic_load_n(
            &ring[consumed & (slot_count - 1)].header_setup,
            __ATOMIC_ACQUIRE);
        uint32_t *completion = (uint32_t *)w->completion.host.pointer;
        uint32_t complete = 0;
        for (uint32_t index = 0; index < ITERATIONS; ++index)
          complete += __atomic_load_n(&completion[index], __ATOMIC_ACQUIRE) != 0;
        fprintf(stderr,
                "AQL producer %u timed out reserving dispatch %u at %" PRIu64
                ": consumed=%" PRIu64 " producer=%" PRIu64
                " next-header=0x%08" PRIx32 " complete=%" PRIu32 "\n",
                context->producer, dispatch, reservation, consumed, producer,
                header, complete);
        context->result = 1;
        return NULL;
      }
      relax_cpu();
    }

    struct aql_dispatch_packet packet;
    initialize_packet(w, &packet,
                      w->arguments.address +
                          (uint64_t)dispatch * sizeof(struct kernel_arguments));
    struct aql_dispatch_packet *slot =
        (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address +
        (reservation & (slot_count - 1));
    memcpy((unsigned char *)slot + sizeof(uint32_t),
           (const unsigned char *)&packet + sizeof(uint32_t),
           sizeof(packet) - sizeof(uint32_t));
    __atomic_store_n(&slot->header_setup, packet.header_setup, __ATOMIC_RELEASE);
    publish_doorbell(doorbell, reservation);
    if (local == 0 && reservation != 0) {
      __atomic_add_fetch(&w->later_published, 1, __ATOMIC_RELEASE);
      uint32_t release;
      do {
        release = __atomic_load_n(&w->hole_release, __ATOMIC_ACQUIRE);
        relax_cpu();
      } while (release == 0);
      if (release != 1) return NULL;
    }

    const uint32_t token = UINT32_C(0x40000000) + dispatch;
    const uint64_t completion_begin = now_ns();
    while (__atomic_load_n(&completion[dispatch], __ATOMIC_ACQUIRE) != token) {
      if (now_ns() - completion_begin >= timeout_ns) {
        fprintf(stderr,
                "AQL producer %u timed out completing dispatch %u: token=0x%08"
                PRIx32 "\n",
                context->producer, dispatch,
                __atomic_load_n(&completion[dispatch], __ATOMIC_ACQUIRE));
        context->result = 1;
        return NULL;
      }
      relax_cpu();
    }
  }
  return NULL;
}

static int join_producers(pthread_t *threads,
                          struct producer_context *contexts, uint32_t count) {
  int result = 0;
  for (uint32_t index = 0; index < count; ++index) {
    int error = pthread_join(threads[index], NULL);
    if (error != 0) {
      fprintf(stderr, "pthread_join failed: %d\n", error);
      result = 1;
    } else if (contexts[index].result != 0) {
      result = 1;
    }
  }
  return result;
}

static int check_reserved_frontier(struct workload *w) {
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  amdf_status_t result = w->api->user_queue_query_status(w->queue, &status);
  if (result != AMDF_STATUS_OK) {
    fprintf(stderr, "AQL reserved-frontier status failed: 0x%016" PRIx64
                    "\n",
            result);
    return 1;
  }
  if (status.producer_index != PRODUCER_THREADS || status.consumed_index != 0) {
    fprintf(stderr,
            "AQL reserved-frontier mismatch: producer=%" PRIu64
            " consumed=%" PRIu64 "\n",
            status.producer_index, status.consumed_index);
    return 1;
  }

  unsigned char seen[PRODUCER_THREADS];
  memset(seen, 0, sizeof(seen));
  const uint64_t slot_count = w->transport.ring_byte_length / PACKET_BYTES;
  struct aql_dispatch_packet *ring =
      (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address;
  for (uint32_t producer = 0; producer < PRODUCER_THREADS; ++producer) {
    const uint64_t reservation = w->reservations[producer];
    if (reservation >= PRODUCER_THREADS || seen[reservation] != 0) {
      fprintf(stderr, "AQL initial reservation is not unique: producer=%u "
                      "reservation=%" PRIu64 "\n",
              producer, reservation);
      return 1;
    }
    seen[reservation] = 1;
    uint32_t header = __atomic_load_n(
        &ring[reservation & (slot_count - 1)].header_setup, __ATOMIC_ACQUIRE);
    if ((header & UINT32_C(0xffff)) != 1) {
      fprintf(stderr, "AQL reserved packet was published early: slot=%" PRIu64
                      " header=0x%08" PRIx32 "\n",
              reservation, header);
      return 1;
    }
  }
  return 0;
}

static int check_published_hole(struct workload *w) {
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  amdf_status_t result = w->api->user_queue_query_status(w->queue, &status);
  if (result != AMDF_STATUS_OK) {
    fprintf(stderr, "AQL published-hole status failed: 0x%016" PRIx64 "\n",
            result);
    return 1;
  }
  if (status.producer_index != PRODUCER_THREADS || status.consumed_index != 0) {
    fprintf(stderr,
            "AQL published-hole mismatch: producer=%" PRIu64
            " consumed=%" PRIu64 "\n",
            status.producer_index, status.consumed_index);
    return 1;
  }

  struct aql_dispatch_packet *ring =
      (struct aql_dispatch_packet *)(uintptr_t)w->transport.ring_address;
  for (uint64_t reservation = 0; reservation < PRODUCER_THREADS;
       ++reservation) {
    uint32_t header =
        __atomic_load_n(&ring[reservation].header_setup, __ATOMIC_ACQUIRE);
    uint32_t expected = reservation == 0 ? 1 : 2;
    if ((header & UINT32_C(0xff)) != expected) {
      fprintf(stderr,
              "AQL published-hole packet mismatch: slot=%" PRIu64
              " type=%" PRIu32 " expected=%" PRIu32 "\n",
              reservation, header & UINT32_C(0xff), expected);
      return 1;
    }
  }
  return 0;
}

static int run_multiple(struct workload *w) {
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint32_t *source = (uint32_t *)w->source.host.pointer;
  uint32_t *target = (uint32_t *)w->target.host.pointer;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  struct kernel_arguments *arguments =
      (struct kernel_arguments *)w->arguments.host.pointer;
  REQUIRE(__atomic_load_n(read_index, __ATOMIC_ACQUIRE) == 0);
  REQUIRE(__atomic_load_n(write_index, __ATOMIC_ACQUIRE) == 0);

  memset(arguments, 0, ITERATIONS * sizeof(*arguments));
  for (uint32_t dispatch = 0; dispatch < ITERATIONS; ++dispatch) {
    const uint32_t token = UINT32_C(0x40000000) + dispatch;
    source[dispatch] = dispatch * 17 + 3;
    target[dispatch] = UINT32_C(0xa5a5a5a5);
    __atomic_store_n(&completion[dispatch], 0, __ATOMIC_RELAXED);
    arguments[dispatch].source = w->source.address + dispatch * sizeof(uint32_t);
    arguments[dispatch].target = w->target.address + dispatch * sizeof(uint32_t);
    arguments[dispatch].completion =
        w->completion.address + dispatch * sizeof(uint32_t);
    arguments[dispatch].count = 1;
    arguments[dispatch].addend = UINT32_C(0x2030405) + dispatch;
    arguments[dispatch].token = token;
  }
  REQUIRE(publish_local(w, ITERATIONS * sizeof(uint32_t)) == 0);

  pthread_t threads[PRODUCER_THREADS];
  struct producer_context contexts[PRODUCER_THREADS];
  uint32_t created = 0;
  w->in_flight = 1;
  for (; created < PRODUCER_THREADS; ++created) {
    contexts[created].workload = w;
    contexts[created].producer = created;
    contexts[created].result = 0;
    int error = pthread_create(&threads[created], NULL, producer_thread,
                               &contexts[created]);
    if (error != 0) {
      fprintf(stderr, "pthread_create failed: %d\n", error);
      __atomic_store_n(&w->multi_start, 2, __ATOMIC_RELEASE);
      break;
    }
  }
  if (created != PRODUCER_THREADS) {
    int join_result = join_producers(threads, contexts, created);
    if (join_result == 0) w->in_flight = 0;
    return 1;
  }

  if (wait_for_at_least(&w->multi_ready, PRODUCER_THREADS,
                        "producer readiness") != 0) {
    __atomic_store_n(&w->multi_start, 2, __ATOMIC_RELEASE);
    int join_result = join_producers(threads, contexts, PRODUCER_THREADS);
    if (join_result == 0) w->in_flight = 0;
    return 1;
  }
  __atomic_store_n(&w->multi_start, 1, __ATOMIC_RELEASE);
  if (wait_for_at_least(&w->first_reserved, PRODUCER_THREADS,
                        "initial reservations") != 0) {
    __atomic_store_n(&w->publish_start, 2, __ATOMIC_RELEASE);
    (void)join_producers(threads, contexts, PRODUCER_THREADS);
    return 1;
  }

  if (check_reserved_frontier(w) != 0) {
    __atomic_store_n(&w->publish_start, 2, __ATOMIC_RELEASE);
    (void)join_producers(threads, contexts, PRODUCER_THREADS);
    return 1;
  }
  __atomic_store_n(&w->publish_start, 1, __ATOMIC_RELEASE);
  if (wait_for_at_least(&w->later_published, PRODUCER_THREADS - 1,
                        "out-of-order publications") != 0) {
    __atomic_store_n(&w->hole_release, 2, __ATOMIC_RELEASE);
    (void)join_producers(threads, contexts, PRODUCER_THREADS);
    return 1;
  }
  if (check_published_hole(w) != 0) {
    __atomic_store_n(&w->hole_release, 2, __ATOMIC_RELEASE);
    (void)join_producers(threads, contexts, PRODUCER_THREADS);
    return 1;
  }
  __atomic_store_n(&w->hole_release, 1, __ATOMIC_RELEASE);

  if (join_producers(threads, contexts, PRODUCER_THREADS) != 0) return 1;

  const uint64_t begin = now_ns();
  for (;;) {
    uint32_t complete = 0;
    for (uint32_t dispatch = 0; dispatch < ITERATIONS; ++dispatch) {
      const uint32_t token = UINT32_C(0x40000000) + dispatch;
      complete += __atomic_load_n(&completion[dispatch], __ATOMIC_ACQUIRE) ==
                  token;
    }
    if (complete == ITERATIONS &&
        __atomic_load_n(read_index, __ATOMIC_ACQUIRE) >= ITERATIONS)
      break;
    if (now_ns() - begin >= timeout_ns) {
      fprintf(stderr,
              "AQL multi-producer completion timeout: complete=%u consumed=%"
              PRIu64 "\n",
              complete, __atomic_load_n(read_index, __ATOMIC_ACQUIRE));
      return 1;
    }
    relax_cpu();
  }

  REQUIRE(observe_local(w, ITERATIONS * sizeof(uint32_t)) == 0);
  unsigned char seen[ITERATIONS];
  memset(seen, 0, sizeof(seen));
  for (uint32_t dispatch = 0; dispatch < ITERATIONS; ++dispatch) {
    REQUIRE(w->reservations[dispatch] < ITERATIONS);
    REQUIRE(seen[w->reservations[dispatch]] == 0);
    seen[w->reservations[dispatch]] = 1;
    REQUIRE(target[dispatch] == source[dispatch] + arguments[dispatch].addend);
  }
  CALL(w->api->user_queue_wait_consumed(w->queue, ITERATIONS, 0, 0));
  amdf_user_queue_status_t status;
  INIT(status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS);
  CALL(w->api->user_queue_query_status(w->queue, &status));
  REQUIRE(status.state == AMDF_QUEUE_STATE_ACTIVE);
  REQUIRE(status.terminal_status == AMDF_STATUS_OK);
  REQUIRE(status.producer_index == ITERATIONS);
  REQUIRE(status.consumed_index == ITERATIONS);
  w->in_flight = 0;
  printf("Verified %u scratch-backed AQL dispatches from %u host producers, "
         "%" PRIu64 " ring wraps, and one recovered publication hole\n",
         ITERATIONS, PRODUCER_THREADS,
         (uint64_t)ITERATIONS /
             (w->transport.ring_byte_length / PACKET_BYTES));
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
    fputs("Retaining GPU resources after incomplete AQL work.\n", stderr);
    return 1;
  }
  REQUIRE(destroy_queue(w) == 0);
  REQUIRE(release_buffer(w, &w->scratch) == 0);
  REQUIRE(release_buffer(w, &w->completion) == 0);
  REQUIRE(release_buffer(w, &w->target) == 0);
  REQUIRE(release_buffer(w, &w->source) == 0);
  REQUIRE(release_buffer(w, &w->arguments) == 0);
  REQUIRE(release_buffer(w, &w->code) == 0);
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
  struct workload workload;
  memset(&workload, 0, sizeof(workload));
  for (int index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "instance") == 0)
      lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
    else if (strcmp(argv[index], "local") == 0)
      workload.local_mode = 1;
    else {
      fprintf(stderr, "Usage: %s [instance] [local]\n", argv[0]);
      return 2;
    }
  }
  int result = prepare(&workload, lifetime);
  if (result == 0) result = run_single(&workload, "normal");
  if (result == 0) result = destroy_queue(&workload);
  if (result == 0)
    result = create_queue(&workload, AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                          AMDF_QUEUE_PRIORITY_LOW);
  if (result == 0) result = run_single(&workload, "low");
  if (result == 0) result = destroy_queue(&workload);
  if (result == 0)
    result = create_queue(&workload, AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                          AMDF_QUEUE_PRIORITY_HIGH);
  if (result == 0) result = run_single(&workload, "high");
  if (result == 0) result = destroy_queue(&workload);
  if (result == 0)
    result = create_queue(&workload, AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                          AMDF_QUEUE_PRIORITY_NORMAL);
  if (result == 0) result = run_barrier_ordering(&workload);
  if (result == 0) result = destroy_queue(&workload);
  if (result == 0)
    result = create_queue(&workload, AMDF_QUEUE_PRODUCER_MODE_MULTI,
                          AMDF_QUEUE_PRIORITY_NORMAL);
  if (result == 0) result = run_multiple(&workload);
  int cleanup_result = cleanup(&workload);
  return result || cleanup_result;
}
