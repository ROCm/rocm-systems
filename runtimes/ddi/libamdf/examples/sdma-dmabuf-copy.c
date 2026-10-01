/* GFX1201 qualification for SYSTEM-memory DMA-BUF export, import, and SDMA use.
 * The provider owns attachment and descriptor lifetimes. Packet construction,
 * publication, completion, and consumption remain explicit caller policy. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define INIT(record, tag)                                                      \
  do {                                                                         \
    memset(&(record), 0, sizeof(record));                                       \
    (record).type = (tag);                                                      \
    (record).structure_size = (uint32_t)sizeof(record);                         \
  } while (0)
#define REQUIRE(expression)                                                     \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "SDMA DMA-BUF: line %d: %s\n", __LINE__, #expression);  \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define CALL(expression)                                                        \
  do {                                                                         \
    amdf_status_t call_status = (expression);                                   \
    if (call_status != AMDF_STATUS_OK) {                                        \
      fprintf(stderr,                                                          \
              "SDMA DMA-BUF: line %d: %s: status=0x%016" PRIx64 "\n",        \
              __LINE__, #expression, call_status);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

enum { PAGE_BYTES = 4096, SLOT_WORDS = 32 };
static const uint64_t timeout_ns = UINT64_C(5000000000);
static const amdf_queue_format_features_t required_format_features =
    AMDF_GPU_SDMA_FORMAT_FEATURE_GCR |
    AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;

struct buffer {
  amdf_memory_t *memory;
  amdf_host_mapping_t *mapping;
  amdf_host_mapping_info_t host;
  amdf_memory_info_t info;
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
  amdf_user_queue_mapping_t *queue_mapping;
  amdf_user_queue_mapping_info_t transport;
  amdf_memory_device_access_t access;
  uint32_t profile;
  uint32_t family;
  amdf_queue_format_features_t format_features;
  struct buffer original;
  struct buffer imported;
  struct buffer target;
  struct buffer completion;
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

static int select_gpu(struct workload *w) {
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
  for (uint32_t i = 0; i < count; ++i) {
    if (endpoints[i].engine_kind != AMDF_ENGINE_KIND_GPU) continue;
    status = w->api->endpoint_open(w->instance, &endpoints[i].id,
                                   &w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    amdf_gpu_endpoint_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO);
    status = w->gpu->endpoint_query_info(w->endpoint, &info);
    if (status != AMDF_STATUS_OK) break;
    if (info.gfx_ip.major == 12 && info.gfx_ip.minor == 0 &&
        info.gfx_ip.stepping == 1) {
      break;
    }
    status = w->api->endpoint_close(w->endpoint);
    if (status != AMDF_STATUS_OK) break;
    w->endpoint = NULL;
  }
  free(endpoints);
  REQUIRE(status == AMDF_STATUS_OK && w->endpoint != NULL);
  return 0;
}

static int select_sdma(struct workload *w) {
  amdf_endpoint_info_t endpoint;
  INIT(endpoint, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO);
  CALL(w->api->endpoint_query_info(w->endpoint, &endpoint));
  for (w->family = 0; w->family < endpoint.queue_family_count; ++w->family) {
    amdf_queue_family_info_t family;
    INIT(family, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO);
    CALL(w->api->endpoint_query_queue_family_info(w->endpoint, w->family,
                                                  &family));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
        family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1 &&
        (family.roles & AMDF_QUEUE_ROLE_CACHE_CONTROL) != 0 &&
        (family.cache_operations &
         (AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM)) ==
            (AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
             AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) &&
        family.format_features == required_format_features) {
      w->format_features = family.format_features;
      return 0;
    }
  }
  REQUIRE(w->family < endpoint.queue_family_count);
  return 0;
}

static int select_profile(struct workload *w) {
  amdf_memory_scope_info_t scope;
  INIT(scope, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
  CALL(w->api->memory_scope_query_info(w->scope, &scope));
  for (w->profile = 0; w->profile < scope.memory_profile_count; ++w->profile) {
    amdf_memory_profile_t profile;
    amdf_memory_access_capabilities_t capabilities;
    INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
    INIT(capabilities, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
    amdf_status_t status = w->api->memory_scope_query_device_profile(
        w->scope, w->profile, 1, &w->access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
    REQUIRE(status == AMDF_STATUS_OK);
    const uint64_t roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                           AMDF_MEMORY_PROFILE_ROLE_IMPORT |
                           AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                           AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    if (profile.memory_class != AMDF_MEMORY_CLASS_SYSTEM ||
        (profile.roles & roles) != roles ||
        (profile.guaranteed_flags & AMDF_MEMORY_FLAG_SHAREABLE) == 0 ||
        profile.external_memory_support_count != 1) {
      continue;
    }
    const amdf_external_memory_support_t *support =
        &profile.external_memory_support[0];
    const uint32_t flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                           AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                           AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                           AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS;
    REQUIRE(support->type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
    REQUIRE((support->flags & flags) == flags);
    REQUIRE((support->flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API) ==
            0);
    REQUIRE(support->source_offset_alignment == PAGE_BYTES);
    REQUIRE(support->byte_length_alignment == 1);
    REQUIRE(profile.import.minimum_alignment == PAGE_BYTES);
    return 0;
  }
  REQUIRE(w->profile < scope.memory_profile_count);
  return 0;
}

static int create_buffer(struct workload *w, struct buffer *buffer,
                         uint64_t bytes) {
  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = w->profile;
  create.access_count = 1;
  create.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  create.byte_length = bytes;
  create.minimum_alignment = PAGE_BYTES;
  create.accesses = &w->access;
  CALL(w->api->memory_create(w->scope, &create, &buffer->memory));
  INIT(buffer->info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CALL(w->api->memory_query_info(buffer->memory, &buffer->info));
  REQUIRE(buffer->info.flags & AMDF_MEMORY_FLAG_SHAREABLE);
  REQUIRE(amdf_physical_memory_id_is_valid(&buffer->info.physical_backing_id));
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(buffer->memory, &map, &buffer->mapping));
  INIT(buffer->host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(buffer->mapping, &buffer->host));
  CALL(w->api->memory_query_address(buffer->memory, 0,
                                    AMDF_MEMORY_ADDRESS_GPU,
                                    &buffer->address));
  return 0;
}

static int import_buffer(struct workload *w, amdf_external_memory_t *external) {
  amdf_memory_device_access_t source_access = w->access;
  source_access.requirements.access = AMDF_MEMORY_ACCESS_READ;
  amdf_memory_import_info_t import;
  INIT(import, AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO);
  import.memory_profile_ordinal = w->profile;
  import.access_count = 1;
  import.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  import.minimum_alignment = PAGE_BYTES;
  import.accesses = &source_access;

  amdf_external_memory_t invalid = *external;
  invalid.release = NULL;
  invalid.release_user_data = NULL;
  ++invalid.source_byte_offset;
  amdf_memory_t *sentinel = (amdf_memory_t *)(uintptr_t)1;
  amdf_memory_t *output = sentinel;
  REQUIRE(w->api->memory_import(w->scope, &import, &invalid, &output) ==
          amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  REQUIRE(output == sentinel && invalid.source_byte_offset == PAGE_BYTES + 1);

  invalid = *external;
  invalid.release = NULL;
  invalid.release_user_data = NULL;
  invalid.payload.file_descriptor = -1;
  output = sentinel;
  REQUIRE(w->api->memory_import(w->scope, &import, &invalid, &output) ==
          amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  REQUIRE(output == sentinel && invalid.payload.file_descriptor == -1);

  invalid = *external;
  invalid.release = NULL;
  invalid.release_user_data = NULL;
  invalid.physical_backing_id.words[0] ^= 1;
  output = sentinel;
  REQUIRE(w->api->memory_import(w->scope, &import, &invalid, &output) ==
          amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  REQUIRE(output == sentinel && invalid.type == external->type);
  REQUIRE(invalid.payload.file_descriptor == external->payload.file_descriptor);
  REQUIRE(invalid.physical_backing_id.words[0] ==
          (external->physical_backing_id.words[0] ^ 1));

  const int consumed_descriptor = (int)external->payload.file_descriptor;
  CALL(w->api->memory_import(w->scope, &import, external,
                             &w->imported.memory));
  amdf_external_memory_t empty;
  memset(&empty, 0, sizeof(empty));
  REQUIRE(memcmp(external, &empty, sizeof(empty)) == 0);
  errno = 0;
  REQUIRE(fcntl(consumed_descriptor, F_GETFD) == -1 && errno == EBADF);

  INIT(w->imported.info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CALL(w->api->memory_query_info(w->imported.memory, &w->imported.info));
  REQUIRE(w->imported.info.memory_profile_ordinal == w->profile);
  REQUIRE(w->imported.info.memory_class == AMDF_MEMORY_CLASS_SYSTEM);
  REQUIRE(w->imported.info.source_byte_offset == PAGE_BYTES);
  REQUIRE(w->imported.info.byte_length == PAGE_BYTES);
  REQUIRE(w->imported.info.native_allocation_byte_length == 3 * PAGE_BYTES);
  REQUIRE(amdf_physical_memory_id_is_equal(
      &w->imported.info.physical_backing_id,
      &w->original.info.physical_backing_id));
  amdf_memory_access_info_t access;
  INIT(access, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO);
  CALL(w->api->memory_query_access_info(w->imported.memory, 0, &access));
  REQUIRE(access.access == AMDF_MEMORY_ACCESS_READ);
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = PAGE_BYTES;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  CALL(w->api->memory_map(w->imported.memory, &map, &w->imported.mapping));
  INIT(w->imported.host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(w->api->host_mapping_query_info(w->imported.mapping,
                                       &w->imported.host));
  CALL(w->api->memory_query_address(w->imported.memory, 0,
                                    AMDF_MEMORY_ADDRESS_GPU,
                                    &w->imported.address));
  return 0;
}

static int verify_child_import(const struct workload *parent,
                               const amdf_external_memory_t *external) {
  struct workload child;
  memset(&child, 0, sizeof(child));
  child.api = parent->api;
  child.gpu = parent->gpu;
  child.profile = parent->profile;
  amdf_instance_create_info_t instance;
  INIT(instance, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
  instance.native_lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
  CALL(child.api->instance_create(&instance, &child.instance));
  uint32_t count = 0;
  CALL(child.api->instance_enumerate_memory_scopes(child.instance, 1,
                                                   &child.scope, &count));
  REQUIRE(count == 1 && child.scope != NULL);
  REQUIRE(select_gpu(&child) == 0);
  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(child.gpu->device_create(child.endpoint, &device, &child.device));
  child.access.device = child.device;
  child.access.requirements.access = AMDF_MEMORY_ACCESS_READ;
  child.access.requirements.flags =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  child.access.requirements.address_kinds =
      UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  amdf_memory_import_info_t request;
  INIT(request, AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO);
  request.memory_profile_ordinal = child.profile;
  request.access_count = 1;
  request.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  request.minimum_alignment = PAGE_BYTES;
  request.accesses = &child.access;
  amdf_external_memory_t borrowed = *external;
  borrowed.release = NULL;
  borrowed.release_user_data = NULL;
  CALL(child.api->memory_import(child.scope, &request, &borrowed,
                                &child.imported.memory));
  amdf_external_memory_t empty = {0};
  REQUIRE(memcmp(&borrowed, &empty, sizeof(empty)) == 0);
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = PAGE_BYTES;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ;
  CALL(child.api->memory_map(child.imported.memory, &map,
                             &child.imported.mapping));
  INIT(child.imported.host, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CALL(child.api->host_mapping_query_info(child.imported.mapping,
                                          &child.imported.host));
  const unsigned char *bytes = child.imported.host.pointer;
  for (uint32_t i = 0; i < PAGE_BYTES; ++i)
    REQUIRE(bytes[i] == (unsigned char)(i * 17));
  CALL(child.api->host_mapping_destroy(child.imported.mapping));
  CALL(child.api->memory_destroy(child.imported.memory));
  CALL(child.api->device_destroy(child.device));
  CALL(child.api->endpoint_close(child.endpoint));
  CALL(child.api->instance_destroy(child.instance));
  return 0;
}

static int verify_cross_process_import(const struct workload *w,
                                        const amdf_external_memory_t *external) {
  pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    const int result = verify_child_import(w, external);
    _exit(result);
  }
  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}

static int check_reexport(struct workload *w) {
  amdf_memory_export_info_t request;
  INIT(request, AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO);
  request.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  request.byte_length = PAGE_BYTES;
  amdf_external_memory_t first, second;
  memset(&first, 0, sizeof(first));
  memset(&second, 0, sizeof(second));
  CALL(w->api->memory_export(w->imported.memory, &request, &first));
  CALL(w->api->memory_export(w->imported.memory, &request, &second));
  REQUIRE(first.source_byte_offset == PAGE_BYTES);
  REQUIRE(first.byte_length == PAGE_BYTES);
  REQUIRE(amdf_physical_memory_id_is_equal(
      &first.physical_backing_id, &w->imported.info.physical_backing_id));
  const int first_fd = (int)first.payload.file_descriptor;
  const int second_fd = (int)second.payload.file_descriptor;
  REQUIRE(first_fd != second_fd);
  const int first_flags = fcntl(first_fd, F_GETFD);
  const int second_flags = fcntl(second_fd, F_GETFD);
  REQUIRE(first_flags >= 0 && (first_flags & FD_CLOEXEC) != 0);
  REQUIRE(second_flags >= 0 && (second_flags & FD_CLOEXEC) != 0);
  w->api->external_memory_release(&first);
  errno = 0;
  REQUIRE(fcntl(first_fd, F_GETFD) == -1 && errno == EBADF);
  REQUIRE(fcntl(second_fd, F_GETFD) != -1);
  w->api->external_memory_release(&second);
  errno = 0;
  REQUIRE(fcntl(second_fd, F_GETFD) == -1 && errno == EBADF);
  return 0;
}

static void encode(uint32_t words[SLOT_WORDS], uint64_t source,
                   uint64_t target, uint64_t fence) {
  const uint32_t commands[SLOT_WORDS] = {
      0x111, 0, 0xc3c00000, 0, 0,
      1, PAGE_BYTES - 1, 0, (uint32_t)source, (uint32_t)(source >> 32),
      (uint32_t)target, (uint32_t)(target >> 32),
      0x111, 0, 0x80400000, 0, 0,
      0x00130005, (uint32_t)fence, (uint32_t)(fence >> 32), 1};
  memcpy(words, commands, sizeof(commands));
}

static int run_sdma(struct workload *w) {
  unsigned char *source = (unsigned char *)w->imported.host.pointer;
  unsigned char *target = (unsigned char *)w->target.host.pointer;
  uint32_t *completion = (uint32_t *)w->completion.host.pointer;
  for (uint32_t i = 0; i < PAGE_BYTES; ++i) source[i] = (unsigned char)(i * 17);
  memset(target, 0xa5, PAGE_BYTES);
  __atomic_store_n(completion, 0, __ATOMIC_RELAXED);
  uint32_t commands[SLOT_WORDS];
  encode(commands, w->imported.address, w->target.address,
         w->completion.address);
  memcpy((void *)(uintptr_t)w->transport.ring_address, commands,
         sizeof(commands));
  w->in_flight = 1;
  uint64_t *write_index =
      (uint64_t *)(uintptr_t)w->transport.write_index_address;
  uint64_t *read_index =
      (uint64_t *)(uintptr_t)w->transport.read_index_address;
  uint64_t *doorbell = (uint64_t *)(uintptr_t)w->transport.doorbell_address;
  __atomic_store_n(write_index, sizeof(commands), __ATOMIC_RELEASE);
  __atomic_store_n(doorbell, sizeof(commands), __ATOMIC_RELEASE);
  const uint64_t begin = now_ns();
  while (__atomic_load_n(completion, __ATOMIC_ACQUIRE) != 1 ||
         __atomic_load_n(read_index, __ATOMIC_ACQUIRE) < sizeof(commands)) {
    if (now_ns() - begin >= timeout_ns) return 1;
#if defined(__x86_64__)
    __asm__ __volatile__("pause");
#endif
  }
  w->in_flight = 0;
  REQUIRE(memcmp(source, target, PAGE_BYTES) == 0);
  CALL(w->api->user_queue_wait_consumed(w->queue, sizeof(commands), 0, 0));
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
  REQUIRE(select_gpu(w) == 0);
  REQUIRE(select_sdma(w) == 0);
  amdf_gpu_device_create_info_t device;
  INIT(device, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
  CALL(w->gpu->device_create(w->endpoint, &device, &w->device));
  memset(&w->access, 0, sizeof(w->access));
  w->access.device = w->device;
  w->access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                  AMDF_MEMORY_ACCESS_WRITE |
                                  AMDF_MEMORY_ACCESS_EXECUTE;
  w->access.requirements.flags =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  w->access.requirements.address_kinds =
      UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  REQUIRE(select_profile(w) == 0);
  REQUIRE(create_buffer(w, &w->original, 3 * PAGE_BYTES) == 0);
  unsigned char *original = (unsigned char *)w->original.host.pointer;
  for (uint32_t i = 0; i < PAGE_BYTES; ++i) {
    original[PAGE_BYTES + i] = (unsigned char)(i * 17);
  }

  amdf_memory_export_info_t export_request;
  INIT(export_request, AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO);
  export_request.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  export_request.byte_offset = PAGE_BYTES;
  export_request.byte_length = PAGE_BYTES;
  amdf_external_memory_t external;
  memset(&external, 0, sizeof(external));
  CALL(w->api->memory_export(w->original.memory, &export_request, &external));
  REQUIRE(external.source_byte_offset == PAGE_BYTES);
  REQUIRE(external.byte_length == PAGE_BYTES);
  REQUIRE(external.release != NULL);
  REQUIRE((external.provenance.words[0] | external.provenance.words[1]) == 0);
  REQUIRE(amdf_physical_memory_id_is_equal(
      &external.physical_backing_id, &w->original.info.physical_backing_id));
  const int descriptor_flags =
      fcntl((int)external.payload.file_descriptor, F_GETFD);
  REQUIRE(descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0);
  REQUIRE(verify_cross_process_import(w, &external) == 0);

  amdf_memory_export_info_t overrun = export_request;
  overrun.byte_offset = 3 * PAGE_BYTES - 1;
  overrun.byte_length = 2;
  amdf_external_memory_t unchanged;
  memset(&unchanged, 0xa5, sizeof(unchanged));
  REQUIRE(w->api->memory_export(w->original.memory, &overrun, &unchanged) ==
          amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  const unsigned char *bytes = (const unsigned char *)&unchanged;
  for (size_t i = 0; i < sizeof(unchanged); ++i) REQUIRE(bytes[i] == 0xa5);

  REQUIRE(import_buffer(w, &external) == 0);
  REQUIRE(memcmp(w->imported.host.pointer, original + PAGE_BYTES,
                 PAGE_BYTES) == 0);
  CALL(w->api->host_mapping_destroy(w->original.mapping));
  w->original.mapping = NULL;
  CALL(w->api->memory_destroy(w->original.memory));
  w->original.memory = NULL;
  REQUIRE(((unsigned char *)w->imported.host.pointer)[17] ==
          (unsigned char)(17 * 17));
  REQUIRE(check_reexport(w) == 0);
  REQUIRE(create_buffer(w, &w->target, PAGE_BYTES) == 0);
  REQUIRE(create_buffer(w, &w->completion, PAGE_BYTES) == 0);
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
  CALL(w->api->user_queue_map(w->queue, NULL, &w->queue_mapping));
  INIT(w->transport, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO);
  CALL(w->api->user_queue_mapping_query_info(w->queue_mapping,
                                             &w->transport));
  REQUIRE(w->transport.format_features == w->format_features);
  return 0;
}

static int release_buffer(struct workload *w, struct buffer *buffer) {
  if (buffer->mapping) {
    CALL(w->api->host_mapping_destroy(buffer->mapping));
    buffer->mapping = NULL;
  }
  if (buffer->memory) {
    CALL(w->api->memory_destroy(buffer->memory));
    buffer->memory = NULL;
  }
  return 0;
}

static int cleanup(struct workload *w) {
  if (w->in_flight) {
    fputs("Retaining DMA-BUF resources after incomplete GPU work.\n", stderr);
    return 1;
  }
  if (w->queue_mapping) {
    CALL(w->api->user_queue_mapping_destroy(w->queue_mapping));
    w->queue_mapping = NULL;
  }
  if (w->queue) {
    CALL(w->api->user_queue_destroy(w->queue));
    w->queue = NULL;
  }
  REQUIRE(release_buffer(w, &w->completion) == 0);
  REQUIRE(release_buffer(w, &w->target) == 0);
  REQUIRE(release_buffer(w, &w->imported) == 0);
  REQUIRE(release_buffer(w, &w->original) == 0);
  if (w->device) CALL(w->api->device_destroy(w->device));
  if (w->endpoint) CALL(w->api->endpoint_close(w->endpoint));
  if (w->instance) CALL(w->api->instance_destroy(w->instance));
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
  if (result == 0) result = run_sdma(&workload);
  if (result == 0) {
    puts("Verified GFX1201 DMA-BUF subrange import, lifetime, re-export, and SDMA");
  }
  const int cleanup_result = cleanup(&workload);
  return result || cleanup_result;
}
