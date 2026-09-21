/* CPU CREATE/REGISTER and explicit subrange mapping; optional GPU allocation.
 * Set AMDF_REQUIRE_GPU=1 to require native GPU memory qualification. */
#define _POSIX_C_SOURCE 200809L
#include <amdf/amdf.h>
#include <amdf/gpu.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "AMDF memory check failed at line %d: %s\n", __LINE__,   \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define INIT(record, tag)                                                      \
  do {                                                                         \
    memset(&(record), 0, sizeof(record));                                      \
    (record).type = (tag);                                                     \
    (record).structure_size = (uint32_t)sizeof(record);                        \
  } while (0)
#define API_ERROR(code) amdf_make_api_status(AMDF_STATUS_CODE_##code)

static int check_mapping(const amdf_api_t *api, amdf_memory_t *memory,
                         unsigned char *registered_base) {
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_offset = 3;
  map.byte_length = 257;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  amdf_host_mapping_t *mapping = NULL;
  CHECK(api->memory_map(memory, &map, &mapping) == AMDF_STATUS_OK);
  amdf_host_mapping_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CHECK(api->host_mapping_query_info(mapping, &info) == AMDF_STATUS_OK);
  CHECK(info.pointer != NULL && info.memory_byte_offset == 3 &&
        info.byte_length == 257);
  CHECK((info.flags & map.flags) == map.flags);
  CHECK(info.byte_offset_granularity == 1 && info.byte_length_granularity == 1);
  if (registered_base)
    CHECK(info.pointer == registered_base + 3);
  volatile unsigned char *bytes = (volatile unsigned char *)info.pointer;
  bytes[0] = 0xa5;
  bytes[256] = 0x5a;
  CHECK(bytes[0] == 0xa5 && bytes[256] == 0x5a);
  if (registered_base) {
    CHECK(info.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
    CHECK(info.cache_line_size != 0);
    CHECK(info.flush.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
    CHECK(info.invalidate.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
  } else {
    CHECK(info.cache_line_size != 0);
    CHECK(info.flush.host_operation == AMDF_HOST_CACHE_OPERATION_FLUSH);
    CHECK(info.invalidate.host_operation ==
          AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  }
  CHECK(api->host_mapping_cache_control(
            mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 1, 255) == AMDF_STATUS_OK);
  CHECK(api->host_mapping_cache_control(mapping,
                                        AMDF_HOST_CACHE_OPERATION_INVALIDATE, 1,
                                        255) == AMDF_STATUS_OK);
  CHECK(api->host_mapping_cache_control(mapping,
                                        AMDF_HOST_CACHE_OPERATION_FLUSH, 257,
                                        0) == AMDF_STATUS_OK);
  CHECK(api->host_mapping_cache_control(mapping,
                                        AMDF_HOST_CACHE_OPERATION_FLUSH, 257,
                                        1) == API_ERROR(INVALID_ARGUMENT));
  amdf_host_mapping_info_t saved;
  info.structure_size = 0;
  memcpy(&saved, &info, sizeof(saved));
  CHECK(api->host_mapping_query_info(mapping, &info) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(memcmp(&saved, &info, sizeof(info)) == 0);
  CHECK(api->host_mapping_destroy(mapping) == AMDF_STATUS_OK);
  return 0;
}

static int check_pair_identity(const amdf_api_t *api,
                               amdf_memory_scope_t *scope,
                               const amdf_memory_create_info_t *create,
                               amdf_memory_t *memory, int registered) {
  /* Separate CREATE calls have distinct known backing. Two REGISTER calls over
   * the same address have no proven physical identity. The API distinguishes
   * those cases even though both reject a cross-resource cache-pair query. */
  amdf_memory_t *peer = NULL;
  CHECK(api->memory_create(scope, create, &peer) == AMDF_STATUS_OK);
  amdf_memory_t *resources[2] = {memory, peer};
  amdf_host_mapping_t *mappings[2] = {NULL, NULL};
  amdf_memory_site_t sites[2];
  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = create->byte_length;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  for (unsigned i = 0; i < 2; ++i) {
    CHECK(api->memory_map(resources[i], &map, &mappings[i]) == AMDF_STATUS_OK);
    INIT(sites[i], AMDF_STRUCTURE_TYPE_MEMORY_SITE);
    sites[i].kind = AMDF_MEMORY_SITE_KIND_HOST;
    sites[i].value.host_mapping = mappings[i];
  }
  amdf_memory_pair_info_t pair;
  INIT(pair, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO);
  /* A single qualified mapping always describes shared backing with itself. */
  amdf_memory_pair_info_t saved;
  memcpy(&saved, &pair, sizeof(saved));
  CHECK(api->memory_query_pair_info(&sites[0], &sites[0], &pair) ==
        AMDF_STATUS_OK);
  CHECK((pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE) != 0);
  memcpy(&saved, &pair, sizeof(saved));
  CHECK(api->memory_query_pair_info(&sites[0], &sites[1], &pair) ==
        (registered ? API_ERROR(UNSUPPORTED) : API_ERROR(FAILED_PRECONDITION)));
  CHECK(memcmp(&saved, &pair, sizeof(pair)) == 0);
  for (unsigned i = 0; i < 2; ++i)
    CHECK(api->host_mapping_destroy(mappings[i]) == AMDF_STATUS_OK);
  CHECK(api->memory_destroy(peer) == AMDF_STATUS_OK);
  return 0;
}

static int check_cpu_profile(const amdf_api_t *api, amdf_memory_scope_t *scope,
                             const amdf_memory_profile_t *profile) {
  const int registered =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  CHECK((profile->roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0);
  const amdf_memory_construction_capabilities_t *limits =
      registered ? &profile->registration : &profile->allocation;
  CHECK(limits->byte_length_granularity == 1);
  CHECK(limits->native_byte_length_granularity != 0);
  CHECK(limits->native_byte_length_prefix == 0);
  unsigned char *host = NULL;
  if (registered) {
    CHECK(limits->registered_host_pointer_alignment == 1);
    CHECK(limits->registered_host_cacheability ==
          AMDF_HOST_CACHEABILITY_WRITE_BACK);
    host = (unsigned char *)malloc(8192);
    CHECK(host != NULL);
    memset(host, 0, 8192);
  }
  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = profile->ordinal;
  create.byte_length = 4097;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.registered_host_pointer = registered ? host + 1 : NULL;
  create.registered_host_cacheability =
      registered ? limits->registered_host_cacheability
                 : (amdf_host_cacheability_t)AMDF_HOST_CACHEABILITY_UNKNOWN;
  amdf_memory_t *memory = NULL;
  CHECK(api->memory_create(scope, &create, &memory) == AMDF_STATUS_OK);
  amdf_memory_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CHECK(api->memory_query_info(memory, &info) == AMDF_STATUS_OK);
  CHECK(info.byte_length == 4097 && info.access_count == 0);
  const uint64_t expected_native_length =
      (info.byte_length + limits->native_byte_length_prefix +
       limits->native_byte_length_granularity - 1) /
      limits->native_byte_length_granularity *
      limits->native_byte_length_granularity;
  CHECK(info.source_byte_offset ==
        (registered ? (uint64_t)(uintptr_t)(host + 1) &
                          (limits->native_byte_length_granularity - 1)
                    : limits->native_byte_length_prefix));
  CHECK(registered || info.native_allocation_byte_length ==
                          expected_native_length);
  CHECK(info.native_allocation_granularity ==
        limits->native_byte_length_granularity);
  CHECK(check_mapping(api, memory, registered ? host + 1 : NULL) == 0);
  CHECK(check_pair_identity(api, scope, &create, memory, registered) == 0);
  if (registered)
    CHECK(host[4] == 0xa5 && host[260] == 0x5a);
  uint64_t address = UINT64_C(0xfeedface);
  CHECK(api->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                  &address) == API_ERROR(OUT_OF_RANGE));
  CHECK(address == UINT64_C(0xfeedface));
  CHECK(api->memory_destroy(memory) == AMDF_STATUS_OK);
  if (registered) {
    host[1] = 0x7a;
    CHECK(host[1] == 0x7a);
    free(host);
  }
  create.registered_host_pointer = NULL;
  create.byte_length = 0;
  amdf_memory_t *sentinel = (amdf_memory_t *)&create;
  memory = sentinel;
  CHECK(api->memory_create(scope, &create, &memory) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(memory == sentinel);
  return 0;
}

static int
check_gpu_registration(const amdf_api_t *api, amdf_memory_scope_t *scope,
                       const amdf_memory_profile_t *profile,
                       const amdf_memory_access_capabilities_t *capabilities,
                       const amdf_memory_device_access_t *access) {
  CHECK((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0);
  CHECK(profile->registration.registered_host_pointer_alignment == 1);
  CHECK(profile->registration.minimum_alignment == 1);
  CHECK(capabilities->device_address.minimum_alignment == 1);
  CHECK((capabilities->guaranteed_flags &
         (AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS)) ==
        (AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS));
  const long page_value = sysconf(_SC_PAGESIZE);
  CHECK(page_value > 0);
  const size_t page = (size_t)page_value;
  CHECK((page & (page - 1)) == 0);
  void *allocation = NULL;
  CHECK(posix_memalign(&allocation, page, page * 3) == 0);
  unsigned char *host = (unsigned char *)allocation + 37;
  const uint64_t bytes = (uint64_t)page + 1;
  memset(allocation, 0, page * 3);

  amdf_memory_create_info_t create;
  INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
  create.memory_profile_ordinal = profile->ordinal;
  create.access_count = 1;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.byte_length = bytes;
  create.registered_host_pointer = host;
  create.registered_host_cacheability =
      profile->registration.registered_host_cacheability;
  create.accesses = access;
  amdf_memory_t *memory = NULL;
  CHECK(api->memory_create(scope, &create, &memory) == AMDF_STATUS_OK);

  amdf_memory_info_t info;
  INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_INFO);
  CHECK(api->memory_query_info(memory, &info) == AMDF_STATUS_OK);
  const uint64_t source_offset = (uint64_t)(uintptr_t)host & (page - 1);
  const uint64_t native_length =
      (source_offset + bytes + page - 1) & ~((uint64_t)page - 1);
  CHECK(info.memory_class == AMDF_MEMORY_CLASS_SYSTEM);
  CHECK(info.flags == AMDF_MEMORY_FLAG_HOST_VISIBLE);
  CHECK(info.source_byte_offset == source_offset);
  CHECK(info.byte_length == bytes);
  CHECK(info.native_allocation_byte_length == native_length);
  CHECK(info.native_allocation_granularity == page);
  CHECK(info.physical_backing_id.words[0] == 0);
  CHECK(info.physical_backing_id.words[1] == 0);

  amdf_memory_access_info_t actual;
  INIT(actual, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO);
  CHECK(api->memory_query_access_info(memory, 0, &actual) == AMDF_STATUS_OK);
  CHECK(actual.access == access->requirements.access);
  CHECK((actual.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0);
  CHECK((actual.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS) != 0);
  uint64_t gpu_address = 0;
  CHECK(api->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                  &gpu_address) == AMDF_STATUS_OK);
  const uint64_t common_bits = (uint64_t)(uintptr_t)host | gpu_address;
  const uint64_t common_alignment = common_bits & (~common_bits + 1);
  CHECK(info.alignment == common_alignment);

  amdf_memory_map_info_t map;
  INIT(map, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO);
  map.byte_length = bytes;
  map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  amdf_host_mapping_t *mapping = NULL;
  CHECK(api->memory_map(memory, &map, &mapping) == AMDF_STATUS_OK);
  amdf_host_mapping_info_t host_info;
  INIT(host_info, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO);
  CHECK(api->host_mapping_query_info(mapping, &host_info) == AMDF_STATUS_OK);
  CHECK(host_info.pointer == host);
  CHECK(host_info.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK);
  CHECK(host_info.flush.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
  CHECK(host_info.invalidate.kind == AMDF_CACHE_TRANSITION_KIND_RANGE);
  CHECK(api->host_mapping_cache_control(mapping,
                                        AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                        bytes) == AMDF_STATUS_OK);
  CHECK(api->host_mapping_destroy(mapping) == AMDF_STATUS_OK);

  create.minimum_alignment = 64;
  amdf_memory_t *sentinel = (amdf_memory_t *)&create;
  amdf_memory_t *rejected = sentinel;
  CHECK(api->memory_create(scope, &create, &rejected) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(rejected == sentinel);
  create.minimum_alignment = 1;
  create.registered_host_pointer = (void *)(UINTPTR_MAX - 7);
  CHECK(api->memory_create(scope, &create, &rejected) ==
        API_ERROR(INVALID_ARGUMENT));
  CHECK(rejected == sentinel);

  CHECK(api->memory_destroy(memory) == AMDF_STATUS_OK);
  host[0] = 0xa5;
  CHECK(host[0] == 0xa5);
  free(allocation);
  return 0;
}

static int check_gpu_memory(const amdf_api_t *api, amdf_instance_t *instance,
                            amdf_memory_scope_t *scope) {
  const void *extension = NULL;
  CHECK(api->query_extension(AMDF_EXTENSION_GPU, 1, 1, &extension) ==
        AMDF_STATUS_OK);
  const amdf_gpu_api_t *gpu = (const amdf_gpu_api_t *)extension;
  uint32_t count = 0;
  CHECK(api->endpoint_enumerate(instance, 0, NULL, &count) == AMDF_STATUS_OK);
  CHECK(count != 0);
  amdf_endpoint_summary_t *summaries =
      (amdf_endpoint_summary_t *)calloc(count, sizeof(*summaries));
  CHECK(summaries != NULL);
  CHECK(api->endpoint_enumerate(instance, count, summaries, &count) ==
        AMDF_STATUS_OK);
  unsigned tested = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (summaries[i].engine_kind != AMDF_ENGINE_KIND_GPU)
      continue;
    amdf_endpoint_t *endpoint = NULL;
    CHECK(api->endpoint_open(instance, &summaries[i].id, &endpoint) ==
          AMDF_STATUS_OK);
    amdf_gpu_device_create_info_t device_create;
    INIT(device_create, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO);
    amdf_device_t *device = NULL;
    amdf_status_t create_status =
        gpu->device_create(endpoint, &device_create, &device);
    if (create_status != AMDF_STATUS_OK)
      fprintf(stderr, "AMDF GPU device creation status: 0x%016" PRIx64 "\n",
              create_status);
    CHECK(create_status == AMDF_STATUS_OK);
    amdf_gpu_device_info_t device_info;
    INIT(device_info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO);
    CHECK(gpu->device_query_info(device, &device_info) == AMDF_STATUS_OK);
    CHECK((device_info.features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) !=
          0);
    amdf_memory_device_access_t access;
    memset(&access, 0, sizeof(access));
    access.device = device;
    access.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
    amdf_memory_scope_info_t scope_info;
    INIT(scope_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
    CHECK(api->memory_scope_query_info(scope, &scope_info) == AMDF_STATUS_OK);
    unsigned allocated = 0, registered = 0;
    for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
         ++ordinal) {
      amdf_memory_profile_t profile;
      amdf_memory_access_capabilities_t caps;
      INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
      INIT(caps, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES);
      amdf_status_t status = api->memory_scope_query_device_profile(
          scope, ordinal, 1, &access, &profile, &caps);
      if (status == API_ERROR(UNSUPPORTED))
        continue;
      CHECK(status == AMDF_STATUS_OK);
      if (profile.roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT) {
        CHECK(profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM);
        CHECK(profile.import.minimum_alignment != 0);
        CHECK(profile.import.maximum_byte_length != 0);
        CHECK(profile.external_memory_support_count == 1);
        CHECK(profile.external_memory_support[0].type ==
              AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD);
        CHECK((profile.external_memory_support[0].flags &
               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT) != 0);
        CHECK((profile.external_memory_support[0].flags &
               AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API) == 0);
      }
      CHECK((caps.guaranteed_access & access.requirements.access) ==
            caps.guaranteed_access);
      CHECK((caps.supported_access & access.requirements.access) ==
            access.requirements.access);
      if (profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        CHECK(check_gpu_registration(api, scope, &profile, &caps, &access) ==
              0);
        ++registered;
        continue;
      }
      if (!(profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE))
        continue;
      amdf_memory_create_info_t create;
      INIT(create, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO);
      create.memory_profile_ordinal = ordinal;
      create.byte_length = profile.allocation.native_byte_length_granularity;
      create.minimum_alignment = profile.allocation.minimum_alignment;
      create.access_count = 1;
      create.accesses = &access;
      create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      amdf_memory_t *memory = NULL;
      CHECK(api->memory_create(scope, &create, &memory) == AMDF_STATUS_OK);
      amdf_memory_access_info_t actual;
      INIT(actual, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO);
      CHECK(api->memory_query_access_info(memory, 0, &actual) ==
            AMDF_STATUS_OK);
      CHECK(actual.access == access.requirements.access);
      uint64_t address = 0;
      CHECK(api->memory_query_address(memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                      &address) == AMDF_STATUS_OK);
      CHECK(address >= caps.device_address.minimum_address &&
            address <= caps.device_address.maximum_address);
      CHECK(check_mapping(api, memory, NULL) == 0);
      CHECK(api->memory_destroy(memory) == AMDF_STATUS_OK);
      ++allocated;
    }
    CHECK(allocated != 0);
    CHECK(registered != 0);
    CHECK(api->device_destroy(device) == AMDF_STATUS_OK);
    CHECK(api->endpoint_close(endpoint) == AMDF_STATUS_OK);
    ++tested;
  }
  free(summaries);
  CHECK(tested != 0);
  printf("AMDF GPU one-access memory passed on %u endpoints\n", tested);
  return 0;
}

int main(int argc, char **argv) {
  const amdf_api_t *api = NULL;
  CHECK(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST,
                       &api) == AMDF_STATUS_OK);
  uint32_t first = AMDF_NATIVE_LIFETIME_PROCESS;
  uint32_t last = AMDF_NATIVE_LIFETIME_INSTANCE;
  if (argc == 2) {
    if (strcmp(argv[1], "process") == 0)
      last = AMDF_NATIVE_LIFETIME_PROCESS;
    else if (strcmp(argv[1], "instance") == 0)
      first = AMDF_NATIVE_LIFETIME_INSTANCE;
    else
      return 2;
  } else if (argc != 1) {
    return 2;
  }
  for (uint32_t lifetime = first; lifetime <= last; ++lifetime) {
    amdf_instance_create_info_t create;
    INIT(create, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    create.native_lifetime = lifetime;
    amdf_instance_t *instance = NULL;
    CHECK(api->instance_create(&create, &instance) == AMDF_STATUS_OK);
    amdf_memory_scope_t *scope = NULL;
    uint32_t count = 0;
    CHECK(api->instance_enumerate_memory_scopes(instance, 1, &scope, &count) ==
          AMDF_STATUS_OK);
    CHECK(count == 1 && scope != NULL);
    amdf_memory_scope_info_t info;
    INIT(info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO);
    CHECK(api->memory_scope_query_info(scope, &info) == AMDF_STATUS_OK);
    unsigned creates = 0, registrations = 0;
    for (uint32_t ordinal = 0; ordinal < info.memory_profile_count; ++ordinal) {
      amdf_memory_profile_t profile;
      INIT(profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE);
      amdf_status_t status = api->memory_scope_query_device_profile(
          scope, ordinal, 0, NULL, &profile, NULL);
      if (status == API_ERROR(UNSUPPORTED))
        continue;
      CHECK(status == AMDF_STATUS_OK);
      if (!(profile.roles & (AMDF_MEMORY_PROFILE_ROLE_CREATE |
                             AMDF_MEMORY_PROFILE_ROLE_REGISTER)))
        continue;
      CHECK(check_cpu_profile(api, scope, &profile) == 0);
      if (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE)
        ++creates;
      if (profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER)
        ++registrations;
    }
    CHECK(creates != 0 && registrations != 0);
    if (getenv("AMDF_REQUIRE_GPU") != NULL)
      CHECK(check_gpu_memory(api, instance, scope) == 0);
    CHECK(api->instance_destroy(instance) == AMDF_STATUS_OK);
  }
  puts("AMDF CPU CREATE/REGISTER, subrange mapping, cache control, and cleanup "
       "passed");
  return 0;
}
