// SPDX-License-Identifier: MIT

// Adapted from libamdf/tests/abi/native_c_smoke.c; the subject is loaded, not
// linked.
#define _GNU_SOURCE
#include <amdf/amdf.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "AMDF contract failed at line %d: %s\n", __LINE__,       \
              #condition);                                                     \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc != 2 || argv[1][0] != '/') {
    fprintf(stderr, "usage: %s /absolute/path/to/libamdf.so\n", argv[0]);
    return 1;
  }
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == NULL) {
    fprintf(stderr, "Cannot load selected runtime %s: %s\n", argv[1],
            dlerror());
    return 1;
  }
  int result = 1;
  const amdf_api_t *api = NULL;
  amdf_instance_t *instance = NULL;
  amdf_query_api_fn_t query = NULL;
  dlerror();
  void *symbol = dlsym(library, "amdf_query_api");
  const char *error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "Missing amdf_query_api: %s\n", error);
    goto cleanup;
  }
  // POSIX dlsym represents callable symbols as void pointers.
  _Static_assert(sizeof(query) == sizeof(symbol),
                 "POSIX function pointer size");
  memcpy(&query, &symbol, sizeof(query));
  CHECK(query != NULL);
  Dl_info loaded;
  CHECK(dladdr(symbol, &loaded) != 0);
  struct stat selected_status, loaded_status;
  CHECK(stat(argv[1], &selected_status) == 0);
  CHECK(stat(loaded.dli_fname, &loaded_status) == 0);
  if (selected_status.st_dev != loaded_status.st_dev ||
      selected_status.st_ino != loaded_status.st_ino) {
    fprintf(stderr, "amdf_query_api resolved outside selected image: %s\n",
            loaded.dli_fname);
    goto cleanup;
  }
  printf("loaded_runtime=%s\n", loaded.dli_fname);

  CHECK(query(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api) ==
        AMDF_STATUS_OK);
  CHECK(api != NULL && api->abi_version == AMDF_ABI_VERSION_3);
  CHECK(api->structure_size >= sizeof(*api));
  CHECK(api->instance_create && api->instance_destroy &&
        api->instance_enumerate_memory_scopes && api->memory_scope_query_info);
  const amdf_api_t *again = api;
  CHECK(query(2, 2, &again) ==
        amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH));
  CHECK(again == api);
  CHECK(query(2, 1, &again) ==
        amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  CHECK(again == api);

  for (uint32_t lifetime = AMDF_NATIVE_LIFETIME_PROCESS;
       lifetime <= AMDF_NATIVE_LIFETIME_INSTANCE; ++lifetime) {
    amdf_instance_create_info_t create = {0};
    create.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.native_lifetime = lifetime;
    CHECK(api->instance_create(&create, &instance) == AMDF_STATUS_OK);
    CHECK(instance != NULL);
    uint32_t count = 0;
    CHECK(api->instance_enumerate_memory_scopes(instance, 0, NULL, &count) ==
          amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    CHECK(count != 0);
    amdf_memory_scope_t *scope = NULL;
    CHECK(api->instance_enumerate_memory_scopes(instance, 1, &scope, &count) ==
          AMDF_STATUS_OK);
    amdf_memory_scope_info_t info = {0};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
    info.structure_size = sizeof(info);
    CHECK(api->memory_scope_query_info(scope, &info) == AMDF_STATUS_OK);
    CHECK(info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM);
    amdf_status_t status = api->instance_destroy(instance);
    if (status != AMDF_STATUS_OK) {
      // A failed destruction preserves an owner only for recovery. Do not
      // release its provider handle or retry implicitly during test cleanup.
      fprintf(stderr, "AMDF instance destruction failed: %" PRIu64 "\n",
              status);
      return 1;
    }
    instance = NULL;
  }
  result = 0;

cleanup:
  if (instance != NULL && api != NULL) {
    if (api->instance_destroy(instance) != AMDF_STATUS_OK) {
      fprintf(stderr,
              "AMDF cleanup failed; retaining provider until process exit\n");
      return 1;
    }
  }
  // NODELETE providers may stay mapped. Each invocation is a fresh process.
  if (dlclose(library) != 0) {
    fprintf(stderr, "AMDF dlclose failed: %s\n", dlerror());
    result = 1;
  }
  if (result == 0)
    puts("PASS runtime-api.amdf.host-contract");
  return result;
}
