// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amdf/amdf.h>
#include <stdint.h>
#include <stdio.h>

#ifndef FIXTURE_SUPPORTS_ABI2
#define FIXTURE_SUPPORTS_ABI2 0
#endif
#ifndef FIXTURE_SCOPE_COUNT
#define FIXTURE_SCOPE_COUNT 1
#endif

#if !FIXTURE_SUPPORTS_ABI2
static uintptr_t instance_storage;
#endif
static uintptr_t scope_storage[FIXTURE_SCOPE_COUNT];

static amdf_status_t AMDF_CALL fixture_create(
    const amdf_instance_create_info_t *info, amdf_instance_t **out_instance) {
  (void)info;
#if FIXTURE_SUPPORTS_ABI2
  (void)out_instance;
  puts("fixture: instance_create reached");
  return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
#else
  *out_instance = (amdf_instance_t *)&instance_storage;
  return AMDF_STATUS_OK;
#endif
}

static amdf_status_t AMDF_CALL fixture_destroy(amdf_instance_t *instance) {
  (void)instance;
  return AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL fixture_enumerate(amdf_instance_t *instance,
                                                 uint32_t capacity,
                                                 amdf_memory_scope_t **scopes,
                                                 uint32_t *out_count) {
  (void)instance;
  *out_count = FIXTURE_SCOPE_COUNT;
  for (uint32_t i = 0; i < capacity && i < FIXTURE_SCOPE_COUNT; ++i)
    scopes[i] = (amdf_memory_scope_t *)&scope_storage[i];
  return capacity < FIXTURE_SCOPE_COUNT
             ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
             : AMDF_STATUS_OK;
}

static amdf_status_t AMDF_CALL fixture_scope_info(
    amdf_memory_scope_t *scope, amdf_memory_scope_info_t *out_info) {
  (void)scope;
  (void)out_info;
  puts("fixture: memory_scope_query_info reached");
  return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
}

static const amdf_api_t fixture_v12 = {
    .abi_version = AMDF_ABI_VERSION_12,
    .structure_size = sizeof(amdf_api_t),
    .instance_create = fixture_create,
    .instance_destroy = fixture_destroy,
    .instance_enumerate_memory_scopes = fixture_enumerate,
    .memory_scope_query_info = fixture_scope_info,
};

#if FIXTURE_SUPPORTS_ABI2
static const amdf_api_t fixture_v2 = {
    .abi_version = AMDF_ABI_VERSION_2,
    .structure_size = sizeof(amdf_api_t),
};
#endif

AMDF_API amdf_status_t AMDF_CALL amdf_query_api(amdf_abi_version_t minimum,
                                                amdf_abi_version_t maximum,
                                                const amdf_api_t **out_api) {
  if (out_api == NULL || minimum > maximum)
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  if (minimum <= AMDF_ABI_VERSION_12 && maximum >= AMDF_ABI_VERSION_12) {
    *out_api = &fixture_v12;
    return AMDF_STATUS_OK;
  }
#if FIXTURE_SUPPORTS_ABI2
  if (minimum <= AMDF_ABI_VERSION_2 && maximum >= AMDF_ABI_VERSION_2) {
    *out_api = &fixture_v2;
    return AMDF_STATUS_OK;
  }
#endif
  return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
}
