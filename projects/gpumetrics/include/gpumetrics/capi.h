// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Flat C API over the gpumetrics Collector: the FFI surface for Rust and other
// bindings. C types only. All strings are UTF-8, NUL-terminated, copied into
// caller buffers.

#ifndef GPUMETRICS_CAPI_H_
#define GPUMETRICS_CAPI_H_

#include <stddef.h>
#include <stdint.h>

#include "gpumetrics/plugin_abi.h"  // for gpum_sample / gpum_value
#include "gpumetrics/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpum_collector gpum_collector;

// Collector options. Any pointer may be NULL / count 0 for defaults. Arrays are
// borrowed for the call only.
typedef struct {
  const char* const* plugin_paths;  // extra search dirs
  uint32_t plugin_paths_count;
  const char* const* plugins;  // restrict to these plugin names/files
  uint32_t plugins_count;
  const char* const* provider_priority;  // conflict resolution order
  uint32_t provider_priority_count;
  uint32_t read_timeout_ms;  // 0 => default
} gpum_collector_options;

// On failure returns NULL and (if non-NULL) writes *status.
gpum_collector* gpum_collector_create(const gpum_collector_options* opts, gpum_status* status);
void gpum_collector_destroy(gpum_collector* c);

// --- topology -------------------------------------------------------------
uint32_t gpum_collector_gpu_count(const gpum_collector* c);
uint32_t gpum_collector_socket_count(const gpum_collector* c);

// Canonical GPU entity id at ordinal `gpu`. NOT_FOUND if out of range.
gpum_status gpum_collector_gpu_entity(const gpum_collector* c, uint32_t gpu, gpum_entity_id* out);

// Describe a GPU: name into name_buf, plus *identity and *partition_count. Any
// out-param may be NULL.
gpum_status gpum_collector_gpu_info(const gpum_collector* c, uint32_t gpu, char* name_buf,
                                    size_t name_cap, gpum_device_identity* identity,
                                    uint32_t* partition_count);

// Partition indices of a GPU into out_partitions (capacity cap); *out_count
// receives the true count (may exceed cap).
gpum_status gpum_collector_gpu_partitions(const gpum_collector* c, uint32_t gpu,
                                          int32_t* out_partitions, uint32_t cap,
                                          uint32_t* out_count);

// --- metric registry ------------------------------------------------------
uint32_t gpum_collector_metric_count(const gpum_collector* c);

// Describe the metric at index `i`. Any out may be NULL.
gpum_status gpum_collector_metric_at(const gpum_collector* c, uint32_t i, char* key_buf,
                                     size_t key_cap, char* unit_buf, size_t unit_cap,
                                     char* provider_buf, size_t provider_cap, gpum_value_type* type,
                                     uint32_t* scope);

// --- selectors ------------------------------------------------------------
// Resolve a selector ("gpu:0", "g0.1", "socket:1", "bdf:...", "uuid:...") into
// an entity id. NOT_FOUND if unresolved.
gpum_status gpum_collector_resolve(const gpum_collector* c, const char* selector,
                                   gpum_entity_id* out);

// --- reads ----------------------------------------------------------------
// Read one metric for one entity.
gpum_status gpum_collector_read(gpum_collector* c, const gpum_entity_id* entity, const char* key,
                                gpum_sample* out_sample);

// Batch read: `n` keys for one entity; out_samples has capacity n.
gpum_status gpum_collector_read_batch(gpum_collector* c, const gpum_entity_id* entity,
                                      const char* const* keys, uint32_t n,
                                      gpum_sample* out_samples);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GPUMETRICS_CAPI_H_
