// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The C ABI between the gpumetrics core and a metrics backend plugin. A plugin
// is a shared library exporting exactly one symbol:
//
//     const gpum_plugin_v1* gpum_plugin_entry_v1(void);
//
// returning a static, immortal vtable. Plain C, so plugins may be authored in
// C, C++, or Rust.
//
// Lifecycle: core dlopen()s the .so RTLD_NOW | RTLD_GLOBAL (RTLD_GLOBAL is
// mandatory so rocprofiler-sdk finds the rocprofiler plugin's
// rocprofiler_configure at hsa_init()), resolves gpum_plugin_entry_v1, checks
// abi_version, calls init(), enumerate() and list_metrics() once, then read()
// per batch, and shutdown() at teardown.

#ifndef GPUMETRICS_PLUGIN_ABI_H_
#define GPUMETRICS_PLUGIN_ABI_H_

#include "gpumetrics/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GPUM_PLUGIN_ABI_V1 1u

// Opaque per-plugin instance state, owned by the plugin.
typedef struct gpum_plugin_ctx gpum_plugin_ctx;

// A device the plugin serves. `identity` carries correlation keys; `name` is a
// human label. Plugin-owned; must stay valid until shutdown().
typedef struct {
  gpum_device_identity identity;
  char name[GPUM_STRING_MAX];
} gpum_device_desc;

// A metric the plugin can provide.
typedef struct {
  char key[GPUM_STRING_MAX];          // "temp.edge"
  char unit[32];                      // "C", "W", "MHz", "bytes", "%", ""
  char description[GPUM_STRING_MAX];
  gpum_value_type type;
  uint32_t scope;                     // OR of gpum_scope_flags
} gpum_metric_desc;

// One read request: which of the plugin's own devices, and which metric.
// `partition` selects a partition (-1 = whole device); plugins that don't model
// partitions treat it as whole device.
typedef struct {
  uint32_t plugin_local_index;
  int32_t partition;
  const char* key;
} gpum_read_req;

// One result, positionally matching the request array.
typedef struct {
  gpum_status status;
  gpum_value_type type;
  gpum_value value;
  uint64_t timestamp_ns;
} gpum_sample;

typedef struct gpum_plugin_v1 {
  uint32_t abi_version;  // must equal GPUM_PLUGIN_ABI_V1
  const char* name;      // stable short name, e.g. "amdsmi"

  // Initialize backend; allocate ctx.
  gpum_status (*init)(gpum_plugin_ctx** out_ctx);

  // Tear down backend and free ctx. Call once per successful init().
  void (*shutdown)(gpum_plugin_ctx* ctx);

  // Point *out_devices at a plugin-owned array of `*out_count` device
  // descriptors (valid until shutdown()).
  gpum_status (*enumerate)(gpum_plugin_ctx* ctx, const gpum_device_desc** out_devices,
                           uint32_t* out_count);

  // Point *out_metrics at a plugin-owned array of `*out_count` metric
  // descriptors (valid until shutdown()).
  gpum_status (*list_metrics)(gpum_plugin_ctx* ctx, const gpum_metric_desc** out_metrics,
                              uint32_t* out_count);

  // Read a batch into caller-allocated `out_samples` (n entries). Fill every
  // entry (GPUM_ERR_UNSUPPORTED etc. for ones it cannot serve) and return
  // GPUM_OK if the batch ran, even if individual samples failed. Non-OK means
  // the whole batch failed.
  gpum_status (*read)(gpum_plugin_ctx* ctx, const gpum_read_req* reqs, uint32_t n,
                      gpum_sample* out_samples);
} gpum_plugin_v1;

// The one exported symbol; name is versioned so the ABI can evolve.
typedef const gpum_plugin_v1* (*gpum_plugin_entry_v1_fn)(void);
#define GPUM_PLUGIN_ENTRY_SYMBOL "gpum_plugin_entry_v1"

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GPUMETRICS_PLUGIN_ABI_H_
