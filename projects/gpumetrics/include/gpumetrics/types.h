// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Fundamental value/status/identity types shared by the C API, C++ core, and
// plugin ABI. Dependency-free and C-compatible so Rust can bind directly.

#ifndef GPUMETRICS_TYPES_H_
#define GPUMETRICS_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------
// A single, non-overloaded status space. "Unsupported" is a first-class status,
// never a magic value inside a metric.
typedef enum {
  GPUM_OK = 0,
  GPUM_ERR_INVALID_ARG = 1,
  GPUM_ERR_NOT_FOUND = 2,      // unknown metric key / entity
  GPUM_ERR_UNSUPPORTED = 3,    // valid request, backend/HW cannot provide it
  GPUM_ERR_NOT_INITIALIZED = 4,
  GPUM_ERR_ALREADY_EXISTS = 5,
  GPUM_ERR_NO_DATA = 6,        // transiently unavailable
  GPUM_ERR_TIMEOUT = 7,
  GPUM_ERR_BACKEND = 8,        // backend/driver call failed
  GPUM_ERR_ABI = 9,            // plugin ABI mismatch / bad plugin
  GPUM_ERR_INTERNAL = 10,
} gpum_status;

const char* gpum_status_string(gpum_status s);

// ---------------------------------------------------------------------------
// Value typing
// ---------------------------------------------------------------------------
typedef enum {
  GPUM_TYPE_U64 = 0,
  GPUM_TYPE_I64 = 1,
  GPUM_TYPE_F64 = 2,
  GPUM_TYPE_STRING = 3,
} gpum_value_type;

#define GPUM_STRING_MAX 256u

// A typed value. GPUM_TYPE_STRING uses `str` (NUL-terminated); otherwise the
// matching scalar field is valid.
typedef struct {
  gpum_value_type type;
  union {
    uint64_t u64;
    int64_t i64;
    double f64;
  };
  char str[GPUM_STRING_MAX];
} gpum_value;

// ---------------------------------------------------------------------------
// Entity model: GPU / partition / socket
// ---------------------------------------------------------------------------
typedef enum {
  GPUM_ENTITY_SOCKET = 0,
  GPUM_ENTITY_GPU = 1,           // whole physical GPU
  GPUM_ENTITY_GPU_PARTITION = 2, // one partition instance of a GPU
} gpum_entity_kind;

// Scope bitmask: which entity kinds a metric is meaningful for.
typedef enum {
  GPUM_SCOPE_SOCKET = 1 << 0,
  GPUM_SCOPE_GPU = 1 << 1,
  GPUM_SCOPE_PARTITION = 1 << 2,
} gpum_scope_flags;

// Canonical, core-assigned identity of an addressable entity.
typedef struct {
  gpum_entity_kind kind;
  uint32_t socket;      // socket ordinal
  uint32_t gpu;         // canonical physical-GPU ordinal (valid for GPU/PARTITION)
  int32_t partition;    // -1 for whole GPU/socket, else partition index
} gpum_entity_id;

// ---------------------------------------------------------------------------
// Device identity: raw keys a plugin reports so the core can correlate the same
// physical GPU across differently ordered backends. A 0 / all-zero field means
// the backend did not provide it. Plugins report the truth (full BDF, real
// partition_index); the core does the physical grouping.
//
// On CPX-partitioned GPUs (e.g. MI350X) most keys fragment per partition: each
// partition is its own PCIe function with a distinct BDF, KFD node id, and UUID.
// The per-physical-GPU keys are the masked BDF (function stripped) and oam_id,
// so the core groups on those.
// ---------------------------------------------------------------------------
typedef struct {
  uint64_t bdf;               // packed PCIe: domain(63-16)|bus(15-8)|dev(7-3)|func(2-0)
  uint32_t oam_id;            // OAM physical slot id (per-board); GPUM_ID_UNKNOWN if absent
  uint32_t kfd_node_id;       // KFD topology node id (per partition on CPX)
  uint32_t drm_render_minor;  // /dev/dri/renderD<minor>
  uint8_t uuid[16];           // GPU UUID (per partition on CPX; all-zero if unknown)
  int32_t partition_index;    // -1 for a whole GPU, else the partition index
  uint32_t plugin_local_index; // index the plugin uses to address this device
  uint32_t socket_id;         // backend socket hint; GPUM_SOCKET_UNKNOWN if none
} gpum_device_identity;

#define GPUM_SOCKET_UNKNOWN 0xFFFFFFFFu
#define GPUM_ID_UNKNOWN 0xFFFFFFFFu

// Pack a PCIe BDF into the representation above.
static inline uint64_t gpum_bdf_pack(uint32_t domain, uint8_t bus, uint8_t dev, uint8_t func) {
  return ((uint64_t)domain << 16) | ((uint64_t)(bus & 0xff) << 8) |
         ((uint64_t)(dev & 0x1f) << 3) | (uint64_t)(func & 0x7);
}

// BDF with the PCIe function stripped: the per-physical-GPU key. All partitions
// of one GPU share it; distinct GPUs differ in bus/device.
static inline uint64_t gpum_bdf_masked(uint64_t bdf) { return bdf & ~(uint64_t)0x7; }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GPUMETRICS_TYPES_H_
