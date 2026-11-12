/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — shared value types.

#ifndef ABCE_TYPES_H_
#define ABCE_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "abce_config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

#define ABCE_KI 1024u

// One choice per possible engine (matches ABCE_MAX_ENGINES in abce_host.h) so
// the fan-out balancer can see and spread across every registered ring, not a
// truncated subset.
#define ABCE_MAX_ENGINE_CHOICES 16u

//===----------------------------------------------------------------------===//
// abce_isa_version_t / abce_packet_caps_t
//===----------------------------------------------------------------------===//

// Hardware instruction-set version used to select SDMA packet layouts.
typedef struct abce_isa_version_t {
  uint32_t major;
  uint32_t minor;
  uint32_t stepping;
} abce_isa_version_t;

typedef enum abce_sdma_ip_version_t {
  ABCE_SDMA_IP_VERSION_OSS4 = 0,  // Legacy packets, no explicit GCR.
  ABCE_SDMA_IP_VERSION_OSS5 = 1,  // Legacy packets with GCR invalidate/writeback.
  ABCE_SDMA_IP_VERSION_OSS7 = 2,  // Scope-bearing packets, no explicit GCR.
} abce_sdma_ip_version_t;

typedef struct abce_packet_caps_t {
  abce_sdma_ip_version_t version;
  bool use_gcr;
  bool scope_fields;
  bool gfx125plus;
  size_t max_linear_copy_size;
} abce_packet_caps_t;

// Derives inherent packet/coherency capabilities from the gfx IP.
static inline abce_packet_caps_t abce_detect_packet_caps(abce_isa_version_t isa) {
  abce_packet_caps_t caps;
  caps.version = ABCE_SDMA_IP_VERSION_OSS4;
  caps.use_gcr = false;
  caps.scope_fields = false;
  caps.gfx125plus = false;
  caps.max_linear_copy_size = 0x3fffe0;

  if (isa.major == 9) {
    caps.max_linear_copy_size =
        (isa.minor >= 4 || (isa.minor == 0 && isa.stepping == 10)) ? 0x3fffffff : 0x3fffff;
    return caps;
  }
  if (isa.major == 10) {
    caps.version = ABCE_SDMA_IP_VERSION_OSS5;
    caps.use_gcr = true;
    caps.max_linear_copy_size = isa.minor < 3 ? 0x3fffff : 0x3fffffff;
    return caps;
  }
  if ((isa.major == 11 || isa.major == 12) && isa.minor < 5) {
    caps.version = ABCE_SDMA_IP_VERSION_OSS5;
    caps.use_gcr = true;
    caps.max_linear_copy_size = 0x3fffffff;
    return caps;
  }
  if ((isa.major == 11 || isa.major == 12) && isa.minor >= 5) {
    caps.version = ABCE_SDMA_IP_VERSION_OSS7;
    caps.scope_fields = true;
    caps.gfx125plus = (isa.major == 12 && isa.minor >= 5);
    caps.max_linear_copy_size = 0x3fffffff;
    return caps;
  }
  return caps;
}

//===----------------------------------------------------------------------===//
// abce_platform_caps_t
//===----------------------------------------------------------------------===//

// Capabilities and policy decisions supplied by the surrounding runtime.
//
// These values cannot be inferred solely from the gfx IP. For example, the same
// IP may be exposed with or without HDP flush support.
typedef struct abce_platform_caps_t {
  bool device_atomic_support;
  bool emit_hdp_flush;
  bool driver_manages_gcr;
} abce_platform_caps_t;

// ROCr-compatible IP defaults. A runtime should override these when link
// topology says otherwise (for example an xGMI host link disables HDP flush).
static inline abce_platform_caps_t abce_detect_default_platform_caps(abce_isa_version_t isa) {
  abce_platform_caps_t caps;
  caps.device_atomic_support = !(isa.major == 7 && isa.minor == 0 && isa.stepping == 1);
  caps.emit_hdp_flush = isa.major >= 9 && !(isa.major == 10 && isa.minor == 1);
  caps.driver_manages_gcr = false;
  return caps;
}

//===----------------------------------------------------------------------===//
// abce_builder_config_t
//===----------------------------------------------------------------------===//

typedef struct abce_builder_config_t {
  bool use_copy_size_override;
  size_t max_linear_copy_size;
  size_t max_fill_size;
} abce_builder_config_t;

// Zero-initializing this struct is NOT the default configuration:
// use_copy_size_override defaults to true. Always initialize through here.
static inline void abce_builder_config_initialize(abce_builder_config_t* out_config) {
  out_config->use_copy_size_override = true;
  out_config->max_linear_copy_size = 0;
  out_config->max_fill_size = 0;
}

//===----------------------------------------------------------------------===//
// Placement modes
//===----------------------------------------------------------------------===//

typedef enum abce_linear_batch_mode_t {
  ABCE_LINEAR_BATCH_MODE_AUTOMATIC = 0,
  ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK = 1,
  ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT = 2,
} abce_linear_batch_mode_t;

typedef enum abce_multicast_mode_t {
  ABCE_MULTICAST_MODE_AUTOMATIC = 0,
  ABCE_MULTICAST_MODE_FORCE_MULTICAST = 1,
  ABCE_MULTICAST_MODE_FORCE_FAN_OUT = 2,
} abce_multicast_mode_t;

//===----------------------------------------------------------------------===//
// Geometry
//===----------------------------------------------------------------------===//

typedef struct abce_dim3_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} abce_dim3_t;

typedef struct abce_pitched_ptr_t {
  void* base;
  size_t pitch;
  size_t slice;
} abce_pitched_ptr_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_TYPES_H_
