////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef AMD_HSA_QUEUE_H
#define AMD_HSA_QUEUE_H

#include "amd_hsa_common.h"
#include "hsa.h"

// AMD Queue Properties.
typedef uint32_t amd_queue_properties32_t;
enum amd_queue_properties_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER, 0, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_IS_PTR64, 1, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS, 2, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, 3, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE, 4, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_PROPERTIES_RESERVED1, 5, 27)
};

// AMD Queue.
#define AMD_QUEUE_ALIGN_BYTES 64
#define AMD_QUEUE_ALIGN __ALIGNED__(AMD_QUEUE_ALIGN_BYTES)

// AMD Queue Capabilities.
typedef uint32_t amd_queue_capabilities32_t;
enum amd_queue_capabilities_t {
  /* This version of CP FW supports dual-scratch and async-reclaim */
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM, 0, 1),

  /*
   * This version of ROCr supports async-reclaim and CP FW may access the
   * V2 fields.
   */
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM, 1, 1),
};

/* This is the original amd_queue_t structure. The definition is only kept
 * for reference purposes. This structure should not be used. */
typedef struct AMD_QUEUE_ALIGN amd_queue_s {
  hsa_queue_t hsa_queue;
  uint32_t caps;
  uint32_t reserved1[3];
  volatile uint64_t write_dispatch_id;
  uint32_t group_segment_aperture_base_hi;
  uint32_t private_segment_aperture_base_hi;
  uint32_t max_cu_id;
  uint32_t max_wave_id;
  volatile uint64_t max_legacy_doorbell_dispatch_id_plus_1;
  volatile uint32_t legacy_doorbell_lock;
  uint32_t reserved2[9];
  volatile uint64_t read_dispatch_id;
  uint32_t read_dispatch_id_field_base_byte_offset;
  uint32_t compute_tmpring_size;
  uint32_t scratch_resource_descriptor[4];
  uint64_t scratch_backing_memory_location;
  uint32_t reserved3[2];
  uint32_t scratch_wave64_lane_byte_size;
  amd_queue_properties32_t queue_properties;
  uint32_t reserved4[2];
  hsa_signal_t queue_inactive_signal;
  uint32_t reserved5[14];
} amd_queue_t;

/*
 * AMD_QUEUE Version 2
 * amd_queue_v2_t is backwards compatible with amd_queue_t structure and can
 * be used with previous versions of CP FW. The added fields tagged as V2 are
 * ignored when running previous versions of CP FW.
 * CP FW will not try to access elements beyond the original 64-bytes
 * (sizeof(amd_queue_t)) unless the AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM bit is set.
 */

#define MAX_NUM_XCC 128
typedef struct scratch_last_used_index_xcc_s {
  volatile uint64_t main;
  volatile uint64_t alt;
} scratch_last_used_index_xcc_t;

typedef struct AMD_QUEUE_ALIGN amd_queue_v2_s {
  hsa_queue_t hsa_queue;
  uint32_t caps;
  uint32_t reserved1[3];
  volatile uint64_t write_dispatch_id;
  uint32_t group_segment_aperture_base_hi;
  uint32_t private_segment_aperture_base_hi;
  uint32_t max_cu_id;
  uint32_t max_wave_id;
  volatile uint64_t max_legacy_doorbell_dispatch_id_plus_1;
  volatile uint32_t legacy_doorbell_lock;
  uint32_t reserved2[9];
  volatile uint64_t read_dispatch_id;
  uint32_t read_dispatch_id_field_base_byte_offset;
  uint32_t compute_tmpring_size;
  uint32_t scratch_resource_descriptor[4];
  uint64_t scratch_backing_memory_location;
  uint64_t scratch_backing_memory_byte_size;
  uint32_t scratch_wave64_lane_byte_size;
  amd_queue_properties32_t queue_properties;
  volatile uint64_t scratch_max_use_index;       /* V2 */
  hsa_signal_t queue_inactive_signal;
  volatile uint64_t alt_scratch_max_use_index;  /* V2 */
  uint32_t alt_scratch_resource_descriptor[4];   /* V2 */
  uint64_t alt_scratch_backing_memory_location;  /* V2 */
  uint32_t alt_scratch_dispatch_limit_x;         /* V2 */
  uint32_t alt_scratch_dispatch_limit_y;         /* V2 */
  uint32_t alt_scratch_dispatch_limit_z;         /* V2 */
  uint32_t alt_scratch_wave64_lane_byte_size;    /* V2 */
  uint32_t alt_compute_tmpring_size;             /* V2 */
  uint32_t reserved5;

  scratch_last_used_index_xcc_t scratch_last_used_index[MAX_NUM_XCC];
} amd_queue_v2_t;

#endif // AMD_HSA_QUEUE_H
