// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AQLMON_AQLMON_H
#define AQLMON_AQLMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AQLMON_MAGIC UINT64_C(0x4145514c4d4f4e31)

// Shared-memory record-stream format version.
// Bump this when aqlmon_shm_header_t or aqlmon_record_t layout changes incompatibly.
#define AQLMON_SHM_VERSION UINT32_C(3)
#define AQLMON_VERSION AQLMON_SHM_VERSION
#define AQLMON_MAX_SHM_NAME 64
#define AQLMON_MAX_KERNEL_NAME 64
#define AQLMON_MAX_CODE_OBJECT_URI 192
#define AQLMON_PACKET_BYTES 64

enum aqlmon_record_kind_t {
  AQLMON_RECORD_PACKET = 1,
  AQLMON_RECORD_DOORBELL = 2,
  AQLMON_RECORD_DROP = 3,
  AQLMON_RECORD_CODE_OBJECT_LIVE = 4,
  AQLMON_RECORD_CODE_OBJECT_DEAD = 5,
  AQLMON_RECORD_DISPATCH_COMPLETE = 6
};

enum aqlmon_record_flags_t {
  AQLMON_FLAG_PRODUCER_VALID = 1u << 0,
  AQLMON_FLAG_KERNEL_NAME_VALID = 1u << 1,
  AQLMON_FLAG_CODE_OBJECT_URI_VALID = 1u << 2,
  AQLMON_FLAG_CODE_OBJECT_LOAD_RANGE_VALID = 1u << 3,
  AQLMON_FLAG_SIGNAL_TIMESTAMPS_VALID = 1u << 4,
  AQLMON_FLAG_INJECTED_SIGNAL = 1u << 5
};

typedef struct aqlmon_shm_header_s {
  uint64_t magic;
  uint32_t version;
  uint32_t header_size;
  uint32_t record_size;
  uint32_t capacity;
  uint64_t write_seq;
  uint64_t dropped_records;
  uint64_t dropped_packets;
  char shm_name[AQLMON_MAX_SHM_NAME];
} aqlmon_shm_header_t;

typedef struct aqlmon_record_s {
  uint32_t size;
  uint16_t kind;
  uint16_t packet_type;
  uint32_t pid;
  uint32_t tid;
  uint64_t seq;
  uint64_t monotonic_ns;
  uint64_t queue_id;
  uint64_t queue_ptr;
  uint64_t queue_base;
  uint64_t queue_size;
  uint64_t dispatch_id;
  uint64_t write_index;
  uint64_t observed_wptr;
  uint64_t observed_rptr;
  uint64_t packet_header;
  uint64_t kernel_object;
  uint64_t code_object_handle;
  uint64_t executable_handle;
  uint64_t agent_handle;
  uint64_t code_object_load_base;
  uint64_t code_object_load_size;
  uint64_t completion_signal;
  uint64_t signal_value;
  uint64_t dispatch_start_ns;
  uint64_t dispatch_end_ns;
  uint64_t doorbell_value;
  uint64_t flags;
  char kernel_name[AQLMON_MAX_KERNEL_NAME];
  char code_object_uri[AQLMON_MAX_CODE_OBJECT_URI];
  uint8_t packet_bytes[AQLMON_PACKET_BYTES];
} aqlmon_record_t;

#ifdef __cplusplus
}
#endif

#endif
