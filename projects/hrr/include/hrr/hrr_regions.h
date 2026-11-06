/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once


#include <stdint.h>
#include <string.h>

#include "hrr_api_args.h" /* hrr_file_header, hrr_event_header */

#define HRR_REGION_MAGIC ((uint32_t)0x52525248u) /* "HRRR" */

#define HRR_REGION_VERSION ((uint16_t)1u)

#define HRR_REGION_EVENT ((uint16_t)0xFFFDu)

/* hrr_region_rec.op */
#define HRR_REGION_ADD ((uint8_t)0u) /* region becomes live at mono_ns  */
#define HRR_REGION_DEL ((uint8_t)1u) /* region stops being live at mono_ns */

/* hrr_region_rec.kind */
#define HRR_REGION_BLOCK ((uint8_t)0u)
#define HRR_REGION_SEGMENT ((uint8_t)1u)

#pragma pack(push, 1)

/* One annotation. Exactly 32 bytes; all fields little-endian. */
typedef struct {
    uint8_t  op;      /* HRR_REGION_ADD / HRR_REGION_DEL                     */
    uint8_t  kind;    /* HRR_REGION_BLOCK / HRR_REGION_SEGMENT               */
    uint8_t  device;  /* device ordinal the region belongs to                */
    uint8_t  flags;   /* reserved; zero                                      */
    uint32_t tag;     /* producer-defined (pool id, stream id, ...); zero ok */
    uint64_t base;    /* capture-time device VA of the region base           */
    uint64_t size;    /* region size in bytes                                */
    int64_t  mono_ns;
} hrr_region_rec;

typedef struct {
    hrr_event_header hdr;   /* event_type = HRR_REGION_EVENT               */
    uint32_t         n;     /* number of hrr_region_rec that follow        */
    uint32_t         flags; /* reserved; zero                              */
    /* hrr_region_rec recs[n] follows immediately. */
} hrr_region_batch;

#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(hrr_region_rec) == 32, "hrr_region_rec must be 32 bytes");
static_assert(sizeof(hrr_region_batch) == 40, "hrr_region_batch must be 40 bytes");
#endif

/* Total on-the-wire size of a batch carrying n records. */
static inline uint32_t hrr_region_batch_bytes(uint32_t n) {
    return (uint32_t)sizeof(hrr_region_batch) + n * (uint32_t)sizeof(hrr_region_rec);
}

/* Build the 8-byte header a sidecar region stream starts with. */
static inline hrr_file_header hrr_make_region_file_header(void) {
    hrr_file_header fh;
    memset(&fh, 0, sizeof(fh));
    fh.magic   = HRR_REGION_MAGIC;
    fh.version = HRR_REGION_VERSION;
    return fh;
}

static inline hrr_region_batch hrr_make_region_batch(uint32_t n, uint64_t timestamp_ns) {
    hrr_region_batch b;
    memset(&b, 0, sizeof(b));
    b.hdr.event_type     = HRR_REGION_EVENT;
    b.hdr.timestamp_ns   = timestamp_ns;
    b.hdr.payload_length = hrr_region_batch_bytes(n);
    b.n                  = n;
    return b;
}
