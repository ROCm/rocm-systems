/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hrr_regions.h - External region annotations.
 *
 * HRR interposes the HIP dispatch table, so it only observes memory that
 * crosses a HIP API. Two things are therefore invisible to it:
 *
 *   1. The per-object *blocks* a framework allocator carves out of a large
 *      hipMalloc segment by host pointer arithmetic (PyTorch's HIP caching
 *      allocator is the canonical case). Playback's range translation recovers
 *      the address of such a pointer but not its bounds, so an intra-segment
 *      overrun lands in a live neighbour at replay instead of faulting.
 *   2. Whole allocations that never reach a HIP entry point at all - a library
 *      calling hsa_amd_memory_pool_allocate directly, a custom VMM pool, or
 *      memory imported from another process. Kernel arguments pointing into
 *      those resolve in no map.
 *
 * Both are the same missing fact: a device VA range that existed at capture
 * time and that HRR could not observe. One record type (hrr_region_rec)
 * describes it, and `kind` selects which of the two cases it is.
 *
 * TRANSPORT
 * ---------
 * Records travel as ordinary HRR events - an 8-byte hrr_file_header followed
 * by hrr_event_header + payload records - so the existing reader parses them,
 * including its torn-tail recovery. Two transports carry the same payload:
 *
 *   sidecar   <archive>/pid-<pid>/regions/<name>.hrrr, appended by a producer
 *             outside libamdhip64. The file header carries HRR_REGION_MAGIC so
 *             it cannot be confused with events.bin. This is the only transport
 *             implemented today; it needs no exported symbol and no capture-side
 *             code, and a producer locates the directory from
 *             $HIP_HRR_CAPTURE_OUTPUT and its own pid.
 *   in-tree   a future producer inside libamdhip64 (e.g. an HSA allocation
 *             hook) can write the identical payload through write_event_raw()
 *             into events.bin with no format change.
 *
 * The format, not an ABI, is the contract: see hrr/producers/README.md for the
 * producer-side specification.
 */

#include <stdint.h>
#include <string.h>

#include "hrr_api_args.h" /* hrr_file_header, hrr_event_header */

/* Sidecar file-header magic. Occupies hrr_file_header.magic in place of
 * HRR_MAGIC so a region stream and an event stream can never be mistaken for
 * one another. */
#define HRR_REGION_MAGIC ((uint32_t)0x52525248u) /* "HRRR" */

/* Region-format version, carried in hrr_file_header.version. Deliberately
 * independent of HRR_VERSION: the region payload does not change when the
 * event wire format does, so a producer pinned to version 1 keeps working
 * across archive-format bumps. */
#define HRR_REGION_VERSION ((uint16_t)1u)

/* hrr_event_header.event_type sentinel, outside the hrr_api_id_t range (like
 * HRR_EOF_MARKER at 0xFFFF). Reserved here even for the sidecar transport so
 * the in-tree transport can use it without a second allocation. */
#define HRR_REGION_EVENT ((uint16_t)0xFFFDu)

/* hrr_region_rec.op */
#define HRR_REGION_ADD ((uint8_t)0u) /* region becomes live at mono_ns  */
#define HRR_REGION_DEL ((uint8_t)1u) /* region stops being live at mono_ns */

/* hrr_region_rec.kind */
/* A sub-range of memory playback already knows about, typically one framework
 * allocator block inside a captured hipMalloc segment. Playback records the
 * bounds and does NOT allocate: the enclosing segment is already contiguous and
 * the layout must stay byte-identical to the captured run. */
#define HRR_REGION_BLOCK ((uint8_t)0u)
/* A whole allocator segment. A producer declares these uniformly and does not
 * need to know what HRR observed. Playback records the bounds, and allocates
 * nothing unless a pointer into the segment turns out to resolve in no map - at
 * which point HIP demonstrably was bypassed for it, and playback materialises
 * the segment so the pointer translates. Contents are unknown to the archive
 * and get the replay fill byte.
 * The bounds do the other half of the work: a pointer inside a segment that
 * belongs to no live block is an intra-segment out-of-bounds or stale pointer,
 * which a contiguous replay segment would otherwise hide. */
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
    /* CLOCK_MONOTONIC nanoseconds, the same clock as
     * hrr_event_header.timestamp_ns (amd::Os::timeNanos). Zero is special: it
     * means "live since before this stream began", which is how a producer
     * declares the regions that already existed when it started observing.
     * That makes baseline seeding an ordinary run of ADD records rather than a
     * separate mechanism. */
    int64_t  mono_ns;
} hrr_region_rec;

/* Payload prefix of an HRR_REGION_EVENT record: header, then n records.
 * A batch of one is legal, so a synchronous producer needs no buffering. */
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

/* Fill the fixed part of a batch record. The caller appends `n` records and
 * writes the whole thing with a single write(), which is what makes a torn
 * tail the only possible damage after a crash. sequence_id is unused by the
 * sidecar transport (records are ordered by mono_ns) and may be zero. */
static inline hrr_region_batch hrr_make_region_batch(uint32_t n, uint64_t timestamp_ns) {
    hrr_region_batch b;
    memset(&b, 0, sizeof(b));
    b.hdr.event_type     = HRR_REGION_EVENT;
    b.hdr.timestamp_ns   = timestamp_ns;
    b.hdr.payload_length = hrr_region_batch_bytes(n);
    b.n                  = n;
    return b;
}
