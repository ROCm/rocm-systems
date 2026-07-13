/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hrr_suballoc.h — Caching-allocator sub-allocation map capture/replay.
 *
 * HRR hooks at the HIP API level, so it only ever sees the large `hipMalloc`
 * *segments* that a framework allocator (e.g. PyTorch's HIP caching allocator)
 * carves up into many per-tensor *blocks*. Without the block layout the
 * replayer cannot tell where one tensor ends and the next begins inside a
 * shared segment, so an out-of-bounds intra-segment write is invisible at
 * replay (it lands in a neighbouring live tensor instead of faulting).
 *
 * To recover that information the capturing process pushes the allocator's
 * layout into HRR via the exported C API (see hip_capture.cpp)
 *   int  hipHrrCaptureActive(void);
 *   void hipHrrCaptureSubAllocSnapshot(const void* blob, uint64_t len);
 *   void hipHrrCaptureSubAllocTimeline(const void* blob, uint64_t len);
 * as two blob kinds: a coarse point-in-time segment->block SNAPSHOT and a
 * precise alloc/free TIMELINE (below). Each blob is stored content-addressed
 * through the normal crash-safe writer (write_blob = temp-file + rename) and a
 * small referencing event is written through write_event_raw — so the existing
 * checkpoint/emergency_finalize crash durability covers the sub-allocation map
 * for free, with no separate signal handler. At replay the timeline
 * reconstructs the exact live block set at each kernel's timestamp, falling
 * back to the latest snapshot when no timeline was captured (see hip_playback).
 *
 * The blob payloads are a compact little-endian binary format (NOT JSON) so the
 * C++ replayer can parse them without a JSON dependency. The Python shim that
 * produces them (hrr/shim/sitecustomize.py) must emit the exact same layout
 * defined below.
 */

#include "hrr_api_args.h"  /* hrr_event_header */

/* event_type sentinels, outside the hrr_api_id_t range (like HRR_EOF_MARKER). */
#define HRR_SUBALLOC_SNAPSHOT ((uint16_t)0xFFFEu)
/* Incremental alloc/free timeline batch (precise per-kernel layout). The blob
 * holds a run of alloc/free/segment events, each carrying an absolute
 * CLOCK_MONOTONIC timestamp (converted from PyTorch's CLOCK_REALTIME time_us by
 * the capture shim), so the replayer can merge them with kernel-launch event
 * timestamps and reconstruct the exact set of live tensor blocks at the moment
 * any kernel ran — independent of how often the shim sampled. */
#define HRR_SUBALLOC_TIMELINE ((uint16_t)0xFFFDu)

/* events.bin record: references the snapshot blob by FNV-1a-128 hash. */
#pragma pack(push, 1)
typedef struct {
    hrr_event_header hdr;           /* event_type = HRR_SUBALLOC_SNAPSHOT */
    int32_t  ret;                   /* unused (0)                          */
    uint64_t blob_hash_lo;          /* snapshot blob hash (FNV-1a-128)     */
    uint64_t blob_hash_hi;
    uint64_t length;                /* snapshot blob length in bytes       */
} hrr_args_suballoc_snapshot;
#pragma pack(pop)

/* ---- snapshot blob binary layout (little-endian) ----
 *
 *   header:
 *     u32 magic   = HRR_SUBALLOC_BLOB_MAGIC
 *     u32 version = HRR_SUBALLOC_BLOB_VERSION
 *     u64 n_segments
 *   then n_segments × segment record:
 *     u64 addr        segment base device VA (== the hipMalloc pointer)
 *     u64 total       segment size in bytes
 *     u32 n_blocks
 *     u32 _pad
 *   then, immediately after each segment record, n_blocks × block record:
 *     u64 off         byte offset of the block from the segment base
 *     u64 size        block size in bytes
 *     u8  active      1 = active_allocated tensor, 0 = free/inactive
 *     u8  _pad[7]
 */
#define HRR_SUBALLOC_BLOB_MAGIC   ((uint32_t)0x42415348u) /* "HSAB" */
#define HRR_SUBALLOC_BLOB_VERSION ((uint32_t)1u)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_segments;
} hrr_suballoc_blob_header;

typedef struct {
    uint64_t addr;
    uint64_t total;
    uint32_t n_blocks;
    uint32_t _pad;
} hrr_suballoc_seg_rec;

typedef struct {
    uint64_t off;
    uint64_t size;
    uint8_t  active;
    uint8_t  _pad[7];
} hrr_suballoc_blk_rec;
#pragma pack(pop)

/* ---- timeline blob binary layout (little-endian) ----
 *
 *   header:
 *     u32 magic   = HRR_SUBALLOC_TL_MAGIC
 *     u32 version = HRR_SUBALLOC_BLOB_VERSION
 *     u64 n_entries
 *   then n_entries × entry record:
 *     u8  action      one of HRR_TL_*
 *     u8  _pad[7]
 *     u64 addr        block (or segment) base device VA
 *     u64 size        block (or segment) size in bytes
 *     i64 mono_ns     CLOCK_MONOTONIC timestamp (== hrr_event_header.timestamp_ns
 *                     clock), converted by the shim from PyTorch time_us.
 *
 * Entries are emitted in PyTorch trace order; the replayer sorts the merged
 * stream by mono_ns anyway. addr for free uses the freed block base.
 */
#define HRR_SUBALLOC_TL_MAGIC ((uint32_t)0x4C545348u) /* "HSTL" */

#define HRR_TL_ALLOC        ((uint8_t)0u)  /* tensor block allocated   */
#define HRR_TL_FREE         ((uint8_t)1u)  /* tensor block freed       */
#define HRR_TL_SEGMENT_ALLOC ((uint8_t)2u) /* hipMalloc segment mapped */
#define HRR_TL_SEGMENT_FREE  ((uint8_t)3u) /* hipMalloc segment freed  */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_entries;
} hrr_suballoc_tl_header;

typedef struct {
    uint8_t  action;
    uint8_t  _pad[7];
    uint64_t addr;
    uint64_t size;
    int64_t  mono_ns;
} hrr_suballoc_tl_rec;
#pragma pack(pop)
