/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file amd_launch_descriptor.h
 *
 * AMD Launch Descriptor and L2 Data Prefetch packet definitions.
 *
 * These structures describe the optional metadata packet attached to a kernel
 * dispatch (amd_launch_descriptor_t) and the per-tile L2 prefetch commands
 * it can carry (amd_data_prefetch_t).
 */

#ifndef AMD_LAUNCH_DESCRIPTOR_H_
#define AMD_LAUNCH_DESCRIPTOR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version constants
 * ========================================================================= */

/** Version 0: invalid / disabled.
 *  All-zero memory is safely treated as "no launch descriptor". */
#define AMD_LAUNCH_DESCRIPTOR_VERSION_NONE       0

/** Version 1: gfx1250 layout.
 *  Supports CU enable, dispatch granularity limiter, TG chunk size, and
 *  L2 prefetch.  Bytes 12-19 are reserved and must be 0. */
#define AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250 1

/** L2 prefetch regions supported by version 1 (gfx1250). */
#define AMD_LAUNCH_DESCRIPTOR_PREFETCH_REGIONS_GFX1250 2

/** Maximum L2 prefetch regions across all known versions.
 *  Used to size the prefetch[] array in amd_launch_descriptor_t.
 *  Update this when a new version raises the limit. */
#define AMD_LAUNCH_DESCRIPTOR_MAX_PREFETCH_REGIONS \
    AMD_LAUNCH_DESCRIPTOR_PREFETCH_REGIONS_GFX1250

/* =========================================================================
 * amd_data_prefetch_t  (16 bytes / 4 DWORDs)
 * ========================================================================= */

/**
 * @brief AMD Data Prefetch command.
 *
 * Prefetches a 2D memory region into L2 cache before kernel execution.
 * The region is described as NUM_BURST bursts of BURST_SIZE each, separated
 * by STRIDE. Total prefetch must not exceed L2 capacity (96 MB max).
 *
 * Used by all launch descriptor versions (gfx12xx V1 and later).
 *
 * Addressing modes (MODE field in dw3):
 *   0 = DISABLED    — Command Processor skips this entry entirely.
 *                     An all-zero entry is safely "unused".
 *   1 = ABSOLUTE_VA — addr_lo / addr_hi contain the 256B-aligned virtual
 *                     address of the tile.
 *   2 = KERNARG_OFS — addr_lo[11:0] is a byte offset into the kernel
 *                     argument buffer. The CP reads an 8-byte pointer from
 *                     (kernarg_base + offset) and uses it as the tile base
 *                     address. addr_lo[23:12] and addr_hi must be 0.
 *                     This makes the descriptor reusable across dispatches
 *                     when the same kernarg slot always holds the pointer.
 *   3 = RESERVED    — must not be used.
 *
 * Total size: 16 bytes (4 DWORDs).
 */
typedef struct amd_data_prefetch_s {
  /**
   * DWORD 0: Control + Geometry
   *
   *  31         16 15          5 4:3   2:1   0
   * +------------+-------------+------+------+----+
   * | NUM_BURST  | BURST_SIZE  |SCOPE | TEMP |COOP|
   * +------------+-------------+------+------+----+
   */
  union {
    struct {
      /** If 0, each XCC prefetches the full packet (non-cooperative).
       *  If 1, prefetching is distributed across XCCs (performance hint;
       *  applicable to SPX/QPX/DPX modes, ignored in CPX mode). */
      uint32_t cooperative : 1;   /* [0]     */
      /** GL2 temporal policy for cache lines:
       *  0 = REGULAR       — normal cache behaviour.
       *  1 = NON_TEMPORAL  — data unlikely to be reused.
       *  2 = HIGH_TEMPORAL — data very likely to be reused; retain longer.
       *  3 = LAST_USE      — hint that this is the final access. */
      uint32_t temporal    : 2;   /* [2:1]   */
      /** GL2 scope for the prefetch:
       *  0 = Reserved (CU), 1 = Reserved (SE),
       *  2 = DEVICE (expected use), 3 = SYSTEM. */
      uint32_t scope       : 2;   /* [4:3]   */
      /** 256B units per burst, encoded as (N - 1).
       *  0 = 256 B, 1 = 512 B, ..., 2047 = 512 KB per burst. */
      uint32_t burst_size  : 11;  /* [15:5]  */
      /** Total burst count, encoded as (N - 1).
       *  0 = 1 burst, 1 = 2 bursts, ..., 65535 = 64K bursts. */
      uint32_t num_burst   : 16;  /* [31:16] */
    };
    uint32_t dw0;
  };
  /**
   * DWORD 1: Chunk Size + Permissions + Address Low
   *
   *  31           8 7:6  5:4   3:0
   * +--------------+-----+-----+------+
   * |   ADDR_LO    | RSV |PERM | CHSZ |
   * +--------------+-----+-----+------+
   */
  union {
    struct {
      /** Bursts per chunk (log2 encoded) for cooperative prefetching:
       *  0=1, 1=2, 2=4, ..., 15=32K bursts per chunk. */
      uint32_t chunk_size              : 4;  /* [3:0]   */
      /** Translation permissions: 0=read, 1=write, 2=execute (expected). */
      uint32_t translation_permissions : 2;  /* [5:4]   */
      /** Reserved. Must be 0. */
      uint32_t reserved1               : 2;  /* [7:6]   */
      /** ABSOLUTE_VA: VA[31:8]; actual VA = addr_lo << 8 (256B aligned).
       *  KERNARG_OFS: byte offset[11:0] into kernarg buffer (must be
       *  8B aligned); bits[23:12] must be 0. */
      uint32_t addr_lo                 : 24; /* [31:8]  */
    };
    uint32_t dw1;
  };
  /**
   * DWORD 2: Address High
   *
   *  31       25 24            0
   * +-----------+---------------+
   * | RESERVED  |   ADDR_HI     |
   * +-----------+---------------+
   *
   * ABSOLUTE_VA: VA[56:32], supports 57-bit virtual addresses.
   * KERNARG_OFS: must be 0.
   */
  union {
    struct {
      uint32_t addr_hi   : 25; /* [24:0]  */
      uint32_t reserved2 : 7;  /* [31:25] */
    };
    uint32_t dw2;
  };
  /**
   * DWORD 3: Stride + Mode
   *
   *  31:30  29:24  23            0
   * +------+------+---------------+
   * | MODE | RSVD |    STRIDE     |
   * +------+------+---------------+
   */
  union {
    struct {
      /** Stride between consecutive bursts in units of 256 B.
       *  For 1D prefetch: set stride = burst_size + 1.
       *  For 2D prefetch: stride is the row pitch.
       *  Max value: (2^24 - 1) * 256 B ≈ 4 GB. */
      uint32_t stride    : 24; /* [23:0]  */
      /** Reserved. Must be 0. */
      uint32_t reserved3 : 6;  /* [29:24] */
      /** Addressing mode and enable (see struct-level comment). */
      uint32_t mode      : 2;  /* [31:30] */
    };
    uint32_t dw3;
  };
} amd_data_prefetch_t; /* 16 bytes (4 DWORDs) */

/* =========================================================================
 * amd_launch_descriptor_t  (52 bytes / 13 DWORDs)
 * ========================================================================= */

/**
 * @brief AMD Launch Descriptor.
 *
 * An optional metadata packet attached to a kernel dispatch that allows
 * software to override or extend dispatch behaviour beyond what the base
 * AQL kernel dispatch packet provides.  The Command Processor reads this
 * descriptor from the metadata queue before launching the kernel and
 * programs the relevant hardware registers prior to dispatch.
 *
 * The descriptor is versioned so that the Command Processor can reject
 * unrecognised formats and so that future hardware generations can extend
 * the layout without breaking existing software. Version 0 (all-zero memory)
 * is explicitly invalid to catch uninitialised descriptors.
 *
 * Features controlled by this descriptor:
 *
 *   CU Enable (all versions)
 *     Restricts which compute units are active for this dispatch.
 *     The same CU selection is applied symmetrically to all enabled
 *     shader engines and replicated identically across all XCCs.
 *     Primarily used for green-context style resource partitioning,
 *     where a fraction of the GPU is reserved for a given workload.
 *
 *   Dispatch Granularity Limiter (all versions)
 *     Caps the maximum number of dispatch granularity units
 *     (work groups or clusters) simultaneously resident on the device,
 *     enabling multiple kernels to coexist on the GPU without exhausting
 *     scratch memory and reducing cache thrashing under high occupancy.
 *
 *   TG Chunk Size (all versions)
 *     Controls how many workgroups are dispatched to one XCC before
 *     round-robining to the next, tuning the locality vs. bandwidth
 *     trade-off for a given kernel.
 *
 *   L2 Prefetch (all versions)
 *     Up to two 2D surface prefetch descriptors that instruct the
 *     Command Processor to warm the L2 cache with the next tile of
 *     data while the current tile is being computed.
 *
 * Memory layout:
 *   Offset  Size  Field
 *   ──────  ────  ──────────────────────────────
 *    0       1    version
 *    1       1    priority
 *    2       1    pm_hint
 *    3       1    reserved0
 *    4       4    cu_enable
 *    8       2    dispatch_granularity_limiter
 *   10       2    reserved1
 *   12       1    reserved2
 *   13       1    reserved3
 *   14       2    tg_chunk_size
 *   16       4    reserved4
 *   20      16    prefetch[0]
 *   36      16    prefetch[1]
 *   ──────  ────
 *   Total: 52 bytes (13 DWORDs)
 */
typedef struct amd_launch_descriptor_s {
  /**
   * DWORD 0 (offset 0): Version + Priority + PM Hint + Reserved
   *
   *  31       24 23      16 15       8 7       0
   * +----------+----------+----------+---------+
   * | RSVD (0) | PM_HINT  | PRIORITY | VERSION |
   * +----------+----------+----------+---------+
   */
  /** Byte 0: ABI version.
   *  Command Processor rejects descriptors with an unrecognised version.
   *  0 = NONE (invalid, ensures zeroed memory is rejected)
   *  1 = gfx1250 layout */
  uint8_t version;
  /** Byte 1: Dispatch priority.
   *  Encoding TBD (pipe priority, SPI wave priority, etc).
   *  0 = default (no priority override). */
  uint8_t priority;
  /** Byte 2: PM firmware hint.
   *  Opaque hint passed to power management firmware.
   *  0 = no hint (default). */
  uint8_t pm_hint;
  /** Byte 3: Reserved. Must be 0. */
  uint8_t reserved0;

  /**
   * DWORD 1 (offset 4): CU Enable
   *
   *  31          10 9:8    7:4    3:0
   * +--------------+-------+------+-------+
   * | RESERVED (0) | SE_EN |  CNT | START |
   * +--------------+-------+------+-------+
   */
  /** Bytes 4–7: CU enable.
   *
   *  Specifies which CUs to enable within each active SE.
   *  This setting is replicated identically to all 8 XCCs.
   *  All enabled SEs receive an identical CU mask built from
   *  START and CNT, enforcing symmetric SE configurations.
   *
   *  START [3:0]: first CU index within each enabled SE (0–15).
   *  CNT   [7:4]: number of contiguous CUs to enable from START.
   *               0 = all active CUs in that SE (NUM_ENABLED_CUS_PER_SE).
   *  SE_EN [9:8]: bit 8 = SE0 enable, bit 9 = SE1 enable.
   *               1 = SE enabled (START+CNT mask applied).
   *               0 = SE disabled (all CUs off).
   *  [31:10]: reserved, must be 0.
   *
   *  0x00000000 = bypass (all CUs enabled, default).
   *  When non-zero:
   *    - SE_EN must be non-zero (at least one SE enabled).
   *    - START + CNT must not exceed NUM_ENABLED_CUS_PER_SE.
   *
   *  Command Processor resolves CNT:
   *    effective_cnt = (CNT == 0) ? NUM_ENABLED_CUS_PER_SE : CNT;
   *  then builds the CU enable mask once from START + effective_cnt
   *  and writes it to COMPUTE_STATIC_THREAD_MGMT_SE{n} for each
   *  enabled SE across all 8 XCCs.  Disabled SEs get a zero mask. */
  union {
    struct {
      uint32_t cu_start    : 4;  /* [3:0]   */
      uint32_t cu_count    : 4;  /* [7:4]   */
      uint32_t se_en       : 2;  /* [9:8]   */
      uint32_t reserved_cu : 22; /* [31:10] must be 0 */
    };
    uint32_t cu_enable;
  };

  /**
   * DWORD 2 (offset 8): Dispatch Granularity Limiter + Reserved
   *
   *  31                16 15                           0
   * +-------------------+-------------------------------+
   * |   RESERVED (0)    | DISPATCH_GRANULARITY_LIMITER  |
   * +-------------------+-------------------------------+
   */
  /** Bytes 8–9: Dispatch granularity limiter.
   *
   *  Maximum number of dispatch granularity units that should be
   *  simultaneously resident on the device for this dispatch.
   *
   *  The granularity unit depends on the dispatch type:
   *    - Non-clustered dispatch (cluster size = 1x1x1):
   *        unit = one work group.
   *    - Clustered dispatch (cluster size > 1x1x1):
   *        unit = one cluster (all work groups in the cluster
   *        are dispatched atomically and count as one unit).
   *
   *  The Command Processor translates this value to the appropriate
   *  hardware wave-count registers by multiplying by the number of
   *  waves per granularity unit for this kernel and architecture.
   *  Software does not need to know wave size, wave32/wave64 mode,
   *  or how ragged-edge work groups are handled.
   *
   *  The Command Processor may additionally use this value to
   *  right-size scratch memory allocation for the dispatch, reducing
   *  over-allocation when the kernel's occupancy is constrained by
   *  VGPR, LDS, or scratch requirements.
   *
   *  Constraints:
   *    - Must be 0 (default) or >= 1 granularity unit.
   *    - Must not be set below the size of one cluster, or the
   *      dispatch may deadlock.
   *    - The value is a chip-wide limit.  The Command Processor
   *      distributes it across active XCCs internally; software
   *      does not need to account for XCC count.
   *    - When cu_enable restricts active shader engines, the
   *      effective limit applies only over those enabled SEs.
   *
   *  0 = no limit (default).  Command Processor uses its own
   *  occupancy and scratch allocation policy. */
  uint16_t dispatch_granularity_limiter;
  /** Bytes 10–11: Reserved. Must be 0. */
  uint16_t reserved1;

  /**
   * DWORD 3 (offset 12): Reserved + TG Chunk Size
   *
   *  31         16 15       8 7       0
   * +-------------+----------+---------+
   * |TG_CHUNK_SIZE|  MBZ     | MBZ     |
   * +-------------+----------+---------+
   */
  /** Byte 12: Reserved. Must be 0. */
  uint8_t reserved2;
  /** Byte 13: Reserved. Must be 0. */
  uint8_t reserved3;
  /** Bytes 14–15: TG chunk size.
   *  Number of workgroups dispatched to one XCC before
   *  round-robining to the next.
   *
   *  The caller sets this field to the desired chunk size.
   *  Command Processor routes the value to the appropriate register.
   *
   *  Values:
   *    0 = inherit from the next level up in the hierarchy
   *        (dispatch -> queue -> device -> platform).
   *    1 = 1 WG per XCC (bandwidth-optimized, current AQL default).
   *    N = N WGs per XCC (locality-optimized, e.g. 16 for GEMM). */
  uint16_t tg_chunk_size;

  /**
   * DWORD 4 (offset 16): Reserved. Must be 0.
   */
  uint32_t reserved4;

  /**
   * DWORDs 5–12 (offsets 20–51): L2 Prefetch Tiles
   *
   * Up to AMD_LAUNCH_DESCRIPTOR_MAX_PREFETCH_REGIONS (currently 2) 2D surface
   * prefetch entries.  The supported count per version is:
   *   version 1 (gfx1250): AMD_LAUNCH_DESCRIPTOR_PREFETCH_REGIONS_GFX1250
   *
   * ── L2 Prefetch Tile 0 (DWORDs 5–8, offsets 20–35) ───────────────
   *
   *  DWORD 5 (offset 20): Control + Geometry
   *
   *   31         16 15          5 4:3   2:1   0
   *  +-----------+-------------+------+------+----+
   *  | NUM_BURST | BURST_SIZE  |SCOPE | TEMP |COOP|
   *  +-----------+-------------+------+------+----+
   *
   *  DWORD 6 (offset 24): Chunk Size + Permissions + Address Low
   *
   *   31                    8 7:6  5:4   3:0
   *  +------------------------+-----+-----+------+
   *  |       ADDR_LO          | RSV |PERM | CHSZ |
   *  +------------------------+-----+-----+------+
   *
   *  DWORD 7 (offset 28): Address High
   *
   *   31       25 24                         0
   *  +-----------+----------------------------+
   *  | RESERVED  |          ADDR_HI           |
   *  +-----------+----------------------------+
   *
   *  DWORD 8 (offset 32): Stride + Mode
   *
   *   31:30 29:24  23                          0
   *  +-----+------+-----------------------------+
   *  |MODE | RSVD |           STRIDE            |
   *  +-----+------+-----------------------------+
   *
   * ── L2 Prefetch Tile 1 (DWORDs 9–12, offsets 36–51) ──────────────
   *
   *  DWORD 9 (offset 36): Control + Geometry
   *
   *   31         16 15          5 4:3   2:1   0
   *  +-----------+-------------+------+------+----+
   *  | NUM_BURST | BURST_SIZE  |SCOPE | TEMP |COOP|
   *  +-----------+-------------+------+------+----+
   *
   *  DWORD 10 (offset 40): Chunk Size + Permissions + Address Low
   *
   *   31                    8 7:6  5:4   3:0
   *  +------------------------+-----+-----+------+
   *  |       ADDR_LO          | RSV |PERM | CHSZ |
   *  +------------------------+-----+-----+------+
   *
   *  DWORD 11 (offset 44): Address High
   *
   *   31       25 24                         0
   *  +-----------+----------------------------+
   *  | RESERVED  |          ADDR_HI           |
   *  +-----------+----------------------------+
   *
   *  DWORD 12 (offset 48): Stride + Mode
   *
   *   31:30 29:24  23                          0
   *  +-----+------+-----------------------------+
   *  |MODE | RSVD |           STRIDE            |
   *  +-----+------+-----------------------------+
   *
   * An all-zero entry (mode == 0) is unused / disabled.
   * Leave all-zero (mode == DISABLED) to skip a slot.
   * Total prefetch across all slots must not exceed L2 capacity.
   */
  amd_data_prefetch_t prefetch[AMD_LAUNCH_DESCRIPTOR_MAX_PREFETCH_REGIONS];
  /* Total: 20 + 32 = 52 bytes (13 DWORDs) */
} amd_launch_descriptor_t;

#ifdef __cplusplus
static_assert(sizeof(amd_data_prefetch_t) == 16,
              "amd_data_prefetch_t must be exactly 16 bytes (4 DWORDs)");
static_assert(sizeof(amd_launch_descriptor_t) == 52,
              "amd_launch_descriptor_t must be exactly 52 bytes (13 DWORDs)");
}
#endif

#endif /* AMD_LAUNCH_DESCRIPTOR_H_ */
