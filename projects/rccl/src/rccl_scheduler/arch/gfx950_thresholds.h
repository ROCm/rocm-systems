/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// gfx950 tuning table. Included once, by rccl_arch_thresholds.cc.

#ifndef RCCL_GFX950_THRESHOLDS_H_
#define RCCL_GFX950_THRESHOLDS_H_

#include "rccl_arch_thresholds.h"

// gfx950: DDA-IPC cap is 64 MiB for AG, 128 MiB for AR/RS and 4 MiB for AlltoAll. No fabric LL/LL128.
static const rcclArchThresholds rcclArchThresholds_gfx950 = {
  // ddaLLMax: no fabric LL on gfx950.
  .ddaLLMax = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- no fabric LL
    0,                    // [3] ReduceScatter   -- no fabric LL
    0,                    // [4] AllReduce       -- no fabric LL
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- no fabric LL
  },
  // ddaLL128Max: no fabric LL128 on gfx950.
  .ddaLL128Max = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- no fabric LL128
    0,                    // [3] ReduceScatter   -- no fabric LL128
    0,                    // [4] AllReduce       -- no fabric LL128
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- no fabric LL128
  },
  // ddaVmmMax: DDA IPC cap per collective (gfx950 uses IPC, not fabric VMM).
  .ddaVmmMax = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    64ULL*1024*1024,      // [2] AllGather
    128ULL*1024*1024,     // [3] ReduceScatter
    128ULL*1024*1024,     // [4] AllReduce
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    4ULL*1024*1024,       // [8] AlltoAll
  },
  // ddaVmmMaxR2: not used on gfx950 (no R2 DDA path).
  .ddaVmmMaxR2 = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- not used
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce       -- not used
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ddaVmmMaxGraph: not used on gfx950.
  .ddaVmmMaxGraph = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- not used
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce       -- not used
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceNonRegMin: lower bound for CE non-registered window. 0 = no lower bound.
  .ceNonRegMin = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- not used
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce       -- not used
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceNonRegMax: upper bound for CE non-registered (2-shot) window. 0 = disabled.
  .ceNonRegMax = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- 0 = disabled
    0,                    // [3] ReduceScatter   -- not used
    256ULL*1024*1024,     // [4] AllReduce
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceRegMin: registered CE lower bound per collective. 0 = no lower bound.
  // Independent of ceNonRegMin (the non-registered / staging path).
  .ceRegMin = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceRegMax: registered CE upper bound per collective. kThreshUnlimited = no cap
  // (CE fires for any size when CTAPolicy_ZERO and ceAvailable). 0 = disabled.
  .ceRegMax = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- not yet measured
    0,                    // [3] ReduceScatter   -- not used
    256ULL*1024*1024,     // [4] AllReduce
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not yet measured
  },
  // symMaxR2: kThreshUnlimited = symk not suppressed by size on gfx950.
  .symMaxR2 = {
    kThreshUnlimited,     // [0] Broadcast      -- not used
    kThreshUnlimited,     // [1] Reduce          -- not used
    kThreshUnlimited,     // [2] AllGather
    kThreshUnlimited,     // [3] ReduceScatter
    kThreshUnlimited,     // [4] AllReduce
    kThreshUnlimited,     // [5] SendRecv        -- not used
    kThreshUnlimited,     // [6] Send            -- not used
    kThreshUnlimited,     // [7] Recv            -- not used
    kThreshUnlimited,     // [8] AlltoAll        -- not used
  },
  .symMaxR2Graph = {
    kThreshUnlimited,     // [0] Broadcast      -- not used
    kThreshUnlimited,     // [1] Reduce          -- not used
    kThreshUnlimited,     // [2] AllGather
    kThreshUnlimited,     // [3] ReduceScatter
    kThreshUnlimited,     // [4] AllReduce
    kThreshUnlimited,     // [5] SendRecv        -- not used
    kThreshUnlimited,     // [6] Send            -- not used
    kThreshUnlimited,     // [7] Recv            -- not used
    kThreshUnlimited,     // [8] AlltoAll        -- not used
  },
  // symMinR2: 0 = no lower-bound suppression on gfx950.
  .symMinR2 = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather
    0,                    // [3] ReduceScatter
    0,                    // [4] AllReduce
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  .forceCeScratch = true,
  .bcRingLLMaxBytes = 0,
  .bcRingLL128MaxBytes = 0,
};

#endif  // RCCL_GFX950_THRESHOLDS_H_
