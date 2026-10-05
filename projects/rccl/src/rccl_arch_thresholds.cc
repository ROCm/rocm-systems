/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Per-architecture DDA/CE dispatch threshold tables and tuning index lookup.
// Moved from graph/tuning.cc to isolate RCCL-specific dispatch data from
// NVIDIA-originated graph infrastructure.

#include "rccl_arch_thresholds.h"
#include "archinfo.h"  // IsArchMatch
#include "net.h"       // rcclPrimaryNic, rcclIBNicTypeAINIC

#include <string>
#include <vector>

// Per-arch DDA/CE dispatch threshold tables.
// DDA arrays: [Broadcast=0, Reduce=1, AllGather=2, ReduceScatter=3, AllReduce=4,
//              SendRecv=5, Send=6, Recv=7, AlltoAll=8]. 0 disables that tier.
// All collectives compare total message bytes against the table values.

// gfx1250 placeholders -- validate against sweep data (AICOMRCCL-1756).
// Index mapping: [Bcast=0, Reduce=1, AG=2, RS=3, AR=4, SR=5, Send=6, Recv=7, A2A=8]
static const rcclArchThresholds rcclArchThresholds_gfx1250 = {
  // ddaLLMax: DDA LL tier ceiling per collective (fabric gfx1250 only).
  .ddaLLMax = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    128ULL*1024,         // [2] AllGather
    16ULL*1024*1024,     // [3] ReduceScatter
    32ULL*1024*1024,     // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    64ULL*1024,          // [8] AlltoAll
  },
  // ddaLL128Max: DDA LL128 tier ceiling per collective.
  // RCCL_PARAM(DdaLL128, ...) defaults to 1 (enabled); set RCCL_DDA_LL128=0 to disable.
  .ddaLL128Max = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    64ULL*1024*1024,     // [2] AllGather
    2ULL*1024*1024,      // [3] ReduceScatter
    32ULL*1024*1024,     // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    1ULL*1024*1024,      // [8] AlltoAll
  },
  // ddaVmmMax: DDA VMM (fabric simple) tier ceiling per collective.
  // Messages above this fall to Ring/CE (or sym kernel for R2).
  .ddaVmmMax = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- 0 = disabled
    0,                   // [3] ReduceScatter   -- 0 = disabled
    16ULL*1024*1024,     // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- 0 = disabled
  },
  // ddaVmmMaxR2: DDA VMM cap when recv buffer is registered (R2 mode).
  // DDA is gated on !symEligible (AR: !symkRequested) on every arch, including
  // gfx1250 fabric; R2 sum/avg never takes DDA. A non-zero entry only shortens
  // DDA when recv is registered and that gate still passes. 0 means use ddaVmmMax.
  .ddaVmmMaxR2 = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- not used
    0,                   // [3] ReduceScatter   -- not used
    4ULL*1024*1024,      // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- not used
  },
  // ddaVmmMaxGraph: DDA VMM cap during graph capture (graphCapturingHint=true).
  // CE AllReduce is blocked by graphModeSeen latch during graph captures.
  // Extending DDA VMM for AR lets DDA fill the window CE would otherwise absorb.
  // 0 means use ddaVmmMax (no graph-specific override for that collective).
  .ddaVmmMaxGraph = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- not used
    0,                   // [3] ReduceScatter   -- not used
    32ULL*1024*1024,    // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- not used
  },
  // ceNonRegMin: lower bound for CE-Scratch window per collective (-R 0).
  // 0 = disabled for that collective.
  .ceNonRegMin = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- 0 = disabled
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce       -- not used
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceNonRegMax: upper bound for CE-Scratch window per collective (-R 0).
  // AR: 0 = 2-shot off. ceARTmpBuf stays at the default 256 MiB unless this entry is larger.
  .ceNonRegMax = {
    0,                    // [0] Broadcast      -- not used
    0,                    // [1] Reduce          -- not used
    0,                    // [2] AllGather       -- 0 = disabled
    0,                    // [3] ReduceScatter   -- not used
    0,                    // [4] AllReduce       -- 0 = 2-shot off
    0,                    // [5] SendRecv        -- not used
    0,                    // [6] Send            -- not used
    0,                    // [7] Recv            -- not used
    0,                    // [8] AlltoAll        -- not used
  },
  // ceRegMax: registered CE upper bound per collective (-R 2).
  .ceRegMax = {
    0,                             // [0] Broadcast      -- not used
    0,                             // [1] Reduce          -- not used
    8ULL*1024*1024*1024,           // [2] AllGather
    0,                             // [3] ReduceScatter   -- not used
    256ULL*1024*1024,              // [4] AllReduce
    0,                             // [5] SendRecv        -- not used
    0,                             // [6] Send            -- not used
    0,                             // [7] Recv            -- not used
    0,                             // [8] AlltoAll        -- not used
  },
  // symMaxR2: suppress symk in favour of CE-registered when recv is registered and
  // msg > threshold. kThreshUnlimited = no suppression; non-zero literal = byte cap;
  // 0 = always suppress (symk not used for this collective on gfx1250).
  .symMaxR2 = {
    0,                    // [0] Broadcast      -- not used on gfx1250
    0,                    // [1] Reduce          -- not used on gfx1250
    2ULL*1024*1024,       // [2] AllGather
    kThreshUnlimited,     // [3] ReduceScatter
    0,                    // [4] AllReduce
    0,                    // [5] SendRecv        -- not used on gfx1250
    0,                    // [6] Send            -- not used on gfx1250
    0,                    // [7] Recv            -- not used on gfx1250
    0,                    // [8] AlltoAll        -- not used on gfx1250
  },
  // Graph capture: CE is blocked, so symk or DDA wins. kThreshUnlimited = no suppression;
  // non-zero literal = byte cap above which symk is withdrawn so DDA can win instead;
  // 0 = always suppress (symk not used for this collective on gfx1250).
  .symMaxR2Graph = {
    0,                    // [0] Broadcast      -- not used on gfx1250
    0,                    // [1] Reduce          -- not used on gfx1250
    2ULL*1024*1024,       // [2] AllGather
    kThreshUnlimited,     // [3] ReduceScatter
    0,                    // [4] AllReduce
    0,                    // [5] SendRecv        -- not used on gfx1250
    0,                    // [6] Send            -- not used on gfx1250
    0,                    // [7] Recv            -- not used on gfx1250
    0,                    // [8] AlltoAll        -- not used on gfx1250
  },
  // symMinR2: suppress symk below this size for R2 buffers so DDA wins in that sub-range.
  // kThreshUnlimited = always suppress (totalBytes < SIZE_MAX always true) for unused collectives.
  .symMinR2 = {
    kThreshUnlimited,     // [0] Broadcast      -- not used on gfx1250
    kThreshUnlimited,     // [1] Reduce          -- not used on gfx1250
    128ULL*1024,          // [2] AllGather
    2ULL*1024*1024,       // [3] ReduceScatter
    8ULL*1024*1024,       // [4] AllReduce
    kThreshUnlimited,     // [5] SendRecv        -- not used on gfx1250
    kThreshUnlimited,     // [6] Send            -- not used on gfx1250
    kThreshUnlimited,     // [7] Recv            -- not used on gfx1250
    kThreshUnlimited,     // [8] AlltoAll        -- not used on gfx1250
  },
};

// gfx950: DDA-IPC cap is 128 MiB for AR/AG/RS and 4 MiB for AlltoAll. No fabric LL/LL128.
static const rcclArchThresholds rcclArchThresholds_gfx950 = {
  .ddaLLMax    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaLL128Max = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaVmmMax      = {0, 0, 128ULL*1024*1024,  128ULL*1024*1024,  128ULL*1024*1024,  0, 0, 0, 4ULL*1024*1024},
  .ddaVmmMaxR2    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaVmmMaxGraph = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ceNonRegMin    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ceNonRegMax = {0, 0, 0, 0, 256ULL*1024*1024, 0, 0, 0, 0},
  .ceRegMax    = {0, 0, 0, 0, 256ULL*1024*1024, 0, 0, 0, 0},
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
  .symMinR2    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
};

// gfx942: DDA-IPC cap is 8 MiB for AR/AG/RS and 4 MiB for AlltoAll. No fabric LL/LL128.
static const rcclArchThresholds rcclArchThresholds_gfx942 = {
  .ddaLLMax    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaLL128Max = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaVmmMax      = {0, 0, 8ULL*1024*1024,    8ULL*1024*1024,    8ULL*1024*1024,    0, 0, 0, 4ULL*1024*1024},
  .ddaVmmMaxR2    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ddaVmmMaxGraph = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ceNonRegMin    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
  .ceNonRegMax = {0, 0, 0, 0, 256ULL*1024*1024, 0, 0, 0, 0},
  .ceRegMax    = {0, 0, 0, 0, 256ULL*1024*1024, 0, 0, 0, 0},
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
  .symMinR2    = {0, 0, 0, 0, 0, 0, 0, 0, 0},
};

const rcclArchThresholds* rcclGetArchThresholds(const char* gcn) {
  if (gcn == nullptr) return nullptr;
  if (IsArchMatch(gcn, "gfx1250")) return &rcclArchThresholds_gfx1250;
  if (IsArchMatch(gcn, "gfx950")) return &rcclArchThresholds_gfx950;
  if (IsArchMatch(gcn, "gfx942")) return &rcclArchThresholds_gfx942;
  return nullptr;
}


