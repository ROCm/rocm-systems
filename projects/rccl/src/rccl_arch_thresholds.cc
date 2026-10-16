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
// SPX table, also used for CPX and when no partition mode is reported. DPX has its own
// copy below (rcclArchThresholds_gfx1250_dpx): a non-tuning change here (a gate such as
// forceCeScratch, a disabled slot) usually belongs in both.
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
    128ULL*1024*1024,    // [2] AllGather       -- beats Ring at 128M on 8-64 ranks (AICOMRCCL-2876)
    2ULL*1024*1024,      // [3] ReduceScatter
    256ULL*1024*1024,    // [4] AllReduce       -- two-shot beats Ring to 256M (AICOMRCCL-2875); its scratch
                         //                        slot reaches 256M from 32 ranks, ~128M at 2-8, 240M at 16
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
  // Rank-scaled caps from the 8-64 rank H13 sweep, AICOMRCCL-1756. The
  // crossovers move with the per-rank share, so on 8+ rank comms these replace the
  // totals above (which stay in force for smaller comms).
  .ddaLLMaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    16ULL*1024,          // [2] AllGather       -- LL beats LL128 to 128K/256K/512K/1M at 8/16/32/64
    256ULL*1024,         // [3] ReduceScatter   -- LL past 2M/4M/8M/16M at 8/16/32/64 trails LL128 up to 2x
    1ULL*1024*1024,      // [4] AllReduce       -- LL two-shot beats LL128 1.4-1.9x to 8M/16M/32M/64M
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- unchanged
  },
  .ddaLL128MaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    16ULL*1024*1024,     // [2] AllGather       -- 128M at 8 ranks, 256M (clipped) from 16
    512ULL*1024,         // [3] ReduceScatter   -- the LL128 slot reaches nRanks x 512K
    0,                   // [4] AllReduce       -- ddaLL128Max total
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- unchanged
  },
  .ddaVmmMaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- VMM off
    2ULL*1024*1024,      // [3] ReduceScatter   -- VMM beats Ring 1.2-1.9x to 16M/32M/64M/128M
    0,                   // [4] AllReduce       -- ddaVmmMax total (see ddaArVmmFirst*)
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- VMM off
  },
  .rankScaledMinRanks = 8,
  .rankScaledMaxBytes = 256ULL*1024*1024,
  // From 32 ranks, AllReduce VMM beats LL two-shot / LL128 1.3-3.7x between 2M and nRanks x 512K.
  .ddaArVmmFirstMinRanks = 32,
  .ddaArVmmFirstMinBytes = 2ULL*1024*1024,
  .ddaArVmmFirstMaxPerRank = 512ULL*1024,
  // VMM peaks at 128 / 128 / 64 / 32 blocks on 8 / 16 / 32 / 64 ranks.
  .ddaVmmBlockBudget = 2048,
  .ddaVmmMinBlocks = 32,
  .ddaVmmMaxBlocks = 128,
  // Ring / Tree crossovers, same sweep (AllReduce LL128: AICOMRCCL-2874/2875).
  .arRingLL128MaxPerRank = 32ULL*1024*1024,       // Ring LL128 beats Simple to 256M/512M/1G/2G at 8-64
  .arTreeLL128Unlimited = true,                   // Tree LL128 beats Tree Simple 1.3-2.9x at 32M-16G
  .arTreeMaxPerRank = 1ULL*1024*1024,             // Tree beats Ring 1.4-2.3x to 8M/16M/32M/64M
  .arTreeLLMaxPerRank = 32ULL*1024,               // Tree LL beats LL128 1.2-1.6x to 256K/512K/1M/2M
  .agRingLL128MaxPerRank = 64ULL*1024*1024,       // Ring LL128 beats Simple to 512M/1G/2G/4G
  .rsRingLL128MinPerRank = 512ULL*1024,           // Ring LL128 beats LL from 1M per rank
  .rsRingLL128MaxPerRank = 8ULL*1024*1024,        // 2/4-byte types: Simple wins past 8M per rank
  .rsRingLL128MaxPerRank1B = 32ULL*1024*1024,     // fp8: LL128 holds to 32M per rank
  .rsRingSimpleCappedMaxPerRank = 32ULL*1024*1024,// 128 channels beat 192 by 1.1-1.3x in (8M, 32M]
  .rsRingSimpleCappedChannels = 128,
  // CE-Scratch trails the DDA / Ring / Direct kernels it displaces between the DDA
  // ceiling and the scratch size (nRanks x 64 MiB) (AICOMRCCL-2876).
  .forceCeScratch = false,
};

// gfx1250 in DPX mode (each GPU split into two partitions), picked by
// RCCL_TUNING_PARTITION_MODE. Started as a copy of the SPX table above; change entries
// here, not there, to tune DPX. Entries marked (DPX) were re-measured on DPX
// nodes at 8/16/32 ranks (rccl-tests -n 10, median of 3 rounds); the other per-entry
// notes are still the SPX measurements.
// Index mapping: [Bcast=0, Reduce=1, AG=2, RS=3, AR=4, SR=5, Send=6, Recv=7, A2A=8]
static const rcclArchThresholds rcclArchThresholds_gfx1250_dpx = {
  // ddaLLMax: DDA LL tier ceiling per collective (fabric gfx1250 only).
  .ddaLLMax = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    512ULL*1024,         // [2] AllGather       -- LL beats LL128 1.4-1.6x to 512K at 8 ranks; the
                         //                        crossover falls with rank count (512K at 16, ~128K
                         //                        at 32), so this is a total, not rank-scaled (DPX)
    16ULL*1024*1024,     // [3] ReduceScatter
    32ULL*1024*1024,     // [4] AllReduce
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    2ULL*1024*1024,      // [8] AlltoAll        -- LL beats Direct 1.2-4.3x to 2M at 8/16/32 (DPX)
  },
  // ddaLL128Max: DDA LL128 tier ceiling per collective.
  // RCCL_PARAM(DdaLL128, ...) defaults to 1 (enabled); set RCCL_DDA_LL128=0 to disable.
  .ddaLL128Max = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    64ULL*1024*1024,     // [2] AllGather       -- Ring beats it 1.1-1.4x at 128M-256M on 8/16/32 (DPX)
    2ULL*1024*1024,      // [3] ReduceScatter
    256ULL*1024*1024,    // [4] AllReduce       -- two-shot beats Ring to 256M (AICOMRCCL-2875); its scratch
                         //                        slot reaches 256M from 32 ranks, ~128M at 2-8, 240M at 16
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
    16ULL*1024*1024,     // [3] ReduceScatter   -- wins only at 16M on 8 ranks; the LL tier covers it
                         //                        below 8 ranks and from 16 (DPX)
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
  // Rank-scaled caps from the 8-64 rank H13 sweep, AICOMRCCL-1756. The
  // crossovers move with the per-rank share, so on 8+ rank comms these replace the
  // totals above (which stay in force for smaller comms).
  .ddaLLMaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- ddaLLMax total (DPX)
    1ULL*1024*1024,      // [3] ReduceScatter   -- LL beats DDA Simple / Ring 1.1-1.5x to 8M/16M/32M
                         //                        at 8/16/32 (DPX)
    1ULL*1024*1024,      // [4] AllReduce       -- LL two-shot beats LL128 1.4-1.9x to 8M/16M/32M/64M
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- unchanged
  },
  .ddaLL128MaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- ddaLL128Max total (DPX)
    512ULL*1024,         // [3] ReduceScatter   -- the LL128 slot reaches nRanks x 512K
    0,                   // [4] AllReduce       -- ddaLL128Max total
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- unchanged
  },
  .ddaVmmMaxPerRank = {
    0,                   // [0] Broadcast      -- not used
    0,                   // [1] Reduce          -- not used
    0,                   // [2] AllGather       -- VMM off
    0,                   // [3] ReduceScatter   -- ddaVmmMax total: Ring LL128 beats VMM 1.3-1.5x at
                         //                        32M/64M on 16/32 ranks (DPX)
    0,                   // [4] AllReduce       -- ddaVmmMax total (see ddaArVmmFirst*)
    0,                   // [5] SendRecv        -- not used
    0,                   // [6] Send            -- not used
    0,                   // [7] Recv            -- not used
    0,                   // [8] AlltoAll        -- VMM off
  },
  .rankScaledMinRanks = 8,
  .rankScaledMaxBytes = 256ULL*1024*1024,
  // From 32 ranks, AllReduce VMM beats LL two-shot between 2M and nRanks x 256K; above that LL
  // two-shot wins 1.2-1.3x (16M on 32 ranks) (DPX).
  .ddaArVmmFirstMinRanks = 32,
  .ddaArVmmFirstMinBytes = 2ULL*1024*1024,
  .ddaArVmmFirstMaxPerRank = 256ULL*1024,
  // VMM peaks at 128 / 128 / 64 / 32 blocks on 8 / 16 / 32 / 64 ranks.
  .ddaVmmBlockBudget = 2048,
  .ddaVmmMinBlocks = 32,
  .ddaVmmMaxBlocks = 128,
  // Ring / Tree crossovers, same sweep (AllReduce LL128: AICOMRCCL-2874/2875).
  .arRingLL128MaxPerRank = 16ULL*1024*1024,       // Ring Simple beats LL128 1.1-1.3x from 256M/512M/1G
                                                  // at 8/16/32 (DPX)
  .arTreeLL128Unlimited = true,                   // Tree LL128 beats Tree Simple 1.3-2.9x at 32M-16G
  .arTreeMaxPerRank = 1ULL*1024*1024,             // Tree beats Ring 1.4-2.3x to 8M/16M/32M/64M
  .arTreeLLMaxPerRank = 256ULL*1024,              // Tree LL beats LL128 1.3-2.1x to 2M/4M/8M at 8/16/32
                                                  // (fp8, DPX)
  .agRingLL128MaxPerRank = 16ULL*1024*1024,       // Ring Simple beats LL128 1.2-1.4x from 256M/1G/2G
                                                  // at 8/16/32 (DPX)
  .rsRingLL128MinPerRank = 1ULL*1024*1024,        // Ring LL beats LL128 1.1-1.2x to 1M per rank
                                                  // (fp8 at 16/32 ranks, DPX)
  .rsRingLL128MaxPerRank = 8ULL*1024*1024,        // 2/4-byte types: Simple wins past 8M per rank
  .rsRingLL128MaxPerRank1B = 16ULL*1024*1024,     // fp8: Ring Simple beats LL128 1.2-1.3x from 32M per
                                                  // rank (DPX)
  .rsRingSimpleCappedMaxPerRank = 32ULL*1024*1024,// 128 channels beat 192 by 1.1-1.3x in (8M, 32M]
  .rsRingSimpleCappedChannels = 128,
  // CE-Scratch trails the DDA / Ring / Direct kernels it displaces between the DDA
  // ceiling and the scratch size (nRanks x 64 MiB) (AICOMRCCL-2876).
  .forceCeScratch = false,
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
  .forceCeScratch = true,
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
  .forceCeScratch = true,
};

const rcclArchThresholds* rcclGetArchThresholds(const char* gcn, rcclTuningPartition partition) {
  if (gcn == nullptr) return nullptr;
  if (IsArchMatch(gcn, "gfx1250")) {
    return partition == RCCL_TUNING_PARTITION_DPX ? &rcclArchThresholds_gfx1250_dpx : &rcclArchThresholds_gfx1250;
  }
  if (IsArchMatch(gcn, "gfx950")) return &rcclArchThresholds_gfx950;
  if (IsArchMatch(gcn, "gfx942")) return &rcclArchThresholds_gfx942;
  return nullptr;
}


