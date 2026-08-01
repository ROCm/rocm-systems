/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_KERNEL_TIMING_H_
#define NCCL_KERNEL_TIMING_H_

#include <stdint.h>

#include "nccl.h"

struct ncclComm;
struct ncclKernelPlan;
struct ncclKernelTimingCtx;

/* One RCCL kernel dispatch, measured from the dispatch packet itself rather
 * than from stream markers. Timestamps are CLOCK_BOOTTIME nanoseconds -- the
 * same domain rocprof uses -- so records merge with an external trace with no
 * calibration, and are directly comparable across the GPUs of a node.
 *
 * Consumers may copy this layout rather than include this header; it is
 * append-only. */
typedef struct {
  uint64_t startNs;
  uint64_t endNs;
  uint64_t seq; /* per-communicator dispatch counter */
  uint64_t commHash;
  uint64_t count;     /* element count of the plan's first collective */
  uint32_t func;      /* ncclFunc_t */
  uint32_t datatype;  /* ncclDataType_t */
  uint32_t nChannels; /* grid.x */
  uint32_t nThreads;  /* block.x */
  int32_t rank;
  uint32_t nColls; /* collectives aggregated into this dispatch */
} ncclKernelTimingRecord;

/* Copies out and releases up to `max` completed records. `got` receives the
 * number written; `dropped` (optional) receives the running count of records
 * lost to buffer overflow, so an under-draining consumer can tell silence from
 * loss. Returns ncclInvalidUsage if timing is not active on this communicator.
 *
 * Not part of the versioned RCCL API: resolve it with dlsym if you want a
 * binary that also runs against builds without it. */
extern "C" ncclResult_t ncclKernelTimingDrain(ncclComm_t comm, ncclKernelTimingRecord* out, int max, int* got,
                                              uint64_t* dropped);

/* Internal */
bool ncclKernelTimingEnabled();
ncclResult_t ncclKernelTimingCommInit(struct ncclComm* comm);
ncclResult_t ncclKernelTimingCommFree(struct ncclComm* comm);

/* Reserves a slot and returns the event to attach as the dispatch's stop event,
 * or nullptr when this launch will not be timed. The reservation is only armed
 * by ncclKernelTimingCommitLaunch, so a launch that fails after reserving is
 * discarded rather than reported with another dispatch's timestamps. */
cudaEvent_t ncclKernelTimingBeginLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, uint32_t nChannels,
                                        uint32_t nThreads, uint64_t* slot);
void ncclKernelTimingCommitLaunch(struct ncclComm* comm, uint64_t slot);

#endif
