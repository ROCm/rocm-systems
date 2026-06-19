/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file ce_fault_inject.h
 * @brief Test-only fault injection API for CE (Copy-Engine) collectives.
 *
 * This header is compiled in only when ENABLE_FAULT_INJECTION is defined
 * (test builds).  Production builds pay zero overhead.
 *
 * Usage pattern (mirrors net_ib_fault_inject.h):
 *
 *   // arm a fault before calling the code under test
 *   ncclCeFaultSet(comm, CE_FAULT_LAUNCH_OP);
 *   ncclResult_t r = ncclCeLaunchBatchOps(comm, &params, stream);
 *   EXPECT_EQ(r, ncclSystemError);
 *   ncclCeFaultClear(comm);
 *
 *   // verify normal operation resumes
 *   r = ncclCeLaunchBatchOps(comm, &params, stream);
 *   EXPECT_EQ(r, ncclSuccess);
 */

#ifndef NCCL_CE_FAULT_INJECT_H_
#define NCCL_CE_FAULT_INJECT_H_

#ifdef ENABLE_FAULT_INJECTION

#include "nccl.h"

// ---------------------------------------------------------------------------
// Fault bits – OR them together when arming multiple faults at once.
// ---------------------------------------------------------------------------

/** Force ncclCeInit() to return ncclSystemError. */
#define CE_FAULT_INIT        0x01U

/** Force ncclPrepUCSync() to return ncclSystemError. */
#define CE_FAULT_SYNC_PREP   0x02U

/** Force ncclCeLaunchBatchOps() to return ncclSystemError. */
#define CE_FAULT_LAUNCH_OP   0x04U

// ---------------------------------------------------------------------------
// Inline implementations – ncclComm is defined in comm.h, ncclResult_t in nccl.h.
// All three functions are defined here so callers need not link against a
// separate translation unit.
// ---------------------------------------------------------------------------

#include "comm.h"   // ncclComm definition (transitively includes nccl.h)

static inline ncclResult_t ncclCeFaultSet(struct ncclComm* comm, uint32_t bits) {
  if (comm == nullptr) return ncclInvalidArgument;
  comm->ceColl.ceFaults |= bits;
  return ncclSuccess;
}

static inline ncclResult_t ncclCeFaultClear(struct ncclComm* comm) {
  if (comm == nullptr) return ncclInvalidArgument;
  comm->ceColl.ceFaults = 0;
  return ncclSuccess;
}

static inline uint32_t ncclCeFaultGet(struct ncclComm* comm) {
  if (comm == nullptr) return 0;
  return comm->ceColl.ceFaults;
}

#endif /* ENABLE_FAULT_INJECTION */
#endif /* NCCL_CE_FAULT_INJECT_H_ */
