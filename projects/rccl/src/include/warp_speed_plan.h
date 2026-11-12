/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_WARP_SPEED_PLAN_H_
#define RCCL_WARP_SPEED_PLAN_H_

#include "archinfo.h"
#include "comm.h"

#ifdef ENABLE_WARP_SPEED
constexpr int RCCL_WARP_SPEED_MAX_PLAN_COLLS = 3;

inline bool rcclShouldCapGroupedPlan(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                     struct ncclTaskColl* collHead, int nPlanColls) {
  if (nPlanColls <= RCCL_WARP_SPEED_MAX_PLAN_COLLS) return false;

  // The single-node gfx950 corruption boundary is not characterized narrowly enough
  // to safely depend on task ordering, datatype, operation, count, or auto mode.
  if (comm->nNodes == 1 && IsArchMatch(comm->archName, "gfx950")) return true;

  return rcclWarpSpeedSupported(comm, plan, collHead, RCCL_WARP_SPEED_MAX_PLAN_COLLS);
}
#endif

#endif
