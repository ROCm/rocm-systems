/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "algorithms/dda/ipc/dda_nranks_relax_consensus.h"
#include "debug.h"

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
ncclResult_t ncclCheckDdaNranksRelaxConsensus(int nranks, std::function<bool(int)> getDdaNranksRelax,
                                              std::function<const char*(int)> getHostname) {
  if (nranks <= 0) return ncclSuccess;

  const bool ref = getDdaNranksRelax(0);
  int nDisagree = 0;
  for (int r = 1; r < nranks; r++) {
    if (getDdaNranksRelax(r) != ref) nDisagree++;
  }
  if (nDisagree == 0) return ncclSuccess;

  WARN("RCCL FATAL: RCCL_DDA_NRANKS_RELAX disagrees across ranks of this communicator (rank 0 on host %s read %d). "
       "This variable is read per process and must be set identically on every rank, or ranks that admit it "
       "diverge from ranks that don't at DDA IPC comm-init time and hang waiting for each other.",
       getHostname(0), ref ? 1 : 0);
  for (int r = 1; r < nranks; r++) {
    if (getDdaNranksRelax(r) == ref) continue;
    WARN("  rank %d host %s RCCL_DDA_NRANKS_RELAX=%d", r, getHostname(r), getDdaNranksRelax(r) ? 1 : 0);
  }
  return ncclInvalidUsage;
}
