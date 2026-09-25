/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/group.cc fakes. See group_fakes.h.

#include "group_fakes.h"

#include <cstdint>

#include "fail_loud.h"
#include "group.h"       // the real declarations of the thread-locals defined below
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM default this stands in for
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclCollPreconnect, ncclCollPreconnect);
#undef ASSERT_HOOK_MATCHES_PROD

// group.cc's thread-local group state. ncclGroupCommJoin is inline in group.h,
// so any TU that joins a comm to a group needs these even though it never calls
// into group.cc. ncclGroupDepth/ncclGroupError sit in nccl_stubs.cc because the
// fail-loud floor already needed them.
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {nullptr};
thread_local int ncclGroupBlocking = -1;  // group.cc's "default mode" sentinel

ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { return ncclSuccess; }

static ncclResult_t DefaultNcclGroupJobAbort(struct ncclGroupJob*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclGroupJob*)> g_ncclGroupJobAbort = DefaultNcclGroupJobAbort;
ncclResult_t ncclGroupJobAbort(struct ncclGroupJob* job) { return g_ncclGroupJobAbort(job); }
ncclResult_t ncclGroupJobComplete(struct ncclGroupJob*) { return ncclSuccess; }

static ncclResult_t DefaultCollPreconnect(struct ncclComm*, bool*) {
  FailLoudUnfaked("group_fakes", "ncclCollPreconnect");
}
std::function<ncclResult_t(struct ncclComm*, bool*)> g_ncclCollPreconnect = DefaultCollPreconnect;
ncclResult_t ncclCollPreconnect(struct ncclComm* comm, bool* algoNeedConnect) {
  return g_ncclCollPreconnect(comm, algoNeedConnect);
}

// Referenced by init.cc but not declared inside it, so the redirected NCCL_PARAM does not cover it.
int64_t ncclParamSingleProcMemRegEnable() { return g_loadParam("SINGLE_PROC_MEM_REG_ENABLE", 0); }  // group.cc:628

void ResetGroupFakes() {
  g_ncclGroupJobAbort = DefaultNcclGroupJobAbort;
  g_ncclCollPreconnect = DefaultCollPreconnect;
}
