/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mgmt_task_enq_fakes.h"

#include "enqueue/mgmt_task_enq.h"

#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclMgmtTaskEnqueue, ncclMgmtTaskEnqueue);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultNcclMgmtTaskEnqueue(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                                               void (*)(void*), struct ncclComm*) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*), void (*)(void*),
                           struct ncclComm*)>
    g_ncclMgmtTaskEnqueue = DefaultNcclMgmtTaskEnqueue;

ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* task, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t comm) {
  return g_ncclMgmtTaskEnqueue(task, func, destructor, comm);
}

void ResetMgmtTaskEnqFakes() {
  g_ncclMgmtTaskEnqueue = DefaultNcclMgmtTaskEnqueue;
}
