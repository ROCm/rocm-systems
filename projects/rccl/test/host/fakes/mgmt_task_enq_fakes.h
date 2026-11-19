/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seam for the symbol defined by src/enqueue/mgmt_task_enq.cc, for every target that does not compile the real file.

#ifndef RCCL_TEST_HOST_MGMT_TASK_ENQ_FAKES_H_
#define RCCL_TEST_HOST_MGMT_TASK_ENQ_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclAsyncJob;
struct ncclComm;

// The default accepts the job without running or freeing it.
extern std::function<ncclResult_t(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*), void (*)(void*),
                                  struct ncclComm*)>
    g_ncclMgmtTaskEnqueue;

void ResetMgmtTaskEnqFakes();

#endif  // RCCL_TEST_HOST_MGMT_TASK_ENQ_FAKES_H_
