/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "group.h"
#include "debug.h"

ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* task, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;

  task->destroyFlag = comm->destroyFlag;
  task->func = func;
  task->destructor = destructor;
  task->abortFlag = comm->abortFlag;
  task->abortFlagDev = comm->abortFlagDev;
  task->childAbortFlag = comm->childAbortFlag;
  task->childAbortFlagDev = comm->childAbortFlagDev;
  task->state = ncclGroupJobRunning;
  task->comm = comm;
  /* check if there are blocking and nonblocking comms at the same time in group. Checked before
   * joining: ncclGroupCommJoin records this comm's mode, and a rejected comm must not stay linked
   * into the group list, because the caller frees it on failure. */
  if (!comm->destroyFlag && ncclGroupBlocking != -1 && ncclGroupBlocking != comm->config.blocking) {
    WARN("Blocking and nonblocking communicators are not allowed in the same group.");
    ret = ncclInvalidArgument;
    // the task hasn't run
    if (destructor) destructor(task);
    return ret;
  }
  ncclGroupCommJoin(comm, ncclGroupTaskTypeMgmtTask);
  if (comm->destroyFlag) ncclGroupBlocking = 1;
  ncclIntruQueueEnqueue(&comm->mgmtTaskQueue, task);

  return ret;
}
