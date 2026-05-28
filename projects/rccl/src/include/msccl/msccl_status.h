/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_STATUS_H_
#define MSCCL_STATUS_H_

#include "msccl/msccl_struct.h"

bool mscclInitialized(const ncclComm_t comm);

void mscclSetInitialized(const ncclComm_t comm, bool initialized = true);

void mscclRemoveRank(const ncclComm_t comm);

mscclStatus& mscclGetStatus(const ncclComm_t comm);

mscclSavedProxyArgs& mscclGetSavedProxyArgs(const ncclComm_t comm);

mscclThreadLocalStatus& mscclGetThreadLocalStatus();

#endif
