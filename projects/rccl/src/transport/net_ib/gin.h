/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_NET_IB_GIN_H_
#define _NCCL_NET_IB_GIN_H_

#include "nccl.h"
#include "rma_multiseg.h"

// After a prefix post, keep the request and return success so Test() drains.
// Callers NCCLCHECK the complete helper and never reach test() on error.
static inline ncclResult_t ncclRmaPostedRequestStatus(ncclResult_t postRet, int posted) {
  if (postRet != ncclSuccess && posted > 0) return ncclSuccess;
  return postRet;
}

// Keep-or-free after ibv_post_send. posted==0 and error: free the slot.
// Otherwise keep *request so Test() drains; mark FAILED if the signaled tail
// was lost. gin.cc applies the IB side effects from these flags.
static inline ncclResult_t ncclRmaCompletePostedRequest(ncclResult_t postRet, int posted, int nWr, int* keepRequest,
                                                       int* markFailed) {
  if (keepRequest) *keepRequest = 0;
  if (markFailed) *markFailed = 0;
  if (postRet != ncclSuccess && posted == 0) return postRet;
  if (keepRequest) *keepRequest = 1;
  if (markFailed) *markFailed = ncclRmaPrefixPostLostSignaledTail(posted, nWr);
  return ncclRmaPostedRequestStatus(postRet, posted);
}

struct ncclGinIbCollComm {
  void* ctx;
  int rank;
  int nranks;
  int connectionId;
  int nConnections;
  int queueDepth;
  void* recvComm;
  void* sendComm;
  void** fullRecvComm;
  void** fullSendComm;
  int dev;
  void* ginCtx;
  struct {
    struct ibv_context* context;
    struct ibv_pd* pd;
  } ib;
  ncclResult_t (*getProperties)(int dev, void* props);
  ncclResult_t (*allGather)(struct ncclGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len);
  ncclResult_t (*allToAll)(struct ncclGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len);
  ncclResult_t (*getGidIndex)(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                              int* gidIndex);
};

#endif
