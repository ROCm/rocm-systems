/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_NET_IB_GIN_H_
#define _NCCL_NET_IB_GIN_H_

#include <stddef.h>
#include <stdint.h>
#include "nccl.h"

// Cap on physical segments per GIN/RMA symmetric buffer. HIP dma-buf export
// describes only the first physical segment, so registration allocates one MR
// per segment up to this limit.
#ifndef NCCL_RMA_MAX_SEGMENTS
#define NCCL_RMA_MAX_SEGMENTS 16
#endif

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
