/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Host-side SDMA ring writer (SDMA 4.4.x packets) for the CE collectives.
 ************************************************************************/
#ifndef NCCL_CE_SDMA_QUEUE_H_
#define NCCL_CE_SDMA_QUEUE_H_

#include <stddef.h>
#include <stdint.h>

#include "nccl.h"

struct ncclCeSdmaHostQueue;

// Process-wide queue for (srcDev, dstDev), created on first use; producers are serialized by a mutex.
ncclResult_t ncclCeSdmaHostQueueGet(int srcDev, int dstDev, struct ncclCeSdmaHostQueue** out);

// Appends complete packets as one submission (one doorbell); blocks while the ring is full.
ncclResult_t ncclCeSdmaHostQueueSubmit(struct ncclCeSdmaHostQueue* q, const uint32_t* dwords, size_t nDwords);

#endif  // NCCL_CE_SDMA_QUEUE_H_
