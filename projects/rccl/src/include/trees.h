/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

ncclResult_t ncclGetBtree(int nranks, int rank, int* u0, int* d1, int* d0, int* parentChildType);
ncclResult_t ncclGetDtree(int nranks, int rank, int* u0, int* d0_0, int* d0_1, int* parentChildType0, int* u1, int* d1_0, int* d1_1, int* parentChildType1);

// Compact tree algorithms optimized for sorted domain ordering.
// Uses half-interleave remapping to keep spine in first half of ranks and push
// cross-domain hops to leaf level where they execute in parallel.
// Enable via NCCL_TREE_COMPACT=1
ncclResult_t ncclGetBtreeCompact(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType);
ncclResult_t ncclGetDtreeCompact(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1, int* d1_0, int* d1_1, int* parentChildType1);

#endif
