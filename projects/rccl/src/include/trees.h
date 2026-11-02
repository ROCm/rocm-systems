/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

#include "nccl.h"

ncclResult_t ncclGetBtree(int nranks, int rank, int* u0, int* d1, int* d0, int* parentChildType);
ncclResult_t ncclGetDtree(int nranks, int rank, int* u0, int* d0_0, int* d0_1, int* parentChildType0, int* u1,
                          int* d1_0, int* d1_1, int* parentChildType1);

// Compact tree algorithms optimized for sorted domain ordering.
// Uses half-interleave remapping so that, in the first tree, every non-leaf node is in the
// first half of ranks and the second half are leaves (the second tree is its mirror/shift).
// Both trees keep cross-domain hops at the leaves only for an even two-way domain split;
// see trees.cc. Enable via NCCL_TREE_COMPACT=1
ncclResult_t ncclGetDtreeCompact(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1,
                                 int* d1_0, int* d1_1, int* parentChildType1);

#endif
