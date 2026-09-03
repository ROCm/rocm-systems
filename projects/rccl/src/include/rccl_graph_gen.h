/*
Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef RCCL_GRAPH_GEN_H_
#define RCCL_GRAPH_GEN_H_

#include "nccl.h"
#include "param.h"

/**
 * Used for enabling using walecki construction of rings in intranode ranks.
 */
RCCL_PARAM_DECLARE(IntraGraphGen);
RCCL_PARAM_DECLARE(InterGraphGen);
void permute_array_inplace(int* input, int length, int* permutation);
void findRingCutIndices(int nChannels, int nNodes /*nodes in the ring graph*/, const int* flattenedRings /* Hamiltonian rings*/, int* cutIndices);
ncclResult_t generateRings(int nNodes, uint8_t nChannels, int* nodeOrder);

#endif  // RCCL_GRAPH_GEN_H_