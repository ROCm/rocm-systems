/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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


// Stub implementations for topo_expl
#include "hipify_rccl/include/comm.h"
#include "hipify_rccl/include/collectives.h"
#include "hipify_rccl/include/net.h"

// Stub for ncclCommCount - just return the nRanks from the comm
extern "C" ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  *count = comm->nRanks;
  return ncclSuccess;
}

// Stub for ncclNvlsRegResourcesQuery - topo_expl doesn't use NVLS
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, struct ncclTaskColl* info, int* recChannels) {
  *recChannels = 0;
  return ncclSuccess;
}

static rcclIBNicInfo rcclPrimaryNicInfo = {rcclIBNicTypeDefault, 0, 0};
rcclIBNicInfo rcclPrimaryNic() {
  return rcclPrimaryNicInfo;
}

bool rcclUseAinic() {
  return false;
}