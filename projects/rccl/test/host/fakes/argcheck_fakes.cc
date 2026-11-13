/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See argcheck_fakes.h.

#include "argcheck_fakes.h"

#include "argcheck.h"
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_commCheck, CommCheck);
ASSERT_HOOK_MATCHES_PROD(g_cudaPtrCheck, CudaPtrCheck);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultCommCheck(struct ncclComm*, const char*, const char*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*, const char*, const char*)> g_commCheck = DefaultCommCheck;

static ncclResult_t DefaultCudaPtrCheck(const void*, struct ncclComm*, const char*, const char*) {
  return ncclSuccess;
}
std::function<ncclResult_t(const void*, struct ncclComm*, const char*, const char*)> g_cudaPtrCheck =
  DefaultCudaPtrCheck;

void ResetArgcheckFakes() {
  g_commCheck = DefaultCommCheck;
  g_cudaPtrCheck = DefaultCudaPtrCheck;
}

ncclResult_t PtrCheck(const void*, const char*, const char*) { return ncclSuccess; }
ncclResult_t CommCheck(struct ncclComm* ptr, const char* opname, const char* ptrname) {
  return g_commCheck(ptr, opname, ptrname);
}
ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  return g_cudaPtrCheck(pointer, comm, ptrname, opname);
}
