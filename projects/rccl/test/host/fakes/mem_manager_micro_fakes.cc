/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See mem_manager_micro_fakes.h.

#include "mem_manager_micro_fakes.h"

#include <cstdarg>
#include <cstdio>

#include "debug.h"
#include "fail_loud.h"

uint32_t ncclDebugLevelMask = 0;  // INFO off, as in nccl_fakes.cc
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam;

void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char* filefunc, int line, const char* fmt, ...) {
  std::fprintf(stderr, "[fake] %s:%d ", filefunc ? filefunc : "?", line);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int ncclGroupBlocking = -1;

std::function<ncclResult_t(int, void*, ncclIpcFd*)> g_getFd;
std::function<ncclResult_t(ncclComm*)> g_commCheck;
std::function<ncclResult_t()> g_groupEnd;
std::function<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(void*))> g_mgmtEnqueue;
std::vector<int> g_closedFds;
int64_t g_rearch = 0;
ncclResult_t g_asyncError = ncclSuccess;
ncclResult_t g_asyncErrorResult = ncclSuccess;

ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm*, int rank, void* handle, ncclIpcFd* fd) {
  return g_getFd(rank, handle, fd);
}
int ncclIpcFdClose(ncclIpcFd fd) {
  g_closedFds.push_back(fd);
  return 0;
}
ncclResult_t CommCheck(struct ncclComm* comm, const char*, const char*) { return g_commCheck(comm); }
ncclResult_t ncclCommEnsureReady(ncclComm_t) { return ncclSuccess; }
ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) {
  ncclGroupDepth--;
  ncclResult_t ret = g_groupEnd();
  ncclGroupError = ncclSuccess;
  return ret;
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t* asyncError) {
  *asyncError = g_asyncError;
  return g_asyncErrorResult;
}
int64_t ncclParamEnqueueRearchEnable() { return g_rearch; }
// Only ncclGroupCommJoin's collective arm reaches this; Suspend/Resume join as SymRegister.
void* ncclMemoryStack::allocateSpilled(ncclMemoryStack*, size_t, size_t) {
  FailLoudUnfaked("mem-manager-test", "ncclMemoryStack::allocateSpilled");
}
ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                                 void (*destructor)(void*), ncclComm_t) {
  return g_mgmtEnqueue(job, func, destructor);
}
