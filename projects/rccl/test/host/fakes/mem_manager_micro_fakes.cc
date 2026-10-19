/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See mem_manager_micro_fakes.h.

#include "mem_manager_micro_fakes.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "debug.h"
#include "fail_loud.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam;

std::vector<std::string> g_warns;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  if (level != NCCL_LOG_WARN) return;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_warns.push_back(buf);
}

thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int ncclGroupBlocking = -1;

std::function<ncclResult_t(int, int)> g_barrier;
std::function<ncclResult_t(void*, int)> g_allGather;
std::function<ncclResult_t(int, int, void*, int)> g_send;
std::function<ncclResult_t(int, int, void*, int)> g_recv;
std::function<ncclResult_t(int, void*, ncclIpcFd*)> g_getFd;
std::function<ncclResult_t(ncclComm*)> g_commCheck;
std::function<ncclResult_t()> g_groupEnd;
std::function<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(void*))> g_mgmtEnqueue;
std::vector<int> g_closedFds;
int64_t g_rearch = 0;
ncclResult_t g_asyncError = ncclSuccess;
ncclResult_t g_asyncErrorResult = ncclSuccess;

// Counts deletes of one watched pointer and keeps its memory, so a double delete is observable.
// new is replaced too so the pair stays malloc/free, which ASan expects.
void* g_watchedDelete = nullptr;
int g_watchedDeletes = 0;
void* operator new(size_t n) {
  void* p = std::malloc(n ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept {
  if (p != nullptr && p == g_watchedDelete) {
    g_watchedDeletes++;
    return;
  }
  std::free(p);
}
void operator delete(void* p, size_t) noexcept {
  operator delete(p);
}
// mem_manager.cc allocates its jobs with new (std::nothrow), so the nothrow and array forms are replaced too.
void* operator new[](size_t n) { return ::operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return std::malloc(n ? n : 1); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return std::malloc(n ? n : 1); }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete[](void* p, size_t) noexcept { operator delete(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { operator delete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { operator delete(p); }

ncclResult_t bootstrapBarrier(void*, int, int, int tag) { return g_barrier(tag, 0); }
ncclResult_t bootstrapAllGather(void*, void* data, int size) { return g_allGather(data, size); }
ncclResult_t bootstrapSend(void*, int peer, int tag, void* data, int size) { return g_send(peer, tag, data, size); }
ncclResult_t bootstrapRecv(void*, int peer, int tag, void* data, int size) { return g_recv(peer, tag, data, size); }
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
