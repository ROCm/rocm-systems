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
#include "comm.h"
#include "collectives.h"
#include "net.h"
#include <climits>

// Minimal libc-backed implementations of the ncclOs* affinity / page-size
// helpers introduced by the NCCL 2.29.x sync. The real implementations live
// in src/os/linux.cc, but linking that whole file into topo_expl pulls in
// header dependencies that clash with topo_expl's own include/utils.h.
// topo_expl only needs CPU-affinity bookkeeping for graph construction and
// the page size for allocator templates, so we provide thin wrappers.
#include <sched.h>
#include <unistd.h>
#include <cstring>

void ncclOsCpuZero(cpu_set_t& affinity) {
  CPU_ZERO(&affinity);
}

void ncclOsCpuSet(cpu_set_t& affinity, int cpu) {
  CPU_SET(cpu, &affinity);
}

bool ncclOsCpuIsSet(const cpu_set_t affinity, int cpu) {
  return CPU_ISSET(cpu, &affinity) != 0;
}

int ncclOsCpuCount(const cpu_set_t affinity) {
  return CPU_COUNT(&affinity);
}

cpu_set_t ncclOsCpuAnd(const cpu_set_t& a, const cpu_set_t& b) {
  cpu_set_t r;
  CPU_AND(&r, &a, &b);
  return r;
}

ncclResult_t ncclOsGetAffinity(cpu_set_t* affinity) {
  if (sched_getaffinity(0, sizeof(cpu_set_t), affinity) != 0) {
    return ncclSystemError;
  }
  return ncclSuccess;
}

size_t ncclOsGetPageSize() {
  long ps = sysconf(_SC_PAGESIZE);
  return ps > 0 ? (size_t)ps : (size_t)4096;
}

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

// Stub for ncclParamWorkArgsBytes - needed by connect.cc and paths.cc
int64_t ncclParamWorkArgsBytes() {
  return INT64_MAX;
}

// Stubs for symbols newly referenced by src/misc/param.cc and debug.h
// after the NCCL 2.28.9 sync. topo_expl does not use the env-plugin
// machinery and does not link debug.cc, so provide minimal definitions.
__attribute__((visibility("default"))) uint64_t ncclDebugMask = 0;

ncclResult_t ncclInitEnv(void) {
  return ncclSuccess;
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return getenv(name);
}

// ncclParamIsCacheDisabled is defined in src/param/param.cc, which topo_expl
// does not link (doing so pulls in the entire param registry/parsers subsystem
// added by the NCCL 2.30 sync). topo_expl never disables param caching, so
// report "not disabled" (i.e. caching enabled, the normal runtime behavior).
extern "C" bool ncclParamIsCacheDisabled(const char* key) {
  return false;
}
