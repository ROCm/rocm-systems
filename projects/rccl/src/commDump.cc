// (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: MIT

#include "nccl.h"
#include <cstring>
#include "comm.h"
#include "device.h"
#include "archinfo.h"
#include "profiler.h"

__attribute__ ((visibility("default")))
ncclResult_t ncclCommDump(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  (void)map;
  if (comm == nullptr) {
    WARN("ncclCommDump comm is null");
    return ncclSuccess;
  }
  if (comm->proxyState == nullptr) {
    WARN("ncclCommDump comm->proxyState is null");
    return ncclSuccess;
  }

  WARN("ncclCommDump() proxy-trace (profiler plugin):");
  ncclProfilerProxyTraceDumpIfAny(comm->profilerContext);

  return ncclSuccess;
}
