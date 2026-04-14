// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "nccl.h"
#include <cstring>
#include "comm.h"
#include "device.h"
#include "archinfo.h"

__attribute__ ((visibility("default")))
ncclResult_t ncclCommDump(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  if (comm == nullptr) {
    WARN("ncclCommDump comm is null");
    return ncclSuccess;
  }
  if (comm->proxyState->proxyTrace == nullptr) {
    WARN("ncclCommDump comm->proxyState->proxyTrace is null");
    return ncclSuccess;
  }

  WARN("ncclCommDump() ProxyTrace:");
  WARN("%s", comm->proxyState->proxyTrace->dump().c_str());

  return ncclSuccess;
}
