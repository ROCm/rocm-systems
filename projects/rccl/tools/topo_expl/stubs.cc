// Stub implementations for topo_expl
#include "hipify_rccl/include/comm.h"
#include "hipify_rccl/include/collectives.h"

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
