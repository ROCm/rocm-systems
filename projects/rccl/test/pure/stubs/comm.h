// Minimal comm.h stub for CPU-only RCCL unit tests.
#pragma once
#include "nccl.h"
#include <cstdint>

struct ncclTaskColl {
  ncclTaskColl* next;
};

struct ncclTaskCollSorter {
  ncclTaskColl* head;
};

inline bool ncclTaskCollSorterEmpty(ncclTaskCollSorter* me) { return me->head == nullptr; }

#define NCCL_MAGIC 0x0280028002800280ULL

struct ncclPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  int64_t busId;
  int sameProcess;
  struct {
    uint64_t cliqueId;
    char uuid[16];
  } fabricInfo;
};

struct ncclComm {
  uint64_t startMagic;
  ncclResult_t asyncResult;
  uint64_t endMagic;
};

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);
