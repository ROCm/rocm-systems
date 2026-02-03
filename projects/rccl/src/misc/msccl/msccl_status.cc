/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "msccl/msccl_status.h"
#include "msccl/msccl_struct.h"

#include "debug.h"
#include "comm.h"
#include <memory>
#include <mutex>
#include <unordered_map>

using namespace std;

struct mscclRankState {
  bool initialized;
  mscclStatus status;
  mscclSavedProxyArgs savedProxyArgs;

  mscclRankState() : initialized(false), status(), savedProxyArgs() {}
  explicit mscclRankState(const mscclRankState&) = default;
};

static mutex rankStatesMutex;
/*
 * @brief rankStates is intended to hold mscclRankState for each communicator in a rccl process.
 * "rankStates" is not threadsafe, hence read/writes on this data strcutures need to be handled explicitly by
 * block of code that is accessing the elements in this map using a lock guard or any mutual exclusion device.
 */
static unordered_map<ncclComm_t, shared_ptr<mscclRankState>> rankStates;

static inline mscclRankState& mscclGetRankState(const ncclComm_t comm) {
  //the following condition comm == nullptr evaluates true when mscclAvailable() called with default params
  if (comm == nullptr) {
    static thread_local shared_ptr<mscclRankState> threadRankState(new mscclRankState());
    return *threadRankState;
  }

  lock_guard<mutex> lock(rankStatesMutex);

  auto rankStateIt = rankStates.find(comm);
  if (rankStateIt == rankStates.end()) {
    // Create a per rank threadRankState rather than per thread
    shared_ptr<mscclRankState> newthreadRankState(new mscclRankState());
    // newthreadRankState->rank = rank;
    rankStateIt = rankStates.insert(make_pair(comm, newthreadRankState)).first;
  }
  return *(rankStateIt->second);
}

bool mscclInitialized(const ncclComm_t comm) {
  return mscclGetRankState(comm).initialized;
}

void mscclSetInitialized(const ncclComm_t comm, bool initialized) {
  auto& state = mscclGetRankState(comm);
  assert(!initialized || !state.initialized);
  state.initialized = initialized;
}

void mscclRemoveRank(const ncclComm_t comm) {
  lock_guard<mutex> lock(rankStatesMutex);
  rankStates.erase(comm);
}

mscclStatus& mscclGetStatus(const ncclComm_t comm) {
  return mscclGetRankState(comm).status;
}

mscclThreadLocalStatus& mscclGetThreadLocalStatus() {
  static thread_local mscclThreadLocalStatus threadLocalStatus;
  return threadLocalStatus;
}

mscclSavedProxyArgs& mscclGetSavedProxyArgs(const ncclComm_t comm) {
  return mscclGetRankState(comm).savedProxyArgs;
}
