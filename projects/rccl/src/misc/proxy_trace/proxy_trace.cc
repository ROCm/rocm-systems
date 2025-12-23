/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "proxy_trace/proxy_trace.h"
#include "debug.h"
#include "device.h"
#include "proxy.h"
#include <map>

constexpr int32_t kFinishedProxyOpItems = 32;
static std::unordered_map<facebook_rccl::ProxyOpStepStatus, std::string>
    proxyStepStatusStrMap = {
        {facebook_rccl::ProxyOpStepStatus::INIT, "INIT"},
        {facebook_rccl::ProxyOpStepStatus::POSTING, "POSTING"},
        {facebook_rccl::ProxyOpStepStatus::SENDING, "SENDING"},
        {facebook_rccl::ProxyOpStepStatus::RECEIVING, "RECEIVING"},
        {facebook_rccl::ProxyOpStepStatus::WAITING_GPU, "WAITING_GPU"},
        {facebook_rccl::ProxyOpStepStatus::FLUSHING, "FLUSHING"},
        {facebook_rccl::ProxyOpStepStatus::DONE, "DONE"},
        {facebook_rccl::ProxyOpStepStatus::UNINITIALIZED, "ILLEGAL"},
};

void facebook_rccl::ProxyTrace::resetAll() {
  activeOps.clear();
  activeOpIdTracker.clear();
  myRank = -1;
  initialized = false;
}

bool facebook_rccl::ProxyTrace::checkActiveOpExist(uint64_t commHash,
                                                   uint64_t opCount,
                                                   uint32_t proxyOpId) const {
  return (activeOps.find(commHash) != activeOps.end() &&
          activeOps.at(commHash).find(opCount) !=
              activeOps.at(commHash).end() &&
          activeOps.at(commHash).at(opCount).find(proxyOpId) !=
              activeOps.at(commHash).at(opCount).end());
}

// Get a unique proxyOpId for a given commHash:opCount
// If the opCount is not found, create a new entry for it and return 0
int64_t facebook_rccl::ProxyTrace::getOrCreateProxyOpId(uint64_t commHash,
                                                        uint64_t opCount) {
  return activeOpIdTracker[commHash][opCount];
}

facebook_rccl::ProxyTraceOp *
facebook_rccl::ProxyTrace::getProxyTraceOpPtr(const ProxyTraceRecordKey &key) {
  if (checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    return &(activeOps.at(key.commHash).at(key.opCount).at(key.proxyOpId));
  }
  return nullptr;
}

void facebook_rccl::ProxyTrace::checkOpCompleted(
    const ProxyTraceRecordKey &key) {
  if (checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    auto &traceOp = activeOps[key.commHash][key.opCount][key.proxyOpId];
    // Remove finished proxyOp or colls to avoid memory leak
    if (traceOp.counters[facebook_rccl::ProxyCounterTypes::DONE] ==
        traceOp.nSteps) {
      traceOp.status = ProxyOpStepStatus::DONE;
      if (finishedOps.size() >= kFinishedProxyOpItems) {
        finishedOps.pop_front();
      }
      finishedOps.push_back({key.str(), traceOp.str()});
      activeOps[key.commHash][key.opCount].erase(key.proxyOpId);
      if (activeOps[key.commHash][key.opCount].empty()) {
        activeOps[key.commHash].erase(key.opCount);
        activeOpIdTracker[key.commHash].erase(key.opCount);
      }
    }
  } else {
    WARN("[proxyTrace] ProxyTraceOp %s not found", key.str().c_str());
  }
}

void facebook_rccl::ProxyTrace::addNewProxyTraceOpImpl(
    const ProxyTraceRecordKey &key, const ProxyTraceExtraInfo &extraInfo,
    ProxyOpType opType, int channelId, int nSteps, uint32_t nbytes,
    int peerRank) {
  if (nSteps > 0 &&
      !checkActiveOpExist(key.commHash, key.opCount, key.proxyOpId)) {
    auto traceOp = facebook_rccl::ProxyTraceOp();
    traceOp.opType = opType;
    traceOp.traceKey = key;
    traceOp.extraInfo = extraInfo;
    traceOp.channelId = channelId;
    traceOp.nSteps = nSteps;
    traceOp.nbytes = nbytes;
    traceOp.myRank = this->myRank;
    traceOp.peerRank = peerRank;
    traceOp.startTs = std::chrono::high_resolution_clock::now();
    traceOp.status = ProxyOpStepStatus::INIT;
    activeOpIdTracker[key.commHash][key.opCount]++;
    activeOps[key.commHash][key.opCount].emplace(key.proxyOpId,
                                                 std::move(traceOp));
  } else if (nSteps == 0) {
    INFO(NCCL_PROXY, "nSteps is 0, ignored %s", key.str().c_str());
  }
}

void facebook_rccl::ProxyTraceOp::computeStatus() {
  ProxyOpStepStatus newStatus;
  int posted = counters[facebook_rccl::ProxyCounterTypes::POSTED];
  int received = counters[facebook_rccl::ProxyCounterTypes::RECEIVED];
  int transmitted = counters[facebook_rccl::ProxyCounterTypes::TRANSMITTED];
  int done = counters[facebook_rccl::ProxyCounterTypes::DONE];
  if (opType == ProxyOpType::RECV) {
    if (posted < nSteps && posted < done + NCCL_STEPS)
      newStatus = ProxyOpStepStatus::POSTING; // Init
    else if (received < posted)
      newStatus = ProxyOpStepStatus::RECEIVING; // Receiving
    else if (received < transmitted)
      newStatus = ProxyOpStepStatus::RECEIVING; // Receiving
    else if (transmitted < received)
      newStatus = ProxyOpStepStatus::FLUSHING; // Flushing
    else if (done < transmitted)
      newStatus = ProxyOpStepStatus::WAITING_GPU; // Waiting on GPU
    else
      newStatus = ProxyOpStepStatus::DONE; // Done
  } else {
    if (posted < nSteps && posted < done + NCCL_STEPS)
      newStatus = ProxyOpStepStatus::POSTING; // Init
    else if (transmitted < posted)
      newStatus = ProxyOpStepStatus::WAITING_GPU; // Waiting on GPU
    else if (done < transmitted)
      newStatus = ProxyOpStepStatus::SENDING; // Sending
    else
      newStatus = ProxyOpStepStatus::DONE; // Done
  }
  this->status = newStatus;
}

std::string facebook_rccl::ProxyTrace::dump(uint64_t commHash) {
  std::string result = fmt::format("commDump for commHash:{}\n", commHash);
  std::map<std::string, std::string> sortedDumpStrMap;
  for (auto &opCountMap : activeOps.at(commHash)) {
    for (auto &proxyOpMap : opCountMap.second) {
      ProxyTraceRecordKey traceKey = {commHash, opCountMap.first,
                                      proxyOpMap.first};
      proxyOpMap.second.computeStatus();
      sortedDumpStrMap[traceKey.str()] = proxyOpMap.second.str();
    }
  }
  for (const auto &pair : sortedDumpStrMap) {
    result += pair.second; //proxyOpStr
  }
  return result;
}

std::string facebook_rccl::ProxyTrace::dump() {
  std::string result = "commDump for all active ops ";
  result += fmt::format("mapSizeMB:{:.2f}\n", getMapSizeMB());

  // maps serialized key to serliazed proxyOp; sorted by key
  std::map<std::string, std::string> sortedDumpStrMap;
  for (auto &commHash_opCountMap : activeOps) {
    for (auto &opCount_proxyOpMap : commHash_opCountMap.second /*opCountMap*/) {
      for (auto &opId_opEntry : opCount_proxyOpMap.second/*proxyOpMap*/) {
        ProxyTraceRecordKey traceKey = {commHash_opCountMap.first, opCount_proxyOpMap.first, opId_opEntry.first};
        opId_opEntry.second.computeStatus();
        sortedDumpStrMap[traceKey.str()] = opId_opEntry.second.str();
      }
    }
  }

  // add the recent finished ops as well
  for (const auto &keyStr_proxyOpStr : finishedOps) {
    sortedDumpStrMap[keyStr_proxyOpStr.first] = keyStr_proxyOpStr.second;
  }
  for (const auto &keyStr_proxyOpStr : sortedDumpStrMap) {
    result += keyStr_proxyOpStr.second;
  }
  return result;
}

std::string facebook_rccl::ProxyTraceOp::str() {
  computeStatus();
  std::string ret = fmt::format(
      "createT:{}, lastT:{}, cntNm:{}, {}, {}, {}->{}({}), "
      "chan:{}, status:{}, ns:{}, nb:{}, po:{}, ke:{}, tail/h:{}, recvT:{}, "
      "connSz/h:{}, trans:{}, flushed:{}, recvd:{}, done:{}\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          startTs.time_since_epoch())
          .count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(
          lastUpdateTs.time_since_epoch())
          .count(),
      static_cast<int>(lastUpdatingCounter), traceKey.str(), extraInfo.str(),
      myRank, peerRank, opType == ProxyOpType::SEND ? "S" : "R", channelId,
      proxyStepStatusStrMap[status], nSteps, nbytes,
      counters[ProxyCounterTypes::POSTED],
      counters[ProxyCounterTypes::KERNEL_COPY_READY],
      counters[ProxyCounterTypes::TAIL_OR_HEAD],
      counters[ProxyCounterTypes::RECV_TAIL],
      counters[ProxyCounterTypes::FIFO_SZ_OR_HEAD_CACHE],
      counters[ProxyCounterTypes::TRANSMITTED],
      counters[ProxyCounterTypes::FLUSHED],
      counters[ProxyCounterTypes::RECEIVED], counters[ProxyCounterTypes::DONE]);
  return ret;
}

float facebook_rccl::ProxyTrace::getMapSizeMB() const {
  float size = 0;
  for (const auto &commHash_opCountMap : activeOps) {
    for (const auto &opCount_proxyOpMap : commHash_opCountMap.second) {
      size += opCount_proxyOpMap.second.size() *
              (sizeof(ProxyTraceOp) +
               sizeof(std::unique_ptr<facebook_rccl::ProxyTraceOp>));
    }
  }
  for (const auto &keyStr_proxyOpStr : finishedOps) {
    size += keyStr_proxyOpStr.first.size() + keyStr_proxyOpStr.second.size();
  }
  return size / 1024.0 / 1024.0;
}

void facebook_rccl::proxyTraceInit(std::unique_ptr<ProxyTrace> &proxyTrace,
                                   int32_t rank, uint64_t commHash) {
  if (proxyTrace) {
    WARN("[proxyTrace] Initializing non-empty proxyTrace! rank: %d, commHash: "
         "%lu",
         rank, commHash);
    return;
  }
  INFO(NCCL_PROXY, "Initializing ProxyTrace, rank: %d, commHash: %lu", rank,
       commHash);
  proxyTrace = std::make_unique<facebook_rccl::ProxyTrace>(rank);
  proxyTrace->initialized = true;
}

void facebook_rccl::updateProxyOpCounter(
    std::unique_ptr<ProxyTrace> &proxyTraceObj,
    const ProxyTraceRecordKey &traceKey, ProxyCounterTypes counter,
    int64_t val) {
  if (proxyTraceObj) {
    auto traceOpPtr = proxyTraceObj->getProxyTraceOpPtr(traceKey);
    if (traceOpPtr) {
      traceOpPtr->counters[counter] = val;
      traceOpPtr->lastUpdateTs = std::chrono::high_resolution_clock::now();
      traceOpPtr->lastUpdatingCounter = counter;
      proxyTraceObj->checkOpCompleted(traceKey);
    }
  }
}

void facebook_rccl::addNewProxyOp(std::unique_ptr<ProxyTrace> &proxyTraceObj,
                                  ProxyTraceRecordKey &key,
                                  const ProxyTraceExtraInfo &extraInfo,
                                  ProxyOpType opType, int channelId, int nSteps,
                                  uint32_t nbytes, int peerRank) {
  if (proxyTraceObj) {
    auto opId = proxyTraceObj->getOrCreateProxyOpId(key.commHash, key.opCount);
    key.proxyOpId = opId;
    proxyTraceObj->addNewProxyTraceOpImpl(key, extraInfo, opType, channelId,
                                          nSteps, nbytes, peerRank);
  }
}
