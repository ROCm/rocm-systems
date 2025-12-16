/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
 
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY 1
#endif
#include <fmt/format.h>
#include <memory>
#include <unordered_map>
namespace facebook_rccl {

enum class ProxyOpStepStatus {
  INIT,
  POSTING,
  SENDING,
  RECEIVING,
  WAITING_GPU,
  FLUSHING,
  DONE,
  UNINITIALIZED,
  NUM_STATUS,
};

// All counters have default value of 0, except for FIFO_SZ_OR_HEAD_CACHE, which
// has default value of -1
enum class ProxyCounterTypes {
  POSTED = 0,
  KERNEL_COPY_READY,
  // ready-to-receive signal received from remote
  RTR_RECV,
  // ready-to-send signal sent to remote
  RTS_SEND,
  RECEIVED,
  TRANSMITTED,
  FLUSHED,
  DONE,
  // tail pointer of data ready to be sent, updated by kernel
  RECV_TAIL,
  // tail for sender, head for receiver, updated by kernel
  TAIL_OR_HEAD,
  // for sender this is data size of D2D copy; for receiver this is head cache,
  // i.e., sub->base + sub->done
  FIFO_SZ_OR_HEAD_CACHE,
  UNINITIALIZED = 100
};

enum class ProxyOpType { SEND, RECV };
// ProxyTraceRecordKey and ProxyTraceExtraInfo is used to pass arguments to
// proxy thread (see ncclProxyOp and ncclProxySubArgs in proxy.h)
struct ProxyTraceRecordKey {
  uint64_t commHash{0};
  int64_t opCount{-1};   // opCount is a unique id for a given collective/p2p
  int64_t proxyOpId{-1}; // id of a proxyOp in an given comm and grouped
                         // collective/p2p (identified as commHash:opCount),
                         // assigned when creating ProxyTraceOp entry
  inline std::string str() const {
    return "<" + std::to_string(commHash) + ":" + std::to_string(opCount) +
           ":" + std::to_string(proxyOpId) + ">";
  }
};

struct ProxyTraceExtraInfo {
  int32_t funcIdx{-1};
  int32_t protocol{-1};
  int32_t pattern{-1};
  uint32_t totalBytes{0};
  uint32_t chunkSize{0};
  inline std::string str() const {
    return fmt::format("[fu,pr,pa,tb,ck]:{},{},{},{},{}", funcIdx, protocol,
                       pattern, totalBytes, chunkSize);
  }
};

// record progress state per comm per collective per proxyOp
struct ProxyTraceOp {
  ProxyTraceRecordKey traceKey;
  ProxyTraceExtraInfo extraInfo;
  int32_t channelId{-1};
  int32_t nSteps{-1};
  uint32_t nbytes{0};
  int32_t myRank{-1};
  int32_t peerRank{-1};
  std::unordered_map<ProxyCounterTypes, int64_t> counters{
      {ProxyCounterTypes::POSTED, 0},
      {ProxyCounterTypes::KERNEL_COPY_READY, 0},
      {ProxyCounterTypes::RTR_RECV, 0},
      {ProxyCounterTypes::RTS_SEND, 0},
      {ProxyCounterTypes::RECEIVED, 0},
      {ProxyCounterTypes::TRANSMITTED, 0},
      {ProxyCounterTypes::FLUSHED, 0},
      {ProxyCounterTypes::DONE, 0},
      {ProxyCounterTypes::RECV_TAIL, 0},
      {ProxyCounterTypes::TAIL_OR_HEAD, 0},
      {ProxyCounterTypes::FIFO_SZ_OR_HEAD_CACHE, -1},
  };
  ProxyCounterTypes lastUpdatingCounter{ProxyCounterTypes::UNINITIALIZED};
  ProxyOpType opType{ProxyOpType::SEND};
  ProxyOpStepStatus status{ProxyOpStepStatus::UNINITIALIZED};
  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> lastUpdateTs{};
  void computeStatus();
  // str the entry to a string
  std::string str();
};

using ProxyActiveOpMap = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<int64_t /* opCount*/,
                       /* proxyOpId : op */
                       std::unordered_map<int64_t, ProxyTraceOp>>>;

using ProxyActiveOpIdTracker =
    std::unordered_map<uint64_t /* commHash*/,
                       std::unordered_map<int64_t /* opCount*/, int64_t>>;

class ProxyTrace {
public:
  ProxyTrace(int32_t rank) : myRank(rank) {}
  ProxyTrace(const ProxyTrace &) = delete;
  ProxyTrace &operator=(const ProxyTrace &) = delete;
  bool initialized{false};
  void checkOpCompleted(const ProxyTraceRecordKey &key);

  void addNewProxyTraceOpImpl(const ProxyTraceRecordKey &key,
                              const ProxyTraceExtraInfo &extraInfo,
                              ProxyOpType opType, int channelId, int nSteps,
                              uint32_t nbytes, int peerRank);

  // Get a unique proxyOpId for a given commHash:opCount
  // If the opCount is not found, create a new entry for it and return 0
  int64_t getOrCreateProxyOpId(uint64_t commHash, uint64_t opCount);

  // Dump all trace for a given communicator
  std::string dump(uint64_t commHash);

  // Dump all active ops
  std::string dump();

  // check if an active send/recv operation exists for a given commHash:opCount
  bool checkActiveOpExist(uint64_t commHash, uint64_t opCount,
                          uint32_t proxyOpId) const;

  ProxyTraceOp *getProxyTraceOpPtr(const ProxyTraceRecordKey &traceKey);
  float getMapSizeMB() const;
  void resetAll();

private:
  int myRank{-1};

  // Current active send/recv operations.
  // Use map to quickly find the record with commHash:opCount:proxyOpId during
  // active progress. Note that each op may not complete in order, e.g.,
  // proxyOpId 1 may finish before proxyOpId 0 if they are to different peers.
  // Thus, the inner-most layer has to still be a map for searching by
  // proxyOpId, no matter other ops are completed or not.
  ProxyActiveOpMap activeOps;
  ProxyActiveOpIdTracker activeOpIdTracker;

  // keep track of the recent completed ops;
  // A record is a pair of traceKey.str() and ProxyTraceOp.str()
  std::deque<std::pair<std::string, std::string>> finishedOps;
};
struct ncclProxySubArgs;
void proxyTraceInit(std::unique_ptr<ProxyTrace> &proxyTrace,
                            int32_t rank, uint64_t commHash);

void updateProxyOpCounter(std::unique_ptr<ProxyTrace> &proxyTraceObj,
                                  const ProxyTraceRecordKey &traceKey,
                                  ProxyCounterTypes counter, int64_t val);

void addNewProxyOp(
    std::unique_ptr<ProxyTrace> &proxyTraceObj, ProxyTraceRecordKey &key,
    const ProxyTraceExtraInfo &extraInfo, ProxyOpType opType, int channelId,
    int nSteps, uint32_t nbytes, int peerRank);
} // namespace facebook_rccl
