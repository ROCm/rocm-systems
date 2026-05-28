/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "latency_profiler/CollTraceUtils.h"
#include "nccl_common.h"
#include "debug.h"

namespace latency_profiler {

float getSizeMb(const std::string& dataType, int count) {
  if (dataType == "ncclInt8" || dataType == "ncclFp8E4M3" ||
      dataType == "ncclFp8E5M2") {
    return count / 1024.0 / 1024.0;
  }

  if (dataType == "ncclFloat16" || dataType == "ncclBfloat16") {
    return 2 * count / 1024.0 / 1024.0;
  }

  if (dataType == "ncclInt32" || dataType == "ncclUint32" ||
      dataType == "ncclFloat32") {
    return 4 * count / 1024.0 / 1024.0;
  }

  if (dataType == "ncclInt64" || dataType == "ncclUint64" ||
      dataType == "ncclFloat64") {
    return 8 * count / 1024.0 / 1024.0;
  }

  throw std::runtime_error("CollTrace: unsupported data type " + dataType);
}

void reportToFile(
    const std::deque<std::vector<CollStats>>& stats,
    const std::string& commHash) {
  for (const auto& oneDumpStats : stats) {
    for (const auto& elem : oneDumpStats) {
      auto size_mb = getSizeMb(elem.dataType, elem.count);
      INFO(NCCL_COLL, "coll_id %ld, percent %d, min_latency_us %f, max_latency_us %f, op_name %s, data_type %s, count %ld, message_size_MB %f, comm_hash %s", elem.collId, elem.percent, elem.minLatencyUs, elem.maxLatencyUs, elem.opName.c_str(), elem.dataType.c_str(), elem.count, size_mb, commHash.c_str());
    }
  }
}

std::vector<CollStats> aggregateResults(
    const std::deque<std::unique_ptr<CollTraceInfo>>& info,
    const std::vector<float>& latencyAllGather,
    int RANKS_PER_HOST,
    int NCCL_COLLTRACE_RECORD_MAX) {
  std::vector<std::pair<float, float>> latencyMetrics;
  for (auto rank = 0; rank < RANKS_PER_HOST; rank++) {
    for (auto i = 0; i < NCCL_COLLTRACE_RECORD_MAX; i++) {
      auto val = latencyAllGather.at(rank * NCCL_COLLTRACE_RECORD_MAX + i);
      if (val == 0) {
        throw std::runtime_error(
            "CollTrace: latency value cannot be zero, CPU all gather failed");
      }
      if (rank == 0) {
        latencyMetrics.emplace_back(val, val);
      } else {
        latencyMetrics.at(i).first =
            std::min<float>(latencyMetrics.at(i).first, val);
        latencyMetrics.at(i).second =
            std::max<float>(latencyMetrics.at(i).second, val);
      }
    }
  }
  std::vector<CollStats> results;
  for (int i = 0; i < info.size(); i++) {
    int percent = 100 *
        (latencyMetrics.at(i).second - latencyMetrics.at(i).first) /
        latencyMetrics.at(i).first;
    results.emplace_back(CollStats(
        (int)info[i]->collId,
        percent,
        latencyMetrics.at(i).first * 1000,
        latencyMetrics.at(i).second * 1000,
        info[i]->opName,
        info[i]->dataType,
        info[i]->count));
  }
  return results;
}

} // namespace latency_profiler
