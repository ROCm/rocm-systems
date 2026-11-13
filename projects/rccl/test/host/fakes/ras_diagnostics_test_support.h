#ifndef RCCL_TEST_HOST_RAS_DIAGNOSTICS_TEST_SUPPORT_H_
#define RCCL_TEST_HOST_RAS_DIAGNOSTICS_TEST_SUPPORT_H_

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "ras/diagnostics.h"
#include "ras/diagnostics_checks_common.h"
#include "fakes/ras_registry_test_support.h"

namespace ras_test {

template <typename Payload>
std::vector<char> BuildRecords(const std::vector<std::pair<rasDiagnosticsRankHeader, Payload>>& records) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(Payload));
  std::vector<char> data(records.size() * stride, 0);
  size_t offset = 0;
  for (const auto& record : records) {
    std::memcpy(data.data() + offset, &record.first, sizeof(record.first));
    std::memcpy(data.data() + offset + sizeof(record.first), &record.second, sizeof(record.second));
    offset += stride;
  }
  return data;
}

struct OwnedComm {
  OwnedComm(uint64_t commHash, uint64_t hostHash, uint64_t pidHash, int rank, bool peerInfoValid = true)
      : comm(std::make_unique<ncclComm>()), peers(std::make_unique<ncclPeerInfo[]>(1)) {
    comm->commHash = commHash;
    comm->peerInfo = peers.get();
    comm->peerInfoValid = peerInfoValid;
    comm->rank = rank;
    comm->nRanks = 8;
    comm->cudaDev = rank + 10;
    comm->nvmlDev = rank + 20;
    comm->busId = 0x1000 + rank;
    comm->localRank = rank % 4;
    comm->localRanks = 4;
    peers[0].hostHash = hostHash;
    peers[0].pidHash = pidHash;
  }

  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclPeerInfo[]> peers;
};

struct ReporterState {
  std::vector<std::string> lines;
  ncclResult_t emitResult = ncclSuccess;
  int calls = 0;
  int failCall = 0;
};

inline ncclResult_t CaptureReport(void* target, const char* line) {
  auto* state = static_cast<ReporterState*>(target);
  state->lines.emplace_back(line);
  ++state->calls;
  return state->failCall == 0 || state->calls == state->failCall ? state->emitResult : ncclSuccess;
}

inline rasDiagnosticsReporter MakeReporter(ReporterState* state) {
  return rasDiagnosticsReporter{CaptureReport, nullptr, state};
}

}

#endif
