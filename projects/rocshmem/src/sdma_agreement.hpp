/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/
#ifndef LIBRARY_SRC_SDMA_AGREEMENT_HPP_
#define LIBRARY_SRC_SDMA_AGREEMENT_HPP_

// Node-local agreement on whether SDMA is used. sdmaEnabled also selects the alltoall algorithm, so
// every rank of the IPC group must reach the same answer. Kept free of HSA/KFD headers so the unit
// tests can build it without USE_SDMA.

#include <string>

namespace rocshmem {

// What one rank reports after trying to wire its SDMA queues. Exchanged as an int.
enum SdmaInitStatus : int {
  kSdmaReady = 0,          // every connect() succeeded
  kSdmaDisabledByEnv = 1,  // ROCSHMEM_SDMA_ENABLED=0 on this rank, nothing was wired
  kSdmaConnectFailed = 2,  // a connect() failed on this rank
};

enum class SdmaDecision {
  kCommit,                 // every rank is ready: keep SDMA
  kAllDisabled,            // every rank disabled SDMA by env: nothing to undo
  kFallbackConnectFailed,  // a connect failed somewhere: every rank drops SDMA
  kFallbackEnvMismatch,    // only some ranks disabled SDMA by env: every rank drops SDMA
};

struct SdmaAgreement {
  SdmaDecision decision{SdmaDecision::kCommit};
  // Comma-separated local ranks that caused a fallback, for the log line every rank prints.
  std::string ranks;
};

// Decide from the statuses of all n node-local ranks, indexed by local rank. Any value other than
// kSdmaReady or kSdmaDisabledByEnv counts as a failed connect, so a garbled exchange falls back
// instead of committing.
inline SdmaAgreement agreeSdmaStatus(const int* statuses, int n) {
  SdmaAgreement out;
  std::string failed;
  std::string disabled;
  int numReady = 0;
  auto append = [](std::string& list, int rank) {
    if (!list.empty()) list += ',';
    list += std::to_string(rank);
  };
  for (int i = 0; i < n; ++i) {
    if (statuses[i] == kSdmaReady) {
      ++numReady;
    } else if (statuses[i] == kSdmaDisabledByEnv) {
      append(disabled, i);
    } else {
      append(failed, i);
    }
  }
  if (!failed.empty()) {
    out.decision = SdmaDecision::kFallbackConnectFailed;
    out.ranks = failed;
  } else if (numReady == n) {
    out.decision = SdmaDecision::kCommit;
  } else if (numReady == 0) {
    out.decision = SdmaDecision::kAllDisabled;
  } else {
    out.decision = SdmaDecision::kFallbackEnvMismatch;
    out.ranks = disabled;
  }
  return out;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_AGREEMENT_HPP_
