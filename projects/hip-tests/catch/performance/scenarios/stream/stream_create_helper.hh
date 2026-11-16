/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>
#include <vector>

#include "hip_test_common.hh"

namespace stream_test {

/** Which backend this process is running on.
 *
 *      GPU_ENABLE_PAL | Windows | Linux
 *      unset or ""    | PAL     | ROCr
 *      "0"            | ROCr    | ROCr
 *      "1"            | PAL     | PAL
 *      "2"            | auto    | auto
 *
 *  "2" hands the choice to the runtime, so it is reported as Unknown and the caller stays
 *  permissive rather than guessing.
 */
enum class Backend { kRocr, kPal, kUnknown };

inline Backend ActiveBackend() {
  const std::string env = TestContext::getEnvVar("GPU_ENABLE_PAL");
  if (env.empty()) {
    return TestContext::get().isWindows() ? Backend::kPal : Backend::kRocr;
  }
  if (env == "0") return Backend::kRocr;
  if (env == "1") return Backend::kPal;
  return Backend::kUnknown;
}


enum class CreateResult {
  kCreated,      ///< All `count` streams were created.
  kUnavailable,  ///< The device cannot provide this many; nothing was left allocated.
};

/** Creates `count` streams, appending them to `streams`.
 *
 *  On kUnavailable the streams created so far are destroyed and `streams` is left empty. The
 *  counts in these sweeps only increase, so a caller that gets kUnavailable should stop rather
 *  than try the larger ones. Errors other than the exhaustion codes fail the test as usual.
 */
inline CreateResult CreateStreams(size_t count, bool nonBlocking,
                                  std::vector<hipStream_t>& streams) {
  for (size_t i = 0; i < count; ++i) {
    hipStream_t s = nullptr;
    const hipError_t err = nonBlocking ? hipStreamCreateWithFlags(&s, hipStreamNonBlocking)
                                       : hipStreamCreate(&s);

    const bool exhausted = (err == hipErrorOutOfMemory) ||
                           (err == hipErrorLaunchFailure && ActiveBackend() != Backend::kRocr);
    if (!exhausted) {
      HIP_CHECK(err);
      streams.push_back(s);
      continue;
    }

    // Best effort: on a lost context these fail too, and HIP_CHECK here would turn a skip into
    // a failure.
    for (hipStream_t created : streams) {
      static_cast<void>(hipStreamDestroy(created));
    }
    streams.clear();
    // The failed creation, and any failed destroy above, leaves a sticky thread-local error
    // behind, and this path does not fail the test. Consume it, so that a later test case in
    // this process does not fail on an error this one left behind.
    static_cast<void>(hipGetLastError());

    return CreateResult::kUnavailable;
  }
  return CreateResult::kCreated;
}

}  // namespace stream_test
