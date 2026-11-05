/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Stream Resolution
 * @{
 * @ingroup HRRTest
 * Handler-level tests of how a recorded stream handle becomes the stream a
 * replayed call is made on. The null stream, hipStreamLegacy and
 * hipStreamPerThread keep their identity; a stream the replay created is the
 * one it maps to; any other handle stops the call instead of reaching HIP as
 * the null stream. The HIP call is a stand-in defined in this file, so no GPU
 * is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <string>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipStreamDestroy(PlaybackContext& ctx, const uint8_t* pl);

namespace {

// What the stand-in below saw.
struct HipCall {
  int calls = 0;
  hipStream_t stream = nullptr;
};
HipCall g_memset;

// Live stream handles for the tests. Never dereferenced: no real HIP call is
// made on them.
hipStream_t fake_stream(uintptr_t n) { return reinterpret_cast<hipStream_t>(0x5000 + n); }

constexpr uint64_t kRecordedStream = 0x7f00000a1000ull;
constexpr uint64_t kOtherStream    = 0x7f00000a2000ull;

hipError_t memset_async(PlaybackContext& ctx, uint64_t stream) {
  hrr_test::ExactRecord<hrr_args_hipMemsetAsync> rec;
  rec.get()->stream = stream;
  rec.get()->sizeBytes = 64;
  REQUIRE(hrr_playback_dispatch[HRR_API_HIPMEMSETASYNC] != nullptr);
  return hrr_playback_dispatch[HRR_API_HIPMEMSETASYNC](ctx, rec.bytes());
}

}  // namespace

// Stand-in for the HIP call. The test executable defines it, so the handler's
// call binds here instead of to the HIP runtime.
extern "C" hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t stream) {
  ++g_memset.calls;
  g_memset.stream = stream;
  return hipSuccess;
}

/**
 * Test Description
 * ----------------
 *   - With --lossy-streams a handle no stream was created for resolves to the
 *     null stream, and says so once.
 */
TEST_CASE("Unit_HRR_StreamResolve_LossyStreamsUseNullStream", "[hrr]") {
  PlaybackContext ctx;
  ctx.lossy_streams = true;
  hipStream_t s = fake_stream(9);
  std::string err = hrr_test::capture_stderr([&] {
    s = ctx.translate_stream(kOtherStream);
    (void)ctx.translate_stream(kOtherStream);
  });
  CHECK(s == nullptr);
  CHECK(err.find("--lossy-streams") != std::string::npos);
  CHECK(err.find("--lossy-streams", err.find("--lossy-streams") + 1) == std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - A replayed hipMemsetAsync reaches HIP on the stream the recording names:
 *     the null stream, hipStreamLegacy, hipStreamPerThread, or the live stream
 *     created for the recorded one.
 *   - On a stream the replay never created the handler makes no HIP call at
 *     all: the null stream would run the memset where the capture did not.
 */
TEST_CASE("Unit_HRR_StreamResolve_HandlerKeepsRecordedStream", "[hrr]") {
  PlaybackContext ctx;
  ctx.record_stream(kRecordedStream, fake_stream(1));

  struct Case { uint64_t rec; hipStream_t want; };
  for (const Case& c : {Case{0, nullptr}, Case{1, hipStreamLegacy},
                        Case{2, hipStreamPerThread}, Case{kRecordedStream, fake_stream(1)}}) {
    INFO("recorded stream 0x" << std::hex << c.rec);
    g_memset = {};
    g_memset.stream = fake_stream(99);
    REQUIRE(memset_async(ctx, c.rec) == hipSuccess);
    CHECK(g_memset.calls == 1);
    CHECK(g_memset.stream == c.want);
  }

  g_memset = {};
  bool thrown = false;
  try {
    (void)memset_async(ctx, kOtherStream);
  } catch (const hrr_unmapped_stream& e) {
    thrown = e.rec == kOtherStream;
  }
  CHECK(thrown);
  CHECK(g_memset.calls == 0);
}

/**
 * Test Description
 * ----------------
 *   - hipStreamDestroy of a stream the replay never created does not end the
 *     replay: there is nothing to destroy, and the work on that stream is
 *     where it is reported.
 */
TEST_CASE("Unit_HRR_StreamResolve_DestroyUnmappedStream", "[hrr]") {
  PlaybackContext ctx;
  hrr_test::ExactRecord<hrr_args_hipStreamDestroy> destroy;
  destroy.get()->stream = kOtherStream;
  CHECK(playback_hipStreamDestroy(ctx, destroy.bytes()) == hipSuccess);
}

/** @} */
