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
 * the null stream. The HIP calls are stand-ins defined in this file, so no GPU
 * is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>
#include <string>

// Defined in hip_playback.cpp; not declared in the public header.
hipError_t playback_hipStreamCreate(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipStreamCreateWithFlags(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipStreamCreateWithPriority(PlaybackContext& ctx, const uint8_t* pl);
hipError_t playback_hipStreamDestroy(PlaybackContext& ctx, const uint8_t* pl);

namespace {

// What the stand-ins below saw.
struct HipCall {
  int calls = 0;
  hipStream_t stream = nullptr;
  int flags = -1;
  int priority = -1;
};
HipCall g_memset;
HipCall g_create;

// Live handles the creation stand-ins hand out. Never dereferenced: no real
// HIP call is made on them.
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

// Stand-ins for the HIP calls. The test executable defines them, so the
// handlers' calls bind here instead of to the HIP runtime.
extern "C" hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t stream) {
  ++g_memset.calls;
  g_memset.stream = stream;
  return hipSuccess;
}

extern "C" hipError_t hipStreamCreate(hipStream_t* stream) {
  ++g_create.calls;
  *stream = fake_stream(g_create.calls);
  return hipSuccess;
}

extern "C" hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  ++g_create.calls;
  g_create.flags = static_cast<int>(flags);
  *stream = fake_stream(g_create.calls);
  return hipSuccess;
}

extern "C" hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                                  int priority) {
  ++g_create.calls;
  g_create.flags = static_cast<int>(flags);
  g_create.priority = priority;
  *stream = fake_stream(g_create.calls);
  return hipSuccess;
}

/**
 * Test Description
 * ----------------
 *   - The recorded values 0, 1 and 2 resolve to the null stream,
 *     hipStreamLegacy and hipStreamPerThread, and a created stream to its live
 *     stream. hipStreamPerThread is left for HIP to resolve on the calling
 *     replay thread, so resolving it adds nothing to the shared map.
 */
TEST_CASE("Unit_HRR_StreamResolve_SpecialAndMappedHandles", "[hrr]") {
  PlaybackContext ctx;
  CHECK(ctx.translate_stream(0) == nullptr);
  CHECK(ctx.translate_stream(1) == hipStreamLegacy);
  CHECK(ctx.translate_stream(2) == hipStreamPerThread);
  CHECK(ctx.stream_map.empty());

  ctx.record_stream(kRecordedStream, fake_stream(1));
  CHECK(ctx.translate_stream(kRecordedStream) == fake_stream(1));
}

/**
 * Test Description
 * ----------------
 *   - A non-zero handle no stream was created for throws hrr_unmapped_stream
 *     carrying the recorded value, instead of resolving to the null stream.
 *   - With --lossy-streams it resolves to the null stream, as replay used to,
 *     and says so once.
 */
TEST_CASE("Unit_HRR_StreamResolve_UnmappedHandleThrows", "[hrr]") {
  PlaybackContext ctx;
  ctx.record_stream(kRecordedStream, fake_stream(1));

  uint64_t thrown = 0;
  try {
    (void)ctx.translate_stream(kOtherStream);
  } catch (const hrr_unmapped_stream& e) {
    thrown = e.rec;
  }
  CHECK(thrown == kOtherStream);

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
 *   - The three standard creators pass the recorded flags and priority to HIP
 *     and map the recorded handle to the stream they made.
 *   - hipStreamDestroy of a stream the replay never created makes no HIP call
 *     and does not end the replay: there is nothing to destroy, and the work
 *     on that stream is where it is reported.
 */
TEST_CASE("Unit_HRR_StreamResolve_CreatorsMapFlagsAndPriority", "[hrr]") {
  PlaybackContext ctx;
  g_create = {};

  hrr_test::ExactRecord<hrr_args_hipStreamCreate> plain;
  plain.get()->stream = 0x7f0000001000ull;
  REQUIRE(playback_hipStreamCreate(ctx, plain.bytes()) == hipSuccess);
  CHECK(ctx.translate_stream(0x7f0000001000ull) == fake_stream(1));

  hrr_test::ExactRecord<hrr_args_hipStreamCreateWithFlags> flagged;
  flagged.get()->stream = 0x7f0000002000ull;
  flagged.get()->flags = hipStreamNonBlocking;
  REQUIRE(playback_hipStreamCreateWithFlags(ctx, flagged.bytes()) == hipSuccess);
  CHECK(g_create.flags == hipStreamNonBlocking);
  CHECK(ctx.translate_stream(0x7f0000002000ull) == fake_stream(2));

  hrr_test::ExactRecord<hrr_args_hipStreamCreateWithPriority> prio;
  prio.get()->stream = 0x7f0000003000ull;
  prio.get()->flags = hipStreamNonBlocking;
  prio.get()->priority = -1;
  REQUIRE(playback_hipStreamCreateWithPriority(ctx, prio.bytes()) == hipSuccess);
  CHECK(g_create.flags == hipStreamNonBlocking);
  CHECK(g_create.priority == -1);
  CHECK(ctx.translate_stream(0x7f0000003000ull) == fake_stream(3));

  hrr_test::ExactRecord<hrr_args_hipStreamDestroy> destroy;
  destroy.get()->stream = kOtherStream;
  CHECK(playback_hipStreamDestroy(ctx, destroy.bytes()) == hipSuccess);
}

/**
 * Test Description
 * ----------------
 *   - The report that ends the replay names the event, the API and the
 *     recorded handle, sets fatal_error and fails the event.
 */
TEST_CASE("Unit_HRR_StreamResolve_ReportNamesEventApiAndHandle", "[hrr]") {
  PlaybackContext ctx;
  hipError_t r = hipSuccess;
  std::string err = hrr_test::capture_stderr(
      [&] { r = hrr_report_unmapped_stream(ctx, 3, 42, "hipMemsetAsync", kOtherStream); });
  INFO(err);
  CHECK(r == hipErrorInvalidHandle);
  CHECK(ctx.fatal_error.load());
  CHECK(err.find("T3 Event 42 (hipMemsetAsync") != std::string::npos);
  CHECK(err.find("0x7f00000a2000") != std::string::npos);
}

/** @} */
